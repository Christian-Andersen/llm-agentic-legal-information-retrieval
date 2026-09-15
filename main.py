import gc
import re
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import torch
from loguru import logger
from pydantic_settings import BaseSettings
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.util import semantic_search
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path("./data/")


class Settings(BaseSettings):
    train: bool = False
    dev: bool = True


settings = Settings()

# --- PREPROCESSING ---


def prepare_val_training_data(val_df: pd.DataFrame, laws_df: pd.DataFrame, courts_df: pd.DataFrame) -> pd.DataFrame:
    """Maps citation IDs to actual text for training supervision."""
    logger.info("Starting validation data preprocessing...")

    laws_dict = dict(zip(laws_df["citation"], laws_df["text"], strict=True))
    courts_dict = dict(zip(courts_df["citation"], courts_df["text"], strict=True))
    all_texts = {**laws_dict, **courts_dict}

    def get_combined_text(citation_str: str) -> str:
        cits = [c.strip() for c in citation_str.split(";")]
        return "\n".join([all_texts.get(c, "") for c in cits if c in all_texts])

    val_df["gold_context"] = val_df["gold_citations"].apply(get_combined_text)
    val_df["hard_negative"] = laws_df["text"].sample(len(val_df), replace=True).to_numpy()

    logger.success(f"Preprocessed {len(val_df)} validation rows.")
    return val_df


# --- PHASE 1: EMBEDDER ---


def run_embedder(  # noqa: PLR0913
    training_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    top_k: int = 50,
    *,
    train: bool = True,
) -> list[dict[str, Any]]:
    logger.info(f"Initializing Embedder Phase with {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={"dtype": torch.bfloat16})

    if train:
        train_examples = [InputExample(texts=[r["query"], r["gold_context"][:4000]]) for _, r in training_df.iterrows()]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)  # ty:ignore[invalid-argument-type]
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        logger.debug("Beginning Embedder fine-tuning...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
        )

    model.max_seq_length = 2048  # Cap sequence length to avoid massive masks
    logger.debug("Encoding corpus and performing semantic search...")
    # Use smaller batch_size and numpy array to save memory
    corpus_embeddings = model.encode(
        corpus_df["text"].tolist(),
        batch_size=64,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(test_df["query"].tolist(), prompt_name="query", convert_to_tensor=True)

    hits = semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)

    test_results = []
    for i, hit_list in enumerate(hits):
        candidates = [corpus_df.iloc[h["corpus_id"]].to_dict() for h in hit_list]
        test_results.append(
            {"query_id": test_df.iloc[i]["query_id"], "query": test_df.iloc[i]["query"], "candidates": candidates}
        )

    # Cleanup
    del model, corpus_embeddings, query_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    logger.success("Embedder phase complete and VRAM cleared.")
    return test_results


# --- PHASE 2: RERANKER ---


def run_reranker(
    training_df: pd.DataFrame,
    test_candidates: list[dict[str, Any]],
    model_name: str = "BAAI/bge-reranker-base",
    top_n: int = 10,
    *,
    train: bool = True,
) -> list[dict[str, Any]]:
    logger.info(f"Initializing Reranker Phase with {model_name}")
    model = CrossEncoder(model_name, num_labels=1)

    if train:
        train_examples = []
        for _, r in training_df.iterrows():
            train_examples.append(InputExample(texts=[r["query"], r["gold_context"]], label=1.0))
            train_examples.append(InputExample(texts=[r["query"], r["hard_negative"]], label=0.0))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)  # ty:ignore[invalid-argument-type]
        model.fit(train_dataloader=train_dataloader, epochs=1)

    reranked_results = []
    for item in test_candidates:
        pairs = [[item["query"], c["text"]] for c in item["candidates"]]
        scores = model.predict(pairs)
        for idx, c in enumerate(item["candidates"]):
            c["rerank_score"] = scores[idx]

        top_c = sorted(item["candidates"], key=lambda x: x["rerank_score"], reverse=True)[:top_n]
        reranked_results.append({"query_id": item["query_id"], "query": item["query"], "top_candidates": top_c})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.success("Reranker phase complete and VRAM cleared.")
    return reranked_results


# --- PHASE 3: LLM ---


def parse_and_validate_citations(
    llm_output: str,
    valid_citations: set[str],
    fallback_citation: str | None,
) -> str:
    """Extract and validate citations from LLM output against corpus."""
    patterns = [
        r"(?:Art\.)\s*\d+[a-z]?(?:\s*Abs\.\s*\d+)?(?:\s*[A-ZÄÖÜ]{2,})?(?:\s*\d+)?",
        r"BGE\s+\d+\s+[IV]+\s+\d+",
    ]
    combined = "|".join(f"(?:{p})" for p in patterns)
    regex = re.compile(combined, re.IGNORECASE)

    raw_citations = regex.findall(llm_output)

    normalized = []
    for c in raw_citations:
        cleaned = " ".join(c.split())
        if cleaned in valid_citations:
            normalized.append(cleaned)

    normalized = list(dict.fromkeys(normalized))

    if not normalized and fallback_citation:
        return fallback_citation

    if not normalized:
        return ""

    return ";".join(normalized)


def train_and_use_llm(
    reranked_results: list[dict[str, Any]],
    corpus_df: pd.DataFrame,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> pl.DataFrame:
    logger.info(f"Initializing LLM Phase with {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ty:ignore[invalid-assignment,unresolved-attribute]

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    valid_citations = set(corpus_df["citation"].astype(str).str.strip().str.strip())

    final_output = []
    logger.info(f"Running inference on {len(reranked_results)} test queries...")
    for item in reranked_results:
        ctx = "\n".join([f"{c['citation']}: {c['text']}" for c in item["top_candidates"]])[:3000]

        messages = [
            {
                "role": "system",
                "content": (
                    "Always output at least one citation ID. If uncertain, make your best guess. "
                    "Output citation IDs separated by semicolons without additional text."
                ),
            },
            {"role": "user", "content": f"Context: {ctx}\n\nQuery: {item['query']}"},
        ]

        prompt = tokenizer.apply_chat_template(  # ty:ignore[unresolved-attribute]
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # ty:ignore[call-non-callable]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)  # ty:ignore[invalid-argument-type]

        raw_pred = tokenizer.decode(  # ty:ignore[unresolved-attribute]
            out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        fallback = item["top_candidates"][0]["citation"] if item["top_candidates"] else None
        validated = parse_and_validate_citations(
            raw_pred.strip(),  # ty:ignore[unresolved-attribute]
            valid_citations,
            fallback,
        )

        final_output.append({"query_id": item["query_id"], "predicted_citations": validated})

    logger.success("LLM generation complete.")
    return pl.DataFrame(final_output)


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    logger.info("Starting Legal RAG Pipeline...")

    # 1. Load Data
    laws_p = pl.read_csv(DATA_DIR / "laws_de.csv").to_pandas()
    courts_p = pl.read_csv(DATA_DIR / "court_considerations.csv").to_pandas()
    val_p = pl.read_csv(DATA_DIR / "val.csv").to_pandas()
    train_p = pl.read_csv(DATA_DIR / "train.csv").to_pandas()
    test_p = pl.read_csv(DATA_DIR / "test.csv").to_pandas()

    # 2. Setup
    training_df = prepare_val_training_data(train_p, laws_p, courts_p)
    val_p = prepare_val_training_data(val_p, laws_p, courts_p)
    full_corpus = pd.concat([laws_p, courts_p], ignore_index=True)

    if settings.dev:
        logger.warning("DEV mode is ON: Sampling training data and corpus for speed.")
        training_df = training_df.sample(n=min(100, len(training_df)), random_state=42)

        gold_cits = set()
        for cits in training_df["gold_citations"]:
            gold_cits.update([c.strip() for c in cits.split(";")])

        gold_corpus = full_corpus[full_corpus["citation"].isin(gold_cits)]
        other_corpus = full_corpus[~full_corpus["citation"].isin(gold_cits)]
        sampled_other = other_corpus.sample(n=min(1000 - len(gold_corpus), len(other_corpus)), random_state=42)
        full_corpus = pd.concat([gold_corpus, sampled_other], ignore_index=True)

    # 3. Pipeline
    test_candidates = run_embedder(training_df=training_df, corpus_df=full_corpus, test_df=test_p, train=settings.train)
    reranked_list = run_reranker(training_df=training_df, test_candidates=test_candidates, train=settings.train)
    submission_df = train_and_use_llm(reranked_list, corpus_df=full_corpus)

    # 4. Finalize
    submission_df.write_csv("submission.csv")
    logger.success("Pipeline finished successfully. Output saved to submission.csv")
