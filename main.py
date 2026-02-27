import gc
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer, losses
from sentence_transformers.util import semantic_search
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# --- LOGGING CONFIGURATION ---
# Configures Loguru to output to both the console and a file

DATA_DIR = Path("./data/")

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


def train_and_use_embedder(
    val_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    top_k: int = 50,
) -> list[dict[str, Any]]:
    logger.info(f"Initializing Embedder Phase with {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={"dtype": torch.bfloat16})

    # Truncate gold_context to avoid OOM during fine-tuning
    train_examples = [InputExample(texts=[r["query"], r["gold_context"][:4000]]) for _, r in val_df.iterrows()]
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
        corpus_df["text"].values,
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


def train_and_use_reranker(
    val_df: pd.DataFrame,
    test_candidates: list[dict[str, Any]],
    model_name: str = "BAAI/bge-reranker-base",
    top_n: int = 10,
) -> list[dict[str, Any]]:
    logger.info(f"Initializing Reranker Phase with {model_name}")
    model = CrossEncoder(model_name, num_labels=1)

    train_examples = []
    for _, r in val_df.iterrows():
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


def train_and_use_llm(
    val_df: pd.DataFrame, reranked_results: list[dict[str, Any]], model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
) -> pl.DataFrame:
    logger.info(f"Initializing LLM Phase with {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load in bfloat16 for performance/memory balance
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    lora_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    def format_chatml(query: str, context: str, gold: str) -> str:
        return (
            f"<|im_start|>system\nOutput ONLY citation IDs separated by semicolons.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context}\n\nQuery: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n{gold}<|im_end|>"
        )

    logger.debug("Formatting validation prompts for SFT...")
    val_prompts = [
        format_chatml(r["query"], r["gold_context"][:1500], r["gold_citations"]) for _, r in val_df.iterrows()
    ]

    trainer = SFTTrainer(
        model=model,  # ty:ignore[invalid-argument-type]
        train_dataset=Dataset.from_dict({"text": val_prompts}),
        dataset_text_field="text",  # ty:ignore[unknown-argument]
        max_seq_length=1024,  # ty:ignore[unknown-argument]
        args=TrainingArguments(
            output_dir="./tmp", per_device_train_batch_size=1, num_train_epochs=1, fp16=True, logging_steps=1
        ),
    )

    logger.info("Beginning LoRA fine-tuning on LLM...")
    trainer.train()

    final_output = []
    logger.info(f"Running inference on {len(reranked_results)} test queries...")
    for item in reranked_results:
        ctx = "\n".join([f"{c['citation']}: {c['text']}" for c in item["top_candidates"]])[:3000]

        messages = [
            {"role": "system", "content": "Output ONLY the relevant citation IDs separated by semicolons."},
            {"role": "user", "content": f"Context: {ctx}\n\nQuery: {item['query']}"},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, temperature=0.0)

        pred = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        final_output.append({"query_id": item["query_id"], "predicted_citations": pred.strip()})

    logger.success("LLM generation complete.")
    return pl.DataFrame(final_output)


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    logger.info("Starting Legal RAG Pipeline...")

    # 1. Load Data
    laws_p = pl.read_csv(DATA_DIR / "laws_de.csv").to_pandas()
    courts_p = pl.read_csv(DATA_DIR / "court_considerations.csv").to_pandas()
    val_p = pl.read_csv(DATA_DIR / "val.csv").to_pandas()
    test_p = pl.read_csv(DATA_DIR / "test.csv").to_pandas()

    # 2. Setup
    val_p = prepare_val_training_data(val_p, laws_p, courts_p)
    full_corpus = pd.concat([laws_p, courts_p], ignore_index=True)

    # 3. Pipeline
    test_candidates = train_and_use_embedder(val_p, full_corpus, test_p)
    reranked_list = train_and_use_reranker(val_p, test_candidates)
    submission_df = train_and_use_llm(val_p, reranked_list)

    # 4. Finalize
    submission_df.write_csv("submission.csv")
    logger.success("Pipeline finished successfully. Output saved to submission.csv")
