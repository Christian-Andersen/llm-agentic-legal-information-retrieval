set dotenv-load

alias c := check
alias r := run
alias ci := continuous_integration

[private]
default:
    @just --list

check:
    uv run -- prek run --all-files

run *args:
    uv run -- main.py {{ args }}

audit:
    uv audit

continuous_integration: audit check
