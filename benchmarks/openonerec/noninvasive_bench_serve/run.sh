#!/usr/bin/env bash
set -euo pipefail

# Non-invasive two-stage OpenOneRec benchmark against a running vllm serve.
#
# Mirrors the generation pipeline in
#   vllm_genrec/vllm/benchmarks/serve.py (benchmark_openopenrec_two_stage)
# but uses HTTP API only — no modifications to vLLM source.
#
#   Stage 1: streaming sampling to generate thinking (stop at </think>)
#   Stage 2: non-streaming beam search with prompt_token (<|sid_begin|>)
#
# Prerequisites:
#   1) Start a vLLM server with --max-logprobs >= 2 * NUM_BEAMS (default 64):
#        vllm serve <model> --max-logprobs 64 [engine args]
#      (vLLM default max_logprobs=20 is too low for beam search with 32 beams)
#   2) Run this script.
#
# Usage:
#   MODEL=<model_name> DATASET_PATH=/path/to/data ./run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:?"MODEL must be set (model name served by vLLM)"}
DATASET_PATH=${DATASET_PATH:?"DATASET_PATH must be set (parquet file or directory)"}
TOKENIZER=${TOKENIZER:-"$MODEL"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8000}
NUM_PROMPTS=${NUM_PROMPTS:-1000}

python "$SCRIPT_DIR/bench.py" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --trust-remote-code \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --enable-thinking \
  --num-beams "${NUM_BEAMS:-32}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-3}" \
  --max-thinking-tokens "${MAX_THINKING_TOKENS:-1000}" \
  --num-return-thinking "${NUM_RETURN_THINKING:-1}" \
  --thinking-temperature "${THINKING_TEMP:-0.6}" \
  --thinking-top-p "${THINKING_TOP_P:-0.95}" \
  --thinking-top-k "${THINKING_TOP_K:-50}" \
  --prompt-token "${PROMPT_TOKEN:-<|sid_begin|>}" \
  --k-values "${K_VALUES:-1,32}" \
  --max-concurrent "${MAX_CONCURRENT:-64}" \
  --rps "${RPS:-inf}" \
  --save-result \
  --save-detailed \
  ${RESULT_DIR:+--result-dir "$RESULT_DIR"}
