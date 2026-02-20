#!/usr/bin/env bash
set -euo pipefail

# Two-stage OpenOneRec (video task) benchmark.
#
# Mirrors the generation pipeline in
#   OpenOneRec/benchmarks/benchmark/base_generator.py
#
#   Stage 1: sampling to generate thinking (stop at </think>)
#   Stage 2: beam search with prompt_token (<|sid_begin|>) for SID generation
#
# If --enable-thinking is omitted, runs single-stage beam search instead.
#
# Prerequisites:
#   1) Start a vLLM server with sufficient max-logprobs:
#        vllm serve <model> --max-logprobs 64 [engine args]
#   2) Run this script.
#
# Adjust --dataset-path to point to either:
#   - a parquet file containing a 'messages' column, OR
#   - a directory containing video/video_test.parquet or video_test.parquet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- tunables (override via env) ----------
MODEL=${MODEL:?"Set MODEL to the served model name"}
TOKENIZER=${TOKENIZER:-"$MODEL"}
DATASET_PATH=${DATASET_PATH:-"/path/to/benchmark_data"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8000}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
ENABLE_THINKING=${ENABLE_THINKING:-true}

# Stage 1 (thinking) params
NUM_RETURN_THINKING=${NUM_RETURN_THINKING:-1}
MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS:-1000}
THINKING_TEMP=${THINKING_TEMP:-0.6}
THINKING_TOP_P=${THINKING_TOP_P:-0.95}
THINKING_TOP_K=${THINKING_TOP_K:-50}

# Stage 2 (beam search) params
NUM_BEAMS=${NUM_BEAMS:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3}
PROMPT_TOKEN=${PROMPT_TOKEN:-"<|sid_begin|>"}

MAX_CONCURRENT=${MAX_CONCURRENT:-64}

# ---------- build CLI args ----------
THINKING_FLAG=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_FLAG="--enable-thinking"
fi

python "$SCRIPT_DIR/bench_two_stage.py" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --num-return-thinking-sequences "$NUM_RETURN_THINKING" \
  --max-new-thinking-tokens "$MAX_THINKING_TOKENS" \
  --thinking-temperature "$THINKING_TEMP" \
  --thinking-top-p "$THINKING_TOP_P" \
  --thinking-top-k "$THINKING_TOP_K" \
  --num-beams "$NUM_BEAMS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prompt-token "$PROMPT_TOKEN" \
  --max-concurrent "$MAX_CONCURRENT" \
  $THINKING_FLAG
