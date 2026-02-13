#!/usr/bin/env bash
set -euo pipefail

# Two-stage OpenOneRec (video task) benchmark via `vllm bench serve`.
#
# Mirrors the generation pipeline in
#   OpenOneRec/benchmarks/benchmark/base_generator.py
#
#   Stage 1: sampling to generate thinking (stop at </think>)
#   Stage 2: beam search with prompt_token (<|sid_begin|>) for SID generation
#
# Prerequisites:
#   1) Start a vLLM server with sufficient max-logprobs:
#        vllm serve <model> --max-logprobs 64 [engine args]
#   2) Run this script.
#
# Uses --backend openai (/v1/completions) since prompts are pre-templated
# via apply_chat_template. The two-stage logic inside serve.py builds
# stage-1 and stage-2 extra_body automatically.
#
# Adjust --dataset-path to point to either:
#   - a parquet file containing a 'messages' column, OR
#   - a directory containing video/video_test.parquet or video_test.parquet

# Path to the vllm binary. By default, use the local vllm_genrec entry point.
# Override with: VLLM_BIN=/path/to/your/vllm ./run.sh
VLLM_BIN=${VLLM_BIN:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/vllm_genrec/vllm"}

DATASET_PATH=${DATASET_PATH:-"/path/to/benchmark_data"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8000}
NUM_PROMPTS=${NUM_PROMPTS:-1000}

"$VLLM_BIN" bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --host "$HOST" \
  --port "$PORT" \
  --dataset-name openopenrec \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --openopenrec-enable-thinking \
  --openopenrec-num-beams "${NUM_BEAMS:-32}" \
  --openopenrec-max-new-tokens "${MAX_NEW_TOKENS:-3}" \
  --openopenrec-max-thinking-tokens "${MAX_THINKING_TOKENS:-1000}" \
  --openopenrec-num-return-thinking "${NUM_RETURN_THINKING:-1}" \
  --openopenrec-thinking-temperature "${THINKING_TEMP:-0.6}" \
  --openopenrec-thinking-top-p "${THINKING_TOP_P:-0.95}" \
  --openopenrec-thinking-top-k "${THINKING_TOP_K:-50}" \
  --openopenrec-prompt-token "${PROMPT_TOKEN:-<|sid_begin|>}" \
  --openopenrec-k-values "${K_VALUES:-1,32}" \
  --save-detailed \
  --ready-check-timeout-sec 120
