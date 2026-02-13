#!/usr/bin/env bash
set -euo pipefail

# Non-invasive two-stage OpenOneRec benchmark using offline vLLM (no server).
#
# Mirrors the generation pipeline in
#   vllm_genrec/vllm/benchmarks/serve.py (benchmark_openopenrec_two_stage)
# but uses `import vllm` directly with vllm.LLM instead of HTTP API.
#
#   Stage 1: sampling to generate thinking (stop at </think>)
#   Stage 2: beam search with prompt_token (<|sid_begin|>) for SID generation
#
# Usage:
#   MODEL=/path/to/model DATASET_PATH=/path/to/data ./run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:?"MODEL must be set (model name or path)"}
DATASET_PATH=${DATASET_PATH:?"DATASET_PATH must be set (parquet file or directory)"}
TOKENIZER=${TOKENIZER:-"$MODEL"}
NUM_PROMPTS=${NUM_PROMPTS:-1000}

python "$SCRIPT_DIR/bench.py" \
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
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  ${ENABLE_PREFIX_CACHING:+--enable-prefix-caching} \
  --save-result \
  --save-detailed \
  ${RESULT_DIR:+--result-dir "$RESULT_DIR"}
