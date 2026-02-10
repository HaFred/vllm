#!/usr/bin/env bash
set -euo pipefail

# Example: profile vLLM OpenAI-compatible server using OpenOneRec (OpenOpenRec) video-style prompts.
#
# 1) Start server (example)
#    vllm serve <your_model> <engine args>
#
# 2) Run benchmark client
#
# Notes:
# - Use --backend openai-chat since prompts are built with tokenizer.apply_chat_template.
# - Pass OpenOneRec-style decoding params via --extra-body.
# - Adjust --dataset-path to point to either:
#     - a parquet file containing a 'messages' column, OR
#     - a directory containing video/video_test.parquet or video_test.parquet

DATASET_PATH=${DATASET_PATH:-"/path/to/data_v1.0"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8000}
NUM_PROMPTS=${NUM_PROMPTS:-1000}

# If your server supports these OpenAI-compatible fields, this approximates:
#   --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1
# You may need to adjust field names to match your server.
EXTRA_BODY=${EXTRA_BODY:-'{"n":32,"use_beam_search":true,"best_of":32}'}

vllm bench serve \
  --backend openai-chat \
  --host "$HOST" \
  --port "$PORT" \
  --dataset-name openopenrec \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --extra-body "$EXTRA_BODY" \
  --openopenrec-enable-thinking
