"""Debug logging controls for vLLM V1.

Mirrors vllm_gr/debug.py so that vanilla vLLM and vLLM-GR can be compared
side-by-side with independent debug toggles.
"""

from __future__ import annotations

import os


def is_vllm_debug_logging_enabled() -> bool:
    value = os.environ.get("VLLM_DEBUG_LOGGING", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
