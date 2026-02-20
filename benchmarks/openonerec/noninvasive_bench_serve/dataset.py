"""
OpenOpenRec Dataset Loading

Loads parquet data and builds prompts for offline vLLM benchmarking.
Logic is identical to:
  - vllm_genrec/vllm/benchmarks/datasets.py (OpenOpenRecDataset)
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    """A single benchmark sample with prompt, groundtruth, and metadata."""

    sample_id: str
    prompt: str
    prompt_len: int
    groundtruth: str | None = None
    metadata: dict[str, Any] | None = None


def load_parquet(dataset_path: str):
    """Load OpenOneRec parquet file (same search order as datasets.py).

    Returns a pandas DataFrame.
    """
    import pandas as pd

    parquet_paths = [
        dataset_path,
        f"{dataset_path}/video/video_test.parquet",
        f"{dataset_path}/video_test.parquet",
    ]

    parquet_path = None
    for p in parquet_paths:
        try:
            if isinstance(p, str):
                if p.endswith(".parquet") and os.path.exists(p):
                    parquet_path = p
                    break
                if not p.endswith(".parquet") and os.path.isfile(p):
                    parquet_path = p
                    break
        except Exception:
            continue

    if parquet_path is None:
        if os.path.isdir(dataset_path):
            raise FileNotFoundError(
                "Could not locate OpenOpenRec parquet under dataset_path. "
                "Expected one of: video/video_test.parquet, video_test.parquet, "
                "or dataset_path pointing directly to a .parquet file. "
                f"Got dataset_path={dataset_path!r}."
            )
        parquet_path = dataset_path

    df = pd.read_parquet(parquet_path)
    if "messages" not in df.columns:
        raise ValueError("OpenOpenRec parquet must contain a 'messages' column.")
    return df


def _convert_messages_format(messages: list) -> list:
    """Normalise content-list messages to plain text (same as datasets.py)."""
    converted = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            converted.append({"role": msg.get("role"), "content": "".join(text_parts)})
        else:
            converted.append(msg)
    return converted


def build_samples(
    df,
    tokenizer,
    num_prompts: int,
    enable_thinking: bool = False,
    seed: int = 0,
    shuffle: bool = True,
) -> list[Sample]:
    """Build Sample objects from a DataFrame.

    Mirrors ``OpenOpenRecDataset.sample()`` from datasets.py.
    """
    rows = list(df.iterrows())
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    samples: list[Sample] = []
    for i, (_, row) in enumerate(rows):
        if len(samples) >= num_prompts:
            break

        item = row.to_dict()
        messages = item.get("messages")
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                continue
        if not isinstance(messages, list):
            continue

        messages = _convert_messages_format(messages)

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_len = len(tokenizer(prompt).input_ids)

        # Extract groundtruth from metadata column
        groundtruth = None
        sample_meta = None
        raw_metadata = item.get("metadata")
        if raw_metadata is not None:
            if isinstance(raw_metadata, str):
                try:
                    sample_meta = json.loads(raw_metadata)
                except Exception:
                    sample_meta = None
            elif isinstance(raw_metadata, dict):
                sample_meta = raw_metadata
            if sample_meta is not None:
                answer = sample_meta.get("answer")
                if answer is not None:
                    groundtruth = str(answer).strip()

        samples.append(
            Sample(
                sample_id=str(i),
                prompt=prompt,
                prompt_len=prompt_len,
                groundtruth=groundtruth,
                metadata=sample_meta,
            )
        )

    return samples
