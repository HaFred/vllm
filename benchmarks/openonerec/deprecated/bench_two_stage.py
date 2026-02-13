#!/usr/bin/env python3
"""
Two-stage benchmark for OpenOneRec recommendation tasks.

Mirrors the two-stage generation in
  OpenOneRec/benchmarks/benchmark/base_generator.py

Stage 1 (thinking):
    Sampling with top_p/top_k, stop at </think>.
    Produces `num_return_thinking_sequences` thinking candidates per prompt.

Stage 2 (beam search):
    For each thinking candidate, append </think> + prompt_token,
    then beam search to generate SID sequences.

When --enable-thinking is NOT set, runs single-stage beam search with
prompt_token appended directly (no thinking stage).

Requires a running vLLM server (e.g. `vllm serve <model> --max-logprobs 64`).
Uses the /v1/completions endpoint for full prompt control.
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_parquet(dataset_path: str) -> pd.DataFrame:
    """Load OpenOneRec parquet file (same search order as datasets.py)."""
    p = Path(dataset_path)
    if p.suffix == ".parquet" and p.is_file():
        return pd.read_parquet(p)
    for candidate in [
        p / "video" / "video_test.parquet",
        p / "video_test.parquet",
    ]:
        if candidate.is_file():
            return pd.read_parquet(candidate)
    raise FileNotFoundError(
        f"No parquet found at {dataset_path}. "
        "Expected a .parquet file or a directory with video/video_test.parquet"
    )


def build_prompts(
    df: pd.DataFrame,
    tokenizer,
    enable_thinking: bool,
    num_prompts: int,
) -> dict[str, str]:
    """Apply chat template to each row and return {sample_id: prompt_text}."""
    prompts: dict[str, str] = {}
    for idx, row in df.iterrows():
        if len(prompts) >= num_prompts:
            break
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts[str(idx)] = prompt_text
    return prompts


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _post_json(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload) as resp:
        body = await resp.json()
        if resp.status != 200:
            err = body.get("error", body)
            raise RuntimeError(f"HTTP {resp.status}: {err}")
        return body


async def _send_with_semaphore(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    pbar: tqdm | None = None,
) -> dict:
    async with sem:
        result = await _post_json(session, url, payload)
        if pbar is not None:
            pbar.update(1)
        return result


# ---------------------------------------------------------------------------
# Stage 1 – thinking (sampling)
# ---------------------------------------------------------------------------

async def run_stage1(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompts: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    """Send sampling requests with stop=["</think>"]."""
    sem = asyncio.Semaphore(args.max_concurrent)
    tasks: list[tuple[str, asyncio.Task]] = []

    pbar = tqdm(total=len(prompts), desc="Stage 1 (thinking)")
    for sample_id, prompt_text in prompts.items():
        payload = {
            "model": model,
            "prompt": prompt_text,
            "max_tokens": args.max_new_thinking_tokens,
            "temperature": args.thinking_temperature,
            "top_p": args.thinking_top_p,
            "n": args.num_return_thinking_sequences,
            "stop": ["</think>"],
            "stream": False,
        }
        if args.thinking_top_k > 0:
            payload["top_k"] = args.thinking_top_k
        t = asyncio.create_task(
            _send_with_semaphore(sem, session, url, payload, pbar)
        )
        tasks.append((sample_id, t))

    results: dict[str, list[str]] = {}
    for sample_id, task in tasks:
        try:
            resp = await task
            results[sample_id] = [c["text"] for c in resp["choices"]]
        except Exception as exc:
            print(f"  [WARN] Stage 1 failed for {sample_id}: {exc}")
    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Stage 2 – beam search
# ---------------------------------------------------------------------------

async def run_stage2(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    stage2_prompts: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    """Send beam search requests for each thinking candidate."""
    sem = asyncio.Semaphore(args.max_concurrent)
    tasks: list[tuple[str, asyncio.Task]] = []

    pbar = tqdm(total=len(stage2_prompts), desc="Stage 2 (beam search)")
    for sample_id, prompt_text in stage2_prompts.items():
        payload = {
            "model": model,
            "prompt": prompt_text,
            "max_tokens": args.max_new_tokens,
            "n": args.num_beams,
            "best_of": args.num_beams,
            "use_beam_search": True,
            "temperature": 0.0,
            "stream": False,
        }
        t = asyncio.create_task(
            _send_with_semaphore(sem, session, url, payload, pbar)
        )
        tasks.append((sample_id, t))

    results: dict[str, list[str]] = {}
    for sample_id, task in tasks:
        try:
            resp = await task
            results[sample_id] = [c["text"] for c in resp["choices"]]
        except Exception as exc:
            print(f"  [WARN] Stage 2 failed for {sample_id}: {exc}")
    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Single-stage (no thinking) – beam search with prompt_token appended
# ---------------------------------------------------------------------------

async def run_single_stage(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompts: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    """Single-stage beam search (prompt_token appended, no thinking)."""
    sem = asyncio.Semaphore(args.max_concurrent)
    tasks: list[tuple[str, asyncio.Task]] = []

    pbar = tqdm(total=len(prompts), desc="Single-stage (beam search)")
    for sample_id, prompt_text in prompts.items():
        full_prompt = prompt_text + args.prompt_token
        payload = {
            "model": model,
            "prompt": full_prompt,
            "max_tokens": args.max_new_tokens,
            "n": args.num_beams,
            "best_of": args.num_beams,
            "use_beam_search": True,
            "temperature": 0.0,
            "stream": False,
        }
        t = asyncio.create_task(
            _send_with_semaphore(sem, session, url, payload, pbar)
        )
        tasks.append((sample_id, t))

    results: dict[str, list[str]] = {}
    for sample_id, task in tasks:
        try:
            resp = await task
            results[sample_id] = [c["text"] for c in resp["choices"]]
        except Exception as exc:
            print(f"  [WARN] Failed for {sample_id}: {exc}")
    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-stage OpenOneRec benchmark (thinking + beam search)"
    )
    # Server
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--base-url", type=str, default=None,
                   help="Override base URL (e.g. http://host:port)")
    p.add_argument("--model", type=str, required=True,
                   help="Model name served by vLLM")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer name/path (defaults to --model)")

    # Dataset
    p.add_argument("--dataset-path", type=str, required=True,
                   help="Path to parquet file or directory")
    p.add_argument("--num-prompts", type=int, default=1000)

    # Thinking (stage 1)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Enable two-stage thinking + beam search")
    p.add_argument("--num-return-thinking-sequences", type=int, default=1,
                   help="Thinking candidates per prompt (stage 1 n)")
    p.add_argument("--max-new-thinking-tokens", type=int, default=1000,
                   help="Max tokens for thinking generation (stage 1)")
    p.add_argument("--thinking-temperature", type=float, default=0.6)
    p.add_argument("--thinking-top-p", type=float, default=0.95)
    p.add_argument("--thinking-top-k", type=int, default=50)

    # Beam search (stage 2 / single stage)
    p.add_argument("--num-beams", type=int, default=32,
                   help="Beam width for stage 2 (or single-stage)")
    p.add_argument("--max-new-tokens", type=int, default=3,
                   help="Max tokens for beam search output")
    p.add_argument("--prompt-token", type=str, default="<|sid_begin|>",
                   help="Token appended before beam search generation")

    # Concurrency
    p.add_argument("--max-concurrent", type=int, default=64,
                   help="Max concurrent HTTP requests")

    return p.parse_args()


async def main():
    args = parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    completions_url = f"{base_url}/v1/completions"
    tokenizer_name = args.tokenizer or args.model

    # Load tokenizer + dataset
    print(f"Tokenizer : {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"Dataset   : {args.dataset_path}")
    df = load_parquet(args.dataset_path)

    print(f"Building prompts (enable_thinking={args.enable_thinking}) ...")
    prompts = build_prompts(df, tokenizer, args.enable_thinking, args.num_prompts)
    print(f"Prompts   : {len(prompts)}")

    connector = aiohttp.TCPConnector(limit=args.max_concurrent)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        if not args.enable_thinking:
            # ---- Single-stage: append prompt_token, beam search ----
            print(f"\n{'='*60}")
            print("Mode: single-stage beam search (no thinking)")
            print(f"  prompt_token={args.prompt_token}")
            print(f"  num_beams={args.num_beams}  max_tokens={args.max_new_tokens}")
            print(f"{'='*60}\n")

            t0 = time.perf_counter()
            results = await run_single_stage(
                session, completions_url, args.model, prompts, args
            )
            elapsed = time.perf_counter() - t0

            print(f"\nCompleted {len(results)}/{len(prompts)} samples")
            print(f"Time      : {elapsed:.2f}s")
            print(f"Throughput: {len(results)/elapsed:.2f} samples/s")
            avg_seqs = sum(len(v) for v in results.values()) / max(len(results), 1)
            print(f"Avg seqs/sample: {avg_seqs:.1f}")
            return

        # ---- Two-stage: thinking + beam search ----
        print(f"\n{'='*60}")
        print("Mode: two-stage (thinking + beam search)")
        print(f"  Stage 1: n={args.num_return_thinking_sequences}, "
              f"max_tokens={args.max_new_thinking_tokens}, "
              f"temp={args.thinking_temperature}, "
              f"top_p={args.thinking_top_p}, top_k={args.thinking_top_k}")
        print(f"  Stage 2: num_beams={args.num_beams}, "
              f"max_tokens={args.max_new_tokens}, "
              f"prompt_token={args.prompt_token}")
        print(f"{'='*60}\n")

        # Stage 1
        t_s1 = time.perf_counter()
        stage1_results = await run_stage1(
            session, completions_url, args.model, prompts, args
        )
        stage1_time = time.perf_counter() - t_s1
        print(f"Stage 1 done: {len(stage1_results)}/{len(prompts)} samples, "
              f"{stage1_time:.2f}s\n")

        # Build stage 2 prompts
        stage2_prompts: dict[str, str] = {}
        for sample_id, thinking_list in stage1_results.items():
            for idx, thinking_text in enumerate(thinking_list):
                tid = f"{sample_id}_thinking_{idx}"
                suffix = thinking_text + "</think>\n" + args.prompt_token
                stage2_prompts[tid] = prompts[sample_id] + suffix

        print(f"Stage 2 candidates: {len(stage2_prompts)}")

        # Stage 2
        t_s2 = time.perf_counter()
        stage2_results = await run_stage2(
            session, completions_url, args.model, stage2_prompts, args
        )
        stage2_time = time.perf_counter() - t_s2
        print(f"Stage 2 done: {len(stage2_results)}/{len(stage2_prompts)} candidates, "
              f"{stage2_time:.2f}s\n")

        # Merge
        final_results: dict[str, list[str]] = defaultdict(list)
        for tid, sequences in stage2_results.items():
            original_id = tid.rsplit("_thinking_", 1)[0]
            thinking_idx = int(tid.rsplit("_thinking_", 1)[1])
            thinking_text = stage1_results[original_id][thinking_idx]
            for seq in sequences:
                combined = f"{thinking_text}</think>\n{args.prompt_token}{seq}"
                final_results[original_id].append(combined)

        total_time = stage1_time + stage2_time
        print(f"{'='*60}")
        print(f"Total samples  : {len(final_results)}")
        print(f"Stage 1 time   : {stage1_time:.2f}s")
        print(f"Stage 2 time   : {stage2_time:.2f}s")
        print(f"Total time     : {total_time:.2f}s")
        print(f"Throughput     : {len(final_results)/total_time:.2f} samples/s")
        avg_seqs = (
            sum(len(v) for v in final_results.values())
            / max(len(final_results), 1)
        )
        print(f"Avg seqs/sample: {avg_seqs:.1f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
