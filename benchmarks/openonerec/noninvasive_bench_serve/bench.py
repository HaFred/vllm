#!/usr/bin/env python3
"""
Non-invasive two-stage benchmark for OpenOneRec recommendation tasks.

Connects to a **running** ``vllm serve`` instance via HTTP API (aiohttp)
and performs all benchmarking externally — no modifications to vLLM source.
All code logic (two-stage generation, beam search, evaluation metrics,
performance measurement) is identical to the ``vllm bench serve`` path
in ``vllm_genrec/vllm/benchmarks/serve.py``.

Stage 1 – thinking (streaming sampling, captures TTFT / ITL).
Stage 2 – beam search (non-streaming, captures E2EL).
Evaluation – pass@k, position1_pass@k, recall@k (same as OpenOneRec).

Prerequisites:
    1) Start a vLLM server:
         vllm serve <model> --max-logprobs 64 [engine args]
    2) Run this script.

Usage:
    python bench.py \\
        --host 127.0.0.1 --port 8000 \\
        --model <model_name> \\
        --dataset-path <path_to_parquet> \\
        --num-prompts 100 \\
        --enable-thinking
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from dataset import Sample, build_samples, load_parquet
from evaluation import evaluate


# ---------------------------------------------------------------------------
# Request I/O dataclasses  (mirrors serve.py / endpoint_request_func.py)
# ---------------------------------------------------------------------------


@dataclass
class RequestOutput:
    """Per-request output with timing metrics."""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""
    start_time: float = 0.0
    all_generated_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Performance metrics  (mirrors serve.py BenchmarkMetrics + calculate_metrics)
# ---------------------------------------------------------------------------


@dataclass
class StageMetrics:
    """Aggregated performance metrics for one stage."""

    completed: int = 0
    failed: int = 0
    total_input: int = 0
    total_output: int = 0
    duration: float = 0.0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    total_token_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    percentiles_ttft_ms: list[tuple[float, float]] = field(default_factory=list)
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    percentiles_tpot_ms: list[tuple[float, float]] = field(default_factory=list)
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    percentiles_itl_ms: list[tuple[float, float]] = field(default_factory=list)
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    std_e2el_ms: float = 0.0
    percentiles_e2el_ms: list[tuple[float, float]] = field(default_factory=list)


def calculate_metrics(
    outputs: list[RequestOutput],
    prompt_lens: list[int],
    dur_s: float,
    tokenizer,
    selected_percentiles: list[float],
) -> tuple[StageMetrics, list[int]]:
    """Calculate performance metrics — same logic as serve.py calculate_metrics."""
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []

    for i, out in enumerate(outputs):
        if out.success:
            output_len = out.output_tokens
            if not output_len:
                output_len = len(
                    tokenizer(out.generated_text, add_special_tokens=False).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += prompt_lens[i]
            if output_len > 1:
                latency_minus_ttft = out.latency - out.ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            itls += out.itl
            ttfts.append(out.ttft)
            e2els.append(out.latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    failed = len(outputs) - completed
    if completed == 0:
        warnings.warn(
            "All requests failed. Check benchmark arguments / server status.",
            stacklevel=2,
        )

    total_output = sum(actual_output_lens)

    m = StageMetrics(
        completed=completed,
        failed=failed,
        total_input=total_input,
        total_output=total_output,
        duration=dur_s,
        request_throughput=completed / dur_s if dur_s > 0 else 0,
        output_throughput=total_output / dur_s if dur_s > 0 else 0,
        total_token_throughput=(total_input + total_output) / dur_s if dur_s > 0 else 0,
        # TTFT
        mean_ttft_ms=float(np.mean(ttfts or 0)) * 1000,
        median_ttft_ms=float(np.median(ttfts or 0)) * 1000,
        std_ttft_ms=float(np.std(ttfts or 0)) * 1000,
        percentiles_ttft_ms=[
            (p, float(np.percentile(ttfts or 0, p)) * 1000)
            for p in selected_percentiles
        ],
        # TPOT
        mean_tpot_ms=float(np.mean(tpots or 0)) * 1000,
        median_tpot_ms=float(np.median(tpots or 0)) * 1000,
        std_tpot_ms=float(np.std(tpots or 0)) * 1000,
        percentiles_tpot_ms=[
            (p, float(np.percentile(tpots or 0, p)) * 1000)
            for p in selected_percentiles
        ],
        # ITL
        mean_itl_ms=float(np.mean(itls or 0)) * 1000,
        median_itl_ms=float(np.median(itls or 0)) * 1000,
        std_itl_ms=float(np.std(itls or 0)) * 1000,
        percentiles_itl_ms=[
            (p, float(np.percentile(itls or 0, p)) * 1000)
            for p in selected_percentiles
        ],
        # E2EL
        mean_e2el_ms=float(np.mean(e2els or 0)) * 1000,
        median_e2el_ms=float(np.median(e2els or 0)) * 1000,
        std_e2el_ms=float(np.std(e2els or 0)) * 1000,
        percentiles_e2el_ms=[
            (p, float(np.percentile(e2els or 0, p)) * 1000)
            for p in selected_percentiles
        ],
    )
    return m, actual_output_lens


def print_stage_metrics(
    m: StageMetrics,
    stage_name: str,
    selected_percentile_metrics: list[str],
) -> None:
    """Print stage performance metrics — same format as serve.py benchmark()."""
    print("{s:{c}^{n}}".format(s=f" {stage_name} Benchmark Result ", n=60, c="="))
    print("{:<45} {:<10}".format("Successful requests:", m.completed))
    print("{:<45} {:<10}".format("Failed requests:", m.failed))
    print("{:<45} {:<10.2f}".format("Benchmark duration (s):", m.duration))
    print("{:<45} {:<10}".format("Total input tokens:", m.total_input))
    print("{:<45} {:<10}".format("Total generated tokens:", m.total_output))
    print("{:<45} {:<10.2f}".format("Request throughput (req/s):", m.request_throughput))
    print("{:<45} {:<10.2f}".format("Output token throughput (tok/s):", m.output_throughput))
    print("{:<45} {:<10.2f}".format(
        "Total token throughput (tok/s):", m.total_token_throughput))

    def _print_one(attr: str, name: str, header: str):
        if attr not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=header, n=60, c="-"))
        print("{:<45} {:<10.2f}".format(
            f"Mean {name} (ms):", getattr(m, f"mean_{attr}_ms")))
        print("{:<45} {:<10.2f}".format(
            f"Median {name} (ms):", getattr(m, f"median_{attr}_ms")))
        for p, value in getattr(m, f"percentiles_{attr}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<45} {:<10.2f}".format(f"P{p_word} {name} (ms):", value))

    _print_one("ttft", "TTFT", "Time to First Token")
    _print_one("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    _print_one("itl", "ITL", "Inter-token Latency")
    _print_one("e2el", "E2EL", "End-to-end Latency")
    print("=" * 60)


# ---------------------------------------------------------------------------
# HTTP request helpers
# ---------------------------------------------------------------------------


async def _send_streaming_completion(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    prompt_len: int,
) -> RequestOutput:
    """Streaming /v1/completions — captures TTFT, ITL, E2EL."""
    output = RequestOutput()
    output.prompt_len = prompt_len

    generated_text = ""
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st

    try:
        async with session.post(url=url, json=payload) as response:
            if response.status == 200:
                first_chunk_received = False
                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    for line in chunk_bytes.decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        data_str = line.removeprefix("data: ")
                        if data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if choices := data.get("choices"):
                            text = choices[0].get("text", "")
                            timestamp = time.perf_counter()
                            if not first_chunk_received:
                                first_chunk_received = True
                                output.ttft = timestamp - st
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            most_recent_timestamp = timestamp
                            generated_text += text
                        elif usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens", 0)
                if first_chunk_received:
                    output.success = True
                else:
                    output.success = False
                    output.error = "No valid chunks received."
                output.generated_text = generated_text
                output.latency = most_recent_timestamp - st
            else:
                output.error = f"HTTP {response.status}: {response.reason}"
                output.success = False
    except Exception:
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))

    return output


async def _send_beam_search_completion(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    prompt_len: int,
) -> RequestOutput:
    """Non-streaming /v1/completions for beam search — captures E2EL + all beams."""
    output = RequestOutput()
    output.prompt_len = prompt_len

    st = time.perf_counter()
    output.start_time = st

    try:
        async with session.post(url=url, json=payload) as response:
            if response.status == 200:
                body = await response.json()
                choices = body.get("choices", [])
                usage = body.get("usage", {})

                output.ttft = time.perf_counter() - st
                output.latency = output.ttft

                if choices:
                    choices.sort(key=lambda c: c.get("index", 0))
                    output.generated_text = choices[0].get("text", "")
                    output.all_generated_texts = [c.get("text", "") for c in choices]
                    output.success = True
                else:
                    output.success = False
                    output.error = "No choices in beam search response."

                output.output_tokens = usage.get("completion_tokens", 0)

                # Approximate ITL for metrics calculation
                if output.output_tokens > 1:
                    approx_itl = output.latency / output.output_tokens
                    output.itl = [approx_itl] * (output.output_tokens - 1)
            else:
                output.error = f"HTTP {response.status}: {response.reason}"
                output.success = False
    except Exception:
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))

    return output


# ---------------------------------------------------------------------------
# Stage 1 – thinking (streaming sampling)
# ---------------------------------------------------------------------------


async def run_stage1(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    samples: list[Sample],
    max_thinking_tokens: int,
    num_return_thinking: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_concurrent: int,
) -> tuple[list[RequestOutput], float]:
    """Send stage-1 streaming requests with stop=["</think>"]."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _send(sample: Sample) -> RequestOutput:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": sample.prompt,
            "max_tokens": max_thinking_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": num_return_thinking,
            "stop": ["</think>"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if top_k > 0:
            payload["top_k"] = top_k
        async with sem:
            result = await _send_streaming_completion(
                session, url, payload, sample.prompt_len,
            )
            pbar.update(1)
            return result

    print(f"\n{'='*60}")
    print("Stage 1/2: Thinking (sampling)")
    print(f"  n={num_return_thinking}, max_tokens={max_thinking_tokens}, "
          f"temp={temperature}, top_p={top_p}, top_k={top_k}, "
          f"stop=['</think>']")
    print(f"  prompts={len(samples)}, max_concurrent={max_concurrent}")
    print(f"{'='*60}")

    pbar = tqdm(total=len(samples), desc="Stage 1 (thinking)")
    t0 = time.perf_counter()
    tasks = [asyncio.create_task(_send(s)) for s in samples]
    outputs: list[RequestOutput] = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    pbar.close()

    completed = sum(1 for o in outputs if o.success)
    print(f"\nStage 1 done: {completed}/{len(samples)} samples, {elapsed:.2f}s")

    return outputs, elapsed


# ---------------------------------------------------------------------------
# Stage 2 – beam search (non-streaming)
# ---------------------------------------------------------------------------


async def run_stage2(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    stage2_prompts: list[str],
    stage2_prompt_lens: list[int],
    num_beams: int,
    max_new_tokens: int,
    max_concurrent: int,
) -> tuple[list[RequestOutput], float]:
    """Send stage-2 non-streaming beam search requests."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _send(prompt: str, prompt_len: int) -> RequestOutput:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "n": num_beams,
            "use_beam_search": True,
            "temperature": 0.0,
            "stream": False,
        }
        async with sem:
            result = await _send_beam_search_completion(
                session, url, payload, prompt_len,
            )
            pbar.update(1)
            return result

    print(f"\n{'='*60}")
    print("Stage 2/2: Beam search")
    print(f"  num_beams={num_beams}, max_tokens={max_new_tokens}")
    print(f"  candidates={len(stage2_prompts)}, max_concurrent={max_concurrent}")
    print(f"{'='*60}")

    pbar = tqdm(total=len(stage2_prompts), desc="Stage 2 (beam search)")
    t0 = time.perf_counter()
    tasks = [
        asyncio.create_task(_send(p, pl))
        for p, pl in zip(stage2_prompts, stage2_prompt_lens)
    ]
    outputs: list[RequestOutput] = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    pbar.close()

    completed = sum(1 for o in outputs if o.success)
    print(f"\nStage 2 done: {completed}/{len(stage2_prompts)} candidates, {elapsed:.2f}s")

    return outputs, elapsed


# ---------------------------------------------------------------------------
# Single-stage – beam search with prompt_token appended (no thinking)
# ---------------------------------------------------------------------------


async def run_single_stage(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    samples: list[Sample],
    prompt_token: str,
    num_beams: int,
    max_new_tokens: int,
    tokenizer,
    max_concurrent: int,
) -> tuple[list[RequestOutput], list[int], float]:
    """Single-stage beam search (prompt_token appended, no thinking)."""
    prompts = [s.prompt + prompt_token for s in samples]
    prompt_lens = [len(tokenizer(p).input_ids) for p in prompts]
    origin_indices = list(range(len(samples)))

    outputs, elapsed = await run_stage2(
        session, url, model,
        stage2_prompts=prompts,
        stage2_prompt_lens=prompt_lens,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        max_concurrent=max_concurrent,
    )
    return outputs, origin_indices, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Non-invasive two-stage OpenOneRec benchmark "
        "(HTTP API against running vllm serve)"
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
    p.add_argument("--trust-remote-code", action="store_true")

    # Dataset
    p.add_argument("--dataset-path", type=str, required=True,
                   help="Path to parquet file or directory")
    p.add_argument("--num-prompts", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)

    # Thinking (stage 1)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Enable two-stage thinking + beam search")
    p.add_argument("--num-return-thinking", type=int, default=1,
                   help="Thinking candidates per prompt (stage 1 n)")
    p.add_argument("--max-thinking-tokens", type=int, default=1000,
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

    # Evaluation
    p.add_argument("--k-values", type=str, default="1,32",
                   help="Comma-separated k values for evaluation metrics "
                   "(pass@k, position1_pass@k, recall@k). Default: '1,32'.")

    # Percentiles
    p.add_argument("--percentile-metrics", type=str,
                   default="ttft,tpot,itl,e2el",
                   help="Comma-separated metrics for percentile reporting.")
    p.add_argument("--metric-percentiles", type=str,
                   default="50,90,95,99",
                   help="Comma-separated percentiles to compute.")

    # Output
    p.add_argument("--save-result", action="store_true",
                   help="Save results to a JSON file")
    p.add_argument("--result-dir", type=str, default=None,
                   help="Directory for result files")
    p.add_argument("--result-filename", type=str, default=None,
                   help="Custom result filename")
    p.add_argument("--save-detailed", action="store_true",
                   help="Include per-sample evaluation details in result JSON")

    return p.parse_args()


async def main_async():
    args = parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    completions_url = f"{base_url}/v1/completions"
    tokenizer_name = args.tokenizer or args.model
    k_values = [int(x) for x in args.k_values.split(",")]
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]
    selected_percentile_metrics = args.percentile_metrics.split(",")

    # ------------------------------------------------------------------ #
    # Load tokenizer + dataset                                             #
    # ------------------------------------------------------------------ #
    from transformers import AutoTokenizer

    print(f"Server    : {base_url}")
    print(f"Tokenizer : {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=args.trust_remote_code
    )

    print(f"Dataset   : {args.dataset_path}")
    df = load_parquet(args.dataset_path)

    print(f"Building prompts (enable_thinking={args.enable_thinking}) ...")
    samples = build_samples(
        df, tokenizer, args.num_prompts,
        enable_thinking=args.enable_thinking,
        seed=args.seed,
    )
    print(f"Samples   : {len(samples)}")

    # ------------------------------------------------------------------ #
    # HTTP session                                                         #
    # ------------------------------------------------------------------ #
    connector = aiohttp.TCPConnector(
        limit=args.max_concurrent,
        limit_per_host=args.max_concurrent,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    try:
        if not args.enable_thinking:
            # ---- Single-stage: append prompt_token, beam search ----
            s2_outputs, origin_indices, stage2_time = await run_single_stage(
                session, completions_url, args.model, samples,
                args.prompt_token, args.num_beams, args.max_new_tokens,
                tokenizer, args.max_concurrent,
            )
            stage1_outputs: list[RequestOutput] = []
            stage1_time = 0.0
        else:
            # ---- Two-stage: thinking + beam search ----
            stage1_outputs, stage1_time = await run_stage1(
                session, completions_url, args.model, samples,
                max_thinking_tokens=args.max_thinking_tokens,
                num_return_thinking=args.num_return_thinking,
                temperature=args.thinking_temperature,
                top_p=args.thinking_top_p,
                top_k=args.thinking_top_k,
                max_concurrent=args.max_concurrent,
            )

            # Log stage-1 output
            for idx, (sample, s1_out) in enumerate(zip(samples, stage1_outputs)):
                if s1_out.success:
                    print(
                        f"[Stage1] idx={idx} | sample_id={sample.sample_id} | "
                        f"thinking[:200]={s1_out.generated_text[:200]!r} | "
                        f"groundtruth[:200]={(sample.groundtruth or '')[:200]!r}"
                    )

            # Build stage-2 prompts
            stage2_prompts: list[str] = []
            stage2_prompt_lens: list[int] = []
            origin_indices: list[int] = []
            continuation = "</think>\n" + args.prompt_token

            for idx, (sample, s1_out) in enumerate(zip(samples, stage1_outputs)):
                if not s1_out.success or not s1_out.generated_text:
                    continue
                suffix = s1_out.generated_text + continuation
                new_prompt = sample.prompt + suffix
                new_prompt_len = len(tokenizer(new_prompt).input_ids)
                stage2_prompts.append(new_prompt)
                stage2_prompt_lens.append(new_prompt_len)
                origin_indices.append(idx)

            s2_outputs, stage2_time = await run_stage2(
                session, completions_url, args.model,
                stage2_prompts=stage2_prompts,
                stage2_prompt_lens=stage2_prompt_lens,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                max_concurrent=args.max_concurrent,
            )

        # Log stage-2 beams
        stage2_all_texts: list[list[str]] = []
        for s2_idx, s2_out in enumerate(s2_outputs):
            orig_idx = origin_indices[s2_idx]
            sample = samples[orig_idx]
            beams = s2_out.all_generated_texts if s2_out.success else []
            stage2_all_texts.append(beams)
            print(
                f"[Stage2] orig_idx={orig_idx} | sample_id={sample.sample_id} | "
                f"n_beams={len(beams)} | "
                f"groundtruth[:200]={(sample.groundtruth or '')[:200]!r}"
            )
            for beam_idx, beam_text in enumerate(beams):
                print(
                    f"[Stage2][Beam] orig_idx={orig_idx} | "
                    f"beam={beam_idx}/{len(beams)} | text={beam_text!r}"
                )

        # -------------------------------------------------------------- #
        # Performance metrics                                              #
        # -------------------------------------------------------------- #
        if stage1_outputs:
            s1_prompt_lens = [s.prompt_len for s in samples]
            s1_metrics, _ = calculate_metrics(
                stage1_outputs, s1_prompt_lens, stage1_time,
                tokenizer, selected_percentiles,
            )
            print_stage_metrics(s1_metrics, "Stage 1", selected_percentile_metrics)

        s2_prompt_lens = [o.prompt_len for o in s2_outputs]
        s2_metrics, _ = calculate_metrics(
            s2_outputs, s2_prompt_lens, stage2_time,
            tokenizer, selected_percentiles,
        )
        print_stage_metrics(s2_metrics, "Stage 2", selected_percentile_metrics)

        # -------------------------------------------------------------- #
        # Evaluation                                                       #
        # -------------------------------------------------------------- #
        eval_samples: list[dict[str, Any]] = []
        for s2_idx in range(len(origin_indices)):
            orig_idx = origin_indices[s2_idx]
            sample = samples[orig_idx]
            beams = (
                stage2_all_texts[s2_idx]
                if s2_idx < len(stage2_all_texts)
                else []
            )
            eval_samples.append({
                "sample_id": sample.sample_id,
                "groundtruth": sample.groundtruth or "",
                "beams": beams,
            })

        eval_metrics = evaluate(eval_samples, k_values)

        # -------------------------------------------------------------- #
        # Combined summary                                                 #
        # -------------------------------------------------------------- #
        total_duration = stage1_time + stage2_time
        total_completed = s2_metrics.completed

        print(f"\n{'='*60}")
        print(f"{'OpenOpenRec Two-Stage Combined Summary':^60}")
        print(f"{'='*60}")
        if args.enable_thinking:
            print(f"{'Stage-1 completed:':<45} {s1_metrics.completed}")
            print(f"{'Stage-1 duration (s):':<45} {stage1_time:<10.2f}")
            print(f"{'Stage-1 request throughput (req/s):':<45} "
                  f"{s1_metrics.request_throughput:<10.2f}")
        print(f"{'Stage-2 completed:':<45} {s2_metrics.completed}")
        print(f"{'Stage-2 duration (s):':<45} {stage2_time:<10.2f}")
        print(f"{'Stage-2 request throughput (req/s):':<45} "
              f"{s2_metrics.request_throughput:<10.2f}")
        print(f"{'Total duration (s):':<45} {total_duration:<10.2f}")
        print(f"{'End-to-end throughput (req/s):':<45} "
              f"{total_completed / total_duration if total_duration > 0 else 0:<10.2f}")

        if eval_metrics:
            print(f"{'-'*60}")
            print(f"{'Evaluation Metrics':^60}")
            print(f"{'-'*60}")
            for k in k_values:
                print(f"{'pass@' + str(k) + ':':<45} "
                      f"{eval_metrics.get(f'pass@{k}', 0.0):<10.4f}")
                print(f"{'position1_pass@' + str(k) + ':':<45} "
                      f"{eval_metrics.get(f'position1_pass@{k}', 0.0):<10.4f}")
                print(f"{'recall@' + str(k) + ':':<45} "
                      f"{eval_metrics.get(f'recall@{k}', 0.0):<10.4f}")
            print(f"{'Evaluated / Total:':<45} "
                  f"{eval_metrics.get('evaluated_samples', 0)}/"
                  f"{eval_metrics.get('total_samples', 0)}")
        print(f"{'='*60}")

        # -------------------------------------------------------------- #
        # Save results                                                     #
        # -------------------------------------------------------------- #
        eval_per_sample = eval_metrics.pop("per_sample", {})

        def _metrics_to_dict(m: StageMetrics) -> dict[str, Any]:
            return {
                "duration": m.duration,
                "completed": m.completed,
                "failed": m.failed,
                "total_input_tokens": m.total_input,
                "total_output_tokens": m.total_output,
                "request_throughput": m.request_throughput,
                "output_throughput": m.output_throughput,
                "total_token_throughput": m.total_token_throughput,
                "mean_ttft_ms": m.mean_ttft_ms,
                "median_ttft_ms": m.median_ttft_ms,
                "std_ttft_ms": m.std_ttft_ms,
                "mean_tpot_ms": m.mean_tpot_ms,
                "median_tpot_ms": m.median_tpot_ms,
                "std_tpot_ms": m.std_tpot_ms,
                "mean_itl_ms": m.mean_itl_ms,
                "median_itl_ms": m.median_itl_ms,
                "std_itl_ms": m.std_itl_ms,
                "mean_e2el_ms": m.mean_e2el_ms,
                "median_e2el_ms": m.median_e2el_ms,
                "std_e2el_ms": m.std_e2el_ms,
            }

        result_json: dict[str, Any] = {
            "openopenrec_two_stage": args.enable_thinking,
            "model": args.model,
            "base_url": base_url,
            "num_prompts": len(samples),
            "max_concurrent": args.max_concurrent,
            "enable_thinking": args.enable_thinking,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "prompt_token": args.prompt_token,
            "k_values": k_values,
        }

        if args.enable_thinking:
            result_json["stage1"] = _metrics_to_dict(s1_metrics)
            result_json["stage1"].update({
                "num_return_thinking": args.num_return_thinking,
                "max_thinking_tokens": args.max_thinking_tokens,
                "temperature": args.thinking_temperature,
                "top_p": args.thinking_top_p,
                "top_k": args.thinking_top_k,
            })

        result_json["stage2"] = _metrics_to_dict(s2_metrics)
        result_json["total_duration"] = total_duration
        result_json["total_completed"] = total_completed
        result_json["total_throughput"] = (
            total_completed / total_duration if total_duration > 0 else 0
        )
        result_json["evaluation"] = eval_metrics

        if args.save_detailed:
            result_json["evaluation_per_sample"] = eval_per_sample

        if args.save_result:
            from datetime import datetime
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            base_model_id = args.model.split("/")[-1]

            if args.result_filename:
                file_name = args.result_filename
            else:
                file_name = f"noninvasive-{base_model_id}-{current_dt}.json"

            if args.result_dir:
                os.makedirs(args.result_dir, exist_ok=True)
                file_name = os.path.join(args.result_dir, file_name)

            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {file_name}")

    finally:
        await session.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
