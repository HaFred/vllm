#!/usr/bin/env python3
"""
Non-invasive two-stage benchmark for OpenOneRec recommendation tasks.

Directly ``import vllm`` and uses ``vllm.LLM`` for offline generation,
without requiring a running vLLM server.  All code logic (two-stage
generation, beam search, evaluation metrics) is identical to the
``vllm bench serve`` path in ``vllm_genrec/vllm/benchmarks/serve.py``.

Stage 1 – thinking (sampling with top_p / top_k, stop at ``</think>``).
Stage 2 – beam search with *prompt_token* appended after thinking.
Evaluation – pass@k, position1_pass@k, recall@k (same as OpenOneRec).

Usage:
    python bench.py \
        --model <model_name_or_path> \
        --dataset-path <path_to_parquet> \
        --num-prompts 100 \
        --enable-thinking
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams

from dataset import Sample, build_samples, load_parquet
from evaluation import evaluate


# ---------------------------------------------------------------------------
# Stage 1 – thinking (sampling)
# ---------------------------------------------------------------------------


def run_stage1(
    llm: LLM,
    samples: list[Sample],
    max_thinking_tokens: int,
    num_return_thinking: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[list[list[str]], list[list[list[int]]], float]:
    """Run stage 1: sampling to generate thinking text.

    Returns:
        stage1_texts: list of (list of thinking texts) per sample.
        stage1_token_ids: list of (list of token-id sequences) per sample.
        elapsed: wall-clock time in seconds.
    """
    sampling_params = SamplingParams(
        max_tokens=max_thinking_tokens,
        n=num_return_thinking,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        stop=["</think>"],
    )

    prompts = [s.prompt for s in samples]

    print(f"\n{'='*60}")
    print("Stage 1/2: Thinking (sampling)")
    print(f"  n={num_return_thinking}, max_tokens={max_thinking_tokens}, "
          f"temp={temperature}, top_p={top_p}, top_k={top_k}, "
          f"stop=['</think>']")
    print(f"  prompts={len(prompts)}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    stage1_texts: list[list[str]] = []
    stage1_token_ids: list[list[list[int]]] = []
    for output in outputs:
        texts = [comp.text for comp in output.outputs]
        token_ids = [list(comp.token_ids) for comp in output.outputs]
        stage1_texts.append(texts)
        stage1_token_ids.append(token_ids)

    completed = sum(1 for texts in stage1_texts if any(t for t in texts))
    print(f"\nStage 1 done: {completed}/{len(samples)} samples, {elapsed:.2f}s")

    return stage1_texts, stage1_token_ids, elapsed


# ---------------------------------------------------------------------------
# Stage 2 – beam search
# ---------------------------------------------------------------------------


def run_stage2(
    llm: LLM,
    samples: list[Sample],
    stage1_texts: list[list[str]],
    stage1_token_ids: list[list[list[int]]],
    prompt_token: str,
    num_beams: int,
    max_new_tokens: int,
    use_token_ids: bool = True,
) -> tuple[list[list[str]], list[int], float]:
    """Run stage 2: beam search on stage-1 outputs.

    Builds stage-2 prompts by appending thinking + ``</think>\\n`` +
    *prompt_token* to the original prompt.

    When *use_token_ids* is True (default), constructs prompts from exact
    token IDs for APC prefix-cache reuse.  Otherwise falls back to text
    concatenation.

    Returns:
        stage2_all_beams: list of (list of beam texts) per stage-2 request.
        origin_indices: maps each stage-2 request back to the original sample index.
        elapsed: wall-clock time in seconds.
    """
    tokenizer = llm.get_tokenizer()
    continuation_text = "</think>\n" + prompt_token

    # Build stage-2 prompts
    stage2_prompts = []          # text or TokensPrompt dicts
    origin_indices: list[int] = []

    if use_token_ids:
        continuation_ids = tokenizer.encode(continuation_text, add_special_tokens=False)

    for idx, sample in enumerate(samples):
        thinking_list = stage1_texts[idx]
        for t_idx, thinking_text in enumerate(thinking_list):
            if not thinking_text:
                continue

            if use_token_ids:
                # Token-ID-based prompt for exact APC prefix match
                prompt_token_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
                thinking_ids = stage1_token_ids[idx][t_idx]
                s2_ids = prompt_token_ids + thinking_ids + continuation_ids
                stage2_prompts.append({"prompt_token_ids": s2_ids})
            else:
                # Text-based prompt (may cause tokenizer boundary mismatches)
                suffix = thinking_text + continuation_text
                stage2_prompts.append({"prompt": sample.prompt + suffix})

            origin_indices.append(idx)

    print(f"\n{'='*60}")
    print("Stage 2/2: Beam search")
    print(f"  num_beams={num_beams}, max_tokens={max_new_tokens}, "
          f"prompt_token={prompt_token!r}")
    print(f"  candidates={len(stage2_prompts)}, "
          f"use_token_ids={use_token_ids}")
    print(f"{'='*60}")

    beam_params = BeamSearchParams(
        beam_width=num_beams,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    t0 = time.perf_counter()
    outputs = llm.beam_search(stage2_prompts, beam_params)
    elapsed = time.perf_counter() - t0

    # Extract all beam texts per request
    # BeamSearchOutput.sequences is a list of BeamSearchSequence, each with .text
    stage2_all_beams: list[list[str]] = []
    for output in outputs:
        beams = [seq.text for seq in output.sequences]
        stage2_all_beams.append(beams)

    print(f"\nStage 2 done: {len(stage2_all_beams)}/{len(stage2_prompts)} candidates, "
          f"{elapsed:.2f}s")

    return stage2_all_beams, origin_indices, elapsed


# ---------------------------------------------------------------------------
# Single-stage (no thinking) – beam search with prompt_token appended
# ---------------------------------------------------------------------------


def run_single_stage(
    llm: LLM,
    samples: list[Sample],
    prompt_token: str,
    num_beams: int,
    max_new_tokens: int,
) -> tuple[list[list[str]], list[int], float]:
    """Single-stage beam search (prompt_token appended, no thinking)."""
    prompts = [{"prompt": s.prompt + prompt_token} for s in samples]
    origin_indices = list(range(len(samples)))

    print(f"\n{'='*60}")
    print("Single-stage beam search (no thinking)")
    print(f"  num_beams={num_beams}, max_tokens={max_new_tokens}, "
          f"prompt_token={prompt_token!r}")
    print(f"  prompts={len(prompts)}")
    print(f"{'='*60}")

    beam_params = BeamSearchParams(
        beam_width=num_beams,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    t0 = time.perf_counter()
    outputs = llm.beam_search(prompts, beam_params)
    elapsed = time.perf_counter() - t0

    stage2_all_beams: list[list[str]] = []
    for output in outputs:
        beams = [seq.text for seq in output.sequences]
        stage2_all_beams.append(beams)

    print(f"\nDone: {len(stage2_all_beams)}/{len(prompts)} samples, {elapsed:.2f}s")

    return stage2_all_beams, origin_indices, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Non-invasive two-stage OpenOneRec benchmark (offline vLLM)"
    )
    # Model
    p.add_argument("--model", type=str, required=True,
                   help="Model name or path for vLLM")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer name/path (defaults to --model)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dtype", type=str, default="auto",
                   help="Data type (auto, float16, bfloat16, etc.)")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Maximum model context length")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--enable-prefix-caching", action="store_true",
                   help="Enable Automatic Prefix Caching (APC) for KV cache "
                   "reuse between stage 1 and stage 2.")

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

    # Evaluation
    p.add_argument("--k-values", type=str, default="1,32",
                   help="Comma-separated k values for evaluation metrics "
                   "(pass@k, position1_pass@k, recall@k). Default: '1,32'.")

    # APC
    p.add_argument("--no-token-ids", action="store_true",
                   help="Use text-based stage-2 prompts instead of token-ID "
                   "based (disables exact APC prefix matching).")

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


def main():
    args = parse_args()

    tokenizer_name = args.tokenizer or args.model
    k_values = [int(x) for x in args.k_values.split(",")]

    # ------------------------------------------------------------------ #
    # Load tokenizer + dataset                                             #
    # ------------------------------------------------------------------ #
    from transformers import AutoTokenizer

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
    # Initialise vLLM engine                                               #
    # ------------------------------------------------------------------ #
    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len

    print(f"\nInitialising vLLM LLM engine ...")
    print(f"  model={args.model}")
    print(f"  tensor_parallel_size={args.tensor_parallel_size}")
    print(f"  enable_prefix_caching={args.enable_prefix_caching}")
    llm = LLM(**engine_kwargs)

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #
    if not args.enable_thinking:
        # ---- Single-stage: append prompt_token, beam search ----
        stage2_all_beams, origin_indices, elapsed = run_single_stage(
            llm, samples, args.prompt_token, args.num_beams, args.max_new_tokens,
        )
        stage1_time = 0.0
        stage2_time = elapsed
    else:
        # ---- Two-stage: thinking + beam search ----
        stage1_texts, stage1_token_ids, stage1_time = run_stage1(
            llm, samples,
            max_thinking_tokens=args.max_thinking_tokens,
            num_return_thinking=args.num_return_thinking,
            temperature=args.thinking_temperature,
            top_p=args.thinking_top_p,
            top_k=args.thinking_top_k,
        )

        # Log stage-1 output
        for idx, (sample, texts) in enumerate(zip(samples, stage1_texts)):
            for t_idx, text in enumerate(texts):
                print(
                    f"[Stage1] idx={idx} | sample_id={sample.sample_id} | "
                    f"thinking_{t_idx}[:200]={text[:200]!r} | "
                    f"groundtruth[:200]={(sample.groundtruth or '')[:200]!r}"
                )

        stage2_all_beams, origin_indices, stage2_time = run_stage2(
            llm, samples,
            stage1_texts=stage1_texts,
            stage1_token_ids=stage1_token_ids,
            prompt_token=args.prompt_token,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_token_ids=not args.no_token_ids,
        )

    # Log stage-2 beams
    for s2_idx, beams in enumerate(stage2_all_beams):
        orig_idx = origin_indices[s2_idx]
        sample = samples[orig_idx]
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

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    # Build evaluation sample dicts
    eval_samples: list[dict[str, Any]] = []
    for s2_idx in range(len(origin_indices)):
        orig_idx = origin_indices[s2_idx]
        sample = samples[orig_idx]
        beams = stage2_all_beams[s2_idx] if s2_idx < len(stage2_all_beams) else []
        eval_samples.append({
            "sample_id": sample.sample_id,
            "groundtruth": sample.groundtruth or "",
            "beams": beams,
        })

    eval_metrics = evaluate(eval_samples, k_values)

    # ------------------------------------------------------------------ #
    # Combined summary                                                    #
    # ------------------------------------------------------------------ #
    total_duration = stage1_time + stage2_time
    total_completed = len(stage2_all_beams)

    print(f"\n{'='*60}")
    print(f"{'OpenOpenRec Two-Stage Combined Summary':^60}")
    print(f"{'='*60}")
    if args.enable_thinking:
        print(f"{'Stage-1 completed:':<45} {len(samples)}")
        print(f"{'Stage-1 duration (s):':<45} {stage1_time:<10.2f}")
    print(f"{'Stage-2 completed:':<45} {total_completed}")
    print(f"{'Stage-2 duration (s):':<45} {stage2_time:<10.2f}")
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

    # ------------------------------------------------------------------ #
    # Save results                                                        #
    # ------------------------------------------------------------------ #
    eval_per_sample = eval_metrics.pop("per_sample", {})

    result_json: dict[str, Any] = {
        "openopenrec_two_stage": args.enable_thinking,
        "model": args.model,
        "num_prompts": len(samples),
        "enable_thinking": args.enable_thinking,
        "enable_prefix_caching": args.enable_prefix_caching,
        "use_token_ids": not args.no_token_ids,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "prompt_token": args.prompt_token,
        "k_values": k_values,
    }

    if args.enable_thinking:
        result_json["stage1"] = {
            "duration": stage1_time,
            "completed": len(samples),
            "num_return_thinking": args.num_return_thinking,
            "max_thinking_tokens": args.max_thinking_tokens,
            "temperature": args.thinking_temperature,
            "top_p": args.thinking_top_p,
            "top_k": args.thinking_top_k,
        }

    result_json["stage2"] = {
        "duration": stage2_time,
        "completed": total_completed,
    }
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

    return result_json


if __name__ == "__main__":
    main()
