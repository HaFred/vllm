---
name: beam-search-prefix-caching
overview: Add explicit prefix-caching support and concurrency-friendly behavior to the offline `LLM.beam_search` API, mirroring how `LLM.generate` integrates with vLLM v1 prefix caching.
todos:
  - id: extend-beamsearchparams
    content: Extend `BeamSearchParams` in `vllm/sampling_params.py` with an optional `skip_reading_prefix_cache` flag and documentation.
    status: completed
  - id: wire-cache-flag-llm-beam
    content: Plumb `BeamSearchParams.skip_reading_prefix_cache` into the internal `SamplingParams` used by `LLM.beam_search`, and reuse a single `SamplingParams` instance across decoding steps.
    status: completed
  - id: doc-concurrency-limit
    content: Tighten and document `concurrency_limit` behavior in `LLM.beam_search` to clarify its impact on batching and ensure safe bounds.
    status: completed
  - id: tests-beam-prefix-cache
    content: Add or extend tests in `tests/samplers/test_beam_search.py` to cover prefix caching and concurrency behavior for offline beam search.
    status: completed
isProject: false
---

## Goal

Implement prefix caching support for offline `LLM.beam_search` in `vllm/entrypoints/llm.py`, making its interaction with the v1 prefix cache explicit and configurable, and ensure the implementation works well with the existing `concurrency_limit` knob for efficient multi-prompt beam search.

## Key design points

- **Prefix caching semantics**
  - Prefix caching is controlled globally by engine config (`enable_prefix_caching`) and per-request by `SamplingParams.skip_reading_prefix_cache` / `PoolingParams.skip_reading_prefix_cache`.
  - Today `LLM.beam_search` internally calls `LLM.generate` with a fresh `SamplingParams` at every decoding step but does not expose any way for callers to control `skip_reading_prefix_cache` specifically for beam search.
  - The plan is to **extend `BeamSearchParams` to carry a cache control flag** and **plumb it into the internal `SamplingParams` created inside beam search**, so callers can opt in/out of reading from the prefix cache explicitly while still relying on the global engine setting for whether the cache exists.
- **Concurrency model**
  - `LLM.beam_search` already exposes a `concurrency_limit` that chunks the input prompts and, for each chunk, batches all active beams into a single `LLM.generate` call per decoding step.
  - The engine itself handles GPU-side batching across that `prompts_batch`, so the main levers we have are: chunk sizing (`concurrency_limit`) and avoiding unnecessary synchronization/overhead between steps.
  - The plan is to **keep the public API but make the behavior clearer and slightly more efficient**:
    - Ensure we construct the internal `SamplingParams` **once per `beam_search` call** and reuse it across decoding steps, rather than rebuilding it on every token.
    - Document and test that `concurrency_limit=None` (or larger than number of prompts) drives full-device utilization, while smaller limits trade throughput for memory.

## File-level changes

- **[vllm/sampling_params.py](vllm/sampling_params.py)**
  - **Extend `BeamSearchParams**`:
    - Add an optional field mirroring other param types, e.g. `skip_reading_prefix_cache: bool | None = None`.
    - (Optional) Add a short docstring comment clarifying that this only affects *reading* from the prefix cache; writes can still occur when the engine has prefix caching enabled.
  - This keeps `BeamSearchParams` consistent with `SamplingParams` / `PoolingParams` and provides a clear carrier for cache behavior from higher-level APIs (offline LLM and OpenAI serving) down into the core.
- **[vllm/entrypoints/llm.py](vllm/entrypoints/llm.py)**
  - **Update `LLM.beam_search` implementation**:
    - When constructing `beam_search_params = SamplingParams(...)` for the per-step `generate` call, plumb through the cache flag from `BeamSearchParams`:
      - If `params.skip_reading_prefix_cache` is `True`, set `beam_search_params.skip_reading_prefix_cache = True` to completely bypass prefix-cache reads during beam search.
      - If it is `False`, set `beam_search_params.skip_reading_prefix_cache = False` to force cache reads when available.
      - If it is `None`, allow `SamplingParams.__post_init__` to fall back to its existing default (which treats the absence of `prompt_logprobs` as "OK to read from cache"), preserving current behavior.
    - Construct `beam_search_params` **once before the token loop** and reuse it for each decoding step (it already has `max_tokens=1` and `skip_clone=True`, so reusing it is safe and avoids per-step allocations).
    - Keep the existing beam expansion logic, but ensure that any new cache-related configuration does not alter the observable decoding behavior (scores, ordering) beyond performance.
  - **Concurrency behavior**:
    - Retain the current `concurrency_limit` API and chunking strategy over prompts.
    - Clarify via docstring/comments that, for a fixed beam width and total number of prompts, you get maximum throughput with `concurrency_limit >= len(prompts)` and that smaller values can be used to bound memory.
    - Optionally, add a small guard so that `concurrency_limit` is always at least `1` and at most `len(prompts)` to avoid degenerate values.
- **Tests
  - [tests/samplers/test_beam_search.py](tests/samplers/test_beam_search.py)**
  - Add/extend tests to cover the new prefix-caching behavior for offline beam search:
    - A new test that constructs an `LLM` with `enable_prefix_caching=True`, runs `generate` and `beam_search` on prompts with shared prefixes, and verifies that:
      - The model runs successfully and beam search outputs match expected shapes/content.
      - When `BeamSearchParams.skip_reading_prefix_cache=True`, behavior is still correct, and (optionally) prefix-cache read metrics do not increase compared to a baseline (using `llm.llm_engine.get_metrics()` or `num_cached_tokens` if feasible).
    - A small test ensuring that `BeamSearchParams(skip_reading_prefix_cache=False)` successfully plumbs through to an internal `SamplingParams` instance (e.g. by probing the engine request objects or logging in a controlled test harness).
    - A smoke test that exercises `concurrency_limit` with `> 1` prompts to ensure batching logic and beam grouping remain correct after the refactor.

## How this addresses your request

- **Prefix caching for `llm.beam_search**`
  - The internal per-step generation calls will now explicitly honor a cache-control flag surfaced through `BeamSearchParams`, aligning beam search with how `SamplingParams`/`PoolingParams` interact with the v1 prefix cache.
  - With `enable_prefix_caching=True` at engine level and `skip_reading_prefix_cache` left as `None` or `False`, beam search will consistently read from and benefit from the prefix cache for repeated prefixes across beams and prompts.
- **Concurrency-aware beam search**
  - The refactored implementation continues to batch **all active beams for up to `concurrency_limit` prompts** in each decoding step into a single `generate` call, which lets the engine exploit GPU parallelism fully.
  - Clearer documentation and tests around `concurrency_limit` will make it easier to choose values that maximize throughput for given beam width and prompt counts while avoiding regressions in correctness.

