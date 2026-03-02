# DKVC: Direct KV Cache Inheritance for Continuation Requests

**Goal**: Enable a child request (e.g. stage-2 beam search) to directly
inherit KV cache blocks from a parent request (e.g. stage-1
thinking/sampling), skipping re-tokenisation and re-prefill of the
shared prefix at the vLLM scheduler/engine level.

---

## 1. Problem Statement

In multi-stage inference pipelines (e.g. two-stage recommendation with
OpenOneRec), stage 2 sends a brand-new completions request whose prompt
is the concatenation of the original prompt + stage-1 output + suffix
tokens.

When this new request arrives at the vLLM engine it goes through the
**full request lifecycle**:

```
HTTP request
  → OpenAI serving layer (tokenize prompt string)
    → InputProcessor.process_inputs() (tokenize, preprocess)
      → EngineCoreRequest (prompt_token_ids)
        → EngineCore.preprocess_add_request()
          → Request.from_engine_core_request() (compute block hashes)
            → Scheduler.add_request() → waiting queue
              → schedule(): get_computed_blocks() (hash-based prefix lookup)
                → allocate_slots() (allocate new blocks for unmatched suffix)
                  → model forward (prefill unmatched tokens)
```

**What APC (Automatic Prefix Caching) already does**:
- At `get_computed_blocks()`, the scheduler walks the block-hash chain
  via `find_longest_cache_hit()`.
- If the stage-1 blocks haven't been evicted, the hash lookup succeeds
  and the scheduler marks those tokens as "computed" → no re-prefill.

**What APC does NOT do (the gaps DKVC addresses)**:
1. **Re-tokenisation**: The full prompt string is still re-tokenised in
   `InputProcessor.process_inputs()`.
2. **Block hash recomputation**: `Request.__init__` recomputes all block
   hashes from scratch.
3. **No guarantee**: APC is opportunistic — blocks can be evicted under
   memory pressure between stage 1 finishing and stage 2 arriving.
4. **Beam search overhead**: `beam_search()` creates
   `beam_width × max_tokens` sub-requests, each going through the full
   lifecycle.

---

## 2. Solution: "Continuation Request"

A **continuation request** is a request type that explicitly inherits
KV cache blocks from a completed (or still-running) parent request.

### Key properties (all implemented):
- **No re-tokenisation**: The engine constructs `prompt_token_ids` from
  the parent's `all_token_ids` + a small `continuation_token_ids` suffix.
- **No hash-based lookup**: The scheduler directly transfers block
  references from the parent, bypassing `find_longest_cache_hit()`.
- **Guaranteed block retention**: Block ref-counts are incremented before
  the parent is freed, preventing eviction.
- **Minimal new tokens to prefill**: Only the suffix tokens (e.g.
  `</think>\n<|sid_begin|>`) need prefill — typically < 10 tokens.

---

## 3. Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │  Client                          │
                    │  POST /v1/completions            │
                    │  { "continuation_of": "req-s1",  │
                    │    "continuation_suffix": "...", │
                    │    "use_beam_search": true }     │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Protocol         │  OpenAI Protocol (protocol.py)   │
                    │  continuation_of: str | None     │
                    │  continuation_suffix: str | None │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 8          │  OpenAI Serving Layer            │
                    │  serving_completion.py           │
                    │  Detect continuation_of field    │
                    │  → route to generate_continuation│
                    │    or beam_search(continuation)  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 9          │  Beam Search (serving_engine.py) │
                    │  First step: generate_continuation│
                    │  with fallback to regular generate│
                    │  Subsequent steps: normal APC    │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 7          │  AsyncLLM.generate_continuation()│
                    │  async_llm.py                    │
                    │  → InputProcessor.process_cont() │
                    │  → skip tokenisation             │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 2          │  InputProcessor                  │
                    │  input_processor.py               │
                    │  process_continuation()           │
                    │  Skip tokenisation entirely      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 1          │  EngineCoreRequest               │
                    │  v1/engine/__init__.py            │
                    │  continuation_of: str | None     │
                    │  continuation_token_ids: [int]   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 3          │  EngineCore.preprocess_add_req() │
                    │  v1/engine/core.py                │
                    │  Resolve parent (running or      │
                    │  finished cache, ext/int ID)     │
                    │  Build prompt = parent + suffix  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 6          │  Request.__init__()              │
                    │  v1/request.py                    │
                    │  continuation_of field            │
                    │  Block hashes computed as normal  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 5          │  Scheduler.schedule()            │
                    │  v1/core/sched/scheduler.py       │
                    │  if request.continuation_of:     │
                    │    → inherit_blocks_from_parent() │
                    │  else:                            │
                    │    → get_computed_blocks()        │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 4          │  KVCacheManager + Coordinator    │
                    │  kv_cache_manager.py              │
                    │  kv_cache_coordinator.py          │
                    │  inherit_blocks_from_parent()    │
                    │  → coordinator.transfer_blocks() │
                    │  Direct block ref transfer       │
                    │  (no hash walk, guaranteed hit)  │
                    └──────────────────────────────────┘
```

---

## 4. Implementation Details Per Layer

### 4.1 Protocol: OpenAI Request Fields

**File**: `vllm/entrypoints/openai/protocol.py`

Two fields were added to the completions request protocol:

```python
continuation_of: str | None = Field(
    default=None,
    description=(
        "If set, this request is a continuation of the specified parent "
        "request. The engine will inherit the parent's KV cache blocks "
        "directly, skipping re-tokenisation and re-prefill of the shared "
        "prefix. Only the continuation_suffix tokens need to be prefilled."
    ),
)
continuation_suffix: str | None = Field(
    default=None,
    description=(
        "Text to append after the parent request's output when using "
        "continuation_of. This small suffix is tokenised server-side. "
        "The full prompt becomes: parent's all_token_ids + "
        "tokenize(continuation_suffix)."
    ),
)
```

These fields are part of the standard request schema (not `extra_body`),
giving clients first-class API support.

---

### 4.2 Layer 1: EngineCoreRequest

**File**: `vllm/v1/engine/__init__.py`

Two optional fields carry continuation metadata through the engine:

```python
# Continuation support: inherit KV cache from a parent request.
# When set, the engine constructs prompt_token_ids from the parent's
# all_token_ids + continuation_token_ids, and the scheduler inherits
# KV cache blocks directly instead of doing hash-based prefix lookup.
continuation_of: str | None = None
continuation_token_ids: list[int] | None = None
```

`EngineCoreRequest` is a `msgspec.Struct` with `omit_defaults=True`, so
these `None`-default fields add zero serialisation overhead for
non-continuation requests.

---

### 4.3 Layer 2: InputProcessor (Skip Tokenisation)

**File**: `vllm/v1/engine/input_processor.py`

The `process_continuation()` method bypasses the tokenisation/preprocessing
pipeline entirely:

```python
def process_continuation(
    self,
    request_id: str,
    parent_request_id: str,
    continuation_token_ids: list[int],
    params: SamplingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
) -> EngineCoreRequest:
```

**What this skips**:
- `self.input_preprocessor.preprocess()` — the full tokenisation +
  multimodal preprocessing pipeline
- Multimodal hash/feature processing
- Platform validation of raw prompt

**What this preserves**:
- LoRA and data-parallel rank validation
- `SamplingParams` cloning and validation
- EOS token ID resolution
- Generation config updates

Returns an `EngineCoreRequest` with `prompt_token_ids=None` (populated
later by `EngineCore.preprocess_add_request()` from the parent's tokens).

---

### 4.4 Layer 3: EngineCore (Prompt Construction from Parent)

**File**: `vllm/v1/engine/core.py`

This is the most substantial layer. Three mechanisms work together:

#### A. Finished-request token cache

```python
# In EngineCore.__init__:
self._finished_request_tokens: OrderedDict[str, list[int]] = OrderedDict()
self._max_finished_token_cache = 1024
```

A bounded LRU cache of recently-finished requests' `all_token_ids`.
This allows continuation requests to find their parent even after the
parent has been freed from the scheduler.

#### B. Bidirectional request ID mapping

```python
self._req_id_bimap: dict[str, str] = {}
```

Maps external ↔ internal request IDs bidirectionally. This is needed
because:
- The HTTP client uses external request IDs in `continuation_of`
- The scheduler and KV cache use internal request IDs
- `cache_finished_request_tokens()` must index by both

#### C. `cache_finished_request_tokens()` method

```python
def cache_finished_request_tokens(self, req_id: str) -> None:
```

Called in three places:
1. After `scheduler.update_from_output()` returns finished requests
   (in the `step()` method)
2. After `scheduler.update_from_output()` in the batch-queue path
3. During `_abort_requests()` — needed because stop-string-triggered
   aborts never go through the normal "finished" path

Caches by **both** internal and external request IDs, and evicts the
oldest entries when over capacity.

#### D. `preprocess_add_request()` continuation handling

When `request.continuation_of` is set:

1. **Try running request**: Look up `request.continuation_of` in
   `scheduler.requests` (direct match or via `_req_id_bimap` for
   external→internal resolution).
2. **Try finished cache**: Look up in `_finished_request_tokens`.
3. **Build prompt**: `prompt_token_ids = parent_tokens + suffix`.
4. **Rewrite continuation_of**: If the parent was found via external ID,
   rewrite to the internal ID so downstream `transfer_blocks()` (which
   indexes by `req_to_blocks`) works correctly.

Block hashes are still computed by `Request.from_engine_core_request()`,
which is intentional — the child request needs its own block hashes so
its blocks can be cached for future APC hits.

---

### 4.5 Layer 4: KVCacheManager + KVCacheCoordinator (Direct Block Inheritance)

**File**: `vllm/v1/core/kv_cache_manager.py`

`inherit_blocks_from_parent()` replaces the hash-based
`get_computed_blocks()` for continuation requests:

```python
def inherit_blocks_from_parent(
    self,
    child_request: Request,
    parent_request_id: str,
) -> tuple[KVCacheBlocks, int]:
```

This delegates to `coordinator.transfer_blocks()` and falls back to
`get_computed_blocks()` if the parent's blocks are no longer available.

**File**: `vllm/v1/core/kv_cache_coordinator.py`

`transfer_blocks()` is implemented on the base `KVCacheCoordinator`
class (not just the unitary coordinator), making it available to all
coordinator types:

```python
def transfer_blocks(
    self,
    from_request_id: str,
    to_request_id: str,
) -> tuple[list[KVCacheBlock], ...] | None:
```

The implementation:
1. Looks up the parent's blocks via `get_blocks(from_request_id)`
2. For each kv_cache_group, increments `ref_cnt` on every parent block
3. Registers the inherited blocks in `manager.req_to_blocks[to_request_id]`
4. Returns the inherited blocks (or `None` if parent has no blocks)

The pre-registered blocks in `req_to_blocks` are then detected by
`allocate_new_computed_blocks()`, which skips its normal
touch/registration path for blocks already present.

**Design decision**: Hash verification is **not** performed during
transfer. Since the child's prompt is constructed from the parent's
exact `all_token_ids`, hash consistency is guaranteed by construction.
This keeps the transfer O(n) in blocks without additional overhead.

---

### 4.6 Layer 5: Scheduler (Continuation-Aware Scheduling)

**File**: `vllm/v1/core/sched/scheduler.py`

A branch in the waiting-request scheduling loop:

```python
# Get already-cached tokens.
if request.num_computed_tokens == 0:
    if getattr(request, 'continuation_of', None) is not None:
        # Continuation request: inherit blocks directly
        # from the parent instead of hash-based lookup.
        new_computed_blocks, num_new_local_computed_tokens = (
            self.kv_cache_manager.inherit_blocks_from_parent(
                request, request.continuation_of
            )
        )
    else:
        # Get locally-cached tokens.
        new_computed_blocks, num_new_local_computed_tokens = (
            self.kv_cache_manager.get_computed_blocks(request)
        )
```

Everything downstream (allocate_slots, scheduling decisions, etc.) is
unchanged because `inherit_blocks_from_parent()` returns the same
`(KVCacheBlocks, int)` tuple as `get_computed_blocks()`.

---

### 4.7 Layer 6: Request Object

**File**: `vllm/v1/request.py`

The `continuation_of` field is added to `Request.__init__()` and
passed through in `from_engine_core_request()`:

```python
def __init__(
    self,
    # ... existing params ...
    block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    continuation_of: str | None = None,
) -> None:
    # ...
    self.continuation_of = continuation_of

@classmethod
def from_engine_core_request(cls, request, block_hasher):
    return cls(
        # ... existing fields ...
        continuation_of=getattr(request, 'continuation_of', None),
    )
```

---

### 4.8 Layer 7: AsyncLLM (Public API)

**File**: `vllm/v1/engine/async_llm.py`

`generate_continuation()` is the public async generator API:

```python
async def generate_continuation(
    self,
    parent_request_id: str,
    continuation_token_ids: list[int],
    sampling_params: SamplingParams,
    request_id: str,
    *,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
) -> AsyncGenerator[RequestOutput, None]:
```

Flow:
1. Calls `input_processor.process_continuation()` to build an
   `EngineCoreRequest` (skipping tokenisation)
2. Passes the `EngineCoreRequest` directly to `add_request()`
3. Yields `RequestOutput` objects like normal `generate()`
4. Has proper error handling with abort on cancel/failure, plus logging

---

### 4.9 Layer 8: OpenAI Serving Layer (HTTP API)

**File**: `vllm/entrypoints/openai/serving_completion.py`

The completions handler detects `continuation_of` and routes accordingly:

```python
continuation_of = getattr(request, 'continuation_of', None)
continuation_suffix = getattr(request, 'continuation_suffix', None)

if isinstance(sampling_params, BeamSearchParams):
    generator = self.beam_search(
        prompt=engine_prompt,
        request_id=request_id,
        params=sampling_params,
        continuation_of=continuation_of,
        continuation_suffix=continuation_suffix,
        # ...
    )
elif continuation_of is not None:
    # Tokenise only the small suffix, not the full prompt
    suffix_token_ids = tokenizer.encode(
        continuation_suffix, add_special_tokens=False,
    ) if continuation_suffix and tokenizer else []
    generator = self.engine_client.generate_continuation(
        parent_request_id=continuation_of,
        continuation_token_ids=suffix_token_ids,
        # ...
    )
else:
    # Normal path
    generator = self.engine_client.generate(...)
```

Both beam-search and non-beam-search continuation requests are handled.
The suffix is tokenised once at the HTTP layer.

---

### 4.10 Layer 9: Beam Search (Continuation-Aware First Step)

**File**: `vllm/entrypoints/openai/serving_engine.py`

The `beam_search()` method accepts `continuation_of` and
`continuation_suffix` parameters:

```python
async def beam_search(
    self,
    prompt: PromptType,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    continuation_of: str | None = None,
    continuation_suffix: str | None = None,
) -> AsyncGenerator[RequestOutput, None]:
```

**First step, first beam**: Uses `generate_continuation()` to inherit
the parent's KV cache. This is wrapped in a **fallback coroutine**
(`_try_continuation`) that catches any exception and falls back to
regular `generate()` with the full prompt:

```python
async def _try_continuation(...):
    try:
        return await collect_from_async_generator(
            self.engine_client.generate_continuation(
                parent_request_id=_parent,
                continuation_token_ids=(_suffix or []),
                # ...
            )
        )
    except Exception as exc:
        logger.warning(
            "DKVC continuation failed for beam search "
            "(parent=%s): %s. Falling back to regular generation.",
            _parent, exc,
        )
        fb_rid = f"{_rid}-fallback"
        return await collect_from_async_generator(
            self.engine_client.generate(_prompt, _params, fb_rid, ...)
        )
```

**Subsequent steps**: Use normal `generate()` with APC handling the
shared prefix naturally, since all beams share the same token prefix.

---

## 5. Benchmark Results

### Setup
- **Dataset**: RecIF — the 2.5k-length user history dataset in OpenOneRec
- **Task**: `task==video` (video recommendation)
- **Load**: `NUM_PROMPTS=1000, RPS=inf`
- Two runs averaged for each configuration.

### Results

| | vLLM 0.16.0 | vLLM 0.16.0 + <br> `--enable-prefix-caching` | vLLM 0.14.0 + DKVC + <br> `--enable-prefix-caching` |
| :---: | :---: | :---: | :---: |
| Total duration (s) | 569.5 | 584.73 | **449.47** |

**Throughput improvement**: ~1.27× ($\frac{569.5}{449.47}$) over
vanilla vLLM 0.16.0.

Note: Vanilla APC (`--enable-prefix-caching` alone) actually slightly
*degrades* performance (569.5 → 584.73s) due to the overhead of hash
computation and cache management without guaranteed hits. DKVC's direct
block transfer eliminates this overhead and guarantees cache hits for
the shared prefix.

### Per-request savings (continuation vs. full re-prefill):

| Operation | Without DKVC | With DKVC |
|-----------|:---:|:---:|
| Tokenise full prompt | O(prompt_len) | **Skipped** (only suffix) |
| Block hash computation | O(prompt_len / block_size) | O(suffix_len / block_size) |
| Prefix cache lookup | O(num_blocks) hash walks | **Skipped** (direct transfer) |
| Prefill compute | O(unmatched_tokens) GPU | O(suffix_tokens) GPU |
| Block eviction risk | Non-zero | **Zero** (ref count held) |

---

## 6. Edge Cases and Safety

### 6.1 Parent already freed
If the parent request's blocks have been freed before the continuation
arrives, `inherit_blocks_from_parent()` falls back to
`get_computed_blocks()` (normal APC path). This is safe but loses the
guaranteed-hit property.

### 6.2 Parent still running
If the parent hasn't finished when the continuation arrives, the
`preprocess_add_request()` method can still find the parent in
`scheduler.requests` and snapshot its current `_all_token_ids`. The
block transfer then works from the parent's currently-allocated blocks.

### 6.3 External vs. internal request IDs
HTTP clients use external request IDs. The `_req_id_bimap` ensures
`continuation_of` works with either external or internal IDs:
- `preprocess_add_request()` resolves external → internal via the bimap
- `cache_finished_request_tokens()` indexes by both directions
- The `continuation_of` field is rewritten to the internal ID before
  reaching the scheduler/KV cache manager

### 6.4 Abort-path token caching
When a request is aborted due to a stop string (detected by the
detokenizer), it never goes through the normal "finished" path.
`_abort_requests()` explicitly calls `cache_finished_request_tokens()`
for each aborted request to ensure continuation support still works.

### 6.5 Block eviction during beam search
After the first beam step inherits blocks, subsequent steps use APC.
If blocks are evicted mid-beam-search, the engine re-prefills as usual.

### 6.6 Token cache overflow
The `_finished_request_tokens` cache is bounded by
`_max_finished_token_cache` (default 1024). When full, the oldest
entries are evicted (LRU via `OrderedDict`). If a continuation request
arrives for an evicted parent, the engine raises a `ValueError` — the
beam search fallback mechanism catches this and uses regular generation.

### 6.7 LoRA compatibility
Continuation requests inherit the parent's LoRA request. LoRA and
data-parallel rank validation is performed in `process_continuation()`.

### 6.8 Beam search fallback
The `_try_continuation()` wrapper in beam search catches **any**
exception from `generate_continuation()` and falls back to regular
`generate()` with a fresh request ID (`{original_rid}-fallback`). This
makes DKVC a best-effort optimization that never blocks inference.

---

## 7. Resolved Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Verify block hash consistency during transfer? | **No** | Hash consistency is guaranteed by construction (child prompt = parent `all_token_ids` + suffix). Skipping verification avoids O(n) overhead. |
| Support continuation from still-running parents? | **Yes** | `preprocess_add_request()` checks `scheduler.requests` first, allowing mid-flight continuations. |
| Finished-token cache scope? | **Per-EngineCore** | Simpler and sufficient for single-server. Bounded to 1024 entries. |
| Use `kv_transfer_params` or new fields? | **New fields** | `kv_transfer_params` is for P/D disaggregation with external KV connectors. Separate `continuation_of` / `continuation_token_ids` fields are semantically cleaner for local block sharing. |
| Where to put protocol fields? | **First-class request fields** | Added directly to the OpenAI protocol (`protocol.py`) rather than relying on `extra_body`, providing schema validation and documentation. |
| `transfer_blocks()` location? | **Base `KVCacheCoordinator`** | Available to all coordinator types (unitary, hybrid, no-prefix-cache), not just `UnitaryKVCacheCoordinator`. |

---

## 8. File Change Summary

| File | Change |
|------|--------|
| `vllm/entrypoints/openai/protocol.py` | `continuation_of`, `continuation_suffix` fields |
| `vllm/v1/engine/__init__.py` | `continuation_of`, `continuation_token_ids` on `EngineCoreRequest` |
| `vllm/v1/engine/input_processor.py` | `process_continuation()` method |
| `vllm/v1/engine/core.py` | `_finished_request_tokens`, `_req_id_bimap`, `cache_finished_request_tokens()`, continuation handling in `preprocess_add_request()` |
| `vllm/v1/request.py` | `continuation_of` field on `Request`, pass-through in `from_engine_core_request()` |
| `vllm/v1/core/kv_cache_manager.py` | `inherit_blocks_from_parent()` method |
| `vllm/v1/core/kv_cache_coordinator.py` | `transfer_blocks()` on base `KVCacheCoordinator` |
| `vllm/v1/core/sched/scheduler.py` | Continuation branch in scheduling loop |
| `vllm/v1/engine/async_llm.py` | `generate_continuation()` method |
| `vllm/entrypoints/openai/serving_completion.py` | Continuation detection and routing |
| `vllm/entrypoints/openai/serving_engine.py` | `beam_search()` continuation support with fallback |

---

## 9. Why "Direct" and Not "Distributed" KV Cache

### 9.1 Naming Rationale

The implementation is named **Direct KV Cache Inheritance (DKVC)** because
its core mechanism is a *direct* ref-count transfer on the same physical
KV cache blocks. No data moves; only ownership metadata is updated. This
is fundamentally different from *distributed* KV cache transfer, which
copies tensor data across network boundaries.

### 9.2 Could a Distributed Approach Achieve Better Performance?

**No.** For the target use case (two-stage recommendation on a single
server), distributed KVC would strictly degrade performance. Here is
a rigorous analysis.

#### A. Data movement: O(0) vs O(N × L × D)

| | Local DKVC (current) | Distributed KVC |
|---|:---:|:---:|
| **What moves** | Nothing. Ref count incremented on same physical blocks. | Entire KV tensor per request |
| **Cost** | O(num\_blocks) integer increments | O(num\_blocks × num\_layers × hidden\_dim) network transfer |

For a 2.5k-token prompt on a 7B model (32 layers, 32 KV heads, 128
head\_dim, block\_size=16):
- **Local DKVC**: ~156 blocks × 1 integer increment ≈ **~0 μs**
- **Distributed KVC**: ~156 blocks × 32 layers × 2 × 32 × 128 × 16 × 2 bytes
  ≈ **~1.3 GB** per request. At 100 Gbps InfiniBand ≈ **~100 ms** per request.

#### B. Synchronization overhead

Distributed KVC requires a multi-step pipeline that doesn't exist in
the local path:

1. Stage 1 finishes → `save_kv_layer()` per layer → `wait_for_save()`
2. Network transfer (RDMA / NCCL / TCP)
3. Stage 2 engine → `start_load_kv()` → `wait_for_layer_load()` per layer
4. Scheduler puts request in `WAITING_FOR_REMOTE_KVS`, polls for completion

Local DKVC is a single atomic operation inside `schedule()`.

#### C. Both stages share the same GPU

Both stages use the same model on the same GPU(s). Distributing them
to separate engines means:
- **2× GPU memory for model weights** (each engine loads the full model)
- **Halved KV cache budget** per engine
- **Network bottleneck** that doesn't exist in the single-engine case
- **Worse GPU utilization**: prefill engine is idle during decode and
  vice versa

#### D. APC already covers the distributed case's value proposition

The distributed KVC's selling point is "stage 2 finds stage 1's cache
even on a different machine." In the single-server case:
- Local DKVC **guarantees** hits via ref-count holding (zero eviction risk)
- Even vanilla APC gets hits most of the time (584.73s vs 569.5s baseline)

Distributed KVC adds massive data-movement cost to solve a problem
(cross-engine cache sharing) that **does not exist** here.

#### E. Performance comparison

| Metric | Local DKVC | Distributed KVC (hypothetical) |
|--------|:---:|:---:|
| Block transfer cost | **O(1) per block** | O(layers × heads × dim) per block |
| Network transfer | **None** | ~1.3 GB per request (2.5k tokens, 7B) |
| Synchronization | **None** | save → transfer → load → ready |
| GPU memory overhead | **None** (shared blocks) | 2× model weights |
| Eviction guarantee | **Yes** | Yes (blocks are copied) |
| Measured throughput | **1.27× baseline** | < 1× baseline (network-bound) |

### 9.3 When Distributed KVC Would Be Appropriate

Distributed KVC (via vLLM's existing `KVConnector` infrastructure) is
valuable in scenarios the current use case does **not** have:

1. **Multi-replica load balancing**: Stage 1 on replica A, stage 2
   routed to replica B — requires KV transfer across network.
2. **P/D disaggregation at scale**: Dedicated prefill and decode pools,
   where prefill GPUs are optimized for throughput and decode GPUs for
   latency.
3. **Memory-constrained deployment**: KV cache for both stages doesn't
   fit on a single GPU — offload to host memory or remote node.

vLLM already provides the full infrastructure for this via
`KVConnectorBase_V1`, with concrete implementations including
`NixlConnector`, `P2pNcclConnector`, `LMCacheConnectorV1`,
`MooncakeConnector`, and `OffloadingConnector`. The scheduler integrates
these via `WAITING_FOR_REMOTE_KVS` status and
`get_num_new_matched_tokens()`. These connectors are orthogonal to DKVC
and can coexist — DKVC handles same-engine inheritance while
`KVConnector` handles cross-engine transfer.

### 9.4 Conclusion

The 1.27× throughput improvement from local DKVC is essentially the
ceiling for scheduler/cache-level optimization in this workload. The
remaining time is dominated by GPU compute for decode steps, which no
caching strategy can eliminate. A distributed approach would add overhead
without removing any compute.
