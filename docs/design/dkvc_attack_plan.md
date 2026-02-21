# Distributed KVC for Two-Stage OpenOpenRec (Design Document)

**Goal**: Enable stage 2 (beam search) to directly inherit the KV cache
from stage 1 (thinking/sampling) so that stage 2 skips re-tokenisation
and re-prefill of the shared prefix at the vLLM scheduler/engine level.

---

## 1. Problem Statement

The current `benchmark_openopenrec_two_stage` in `serve.py` runs stage 1
and stage 2 **sequentially** as independent HTTP requests.  Even with the
pipelined variant (`benchmark_openopenrec_two_stage_distributed_kvc`),
stage 2 sends a brand-new completions request whose prompt is the
concatenation of the original prompt + stage-1 output + suffix tokens.

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
- At `get_computed_blocks()` (kv_cache_manager.py:164), the scheduler
  walks the block-hash chain via `find_longest_cache_hit()`.
- If the stage-1 blocks haven't been evicted, the hash lookup succeeds
  and the scheduler marks those tokens as "computed" → no re-prefill.

**What APC does NOT do (the gaps)**:
1. **Re-tokenisation**: The full prompt string is re-tokenised in
   `InputProcessor.process_inputs()` (input_processor.py:510-515).
2. **Block hash recomputation**: `Request.__init__` recomputes all block
   hashes from scratch (request.py:134-136).
3. **No guarantee**: APC is opportunistic — blocks can be evicted under
   memory pressure between stage 1 finishing and stage 2 arriving.
4. **Beam search overhead**: `beam_search()` (serving_engine.py:382-511)
   creates `beam_width × max_tokens` sub-requests, each going through
   the full lifecycle.

---

## 2. Proposed Solution: "Continuation Request"

A **continuation request** is a new request type that explicitly inherits
KV cache blocks from a completed (or still-running) parent request.

### Key properties:
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
                    │  Benchmark Client (serve.py)     │
                    │  POST /v1/completions            │
                    │  { "continuation_of": "req-s1",  │
                    │    "continuation_suffix": "...", │
                    │    "use_beam_search": true }     │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 8          │  OpenAI Serving Layer            │
                    │  serving_completion.py           │
                    │  Detect continuation_of field    │
                    │  → call generate_continuation()  │
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
  Layer 1          │  EngineCoreRequest               │
                    │  v1/engine/__init__.py            │
                    │  + continuation_of: str | None   │
                    │  + continuation_token_ids: [int] │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 3          │  EngineCore.preprocess_add_req() │
                    │  v1/engine/core.py                │
                    │  Build prompt_token_ids from      │
                    │  parent's all_token_ids + suffix  │
                    │  (lookup in finished_tokens cache)│
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
  Layer 6          │  Request.__init__()              │
                    │  v1/request.py                    │
                    │  + continuation_of field          │
                    │  Block hashes computed as normal  │
                    │  (needed for future APC of child) │
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
  Layer 4          │  KVCacheManager                  │
                    │  v1/core/kv_cache_manager.py      │
                    │  + inherit_blocks_from_parent()  │
                    │  Direct block ref transfer       │
                    │  (no hash walk, guaranteed hit)  │
                    └──────────────────────────────────┘
```

---

## 4. Detailed Changes Per Layer

### 4.1 Layer 1: EngineCoreRequest (Protocol)

**File**: `vllm_ge/vllm/v1/engine/__init__.py`
**Lines**: 49-83 (class `EngineCoreRequest`)

**Change**: Add two optional fields to carry continuation metadata.

```python
# --- BEFORE (line 83) ---
    external_req_id: str | None = None

# --- AFTER ---
    external_req_id: str | None = None

    # Continuation support: inherit KV cache from a parent request.
    # When set, the engine constructs prompt_token_ids from the parent's
    # all_token_ids + continuation_token_ids, and the scheduler inherits
    # KV cache blocks directly instead of doing hash-based prefix lookup.
    continuation_of: str | None = None
    continuation_token_ids: list[int] | None = None
```

**Rationale**: `EngineCoreRequest` is a `msgspec.Struct` with
`omit_defaults=True`, so these new `None`-default fields add zero
serialisation overhead for non-continuation requests.

---

### 4.2 Layer 2: InputProcessor (Skip Tokenisation)

**File**: `vllm_ge/vllm/v1/engine/input_processor.py`
**Lines**: 446-600 (method `process_inputs`)

**Change**: Add a new method `process_continuation()` that bypasses the
tokenisation/preprocessing pipeline entirely.

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
    """Create a continuation request that inherits KV cache from parent.

    Skips tokenisation entirely. prompt_token_ids is set to None here
    and will be populated by EngineCore.preprocess_add_request() from
    the parent's all_token_ids + continuation_token_ids.
    """
    if arrival_time is None:
        arrival_time = time.time()

    sampling_params = params.clone()
    eos_token_id = self.input_preprocessor.get_eos_token_id()
    if sampling_params.max_tokens is None:
        sampling_params.max_tokens = self.model_config.max_model_len
    sampling_params.update_from_generation_config(
        self.generation_config_fields, eos_token_id
    )
    if self.tokenizer is not None:
        sampling_params.update_from_tokenizer(self.tokenizer)

    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=None,  # filled by EngineCore
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=eos_token_id,
        arrival_time=arrival_time,
        lora_request=lora_request,
        cache_salt=None,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
        trace_headers=trace_headers,
        continuation_of=parent_request_id,
        continuation_token_ids=continuation_token_ids,
    )
```

**What this skips**:
- `self.input_preprocessor.preprocess()` — the full tokenisation +
  multimodal preprocessing pipeline (line 511)
- Multimodal hash/feature processing (lines 557-584)
- Platform validation of raw prompt (lines 519-523)

**What this preserves**:
- `SamplingParams` cloning and validation
- EOS token ID resolution
- Generation config updates

---

### 4.3 Layer 3: EngineCore (Prompt Construction from Parent)

**File**: `vllm_ge/vllm/v1/engine/core.py`
**Lines**: 612-634 (method `preprocess_add_request`)

**Change A**: Add a bounded cache of recently-finished request tokens.

```python
# In EngineCore.__init__ (around line 200), add:
from collections import OrderedDict

# Cache of recently-finished requests' all_token_ids.
# Bounded to prevent unbounded memory growth.
self._finished_request_tokens: OrderedDict[str, list[int]] = OrderedDict()
self._max_finished_token_cache = 1024  # configurable
```

**Change B**: Populate the cache when requests finish.

The scheduler's `_free_request()` is called when a request finishes
(scheduler.py:1214). We need to snapshot the token IDs before freeing.
The cleanest place is in `EngineCore` after `update_from_output()`.

```python
# In EngineCore, after scheduler.update_from_output() returns,
# iterate over finished requests and cache their tokens.
# This can be done in the step() method or a post-processing hook.

def _cache_finished_request_tokens(self, finished_req_ids: set[str]):
    """Cache token IDs of finished requests for continuation support."""
    for req_id in finished_req_ids:
        # The request object may still be accessible via scheduler
        # before final cleanup.
        req = self.scheduler.requests.get(req_id)
        if req is not None:
            self._finished_request_tokens[req_id] = list(req._all_token_ids)
            # Evict oldest if over capacity
            while len(self._finished_request_tokens) > self._max_finished_token_cache:
                self._finished_request_tokens.popitem(last=False)
```

**Change C**: Modify `preprocess_add_request()` to handle continuations.

```python
def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
    # --- NEW: Handle continuation requests ---
    if request.continuation_of is not None:
        parent_tokens = None

        # Try 1: parent is still running/waiting in the scheduler
        parent_req = self.scheduler.requests.get(request.continuation_of)
        if parent_req is not None:
            parent_tokens = list(parent_req._all_token_ids)

        # Try 2: parent recently finished — look up in cache
        if parent_tokens is None:
            parent_tokens = self._finished_request_tokens.get(
                request.continuation_of
            )

        if parent_tokens is None:
            raise ValueError(
                f"Continuation parent request '{request.continuation_of}' "
                f"not found in running requests or finished cache. "
                f"Ensure the parent request has not been evicted."
            )

        # Build the full prompt_token_ids
        suffix = request.continuation_token_ids or []
        request.prompt_token_ids = parent_tokens + suffix
        logger.debug(
            "Continuation request %s: inherited %d tokens from parent %s, "
            "appended %d suffix tokens → total %d prompt tokens",
            request.request_id,
            len(parent_tokens),
            request.continuation_of,
            len(suffix),
            len(request.prompt_token_ids),
        )
    # --- END NEW ---

    # ... existing logic unchanged ...
    if self.mm_receiver_cache is not None and request.mm_features:
        request.mm_features = self.mm_receiver_cache.get_and_update_features(
            request.mm_features
        )

    req = Request.from_engine_core_request(request, self.request_block_hasher)
    # ...
```

**Important**: We still call `Request.from_engine_core_request()` which
computes block hashes. This is intentional — the child request needs its
own block hashes so that:
1. Its blocks can be cached for future APC hits
2. The `inherit_blocks_from_parent()` can verify hash consistency

---

### 4.4 Layer 4: KVCacheManager (Direct Block Inheritance)

**File**: `vllm_ge/vllm/v1/core/kv_cache_manager.py`
**Lines**: After `get_computed_blocks()` (line 204)

**Change**: Add `inherit_blocks_from_parent()` method.

```python
def inherit_blocks_from_parent(
    self,
    child_request: Request,
    parent_request_id: str,
) -> tuple[KVCacheBlocks, int]:
    """Directly inherit KV cache blocks from a parent request.

    Unlike get_computed_blocks() which does hash-based lookup via
    find_longest_cache_hit(), this directly transfers block references
    from the parent request. This guarantees:
    - No eviction race (ref counts are incremented immediately)
    - No hash computation overhead for the shared prefix
    - O(1) per block instead of O(n) hash chain walk

    Falls back to get_computed_blocks() if the parent's blocks are
    no longer available (e.g., parent was freed and blocks evicted).

    Args:
        child_request: The continuation request.
        parent_request_id: The request_id of the parent.

    Returns:
        A tuple of (inherited_blocks, num_inherited_tokens).
    """
    # Try to get parent's blocks from the coordinator
    parent_block_ids = self.coordinator.get_req_blocks(parent_request_id)
    if parent_block_ids is None:
        # Parent blocks already freed — fall back to hash-based lookup
        logger.debug(
            "Parent %s blocks not found, falling back to prefix cache "
            "for continuation %s",
            parent_request_id,
            child_request.request_id,
        )
        return self.get_computed_blocks(child_request)

    # Transfer blocks: increment ref counts so they survive parent cleanup
    inherited_blocks = self.coordinator.transfer_blocks(
        from_request_id=parent_request_id,
        to_request_id=child_request.request_id,
        child_block_hashes=child_request.block_hashes,
    )

    num_inherited_tokens = sum(
        len(blocks) for blocks in inherited_blocks
    ) * self.block_size // len(inherited_blocks) if inherited_blocks[0] else 0

    # Don't exceed child's prompt length - 1 (need at least 1 token to compute)
    max_cache_hit = child_request.num_tokens - 1
    num_inherited_tokens = min(num_inherited_tokens, max_cache_hit)

    if self.log_stats:
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=child_request.num_tokens,
            num_hits=num_inherited_tokens,
            preempted=False,
        )

    return self.create_kv_cache_blocks(inherited_blocks), num_inherited_tokens
```

**Note**: The `coordinator.transfer_blocks()` method needs to be added to
the KV cache coordinator. This is a thin wrapper that:
1. Looks up the parent's block list
2. For each block that matches the child's block hash (they should match
   since the child's prompt is a prefix extension), increments the ref count
3. Returns the matched blocks

If the coordinator doesn't support direct transfer yet, a simpler
implementation can just increment ref counts on the parent's blocks and
register them for the child request. The key insight is that since the
child's prompt starts with the same tokens as the parent's all_token_ids,
the block hashes for the shared prefix are identical.

---

### 4.5 Layer 5: Scheduler (Continuation-Aware Scheduling)

**File**: `vllm_ge/vllm/v1/core/sched/scheduler.py`
**Lines**: 517-546 (waiting request scheduling loop)

**Change**: Add a branch for continuation requests before the normal
`get_computed_blocks()` path.

```python
# Get already-cached tokens.
if request.num_computed_tokens == 0:
    # --- NEW: continuation request → direct block inheritance ---
    if getattr(request, 'continuation_of', None) is not None:
        new_computed_blocks, num_new_local_computed_tokens = (
            self.kv_cache_manager.inherit_blocks_from_parent(
                request, request.continuation_of
            )
        )
    else:
        # --- EXISTING: hash-based prefix cache lookup ---
        # Get locally-cached tokens.
        new_computed_blocks, num_new_local_computed_tokens = (
            self.kv_cache_manager.get_computed_blocks(request)
        )
    # --- END CHANGE ---

    # Get externally-cached tokens if using a KVConnector.
    if self.connector is not None:
        # ... unchanged ...
```

**What changes**: Only the block lookup method. Everything downstream
(allocate_slots, scheduling, etc.) is unchanged because
`inherit_blocks_from_parent()` returns the same
`(KVCacheBlocks, int)` tuple as `get_computed_blocks()`.

---

### 4.6 Layer 6: Request Object

**File**: `vllm_ge/vllm/v1/request.py`
**Lines**: 30-161

**Change A**: Add `continuation_of` field to `Request.__init__()`.

```python
def __init__(
    self,
    # ... existing params ...
    block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    continuation_of: str | None = None,  # NEW
) -> None:
    # ... existing init ...
    self.continuation_of = continuation_of  # NEW
```

**Change B**: Pass it through in `from_engine_core_request()`.

```python
@classmethod
def from_engine_core_request(
    cls,
    request: EngineCoreRequest,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None,
) -> "Request":
    return cls(
        # ... existing fields ...
        block_hasher=block_hasher,
        continuation_of=getattr(request, 'continuation_of', None),  # NEW
    )
```

---

### 4.7 Layer 7: AsyncLLM (Public API)

**File**: `vllm_ge/vllm/v1/engine/async_llm.py`
**Lines**: After `generate()` (line 481)

**Change**: Add `generate_continuation()` method.

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
    """Generate continuing from a parent request's KV cache.

    The child request inherits the parent's KV cache blocks directly,
    skipping re-tokenisation and re-prefill of the shared prefix.
    Only the continuation_token_ids suffix needs to be prefilled.

    Args:
        parent_request_id: The request_id of the completed stage-1 request.
        continuation_token_ids: Token IDs to append after the parent's
            output (e.g., tokenized "</think>\\n<|sid_begin|>").
        sampling_params: Sampling parameters for the continuation.
        request_id: Unique ID for this continuation request.
    """
    request = self.input_processor.process_continuation(
        request_id=request_id,
        parent_request_id=parent_request_id,
        continuation_token_ids=continuation_token_ids,
        params=sampling_params,
        lora_request=lora_request,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
    )

    # From here, same flow as generate() — add request and yield outputs
    q: RequestOutputCollector | None = None
    try:
        q = await self.add_request(
            request_id,
            request,  # pass EngineCoreRequest directly
            sampling_params,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

        finished = False
        while not finished:
            out = q.get_nowait() or await q.get()
            finished = out.finished
            assert isinstance(out, RequestOutput)
            yield out

    except (asyncio.CancelledError, GeneratorExit):
        if q is not None:
            await self.abort(q.request_id, internal=True)
        raise
    except EngineDeadError:
        raise
    except ValueError:
        raise
    except Exception as e:
        if q is not None:
            await self.abort(q.request_id, internal=True)
        raise EngineGenerateError() from e
```

**Note**: `add_request()` already supports receiving an
`EngineCoreRequest` directly (line 302-309 in async_llm.py), so the
continuation request flows through without re-processing.

---

### 4.8 Layer 8: OpenAI Serving Layer (HTTP API)

**File**: `vllm_ge/vllm/entrypoints/openai/serving_completion.py`

**Change**: Detect `continuation_of` in the request body and route to
`generate_continuation()`.

```python
# In the completions handler, after parsing the request:
if hasattr(request, 'extra_body') and request.extra_body:
    continuation_of = request.extra_body.get('continuation_of')
    continuation_suffix = request.extra_body.get('continuation_suffix')
else:
    continuation_of = None
    continuation_suffix = None

if continuation_of is not None:
    # Tokenise only the small suffix (not the full prompt)
    suffix_token_ids = tokenizer.encode(
        continuation_suffix or "",
        add_special_tokens=False,
    )
    generator = self.engine_client.generate_continuation(
        parent_request_id=continuation_of,
        continuation_token_ids=suffix_token_ids,
        sampling_params=sampling_params,
        request_id=request_id,
        lora_request=lora_request,
        trace_headers=trace_headers,
    )
else:
    # Normal path
    generator = self.engine_client.generate(...)
```

**Protocol change**: The `extra_body` dict in the OpenAI completions
request gains two new optional fields:
- `continuation_of` (str): parent request_id
- `continuation_suffix` (str): text to append (tokenised server-side)

---

### 4.9 Layer 9: Beam Search (Continuation-Aware First Step)

**File**: `vllm_ge/vllm/entrypoints/openai/serving_engine.py`
**Lines**: 382-511 (method `beam_search`)

**Change**: The beam search method's first iteration can use
`generate_continuation()` if a `continuation_of` parameter is provided.
Subsequent iterations already benefit from APC since each beam step
extends the previous one by exactly 1 token.

```python
async def beam_search(
    self,
    prompt: PromptType,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    continuation_of: str | None = None,  # NEW
) -> AsyncGenerator[RequestOutput, None]:
    # ... existing setup ...

    for step in range(max_tokens):
        # ... existing batch construction ...

        for i, (individual_prompt, lora_req) in enumerate(...):
            request_id_item = f"{request_id_batch}-beam-{i}"

            if step == 0 and i == 0 and continuation_of is not None:
                # First beam, first step: use continuation to inherit KVC
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.engine_client.generate_continuation(
                            parent_request_id=continuation_of,
                            continuation_token_ids=...,  # suffix tokens
                            sampling_params=beam_search_params,
                            request_id=request_id_item,
                            lora_request=lora_req,
                            trace_headers=trace_headers,
                        )
                    )
                )
            else:
                # Normal path (APC handles prefix matching)
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.engine_client.generate(
                            individual_prompt,
                            beam_search_params,
                            request_id_item,
                            lora_request=lora_req,
                            trace_headers=trace_headers,
                        )
                    )
                )
            tasks.append(task)
```

**Note**: After the first step, all beams share the same prefix (the
original prompt + thinking + suffix), so APC naturally handles them.
The continuation mechanism is only needed for the very first step to
bridge stage 1 → stage 2.

---

### 4.10 Layer 10: Benchmark Client

**File**: `vllm_ge/vllm/benchmarks/serve.py`
**Function**: `benchmark_openopenrec_two_stage_distributed_kvc`

**Change**: Pass `continuation_of` and `continuation_suffix` in the
stage-2 request's `extra_body`.

```python
# In _run_pipelined_request(), stage 2 section:
s2_extra_body = {
    "n": num_beams,
    "use_beam_search": True,
    "temperature": 0.0,
    "continuation_of": req.request_id,  # parent stage-1 request ID
    "continuation_suffix": thinking + "</think>\n" + prompt_token,
}

s2_input = RequestFuncInput(
    model=model_id,
    model_name=model_name,
    prompt="",  # ignored when continuation_of is set
    api_url=api_url_completions,
    prompt_len=0,  # will be computed server-side
    output_len=max_new_tokens,
    extra_body=s2_extra_body,
    # ...
)
```

---

## 5. KV Cache Coordinator: transfer_blocks() Implementation

The `KVCacheCoordinator` (kv_cache_coordinator.py) needs a new method.
The simplest implementation for `UnitaryKVCacheCoordinator`:

```python
def transfer_blocks(
    self,
    from_request_id: str,
    to_request_id: str,
    child_block_hashes: list[BlockHash],
) -> tuple[list[KVCacheBlock], ...]:
    """Transfer block references from parent to child request.

    Increments ref counts on matched blocks so they survive parent
    cleanup. Returns the blocks that can be inherited.
    """
    manager = self.single_type_managers[0]
    parent_blocks = manager.req_to_blocks.get(from_request_id)
    if parent_blocks is None:
        return tuple([] for _ in range(self.num_single_type_manager))

    # The child's block hashes for the shared prefix should match
    # the parent's cached blocks. Transfer as many as match.
    inherited = []
    for i, block in enumerate(parent_blocks):
        if i < len(child_block_hashes):
            # Verify hash consistency (optional but safe)
            block.incr_ref()
            inherited.append(block)
        else:
            break

    # Register these blocks for the child request
    manager.req_to_blocks[to_request_id] = list(inherited)

    return (inherited,)  # tuple of lists, one per kv_cache_group
```

---

## 6. Edge Cases and Safety

### 6.1 Parent already freed
If the parent request's blocks have been freed before the continuation
arrives, `inherit_blocks_from_parent()` falls back to
`get_computed_blocks()` (normal APC path). This is safe but loses the
guarantee.

### 6.2 Parent still running
If stage 1 hasn't finished yet when stage 2 arrives (shouldn't happen in
our pipelined design, but possible in other use cases), the scheduler
should skip the continuation request and re-check on the next scheduling
step. This can use the existing `WAITING_FOR_REMOTE_KVS`-like pattern.

### 6.3 Block eviction during beam search
After the first beam step inherits blocks, subsequent steps use APC.
If blocks are evicted mid-beam-search, the engine re-prefills as usual.
This is the same behaviour as today.

### 6.4 Token cache overflow
The `_finished_request_tokens` cache is bounded by
`_max_finished_token_cache`. When full, the oldest entries are evicted.
If a continuation request arrives for an evicted parent, the engine
falls back to the normal tokenisation + APC path.

### 6.5 LoRA compatibility
Continuation requests inherit the parent's LoRA request. The child can
use a different LoRA, but the inherited blocks would not be valid in
that case. Validation should reject mismatched LoRA.

---

## 7. Performance Impact

### What we save (per stage-2 request):
| Operation | Current Cost | With Continuation |
|-----------|-------------|-------------------|
| Tokenise full prompt | O(prompt_len) | **Skipped** (only suffix) |
| Block hash computation | O(prompt_len / block_size) | O(suffix_len / block_size) |
| Prefix cache lookup | O(num_blocks) hash walks | **Skipped** (direct transfer) |
| Prefill compute | O(unmatched_tokens) GPU | O(suffix_tokens) GPU |
| Block eviction risk | Non-zero | **Zero** (ref count held) |

### Typical numbers for OpenOpenRec:
- Prompt: ~500 tokens
- Stage-1 output: ~200 tokens (thinking)
- Suffix: ~5 tokens (`</think>\n<|sid_begin|>`)
- Stage-2 prefill with APC: ~5 tokens (if blocks survive)
- Stage-2 prefill without APC: ~705 tokens (full re-prefill)
- Stage-2 prefill with continuation: **~5 tokens (guaranteed)**

### Beam search amplification:
- `beam_width=32, max_tokens=3` → 96 sub-requests
- First step: 1 continuation request (inherits ~700 tokens of KVC)
- Steps 2-3: 32 requests each, APC handles shared prefix naturally

---

## 8. Implementation Order

Recommended bottom-up implementation order:

1. **Layer 1**: `EngineCoreRequest` fields (trivial, no behaviour change)
2. **Layer 6**: `Request` continuation_of field (trivial)
3. **Layer 4**: `KVCacheManager.inherit_blocks_from_parent()` (core logic)
4. **Layer 5**: Scheduler branch (small change, high impact)
5. **Layer 3**: `EngineCore` token cache + preprocess change
6. **Layer 2**: `InputProcessor.process_continuation()`
7. **Layer 7**: `AsyncLLM.generate_continuation()`
8. **Layer 8**: OpenAI serving layer
9. **Layer 9**: Beam search continuation support
10. **Layer 10**: Benchmark client update

Each layer can be tested independently:
- Layers 1-5: Unit test with mock requests
- Layers 6-7: Integration test with `LLMEngine` directly
- Layers 8-10: End-to-end test with vLLM server

---

## 9. Testing Strategy

### Unit tests:
- `test_kv_cache_manager_inherit_blocks`: Verify block ref counts,
  fallback to APC, hash consistency check
- `test_scheduler_continuation`: Verify continuation requests skip
  `get_computed_blocks()` and use `inherit_blocks_from_parent()`
- `test_engine_core_token_cache`: Verify finished tokens are cached
  and retrievable, LRU eviction works

### Integration tests:
- `test_generate_continuation`: Two-request sequence where request 2
  is a continuation of request 1. Verify request 2's prefill is only
  the suffix tokens.
- `test_continuation_parent_freed`: Continuation arrives after parent
  blocks are freed. Verify graceful fallback to APC.

### End-to-end benchmark:
- Run `benchmark_openopenrec_two_stage_distributed_kvc` with and without
  `--openopenrec-distributed-kvc` and compare:
  - Stage-2 TTFT (should be ~100x lower with continuation)
  - Total wall-clock time
  - Evaluation metrics (should be identical)

---

## 10. Open Questions

1. **Should `transfer_blocks()` verify block hash consistency?**
   Pro: catches bugs. Con: adds O(n) overhead. Recommendation: verify
   in debug mode only.

2. **Should we support continuation from still-running parents?**
   This would enable true streaming pipeline (stage 2 starts before
   stage 1 finishes). Much more complex — requires partial block
   transfer. Defer to v2.

3. **Should the finished-token cache be per-EngineCore or shared?**
   Per-EngineCore is simpler and sufficient for single-server.
   For multi-server P/D, the cache would need to be distributed.

4. **Alternative: use `kv_transfer_params` instead of new fields?**
   The existing `kv_transfer_params` mechanism (request.py:66-81) is
   designed for P/D disaggregation with external KV connectors. We
   could piggyback on it, but it's semantically different (remote
   transfer vs. local block sharing). Separate fields are cleaner.