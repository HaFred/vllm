# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for distributed KV cache (dkvc) continuation support.

Covers:
- KVCacheCoordinator.transfer_blocks()
- KVCacheManager.inherit_blocks_from_parent()
- Scheduler continuation_of branch
"""

import logging

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinatorNoPrefixCache,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 200


def _make_kv_cache_config(
    block_size: int = BLOCK_SIZE,
    num_blocks: int = NUM_GPU_BLOCKS,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def _make_coordinator(
    enable_caching: bool = False,
    block_size: int = BLOCK_SIZE,
    num_blocks: int = NUM_GPU_BLOCKS,
):
    kv_cache_config = _make_kv_cache_config(block_size, num_blocks)
    return get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        use_eagle=False,
        enable_caching=enable_caching,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=block_size,
    )


def _make_kv_cache_manager(
    enable_caching: bool = True,
    block_size: int = BLOCK_SIZE,
    num_blocks: int = NUM_GPU_BLOCKS,
    log_stats: bool = True,
) -> KVCacheManager:
    kv_cache_config = _make_kv_cache_config(block_size, num_blocks)
    return KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        hash_block_size=block_size,
        enable_caching=enable_caching,
        log_stats=log_stats,
    )


def _allocate_blocks_for_request(
    coordinator, request_id: str, num_blocks: int
) -> list[KVCacheBlock]:
    """Manually allocate blocks from the pool and register them for a request."""
    blocks = coordinator.block_pool.allocate(num_blocks)
    manager = coordinator.single_type_managers[0]
    manager.req_to_blocks[request_id] = list(blocks)
    return list(blocks)


# ---------------------------------------------------------------------------
# Tests: KVCacheCoordinator.transfer_blocks
# ---------------------------------------------------------------------------


class TestTransferBlocks:
    """Tests for KVCacheCoordinator.transfer_blocks()."""

    def test_transfer_blocks_basic(self):
        """Blocks are transferred and ref counts incremented."""
        coordinator = _make_coordinator(enable_caching=False)
        parent_blocks = _allocate_blocks_for_request(
            coordinator, "parent", 3
        )
        original_ref_cnts = [b.ref_cnt for b in parent_blocks]
        logger.debug(
            "Before transfer: parent block ids=%s, ref_cnts=%s",
            [b.block_id for b in parent_blocks],
            original_ref_cnts,
        )

        result = coordinator.transfer_blocks("parent", "child")

        assert result is not None, "transfer_blocks should return blocks"
        assert len(result) == 1, "Single kv_cache_group expected"
        assert len(result[0]) == 3

        for i, block in enumerate(result[0]):
            assert block.block_id == parent_blocks[i].block_id
            assert block.ref_cnt == original_ref_cnts[i] + 1
            logger.debug(
                "After transfer: block_id=%d, ref_cnt=%d (was %d)",
                block.block_id, block.ref_cnt, original_ref_cnts[i],
            )

        # Verify child is registered in req_to_blocks
        manager = coordinator.single_type_managers[0]
        assert "child" in manager.req_to_blocks
        assert len(manager.req_to_blocks["child"]) == 3

    def test_transfer_blocks_parent_not_found(self):
        """Returns None when parent has no blocks."""
        coordinator = _make_coordinator(enable_caching=False)

        result = coordinator.transfer_blocks("nonexistent", "child")

        assert result is None
        logger.debug("transfer_blocks returned None for missing parent")

    def test_transfer_blocks_parent_empty(self):
        """Returns None when parent has empty block list."""
        coordinator = _make_coordinator(enable_caching=False)
        manager = coordinator.single_type_managers[0]
        manager.req_to_blocks["parent"] = []

        result = coordinator.transfer_blocks("parent", "child")

        assert result is None
        logger.debug("transfer_blocks returned None for empty parent blocks")

    def test_transfer_blocks_preserves_parent(self):
        """Parent's blocks remain registered after transfer."""
        coordinator = _make_coordinator(enable_caching=False)
        _allocate_blocks_for_request(coordinator, "parent", 2)

        coordinator.transfer_blocks("parent", "child")

        manager = coordinator.single_type_managers[0]
        assert "parent" in manager.req_to_blocks
        assert len(manager.req_to_blocks["parent"]) == 2
        logger.debug("Parent blocks still registered after transfer")

    def test_transfer_blocks_multiple_children(self):
        """Multiple children can inherit from the same parent."""
        coordinator = _make_coordinator(enable_caching=False)
        parent_blocks = _allocate_blocks_for_request(
            coordinator, "parent", 2
        )
        initial_ref_cnts = [b.ref_cnt for b in parent_blocks]

        coordinator.transfer_blocks("parent", "child_1")
        coordinator.transfer_blocks("parent", "child_2")

        for i, block in enumerate(parent_blocks):
            assert block.ref_cnt == initial_ref_cnts[i] + 2
            logger.debug(
                "Block %d ref_cnt=%d after 2 children (was %d)",
                block.block_id, block.ref_cnt, initial_ref_cnts[i],
            )

        manager = coordinator.single_type_managers[0]
        assert "child_1" in manager.req_to_blocks
        assert "child_2" in manager.req_to_blocks


# ---------------------------------------------------------------------------
# Tests: KVCacheManager.inherit_blocks_from_parent
# ---------------------------------------------------------------------------


class TestInheritBlocksFromParent:
    """Tests for KVCacheManager.inherit_blocks_from_parent()."""

    def _make_request(
        self,
        request_id: str,
        num_tokens: int,
        block_size: int = BLOCK_SIZE,
    ) -> Request:
        """Create a minimal Request for testing."""
        init_none_hash(sha256)
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher
        block_hasher = get_request_block_hasher(block_size, sha256)
        return Request(
            request_id=request_id,
            prompt_token_ids=[0] * num_tokens,
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            block_hasher=block_hasher,
        )

    def test_inherit_success(self):
        """Child inherits blocks from parent, returns correct token count."""
        mgr = _make_kv_cache_manager(enable_caching=False, log_stats=True)

        # Allocate parent blocks
        parent_blocks = _allocate_blocks_for_request(
            mgr.coordinator, "parent", 3
        )
        logger.debug(
            "Allocated %d blocks for parent: ids=%s",
            len(parent_blocks),
            [b.block_id for b in parent_blocks],
        )

        # Child has enough tokens to benefit from inheritance
        child_num_tokens = 3 * BLOCK_SIZE + 5  # More than parent's blocks
        child = self._make_request("child", child_num_tokens)

        kv_blocks, num_tokens = mgr.inherit_blocks_from_parent(
            child, "parent"
        )

        expected_tokens = min(3 * BLOCK_SIZE, child_num_tokens - 1)
        assert num_tokens == expected_tokens
        assert isinstance(kv_blocks, KVCacheBlocks)
        assert len(kv_blocks.blocks) == 1
        assert len(kv_blocks.blocks[0]) == 3
        logger.debug(
            "Inherited %d tokens from %d blocks",
            num_tokens, len(kv_blocks.blocks[0]),
        )

    def test_inherit_caps_at_child_prompt_minus_one(self):
        """Inherited tokens capped at child.num_tokens - 1."""
        mgr = _make_kv_cache_manager(enable_caching=False, log_stats=True)

        # Parent has many blocks, child has few tokens
        _allocate_blocks_for_request(mgr.coordinator, "parent", 5)
        child_num_tokens = 2 * BLOCK_SIZE  # Only 32 tokens
        child = self._make_request("child", child_num_tokens)

        _, num_tokens = mgr.inherit_blocks_from_parent(child, "parent")

        assert num_tokens == child_num_tokens - 1
        logger.debug(
            "Inherited tokens capped at %d (child_tokens=%d)",
            num_tokens, child_num_tokens,
        )

    def test_inherit_fallback_when_parent_missing(self):
        """Falls back to get_computed_blocks when parent has no blocks."""
        mgr = _make_kv_cache_manager(enable_caching=False, log_stats=True)
        child = self._make_request("child", 50)

        kv_blocks, num_tokens = mgr.inherit_blocks_from_parent(
            child, "nonexistent_parent"
        )

        # Fallback: no caching enabled, so 0 computed tokens
        assert num_tokens == 0
        logger.debug(
            "Fallback: inherit returned %d tokens for missing parent",
            num_tokens,
        )

    def test_inherit_records_prefix_cache_stats(self):
        """Stats are recorded when log_stats is enabled."""
        mgr = _make_kv_cache_manager(enable_caching=False, log_stats=True)
        _allocate_blocks_for_request(mgr.coordinator, "parent", 2)
        child = self._make_request("child", 3 * BLOCK_SIZE)

        mgr.inherit_blocks_from_parent(child, "parent")

        stats = mgr.make_prefix_cache_stats()
        assert stats is not None
        assert stats.requests > 0
        logger.debug("Prefix cache stats after inherit: %s", stats)


# ---------------------------------------------------------------------------
# Tests: Scheduler continuation_of branch
# ---------------------------------------------------------------------------


class TestSchedulerContinuation:
    """Test that the scheduler uses inherit_blocks_from_parent
    when request.continuation_of is set."""

    def test_continuation_request_scheduled(self):
        """A request with continuation_of is added and scheduled."""
        scheduler = create_scheduler(
            enable_prefix_caching=False,
            num_blocks=10000,
        )

        # Add a parent request and schedule it
        parent_requests = create_requests(
            num_requests=1, num_tokens=32, req_ids=["parent-0"]
        )
        parent = parent_requests[0]
        scheduler.add_request(parent)
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1
        logger.debug("Parent scheduled: %s", parent.request_id)

        # Create a continuation request
        init_none_hash(sha256)
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher
        block_hasher = get_request_block_hasher(16, sha256)
        child = Request(
            request_id="child-0",
            prompt_token_ids=[0] * 48,  # parent tokens + suffix
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            block_hasher=block_hasher,
            continuation_of="parent-0",
        )

        scheduler.add_request(child)
        assert child.request_id in scheduler.requests
        assert child.continuation_of == "parent-0"
        logger.debug(
            "Child request added: %s, continuation_of=%s",
            child.request_id, child.continuation_of,
        )

        # Schedule the child — this should hit the continuation branch
        output2 = scheduler.schedule()
        assert "child-0" in output2.num_scheduled_tokens
        logger.debug(
            "Child scheduled with %d tokens",
            output2.num_scheduled_tokens["child-0"],
        )

    def test_continuation_of_none_uses_normal_path(self):
        """Requests without continuation_of use the normal code path."""
        scheduler = create_scheduler(enable_prefix_caching=False)
        requests = create_requests(num_requests=2, num_tokens=32)

        for req in requests:
            assert req.continuation_of is None
            scheduler.add_request(req)

        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 2
        logger.debug("Normal requests scheduled without continuation path")


# ---------------------------------------------------------------------------
# Tests: Request.continuation_of propagation
# ---------------------------------------------------------------------------


class TestRequestContinuationOf:
    """Test that continuation_of is properly propagated through Request."""

    def test_request_init_with_continuation_of(self):
        """Request stores continuation_of correctly."""
        init_none_hash(sha256)
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher
        block_hasher = get_request_block_hasher(16, sha256)

        req = Request(
            request_id="test-req",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            block_hasher=block_hasher,
            continuation_of="parent-req",
        )

        assert req.continuation_of == "parent-req"
        logger.debug(
            "Request %s has continuation_of=%s",
            req.request_id, req.continuation_of,
        )

    def test_request_init_without_continuation_of(self):
        """Request defaults continuation_of to None."""
        init_none_hash(sha256)
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher
        block_hasher = get_request_block_hasher(16, sha256)

        req = Request(
            request_id="test-req-2",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            block_hasher=block_hasher,
        )

        assert req.continuation_of is None
        logger.debug(
            "Request %s has continuation_of=%s (default)",
            req.request_id, req.continuation_of,
        )

    def test_from_engine_core_request_propagates_continuation_of(self):
        """Request.from_engine_core_request passes continuation_of through."""
        from vllm.v1.engine import EngineCoreRequest

        ecr = EngineCoreRequest(
            request_id="ecr-1",
            prompt_token_ids=[10, 20, 30],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            continuation_of="parent-ecr",
            continuation_token_ids=[40, 50],
        )

        req = Request.from_engine_core_request(ecr, block_hasher=None)

        assert req.continuation_of == "parent-ecr"
        logger.debug(
            "from_engine_core_request: continuation_of=%s",
            req.continuation_of,
        )

    def test_from_engine_core_request_none_continuation(self):
        """from_engine_core_request with no continuation_of yields None."""
        from vllm.v1.engine import EngineCoreRequest

        ecr = EngineCoreRequest(
            request_id="ecr-2",
            prompt_token_ids=[10, 20, 30],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )

        req = Request.from_engine_core_request(ecr, block_hasher=None)

        assert req.continuation_of is None
        logger.debug(
            "from_engine_core_request: continuation_of=%s (default)",
            req.continuation_of,
        )
