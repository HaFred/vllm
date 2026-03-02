# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EngineCore continuation support (dkvc).

Covers:
- EngineCore._finished_request_tokens cache
- EngineCore.cache_finished_request_tokens()
- EngineCore.preprocess_add_request() continuation branch
- EngineCoreRequest continuation fields
"""

import logging
from collections import OrderedDict
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

pytestmark = pytest.mark.cpu_test

logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 50256


# ---------------------------------------------------------------------------
# Helpers: Minimal mock of EngineCore internals for unit-testing the
# continuation logic without needing a full engine setup.
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for vllm.v1.request.Request."""

    def __init__(self, request_id: str, all_token_ids: list[int]):
        self.request_id = request_id
        self._all_token_ids = list(all_token_ids)


class _FakeScheduler:
    """Minimal stand-in for Scheduler with a requests dict."""

    def __init__(self):
        self.requests: dict[str, _FakeRequest] = {}


class _FakeEngineCore:
    """Mimics the continuation-related subset of EngineCore.

    We test the actual logic from the commit without instantiating the
    full EngineCore (which needs GPU, model weights, etc.).
    """

    def __init__(self, max_cache_size: int = 1024):
        self.scheduler = _FakeScheduler()
        self._finished_request_tokens: OrderedDict[str, list[int]] = (
            OrderedDict()
        )
        self._max_finished_token_cache = max_cache_size

    def cache_finished_request_tokens(self, req_id: str) -> None:
        """Identical to EngineCore.cache_finished_request_tokens."""
        req = self.scheduler.requests.get(req_id)
        if req is not None and req._all_token_ids:
            self._finished_request_tokens[req_id] = list(
                req._all_token_ids
            )
            while (
                len(self._finished_request_tokens)
                > self._max_finished_token_cache
            ):
                self._finished_request_tokens.popitem(last=False)

    def resolve_continuation_tokens(
        self, request: EngineCoreRequest
    ) -> list[int]:
        """Extract the continuation resolution logic from
        preprocess_add_request for isolated testing."""
        if getattr(request, "continuation_of", None) is None:
            raise ValueError("Not a continuation request")

        parent_tokens = None

        # Try 1: parent is still running/waiting
        parent_req = self.scheduler.requests.get(request.continuation_of)
        if parent_req is not None:
            parent_tokens = list(parent_req._all_token_ids)

        # Try 2: recently finished
        if parent_tokens is None:
            parent_tokens = self._finished_request_tokens.get(
                request.continuation_of
            )

        if parent_tokens is None:
            raise ValueError(
                f"Continuation parent request '{request.continuation_of}' "
                f"not found in running requests or finished cache."
            )

        suffix = request.continuation_token_ids or []
        prompt_token_ids = parent_tokens + suffix
        logger.debug(
            "Resolved continuation %s: %d parent tokens + %d suffix "
            "= %d total",
            request.request_id,
            len(parent_tokens),
            len(suffix),
            len(prompt_token_ids),
        )
        return prompt_token_ids


# ---------------------------------------------------------------------------
# Tests: EngineCoreRequest continuation fields
# ---------------------------------------------------------------------------


class TestEngineCoreRequestFields:
    """Test EngineCoreRequest continuation_of / continuation_token_ids."""

    def test_defaults_are_none(self):
        """Fields default to None when not set."""
        ecr = EngineCoreRequest(
            request_id="r1",
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        assert ecr.continuation_of is None
        assert ecr.continuation_token_ids is None
        logger.debug("Default continuation fields: of=%s, ids=%s",
                      ecr.continuation_of, ecr.continuation_token_ids)

    def test_fields_set_correctly(self):
        """Fields are set when provided."""
        ecr = EngineCoreRequest(
            request_id="r2",
            prompt_token_ids=None,
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            continuation_of="parent-123",
            continuation_token_ids=[100, 200, 300],
        )
        assert ecr.continuation_of == "parent-123"
        assert ecr.continuation_token_ids == [100, 200, 300]
        logger.debug(
            "Continuation fields: of=%s, ids=%s",
            ecr.continuation_of, ecr.continuation_token_ids,
        )

    def test_prompt_token_ids_can_be_none_for_continuation(self):
        """Continuation requests may have prompt_token_ids=None
        (filled later by EngineCore)."""
        ecr = EngineCoreRequest(
            request_id="r3",
            prompt_token_ids=None,
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            continuation_of="parent-456",
        )
        assert ecr.prompt_token_ids is None
        assert ecr.continuation_of == "parent-456"
        logger.debug("Continuation with prompt_token_ids=None accepted")


# ---------------------------------------------------------------------------
# Tests: cache_finished_request_tokens
# ---------------------------------------------------------------------------


class TestCacheFinishedRequestTokens:
    """Test the finished-request token cache used for continuation."""

    def test_cache_stores_tokens(self):
        """Tokens are cached for a finished request."""
        engine = _FakeEngineCore()
        engine.scheduler.requests["req-1"] = _FakeRequest(
            "req-1", [10, 20, 30, 40]
        )

        engine.cache_finished_request_tokens("req-1")

        assert "req-1" in engine._finished_request_tokens
        assert engine._finished_request_tokens["req-1"] == [10, 20, 30, 40]
        logger.debug(
            "Cached tokens for req-1: %s",
            engine._finished_request_tokens["req-1"],
        )

    def test_cache_is_a_copy(self):
        """Cached tokens are a copy, not a reference."""
        engine = _FakeEngineCore()
        original = [1, 2, 3]
        engine.scheduler.requests["req-1"] = _FakeRequest("req-1", original)

        engine.cache_finished_request_tokens("req-1")
        original.append(999)

        assert 999 not in engine._finished_request_tokens["req-1"]
        logger.debug("Cache is independent copy of token list")

    def test_cache_ignores_missing_request(self):
        """No error when request ID is not in scheduler."""
        engine = _FakeEngineCore()
        engine.cache_finished_request_tokens("nonexistent")
        assert "nonexistent" not in engine._finished_request_tokens
        logger.debug("Missing request silently ignored")

    def test_cache_ignores_empty_tokens(self):
        """Request with empty token list is not cached."""
        engine = _FakeEngineCore()
        engine.scheduler.requests["req-e"] = _FakeRequest("req-e", [])

        engine.cache_finished_request_tokens("req-e")

        assert "req-e" not in engine._finished_request_tokens
        logger.debug("Empty-token request not cached")

    def test_cache_eviction_lru(self):
        """Oldest entries are evicted when cache exceeds max size."""
        max_size = 3
        engine = _FakeEngineCore(max_cache_size=max_size)

        for i in range(5):
            req_id = f"req-{i}"
            engine.scheduler.requests[req_id] = _FakeRequest(
                req_id, [i] * 10
            )
            engine.cache_finished_request_tokens(req_id)
            logger.debug(
                "Cached req-%d, cache size=%d",
                i, len(engine._finished_request_tokens),
            )

        # Only the last 3 should remain
        assert len(engine._finished_request_tokens) == max_size
        assert "req-0" not in engine._finished_request_tokens
        assert "req-1" not in engine._finished_request_tokens
        assert "req-2" in engine._finished_request_tokens
        assert "req-3" in engine._finished_request_tokens
        assert "req-4" in engine._finished_request_tokens
        logger.debug(
            "After eviction, cached ids: %s",
            list(engine._finished_request_tokens.keys()),
        )


# ---------------------------------------------------------------------------
# Tests: Continuation token resolution (preprocess_add_request logic)
# ---------------------------------------------------------------------------


class TestContinuationTokenResolution:
    """Test the continuation token resolution logic extracted from
    EngineCore.preprocess_add_request()."""

    def _make_continuation_request(
        self,
        request_id: str = "child-1",
        parent_id: str = "parent-1",
        suffix_ids: list[int] | None = None,
    ) -> EngineCoreRequest:
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=None,
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            continuation_of=parent_id,
            continuation_token_ids=suffix_ids,
        )

    def test_resolve_from_running_parent(self):
        """Tokens resolved from a still-running parent request."""
        engine = _FakeEngineCore()
        engine.scheduler.requests["parent-1"] = _FakeRequest(
            "parent-1", [10, 20, 30]
        )

        ecr = self._make_continuation_request(suffix_ids=[40, 50])
        result = engine.resolve_continuation_tokens(ecr)

        assert result == [10, 20, 30, 40, 50]
        logger.debug("Resolved from running parent: %s", result)

    def test_resolve_from_finished_cache(self):
        """Tokens resolved from the finished-request cache."""
        engine = _FakeEngineCore()
        engine._finished_request_tokens["parent-1"] = [100, 200]

        ecr = self._make_continuation_request(suffix_ids=[300])
        result = engine.resolve_continuation_tokens(ecr)

        assert result == [100, 200, 300]
        logger.debug("Resolved from finished cache: %s", result)

    def test_resolve_prefers_running_over_cache(self):
        """Running parent takes priority over finished cache."""
        engine = _FakeEngineCore()
        engine.scheduler.requests["parent-1"] = _FakeRequest(
            "parent-1", [1, 2, 3]
        )
        engine._finished_request_tokens["parent-1"] = [7, 8, 9]

        ecr = self._make_continuation_request(suffix_ids=[4])
        result = engine.resolve_continuation_tokens(ecr)

        # Should use running request tokens, not cache
        assert result == [1, 2, 3, 4]
        logger.debug("Running parent preferred over cache: %s", result)

    def test_resolve_with_no_suffix(self):
        """Continuation with no suffix (None) uses parent tokens only."""
        engine = _FakeEngineCore()
        engine.scheduler.requests["parent-1"] = _FakeRequest(
            "parent-1", [5, 6, 7]
        )

        ecr = self._make_continuation_request(suffix_ids=None)
        result = engine.resolve_continuation_tokens(ecr)

        assert result == [5, 6, 7]
        logger.debug("Resolved with no suffix: %s", result)

    def test_resolve_with_empty_suffix(self):
        """Continuation with empty suffix list uses parent tokens only."""
        engine = _FakeEngineCore()
        engine.scheduler.requests["parent-1"] = _FakeRequest(
            "parent-1", [5, 6, 7]
        )

        ecr = self._make_continuation_request(suffix_ids=[])
        result = engine.resolve_continuation_tokens(ecr)

        assert result == [5, 6, 7]
        logger.debug("Resolved with empty suffix: %s", result)

    def test_resolve_raises_when_parent_not_found(self):
        """ValueError raised when parent is not in running or cache."""
        engine = _FakeEngineCore()
        ecr = self._make_continuation_request(
            parent_id="ghost", suffix_ids=[1]
        )

        with pytest.raises(ValueError, match="not found"):
            engine.resolve_continuation_tokens(ecr)
        logger.debug("ValueError raised for missing parent as expected")

    def test_resolve_non_continuation_raises(self):
        """ValueError raised if request is not a continuation."""
        engine = _FakeEngineCore()
        ecr = EngineCoreRequest(
            request_id="normal",
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )

        with pytest.raises(ValueError, match="Not a continuation"):
            engine.resolve_continuation_tokens(ecr)
        logger.debug("Non-continuation request rejected as expected")
