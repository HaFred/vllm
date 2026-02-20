# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OpenAI CompletionRequest continuation fields (dkvc).

Covers:
- CompletionRequest.continuation_of field
- CompletionRequest.continuation_suffix field
- Default values and explicit setting
"""

import logging

import pytest

from vllm.entrypoints.openai.protocol import CompletionRequest

pytestmark = pytest.mark.cpu_test

logger = logging.getLogger(__name__)


class TestCompletionRequestContinuationFields:
    """Test continuation_of and continuation_suffix on CompletionRequest."""

    def test_defaults_to_none(self):
        """Both continuation fields default to None."""
        req = CompletionRequest(
            model="test-model",
            prompt="Hello world",
        )
        assert req.continuation_of is None
        assert req.continuation_suffix is None
        logger.debug(
            "Default fields: continuation_of=%s, continuation_suffix=%s",
            req.continuation_of, req.continuation_suffix,
        )

    def test_continuation_of_set(self):
        """continuation_of can be set to a parent request ID."""
        req = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            continuation_of="parent-req-123",
        )
        assert req.continuation_of == "parent-req-123"
        assert req.continuation_suffix is None
        logger.debug("continuation_of=%s", req.continuation_of)

    def test_continuation_suffix_set(self):
        """continuation_suffix can be set alongside continuation_of."""
        req = CompletionRequest(
            model="test-model",
            prompt="placeholder",
            continuation_of="parent-req-456",
            continuation_suffix="</think>\n<|sid_begin|>",
        )
        assert req.continuation_of == "parent-req-456"
        assert req.continuation_suffix == "</think>\n<|sid_begin|>"
        logger.debug(
            "continuation_of=%s, continuation_suffix=%r",
            req.continuation_of, req.continuation_suffix,
        )

    def test_continuation_suffix_without_continuation_of(self):
        """continuation_suffix can technically be set without continuation_of
        (the engine will ignore it)."""
        req = CompletionRequest(
            model="test-model",
            prompt="test",
            continuation_suffix="some suffix",
        )
        assert req.continuation_of is None
        assert req.continuation_suffix == "some suffix"
        logger.debug(
            "Suffix without parent: continuation_of=%s, suffix=%r",
            req.continuation_of, req.continuation_suffix,
        )

    def test_extra_body_continuation_fields(self):
        """Fields can be set via model_validate (simulating extra_body)."""
        data = {
            "model": "test-model",
            "prompt": "hello",
            "continuation_of": "parent-abc",
            "continuation_suffix": "</think>\ntoken",
        }
        req = CompletionRequest.model_validate(data)
        assert req.continuation_of == "parent-abc"
        assert req.continuation_suffix == "</think>\ntoken"
        logger.debug(
            "From model_validate: continuation_of=%s, suffix=%r",
            req.continuation_of, req.continuation_suffix,
        )

    def test_serialization_roundtrip(self):
        """Continuation fields survive serialization roundtrip."""
        req = CompletionRequest(
            model="test-model",
            prompt="test",
            continuation_of="parent-rt",
            continuation_suffix="suffix-rt",
        )
        dumped = req.model_dump()
        assert dumped["continuation_of"] == "parent-rt"
        assert dumped["continuation_suffix"] == "suffix-rt"
        logger.debug("Serialized continuation fields: %s",
                      {k: dumped[k] for k in
                       ("continuation_of", "continuation_suffix")})

    def test_none_fields_excluded_in_dump(self):
        """None continuation fields appear as None in dump."""
        req = CompletionRequest(
            model="test-model",
            prompt="test",
        )
        dumped = req.model_dump()
        assert dumped.get("continuation_of") is None
        assert dumped.get("continuation_suffix") is None
        logger.debug("None fields in dump verified")
