# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.collection_utils import flatten_2d_lists
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    async_tensor_h2d,
    cuda_device_count_stateless,
    current_stream,
    direct_register_custom_op,
    get_dtype_size,
    get_kv_cache_torch_dtype,
    make_tensor_with_pad,
    set_random_seed,
    supports_custom_op,
    supports_dynamo,
)

MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len
