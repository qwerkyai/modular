# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Config for Mamba models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Any

logger = logging.getLogger("max.pipelines")

from max.dtype import DType
from max.graph import DeviceRef
from max.nn import (
    ReturnHiddenStates,
    ReturnLogits,
)
from max.nn.float8_config import Float8Config
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
    PipelineConfig,
    upper_bounded_default,
)
from transformers import AutoConfig


@dataclass
class MambaConfigBase(MAXModelConfigBase):
    """Base configuration for Mamba models."""

    # Required fields
    max_seq_len: int
    vocab_size: int
    dtype: DType
    devices: list[DeviceRef]
    float8_config: Float8Config | None
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    ssm_cfg: dict[str, Any] | None = None
    attn_layer_idx: list[int] | None = None
    attn_cfg: dict[str, Any] | None = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class MambaConfig(MAXModelConfig, MambaConfigBase):
    """Implementation of MAXModelConfig for Mamba models."""

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.n_layer

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=getattr(
                    huggingface_config, "max_position_embeddings", 2048
                ),
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Mamba, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({getattr(huggingface_config, 'max_position_embeddings', 2048)})."
            ) from e

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict,
        dtype: DType,
        n_devices: int,
    ) -> MambaConfig:
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs[:n_devices]
        ]

        # Parse the float8 config from compressed-tensors or FBGEMM.
        from max.pipelines.lib.float8 import parse_float8_config

        float8_config = parse_float8_config(
            huggingface_config, state_dict, dtype
        )

        # Get Mamba-specific config values
        d_model = huggingface_config.d_model
        d_intermediate = huggingface_config.d_intermediate
        n_layer = huggingface_config.n_layer
        ssm_cfg = huggingface_config.ssm_cfg
        attn_layer_idx = huggingface_config.attn_layer_idx
        attn_cfg = huggingface_config.attn_cfg
        rms_norm = huggingface_config.rms_norm
        residual_in_fp32 = huggingface_config.residual_in_fp32
        fused_add_norm = huggingface_config.fused_add_norm
        pad_vocab_size_multiple = huggingface_config.pad_vocab_size_multiple
        tie_embeddings = huggingface_config.tie_embeddings
        
        return MambaConfig(
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            max_seq_len=MambaConfig.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            d_model=d_model,
            d_intermediate=d_intermediate,
            n_layer=n_layer,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            pad_vocab_size_multiple=pad_vocab_size_multiple,
            tie_embeddings=tie_embeddings,
            devices=device_refs,
            float8_config=float8_config,
            use_subgraphs=pipeline_config.model_config.use_subgraphs,
            data_parallel_degree=pipeline_config.model_config.data_parallel_degree,
        )

