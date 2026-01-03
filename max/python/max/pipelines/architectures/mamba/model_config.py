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

from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.nn import (
    DistributedGemmConfig,
    ReturnHiddenStates,
    ReturnLogits,
)
from max.nn.float8_config import Float8Config
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
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
    hidden_size: int
    num_hidden_layers: int
    max_seq_len: int
    intermediate_size: int
    vocab_size: int
    dtype: DType
    model_quantization_encoding: QuantizationEncoding | None
    quantization_config: QuantizationConfig | None
    kv_params: KVCacheParams
    return_logits: ReturnLogits
    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    norm_dtype: DType | None
    rms_norm_eps: float | None
    tie_word_embeddings: bool
    devices: list[DeviceRef]
    float8_config: Float8Config | None
    lora_config: LoRAConfig | None = None
    dist_gemm_config: DistributedGemmConfig | None = None
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    d_state: int = 16
    dt_rank: int | None = None
    conv_kernel: int = 4
    x_proj_dim: int | None = None  # Can be inferred from weight shape

    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class MambaConfig(MAXModelConfig, MambaConfigBase):
    """Implementation of MAXModelConfig for Mamba models."""

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        # TODO: Implement KV cache params for Mamba
        # Mamba uses state space models, so KV cache may work differently
        # For now, use PAGED strategy if prefix caching is enabled
        cache_strategy = kv_cache_config.cache_strategy
        if kv_cache_config.enable_prefix_caching:
            from max.nn.kv_cache import KVCacheStrategy
            if cache_strategy == KVCacheStrategy.MODEL_DEFAULT:
                cache_strategy = KVCacheStrategy.PAGED
        
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=1,  # Placeholder - Mamba doesn't use attention heads
            head_dim=1,  # Placeholder
            num_layers=MambaConfig.get_num_layers(huggingface_config),
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            devices=devices,
            data_parallel_degree=pipeline_config.model_config.data_parallel_degree,
        )

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
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
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

        # Determine norm_dtype.
        # Note: Weight adapter removes "backbone." and "model." prefixes,
        # so we check for the weight name after prefix removal.
        norm_dtype = None
        if "embeddings.weight" in state_dict:
            weight_data = state_dict["embeddings.weight"]
            if weight_data is not None and hasattr(weight_data, "dtype"):
                norm_dtype = weight_data.dtype
        elif "embedding.weight" in state_dict:
            weight_data = state_dict["embedding.weight"]
            if weight_data is not None and hasattr(weight_data, "dtype"):
                norm_dtype = weight_data.dtype

        # Get RMS norm epsilon
        rms_norm_eps = None
        if norm_method == "rms_norm":
            rms_norm_eps = getattr(
                huggingface_config, "layer_norm_epsilon", 1e-5
            )

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        tie_word_embeddings = getattr(
            huggingface_config, "tie_word_embeddings", False
        )

        # Get Mamba-specific config values
        d_state = getattr(huggingface_config, "d_state", 16)
        dt_rank = getattr(huggingface_config, "dt_rank", None)
        conv_kernel = getattr(huggingface_config, "conv_kernel", 4)
        
        # Infer x_proj_dim and dt_rank from x_proj weight shape if available
        intermediate_size = getattr(
            huggingface_config, "d_inner", huggingface_config.d_model * 2
        )
        x_proj_dim = None
        x_proj_key = "layers.0.mixer.x_proj.weight"
        if x_proj_key in state_dict:
            weight_data = state_dict[x_proj_key]
            if weight_data is not None and hasattr(weight_data, "shape"):
                # x_proj weight shape is [out_dim, in_dim]
                # Convert Dim to int if needed - handle both Dim and int
                shape_0 = weight_data.shape[0]
                if hasattr(shape_0, 'value'):
                    x_proj_dim = int(shape_0.value)
                else:
                    x_proj_dim = int(shape_0)
                
                # Try to infer dt_rank from x_proj_dim
                # x_proj projects to dt_rank + 2 * n_groups * d_state
                # where n_groups = intermediate_size // d_state
                n_groups = intermediate_size // d_state
                # Calculate dt_rank from: x_proj_dim = dt_rank + 2 * n_groups * d_state
                calculated_dt_rank = x_proj_dim - 2 * n_groups * d_state
                if calculated_dt_rank > 0:
                    dt_rank = calculated_dt_rank
        
        # If dt_rank is still None, calculate it
        if dt_rank is None:
            hidden_size = huggingface_config.d_model
            dt_rank = max(16, hidden_size // 16)
        
        return MambaConfig(
            hidden_size=huggingface_config.d_model,
            num_hidden_layers=huggingface_config.n_layer,
            intermediate_size=getattr(
                huggingface_config, "d_inner", huggingface_config.d_model * 2
            ),
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            model_quantization_encoding=pipeline_config.model_config.graph_quantization_encoding,
            quantization_config=pipeline_config.model_config._quant_config,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_seq_len=MambaConfig.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            kv_params=MambaConfig.get_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            norm_method=norm_method,
            norm_dtype=norm_dtype,
            d_state=d_state,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
            x_proj_dim=x_proj_dim,
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            devices=device_refs,
            float8_config=float8_config,
            use_subgraphs=pipeline_config.model_config.use_subgraphs,
            dist_gemm_config=DistributedGemmConfig.generate(),
            lora_config=pipeline_config.lora_config,
            data_parallel_degree=pipeline_config.model_config.data_parallel_degree,
        )

