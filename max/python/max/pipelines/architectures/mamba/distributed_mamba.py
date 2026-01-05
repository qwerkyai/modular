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
"""Build a Mamba model that runs on multiple devices."""

from __future__ import annotations

import functools
import logging
from collections import defaultdict

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.quantization import QuantizationEncoding
from max.nn import (
    ColumnParallelLinear,
    DistributedTransformer,
    DistributedTransformerBlock,
    Linear,
    RMSNorm,
    Signals,
    VocabParallelEmbedding,
)

logger = logging.getLogger("max.pipelines")
from .model_config import MambaConfig


class DistributedMamba(DistributedTransformer):
    def __init__(self, config: MambaConfig) -> None:
        assert len(config.devices) > 1
        self.config = config

        if config.quantization_config:
            raise ValueError(
                "Model contains GPTQ weights. This is currently not supported with multiple GPUs."
            )

        if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
            raise ValueError(
                "`norm_method` must be `RMSNorm` and `rms_norm_eps` cannot be "
                "None for model that uses `RMSNorm`."
            )

        create_distributed_norm = functools.partial(
            RMSNorm,
            dim=config.hidden_size,
            dtype=config.norm_dtype or config.dtype,
            eps=config.rms_norm_eps,
        )

        fp8_cfg = config.float8_config
        linear_cls = functools.partial(Linear, float8_config=fp8_cfg)

        layers = []
        sublayer_groupings_dict = defaultdict(list)

        for layer_idx in range(config.num_hidden_layers):
            # Deal with the float8 case where individual layers are ignored
            # specially: assume bfloat16 dtype for "ignored" layers in fp8
            # quantized models.
            mlp_dtype = (
                DType.bfloat16
                if fp8_cfg and layer_idx not in fp8_cfg.mlp_in_float8
                else config.dtype
            )

            sublayer_groupings_dict[mlp_dtype].append(layer_idx)

            # TODO: Implement Mamba-specific distributed layers
            # For now, creating placeholder structure
            # layers.append(...)

        subgraph_layer_groups = list(sublayer_groupings_dict.values())

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if fp8_cfg and fp8_cfg.embedding_output_dtype:
            embedding_output_dtype = fp8_cfg.embedding_output_dtype

        embedding_layer = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices,
            quantization_encoding=embedding_output_quantization,
        )
        output = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            devices=config.devices,
            tied_weight=(
                embedding_layer.weight if config.tie_word_embeddings else None
            ),
            quantization_encoding=embedding_output_quantization,
        )

        # TODO: Initialize with proper Mamba distributed architecture
        # For now, using placeholder structure
        super().__init__(
            dim=config.hidden_size,
            n_heads=1,  # Placeholder - Mamba doesn't use attention heads
            layers=layers,
            norm=create_distributed_norm(),
            output=output,
            embedding=embedding_layer,
            devices=config.devices,
            rope=None,  # Mamba doesn't use RoPE
            return_logits=config.return_logits,
            use_subgraphs=config.use_subgraphs,
            subgraph_layer_groups=subgraph_layer_groups,
            logits_scaling=1.0,
        )

    def input_types(
        self,
    ) -> tuple[TensorType | BufferType, ...]:
        # TODO: Move input symbol computation from the manager classes.
        device_ref = self.config.devices[0]

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        signals = Signals(devices=self.config.devices)

        # Explicitly construct tuple with mixed types
        signal_buffer_types: list[BufferType] = signals.input_types()

        # Build the complete input types list
        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        all_input_types.extend(signal_buffer_types)

        return tuple(all_input_types)

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: list[BufferValue],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Forward pass for distributed Mamba model without KV cache.
        
        Args:
            tokens: Input token IDs.
            signal_buffers: Signal buffers for multi-GPU communication.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged tensor processing.
            
        Returns:
            Tuple of output tensors (logits, and optionally offsets and hidden states).
        """
        # TODO: Implement proper distributed Mamba forward pass
        # For now, this is a placeholder that doesn't use KV cache
        raise NotImplementedError(
            "DistributedMamba.__call__ is not yet implemented. "
            "Mamba models do not use KV cache."
        )

