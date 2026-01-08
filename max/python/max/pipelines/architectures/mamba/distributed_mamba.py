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
    Module,
    Signals,
    VocabParallelEmbedding,
)

logger = logging.getLogger("max.pipelines")
from .model_config import MambaConfig


class DistributedMamba(Module):
    def __init__(self, config: MambaConfig) -> None:
        assert len(config.devices) > 1
        self.config = config

        # TODO: Implement distributed Mamba forward pass
        raise NotImplementedError(
         "DistributedMamba.__init__ is not yet implemented. "
         "Distributed Mamba models are not yet supported."
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

