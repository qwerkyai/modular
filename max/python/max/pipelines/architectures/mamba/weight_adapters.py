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

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import MAXModelConfig, PipelineConfig, SupportedEncoding
from transformers import AutoConfig

# Maps from Safetensor to MAX weight names.
MAMBA_SAFETENSOR_MAPPING = {
    "backbone.": "",  # Removes the "backbone" prefix if present.
    "model.": "",  # Removes the "model" prefix.
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format."""
    new_state_dict: dict[str, WeightData] = {}
    # Map the weight names.
    for safetensor_name, value in state_dict.items():
        max_name = safetensor_name
        for before, after in MAMBA_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    model_config = pipeline_config.model_config

    if model_config._applied_dtype_cast_from:
        cast_from = model_config._applied_dtype_cast_from
        cast_to = model_config._applied_dtype_cast_to
        assert cast_to, (
            "Invalid configuration: _applied_dtype_cast_to is not set but _applied_dtype_cast_from is set. "
            "This should not happen."
        )
        for key, weight_data in new_state_dict.items():
            if weight_data.dtype == cast_from.dtype:
                new_state_dict[key] = weight_data.astype(cast_to.dtype)

    return new_state_dict

