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
    # Mamba-specific weight name mappings
    # HuggingFace format: layers.{i}.mixer.A_log -> MAX format: layers.{i}.mixer.A_log
    # HuggingFace format: layers.{i}.mixer.D -> MAX format: layers.{i}.mixer.D
    # These should match automatically, but we handle any prefix differences here.
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.
    
    This function handles weight name mapping from HuggingFace Mamba format to MAX format.
    Key weight names that are handled:
    - layers.{i}.mixer.A_log: State transition matrix in log space (intermediate_size, d_state)
    - layers.{i}.mixer.D: Skip connection parameter (intermediate_size,)
    - layers.{i}.mixer.conv1d.weight: Causal 1D convolution weight (intermediate_size, conv_width)
    - layers.{i}.mixer.conv1d.bias: Causal 1D convolution bias (intermediate_size,)
    - layers.{i}.mixer.in_proj.weight: Input projection weight
    - layers.{i}.mixer.x_proj.weight: State space parameter projection weight
    - layers.{i}.mixer.dt_proj.weight: Delta projection weight
    - layers.{i}.mixer.dt_proj.bias: Delta projection bias (dt_bias)
    - layers.{i}.mixer.out_proj.weight: Output projection weight
    
    Args:
        state_dict: Dictionary mapping weight names to Weights objects from safetensors.
        huggingface_config: HuggingFace model configuration.
        pipeline_config: MAX pipeline configuration.
        **unused_kwargs: Additional unused keyword arguments.
        
    Returns:
        Dictionary mapping MAX weight names to WeightData objects.
    """
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

