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
"""Mamba block implementation based on the reference implementation.

Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, Weight, ops
from max.nn import Layer, Linear, Module
from max.nn.conv import causal_conv1d_fn, causal_conv1d_update_fn
from max.nn.norm import layer_norm_fn
from max.nn.norm.layer_norm import LayerNorm
from max.nn.norm.rms_norm import RMSNorm


class MambaSSM(Module):
    """State Space Model layer for Mamba.
    
    Implements the Mamba SSM layer based on the reference implementation:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    
    The layer performs:
    1. Input projection and gating (gate + up)
    2. Causal 1D convolution
    3. State space parameter projection (B, C, delta)
    4. Selective scan computation
    5. Output projection
    
    Note: Full implementation with selective_scan_fwd requires proper handling
    of ragged tensors and tensor reshaping. This is a foundational implementation
    that can be extended as kernel support improves.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: DType,
        device: DeviceRef,
        linear_cls: Callable[..., Linear],
        d_state: int = 16,
        dt_rank: str | int = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        delta_softplus: bool = True,
        conv_kernel: int = 4,
        x_proj_dim: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        use_fast_path: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        """Initialize the Mamba SSM layer.
        
        Matches the reference implementation:
        https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        
        Args:
            hidden_size: Hidden dimension size (d_model in reference).
            intermediate_size: Intermediate dimension for SSM expansion (d_inner in reference).
            dtype: Data type for weights.
            device: Device to place weights on.
            linear_cls: Linear layer class to use.
            d_state: State space dimension (default: 16).
            dt_rank: Rank of delta projection. "auto" sets it to ceil(hidden_size / 16).
            conv_bias: Whether to use bias in conv1d.
            bias: Whether to use bias in linear projections.
            delta_softplus: Whether to apply softplus to delta values (default: True).
            conv_kernel: Convolution kernel size (default: 4).
            x_proj_dim: Output dimension of x_proj (inferred from weights if None).
            dt_min: Minimum value for dt initialization (default: 0.001).
            dt_max: Maximum value for dt initialization (default: 0.1).
            dt_init: Initialization method for dt_proj weights ("random" or "constant").
            dt_scale: Scale factor for dt_proj weight initialization (default: 1.0).
            dt_init_floor: Floor value for dt initialization (default: 1e-4).
            use_fast_path: Whether to use fast fused kernel path when available (default: True).
            layer_idx: Layer index for inference caching (default: None).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.device = device
        self.d_state = d_state
        self.delta_softplus = delta_softplus
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        
        # Determine dt_rank - match reference: math.ceil(d_model / 16)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(hidden_size / 16)
        else:
            self.dt_rank = int(dt_rank)
        
        # Input projection: projects to intermediate_size * 2 (gate + up)
        self.in_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=intermediate_size * 2,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )
        
        # Conv1d layer for sequence processing
        # The conv1d processes the gated input with a causal convolution
        # Weight shape: (intermediate_size, conv_width)
        # Use conv_kernel from config
        self.conv_width = conv_kernel
        
        # Use Conv1D for the convolution
        # Note: Conv1D doesn't have a causal parameter, so we'll use causal_conv1d_fn
        # or handle causal padding manually
        # For now, create Conv1D and we'll use causal_conv1d_fn in forward
        self.conv1d_weight = Weight(
            name="conv1d_weight",
            dtype=dtype,
            shape=(intermediate_size, self.conv_width),  # (channels, width)
            device=device,
        )
        self.conv1d_bias: Weight | None = None
        if conv_bias:
            self.conv1d_bias = Weight(
                name="conv1d_bias",
                dtype=dtype,
                shape=(intermediate_size,),
                device=device,
            )
        
        # State space parameter projections
        # x_proj: projects conv output to (B, C, delta) parameters
        # B and C have shape (batch, n_groups, dstate, seqlen)
        # delta has shape (batch, intermediate_size, seqlen)
        # We use n_groups = intermediate_size // d_state for grouping
        self.n_groups = intermediate_size // d_state
        
        # x_proj_dim can be provided directly (inferred from weights) or calculated
        # If not provided, calculate: dt_rank + 2 * n_groups * d_state
        if x_proj_dim is not None:
            # Use provided x_proj_dim (inferred from actual weight shape)
            # Recalculate dt_rank to match the actual x_proj_dim
            # Formula: x_proj_dim = dt_rank + 2 * n_groups * d_state
            # So: dt_rank = x_proj_dim - 2 * n_groups * d_state
            bc_dim = 2 * self.n_groups * d_state
            calculated_dt_rank = x_proj_dim - bc_dim
            if calculated_dt_rank > 0:
                # Normal case: x_proj_dim matches expected formula
                self.dt_rank = calculated_dt_rank
                self.bc_dim = bc_dim
            else:
                # x_proj_dim doesn't match expected formula
                # This can happen if the model architecture is different
                # For some models, x_proj might project to intermediate_size directly
                # In that case, we need to recalculate n_groups based on x_proj_dim
                # Try: if x_proj_dim == intermediate_size, then maybe it's a different architecture
                if x_proj_dim == intermediate_size:
                    # Special case: x_proj projects to intermediate_size
                    # This might mean dt_rank = intermediate_size and no BC projection
                    # But that doesn't match standard Mamba architecture
                    # For now, use a heuristic: assume dt_rank is a reasonable fraction
                    self.dt_rank = max(16, min(x_proj_dim // 4, hidden_size // 16))
                    self.bc_dim = x_proj_dim - self.dt_rank
                    # Recalculate n_groups to match bc_dim
                    if self.bc_dim > 0 and d_state > 0:
                        # bc_dim = 2 * n_groups * d_state, so n_groups = bc_dim / (2 * d_state)
                        self.n_groups = self.bc_dim // (2 * d_state)
                        if self.n_groups == 0:
                            self.n_groups = 1
                else:
                    # Use minimum dt_rank and rest for BC
                    min_dt_rank = max(16, hidden_size // 16)
                    if x_proj_dim > min_dt_rank:
                        self.dt_rank = min_dt_rank
                        self.bc_dim = x_proj_dim - min_dt_rank
                        # Recalculate n_groups to match bc_dim
                        if self.bc_dim > 0 and d_state > 0:
                            self.n_groups = self.bc_dim // (2 * d_state)
                            if self.n_groups == 0:
                                self.n_groups = 1
                    else:
                        # x_proj_dim is too small, use it all for dt_rank
                        self.dt_rank = x_proj_dim
                        self.bc_dim = 0
                        self.n_groups = 1  # Minimum to avoid division by zero
        else:
            x_proj_dim = self.dt_rank + 2 * self.n_groups * d_state  # dt_rank + B + C
            self.bc_dim = 2 * self.n_groups * d_state
        
        self.x_proj = linear_cls(
            in_dim=intermediate_size,
            out_dim=x_proj_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        
        # Delta projection parameters
        self.dt_proj = linear_cls(
            in_dim=self.dt_rank,
            out_dim=intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=True,  # dt_bias
        )
        
        # State transition matrix A (intermediate_size, d_state)
        # A is initialized in log space and shared across all sequences
        # A_log is a learnable parameter, A = exp(A_log)
        # Shape: (intermediate_size, d_state)
        # 
        # Weight loading: This weight is automatically registered by Module.__setattr__
        # and will be loaded from state dict with name "layers.{i}.mixer.A_log"
        # when load_state_dict() is called. The weight should be in log space
        # (i.e., values should be log(A), not A itself).
        self.A_log = Weight(
            name="A_log",
            dtype=dtype,
            shape=(intermediate_size, d_state),
            device=device,
        )
        
        # Skip connection D (optional, shape: intermediate_size)
        # D is a learnable parameter for the skip connection
        # 
        # Weight loading: This weight is automatically registered by Module.__setattr__
        # and will be loaded from state dict with name "layers.{i}.mixer.D"
        # when load_state_dict() is called.
        self.D = Weight(
            name="D",
            dtype=dtype,
            shape=(intermediate_size,),
            device=device,
        )
        
        # Output projection
        self.out_proj = linear_cls(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )
        
        # State buffers for autoregressive generation
        # These will be initialized on first use
        self._conv_state: Weight | None = None  # (batch, intermediate_size, width-1)
        self._ssm_state: Weight | None = None  # (batch, intermediate_size, d_state)

    def __call__(
        self,
        x: TensorValue,
        input_row_offsets: TensorValue | None = None,
        inference_params: Optional[dict] = None,
        **kwargs,
    ) -> TensorValue:
        """Forward pass through the SSM layer.
        
        Args:
            x: Input hidden states of shape (batch * seqlen, hidden_size) or (batch, seqlen, hidden_size).
            input_row_offsets: Row offsets for ragged tensor processing.
            inference_params: Optional inference parameters for caching.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Output after SSM transformation.
        """
        # Step 1: Check if we should use step method (autoregressive mode)
        # Only use step when we *know* we have a single token (seqlen==1); otherwise fall back to prefill
        use_step = False
        if inference_params is not None and self.layer_idx is not None:
            seqlen_offset = inference_params.get("seqlen_offset", 0)

            def _is_one(dim: Dim | int) -> bool:
                try:
                    return int(dim) == 1
                except (TypeError, ValueError):
                    return False

            # Require concrete single-token inputs to enter step path
            use_step = seqlen_offset > 0 and _is_one(x.shape[0]) and (
                input_row_offsets is None or _is_one(input_row_offsets.shape[0] - 1)
            )
        
        # Infer batch and seqlen early for step method
        # For now, simplify and use batch_size=1 to avoid ragged tensor complexity
        batch_size = 1
        seqlen = x.shape[0]
        
        # If using step method, call it directly with original hidden_states
        if use_step:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch_size)
            
            # Reshape hidden_states for step: (batch * seqlen, hidden_size) -> (batch, seqlen, hidden_size)
            # Step expects (batch, 1, hidden_size) for single token
            # Reference asserts seqlen == 1, so we reshape accordingly
            hidden_states_step = ops.reshape(
                x,
                shape=[batch_size, seqlen, self.hidden_size],
            )
            
            # For seqlen=1, reshape to (batch, 1, hidden_size)
            # For seqlen>1, we'd need to loop, but typically step is called with seqlen=1
            # Reshape to (batch, 1, hidden_size) - step method will handle both shapes
            if hasattr(seqlen, '__int__'):
                try:
                    if int(seqlen) == 1:
                        hidden_states_step = ops.reshape(hidden_states_step, shape=[batch_size, 1, self.hidden_size])
                except (TypeError, ValueError):
                    # Symbolic dimension, assume seqlen=1 and reshape
                    hidden_states_step = ops.reshape(hidden_states_step, shape=[batch_size, 1, self.hidden_size])
            else:
                # Assume seqlen=1 for autoregressive mode
                hidden_states_step = ops.reshape(hidden_states_step, shape=[batch_size, 1, self.hidden_size])
            
            # Call step method
            out_step, conv_state, ssm_state = self.step(
                hidden_states_step,
                conv_state,
                ssm_state,
            )
            
            # Update cache
            if "key_value_memory_dict" in inference_params:
                inference_params["key_value_memory_dict"][self.layer_idx] = (conv_state, ssm_state)
            
            # Reshape output: (batch, 1, hidden_size) -> (batch, hidden_size)
            ss_output = ops.reshape(out_step, shape=[batch_size, self.hidden_size])
            return ss_output
        
        # Step 2: Input projection and gating (for prefill mode)
        # Project input - matches reference: "We do matmul and transpose BLH -> HBL at the same time"
        # In MAX, we work with flattened tensors, so we just do the projection
        xz = self.in_proj(x)  # (batch * seqlen, intermediate_size * 2)
        
        # Split into x and z (gate and up in reference terminology)
        # Reference splits as: x, z = xz.chunk(2, dim=1)
        x, z = ops.split(
            xz,
            [self.intermediate_size, self.intermediate_size],
            axis=-1,
        )
        
        # Step 3: Infer batch and seqlen for reshaping
        # For now, simplify and use batch_size=1 to avoid ragged tensor complexity
        # TODO: Implement proper ragged tensor support
        # Force graph rebuild: updated 2025-01-04
        batch_size = 1
        seqlen = x.shape[0]

        x_conv_input = ops.reshape(
            x,
            shape=[batch_size, self.intermediate_size, seqlen],
        )

        z_conv = ops.reshape(
            z,
            shape=[batch_size, self.intermediate_size, seqlen],
        )

        # Step 4: Prefill selective scan (full sequence)
        # Apply conv1d to x
        x_conv = causal_conv1d_fn(
            x_conv_input,
            self.conv1d_weight,
            bias=self.conv1d_bias,
            algorithm="optimized",
            activation="silu",
        )  # (batch, intermediate_size, seqlen)

        # Reshape back to flattened format for projections
        x_flat = ops.reshape(
            x_conv,
            shape=[batch_size * seqlen, self.intermediate_size],
        )

        # Step 5: Project to get B, C, delta
        x_proj = self.x_proj(x_flat)  # (batch * seqlen, dt_rank + 2 * n_groups * d_state)

        # Split x_proj into delta, B, C
        bc_dim = getattr(self, 'bc_dim', 2 * self.n_groups * self.d_state)
        dt, BC = ops.split(
            x_proj,
            [self.dt_rank, bc_dim],
            axis=-1,
        )

        # Project delta
        dt_proj = self.dt_proj(dt)  # (batch * seqlen, intermediate_size)

        # Split BC into B and C
        B, C = ops.split(
            BC,
            [self.n_groups * self.d_state, self.n_groups * self.d_state],
            axis=-1,
        )

        # Step 6: Reshape for selective scan forward
        u = ops.reshape(x_flat, shape=[batch_size, self.intermediate_size, seqlen])
        delta = ops.reshape(dt_proj, shape=[batch_size, self.intermediate_size, seqlen])
        B_reshaped = ops.reshape(
            B,
            shape=[batch_size, self.n_groups, self.d_state, seqlen],
        )
        C_reshaped = ops.reshape(
            C,
            shape=[batch_size, self.n_groups, self.d_state, seqlen],
        )

        # Prepare A and D
        A_log_cast = self.A_log.cast(self.dtype)
        A = ops.negate(ops.exp(A_log_cast))  # (intermediate_size, d_state)
        D = self.D.cast(self.dtype)

        # delta_bias
        if hasattr(self.dt_proj, 'bias') and self.dt_proj.bias is not None:
            delta_bias = self.dt_proj.bias.cast(self.dtype)
        else:
            delta_bias = ops.constant(0.0, dtype=self.dtype, device=x.device)
            delta_bias = ops.broadcast_to(delta_bias, shape=[self.intermediate_size])

        # selective_scan_fwd outputs
        chunk_size = 2048
        num_chunks = (seqlen + Dim(chunk_size) - Dim(1)) // Dim(chunk_size)

        output_type = TensorType(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size, seqlen],
            device=x.device,
        )
        x_checkpoint_type = TensorType(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size, num_chunks, 2 * self.d_state],
            device=x.device,
        )
        out_z_type = TensorType(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size, seqlen],
            device=x.device,
        )

        results = ops.custom(
            "selective_scan_fwd",
            device=x.device,
            values=[u, delta, A, B_reshaped, C_reshaped, D, z_conv, delta_bias],
            out_types=[output_type, x_checkpoint_type, out_z_type],
            parameters={"delta_softplus": self.delta_softplus},
        )

        ss_output = results[0].tensor  # (batch, intermediate_size, seqlen)

        # Flatten and project out
        ss_output_flat = ops.reshape(
            ss_output,
            shape=[batch_size * seqlen, self.intermediate_size],
        )
        return self.out_proj(ss_output_flat)

        # Step 10: Reshape back to flattened format
        # (batch, intermediate_size, seqlen) -> (batch * seqlen, intermediate_size)
        # With batch_size=1, this is just (seqlen, intermediate_size)
        ss_output_flat = ops.reshape(
            ss_output,
            shape=[batch_size * seqlen, self.intermediate_size],
        )

        # Step 11: Output projection
        # Project back to hidden_size
        return self.out_proj(ss_output_flat)
        ss_output = ops.reshape(
            ss_output,
            shape=[batch_size * seqlen, self.intermediate_size],
        )
        
        # Step 11: Output projection
        # Project back to hidden_size
        return self.out_proj(ss_output)
    
    def step(
        self,
        hidden_states: TensorValue,
        conv_state: TensorValue,
        ssm_state: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Single step forward pass for autoregressive generation.
        
        Matches reference implementation:
        https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        
        Args:
            hidden_states: Input hidden states of shape (batch, 1, hidden_size) or (batch, hidden_size).
            conv_state: Convolution state buffer (batch, intermediate_size, width-1).
            ssm_state: SSM state buffer (batch, intermediate_size, d_state).
            
        Returns:
            Tuple of (output, conv_state, ssm_state) where:
            - output: Output tensor of shape (batch, 1, hidden_size).
            - conv_state: Updated convolution state.
            - ssm_state: Updated SSM state.
        """
        # Reference: assert hidden_states.shape[1] == 1
        # Handle both (batch, 1, hidden_size) and (batch, hidden_size) shapes
        if len(hidden_states.shape.static_dims) == 3:
            # (batch, 1, hidden_size) -> (batch, hidden_size)
            batch_size = hidden_states.shape[0]
            hidden_states = ops.reshape(hidden_states, shape=[batch_size, self.hidden_size])
        else:
            # (batch, hidden_size)
            batch_size = hidden_states.shape[0]
        
        # Project input: xz = self.in_proj(hidden_states.squeeze(1))
        xz = self.in_proj(hidden_states)  # (batch, intermediate_size * 2)
        
        # Split into x and z
        x, z = ops.split(
            xz,
            [self.intermediate_size, self.intermediate_size],
            axis=-1,
        )  # (batch, intermediate_size) each
        
        # Conv step
        # Reshape x for conv1d: (batch, intermediate_size) -> (batch, intermediate_size, 1)
        x_conv = ops.reshape(x, shape=[batch_size, self.intermediate_size, 1])
        
        # Use causal_conv1d_update_fn
        x_conv = causal_conv1d_update_fn(
            x_conv,
            conv_state,
            self.conv1d_weight,
            bias=self.conv1d_bias,
            activation="silu",
        )  # (batch, intermediate_size, 1)
        
        # Reshape back: (batch, intermediate_size, 1) -> (batch, intermediate_size)
        x = ops.reshape(x_conv, shape=[batch_size, self.intermediate_size])
        
        # Project to get dt, B, C
        x_db = self.x_proj(x)  # (batch, dt_rank + 2 * d_state)
        
        # Split into dt, B, C
        bc_dim = getattr(self, 'bc_dim', 2 * self.n_groups * self.d_state)
        dt, BC = ops.split(
            x_db,
            [self.dt_rank, bc_dim],
            axis=-1,
        )
        
        # Project delta (don't add dt_bias here, kernel handles it)
        dt = self.dt_proj(dt)  # (batch, intermediate_size)
        
        # Split BC into B and C
        B, C = ops.split(
            BC,
            [self.n_groups * self.d_state, self.n_groups * self.d_state],
            axis=-1,
        )
        
        # Reshape B and C for update kernel: (batch, n_groups, d_state)
        B_grouped = ops.reshape(B, shape=[batch_size, self.n_groups, self.d_state])
        C_grouped = ops.reshape(C, shape=[batch_size, self.n_groups, self.d_state])
        
        # Prepare A and D (same as forward)
        A_log_cast = self.A_log.cast(self.dtype)
        A = ops.negate(ops.exp(A_log_cast))  # (intermediate_size, d_state)
        D = self.D.cast(self.dtype)

        # Get delta_bias
        if hasattr(self.dt_proj, 'bias') and self.dt_proj.bias is not None:
            delta_bias = self.dt_proj.bias.cast(self.dtype)
        else:
            delta_bias = ops.constant(0.0, dtype=self.dtype, device=hidden_states.device)
            delta_bias = ops.broadcast_to(delta_bias, shape=[self.intermediate_size])
        
        # SSM step: call selective_scan_update
        state_out_type = TensorType(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size, self.d_state],
            device=hidden_states.device,
        )
        output_type = TensorType(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size],
            device=hidden_states.device,
        )
        
        results = ops.custom(
            "selective_scan_update",
            device=hidden_states.device,
            values=[ssm_state, x, dt, A, B_grouped, C_grouped, D, z, delta_bias],
            out_types=[state_out_type, output_type],
            parameters={"delta_softplus": self.delta_softplus},
        )
        
        # Update states
        ssm_state = results[0].tensor  # Updated SSM state
        y = results[1].tensor  # (batch, intermediate_size)
        
        # Output projection
        out = self.out_proj(y)  # (batch, hidden_size)
        
        # Reshape to (batch, 1, hidden_size) for consistency
        out = ops.reshape(out, shape=[batch_size, 1, self.hidden_size])
        
        return out, conv_state, ssm_state
    
    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: DType | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Allocate inference cache buffers for autoregressive generation.
        
        Matches reference implementation:
        https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        
        Args:
            batch_size: Batch size for the cache.
            max_seqlen: Maximum sequence length (not used for Mamba, but kept for compatibility).
            dtype: Data type for cache (uses conv1d weight dtype if None).
            
        Returns:
            Tuple of (conv_state, ssm_state) buffers.
        """
        conv_dtype = self.conv1d_weight.dtype if dtype is None else dtype
        conv_state = ops.constant(0.0, dtype=conv_dtype, device=self.device)
        conv_state = ops.broadcast_to(
            conv_state,
            shape=[batch_size, self.intermediate_size, self.conv_width - 1],
        )
        
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = ops.constant(0.0, dtype=ssm_dtype, device=self.device)
        ssm_state = ops.broadcast_to(
            ssm_state,
            shape=[batch_size, self.intermediate_size, self.d_state],
        )
        
        return conv_state, ssm_state
    
    def _get_states_from_cache(
        self,
        inference_params: dict,
        batch_size: int,
        initialize_states: bool = False,
    ) -> tuple[TensorValue, TensorValue]:
        """Get or create state buffers from inference cache.
        
        Matches reference implementation:
        https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        
        Args:
            inference_params: Inference parameters dict containing key_value_memory_dict.
            batch_size: Batch size for the states.
            initialize_states: Whether to initialize states to zero.
            
        Returns:
            Tuple of (conv_state, ssm_state) buffers.
        """
        assert self.layer_idx is not None, "layer_idx must be set for inference caching"
        
        if "key_value_memory_dict" not in inference_params:
            inference_params["key_value_memory_dict"] = {}
        
        cache_dict = inference_params["key_value_memory_dict"]
        
        if self.layer_idx not in cache_dict:
            # Allocate new states
            conv_state, ssm_state = self.allocate_inference_cache(
                batch_size,
                max_seqlen=1,  # Not used but required by signature
            )
            cache_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = cache_dict[self.layer_idx]
            if initialize_states:
                # Zero out states
                conv_state = ops.constant(0.0, dtype=conv_state.dtype, device=self.device)
                conv_state = ops.broadcast_to(conv_state, shape=conv_state.shape)
                ssm_state = ops.constant(0.0, dtype=ssm_state.dtype, device=self.device)
                ssm_state = ops.broadcast_to(ssm_state, shape=ssm_state.shape)
                cache_dict[self.layer_idx] = (conv_state, ssm_state)
        
        return conv_state, ssm_state


class Block(Module):
    """Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.
    
    This Block has a slightly different structure compared to a regular
    prenorm Transformer block.
    The standard block is: LN -> MHA/MLP -> Add.
    Here we have: Add -> LN -> Mixer, returning both
    the hidden_states (output of the mixer) and the residual.
    This is purely for performance reasons, as we can fuse add and LayerNorm.
    The residual needs to be provided (except for the very first block).
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py
    """

    def __init__(
        self,
        dim: int,
        mixer: Module,
        mlp: Layer | None = None,
        norm: Layer | None = None,
        norm2: Layer | None = None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ) -> None:
        """Initialize a Mamba block.
        
        Args:
            dim: Hidden dimension size.
            mixer: The mixer class (SSM layer, replaces attention in transformers).
            mlp: Optional MLP layer. If None, MLP is skipped.
            norm: Normalization layer (typically RMSNorm or LayerNorm) for the mixer path.
            norm2: Optional second normalization layer for the MLP path.
            fused_add_norm: Whether to use fused add-norm operations via layer_norm_fn.
            residual_in_fp32: Whether to keep residual in float32 precision.
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm
        self.mixer = mixer
        if mlp is not None:
            self.norm2 = norm2
            self.mlp = mlp
        else:
            self.mlp = None
            self.norm2 = None
        
        if self.fused_add_norm:
            # Verify that layer_norm_fn is available
            if layer_norm_fn is None:
                raise ValueError("layer_norm_fn import fails")
            # Only LayerNorm and RMSNorm are supported for fused_add_norm
            if self.norm is not None and not isinstance(self.norm, (LayerNorm, RMSNorm)):
                raise ValueError(
                    "Only LayerNorm and RMSNorm are supported for fused_add_norm, "
                    f"got {type(self.norm)}"
                )
            if self.norm2 is not None and not isinstance(self.norm2, (LayerNorm, RMSNorm)):
                raise ValueError(
                    "Only LayerNorm and RMSNorm are supported for fused_add_norm, "
                    f"got {type(self.norm2)}"
                )

    def __call__(
        self,
        hidden_states: TensorValue,
        residual: Optional[TensorValue] = None,
        input_row_offsets: TensorValue | None = None,
        inference_params: Optional[dict] = None,
        **mixer_kwargs,
    ) -> tuple[TensorValue, TensorValue]:
        """Forward pass through the block.
        
        Args:
            hidden_states: The sequence to the block (required).
            residual: Optional residual tensor. If None, uses hidden_states as residual.
            input_row_offsets: Row offsets for ragged tensor processing.
            inference_params: Optional inference parameters for caching.
            **mixer_kwargs: Additional keyword arguments for the mixer.
            
        Returns:
            Tuple of (hidden_states, residual) where:
            - hidden_states: Output hidden states after mixer (and optionally MLP)
            - residual: Updated residual tensor
        """
        if not self.fused_add_norm:
            # Non-fused path: add residual first, then normalize
            if self.norm is not None:
                if residual is not None:
                    # Use rebind to assert shapes are equivalent at runtime
                    residual = hidden_states.rebind(residual.shape, message="Asserting hidden_states and residual have equivalent shapes for addition") + residual
                else:
                    residual = hidden_states
                # Cast to norm dtype for normalization
                norm_dtype = self.norm.weight.dtype if hasattr(self.norm, 'weight') else residual.dtype
                hidden_states = self.norm(residual.cast(norm_dtype))
                if self.residual_in_fp32:
                    residual = residual.cast(DType.float32)
            else:
                # No norm, just use hidden_states as residual
                residual = (hidden_states + residual) if residual is not None else hidden_states
        else:
            # Fused path: use layer_norm_fn for fused add-norm operation
            if self.norm is not None:
                # Get weight and bias, handling device placement
                norm_weight = self.norm.weight.cast(hidden_states.dtype)
                if hidden_states.device:
                    norm_weight = norm_weight.to(hidden_states.device)
                
                # Get bias (can be None for RMSNorm)
                norm_bias = None
                if isinstance(self.norm, LayerNorm) and self.norm.bias is not None:
                    norm_bias = self.norm.bias.cast(hidden_states.dtype)
                    if hidden_states.device:
                        norm_bias = norm_bias.to(hidden_states.device)
                
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    norm_weight,
                    bias=norm_bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                    is_rms_norm=isinstance(self.norm, RMSNorm),
                )
            else:
                # No norm, just use hidden_states as residual
                residual = (hidden_states + residual) if residual is not None else hidden_states
        
        # Apply mixer (SSM)
        if input_row_offsets is not None:
            hidden_states = self.mixer(
                hidden_states,
                input_row_offsets=input_row_offsets,
                inference_params=inference_params,
                **mixer_kwargs,
            )
        else:
            hidden_states = self.mixer(
                hidden_states,
                inference_params=inference_params,
                **mixer_kwargs,
            )

        # Apply MLP if present
        if self.mlp is not None:
            if not self.fused_add_norm:
                # Non-fused path: add residual first, then normalize
                if self.norm2 is not None:
                    # Use rebind to assert shapes are equivalent at runtime
                    residual = hidden_states.rebind(residual.shape, message="Asserting hidden_states and residual have equivalent shapes for MLP residual addition") + residual
                    # Cast to norm dtype for normalization
                    norm2_dtype = self.norm2.weight.dtype if hasattr(self.norm2, 'weight') else residual.dtype
                    hidden_states = self.norm2(residual.to(dtype=norm2_dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(dtype=DType.float32)
                else:
                    # No norm2, just add residual
                    # Use rebind to assert shapes are equivalent at runtime
                    residual = hidden_states.rebind(residual.shape, message="Asserting hidden_states and residual have equivalent shapes for MLP residual addition (no norm2)") + residual
            else:
                # Fused path: use layer_norm_fn for fused add-norm operation
                if self.norm2 is not None:
                    # Get weight and bias, handling device placement
                    norm2_weight = self.norm2.weight.cast(hidden_states.dtype)
                    if hidden_states.device:
                        norm2_weight = norm2_weight.to(hidden_states.device)
                    
                    # Get bias (can be None for RMSNorm)
                    norm2_bias = None
                    if isinstance(self.norm2, LayerNorm) and self.norm2.bias is not None:
                        norm2_bias = self.norm2.bias.cast(hidden_states.dtype)
                        if hidden_states.device:
                            norm2_bias = norm2_bias.to(hidden_states.device)
                    
                    hidden_states, residual = layer_norm_fn(
                        hidden_states,
                        norm2_weight,
                        bias=norm2_bias,
                        residual=residual,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        eps=self.norm2.eps,
                        is_rms_norm=isinstance(self.norm2, RMSNorm),
                    )
                else:
                    # No norm2, just add residual
                    # Use rebind to assert shapes are equivalent at runtime
                    residual = hidden_states.rebind(residual.shape, message="Asserting hidden_states and residual have equivalent shapes for MLP residual addition (fused, no norm2)") + residual
            
            # Apply MLP
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

