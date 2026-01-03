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

from collections.abc import Callable
from typing import Optional

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Layer, Linear, Module
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
        delta_softplus: bool = False,
        conv_kernel: int = 4,
        x_proj_dim: int | None = None,
    ) -> None:
        """Initialize the Mamba SSM layer.
        
        Args:
            hidden_size: Hidden dimension size.
            intermediate_size: Intermediate dimension for SSM expansion.
            dtype: Data type for weights.
            device: Device to place weights on.
            linear_cls: Linear layer class to use.
            d_state: State space dimension (default: 16).
            dt_rank: Rank of delta projection. "auto" sets it to max(16, hidden_size // 16).
            conv_bias: Whether to use bias in conv1d.
            bias: Whether to use bias in linear projections.
            delta_softplus: Whether to apply softplus to delta values.
            conv_kernel: Convolution kernel size (default: 4).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.device = device
        self.d_state = d_state
        self.delta_softplus = delta_softplus
        
        # Determine dt_rank
        if dt_rank == "auto":
            self.dt_rank = max(16, hidden_size // 16)
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
        from max.nn.conv import Conv1D, causal_conv1d_fn
        
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
            pass
        else:
            x_proj_dim = self.dt_rank + 2 * self.n_groups * d_state  # dt_rank + B + C
        
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
        # Step 1: Input projection and gating
        # x shape: (batch * seqlen, hidden_size) or (batch, seqlen, hidden_size)
        # We need to handle both flattened and unflattened cases
        original_shape = x.shape
        needs_reshape = len(original_shape.static_dims) == 2
        
        if needs_reshape:
            # Flattened case: (batch * seqlen, hidden_size)
            # We'll need to infer batch and seqlen from input_row_offsets if available
            # For now, assume we can work with the flattened tensor
            batch_seqlen, hidden_dim = original_shape.static_dims
            # We'll need to reshape for conv1d: (batch, hidden_size, seqlen)
            # But we don't know batch/seqlen split without input_row_offsets
            # For simplicity, assume we receive (batch, seqlen, hidden_size) format
            pass
        
        # Project input
        xz = self.in_proj(x)  # (batch * seqlen, intermediate_size * 2)
        
        # Split into gate and up
        gate, up = ops.split(
            xz,
            [self.intermediate_size, self.intermediate_size],
            axis=-1,
        )
        
        # Apply SiLU gate
        x = ops.silu(gate) * up  # (batch * seqlen, intermediate_size)
        
        # Step 2: Infer batch and seqlen for reshaping
        # We need this for both conv1d and selective scan
        if input_row_offsets is not None:
            # input_row_offsets has shape (batch_size + 1,)
            batch_size = input_row_offsets.shape[0] - 1
            total_seqlen = x.shape[0]
            # Use Dim division for symbolic dimensions
            from max.graph import Dim
            seqlen = Dim(total_seqlen) // Dim(batch_size)
        else:
            # No input_row_offsets: assume single batch
            batch_size = 1
            seqlen = x.shape[0]
        
        # Step 3: Causal conv1d
        # causal_conv1d_fn expects (batch, channels, seqlen) format
        # Reshape x from (batch * seqlen, intermediate_size) to (batch, intermediate_size, seqlen)
        # Use rebind to assert the reshape is valid: total_seqlen == batch_size * seqlen
        from max.graph import Dim
        batch_seqlen = Dim(batch_size) * Dim(seqlen)
        x_rebound = x.rebind([batch_seqlen, self.intermediate_size], 
                             message="Asserting batch_size * seqlen == total_seqlen for reshape")
        x_conv = ops.reshape(
            x_rebound,
            shape=[batch_size, self.intermediate_size, seqlen],
        )
        
        # Apply causal conv1d using the functional API
        # causal_conv1d_fn expects (batch, channels, seqlen) and returns same shape
        from max.nn.conv import causal_conv1d_fn
        
        x_conv = causal_conv1d_fn(
            x_conv,
            self.conv1d_weight,
            bias=self.conv1d_bias,
            algorithm="optimized",
            activation="none",  # No activation, SiLU already applied
        )  # (batch, intermediate_size, seqlen)
        
        # Reshape back to flattened format
        x = ops.reshape(
            x_conv,
            shape=[batch_size * seqlen, self.intermediate_size],
        )
        
        # Step 4: Project to get B, C, delta
        x_proj = self.x_proj(x)  # (batch * seqlen, dt_rank + 2 * n_groups * d_state)
        
        # Split x_proj into delta, B, C
        dt, BC = ops.split(
            x_proj,
            [self.dt_rank, 2 * self.n_groups * self.d_state],
            axis=-1,
        )
        
        # Project delta
        dt = self.dt_proj(dt)  # (batch * seqlen, intermediate_size)
        
        # Split BC into B and C
        B, C = ops.split(
            BC,
            [self.n_groups * self.d_state, self.n_groups * self.d_state],
            axis=-1,
        )
        
        # Step 5: Reshape for selective scan
        # Selective scan expects:
        # - u: (batch, dim, seqlen)
        # - delta: (batch, dim, seqlen)
        # - B: (batch, n_groups, dstate, seqlen)
        # - C: (batch, n_groups, dstate, seqlen)
        # - A: (dim, dstate)
        # - D: (dim,)
        # - z: (batch, dim, seqlen) - optional
        # - delta_bias: (dim,) - optional
        
        # batch_size and seqlen are already computed above in Step 2
        # Reshape tensors for selective scan
        # x: (batch * seqlen, intermediate_size) -> (batch, intermediate_size, seqlen)
        u = ops.reshape(
            x,
            shape=[batch_size, self.intermediate_size, seqlen],
        )
        
        # delta: (batch * seqlen, intermediate_size) -> (batch, intermediate_size, seqlen)
        delta = ops.reshape(
            dt,
            shape=[batch_size, self.intermediate_size, seqlen],
        )
        
        # B: (batch * seqlen, n_groups * d_state) -> (batch, n_groups, d_state, seqlen)
        B = ops.reshape(
            B,
            shape=[batch_size, self.n_groups, self.d_state, seqlen],
        )
        
        # C: (batch * seqlen, n_groups * d_state) -> (batch, n_groups, d_state, seqlen)
        C = ops.reshape(
            C,
            shape=[batch_size, self.n_groups, self.d_state, seqlen],
        )
        
        # Step 6: Prepare A and D parameters
        # A should be (intermediate_size, d_state)
        # A_log is stored in log space, so A = exp(A_log)
        # Get A_log weight and convert to A
        # Weight tensors are automatically handled by the graph, just use them directly
        A = ops.exp(self.A_log)  # (intermediate_size, d_state)
        
        # D should be (intermediate_size,)
        # Get D weight (already a Weight, will be handled by graph)
        D = self.D
        
        # Create empty z tensor (optional gating, not used in basic Mamba)
        # z is used for gating but is typically not needed in the basic implementation
        z_shape = [batch_size, self.intermediate_size, seqlen]
        z = ops.constant(0.0, dtype=self.dtype, device=x.device)
        z = ops.broadcast_to(z, shape=z_shape)
        
        # delta_bias comes from dt_proj.bias, extract it if available
        # For now, create empty delta_bias (dt_proj.bias is handled in dt_proj)
        delta_bias = ops.constant(0.0, dtype=self.dtype, device=x.device)
        delta_bias = ops.broadcast_to(delta_bias, shape=[self.intermediate_size])
        
        # Step 7: Call selective_scan_fwd operation
        # The operation returns: (output, x_checkpoint, out_z)
        # We only need the output
        from max.graph import TensorType as TT
        
        # Compute number of chunks for checkpoint tensor
        # Chunk size is typically 2048 for efficient computation
        chunk_size = 2048
        num_chunks = (seqlen + chunk_size - 1) // chunk_size if seqlen > 0 else 1
        
        # Define output types
        output_type = TT(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size, seqlen],
            device=x.device,
        )
        # x_checkpoint: (batch, dim, num_chunks, 2*dstate)
        x_checkpoint_type = TT(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size, num_chunks, 2 * self.d_state],
            device=x.device,
        )
        out_z_type = TT(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size, seqlen],
            device=x.device,
        )
        
        # Call selective_scan_fwd with delta_softplus parameter
        results = ops.custom(
            "selective_scan_fwd",
            device=x.device,
            values=[u, delta, A, B, C, D, z, delta_bias],
            out_types=[output_type, x_checkpoint_type, out_z_type],
            parameters={"delta_softplus": self.delta_softplus},
        )
        
        # Extract output (first result)
        ss_output = results[0].tensor  # (batch, intermediate_size, seqlen)
        
        # Step 8: Reshape back to flattened format
        # (batch, intermediate_size, seqlen) -> (batch * seqlen, intermediate_size)
        ss_output = ops.reshape(
            ss_output,
            shape=[batch_size * seqlen, self.intermediate_size],
        )
        
        # Step 9: Output projection
        # Project back to hidden_size
        return self.out_proj(ss_output)


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
                residual = (hidden_states + residual) if residual is not None else hidden_states
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
                    residual = hidden_states + residual
                    # Cast to norm dtype for normalization
                    norm2_dtype = self.norm2.weight.dtype if hasattr(self.norm2, 'weight') else residual.dtype
                    hidden_states = self.norm2(residual.to(dtype=norm2_dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(dtype=DType.float32)
                else:
                    # No norm2, just add residual
                    residual = hidden_states + residual
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
                    residual = hidden_states + residual
            
            # Apply MLP
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

