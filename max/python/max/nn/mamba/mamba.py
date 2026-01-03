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
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Layer, Linear, Module
from max.nn.norm import layer_norm_fn
from max.nn.norm.layer_norm import LayerNorm
from max.nn.norm.rms_norm import RMSNorm


class MambaSSM(Module):
    """State Space Model layer for Mamba.
    
    This is a placeholder implementation. The actual SSM computation
    will need to be implemented based on the Mamba paper's specifications.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: DType,
        device: DeviceRef,
        linear_cls: Callable[..., Linear],
    ) -> None:
        """Initialize the Mamba SSM layer.
        
        Args:
            hidden_size: Hidden dimension size.
            intermediate_size: Intermediate dimension for SSM expansion.
            dtype: Data type for weights.
            device: Device to place weights on.
            linear_cls: Linear layer class to use.
        """
        super().__init__()
        # TODO: Implement actual SSM components:
        # - Input projection (in_proj)
        # - State space parameters (A, B, C, D)
        # - Output projection (out_proj)
        # - Gating mechanism
        
        # Placeholder: simple projection for now
        self.in_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=intermediate_size * 2,  # For gating (gate + up)
            dtype=dtype,
            device=device,
        )
        self.out_proj = linear_cls(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        x: TensorValue,
        input_row_offsets: TensorValue,
        inference_params: Optional[dict] = None,
        **kwargs,
    ) -> TensorValue:
        """Forward pass through the SSM layer.
        
        Args:
            x: Input hidden states.
            input_row_offsets: Row offsets for ragged tensor processing.
            inference_params: Optional inference parameters for caching.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Output after SSM transformation.
        """
        # TODO: Implement actual SSM computation
        # For now, using a simple gated projection as placeholder
        proj = self.in_proj(x)
        
        # Split into gate and up projections
        gate, up = ops.split(
            proj, [proj.shape.static_dims[-1] // 2, proj.shape.static_dims[-1] // 2], axis=-1
        )
        
        # Apply SiLU gate
        gated = ops.silu(gate) * up
        
        # Output projection
        return self.out_proj(gated)


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
                hidden_states = self.norm(residual.to(dtype=norm_dtype))
                if self.residual_in_fp32:
                    residual = residual.to(dtype=DType.float32)
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

