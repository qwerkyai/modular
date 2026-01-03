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
"""Mamba-specific kernel registrations.

This package contains only the custom operations required for Mamba models:
- causal_conv1d: Causal 1D convolution with optional SiLU activation
- selective_scan_fwd: Selective scan forward pass for Mamba SSM

These operations are separated from MOGGKernelAPI to avoid conflicts with
default kernel registrations.
"""

from math import ceildiv

import compiler_internal as compiler
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from runtime.asyncrt import DeviceContextPtr

from nn.causal_conv1d import (
    causal_conv1d_channel_first_fwd_cpu,
    causal_conv1d_channel_first_fwd_gpu,
    causal_conv1d_update_cpu,
    causal_conv1d_update_gpu,
)
from nn.selective_scan import (
    selective_scan_fwd_cpu,
    selective_scan_fwd_gpu,
    selective_scan_update_cpu,
    selective_scan_update_gpu,
)

from utils.index import IndexList
from tensor import InputTensor, OutputTensor
from tensor.managed_tensor_slice import _FusedInputTensor as FusedInputTensor


# ============================================================================
# Causal Conv1D Registration
# ============================================================================

@compiler.register("causal_conv1d")
struct CausalConv1D[activation: StaticString]:
    """Causal 1D convolution operation with bias.
    
    Performs causal (autoregressive) 1D convolution where each output position
    depends only on current and past input positions. Supports optional SiLU
    activation with SIMD-vectorized implementations for widths 1, 2, 3, 4.
    
    Parameters:
        activation: Activation function to apply after convolution.
            - "none": No activation (identity).
            - "silu": SiLU/Swish activation (x * sigmoid(x)).
    
    Tensor Shapes:
        - input: (batch, channels, seqlen) - Input sequence tensor.
        - weight: (channels, width) - Convolution weights per channel.
        - bias: (channels,) - Per-channel bias to add.
        - output: (batch, channels, seqlen) - Output tensor (same shape as input).
    """
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: FusedInputTensor[dtype=dtype, rank=rank],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if input.rank != 3:
            raise Error("Input tensor must be rank 3 (batch, channels, seqlen)")
        if weight.rank != 2:
            raise Error("Weight tensor must be rank 2 (channels, width)")
        if bias.rank != 1:
            raise Error("Bias tensor must be rank 1 (channels,)")
        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")

        var X = input.to_layout_tensor()
        var W = weight.to_layout_tensor()
        var O = output.to_layout_tensor()
        var B = bias.to_layout_tensor()

        var batch_size: Int = input.dim_size(0)
        var dim: Int = input.dim_size(1)
        var seqlen: Int = input.dim_size(2)
        var width: Int = weight.dim_size(1)
        
        var x_batch_stride: UInt32 = input.strides()[0]
        var x_c_stride: UInt32 = input.strides()[1]
        var x_l_stride: UInt32 = input.strides()[2]
        
        var weight_c_stride: UInt32 = weight.strides()[0]
        var weight_width_stride: UInt32 = weight.strides()[1]
        
        var out_batch_stride: UInt32 = output.strides()[0]
        var out_c_stride: UInt32 = output.strides()[1]
        var out_l_stride: UInt32 = output.strides()[2]
        
        var silu_activation = Self.activation == "silu"
        
        @parameter
        if is_cpu[target]():
            causal_conv1d_channel_first_fwd_cpu[
                X.dtype,
                X.layout,
                W.dtype,
                W.layout,
                O.dtype,
                O.layout,
                B.dtype,
                B.layout,
            ](
                batch_size,
                dim,
                seqlen,
                width,
                X,
                W,
                O,
                B,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation,
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx.get_device_context()
            comptime kNThreads = 128
            comptime kNElts = 4
            if width == 1:
                comptime kWidth = 1
                var compiled_func = gpu_ctx.compile_function_checked[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function_checked(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    silu_activation_int8,
                    grid_dim=(ceildiv(X.dim(2), kNThreads * kNElts), X.dim(1), X.dim(0)),
                    block_dim=(kNThreads),
                )
            elif width == 2:
                comptime kWidth = 2
                var compiled_func = gpu_ctx.compile_function_checked[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function_checked(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    silu_activation_int8,
                    grid_dim=(ceildiv(X.dim(2), kNThreads * kNElts), X.dim(1), X.dim(0)),
                    block_dim=(kNThreads),
                )
            elif width == 3:
                comptime kWidth = 3
                var compiled_func = gpu_ctx.compile_function_checked[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function_checked(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    silu_activation_int8,
                    grid_dim=(ceildiv(X.dim(2), kNThreads * kNElts), X.dim(1), X.dim(0)),
                    block_dim=(kNThreads),
                )
            elif width == 4:
                comptime kWidth = 4
                var compiled_func = gpu_ctx.compile_function_checked[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function_checked(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    silu_activation_int8,
                    grid_dim=(ceildiv(X.dim(2), kNThreads * kNElts), X.dim(1), X.dim(0)),
                    block_dim=(kNThreads),
                )
            else:
                raise Error("Unsupported kernel width: only widths 1, 2, 3, 4 are supported")
        else:
            raise Error("Unsupported target device")
    
    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
    ) -> IndexList[rank]:
        return input.shape()


# ===----------------------------------------------------------------------=== #
# Selective Scan Forward Operation
# ===----------------------------------------------------------------------=== #

@compiler.register("selective_scan_fwd")
struct SelectiveScanFwd[delta_softplus: Bool = False]:
    """Selective scan forward pass operation for Mamba SSM.
    
    Performs the selective scan computation used in Mamba state space models.
    This is the core operation that processes sequences through the SSM.
    
    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.
    
    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - out_z: (batch, dim, seqlen) - Gated output (if z is provided)
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim, seqlen) - Gating tensor (optional, can be empty)
        - delta_bias: (dim,) - Delta bias (optional, can be empty)
    """
    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        x: OutputTensor[dtype=dtype, rank=4],
        out_z: OutputTensor[dtype=dtype, rank=3],
        u: FusedInputTensor[dtype=dtype, rank=3],
        delta: InputTensor[dtype=dtype, rank=3],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=4],
        C: InputTensor[dtype=dtype, rank=4],
        D: InputTensor[dtype=dtype, rank=1],
        z: InputTensor[dtype=dtype, rank=3],
        delta_bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if output.rank != 3:
            raise Error("Output tensor must be rank 3 (batch, dim, seqlen)")
        if u.rank != 3:
            raise Error("Input tensor u must be rank 3 (batch, dim, seqlen)")
        if delta.rank != 3:
            raise Error("Delta tensor must be rank 3 (batch, dim, seqlen)")
        if A.rank != 2:
            raise Error("A tensor must be rank 2 (dim, dstate)")
        if B.rank != 4:
            raise Error("B tensor must be rank 4 (batch, n_groups, dstate, seqlen)")
        if C.rank != 4:
            raise Error("C tensor must be rank 4 (batch, n_groups, dstate, seqlen)")
        
        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var dstate = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups
        
        var output_lt = output.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var out_z_lt = out_z.to_layout_tensor()
        var u_lt = u.to_layout_tensor()
        var delta_lt = delta.to_layout_tensor()
        var A_lt = A.to_layout_tensor()
        var B_lt = B.to_layout_tensor()
        var C_lt = C.to_layout_tensor()
        var D_lt = D.to_layout_tensor()
        var z_lt = z.to_layout_tensor()
        var delta_bias_lt = delta_bias.to_layout_tensor()
        
        var output_strides = output.strides()
        var x_strides = x.strides()
        var out_z_strides = out_z.strides()
        var u_strides = u.strides()
        var delta_strides = delta.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        var D_strides = D.strides()
        var z_strides = z.strides()
        var delta_bias_strides = delta_bias.strides()
        
        comptime delta_softplus_int8: Int8 = Int8(1) if Self.delta_softplus else Int8(0)
        
        @parameter
        if is_cpu[target]():
            selective_scan_fwd_cpu[
                dtype,
                output_lt.layout,
                x_lt.layout,
                out_z_lt.layout,
                u_lt.layout,
                delta_lt.layout,
                A_lt.layout,
                B_lt.layout,
                C_lt.layout,
                D_lt.layout,
                z_lt.layout,
                delta_bias_lt.layout,
            ](
                batch,
                dim,
                seqlen,
                dstate,
                group_size,
                delta_softplus_int8,
                output_lt,
                x_lt,
                out_z_lt,
                u_lt,
                delta_lt,
                A_lt,
                B_lt,
                C_lt,
                D_lt,
                z_lt,
                delta_bias_lt,
                UInt32(output_strides[0]),
                UInt32(output_strides[1]),
                UInt32(output_strides[2]),
                UInt32(x_strides[0]),
                UInt32(x_strides[1]),
                UInt32(x_strides[2]),
                UInt32(x_strides[3]),
                UInt32(out_z_strides[0]),
                UInt32(out_z_strides[1]),
                UInt32(out_z_strides[2]),
                UInt32(u_strides[0]),
                UInt32(u_strides[1]),
                UInt32(u_strides[2]),
                UInt32(delta_strides[0]),
                UInt32(delta_strides[1]),
                UInt32(delta_strides[2]),
                UInt32(A_strides[0]),
                UInt32(A_strides[1]),
                UInt32(B_strides[0]),
                UInt32(B_strides[1]),
                UInt32(B_strides[2]),
                UInt32(B_strides[3]),
                UInt32(C_strides[0]),
                UInt32(C_strides[1]),
                UInt32(C_strides[2]),
                UInt32(C_strides[3]),
                UInt32(D_strides[0]),
                UInt32(z_strides[0]),
                UInt32(z_strides[1]),
                UInt32(z_strides[2]),
                UInt32(delta_bias_strides[0]),
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)
            
            var compiled_kernel = gpu_ctx.compile_function_checked[
                selective_scan_fwd_gpu[
                    dtype,
                    output_lt.layout,
                    x_lt.layout,
                    out_z_lt.layout,
                    u_lt.layout,
                    delta_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                    D_lt.layout,
                    z_lt.layout,
                    delta_bias_lt.layout,
                ],
                selective_scan_fwd_gpu[
                    dtype,
                    output_lt.layout,
                    x_lt.layout,
                    out_z_lt.layout,
                    u_lt.layout,
                    delta_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                    D_lt.layout,
                    z_lt.layout,
                    delta_bias_lt.layout,
                ]
            ]()
            
            gpu_ctx.enqueue_function_checked(
                compiled_kernel,
                total_batch_dim,
                batch,
                dim,
                seqlen,
                dstate,
                group_size,
                delta_softplus_int8,
                output_lt,
                x_lt,
                out_z_lt,
                u_lt,
                delta_lt,
                A_lt,
                B_lt,
                C_lt,
                D_lt,
                z_lt,
                delta_bias_lt,
                UInt32(output_strides[0]),
                UInt32(output_strides[1]),
                UInt32(output_strides[2]),
                UInt32(x_strides[0]),
                UInt32(x_strides[1]),
                UInt32(x_strides[2]),
                UInt32(x_strides[3]),
                UInt32(out_z_strides[0]),
                UInt32(out_z_strides[1]),
                UInt32(out_z_strides[2]),
                UInt32(u_strides[0]),
                UInt32(u_strides[1]),
                UInt32(u_strides[2]),
                UInt32(delta_strides[0]),
                UInt32(delta_strides[1]),
                UInt32(delta_strides[2]),
                UInt32(A_strides[0]),
                UInt32(A_strides[1]),
                UInt32(B_strides[0]),
                UInt32(B_strides[1]),
                UInt32(B_strides[2]),
                UInt32(B_strides[3]),
                UInt32(C_strides[0]),
                UInt32(C_strides[1]),
                UInt32(C_strides[2]),
                UInt32(C_strides[3]),
                UInt32(D_strides[0]),
                UInt32(z_strides[0]),
                UInt32(z_strides[1]),
                UInt32(z_strides[2]),
                UInt32(delta_bias_strides[0]),
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            raise Error("Unsupported target: " + target)
    
    @staticmethod
    fn shape[
        dtype: DType,
    ](
        u: InputTensor[dtype=dtype, rank=3],
        delta: InputTensor[dtype=dtype, rank=3],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=4],
        C: InputTensor[dtype=dtype, rank=4],
        D: InputTensor[dtype=dtype, rank=1],
        z: InputTensor[dtype=dtype, rank=3],
        delta_bias: InputTensor[dtype=dtype, rank=1],
    ) -> IndexList[3]:
        return u.shape()

