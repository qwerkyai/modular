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

from math import exp
from sys.info import simd_width_of

from algorithm.functional import _get_start_indices_of_nth_subvolume
from gpu.host import DeviceContext
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from layout._fillers import rand
from layout.int_tuple import fill_like
from memory import alloc
from nn.causal_conv1d import (
    naive_causal_conv1d_channel_first_fwd_cpu,
    naive_causal_conv1d_channel_first_fwd_gpu,
    causal_conv1d_channel_first_fwd_gpu,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


@always_inline
fn silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[DType.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Scalar[DType.float32](1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


fn run_causal_conv1d_gpu[
    dtype: DType,
    algorithm: StaticString,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    rtol: Float64 = 0.01,
    ctx: DeviceContext,
) raises:
    """Test causal conv1d GPU kernel against CPU reference."""
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    
    var input_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var input_h = LayoutTensor[dtype, layout_3d](
        input_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var weight_heap = alloc[Scalar[dtype]](dim * width)
    var weight_h = LayoutTensor[dtype, layout_2d](
        weight_heap, RuntimeLayout[layout_2d].row_major(Index(dim, width))
    )
    var bias_heap = alloc[Scalar[dtype]](dim)
    var bias_h = LayoutTensor[dtype, layout_1d](
        bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    var result_gpu_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var result_gpu_h = LayoutTensor[dtype, layout_3d](
        result_gpu_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    ).fill(0)
    var result_cpu_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var result_cpu_h = LayoutTensor[dtype, layout_3d](
        result_cpu_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    ).fill(0)
    
    # Initialize input data
    rand(input_h)
    rand(weight_h)
    rand(bias_h)
    
    var input_buf = input_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var result_gpu_buf = result_gpu_h
    var result_cpu_buf = result_cpu_h
    
    # Strides for channel-first layout (B, C, L)
    var x_batch_stride: UInt32 = dim * seqlen
    var x_c_stride: UInt32 = seqlen
    var x_l_stride: UInt32 = 1
    var weight_c_stride: UInt32 = width
    var weight_width_stride: UInt32 = 1
    var out_batch_stride: UInt32 = dim * seqlen
    var out_c_stride: UInt32 = seqlen
    var out_l_stride: UInt32 = 1
    
    var silu_activation = activation == "silu"
    
    # Run CPU reference
    naive_causal_conv1d_channel_first_fwd_cpu[
        input_buf.dtype,
        input_buf.layout,
        weight_buf.dtype,
        weight_buf.layout,
        result_cpu_buf.dtype,
        result_cpu_buf.layout,
        bias_buf.dtype,
        bias_buf.layout,
    ](
        batch,
        dim,
        seqlen,
        width,
        input_buf,
        weight_buf,
        result_cpu_buf,
        bias_buf,
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
    
    # Run GPU kernel
    @parameter
    if algorithm == "naive":
        comptime BX = 32
        var compiled_func = ctx.compile_function_checked[
            naive_causal_conv1d_channel_first_fwd_gpu[
                input_buf.dtype,
                input_buf.layout,
                weight_buf.dtype,
                weight_buf.layout,
                result_gpu_buf.dtype,
                result_gpu_buf.layout,
                BX,
                bias_buf.dtype,
                bias_buf.layout,
            ],
            naive_causal_conv1d_channel_first_fwd_gpu[
                input_buf.dtype,
                input_buf.layout,
                weight_buf.dtype,
                weight_buf.layout,
                result_gpu_buf.dtype,
                result_gpu_buf.layout,
                BX,
                bias_buf.dtype,
                bias_buf.layout,
            ],
        ]()
        var silu_activation_int8 = Int8(silu_activation)
        ctx.enqueue_function_checked(
            compiled_func,
            batch,
            dim,
            seqlen,
            width,
            input_buf,
            weight_buf,
            result_gpu_buf,
            bias_buf,
            x_batch_stride,
            x_c_stride,
            x_l_stride,
            weight_c_stride,
            weight_width_stride,
            out_batch_stride,
            out_c_stride,
            out_l_stride,
            silu_activation_int8,
            grid_dim=(batch, ceildiv(dim, BX), ceildiv(seqlen, BX)),
            block_dim=(BX),
        )
    elif algorithm == "optimized":
        comptime kNThreads = 128
        comptime kNElts = 4
        # Create dummy xx2D tensor for optimized kernel
        comptime layout_2d_xx = Layout.row_major[2]()
        var xx2D_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
        var xx2D_h = LayoutTensor[dtype, layout_2d_xx](
            xx2D_heap, RuntimeLayout[layout_2d_xx].row_major(Index(batch * dim, seqlen))
        )
        var xx2D_buf = xx2D_h
        
        if width == 1:
            comptime kWidth = 1
            var compiled_func_opt = ctx.compile_function_checked[
                causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    xx2D_buf.layout,
                    kNThreads,
                    kWidth,
                    kNElts,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
                causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    xx2D_buf.layout,
                    kNThreads,
                    kWidth,
                    kNElts,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
            ]()
            var silu_activation_int8_opt = Int8(silu_activation)
            ctx.enqueue_function_checked(
                compiled_func_opt,
                batch,
                dim,
                seqlen,
                width,
                input_buf,
                weight_buf,
                result_gpu_buf,
                bias_buf,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation_int8_opt,
                grid_dim=(ceildiv(input_buf.dim(2), kNThreads * kNElts), input_buf.dim(1), input_buf.dim(0)),
                block_dim=(kNThreads),
            )
        elif width == 2:
            comptime kWidth = 2
            var compiled_func_opt = ctx.compile_function_checked[
                causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    xx2D_buf.layout,
                    kNThreads,
                    kWidth,
                    kNElts,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
                causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    xx2D_buf.layout,
                    kNThreads,
                    kWidth,
                    kNElts,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
            ]()
            var silu_activation_int8_opt = Int8(silu_activation)
            ctx.enqueue_function_checked(
                compiled_func_opt,
                batch,
                dim,
                seqlen,
                width,
                input_buf,
                weight_buf,
                result_gpu_buf,
                bias_buf,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation_int8_opt,
                grid_dim=(ceildiv(input_buf.dim(2), kNThreads * kNElts), input_buf.dim(1), input_buf.dim(0)),
                block_dim=(kNThreads),
            )
        elif width == 3:
            comptime kWidth = 3
            var compiled_func_opt = ctx.compile_function_checked[
                causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    xx2D_buf.layout,
                    kNThreads,
                    kWidth,
                    kNElts,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
                causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    xx2D_buf.layout,
                    kNThreads,
                    kWidth,
                    kNElts,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
            ]()
            var silu_activation_int8_opt = Int8(silu_activation)
            ctx.enqueue_function_checked(
                compiled_func_opt,
                batch,
                dim,
                seqlen,
                width,
                input_buf,
                weight_buf,
                result_gpu_buf,
                bias_buf,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation_int8_opt,
                grid_dim=(ceildiv(input_buf.dim(2), kNThreads * kNElts), input_buf.dim(1), input_buf.dim(0)),
                block_dim=(kNThreads),
            )
        elif width == 4:
            comptime kWidth = 4
            var compiled_func_opt = ctx.compile_function_checked[
                causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    xx2D_buf.layout,
                    kNThreads,
                    kWidth,
                    kNElts,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
                causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    xx2D_buf.layout,
                    kNThreads,
                    kWidth,
                    kNElts,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
            ]()
            var silu_activation_int8_opt = Int8(silu_activation)
            ctx.enqueue_function_checked(
                compiled_func_opt,
                batch,
                dim,
                seqlen,
                width,
                input_buf,
                weight_buf,
                result_gpu_buf,
                bias_buf,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation_int8_opt,
                grid_dim=(ceildiv(input_buf.dim(2), kNThreads * kNElts), input_buf.dim(1), input_buf.dim(0)),
                block_dim=(kNThreads),
            )
        else:
            # Fall back to naive for unsupported widths
            comptime BX = 32
            var compiled_func_fallback = ctx.compile_function_checked[
                naive_causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    BX,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
                naive_causal_conv1d_channel_first_fwd_gpu[
                    input_buf.dtype,
                    input_buf.layout,
                    weight_buf.dtype,
                    weight_buf.layout,
                    result_gpu_buf.dtype,
                    result_gpu_buf.layout,
                    BX,
                    bias_buf.dtype,
                    bias_buf.layout,
                ],
            ]()
            var silu_activation_int8_fallback = Int8(silu_activation)
            ctx.enqueue_function_checked(
                compiled_func_fallback,
                batch,
                dim,
                seqlen,
                width,
                input_buf,
                weight_buf,
                result_gpu_buf,
                bias_buf,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation_int8_fallback,
                grid_dim=(batch, ceildiv(dim, BX), ceildiv(seqlen, BX)),
                block_dim=(BX),
            )
        xx2D_heap.free()
    
    # Synchronize GPU
    ctx.sync()
    
    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            result_gpu_h.ptr[i],
            result_cpu_h.ptr[i],
            rtol=rtol,
        )
    
    # Cleanup
    input_heap.free()
    weight_heap.free()
    bias_heap.free()
    result_gpu_heap.free()
    result_cpu_heap.free()


def main():
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        print("GPU not available, skipping GPU tests")
        return
    
    # Test basic cases
    run_causal_conv1d_gpu[DType.float32, "naive", "none"](2, 4, 8, 3, ctx=ctx)
    print("✓ Basic GPU causal conv1d test passed")
    
    run_causal_conv1d_gpu[DType.float32, "naive", "silu"](2, 4, 8, 3, ctx=ctx)
    print("✓ GPU causal conv1d with SiLU test passed")
    
    # Test optimized algorithm with supported widths
    run_causal_conv1d_gpu[DType.float32, "optimized", "none"](2, 8, 16, 1, ctx=ctx)
    print("✓ Optimized GPU causal conv1d width 1 test passed")
    
    run_causal_conv1d_gpu[DType.float32, "optimized", "none"](2, 8, 16, 2, ctx=ctx)
    print("✓ Optimized GPU causal conv1d width 2 test passed")
    
    run_causal_conv1d_gpu[DType.float32, "optimized", "none"](2, 8, 16, 3, ctx=ctx)
    print("✓ Optimized GPU causal conv1d width 3 test passed")
    
    run_causal_conv1d_gpu[DType.float32, "optimized", "none"](2, 8, 16, 4, ctx=ctx)
    print("✓ Optimized GPU causal conv1d width 4 test passed")
    
    # Test larger sequences
    run_causal_conv1d_gpu[DType.float32, "naive", "none"](2, 16, 128, 3, ctx=ctx)
    print("✓ Large sequence GPU test passed")

