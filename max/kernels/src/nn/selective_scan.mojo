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

# ===----------------------------------------------------------------------=== #
# Core selective scan kernel implementation
# ===----------------------------------------------------------------------=== #
# This file contains the core GPU and CPU kernel implementations for selective scan.
# 
# The kernel functions `selective_scan_fwd_gpu` and `selective_scan_fwd_cpu` are
# the actual kernels that perform the selective scan computation.
#
# For MAX framework integration, see MOGGKernelAPI.mojo.
# ===----------------------------------------------------------------------=== #

from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from utils.index import IndexList
from memory import UnsafePointer
from algorithm import sync_parallelize
import math
from math import ceildiv, exp2

# LOG2E constant for converting exp to exp2 (faster on GPU)
comptime LOG2E = 1.4426950408889634
comptime MAX_DSTATE = 16

# Optimized softplus - matches CUDA reference implementation
# Uses log1pf(expf(x)) for better numerical stability when x <= 20
fn softplus(val: Float32) -> Float32:
    if val > 20.0:
        return val
    # log1p(exp(x)) = log(1 + exp(x)) for better numerical stability
    var exp_val = math.exp(val)
    return math.log(1.0 + exp_val)

# Optimized sigmoid using fast approximation for large negative values
fn sigmoid(val: Float32) -> Float32:
    if val < -20.0:
        return 0.0
    var exp_neg = math.exp(-val)
    return 1.0 / (1.0 + exp_neg)

# Optimized SiLU (swish) activation
fn silu(val: Float32) -> Float32:
    if val < -20.0:
        return 0.0
    var exp_neg = math.exp(-val)
    return val / (1.0 + exp_neg)

# GPU kernel function - must be at module scope for compile_function_checked
fn selective_scan_fwd_gpu[
    kernel_dtype: DType,
    output_layout: Layout,
    x_layout: Layout,
    out_z_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
](
    total_batch_dim: Int,
    batch: Int,
    dim: Int,
    seqlen: Int,
    dstate: Int,
    group_size: Int,
    delta_softplus: Int8,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    out_z: LayoutTensor[kernel_dtype, out_z_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    delta_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    # Strides
    output_b_stride: UInt32,
    output_d_stride: UInt32,
    output_t_stride: UInt32,
    x_b_stride: UInt32,
    x_d_stride: UInt32,
    x_chunk_stride: UInt32,
    x_n_stride: UInt32,
    out_z_b_stride: UInt32,
    out_z_d_stride: UInt32,
    out_z_t_stride: UInt32,
    u_b_stride: UInt32,
    u_d_stride: UInt32,
    u_t_stride: UInt32,
    delta_b_stride: UInt32,
    delta_d_stride: UInt32,
    delta_t_stride: UInt32,
    A_d_stride: UInt32,
    A_n_stride: UInt32,
    B_b_stride: UInt32,
    B_g_stride: UInt32,
    B_n_stride: UInt32,
    B_t_stride: UInt32,
    C_b_stride: UInt32,
    C_g_stride: UInt32,
    C_n_stride: UInt32,
    C_t_stride: UInt32,
    D_stride: UInt32,
    z_b_stride: UInt32,
    z_d_stride: UInt32,
    z_t_stride: UInt32,
    delta_bias_stride: UInt32,
):
    """GPU kernel for selective scan operation.
    
    Each thread processes one (batch, dim) pair and iterates through the sequence.
    """
    # Calculate which (batch, dim) this thread is responsible for
    var thread_id = block_dim.x * block_idx.x + thread_idx.x
    var thread_id_int = Int(thread_id)
    if thread_id_int >= total_batch_dim:
        return
        
    var b = thread_id_int // dim
    var d = thread_id_int % dim
    
    # Additional bounds checking
    if b >= batch or d >= dim:
        return
        
    var group_id = d // group_size
    
    # Local state storage (max dstate 16 to fit in registers)
    # Note: Using large SIMD sizes (e.g. 256) causes register spilling and massive performance loss
    var state = SIMD[DType.float32, MAX_DSTATE](0.0)
    var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
    var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)
    
    # Pre-load A values for this dim and pre-multiply by LOG2E for faster exp2
    # This optimization converts exp(A * delta) to exp2(A * LOG2E * delta)
    # which is faster on GPUs
    var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
    var has_delta_bias = delta_bias.dim(0) > 0
    var delta_bias_val = Float32(0.0)
    if has_delta_bias:
        var bias_offset = UInt32(d) * delta_bias_stride
        delta_bias_val = Scalar[kernel_dtype](delta_bias.ptr.offset(bias_offset).load()).cast[DType.float32]()
    
    var has_D = D.dim(0) > 0
    var D_val = Float32(0.0)
    if has_D:
        var D_offset = UInt32(d) * D_stride
        D_val = Scalar[kernel_dtype](D.ptr.offset(D_offset).load()).cast[DType.float32]()
    
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
    var has_z = z.dim(0) > 0
    var has_out_z = out_z.dim(0) > 0
    
    # Pre-multiply A by LOG2E for exp2 optimization
    for n in range(dstate):
        var A_offset = UInt32(d) * A_d_stride + UInt32(n) * A_n_stride
        A_vals[n] = Scalar[kernel_dtype](A.ptr.offset(A_offset).load()).cast[DType.float32]() * LOG2E

    var chunk_size = 2048
    var t_in_chunk = 0
    var chunk_idx = 0
    
    # Initialize running offsets for strength reduction
    var curr_u_offset = UInt32(b) * u_b_stride + UInt32(d) * u_d_stride
    var curr_delta_offset = UInt32(b) * delta_b_stride + UInt32(d) * delta_d_stride
    var curr_output_offset = UInt32(b) * output_b_stride + UInt32(d) * output_d_stride
    var curr_B_offset = UInt32(b) * B_b_stride + UInt32(group_id) * B_g_stride
    var curr_C_offset = UInt32(b) * C_b_stride + UInt32(group_id) * C_g_stride
    var curr_z_offset = UInt32(b) * z_b_stride + UInt32(d) * z_d_stride
    var curr_out_z_offset = UInt32(b) * out_z_b_stride + UInt32(d) * out_z_d_stride
    
    # Process sequence sequentially for this (batch, dim)
    # OPTIMIZED: Tiled loading with pre-loaded B/C tiles and buffered outputs
    comptime TILE_SIZE = 8  # Sweet spot: larger causes register spilling
    var aligned_seqlen = seqlen - (seqlen % TILE_SIZE)
    var t = 0
    
    # Check if we can use contiguous loads (stride == 1)
    var u_contiguous = u_t_stride == 1
    var delta_contiguous = delta_t_stride == 1
    var z_contiguous = z_t_stride == 1
    var B_contiguous = B_t_stride == 1
    var C_contiguous = C_t_stride == 1
    var output_contiguous = output_t_stride == 1
    
    # Fast path: Tiled loading for the aligned portion
    while t < aligned_seqlen:
        # Load u and delta vectors
        var u_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var delta_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var z_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
        
        if u_contiguous:
            u_vec = u.ptr.offset(curr_u_offset).load[width=TILE_SIZE]()
        else:
            for i in range(TILE_SIZE):
                u_vec[i] = u.ptr.offset(curr_u_offset + UInt32(i) * u_t_stride).load()
                
        if delta_contiguous:
            delta_vec = delta.ptr.offset(curr_delta_offset).load[width=TILE_SIZE]()
        else:
            for i in range(TILE_SIZE):
                delta_vec[i] = delta.ptr.offset(curr_delta_offset + UInt32(i) * delta_t_stride).load()
                
        if has_z:
            if z_contiguous:
                z_vec = z.ptr.offset(curr_z_offset).load[width=TILE_SIZE]()
            else:
                for i in range(TILE_SIZE):
                    z_vec[i] = z.ptr.offset(curr_z_offset + UInt32(i) * z_t_stride).load()
        
        # PRE-LOAD B/C TILES: Load B[n, t:t+TILE] and C[n, t:t+TILE] for all n
        # This avoids redundant address calculations inside the inner loop
        # Using flat arrays: B_tile[n * TILE_SIZE + i] = B[n, t+i]
        var B_tile_0 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_1 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_2 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_3 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_4 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_5 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_6 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_7 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_8 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_9 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_10 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_11 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_12 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_13 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_14 = SIMD[DType.float32, TILE_SIZE](0.0)
        var B_tile_15 = SIMD[DType.float32, TILE_SIZE](0.0)
        
        var C_tile_0 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_1 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_2 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_3 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_4 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_5 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_6 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_7 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_8 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_9 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_10 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_11 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_12 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_13 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_14 = SIMD[DType.float32, TILE_SIZE](0.0)
        var C_tile_15 = SIMD[DType.float32, TILE_SIZE](0.0)
        
        # Load B tiles - vector load if B is contiguous in t
        if B_contiguous:
            if dstate > 0:
                B_tile_0 = B.ptr.offset(curr_B_offset + UInt32(0) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 1:
                B_tile_1 = B.ptr.offset(curr_B_offset + UInt32(1) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 2:
                B_tile_2 = B.ptr.offset(curr_B_offset + UInt32(2) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 3:
                B_tile_3 = B.ptr.offset(curr_B_offset + UInt32(3) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 4:
                B_tile_4 = B.ptr.offset(curr_B_offset + UInt32(4) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 5:
                B_tile_5 = B.ptr.offset(curr_B_offset + UInt32(5) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 6:
                B_tile_6 = B.ptr.offset(curr_B_offset + UInt32(6) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 7:
                B_tile_7 = B.ptr.offset(curr_B_offset + UInt32(7) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 8:
                B_tile_8 = B.ptr.offset(curr_B_offset + UInt32(8) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 9:
                B_tile_9 = B.ptr.offset(curr_B_offset + UInt32(9) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 10:
                B_tile_10 = B.ptr.offset(curr_B_offset + UInt32(10) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 11:
                B_tile_11 = B.ptr.offset(curr_B_offset + UInt32(11) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 12:
                B_tile_12 = B.ptr.offset(curr_B_offset + UInt32(12) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 13:
                B_tile_13 = B.ptr.offset(curr_B_offset + UInt32(13) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 14:
                B_tile_14 = B.ptr.offset(curr_B_offset + UInt32(14) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 15:
                B_tile_15 = B.ptr.offset(curr_B_offset + UInt32(15) * B_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
        else:
            # Scalar fallback for non-contiguous B
            for i in range(TILE_SIZE):
                var b_base = curr_B_offset + UInt32(i) * B_t_stride
                if dstate > 0: B_tile_0[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(0) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 1: B_tile_1[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(1) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 2: B_tile_2[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(2) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 3: B_tile_3[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(3) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 4: B_tile_4[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(4) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 5: B_tile_5[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(5) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 6: B_tile_6[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(6) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 7: B_tile_7[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(7) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 8: B_tile_8[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(8) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 9: B_tile_9[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(9) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 10: B_tile_10[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(10) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 11: B_tile_11[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(11) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 12: B_tile_12[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(12) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 13: B_tile_13[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(13) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 14: B_tile_14[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(14) * B_n_stride).load()).cast[DType.float32]()
                if dstate > 15: B_tile_15[i] = Scalar[kernel_dtype](B.ptr.offset(b_base + UInt32(15) * B_n_stride).load()).cast[DType.float32]()
        
        # Load C tiles - vector load if C is contiguous in t
        if C_contiguous:
            if dstate > 0:
                C_tile_0 = C.ptr.offset(curr_C_offset + UInt32(0) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 1:
                C_tile_1 = C.ptr.offset(curr_C_offset + UInt32(1) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 2:
                C_tile_2 = C.ptr.offset(curr_C_offset + UInt32(2) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 3:
                C_tile_3 = C.ptr.offset(curr_C_offset + UInt32(3) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 4:
                C_tile_4 = C.ptr.offset(curr_C_offset + UInt32(4) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 5:
                C_tile_5 = C.ptr.offset(curr_C_offset + UInt32(5) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 6:
                C_tile_6 = C.ptr.offset(curr_C_offset + UInt32(6) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 7:
                C_tile_7 = C.ptr.offset(curr_C_offset + UInt32(7) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 8:
                C_tile_8 = C.ptr.offset(curr_C_offset + UInt32(8) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 9:
                C_tile_9 = C.ptr.offset(curr_C_offset + UInt32(9) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 10:
                C_tile_10 = C.ptr.offset(curr_C_offset + UInt32(10) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 11:
                C_tile_11 = C.ptr.offset(curr_C_offset + UInt32(11) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 12:
                C_tile_12 = C.ptr.offset(curr_C_offset + UInt32(12) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 13:
                C_tile_13 = C.ptr.offset(curr_C_offset + UInt32(13) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 14:
                C_tile_14 = C.ptr.offset(curr_C_offset + UInt32(14) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
            if dstate > 15:
                C_tile_15 = C.ptr.offset(curr_C_offset + UInt32(15) * C_n_stride).load[width=TILE_SIZE]().cast[DType.float32]()
        else:
            # Scalar fallback for non-contiguous C
            for i in range(TILE_SIZE):
                var c_base = curr_C_offset + UInt32(i) * C_t_stride
                if dstate > 0: C_tile_0[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(0) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 1: C_tile_1[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(1) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 2: C_tile_2[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(2) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 3: C_tile_3[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(3) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 4: C_tile_4[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(4) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 5: C_tile_5[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(5) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 6: C_tile_6[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(6) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 7: C_tile_7[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(7) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 8: C_tile_8[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(8) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 9: C_tile_9[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(9) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 10: C_tile_10[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(10) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 11: C_tile_11[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(11) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 12: C_tile_12[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(12) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 13: C_tile_13[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(13) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 14: C_tile_14[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(14) * C_n_stride).load()).cast[DType.float32]()
                if dstate > 15: C_tile_15[i] = Scalar[kernel_dtype](C.ptr.offset(c_base + UInt32(15) * C_n_stride).load()).cast[DType.float32]()
        
        # Buffer for output values to enable vector stores
        var output_buffer = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var out_z_buffer = SIMD[kernel_dtype, TILE_SIZE](0.0)
        
        # Process tile with pre-loaded B/C values
        for i in range(TILE_SIZE):
            t_in_chunk += 1
            
            # Extract scalars from pre-loaded vectors
            var u_val = u_vec[i].cast[DType.float32]()
            var delta_val = delta_vec[i].cast[DType.float32]()
            
            # Apply delta bias and softplus
            if has_delta_bias:
                delta_val += delta_bias_val
            if delta_softplus_bool:
                delta_val = softplus(delta_val)
                
            var delta_u = delta_val * u_val
            
            # Extract B/C values for this timestep from pre-loaded tiles
            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            
            if dstate > 0: B_vals[0] = B_tile_0[i]
            if dstate > 1: B_vals[1] = B_tile_1[i]
            if dstate > 2: B_vals[2] = B_tile_2[i]
            if dstate > 3: B_vals[3] = B_tile_3[i]
            if dstate > 4: B_vals[4] = B_tile_4[i]
            if dstate > 5: B_vals[5] = B_tile_5[i]
            if dstate > 6: B_vals[6] = B_tile_6[i]
            if dstate > 7: B_vals[7] = B_tile_7[i]
            if dstate > 8: B_vals[8] = B_tile_8[i]
            if dstate > 9: B_vals[9] = B_tile_9[i]
            if dstate > 10: B_vals[10] = B_tile_10[i]
            if dstate > 11: B_vals[11] = B_tile_11[i]
            if dstate > 12: B_vals[12] = B_tile_12[i]
            if dstate > 13: B_vals[13] = B_tile_13[i]
            if dstate > 14: B_vals[14] = B_tile_14[i]
            if dstate > 15: B_vals[15] = B_tile_15[i]
            
            if dstate > 0: C_vals[0] = C_tile_0[i]
            if dstate > 1: C_vals[1] = C_tile_1[i]
            if dstate > 2: C_vals[2] = C_tile_2[i]
            if dstate > 3: C_vals[3] = C_tile_3[i]
            if dstate > 4: C_vals[4] = C_tile_4[i]
            if dstate > 5: C_vals[5] = C_tile_5[i]
            if dstate > 6: C_vals[6] = C_tile_6[i]
            if dstate > 7: C_vals[7] = C_tile_7[i]
            if dstate > 8: C_vals[8] = C_tile_8[i]
            if dstate > 9: C_vals[9] = C_tile_9[i]
            if dstate > 10: C_vals[10] = C_tile_10[i]
            if dstate > 11: C_vals[11] = C_tile_11[i]
            if dstate > 12: C_vals[12] = C_tile_12[i]
            if dstate > 13: C_vals[13] = C_tile_13[i]
            if dstate > 14: C_vals[14] = C_tile_14[i]
            if dstate > 15: C_vals[15] = C_tile_15[i]
            
            # SIMD Math
            var a_t = exp2(A_vals * delta_val)
            var b_t = B_vals * delta_u
            state = state * a_t + b_t
            var output_val = (state * C_vals).reduce_add()
            
            cum_b = cum_b * a_t + b_t
            cum_a = cum_a * a_t
            
            if has_D:
                output_val += D_val * u_val
            
            # Buffer output for vector store
            output_buffer[i] = output_val.cast[kernel_dtype]()
            
            if has_z:
                var z_val = z_vec[i].cast[DType.float32]()
                var out_z_val = output_val * silu(z_val)
                out_z_buffer[i] = out_z_val.cast[kernel_dtype]()
            
            # Checkpoint handling
            var current_t = t + i
            var is_chunk_boundary = (t_in_chunk == chunk_size)
            var is_last_step = (current_t == seqlen - 1)
            
            if is_chunk_boundary or is_last_step:
                for n in range(dstate):
                    var x_offset_a = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride + UInt32(chunk_idx) * x_chunk_stride + UInt32(n * 2) * x_n_stride
                    var x_offset_b = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride + UInt32(chunk_idx) * x_chunk_stride + UInt32(n * 2 + 1) * x_n_stride
                    x.ptr.offset(x_offset_a).store(Scalar[kernel_dtype](cum_a[n].cast[kernel_dtype]()))
                    x.ptr.offset(x_offset_b).store(Scalar[kernel_dtype](cum_b[n].cast[kernel_dtype]()))
                    cum_a[n] = 1.0
                    cum_b[n] = 0.0
                
                if is_chunk_boundary:
                    chunk_idx += 1
                    t_in_chunk = 0
        
        # Vector store outputs if contiguous
        if output_contiguous:
            output.ptr.offset(curr_output_offset).store[width=TILE_SIZE](output_buffer)
        else:
            for i in range(TILE_SIZE):
                var out_off = curr_output_offset + UInt32(i) * output_t_stride
                output.ptr.offset(out_off).store(output_buffer[i])
        
        if has_z and has_out_z:
            if out_z_t_stride == 1:
                out_z.ptr.offset(curr_out_z_offset).store[width=TILE_SIZE](out_z_buffer)
            else:
                for i in range(TILE_SIZE):
                    var out_z_off = curr_out_z_offset + UInt32(i) * out_z_t_stride
                    out_z.ptr.offset(out_z_off).store(out_z_buffer[i])

        # Advance global offsets by TILE_SIZE
        curr_u_offset += u_t_stride * UInt32(TILE_SIZE)
        curr_delta_offset += delta_t_stride * UInt32(TILE_SIZE)
        curr_output_offset += output_t_stride * UInt32(TILE_SIZE)
        curr_B_offset += B_t_stride * UInt32(TILE_SIZE)
        curr_C_offset += C_t_stride * UInt32(TILE_SIZE)
        curr_z_offset += z_t_stride * UInt32(TILE_SIZE)
        curr_out_z_offset += out_z_t_stride * UInt32(TILE_SIZE)
        
        t += TILE_SIZE

    # Tail loop (scalar)
    while t < seqlen:
        t_in_chunk += 1
        # ... (same body as original loop) ...
        # Copied from original for correctness
        var u_val = Scalar[kernel_dtype](u.ptr.offset(curr_u_offset).load()).cast[DType.float32]()
        var delta_val = Scalar[kernel_dtype](delta.ptr.offset(curr_delta_offset).load()).cast[DType.float32]()
        if has_delta_bias: delta_val += delta_bias_val
        if delta_softplus_bool: delta_val = softplus(delta_val)
        var delta_u = delta_val * u_val
        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        for n in range(dstate):
            B_vals[n] = Scalar[kernel_dtype](B.ptr.offset(curr_B_offset + UInt32(n) * B_n_stride).load()).cast[DType.float32]()
            C_vals[n] = Scalar[kernel_dtype](C.ptr.offset(curr_C_offset + UInt32(n) * C_n_stride).load()).cast[DType.float32]()
        var a_t = exp2(A_vals * delta_val)
        var b_t = B_vals * delta_u
        state = state * a_t + b_t
        var output_val = (state * C_vals).reduce_add()
        cum_b = cum_b * a_t + b_t
        cum_a = cum_a * a_t
        if has_D: output_val += D_val * u_val
        output.ptr.offset(curr_output_offset).store(Scalar[kernel_dtype](output_val.cast[kernel_dtype]()))
        if has_z:
            var z_val = Scalar[kernel_dtype](z.ptr.offset(curr_z_offset).load()).cast[DType.float32]()
            var out_z_val = output_val * silu(z_val)
            if has_out_z: out_z.ptr.offset(curr_out_z_offset).store(Scalar[kernel_dtype](out_z_val.cast[kernel_dtype]()))
        
        curr_u_offset += u_t_stride
        curr_delta_offset += delta_t_stride
        curr_output_offset += output_t_stride
        curr_B_offset += B_t_stride
        curr_C_offset += C_t_stride
        curr_z_offset += z_t_stride
        curr_out_z_offset += out_z_t_stride
        
        var is_chunk_boundary = (t_in_chunk == chunk_size)
        var is_last_step = (t == seqlen - 1)
        if is_chunk_boundary or is_last_step:
            for n in range(dstate):
                var x_offset_a = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride + UInt32(chunk_idx) * x_chunk_stride + UInt32(n * 2) * x_n_stride
                var x_offset_b = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride + UInt32(chunk_idx) * x_chunk_stride + UInt32(n * 2 + 1) * x_n_stride
                x.ptr.offset(x_offset_a).store(Scalar[kernel_dtype](cum_a[n].cast[kernel_dtype]()))
                x.ptr.offset(x_offset_b).store(Scalar[kernel_dtype](cum_b[n].cast[kernel_dtype]()))
                cum_a[n] = 1.0
                cum_b[n] = 0.0
            if is_chunk_boundary:
                chunk_idx += 1
                t_in_chunk = 0
        t += 1


# ===----------------------------------------------------------------------=== #
# Selective Scan Update Kernel (Single Step / Autoregressive)
# ===----------------------------------------------------------------------=== #
# This kernel is used for incremental/autoregressive inference where you
# process one token at a time and update the hidden state.
#
# Reference: mamba_ssm/ops/triton/selective_state_update.py
#
# Algorithm:
#   dt = dt + dt_bias (if has_dt_bias)
#   dt = softplus(dt) (if dt_softplus)
#   dA = exp(A * dt)
#   dB = B * dt
#   state = state * dA + dB * x
#   out = sum(state * C, axis=-1)
#   out += x * D (if has_D)
#   out *= z * sigmoid(z) (if has_z)
# ===----------------------------------------------------------------------=== #

fn selective_scan_update_gpu[
    kernel_dtype: DType,
    state_out_layout: Layout,
    output_layout: Layout,
    state_in_layout: Layout,
    x_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    dt_bias_layout: Layout,
](
    total_batch_dim: Int,
    batch: Int,
    dim: Int,
    dstate: Int,
    delta_softplus: Int8,
    # Outputs first (MAX convention)
    state_out: LayoutTensor[kernel_dtype, state_out_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    # Inputs
    state_in: LayoutTensor[kernel_dtype, state_in_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    dt: LayoutTensor[kernel_dtype, dt_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    dt_bias: LayoutTensor[kernel_dtype, dt_bias_layout, MutAnyOrigin],
    # Strides for state_out
    state_out_b_stride: UInt32,
    state_out_d_stride: UInt32,
    state_out_n_stride: UInt32,
    output_b_stride: UInt32,
    output_d_stride: UInt32,
    # Strides for state_in
    state_in_b_stride: UInt32,
    state_in_d_stride: UInt32,
    state_in_n_stride: UInt32,
    x_b_stride: UInt32,
    x_d_stride: UInt32,
    dt_b_stride: UInt32,
    dt_d_stride: UInt32,
    A_d_stride: UInt32,
    A_n_stride: UInt32,
    B_b_stride: UInt32,
    B_n_stride: UInt32,
    C_b_stride: UInt32,
    C_n_stride: UInt32,
    D_stride: UInt32,
    z_b_stride: UInt32,
    z_d_stride: UInt32,
    dt_bias_stride: UInt32,
):
    """GPU kernel for selective scan update (single step).
    
    Each thread processes one (batch, dim) pair.
    Reads initial state from state_in, writes updated state to state_out.
    
    Args:
        state_out: Output SSM state tensor (batch, dim, dstate)
        output: Output tensor (batch, dim)
        state_in: Input SSM state tensor (batch, dim, dstate)
        x: Input tensor (batch, dim)
        dt: Delta/timestep tensor (batch, dim)
        A: State transition matrix (dim, dstate)
        B: Input projection (batch, dstate)
        C: Output projection (batch, dstate)
        D: Optional skip connection (dim,)
        z: Optional gate (batch, dim)
        dt_bias: Optional delta bias (dim,)
    """
    # Calculate which (batch, dim) this thread is responsible for
    var thread_id = block_dim.x * block_idx.x + thread_idx.x
    var thread_id_int = Int(thread_id)
    if thread_id_int >= total_batch_dim:
        return
        
    var b = thread_id_int // dim
    var d = thread_id_int % dim
    
    # Additional bounds checking
    if b >= batch or d >= dim:
        return
    
    # Load dt value
    var dt_offset = UInt32(b) * dt_b_stride + UInt32(d) * dt_d_stride
    var dt_val = Scalar[kernel_dtype](dt.ptr.offset(dt_offset).load()).cast[DType.float32]()
    
    # Apply dt_bias if present
    var has_dt_bias = dt_bias.dim(0) > 0
    if has_dt_bias:
        var bias_offset = UInt32(d) * dt_bias_stride
        var bias_val = Scalar[kernel_dtype](dt_bias.ptr.offset(bias_offset).load()).cast[DType.float32]()
        dt_val += bias_val
    
    # Apply softplus if requested
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
    if delta_softplus_bool:
        dt_val = softplus(dt_val)
    
    # Load x value
    var x_offset = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride
    var x_val = Scalar[kernel_dtype](x.ptr.offset(x_offset).load()).cast[DType.float32]()
    
    # Load A values for this dim and pre-multiply by LOG2E for faster exp2
    var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
    for n in range(dstate):
        var A_offset = UInt32(d) * A_d_stride + UInt32(n) * A_n_stride
        A_vals[n] = Scalar[kernel_dtype](A.ptr.offset(A_offset).load()).cast[DType.float32]() * LOG2E
    
    # Compute dA = exp2(A * LOG2E * dt) = exp(A * dt)
    var dA = exp2(A_vals * dt_val)
    
    # Load B values
    var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
    for n in range(dstate):
        var B_offset = UInt32(b) * B_b_stride + UInt32(n) * B_n_stride
        B_vals[n] = Scalar[kernel_dtype](B.ptr.offset(B_offset).load()).cast[DType.float32]()
    
    # Compute dB = B * dt
    var dB = B_vals * dt_val
    
    # Load current state from state_in
    var state_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
    for n in range(dstate):
        var state_offset = UInt32(b) * state_in_b_stride + UInt32(d) * state_in_d_stride + UInt32(n) * state_in_n_stride
        state_vals[n] = Scalar[kernel_dtype](state_in.ptr.offset(state_offset).load()).cast[DType.float32]()
    
    # Update state: state = state * dA + dB * x
    state_vals = state_vals * dA + dB * x_val
    
    # Store updated state to state_out
    for n in range(dstate):
        var state_offset = UInt32(b) * state_out_b_stride + UInt32(d) * state_out_d_stride + UInt32(n) * state_out_n_stride
        state_out.ptr.offset(state_offset).store(Scalar[kernel_dtype](state_vals[n].cast[kernel_dtype]()))
    
    # Load C values
    var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
    for n in range(dstate):
        var C_offset = UInt32(b) * C_b_stride + UInt32(n) * C_n_stride
        C_vals[n] = Scalar[kernel_dtype](C.ptr.offset(C_offset).load()).cast[DType.float32]()
    
    # Compute output: out = sum(state * C, axis=-1)
    var out_val = (state_vals * C_vals).reduce_add()
    
    # Add skip connection if D is present
    var has_D = D.dim(0) > 0
    if has_D:
        var D_offset = UInt32(d) * D_stride
        var D_val = Scalar[kernel_dtype](D.ptr.offset(D_offset).load()).cast[DType.float32]()
        out_val += x_val * D_val
    
    # Apply gating if z is present
    var has_z = z.dim(0) > 0
    if has_z:
        var z_offset = UInt32(b) * z_b_stride + UInt32(d) * z_d_stride
        var z_val = Scalar[kernel_dtype](z.ptr.offset(z_offset).load()).cast[DType.float32]()
        out_val *= z_val * sigmoid(z_val)  # z * sigmoid(z) = silu(z) but formulated differently
    
    # Store output
    var out_offset = UInt32(b) * output_b_stride + UInt32(d) * output_d_stride
    output.ptr.offset(out_offset).store(Scalar[kernel_dtype](out_val.cast[kernel_dtype]()))


fn selective_scan_update_cpu[
    kernel_dtype: DType,
    state_out_layout: Layout,
    output_layout: Layout,
    state_in_layout: Layout,
    x_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    dt_bias_layout: Layout,
](
    batch: Int,
    dim: Int,
    dstate: Int,
    delta_softplus: Int8,
    # Outputs first (MAX convention)
    state_out: LayoutTensor[kernel_dtype, state_out_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    # Inputs
    state_in: LayoutTensor[kernel_dtype, state_in_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    dt: LayoutTensor[kernel_dtype, dt_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    dt_bias: LayoutTensor[kernel_dtype, dt_bias_layout, MutAnyOrigin],
    # Strides for state_out
    state_out_b_stride: UInt32,
    state_out_d_stride: UInt32,
    state_out_n_stride: UInt32,
    output_b_stride: UInt32,
    output_d_stride: UInt32,
    # Strides for state_in
    state_in_b_stride: UInt32,
    state_in_d_stride: UInt32,
    state_in_n_stride: UInt32,
    x_b_stride: UInt32,
    x_d_stride: UInt32,
    dt_b_stride: UInt32,
    dt_d_stride: UInt32,
    A_d_stride: UInt32,
    A_n_stride: UInt32,
    B_b_stride: UInt32,
    B_n_stride: UInt32,
    C_b_stride: UInt32,
    C_n_stride: UInt32,
    D_stride: UInt32,
    z_b_stride: UInt32,
    z_d_stride: UInt32,
    dt_bias_stride: UInt32,
):
    """CPU kernel for selective scan update (single step).
    
    Parallelized over batch and dimension.
    """
    var has_dt_bias = dt_bias.dim(0) > 0
    var has_D = D.dim(0) > 0
    var has_z = z.dim(0) > 0
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
    
    @parameter
    fn worker(idx: Int):
        var b = idx // dim
        var d = idx % dim
        
        # Load dt value
        var dt_offset = UInt32(b) * dt_b_stride + UInt32(d) * dt_d_stride
        var dt_val = Scalar[kernel_dtype](dt.ptr.offset(dt_offset).load()).cast[DType.float32]()
        
        # Apply dt_bias if present
        if has_dt_bias:
            var bias_offset = UInt32(d) * dt_bias_stride
            var bias_val = Scalar[kernel_dtype](dt_bias.ptr.offset(bias_offset).load()).cast[DType.float32]()
            dt_val += bias_val
        
        # Apply softplus if requested
        if delta_softplus_bool:
            dt_val = softplus(dt_val)
        
        # Load x value
        var x_offset = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride
        var x_val = Scalar[kernel_dtype](x.ptr.offset(x_offset).load()).cast[DType.float32]()
        
        # Load A values and pre-multiply by LOG2E
        var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        for n in range(dstate):
            var A_offset = UInt32(d) * A_d_stride + UInt32(n) * A_n_stride
            A_vals[n] = Scalar[kernel_dtype](A.ptr.offset(A_offset).load()).cast[DType.float32]() * LOG2E
        
        # Compute dA
        var dA = exp2(A_vals * dt_val)
        
        # Load B values
        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        for n in range(dstate):
            var B_offset = UInt32(b) * B_b_stride + UInt32(n) * B_n_stride
            B_vals[n] = Scalar[kernel_dtype](B.ptr.offset(B_offset).load()).cast[DType.float32]()
        
        # Compute dB
        var dB = B_vals * dt_val
        
        # Load current state from state_in
        var state_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        for n in range(dstate):
            var state_offset = UInt32(b) * state_in_b_stride + UInt32(d) * state_in_d_stride + UInt32(n) * state_in_n_stride
            state_vals[n] = Scalar[kernel_dtype](state_in.ptr.offset(state_offset).load()).cast[DType.float32]()
        
        # Update state
        state_vals = state_vals * dA + dB * x_val
        
        # Store updated state to state_out
        for n in range(dstate):
            var state_offset = UInt32(b) * state_out_b_stride + UInt32(d) * state_out_d_stride + UInt32(n) * state_out_n_stride
            state_out.ptr.offset(state_offset).store(Scalar[kernel_dtype](state_vals[n].cast[kernel_dtype]()))
        
        # Load C values
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        for n in range(dstate):
            var C_offset = UInt32(b) * C_b_stride + UInt32(n) * C_n_stride
            C_vals[n] = Scalar[kernel_dtype](C.ptr.offset(C_offset).load()).cast[DType.float32]()
        
        # Compute output
        var out_val = (state_vals * C_vals).reduce_add()
        
        # Add skip connection
        if has_D:
            var D_offset = UInt32(d) * D_stride
            var D_val = Scalar[kernel_dtype](D.ptr.offset(D_offset).load()).cast[DType.float32]()
            out_val += x_val * D_val
        
        # Apply gating
        if has_z:
            var z_offset = UInt32(b) * z_b_stride + UInt32(d) * z_d_stride
            var z_val = Scalar[kernel_dtype](z.ptr.offset(z_offset).load()).cast[DType.float32]()
            out_val *= z_val * sigmoid(z_val)
        
        # Store output
        var out_offset = UInt32(b) * output_b_stride + UInt32(d) * output_d_stride
        output.ptr.offset(out_offset).store(Scalar[kernel_dtype](out_val.cast[kernel_dtype]()))

    sync_parallelize[worker](batch * dim)


fn selective_scan_fwd_cpu[
    kernel_dtype: DType,
    output_layout: Layout,
    x_layout: Layout,
    out_z_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    dstate: Int,
    group_size: Int,
    delta_softplus: Int8,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    out_z: LayoutTensor[kernel_dtype, out_z_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    delta_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    # Strides
    output_b_stride: UInt32,
    output_d_stride: UInt32,
    output_t_stride: UInt32,
    x_b_stride: UInt32,
    x_d_stride: UInt32,
    x_chunk_stride: UInt32,
    x_n_stride: UInt32,
    out_z_b_stride: UInt32,
    out_z_d_stride: UInt32,
    out_z_t_stride: UInt32,
    u_b_stride: UInt32,
    u_d_stride: UInt32,
    u_t_stride: UInt32,
    delta_b_stride: UInt32,
    delta_d_stride: UInt32,
    delta_t_stride: UInt32,
    A_d_stride: UInt32,
    A_n_stride: UInt32,
    B_b_stride: UInt32,
    B_g_stride: UInt32,
    B_n_stride: UInt32,
    B_t_stride: UInt32,
    C_b_stride: UInt32,
    C_g_stride: UInt32,
    C_n_stride: UInt32,
    C_t_stride: UInt32,
    D_stride: UInt32,
    z_b_stride: UInt32,
    z_d_stride: UInt32,
    z_t_stride: UInt32,
    delta_bias_stride: UInt32,
):
    """CPU kernel for selective scan operation.
    
    Parallelized over batch and dimension.
    """
    
    @parameter
    fn worker(idx: Int):
        var b = idx // dim
        var d = idx % dim
        
        var group_id = d // group_size
        
        # Local state storage (max dstate 16)
        var state = SIMD[DType.float32, MAX_DSTATE](0.0)
        var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
        var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)
        
        # Pre-load A values for this dim and pre-multiply by LOG2E for faster exp2
        var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var has_delta_bias = delta_bias.dim(0) > 0
        var delta_bias_val = Float32(0.0)
        if has_delta_bias:
            var bias_offset = UInt32(d) * delta_bias_stride
            delta_bias_val = Scalar[kernel_dtype](delta_bias.ptr.offset(bias_offset).load()).cast[DType.float32]()
        
        var has_D = D.dim(0) > 0
        var D_val = Float32(0.0)
        if has_D:
            var D_offset = UInt32(d) * D_stride
            D_val = Scalar[kernel_dtype](D.ptr.offset(D_offset).load()).cast[DType.float32]()
        
        var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
        var has_z = z.dim(0) > 0
        var has_out_z = out_z.dim(0) > 0
        
        for n in range(dstate):
            var A_offset = UInt32(d) * A_d_stride + UInt32(n) * A_n_stride
            A_vals[n] = Scalar[kernel_dtype](A.ptr.offset(A_offset).load()).cast[DType.float32]() * LOG2E

        var chunk_size = 2048
        var t_in_chunk = 0
        var chunk_idx = 0
        
        var curr_u_offset = UInt32(b) * u_b_stride + UInt32(d) * u_d_stride
        var curr_delta_offset = UInt32(b) * delta_b_stride + UInt32(d) * delta_d_stride
        var curr_output_offset = UInt32(b) * output_b_stride + UInt32(d) * output_d_stride
        var curr_B_offset = UInt32(b) * B_b_stride + UInt32(group_id) * B_g_stride
        var curr_C_offset = UInt32(b) * C_b_stride + UInt32(group_id) * C_g_stride
        var curr_z_offset = UInt32(b) * z_b_stride + UInt32(d) * z_d_stride
        var curr_out_z_offset = UInt32(b) * out_z_b_stride + UInt32(d) * out_z_d_stride
        
        comptime TILE_SIZE = 4
        var aligned_seqlen = seqlen - (seqlen % TILE_SIZE)
        var t = 0
        
        while t < aligned_seqlen:
            var u_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
            var delta_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
            var z_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
            
            if u_t_stride == 1:
                u_vec = u.ptr.offset(curr_u_offset).load[width=TILE_SIZE]()
            else:
                for i in range(TILE_SIZE):
                    u_vec[i] = u.ptr.offset(curr_u_offset + UInt32(i) * u_t_stride).load()
                    
            if delta_t_stride == 1:
                delta_vec = delta.ptr.offset(curr_delta_offset).load[width=TILE_SIZE]()
            else:
                for i in range(TILE_SIZE):
                    delta_vec[i] = delta.ptr.offset(curr_delta_offset + UInt32(i) * delta_t_stride).load()
                    
            if has_z:
                if z_t_stride == 1:
                    z_vec = z.ptr.offset(curr_z_offset).load[width=TILE_SIZE]()
                else:
                    for i in range(TILE_SIZE):
                        z_vec[i] = z.ptr.offset(curr_z_offset + UInt32(i) * z_t_stride).load()
            
            for i in range(TILE_SIZE):
                t_in_chunk += 1
                
                var u_val = u_vec[i].cast[DType.float32]()
                var delta_val = delta_vec[i].cast[DType.float32]()
                
                if has_delta_bias:
                    delta_val += delta_bias_val
                
                if delta_softplus_bool:
                    delta_val = softplus(delta_val)
                    
                var delta_u = delta_val * u_val
                
                var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
                
                for n in range(dstate):
                    var b_off = curr_B_offset + UInt32(i) * B_t_stride + UInt32(n) * B_n_stride
                    var c_off = curr_C_offset + UInt32(i) * C_t_stride + UInt32(n) * C_n_stride
                    
                    B_vals[n] = Scalar[kernel_dtype](B.ptr.offset(b_off).load()).cast[DType.float32]()
                    C_vals[n] = Scalar[kernel_dtype](C.ptr.offset(c_off).load()).cast[DType.float32]()
                
                var a_t = exp2(A_vals * delta_val)
                var b_t = B_vals * delta_u
                state = state * a_t + b_t
                var output_val = (state * C_vals).reduce_add()
                
                cum_b = cum_b * a_t + b_t
                cum_a = cum_a * a_t
                
                if has_D:
                    output_val += D_val * u_val
                
                var final_val = output_val.cast[kernel_dtype]()
                
                if has_z:
                    var z_val = z_vec[i].cast[DType.float32]()
                    var out_z_val = output_val * silu(z_val)
                    if has_out_z:
                        var out_z_off = curr_out_z_offset + UInt32(i) * out_z_t_stride
                        out_z.ptr.offset(out_z_off).store(Scalar[kernel_dtype](out_z_val.cast[kernel_dtype]()))
                
                var out_off = curr_output_offset + UInt32(i) * output_t_stride
                output.ptr.offset(out_off).store(Scalar[kernel_dtype](final_val))
                
                var is_chunk_boundary = (t_in_chunk == chunk_size)
                var current_t = t + i
                var is_last_step = (current_t == seqlen - 1)
                
                if is_chunk_boundary or is_last_step:
                    for n in range(dstate):
                        var x_offset_a = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride + UInt32(chunk_idx) * x_chunk_stride + UInt32(n * 2) * x_n_stride
                        var x_offset_b = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride + UInt32(chunk_idx) * x_chunk_stride + UInt32(n * 2 + 1) * x_n_stride
                        x.ptr.offset(x_offset_a).store(Scalar[kernel_dtype](cum_a[n].cast[kernel_dtype]()))
                        x.ptr.offset(x_offset_b).store(Scalar[kernel_dtype](cum_b[n].cast[kernel_dtype]()))
                        cum_a[n] = 1.0
                        cum_b[n] = 0.0
                    
                    if is_chunk_boundary:
                        chunk_idx += 1
                        t_in_chunk = 0

            curr_u_offset += u_t_stride * UInt32(TILE_SIZE)
            curr_delta_offset += delta_t_stride * UInt32(TILE_SIZE)
            curr_output_offset += output_t_stride * UInt32(TILE_SIZE)
            curr_B_offset += B_t_stride * UInt32(TILE_SIZE)
            curr_C_offset += C_t_stride * UInt32(TILE_SIZE)
            curr_z_offset += z_t_stride * UInt32(TILE_SIZE)
            curr_out_z_offset += out_z_t_stride * UInt32(TILE_SIZE)
            
            t += TILE_SIZE

        while t < seqlen:
            t_in_chunk += 1
            var u_val = Scalar[kernel_dtype](u.ptr.offset(curr_u_offset).load()).cast[DType.float32]()
            var delta_val = Scalar[kernel_dtype](delta.ptr.offset(curr_delta_offset).load()).cast[DType.float32]()
            if has_delta_bias: delta_val += delta_bias_val
            if delta_softplus_bool: delta_val = softplus(delta_val)
            var delta_u = delta_val * u_val
            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                B_vals[n] = Scalar[kernel_dtype](B.ptr.offset(curr_B_offset + UInt32(n) * B_n_stride).load()).cast[DType.float32]()
                C_vals[n] = Scalar[kernel_dtype](C.ptr.offset(curr_C_offset + UInt32(n) * C_n_stride).load()).cast[DType.float32]()
            var a_t = exp2(A_vals * delta_val)
            var b_t = B_vals * delta_u
            state = state * a_t + b_t
            var output_val = (state * C_vals).reduce_add()
            cum_b = cum_b * a_t + b_t
            cum_a = cum_a * a_t
            if has_D: output_val += D_val * u_val
            output.ptr.offset(curr_output_offset).store(Scalar[kernel_dtype](output_val.cast[kernel_dtype]()))
            if has_z:
                var z_val = Scalar[kernel_dtype](z.ptr.offset(curr_z_offset).load()).cast[DType.float32]()
                var out_z_val = output_val * silu(z_val)
                if has_out_z: out_z.ptr.offset(curr_out_z_offset).store(Scalar[kernel_dtype](out_z_val.cast[kernel_dtype]()))
            
            curr_u_offset += u_t_stride
            curr_delta_offset += delta_t_stride
            curr_output_offset += output_t_stride
            curr_B_offset += B_t_stride
            curr_C_offset += C_t_stride
            curr_z_offset += z_t_stride
            curr_out_z_offset += out_z_t_stride
            
            var is_chunk_boundary = (t_in_chunk == chunk_size)
            var is_last_step = (t == seqlen - 1)
            if is_chunk_boundary or is_last_step:
                for n in range(dstate):
                    var x_offset_a = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride + UInt32(chunk_idx) * x_chunk_stride + UInt32(n * 2) * x_n_stride
                    var x_offset_b = UInt32(b) * x_b_stride + UInt32(d) * x_d_stride + UInt32(chunk_idx) * x_chunk_stride + UInt32(n * 2 + 1) * x_n_stride
                    x.ptr.offset(x_offset_a).store(Scalar[kernel_dtype](cum_a[n].cast[kernel_dtype]()))
                    x.ptr.offset(x_offset_b).store(Scalar[kernel_dtype](cum_b[n].cast[kernel_dtype]()))
                    cum_a[n] = 1.0
                    cum_b[n] = 0.0
                if is_chunk_boundary:
                    chunk_idx += 1
                    t_in_chunk = 0
            t += 1

    sync_parallelize[worker](batch * dim)
