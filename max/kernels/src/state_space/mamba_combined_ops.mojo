# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Mamba-2 fused combined op registration (conv1d + split + selective scan).

Registers the mamba_split_conv1d_scan_combined op for use from the graph.
This is the Mamba-2 style fused prefill path: packed zxbcdt input,
conv on xBC, selective scan, optional RMSNorm and output projection.
"""

from math import ceildiv

import compiler_internal as compiler
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

from state_space.selective_scan import (
    mamba_split_conv1d_scan_combined_cpu,
    mamba_split_conv1d_scan_combined_gpu,
)


# =============================================================================
# Mamba Split Conv1D Scan Combined (Mamba-2 fused prefill)
# =============================================================================


@compiler.register("mamba_split_conv1d_scan_combined")
struct MambaSplitConv1DScanCombined[
    delta_softplus: Bool = True,
    norm_before_gate: Bool = False,
    has_rmsnorm: Bool = False,
    has_outproj: Bool = False,
    chunk_size: Int = 2048,
]:
    """Fused Mamba-2 prefill: conv1d on xBC + selective scan + optional norm/outproj.

    Input zxbcdt layout: (batch, seqlen, 2*dim + 2*ngroups*dstate + nheads)
    - Channels 0..dim-1: z
    - Channels dim..dim+2*ngroups*dstate-1: xBC (before conv)
    - Channels 2*dim+2*ngroups*dstate..end: dt

    Parameters:
        delta_softplus: Apply softplus to delta.
        norm_before_gate: If True, apply RMSNorm before gating.
        has_rmsnorm: Use rmsnorm_weight (otherwise pass empty tensor).
        has_outproj: Use outproj_weight/outproj_bias (otherwise pass empty).
        chunk_size: Chunk size for selective scan (default 2048).
    """

    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        x: OutputTensor[dtype=dtype, rank=4],
        out_z: OutputTensor[dtype=dtype, rank=3],
        dt: OutputTensor[dtype=dtype, rank=3],
        B: OutputTensor[dtype=dtype, rank=4],
        C: OutputTensor[dtype=dtype, rank=4],
        z: OutputTensor[dtype=dtype, rank=3],
        zxbcdt: InputTensor[dtype=dtype, rank=3],
        conv_weight: InputTensor[dtype=dtype, rank=2],
        conv_bias: InputTensor[dtype=dtype, rank=1],
        dt_bias: InputTensor[dtype=dtype, rank=1],
        A: InputTensor[dtype=dtype, rank=1],
        D: InputTensor[dtype=dtype, rank=2],
        rmsnorm_weight: InputTensor[dtype=dtype, rank=1],
        outproj_weight: InputTensor[dtype=dtype, rank=2],
        outproj_bias: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var batch = zxbcdt.dim_size(0)
        var seqlen = zxbcdt.dim_size(1)
        var zxbcdt_channels = zxbcdt.dim_size(2)
        var nheads = A.dim_size(0)
        var ngroups = B.dim_size(1)
        var dstate = B.dim_size(2)
        var width = conv_weight.dim_size(1)

        if dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: " + String(dstate) + ". Expected 8 or 16."
            )
        # dim = (zxbcdt_channels - nheads) // 2 - ngroups * dstate
        var dim = (zxbcdt_channels - nheads) // 2 - ngroups * dstate
        if dim <= 0 or (dim // nheads) * nheads != dim:
            raise Error(
                "Invalid dim from zxbcdt_channels; dim must be positive and divisible by nheads"
            )
        var headdim = dim // nheads

        var n_chunks = ceildiv(seqlen, Self.chunk_size)

        # Validate output shapes
        if output.dim_size(0) != batch or output.dim_size(1) != seqlen:
            raise Error("output shape must be (batch, seqlen, out_dim)")
        if x.dim_size(0) != batch or x.dim_size(1) != dim or x.dim_size(2) != n_chunks or x.dim_size(3) != 2 * dstate:
            raise Error("x shape must be (batch, dim, n_chunks, 2*dstate)")
        if out_z.dim_size(0) != batch or out_z.dim_size(1) != dim or out_z.dim_size(2) != seqlen:
            raise Error("out_z shape must be (batch, dim, seqlen)")
        if dt.dim_size(0) != batch or dt.dim_size(1) != nheads or dt.dim_size(2) != seqlen:
            raise Error("dt shape must be (batch, nheads, seqlen)")
        if B.dim_size(0) != batch or B.dim_size(1) != ngroups or B.dim_size(2) != dstate or B.dim_size(3) != seqlen:
            raise Error("B shape must be (batch, ngroups, dstate, seqlen)")
        if C.dim_size(0) != batch or C.dim_size(1) != ngroups or C.dim_size(2) != dstate or C.dim_size(3) != seqlen:
            raise Error("C shape must be (batch, ngroups, dstate, seqlen)")
        if z.dim_size(0) != batch or z.dim_size(1) != dim or z.dim_size(2) != seqlen:
            raise Error("z shape must be (batch, dim, seqlen)")

        var zxbcdt_lt = zxbcdt.to_layout_tensor()
        var conv_weight_lt = conv_weight.to_layout_tensor()
        var conv_bias_lt = conv_bias.to_layout_tensor()
        var dt_bias_lt = dt_bias.to_layout_tensor()
        var A_lt = A.to_layout_tensor()
        var D_lt = D.to_layout_tensor()
        var rmsnorm_weight_lt = rmsnorm_weight.to_layout_tensor()
        var outproj_weight_lt = outproj_weight.to_layout_tensor()
        var outproj_bias_lt = outproj_bias.to_layout_tensor()
        var output_lt = output.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var out_z_lt = out_z.to_layout_tensor()
        var dt_lt = dt.to_layout_tensor()
        var B_lt = B.to_layout_tensor()
        var C_lt = C.to_layout_tensor()
        var z_lt = z.to_layout_tensor()

        comptime delta_softplus_int8: Int8 = Int8(1) if Self.delta_softplus else Int8(0)
        comptime norm_before_gate_int8: Int8 = Int8(1) if Self.norm_before_gate else Int8(0)
        comptime has_rmsnorm_int8: Int8 = Int8(1) if Self.has_rmsnorm else Int8(0)
        comptime has_outproj_int8: Int8 = Int8(1) if Self.has_outproj else Int8(0)

        @parameter
        if is_cpu[target]():
            if dstate == 16:
                mamba_split_conv1d_scan_combined_cpu[
                    dtype,
                    16,
                    zxbcdt_lt.layout,
                    conv_weight_lt.layout,
                    conv_bias_lt.layout,
                    output_lt.layout,
                    x_lt.layout,
                    out_z_lt.layout,
                    dt_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                    D_lt.layout,
                    z_lt.layout,
                    dt_bias_lt.layout,
                    rmsnorm_weight_lt.layout,
                    outproj_weight_lt.layout,
                    outproj_bias_lt.layout,
                ](
                    batch,
                    seqlen,
                    dim,
                    nheads,
                    headdim,
                    ngroups,
                    width,
                    Self.chunk_size,
                    delta_softplus_int8,
                    norm_before_gate_int8,
                    has_rmsnorm_int8,
                    has_outproj_int8,
                    zxbcdt_lt,
                    conv_weight_lt,
                    conv_bias_lt,
                    dt_bias_lt,
                    A_lt,
                    D_lt,
                    x_lt,
                    out_z_lt,
                    dt_lt,
                    B_lt,
                    C_lt,
                    z_lt,
                    rmsnorm_weight_lt,
                    outproj_weight_lt,
                    outproj_bias_lt,
                    output_lt,
                    epsilon,
                )
            else:
                mamba_split_conv1d_scan_combined_cpu[
                    dtype,
                    8,
                    zxbcdt_lt.layout,
                    conv_weight_lt.layout,
                    conv_bias_lt.layout,
                    output_lt.layout,
                    x_lt.layout,
                    out_z_lt.layout,
                    dt_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                    D_lt.layout,
                    z_lt.layout,
                    dt_bias_lt.layout,
                    rmsnorm_weight_lt.layout,
                    outproj_weight_lt.layout,
                    outproj_bias_lt.layout,
                ](
                    batch,
                    seqlen,
                    dim,
                    nheads,
                    headdim,
                    ngroups,
                    width,
                    Self.chunk_size,
                    delta_softplus_int8,
                    norm_before_gate_int8,
                    has_rmsnorm_int8,
                    has_outproj_int8,
                    zxbcdt_lt,
                    conv_weight_lt,
                    conv_bias_lt,
                    dt_bias_lt,
                    A_lt,
                    D_lt,
                    x_lt,
                    out_z_lt,
                    dt_lt,
                    B_lt,
                    C_lt,
                    z_lt,
                    rmsnorm_weight_lt,
                    outproj_weight_lt,
                    outproj_bias_lt,
                    output_lt,
                    epsilon,
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    mamba_split_conv1d_scan_combined_gpu[
                        dtype,
                        DSTATE_VAL,
                        zxbcdt_lt.layout,
                        conv_weight_lt.layout,
                        conv_bias_lt.layout,
                        output_lt.layout,
                        x_lt.layout,
                        out_z_lt.layout,
                        dt_lt.layout,
                        A_lt.layout,
                        B_lt.layout,
                        C_lt.layout,
                        D_lt.layout,
                        z_lt.layout,
                        dt_bias_lt.layout,
                        rmsnorm_weight_lt.layout,
                        outproj_weight_lt.layout,
                        outproj_bias_lt.layout,
                    ],
                    mamba_split_conv1d_scan_combined_gpu[
                        dtype,
                        DSTATE_VAL,
                        zxbcdt_lt.layout,
                        conv_weight_lt.layout,
                        conv_bias_lt.layout,
                        output_lt.layout,
                        x_lt.layout,
                        out_z_lt.layout,
                        dt_lt.layout,
                        A_lt.layout,
                        B_lt.layout,
                        C_lt.layout,
                        D_lt.layout,
                        z_lt.layout,
                        dt_bias_lt.layout,
                        rmsnorm_weight_lt.layout,
                        outproj_weight_lt.layout,
                        outproj_bias_lt.layout,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    seqlen,
                    dim,
                    nheads,
                    headdim,
                    ngroups,
                    width,
                    Self.chunk_size,
                    delta_softplus_int8,
                    norm_before_gate_int8,
                    has_rmsnorm_int8,
                    has_outproj_int8,
                    zxbcdt_lt,
                    conv_weight_lt,
                    conv_bias_lt,
                    dt_bias_lt,
                    A_lt,
                    D_lt,
                    x_lt,
                    out_z_lt,
                    dt_lt,
                    B_lt,
                    C_lt,
                    z_lt,
                    rmsnorm_weight_lt,
                    outproj_weight_lt,
                    outproj_bias_lt,
                    output_lt,
                    epsilon,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    mamba_split_conv1d_scan_combined_gpu[
                        dtype,
                        DSTATE_VAL,
                        zxbcdt_lt.layout,
                        conv_weight_lt.layout,
                        conv_bias_lt.layout,
                        output_lt.layout,
                        x_lt.layout,
                        out_z_lt.layout,
                        dt_lt.layout,
                        A_lt.layout,
                        B_lt.layout,
                        C_lt.layout,
                        D_lt.layout,
                        z_lt.layout,
                        dt_bias_lt.layout,
                        rmsnorm_weight_lt.layout,
                        outproj_weight_lt.layout,
                        outproj_bias_lt.layout,
                    ],
                    mamba_split_conv1d_scan_combined_gpu[
                        dtype,
                        DSTATE_VAL,
                        zxbcdt_lt.layout,
                        conv_weight_lt.layout,
                        conv_bias_lt.layout,
                        output_lt.layout,
                        x_lt.layout,
                        out_z_lt.layout,
                        dt_lt.layout,
                        A_lt.layout,
                        B_lt.layout,
                        C_lt.layout,
                        D_lt.layout,
                        z_lt.layout,
                        dt_bias_lt.layout,
                        rmsnorm_weight_lt.layout,
                        outproj_weight_lt.layout,
                        outproj_bias_lt.layout,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    seqlen,
                    dim,
                    nheads,
                    headdim,
                    ngroups,
                    width,
                    Self.chunk_size,
                    delta_softplus_int8,
                    norm_before_gate_int8,
                    has_rmsnorm_int8,
                    has_outproj_int8,
                    zxbcdt_lt,
                    conv_weight_lt,
                    conv_bias_lt,
                    dt_bias_lt,
                    A_lt,
                    D_lt,
                    x_lt,
                    out_z_lt,
                    dt_lt,
                    B_lt,
                    C_lt,
                    z_lt,
                    rmsnorm_weight_lt,
                    outproj_weight_lt,
                    outproj_bias_lt,
                    output_lt,
                    epsilon,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
        else:
            raise Error("Unsupported target: " + target)
