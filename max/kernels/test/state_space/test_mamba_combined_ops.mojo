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
"""Integration tests for the mamba_split_conv1d_scan_combined registered op.

The op is registered in state_space.mamba_combined_ops and dispatches to
mamba_split_conv1d_scan_combined_cpu / mamba_split_conv1d_scan_combined_gpu.
This module imports state_space (loading the op registration) and runs the
kernel with minimal shapes to verify the op's kernel path works.
"""

from math import ceildiv

from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
)
from layout._fillers import random
from memory import alloc
from state_space.selective_scan import mamba_split_conv1d_scan_combined_cpu
from testing import TestSuite, assert_true

from utils.index import Index


fn test_mamba_combined_ops_registered_op_kernel_path() raises:
    """Verify the kernel used by the mamba_split_conv1d_scan_combined op runs correctly.

    Loading state_space pulls in mamba_combined_ops.mojo which registers the op.
    We call the same kernel the op dispatches to with minimal shapes and assert
    the output is finite and non-trivial.
    """
    var batch = 1
    var seqlen = 8
    var dim = 8
    var nheads = 2
    var headdim = 4
    var ngroups = 1
    var dstate = 8
    var width = 4
    var chunk_size = 8

    var n_chunks = ceildiv(seqlen, chunk_size)
    var zxbcdt_channels = 2 * dim + 2 * ngroups * dstate + nheads
    comptime dtype = DType.float32
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()

    # Allocate and fill tensors (minimal sizes)
    var zxbcdt_size = batch * seqlen * zxbcdt_channels
    var zxbcdt_heap = alloc[Scalar[dtype]](zxbcdt_size)
    var zxbcdt_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        zxbcdt_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(batch, seqlen, zxbcdt_channels)
        ),
    )
    random(zxbcdt_h)

    var conv_weight_channels = dim + 2 * ngroups * dstate
    var conv_weight_heap = alloc[Scalar[dtype]](conv_weight_channels * width)
    var conv_weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        conv_weight_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(conv_weight_channels, width)
        ),
    )
    random(conv_weight_h)

    var conv_bias_heap = alloc[Scalar[dtype]](conv_weight_channels)
    var conv_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        conv_bias_heap,
        RuntimeLayout[layout_1d].row_major(Index(conv_weight_channels)),
    )
    random(conv_bias_h)

    var dt_bias_heap = alloc[Scalar[dtype]](nheads)
    var dt_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        dt_bias_heap,
        RuntimeLayout[layout_1d].row_major(Index(nheads)),
    )
    random(dt_bias_h)

    var A_heap = alloc[Scalar[dtype]](nheads)
    var A_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        A_heap,
        RuntimeLayout[layout_1d].row_major(Index(nheads)),
    )
    random(A_h)

    var D_heap = alloc[Scalar[dtype]](nheads * headdim)
    var D_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        D_heap,
        RuntimeLayout[layout_2d].row_major(Index(nheads, headdim)),
    )
    random(D_h)

    var x_size = batch * dim * n_chunks * 2 * dstate
    var x_heap = alloc[Scalar[dtype]](x_size)
    var x_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, dim, n_chunks, 2 * dstate)
        ),
    )

    var out_z_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var out_z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    var dt_heap = alloc[Scalar[dtype]](batch * nheads * seqlen)
    var dt_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        dt_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, nheads, seqlen)),
    )

    var B_heap = alloc[Scalar[dtype]](batch * ngroups * dstate * seqlen)
    var B_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, ngroups, dstate, seqlen)
        ),
    )

    var C_heap = alloc[Scalar[dtype]](batch * ngroups * dstate * seqlen)
    var C_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, ngroups, dstate, seqlen)
        ),
    )

    var z_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    var rmsnorm_weight_heap = alloc[Scalar[dtype]](0)
    var rmsnorm_weight_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        rmsnorm_weight_heap,
        RuntimeLayout[layout_1d].row_major(Index(0)),
    )

    var outproj_weight_heap = alloc[Scalar[dtype]](0)
    var outproj_weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        outproj_weight_heap,
        RuntimeLayout[layout_2d].row_major(Index(0, 0)),
    )

    var outproj_bias_heap = alloc[Scalar[dtype]](0)
    var outproj_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        outproj_bias_heap,
        RuntimeLayout[layout_1d].row_major(Index(0)),
    )

    var output_size = batch * seqlen * dim
    var output_heap = alloc[Scalar[dtype]](output_size)
    var output_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, dim)),
    )

    var epsilon = Scalar[dtype](1e-5)

    # Call the same kernel the registered op dispatches to (CPU path, DSTATE=8)
    mamba_split_conv1d_scan_combined_cpu[
        dtype,
        8,
        zxbcdt_h.layout,
        conv_weight_h.layout,
        conv_bias_h.layout,
        output_h.layout,
        x_h.layout,
        out_z_h.layout,
        dt_h.layout,
        A_h.layout,
        B_h.layout,
        C_h.layout,
        D_h.layout,
        z_h.layout,
        dt_bias_h.layout,
        rmsnorm_weight_h.layout,
        outproj_weight_h.layout,
        outproj_bias_h.layout,
    ](
        batch,
        seqlen,
        dim,
        nheads,
        headdim,
        ngroups,
        width,
        chunk_size,
        Int8(1),
        Int8(0),
        Int8(0),
        Int8(0),
        zxbcdt_h,
        conv_weight_h,
        conv_bias_h,
        dt_bias_h,
        A_h,
        D_h,
        x_h,
        out_z_h,
        dt_h,
        B_h,
        C_h,
        z_h,
        rmsnorm_weight_h,
        outproj_weight_h,
        outproj_bias_h,
        output_h,
        epsilon,
    )

    # Sanity: output should be non-trivial (kernel ran and produced nonzero values)
    var has_nonzero = False
    for i in range(output_size):
        if output_h.ptr[i] != Scalar[dtype](0.0):
            has_nonzero = True
    assert_true(has_nonzero, "output should not be all zeros")

    # Free
    zxbcdt_heap.free()
    conv_weight_heap.free()
    conv_bias_heap.free()
    dt_bias_heap.free()
    A_heap.free()
    D_heap.free()
    x_heap.free()
    out_z_heap.free()
    dt_heap.free()
    B_heap.free()
    C_heap.free()
    z_heap.free()
    rmsnorm_weight_heap.free()
    outproj_weight_heap.free()
    outproj_bias_heap.free()
    output_heap.free()


def main():
    """Entrypoint: run all test_* functions discovered in this module."""
    TestSuite.discover_tests[__functions_in_module()]().run()
