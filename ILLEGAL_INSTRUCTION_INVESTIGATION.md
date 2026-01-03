# Illegal Instruction Error Investigation

## Error Location
- **File**: `max/python/max/engine/api.py`
- **Line**: 422 (`_model._load(weights_registry_real)`)
- **Error**: `Fatal Python error: Illegal instruction`

## System Information
- **CPU**: KVM virtualized processor
- **CPU Flags**: Limited instruction set (no AVX, AVX2, AVX-512 visible)
- **Architecture**: x86-64

## Root Cause Analysis

The "Illegal instruction" error occurs when the compiled Mojo kernel attempts to execute CPU instructions that are not available on the target system. This typically happens when:

1. **SIMD/Vectorization**: The kernel uses SIMD instructions (SSE, AVX, etc.) that aren't supported
2. **CPU-specific optimizations**: The compiler generates code optimized for newer CPUs
3. **sync_parallelize**: The `sync_parallelize` function in `causal_conv1d_channel_first_fwd_cpu` may use threading primitives that require specific CPU features

## Key Differences: MambaKernelAPI vs MOGGKernelAPI

Both kernels use the same underlying CPU function (`causal_conv1d_channel_first_fwd_cpu`), but:
- **MOGGKernelAPI**: Works (or fails with "Operation overriding error" - a different issue)
- **MambaKernelAPI**: Fails with "Illegal instruction"

This suggests the issue might be:
1. **Compilation flags**: Different compilation settings between packages
2. **Dependencies**: Missing or different dependencies affecting code generation
3. **Registration order**: Kernel registration timing affecting optimization decisions

## Potential Solutions

### 1. Force GPU Execution
If a GPU is available, we could force GPU execution to bypass CPU code generation:
- Modify the kernel to prefer GPU when available
- Check if the test environment has GPU access

### 2. Disable CPU Optimizations
- Add compiler flags to disable aggressive SIMD optimizations
- Use a simpler CPU implementation without `sync_parallelize`

### 3. Check Compilation Flags
- Compare BUILD.bazel files between MambaKernelAPI and MOGGKernelAPI
- Ensure consistent compilation settings

### 4. CPU Feature Detection
- Add runtime CPU feature detection
- Fall back to a simpler implementation if advanced features aren't available

## Next Steps

1. **Verify GPU availability**: Check if GPU execution is possible
2. **Compare BUILD files**: Ensure MambaKernelAPI has the same compilation flags as MOGGKernelAPI
3. **Test with simpler CPU path**: Temporarily disable `sync_parallelize` to test if that's the issue
4. **Check Mojo compiler version**: Ensure consistent compiler versions

## Files to Investigate

- `max/kernels/src/Mogg/MambaKernelAPI/BUILD.bazel`
- `max/kernels/src/Mogg/MOGGKernelAPI/BUILD.bazel`
- `max/kernels/src/nn/causal_conv1d.mojo` (CPU implementation)
- `max/python/max/engine/api.py` (line 422)

