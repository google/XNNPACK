// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params) \
    XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL_MULTIPASS
#define XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

//SCALAR
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, 9, 8, 1, false, float, struct xnn_f32_default_params, NULL)


#if (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_argmaxpool_ukernel_9p8x__neon_c4, 9, 8, 4,false, float, struct xnn_f32_default_params, NULL)


#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4, 9, 8, 4, false, float, struct xnn_f32_default_params, NULL)

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_9p8x__wasmsimd_c4, 9, 8, 4, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_RISCV && (XNN_ENABLE_RISCV_VECTOR)

XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_argmaxpool_ukernel_9p8x__rvv_u1v,  9, 8, 1, true, float, struct xnn_f32_default_params, NULL)
#endif // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL_MULTIPASS
#endif
