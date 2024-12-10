// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params) \
    XNN_UKERNEL_UNIPASS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL_UNIPASS
#define XNN_UKERNEL_UNIPASS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

//SCALAR
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_4x__scalar_c1, 4, 0, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_9x__scalar_c1, 9, 0, 1, false, float, struct xnn_f32_default_params, NULL)

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_argmaxpool_ukernel_4x__neon_c4, 4, 0, 4,false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_argmaxpool_ukernel_9x__neon_c4, 9, 0, 4, false,float, struct xnn_f32_default_params, NULL)

#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_4x__sse2_c4, 4, 0, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_9x__sse2_c4,  9, 0, 4, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_4x__wasmsimd_c4, 4, 0, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_argmaxpool_ukernel_9x__wasmsimd_c4, 9, 0, 4, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_RISCV && (XNN_ENABLE_RISCV_VECTOR)

XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_argmaxpool_ukernel_4x__rvv_u1v,  4, 0, 1, true, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_argmaxpool_ukernel_9x__rvv_u1v, 9, 0, 1, true, float, struct xnn_f32_default_params, NULL)

#endif // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL_UNIPASS
#endif
