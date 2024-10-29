// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif


#if XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v, 1, true, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v, 2, true, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v, 4, true, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v, 8, true, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__sse_sqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__sse_sqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__sse_sqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__sse_rsqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__sse_rsqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__sse_rsqrt_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_sqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_sqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_sqrt_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_rsqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_rsqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_rsqrt_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f32_vsqrt_ukernel__fma3_rsqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32, 32, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48, 48, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__scalar_sqrt_u1, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__scalar_sqrt_u2, 2, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vsqrt_ukernel__scalar_sqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
