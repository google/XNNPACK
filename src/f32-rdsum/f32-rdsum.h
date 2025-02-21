// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, datatype, output_type, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

//SCALAR
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdsum_ukernel_7p7x__scalar_c4, 7, 7, 4, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_rdsum_ukernel_7p7x__neon_c16, 7, 7, 16, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_rdsum_ukernel_7p7x__neon_c32, 7, 7, 32, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_rdsum_ukernel_7p7x__neon_c64, 7, 7, 64, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, 7, 7, 1, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, 7, 7, 2, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, 7, 7, 4, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdsum_ukernel_7p7x__sse_c16, 7, 7, 16, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdsum_ukernel_7p7x__sse_c32, 7, 7, 32, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdsum_ukernel_7p7x__sse_c64, 7, 7, 64, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_rdsum_ukernel_7p7x__avx_c16, 7, 7, 16, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_rdsum_ukernel_7p7x__avx_c32, 7, 7, 32, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_rdsum_ukernel_7p7x__avx_c64, 7, 7, 64, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, 7, 7, 16, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, 7, 7, 32, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, 7, 7, 64, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, 7, 7, 16, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, 7, 7, 32, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, 7, 7, 64, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
