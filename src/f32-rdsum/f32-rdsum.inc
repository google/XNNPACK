// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

XNN_UKERNEL(0, xnn_f32_rdsum_ukernel_7p7x__scalar_c4, 7, 4, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rdsum_ukernel_7p7x__neon_c16, 7, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rdsum_ukernel_7p7x__neon_c32, 7, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rdsum_ukernel_7p7x__neon_c64, 7, 64, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, 7, 1, true, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, 7, 2, true, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, 7, 4, true, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(0, xnn_f32_rdsum_ukernel_7p7x__sse2_c16, 7, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(0, xnn_f32_rdsum_ukernel_7p7x__sse2_c32, 7, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(0, xnn_f32_rdsum_ukernel_7p7x__sse2_c64, 7, 64, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rdsum_ukernel_7p7x__avx_c16, 7, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rdsum_ukernel_7p7x__avx_c32, 7, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rdsum_ukernel_7p7x__avx_c64, 7, 64, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, 7, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, 7, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, 7, 64, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rdsum_ukernel_7p7x__avx512f_c128, 7, 128, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, 7, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(0, xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, 7, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(0, xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, 7, 64, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

