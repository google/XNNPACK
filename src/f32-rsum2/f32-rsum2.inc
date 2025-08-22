// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rsum2_ukernel__neon_u4, 4, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rsum2_ukernel__neon_u8, 8, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rsum2_ukernel__neon_u8_acc2, 8, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rsum2_ukernel__neon_u12, 12, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rsum2_ukernel__neon_u12_acc3, 12, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rsum2_ukernel__neon_u16, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rsum2_ukernel__neon_u16_acc2, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rsum2_ukernel__neon_u16_acc4, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__sse2_u4, 4, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__sse2_u8, 8, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__sse2_u8_acc2, 8, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__sse2_u12, 12, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__sse2_u12_acc3, 12, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__sse2_u16, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__sse2_u16_acc2, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__sse2_u16_acc4, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rsum2_ukernel__avx_u8, 8, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rsum2_ukernel__avx_u16, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rsum2_ukernel__avx_u16_acc2, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rsum2_ukernel__avx_u24, 24, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rsum2_ukernel__avx_u24_acc3, 24, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rsum2_ukernel__avx_u32, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rsum2_ukernel__avx_u32_acc2, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rsum2_ukernel__avx_u32_acc4, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rsum2_ukernel__avx512f_u16, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rsum2_ukernel__avx512f_u32, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rsum2_ukernel__avx512f_u32_acc2, 32, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rsum2_ukernel__avx512f_u48, 48, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rsum2_ukernel__avx512f_u48_acc3, 48, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rsum2_ukernel__avx512f_u64, 64, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rsum2_ukernel__avx512f_u64_acc2, 64, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rsum2_ukernel__avx512f_u64_acc4, 64, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__wasmsimd_u4, 4, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__wasmsimd_u8, 8, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__wasmsimd_u8_acc2, 8, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__wasmsimd_u12, 12, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__wasmsimd_u12_acc3, 12, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__wasmsimd_u16, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__wasmsimd_u16_acc2, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__wasmsimd_u16_acc4, 16, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__scalar_u1, 1, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__scalar_u2, 2, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__scalar_u2_acc2, 2, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__scalar_u3, 3, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__scalar_u3_acc3, 3, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__scalar_u4, 4, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__scalar_u4_acc2, 4, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_rsum2_ukernel__scalar_u4_acc4, 4, false, float, float, struct xnn_f32_scale_params, xnn_init_f32_scale_scalar_params)
