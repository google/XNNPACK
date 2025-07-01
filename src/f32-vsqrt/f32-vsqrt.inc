// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)

XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__neon_rsqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__neon_rsqrt_u8, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsqrt_ukernel__neon_rsqrt_u16, 4, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v, 1, true, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v, 2, true, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v, 4, true, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v, 8, true, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__sse2_sqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__sse2_sqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__sse2_sqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__sse2_rsqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__sse2_rsqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__sse2_rsqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_sqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_sqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_sqrt_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_rsqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_rsqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsqrt_ukernel__avx_rsqrt_u32, 32, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48, 48, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__scalar_sqrt_u1, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__scalar_sqrt_u2, 2, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsqrt_ukernel__scalar_sqrt_u4, 4, false, float, struct xnn_f32_default_params, NULL)
