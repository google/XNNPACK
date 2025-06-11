// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vmul_minmax_fp32_ukernel__neon_ld64_u8, 8, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vmul_minmax_fp32_ukernel__neon_ld64_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vmul_minmax_fp32_ukernel__neon_ld128_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_v8, xnn_qs8_vmul_minmax_fp32_ukernel__neonv8_ld64_u8, 8, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_v8, xnn_qs8_vmul_minmax_fp32_ukernel__neonv8_ld64_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_v8, xnn_qs8_vmul_minmax_fp32_ukernel__neonv8_ld128_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(0, xnn_qs8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u8, 8, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_u8, 8, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_u8, 8, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_qs8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, 8, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_vmul_minmax_fp32_ukernel__rvv_u1v, 1, true, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_vmul_minmax_fp32_ukernel__rvv_u2v, 2, true, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

XNN_UKERNEL(0, xnn_qs8_vmul_minmax_fp32_ukernel__scalar_u1, 1, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vmul_minmax_fp32_ukernel__scalar_u2, 2, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vmul_minmax_fp32_ukernel__scalar_u4, 4, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_scalar_params)
