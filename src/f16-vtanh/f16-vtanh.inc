// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// Direct FP16
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__scalar_rational_5_4_div_u1, 1, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__scalar_rational_5_4_div_u2, 2, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__scalar_rational_5_4_div_u4, 4, false, xnn_float16, struct xnn_f16_default_params, NULL)

#if XNN_ARCH_WASMRELAXEDSIMDFP16
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_div_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_div_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_div_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_nr_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_nr_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_nr_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMDFP16

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vtanh_ukernel__neonfp16arith_rational_5_4_div_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vtanh_ukernel__neonfp16arith_rational_5_4_div_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vtanh_ukernel__neonfp16arith_rational_5_4_div_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vtanh_ukernel__neonfp16arith_rational_5_4_nr_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vtanh_ukernel__neonfp16arith_rational_5_4_nr_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vtanh_ukernel__neonfp16arith_rational_5_4_nr_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if (XNN_ARCH_X86 || XNN_ARCH_X86_64) && XNN_ENABLE_AVX512FP16
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vtanh_ukernel__avx512fp16_rational_5_4_div_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vtanh_ukernel__avx512fp16_rational_5_4_div_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vtanh_ukernel__avx512fp16_rational_5_4_div_u128, 128, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vtanh_ukernel__avx512fp16_rational_5_4_nr_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vtanh_ukernel__avx512fp16_rational_5_4_nr_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vtanh_ukernel__avx512fp16_rational_5_4_nr_u128, 128, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // (XNN_ARCH_X86 || XNN_ARCH_X86_64) && XNN_ENABLE_AVX512FP16

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_FP16_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_div_u1v, 1, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_div_u2v, 2, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_div_u4v, 4, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_div_u8v, 8, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_nr_u1v, 1, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_nr_u2v, 2, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_nr_u4v, 4, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vtanh_ukernel__rvvfp16arith_rational_5_4_nr_u8v, 8, true, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_FP16_VECTOR

// FP32 Accumulation
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__scalar_rational_5_4_div_u1, 1, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__scalar_rational_5_4_div_u2, 2, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__scalar_rational_5_4_div_u4, 4, false, xnn_float16, struct xnn_f16_default_params, NULL)

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_div_u4, 4, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_div_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_div_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_nr_u4, 4, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_nr_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32acc_vtanh_ukernel__wasmrelaxedsimd_rational_5_4_nr_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon_fp16, xnn_f16_f32acc_vtanh_ukernel__neonfp16_rational_5_4_div_u4, 4, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16, xnn_f16_f32acc_vtanh_ukernel__neonfp16_rational_5_4_div_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16, xnn_f16_f32acc_vtanh_ukernel__neonfp16_rational_5_4_div_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16, xnn_f16_f32acc_vtanh_ukernel__neonfp16_rational_5_4_nr_u4, 4, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16, xnn_f16_f32acc_vtanh_ukernel__neonfp16_rational_5_4_nr_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16, xnn_f16_f32acc_vtanh_ukernel__neonfp16_rational_5_4_nr_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64



#if XNN_ENABLE_F16C && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_vtanh_ukernel__f16c_rational_5_4_div_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_vtanh_ukernel__f16c_rational_5_4_div_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_vtanh_ukernel__f16c_rational_5_4_div_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_vtanh_ukernel__f16c_rational_5_4_nr_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_vtanh_ukernel__f16c_rational_5_4_nr_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_vtanh_ukernel__f16c_rational_5_4_nr_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_F16C && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f16_f32acc_vtanh_ukernel__avx512f_rational_5_4_div_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f16_f32acc_vtanh_ukernel__avx512f_rational_5_4_div_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f16_f32acc_vtanh_ukernel__avx512f_rational_5_4_div_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f16_f32acc_vtanh_ukernel__avx512f_rational_5_4_nr_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f16_f32acc_vtanh_ukernel__avx512f_rational_5_4_nr_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f16_f32acc_vtanh_ukernel__avx512f_rational_5_4_nr_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
XNN_UKERNEL(xnn_arch_hvx, xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_div_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_hvx, xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_div_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_hvx, xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_div_u128, 128, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_hvx, xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_nr_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_hvx, xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_nr_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_hvx, xnn_f16_f32acc_vtanh_ukernel__hvx_rational_5_4_nr_u128, 128, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
