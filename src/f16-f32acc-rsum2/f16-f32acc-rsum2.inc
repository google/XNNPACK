// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u8, 8, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u16, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u24, 24, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u32, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u8_acc2, 8, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u16_acc2, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u24_acc3, 24, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u32_acc2, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u16_acc4, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u32_acc4, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u24_acc6, 24, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u32_acc8, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_F16C && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rsum2_ukernel__f16c_u8, 8, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rsum2_ukernel__f16c_u16, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rsum2_ukernel__f16c_u24, 24, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rsum2_ukernel__f16c_u32, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rsum2_ukernel__f16c_u16_acc2, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rsum2_ukernel__f16c_u24_acc3, 24, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rsum2_ukernel__f16c_u32_acc2, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rsum2_ukernel__f16c_u32_acc4, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
#endif  // XNN_ENABLE_F16C && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u16, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u32, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u48, 48, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u64, 64, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u128, 128, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u128_acc2, 128, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u32_acc2, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u48_acc3, 48, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u64_acc2, 64, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u64_acc4, 64, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rsum2_ukernel__avx512skx_u128_acc4, 128, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

