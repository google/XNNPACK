// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rdsum2_ukernel_7p7x__neonfp16arith_u16, 7, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rdsum2_ukernel_7p7x__neonfp16arith_u32, 7, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_f32acc_rdsum2_ukernel_7p7x__neonfp16arith_u64, 7, 64, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_F16C && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rdsum2_ukernel_7p7x__f16c_u16, 7, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rdsum2_ukernel_7p7x__f16c_u32, 7, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rdsum2_ukernel_7p7x__f16c_u64, 7, 64, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32acc_rdsum2_ukernel_7p7x__f16c_u128, 7, 128, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
#endif  // XNN_ENABLE_F16C && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rdsum2_ukernel_7p7x__avx512skx_u16, 7, 16, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rdsum2_ukernel_7p7x__avx512skx_u32, 7, 32, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rdsum2_ukernel_7p7x__avx512skx_u64, 7, 64, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32acc_rdsum2_ukernel_7p7x__avx512skx_u128, 7, 128, false, xnn_float16, float, struct xnn_f16_f32acc_scale_params, xnn_init_f16_f32acc_scale_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

