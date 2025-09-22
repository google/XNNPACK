// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, 8, false, xnn_float16, struct xnn_f16_elu_params, xnn_init_f16_elu_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, 16, false, xnn_float16, struct xnn_f16_elu_params, xnn_init_f16_elu_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f16_velu_ukernel__avx2_rr1_p3_u8, 8, false, xnn_float16, struct xnn_f16_elu_params, xnn_init_f16_elu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f16_velu_ukernel__avx2_rr1_p3_u16, 16, false, xnn_float16, struct xnn_f16_elu_params, xnn_init_f16_elu_scalar_params)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

