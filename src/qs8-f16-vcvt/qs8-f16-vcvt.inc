// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#ifndef XNN_QUANTIZED
#define XNN_QUANTIZED(T) T
#define XNN_DEFINED_QUANTIZED
#endif


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8, 8, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16, 16, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24, 24, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32, 32, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_f16_vcvt_ukernel__avx2_u16, 16, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_f16_vcvt_ukernel__avx2_u24, 24, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_f16_vcvt_ukernel__avx2_u32, 32, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_f16_vcvt_ukernel__avx2_u64, 64, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif

