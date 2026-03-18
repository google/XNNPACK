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
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8, 8, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16, 16, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24, 24, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32, 32, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64, 64, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

XNN_UKERNEL(xnn_arch_none, xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u1, 1, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2, 2, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3, 3, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4, 4, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u1, 1, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2, 2, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3, 3, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4, 4, false, xnn_float16, XNN_QUANTIZED(int8_t), struct xnn_f16_qs8_cvt_params, xnn_init_f16_qs8_cvt_scalar_params)


#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif
