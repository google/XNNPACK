// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_CVT_UKERNEL_WITH_PARAMS
#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out, params_type, init_params) \
    XNN_CVT_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out)
#define XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_CVT_UKERNEL
#define XNN_CVT_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out) \
    XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out, void, /*init_params=*/nullptr)
#define XNN_DEFINED_CVT_UKERNEL
#endif

#ifndef XNN_QUANTIZED
#define XNN_QUANTIZED(T) T
#define XNN_DEFINED_QUANTIZED
#endif


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8, 8, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16, 16, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24, 24, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32, 32, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_f16_vcvt_ukernel__avx2_u16, 16, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_f16_vcvt_ukernel__avx2_u24, 24, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_f16_vcvt_ukernel__avx2_u32, 32, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_f16_vcvt_ukernel__avx2_u64, 64, false, XNN_QUANTIZED(int8_t), xnn_float16, struct xnn_qs8_f16_cvt_params, xnn_init_qs8_f16_cvt_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifdef XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#undef XNN_CVT_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_CVT_UKERNEL
#undef XNN_DEFINED_CVT_UKERNEL
#undef XNN_CVT_UKERNEL
#endif

#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif

