// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_QUANTIZED
#define XNN_QUANTIZED(T) T
#define XNN_DEFINED_QUANTIZED
#endif

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_qs8_vcvt_ukernel__avx512skx_u32, 32, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_qs8_vcvt_ukernel__avx512skx_u64, 64, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_qs8_vcvt_ukernel__avx512skx_u96, 96, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_qs8_vcvt_ukernel__avx512skx_u128, 128, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon_v8, xnn_bf16_qs8_vcvt_ukernel__neonv8_u8, 8, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_v8, xnn_bf16_qs8_vcvt_ukernel__neonv8_u16, 16, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_v8, xnn_bf16_qs8_vcvt_ukernel__neonv8_u24, 24, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_v8, xnn_bf16_qs8_vcvt_ukernel__neonv8_u32, 32, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

XNN_UKERNEL(xnn_arch_none, xnn_bf16_qs8_vcvt_ukernel__scalar_imagic_u1, 1, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_bf16_qs8_vcvt_ukernel__scalar_imagic_u2, 2, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_bf16_qs8_vcvt_ukernel__scalar_imagic_u3, 3, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_bf16_qs8_vcvt_ukernel__scalar_imagic_u4, 4, false, xnn_bfloat16, XNN_QUANTIZED(int8_t), struct xnn_bf16_qs8_cvt_params, xnn_init_bf16_qs8_cvt_scalar_params)


#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif
