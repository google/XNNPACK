// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#ifndef XNN_QUANTIZED
#define XNN_QUANTIZED(T) T
#define XNN_DEFINED_QUANTIZED
#endif

XNN_UKERNEL(xnn_arch_none, xnn_f16_qu8_vcvt_ukernel__scalar_imagic_u1, 1, false, xnn_float16, XNN_QUANTIZED(uint8_t), struct xnn_f16_qu8_cvt_params, xnn_init_f16_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qu8_vcvt_ukernel__scalar_imagic_u2, 2, false, xnn_float16, XNN_QUANTIZED(uint8_t), struct xnn_f16_qu8_cvt_params, xnn_init_f16_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qu8_vcvt_ukernel__scalar_imagic_u3, 3, false, xnn_float16, XNN_QUANTIZED(uint8_t), struct xnn_f16_qu8_cvt_params, xnn_init_f16_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f16_qu8_vcvt_ukernel__scalar_imagic_u4, 4, false, xnn_float16, XNN_QUANTIZED(uint8_t), struct xnn_f16_qu8_cvt_params, xnn_init_f16_qu8_cvt_scalar_params)


#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif
