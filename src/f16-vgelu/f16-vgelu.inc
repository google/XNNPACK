// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

XNN_UKERNEL(xnn_arch_none, xnn_f16_vgelu_ukernel__scalar_rational_6_4_div_u1, 1, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vgelu_ukernel__scalar_rational_6_4_div_u2, 2, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vgelu_ukernel__scalar_rational_6_4_div_u4, 4, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vgelu_ukernel__scalar_rational_6_4_div_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vgelu_ukernel__neonfp16arith_rational_6_4_div_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vgelu_ukernel__neonfp16arith_rational_6_4_div_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vgelu_ukernel__neonfp16arith_rational_6_4_div_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vgelu_ukernel__avx512fp16_rational_6_4_div_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vgelu_ukernel__avx512fp16_rational_6_4_div_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_vgelu_ukernel__avx512fp16_rational_6_4_div_u96, 96, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
