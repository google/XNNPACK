// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, channel_tile, primary_tile, datatype, params_type, init_params

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_maxpool_minmax_ukernel_9p__neonfp16arith_u8, 8, 9, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f16_maxpool_minmax_ukernel_9p__sse41_u8, 8, 9, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f16_maxpool_minmax_ukernel_9p__avx2_u16, 16, 9, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


