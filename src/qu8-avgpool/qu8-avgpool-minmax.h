// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, requantize_fn, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_MULTIPASS(xnn_arch_arm_neon, xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_qu8_requantize_fp32, 8, 8, 9, 8, xnn_init_qu8_avgpool_minmax_fp32_scalar_params)
XNN_UKERNEL_UNIPASS(xnn_arch_arm_neon, xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_qu8_requantize_fp32, 8, 8, 9, 0, xnn_init_qu8_avgpool_minmax_fp32_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_MULTIPASS(0, xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_qu8_requantize_fp32, 8, 8, 9, 8, xnn_init_qu8_avgpool_minmax_fp32_scalar_params)
XNN_UKERNEL_UNIPASS(0, xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_qu8_requantize_fp32, 8, 8, 9, 0, xnn_init_qu8_avgpool_minmax_fp32_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

XNN_UKERNEL_MULTIPASS(0, xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_qu8_requantize_fp32, 1, 1, 9, 8, xnn_init_qu8_avgpool_minmax_fp32_scalar_params)
XNN_UKERNEL_UNIPASS(0, xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_qu8_requantize_fp32, 1, 1, 9, 0, xnn_init_qu8_avgpool_minmax_fp32_scalar_params)

