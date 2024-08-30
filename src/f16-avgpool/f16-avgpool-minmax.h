// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, 8, 8, 9, 8, xnn_init_f16_scaleminmax_scalar_params)
XNN_UKERNEL_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, 8, 8, 9, 0, xnn_init_f16_scaleminmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_MULTIPASS(xnn_arch_x86_f16c, xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, 8, 8, 9, 8, xnn_init_f16_scaleminmax_scalar_params)
XNN_UKERNEL_UNIPASS(xnn_arch_x86_f16c, xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, 8, 8, 9, 0, xnn_init_f16_scaleminmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
