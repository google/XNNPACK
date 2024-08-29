// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax, params_type, init_params

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8, 8, 8, 9, 8, -16384, 16384, xnn_init_f16_minmax_params_fn, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f16_maxpool_minmax_ukernel_9p8x__f16c_c8, 8, 8, 9, 8, -16384, 16384, xnn_init_f16_minmax_params_fn, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

