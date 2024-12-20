// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, pixel_tile, datatype, weight_type, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, channel_tile, pixel_tile, datatype, weight_type, params_type, init_params)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, pixel_tile, datatype, weight_type, params_type, init_params) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, pixel_tile, datatype, weight_type, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_fp16_arith, xnn_f16_ibilinear_ukernel__neonfp16arith_c8, 8, 1, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_fp16_arith, xnn_f16_ibilinear_ukernel__neonfp16arith_c16, 16, 1, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f16_ibilinear_ukernel__fma3_c8, 8, 1, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f16_ibilinear_ukernel__fma3_c16, 16, 1, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif
#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
