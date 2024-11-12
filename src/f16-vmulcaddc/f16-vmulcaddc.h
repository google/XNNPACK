// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, channel_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, row_tile, channel_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, row_tile, channel_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, channel_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_X86, xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, 2, 8, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_X86, xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, 2, 16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, 2, 28, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, 2, 16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
