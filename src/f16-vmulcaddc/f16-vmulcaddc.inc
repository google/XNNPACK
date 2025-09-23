// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_FMA3 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_fma3, xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_fma3, xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, 2, 16, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_FMA3 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_fp16_arith, xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_fp16_arith, xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, 2, 16, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
