// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_DWCONV_UNIPASS(arch, name, first_pass_tile, middle_pass_tile, last_pass_tile, channel_tile, channel_subtile, channel_round, datatype, weights_type, buffer_type,params_type, init_fn)

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__neonfp16arith, 5, 5, 5, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__neonfp16arith_acc2, 5, 5, 5, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__neonfp16arith, 5, 5, 5, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__neonfp16arith_acc2, 5, 5, 5, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith, 5, 5, 5, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith_acc2, 5, 5, 5, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__neonfp16arith, 6, 6, 7, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__neonfp16arith_acc2, 6, 6, 7, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__neonfp16arith, 6, 6, 7, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__neonfp16arith_acc2, 6, 6, 7, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__neonfp16arith, 6, 6, 7, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__neonfp16arith_acc2, 6, 6, 7, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__neonfp16arith, 8, 8, 9, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__neonfp16arith_acc2, 8, 8, 9, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__neonfp16arith, 8, 8, 9, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__neonfp16arith_acc2, 8, 8, 9, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__neonfp16arith, 8, 8, 9, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__neonfp16arith_acc2, 8, 8, 9, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, 5, 5, 5, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3_acc2, 5, 5, 5, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, 5, 5, 5, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3_acc2, 5, 5, 5, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, 5, 5, 5, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3_acc2, 5, 5, 5, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__fma3, 6, 6, 7, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__fma3_acc2, 6, 6, 7, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__fma3, 6, 6, 7, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__fma3_acc2, 6, 6, 7, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__fma3, 6, 6, 7, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__fma3_acc2, 6, 6, 7, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3, 8, 8, 9, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3_acc2, 8, 8, 9, 8, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__fma3, 8, 8, 9, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__fma3_acc2, 8, 8, 9, 16, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3, 8, 8, 9, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3_acc2, 8, 8, 9, 32, 8, 4, xnn_float16, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


