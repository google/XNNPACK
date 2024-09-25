// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_DWCONV_UNIPASS(arch, name, c_block, pipelined, cr, kr, datatype, weights_type,params_type, init_fn)

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_3p8c__neonfp16arith, 8, false, 8, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_3p8c__neonfp16arith_acc2, 8, false, 8, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_3p16c__neonfp16arith, 16, false, 16, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_3p16c__neonfp16arith_acc2, 16, false, 16, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_3p32c__neonfp16arith, 32, false, 32, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_3p32c__neonfp16arith_acc2, 32, false, 32, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_4p8c__neonfp16arith, 8, false, 8, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_4p8c__neonfp16arith_acc2, 8, false, 8, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith, 16, false, 16, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith_acc2, 16, false, 16, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_4p32c__neonfp16arith, 32, false, 32, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_4p32c__neonfp16arith_acc2, 32, false, 32, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_9p8c__neonfp16arith, 8, false, 8, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_9p8c__neonfp16arith_acc2, 8, false, 8, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_9p16c__neonfp16arith, 16, false, 16, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_9p16c__neonfp16arith_acc2, 16, false, 16, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_9p32c__neonfp16arith, 32, false, 32, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_9p32c__neonfp16arith_acc2, 32, false, 32, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith, 8, false, 8, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith_acc2, 8, false, 8, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_25p16c__neonfp16arith, 16, false, 16, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_25p16c__neonfp16arith_acc2, 16, false, 16, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_25p32c__neonfp16arith, 32, false, 32, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv_minmax_ukernel_25p32c__neonfp16arith_acc2, 32, false, 32, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_3p8c__fma3, 8, false, 8, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_3p8c__fma3_acc2, 8, false, 8, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_3p16c__fma3, 16, false, 16, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_3p16c__fma3_acc2, 16, false, 16, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_3p32c__fma3, 32, false, 32, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_3p32c__fma3_acc2, 32, false, 32, 3, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_4p8c__fma3, 8, false, 8, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_4p8c__fma3_acc2, 8, false, 8, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_4p16c__fma3, 16, false, 16, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_4p16c__fma3_acc2, 16, false, 16, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_4p32c__fma3, 32, false, 32, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_4p32c__fma3_acc2, 32, false, 32, 4, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_9p8c__fma3, 8, false, 8, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_9p8c__fma3_acc2, 8, false, 8, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_9p16c__fma3, 16, false, 16, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_9p16c__fma3_acc2, 16, false, 16, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_9p32c__fma3, 32, false, 32, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_9p32c__fma3_acc2, 32, false, 32, 9, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_25p8c__fma3, 8, false, 8, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_25p8c__fma3_acc2, 8, false, 8, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_25p16c__fma3, 16, false, 16, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_25p16c__fma3_acc2, 16, false, 16, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_25p32c__fma3, 32, false, 32, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_fma3, xnn_f16_dwconv_minmax_ukernel_25p32c__fma3_acc2, 32, false, 32, 25, xnn_float16, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


