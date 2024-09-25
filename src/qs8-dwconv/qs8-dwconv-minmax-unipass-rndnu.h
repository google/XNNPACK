// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_DWCONV_UNIPASS(arch, name, c_block, pipelined, cr, kr, datatype, weights_type,params_type, init_fn)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, 8, false, 8, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, 8, false, 8, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, 8, false, 8, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, 16, false, 16, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, 16, false, 16, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, 16, false, 16, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, 16, false, 16, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, 16, false, 16, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, 32, false, 32, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, 8, false, 8, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, 8, false, 8, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, 8, false, 8, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, 16, false, 16, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, 16, false, 16, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, 16, false, 16, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, 16, false, 16, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, 16, false, 16, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, 32, false, 32, 25, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

XNN_DWCONV_UNIPASS(0, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, 1, false, 1, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, 2, false, 2, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, 4, false, 4, 9, int8_t, void, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_scalar_params)

