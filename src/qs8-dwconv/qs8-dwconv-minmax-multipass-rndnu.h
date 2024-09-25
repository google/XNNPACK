// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_DWCONV_UNIPASS(arch, name, first_pass_tile, middle_pass_tile, last_pass_tile, channel_tile, channel_subtile, channel_round, datatype, weights_type, buffer_type,params_type, init_fn)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l8c8s8r__neon_mla8_ld64, 5, 5, 5, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l8c8s8r__neon_mul8_ld64, 5, 5, 5, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l8c8s8r__neon_mul16, 5, 5, 5, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mla8_ld64, 5, 5, 5, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mla8_ld128, 5, 5, 5, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mul8_ld64, 5, 5, 5, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mul8_ld128, 5, 5, 5, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mul16, 5, 5, 5, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l32c8s8r__neon_mul16, 5, 5, 5, 32, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l8c8s8r__neon_mla8_ld64, 6, 6, 7, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l8c8s8r__neon_mul8_ld64, 6, 6, 7, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l8c8s8r__neon_mul16, 6, 6, 7, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mla8_ld64, 6, 6, 7, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mla8_ld128, 6, 6, 7, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mul8_ld64, 6, 6, 7, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mul8_ld128, 6, 6, 7, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mul16, 6, 6, 7, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l32c8s8r__neon_mul16, 6, 6, 7, 32, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l8c8s8r__neon_mla8_ld64, 8, 8, 9, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l8c8s8r__neon_mul8_ld64, 8, 8, 9, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l8c8s8r__neon_mul16, 8, 8, 9, 8, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mla8_ld64, 8, 8, 9, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mla8_ld128, 8, 8, 9, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mul8_ld64, 8, 8, 9, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mul8_ld128, 8, 8, 9, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mul16, 8, 8, 9, 16, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l32c8s8r__neon_mul16, 8, 8, 9, 32, 8, 8, int8_t, void, int32_t, union xnn_qs8_conv_minmax_params, xnn_init_qs8_conv_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


