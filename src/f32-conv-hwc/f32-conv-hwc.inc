// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


XNN_UKERNEL(xnn_arch_none, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__scalar_1x1, 3, 2, 1, 0, 3, 4, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__scalar_1x1, 3, 2, 1, 1, 3, 4, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__neon_2x1, 3, 2, 1, 0, 3, 4, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__neon_2x2, 3, 2, 1, 0, 3, 4, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x8__neon_2x1, 3, 2, 1, 0, 3, 8, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x8__neon_2x2, 3, 2, 1, 0, 3, 8, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__neon_2x1, 3, 2, 1, 1, 3, 4, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__neon_2x2, 3, 2, 1, 1, 3, 4, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__neon_2x1, 3, 2, 1, 1, 3, 8, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__neon_2x2, 3, 2, 1, 1, 3, 8, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__aarch64_neonfma_2x1, 3, 2, 1, 0, 3, 4, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__aarch64_neonfma_2x2, 3, 2, 1, 0, 3, 4, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x8__aarch64_neonfma_2x1, 3, 2, 1, 0, 3, 8, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x8__aarch64_neonfma_2x2, 3, 2, 1, 0, 3, 8, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x1, 3, 2, 1, 1, 3, 4, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, 3, 2, 1, 1, 3, 4, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__aarch64_neonfma_2x1, 3, 2, 1, 1, 3, 8, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__aarch64_neonfma_2x2, 3, 2, 1, 1, 3, 8, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM64
