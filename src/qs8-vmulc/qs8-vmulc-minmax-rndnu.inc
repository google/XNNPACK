// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, 8, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_rndnu_neon_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_rndnu_neon_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, 16, false, int8_t, union xnn_qs8_mul_minmax_params, xnn_init_qs8_mul_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

