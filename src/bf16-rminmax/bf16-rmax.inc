// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_rmax_ukernel__neon_u8, 8, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_rmax_ukernel__neon_u16_acc2, 16, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_rmax_ukernel__neon_u24_acc3, 24, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_rmax_ukernel__neon_u32_acc2, 32, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_rmax_ukernel__neon_u32_acc4, 32, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rmax_ukernel__scalar_u1, 1, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rmax_ukernel__scalar_u2_acc2, 2, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rmax_ukernel__scalar_u3_acc3, 3, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rmax_ukernel__scalar_u4_acc2, 4, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rmax_ukernel__scalar_u4_acc4, 4, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
