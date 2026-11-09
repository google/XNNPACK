// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_vadd_ukernel__neon_u8, 8, false, xnn_bfloat16, struct xnn_bf16_default_params, ((xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_vadd_ukernel__neon_u16, 16, false, xnn_bfloat16, struct xnn_bf16_default_params, ((xnn_init_bf16_default_params_fn) NULL))
#endif  // XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_BF16 && XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_vadd_ukernel__neonbf16_u8, 8, false, xnn_bfloat16, struct xnn_bf16_default_params, ((xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_vadd_ukernel__neonbf16_u16, 16, false, xnn_bfloat16, struct xnn_bf16_default_params, ((xnn_init_bf16_default_params_fn) NULL))
#endif  // XNN_ENABLE_ARM_BF16 && XNN_ARCH_ARM64
