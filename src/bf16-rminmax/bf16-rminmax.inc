// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


XNN_UKERNEL(xnn_arch_none, xnn_bf16_rminmax_ukernel__scalar_u1, 1, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rminmax_ukernel__scalar_u2_acc2, 2, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rminmax_ukernel__scalar_u3_acc3, 3, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rminmax_ukernel__scalar_u4_acc2, 4, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_bf16_rminmax_ukernel__scalar_u4_acc4, 4, false, xnn_bfloat16, xnn_bfloat16, struct xnn_bf16_default_params, ((struct xnn_init_bf16_default_params_fn) NULL))
