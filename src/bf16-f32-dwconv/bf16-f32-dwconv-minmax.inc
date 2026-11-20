// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// Arguments are:
// XNN_UKERNEL(arch, name, c_block, pipelined, cr, kr, datatype, weights_type,params_type, init_fn)
XNN_UKERNEL(0, xnn_bf16_f32_dwconv_minmax_ukernel_3p2c__scalar, 2, false, 2, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(0, xnn_bf16_f32_dwconv_minmax_ukernel_4p2c__scalar, 2, false, 2, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(0, xnn_bf16_f32_dwconv_minmax_ukernel_9p2c__scalar, 2, false, 2, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(0, xnn_bf16_f32_dwconv_minmax_ukernel_25p2c__scalar_acc2, 2, false, 2, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)



