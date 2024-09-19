// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_DWCONV_UNIPASS(arch, name, c_block, pipelined, cr, kr, datatype, weights_type,params_type, init_fn)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_3p4c__wasmsimd, 4, false, 4, 3, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_3p8c__wasmsimd, 8, false, 8, 3, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_4p4c__wasmsimd, 4, false, 4, 4, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_4p8c__wasmsimd, 8, false, 8, 4, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p4c__wasmsimd, 4, false, 4, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2, 4, false, 4, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p8c__wasmsimd, 8, false, 8, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2, 8, false, 8, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_25p4c__wasmsimd, 4, false, 4, 25, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_25p8c__wasmsimd, 8, false, 8, 25, float, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma, 4, false, 4, 3, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma, 8, false, 8, 3, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma, 4, false, 4, 4, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma, 8, false, 8, 4, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma, 4, false, 4, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma, 8, false, 8, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma, 4, false, 4, 25, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma, 8, false, 8, 25, float, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_3p1c__scalar, 1, false, 1, 3, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_3p1c__scalar_acc2, 1, false, 1, 3, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_3p2c__scalar, 2, false, 2, 3, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_3p2c__scalar_acc2, 2, false, 2, 3, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_4p1c__scalar, 1, false, 1, 4, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_4p1c__scalar_acc2, 1, false, 1, 4, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_4p2c__scalar, 2, false, 2, 4, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_4p2c__scalar_acc2, 2, false, 2, 4, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p1c__scalar, 1, false, 1, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p1c__scalar_acc2, 1, false, 1, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p2c__scalar, 2, false, 2, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_9p2c__scalar_acc2, 2, false, 2, 9, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_25p1c__scalar, 1, false, 1, 25, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_25p1c__scalar_acc2, 1, false, 1, 25, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_25p2c__scalar, 2, false, 2, 25, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_UNIPASS(0, xnn_f32_dwconv_ukernel_25p2c__scalar_acc2, 2, false, 2, 25, float, float, struct xnn_f32_default_params, NULL)

