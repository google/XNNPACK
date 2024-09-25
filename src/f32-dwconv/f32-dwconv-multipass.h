// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_DWCONV_UNIPASS(arch, name, first_pass_tile, middle_pass_tile, last_pass_tile, channel_tile, channel_subtile, channel_round, datatype, weights_type, buffer_type,params_type, init_fn)

XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar, 2, 2, 2, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2, 2, 2, 2, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar, 2, 2, 2, 4, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2, 2, 2, 2, 4, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_3f3m3l1c1s1r__scalar, 3, 3, 3, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_3f3m3l1c1s1r__scalar_acc2, 3, 3, 3, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar, 5, 5, 5, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2, 5, 5, 5, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar, 6, 6, 7, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2, 6, 6, 7, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar, 8, 8, 9, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2, 8, 8, 9, 1, 1, 1, float, float, float, struct xnn_f32_default_params, NULL)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_3f3m3l4c4s4r__wasmsimd, 3, 3, 3, 4, 4, 4, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_3f3m3l4c4s4r__wasmsimd_acc2, 3, 3, 3, 4, 4, 4, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_3f3m3l8c4s4r__wasmsimd, 3, 3, 3, 8, 4, 4, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_3f3m3l8c4s4r__wasmsimd_acc2, 3, 3, 3, 8, 4, 4, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd, 5, 5, 5, 4, 4, 4, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2, 5, 5, 5, 4, 4, 4, float, float, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, 5, 5, 5, 4, 4, 4, float, float, float, struct xnn_f32_default_params, NULL)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, 5, 5, 5, 4, 4, 4, float, float, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


