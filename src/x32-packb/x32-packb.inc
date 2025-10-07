// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// Arguments are:
// XNN_UKERNEL(arch, ukernel, channel_tile, channel_subtile, channel_round)

XNN_UKERNEL(xnn_arch_none, xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_float, 2, 1, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int, 2, 1, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_float, 2, 2, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_int, 2, 2, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float, 4, 1, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_int, 4, 1, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_float, 4, 4, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int, 4, 4, 1)


