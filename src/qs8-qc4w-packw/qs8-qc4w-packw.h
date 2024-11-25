// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, nr, kr, sr, kblock, nr_scale
XNN_UKERNEL(0, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 8, 1)
XNN_UKERNEL(0, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 8, 1)
XNN_UKERNEL(0, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x32c8__scalar, 32, 8, 1, 8, 1)
