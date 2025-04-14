
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, nr, kr, sr, kblock, bl, nr_scale, izp
XNN_QB4_UKERNEL(0, xnn_qb4_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 32,
                32, 1, 8)
XNN_QB4_UKERNEL(0, xnn_qb4_packw_gemm_goi_ukernel_x16c4__scalar, 16, 4, 1, 32,
                32, 1, 8)
