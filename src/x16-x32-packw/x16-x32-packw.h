// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, nr, kr, sr, kblock, nr_scale
XNN_UKERNEL(0, xnn_x16_x32_packw_gemm_goi_ukernel_x32c2__scalar, 32, 2, 1, 32, 1)
XNN_GIO_UKERNEL(0, xnn_x16_x32_packw_gemm_gio_ukernel_x32c2__scalar, 32, 2, 1, 32, 1)
