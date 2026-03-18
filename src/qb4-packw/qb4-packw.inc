
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, nr, kr, sr, kblock, bl, nr_scale, izp
XNN_QB4_UKERNEL(xnn_arch_none, xnn_qb4_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 32,
                32, 1, 8)
XNN_QB4_UKERNEL(xnn_arch_none, xnn_qb4_packw_gemm_goi_ukernel_x16c4__scalar, 16, 4, 1, 32,
                32, 1, 8)

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_DOTPROD
XNN_QB4_UKERNEL(xnn_arch_arm_neon_dot,
                xnn_qb4_packw_gemm_goi_ukernel_x16c4__aarch64_neondot, 16, 4, 1, 32, 32,
                1, 8)
XNN_QB4_UKERNEL(xnn_arch_arm_neon_dot,
                xnn_qb4_packw_gemm_goi_ukernel_x16c8__aarch64_neondot, 16, 8, 1, 32, 32,
                1, 8)
#endif
