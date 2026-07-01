// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

// Arguments are:
// XNN_QD8_UKERNEL(arch, ukernel, nr, kr, sr, kblock, nr_scale, izp)
// XNN_QS8_UKERNEL(arch, ukernel, nr, kr, sr, kblock, nr_scale, izp)

XNN_QD8_UKERNEL(xnn_arch_none, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x2c8__scalar, 2, 8, 1, 32, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_none, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x4c8__scalar, 4, 8, 1, 32, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_none, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 32, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_none, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 32, 1, 0)

XNN_QD8_UKERNEL(xnn_arch_none, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x2c4__scalar, 2, 4, 1, 16, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_none, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x4c4__scalar, 4, 4, 1, 16, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_none, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x8c4__scalar, 8, 4, 1, 16, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_none, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x16c4__scalar, 16, 4, 1, 16, 1, 0)

XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x2c8__scalar, 2, 8, 1, 32, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x4c8__scalar, 4, 8, 1, 32, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 32, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 32, 1, 128)

XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x2c4__scalar, 2, 4, 1, 16, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x4c4__scalar, 4, 4, 1, 16, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x8c4__scalar, 8, 4, 1, 16, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x16c4__scalar, 16, 4, 1, 16, 1, 128)

XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x2c8__scalar, 2, 8, 1, 32, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x4c8__scalar, 4, 8, 1, 32, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 32, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 32, 1, 0)

XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x2c4__scalar, 2, 4, 1, 16, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x4c4__scalar, 4, 4, 1, 16, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x8c4__scalar, 8, 4, 1, 16, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_none, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x16c4__scalar, 16, 4, 1, 16, 1, 0)

#if XNN_ENABLE_SSSE3 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_QD8_UKERNEL(xnn_arch_x86_ssse3, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x4c8__ssse3, 4, 8, 1, 32, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_x86_ssse3, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x8c8__ssse3, 8, 8, 1, 32, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_ssse3, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x4c8__ssse3, 4, 8, 1, 32, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_x86_ssse3, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x8c8__ssse3, 8, 8, 1, 32, 1, 128)
#endif

#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_QD8_UKERNEL(xnn_arch_x86_avx2, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x8c8__avx2, 8, 8, 1, 32, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_x86_avx2, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x16c8__avx2, 16, 8, 1, 32, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x8c8__avx2, 8, 8, 1, 32, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x16c8__avx2, 16, 8, 1, 32, 1, 128)
#endif

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_QD8_UKERNEL(xnn_arch_arm_neon_dot, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x8c4__neondot, 8, 4, 1, 16, 1, 0)
XNN_QD8_UKERNEL(xnn_arch_arm_neon_dot, xnn_qd8_qc2w_packw_gemm_goi_ukernel_x16c4__neondot, 16, 4, 1, 16, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_arm_neon_dot, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x8c4__neondot, 8, 4, 1, 16, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_arm_neon_dot, xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x16c4__neondot, 16, 4, 1, 16, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_arm_neon_dot, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x8c4__neondot, 8, 4, 1, 16, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_arm_neon_dot, xnn_qs8_qc2w_packw_gemm_goi_ukernel_x16c4__neondot, 16, 4, 1, 16, 1, 0)
#endif
