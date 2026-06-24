// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

// Arguments are:
// XNN_UKERNEL(arch, ukernel, nr, kr, sr, kblock, nr_scale, izp)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x4c8__scalar, 4, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x32c8__scalar, 32, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x4c8__scalar, 4, 8, 1, 16, 1, 128)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 16, 1, 128)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 16, 1, 128)

XNN_QC4UW_UKERNEL(xnn_arch_none, xnn_qs8_qc4uw_packw_gemm_goi_ukernel_x4c8__scalar, 4, 8, 1, 16, 1, 0)
XNN_QC4UW_UKERNEL(xnn_arch_none, xnn_qs8_qc4uw_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 16, 1, 0)
XNN_QC4UW_UKERNEL(xnn_arch_none, xnn_qs8_qc4uw_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 16, 1, 0)
XNN_QC4UW_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc4uw_packw_gemm_goi_ukernel_x4c8__scalar, 4, 8, 1, 16, 1, 128)
XNN_QC4UW_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc4uw_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 16, 1, 128)
XNN_QC4UW_UKERNEL(xnn_arch_none, xnn_qs8_to_qu8_qc4uw_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 16, 1, 128)


#if XNN_ENABLE_SSE2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_sse2, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x4c8__sse2, 4, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_x86_sse2, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__sse2, 8, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_x86_sse2, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x4c8__sse2, 4, 8, 1, 16, 1, 128)
XNN_UKERNEL(xnn_arch_x86_sse2, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x8c8__sse2, 8, 8, 1, 16, 1, 128)

XNN_QC4UW_UKERNEL(xnn_arch_x86_sse2, xnn_qs8_qc4uw_packw_gemm_goi_ukernel_x4c8__sse2, 4, 8, 1, 16, 1, 0)
XNN_QC4UW_UKERNEL(xnn_arch_x86_sse2, xnn_qs8_qc4uw_packw_gemm_goi_ukernel_x8c8__sse2, 8, 8, 1, 16, 1, 0)
XNN_QC4UW_UKERNEL(xnn_arch_x86_sse2, xnn_qs8_to_qu8_qc4uw_packw_gemm_goi_ukernel_x4c8__sse2, 4, 8, 1, 16, 1, 128)
XNN_QC4UW_UKERNEL(xnn_arch_x86_sse2, xnn_qs8_to_qu8_qc4uw_packw_gemm_goi_ukernel_x8c8__sse2, 8, 8, 1, 16, 1, 128)
#endif

#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__avxvnni, 8, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__avxvnni_prfm, 8, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x8c8__avxvnni, 8, 8, 1, 16, 1, 128)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x8c8__avxvnni_prfm, 8, 8, 1, 16, 1, 128)

XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__avxvnni, 16, 8, 1, 32, 1, 0)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__avxvnni_prfm, 16, 8, 1, 32, 1, 0)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x16c8__avxvnni, 16, 8, 1, 32, 1, 128)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x16c8__avxvnni_prfm, 16, 8, 1, 32, 1, 128)
#endif

#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__avx256vnni, 8, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__avx256vnni_prfm, 8, 8, 1, 16, 1, 0)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x8c8__avx256vnni, 8, 8, 1, 16, 1, 128)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x8c8__avx256vnni_prfm, 8, 8, 1, 16, 1, 128)

XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__avx256vnni, 16, 8, 1, 32, 1, 0)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__avx256vnni_prfm, 16, 8, 1, 32, 1, 0)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x16c8__avx256vnni, 16, 8, 1, 32, 1, 128)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_to_qu8_qc4w_packw_gemm_goi_ukernel_x16c8__avx256vnni_prfm, 16, 8, 1, 32, 1, 128)
#endif



