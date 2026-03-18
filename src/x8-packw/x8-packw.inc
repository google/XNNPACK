// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// Arguments are:
// XNN_UKERNEL(arch, ukernel, nr, kr, sr, kblock, nr_scale)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x2__scalar_u2, 2, 1, 1, 2, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x4__scalar_u2, 4, 1, 1, 2, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2, 8, 1, 1, 2, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x16__scalar_u2, 16, 1, 1, 2, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x32__scalar_u2, 32, 1, 1, 2, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x2__scalar_u4, 2, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x4__scalar_u4, 4, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x16__scalar_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_goi_ukernel_x32__scalar_u4, 32, 1, 1, 4, 1)

XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x8c8__scalar, 8, 8, 1, 8, 1)

#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_x8_packw_gemm_goi_ukernel_x8c8__avx2, 8, 8, 1, 8, 1)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_x8_packw_gemm_goi_ukernel_x8c8__avx2_prfm, 8, 8, 1, 8, 1)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_x8_packw_gemm_goi_ukernel_x16c8__avx2, 16, 8, 1, 8, 1)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_x8_packw_gemm_goi_ukernel_x16c8__avx2_prfm, 16, 8, 1, 8, 1)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avx256skx, xnn_x8_packw_gemm_goi_ukernel_x8c8__avx256skx, 8, 8, 1, 8, 1)
XNN_UKERNEL(xnn_arch_x86_avx256skx, xnn_x8_packw_gemm_goi_ukernel_x8c8__avx256skx_prfm, 8, 8, 1, 8, 1)
XNN_UKERNEL(xnn_arch_x86_avx256skx, xnn_x8_packw_gemm_goi_ukernel_x16c8__avx256skx, 16, 8, 1, 8, 1)
XNN_UKERNEL(xnn_arch_x86_avx256skx, xnn_x8_packw_gemm_goi_ukernel_x16c8__avx256skx_prfm, 16, 8, 1, 8, 1)
#endif

