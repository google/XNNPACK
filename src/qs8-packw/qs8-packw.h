// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, izp
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x8c4__scalar, 8, 4, 1, 4, 1, 0)
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x16c4__scalar, 16, 4, 1, 4, 1, 0)
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x32c4__scalar, 32, 4, 1, 4, 1, 0)
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x64c4__scalar, 64, 4, 1, 4, 1, 0)

XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 8, 1, 0)

XNN_QS8_UKERNEL(0, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 8, 1, 128)
XNN_QS8_UKERNEL(0, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 8, 1, 128)

XNN_QS8_GIO_UKERNEL(0, xnn_qs8_packw_gemm_gio_ukernel_x8c4__scalar, 8, 4, 1, 4, 1, 0)
XNN_QS8_GIO_UKERNEL(0, xnn_qs8_packw_gemm_gio_ukernel_x16c4__scalar, 16, 4, 1, 4, 1, 0)
XNN_QS8_GIO_UKERNEL(0, xnn_qs8_packw_gemm_gio_ukernel_x32c4__scalar, 32, 4, 1, 4, 1, 0)
XNN_QS8_GIO_UKERNEL(0, xnn_qs8_packw_gemm_gio_ukernel_x64c4__scalar, 64, 4, 1, 4, 1, 0)

XNN_QS8_GIO_UKERNEL(0, xnn_qs8_packw_gemm_gio_ukernel_x8c8__scalar, 8, 8, 1, 8, 1, 0)
XNN_QS8_GIO_UKERNEL(0, xnn_qs8_packw_gemm_gio_ukernel_x16c8__scalar, 16, 8, 1, 8, 1, 0)

XNN_QS8_GIO_UKERNEL(0, xnn_qs8_to_qu8_packw_gemm_gio_ukernel_x8c8__scalar, 8, 8, 1, 8, 1, 128)
XNN_QS8_GIO_UKERNEL(0, xnn_qs8_to_qu8_packw_gemm_gio_ukernel_x16c8__scalar, 16, 8, 1, 8, 1, 128)

#if XNN_ARCH_X86_64 || XNN_ARCH_X86
XNN_QS8_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_packw_gemm_goi_ukernel_x8c8__avx2_madd, 8, 8, 1, 8, 1, 0)
#endif

#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_QS8_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_packw_gemm_goi_ukernel_x8c8__avxvnni, 8, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_packw_gemm_goi_ukernel_x8c8__avxvnni_prfm, 8, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__avxvnni, 8, 8, 1, 8, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__avxvnni_prfm, 8, 8, 1, 8, 1, 128)

XNN_QS8_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_packw_gemm_goi_ukernel_x16c8__avxvnni, 16, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_packw_gemm_goi_ukernel_x16c8__avxvnni_prfm, 16, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__avxvnni, 16, 8, 1, 8, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__avxvnni_prfm, 16, 8, 1, 8, 1, 128)

XNN_QS8_GIO_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_packw_gemm_gio_ukernel_x8c8__avxvnni, 8, 8, 1, 8, 1, 0)
XNN_QS8_GIO_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_packw_gemm_gio_ukernel_x8c8__avxvnni_prfm, 8, 8, 1, 8, 1, 0)
#endif

#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_packw_gemm_goi_ukernel_x64c4__avx256vnni, 64, 4, 1, 4, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_packw_gemm_goi_ukernel_x64c4__avx256vnni_prfm, 64, 4, 1, 4, 1, 0)

XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_packw_gemm_goi_ukernel_x8c8__avx256vnni, 8, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_packw_gemm_goi_ukernel_x8c8__avx256vnni_prfm, 8, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__avx256vnni, 8, 8, 1, 8, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__avx256vnni_prfm, 8, 8, 1, 8, 1, 128)

XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_packw_gemm_goi_ukernel_x16c8__avx256vnni, 16, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_packw_gemm_goi_ukernel_x16c8__avx256vnni_prfm, 16, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__avx256vnni, 16, 8, 1, 8, 1, 128)
XNN_QS8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__avx256vnni_prfm, 16, 8, 1, 8, 1, 128)
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x8c8__wasmrelaxedsimd, 8, 8, 1, 8, 1, 0)
XNN_QS8_UKERNEL(0, xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__wasmrelaxedsimd, 8, 8, 1, 8, 1, 128)
#endif // XNN_ARCH_WASMRELAXEDSIMD
