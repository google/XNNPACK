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

XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x2__scalar, 2, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x4__scalar, 4, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x8__scalar, 8, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x16__scalar, 16, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x32__scalar, 32, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x64__scalar, 64, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x8c8__scalar, 8, 8, 1, 8, 1)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x8_packw_gemm_goi_ukernel_x8c4__neon, 8, 4, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x8_packw_gemm_goi_ukernel_x16c4__neon, 16, 4, 1, 4, 1)

XNN_GIO_UKERNEL(xnn_arch_arm_neon, xnn_x8_packw_gemm_gio_ukernel_x16__neon_u2, 16, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_arm_neon, xnn_x8_packw_gemm_gio_ukernel_x32__neon_u2, 32, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_arm_neon, xnn_x8_packw_gemm_gio_ukernel_x48__neon_u2, 48, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_arm_neon, xnn_x8_packw_gemm_gio_ukernel_x64__neon_u2, 64, 1, 1, 1, 1)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x16__wasmsimd_u2, 16, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x32__wasmsimd_u2, 32, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x48__wasmsimd_u2, 48, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x8_packw_gemm_gio_ukernel_x64__wasmsimd_u2, 64, 1, 1, 1, 1)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_HEXAGON
XNN_GIO_UKERNEL(xnn_arch_hexagon_hvx, xnn_x8_packw_gemm_gio_ukernel_x128__hvx_u2, 128, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_hexagon_hvx, xnn_x8_packw_gemm_gio_ukernel_x256__hvx_u2, 256, 1, 1, 1, 1)
#endif  // XNN_ARCH_HEXAGON


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

