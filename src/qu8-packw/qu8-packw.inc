// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

// Arguments are:
// XNN_QU8_UKERNEL(arch, ukernel, nr, kr, sr, kblock, nr_scale)

#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_QU8_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_packw_gemm_goi_ukernel_x8c8__avx2_madd, 8, 8, 1, 8, 1)
XNN_QU8_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_packw_gemm_goi_ukernel_x16c8__avx2_madd, 16, 8, 1, 8, 1)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_QU8_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qu8_packw_gemm_goi_ukernel_x16c8__avx256vnni, 16, 8, 1, 8, 1)
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
