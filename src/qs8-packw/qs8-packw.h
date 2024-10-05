// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_QS8_UKERNEL_WITH_PARAMS
#define XNN_QS8_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, params_type, init_params) \
    XNN_QS8_UKERNEL(arch_flags, ukernel, kblock)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_QS8_UKERNEL
#define XNN_QS8_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
    XNN_QS8_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, nr, kr, sr, kblock, nr_scale
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x8c4__scalar, 8, 4, 1, 4, 1)
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x16c4__scalar, 16, 4, 1, 4, 1)
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x32c4__scalar, 32, 4, 1, 4, 1)
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x64c4__scalar, 64, 4, 1, 4, 1)

XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x8c8__scalar, 8, 8, 1, 8, 1)
XNN_QS8_UKERNEL(0, xnn_qs8_packw_gemm_goi_ukernel_x16c8__scalar, 16, 8, 1, 8, 1)

// TODO: immintrin.h only provide _mm256_insert_epi64 for __x86_64__
#if XNN_ENABLE_AVXVNNIINT8 && XNN_ARCH_X86_64
XNN_QS8_UKERNEL(xnn_arch_x86_avxvnniint8, xnn_qs8_packw_gemm_goi_ukernel_x8c8__avxvnniint8, 8, 8, 1, 8, 1)
#endif

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_QS8_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_QS8_UKERNEL
#endif


