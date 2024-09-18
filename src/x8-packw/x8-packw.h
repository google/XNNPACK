// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, kblock)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, nr, kr, sr, kblock, nr_scale

XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x2__scalar_u2, 2, 1, 1, 2, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x4__scalar_u2, 4, 1, 1, 2, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2, 8, 1, 1, 2, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x16__scalar_u2, 16, 1, 1, 2, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x32__scalar_u2, 32, 1, 1, 2, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x2__scalar_u4, 2, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x4__scalar_u4, 4, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x16__scalar_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x8_packw_gemm_goi_ukernel_x32__scalar_u4, 32, 1, 1, 4, 1)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

