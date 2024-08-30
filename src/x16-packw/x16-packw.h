// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, unroll)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, nr, kr, sr, kblock, nr_scale

XNN_UKERNEL(0, xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4, 32, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4, 64, 1, 1, 4, 1)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm, 8, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8, 8, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm, 8, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12, 8, 1, 1, 12, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm, 8, 1, 1, 12, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16, 8, 1, 1, 16, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm, 8, 1, 1, 16, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8, 16, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm, 16, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12, 16, 1, 1, 12, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm, 16, 1, 1, 12, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16, 16, 1, 1, 16, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm, 16, 1, 1, 16, 1)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16, 8, 1, 1, 16, 1)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm, 8, 1, 1, 16, 1)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16, 16, 1, 1, 16, 1)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm, 16, 1, 1, 16, 1)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

