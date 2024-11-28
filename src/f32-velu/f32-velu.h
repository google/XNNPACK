// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_velu_ukernel__neon_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_velu_ukernel__neon_rr2_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_velu_ukernel__neon_rr2_p6_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_velu_ukernel__neon_rr2_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_velu_ukernel__neonfma_rr1_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_velu_ukernel__neonfma_rr1_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_velu_ukernel__neonfma_rr1_p6_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_velu_ukernel__neonfma_rr1_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__sse2_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__sse2_rr2_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__sse2_rr2_p6_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__sse2_rr2_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_velu_ukernel__sse41_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_velu_ukernel__sse41_rr2_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_velu_ukernel__sse41_rr2_p6_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_velu_ukernel__sse41_rr2_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u24, 24, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u24, 24, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_p6_u24, 24, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_velu_ukernel__avx_rr2_p6_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u24, 24, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u24, 24, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u24, 24, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_p6_u24, 24, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_velu_ukernel__avx2_rr1_p6_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u48, 48, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u64, 64, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_velu_ukernel__avx512f_rr1_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_velu_ukernel__avx512f_rr1_p6_u32, 32, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_velu_ukernel__avx512f_rr1_p6_u48, 48, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_velu_ukernel__avx512f_rr1_p6_u64, 64, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u8, 8, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u12, 12, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u16, 16, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u1, 1, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u2, 2, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u3, 3, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u5, 5, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u6, 6, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_p6_u1, 1, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_p6_u2, 2, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_p6_u3, 3, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_p6_u5, 5, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__wasm_rr2_p6_u6, 6, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u1, 1, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u2, 2, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u3, 3, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u5, 5, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u6, 6, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_p6_u1, 1, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_p6_u2, 2, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_p6_u3, 3, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_p6_u4, 4, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_p6_u5, 5, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_velu_ukernel__scalar_rr2_p6_u6, 6, false, float, struct xnn_f32_elu_params, xnn_init_f32_elu_scalar_params)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
