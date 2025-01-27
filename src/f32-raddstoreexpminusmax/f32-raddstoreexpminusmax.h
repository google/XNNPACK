// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, element_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, element_tile,  datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile,  datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u8, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc2, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc2, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc4, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc4, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u8, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx256skx, xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc2, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx256skx, xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u8, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx256skx, xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx256skx, xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc4, 32, float, struct xnn_f32_default_params, NULL)
#endif //XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u16, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u32_acc2, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc2, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc4, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u16, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u32_acc2, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc4, 64, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0,xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, 16, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, 16, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u32, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u64_acc2, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc2, 128, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc4, 128, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, 2 * xnn_init_hardware_config()->vlenb / sizeof(float), float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, 4 * xnn_init_hardware_config()->vlenb / sizeof(float), float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, 16, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u1, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, 2, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u1, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, 2, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, 4, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, 4, float, struct xnn_f32_default_params, NULL)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif