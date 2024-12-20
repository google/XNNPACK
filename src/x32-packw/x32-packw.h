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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2, 2, 1, 1, 2, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm, 2, 1, 1, 2, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm, 8, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8, 8, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm, 8, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4, 8, 1, 4, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm, 8, 1, 4, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8, 8, 1, 4, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8_prfm, 8, 1, 4, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4, 12, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4_prfm, 12, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8, 12, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8_prfm, 12, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8, 16, 1, 1, 8, 1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm, 16, 1, 1, 8, 1)

XNN_GIO_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_gio_ukernel_x4__neon_u2, 4, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_gio_ukernel_x8__neon_u2, 8, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_gio_ukernel_x12__neon_u2, 12, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_gio_ukernel_x16__neon_u2, 16, 1, 1, 1, 1)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4, 2, 4, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4_prfm, 2, 4, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4_prfm, 8, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8, 8, 1, 1, 8, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8_prfm, 8, 1, 1, 8, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4, 8, 1, 4, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4_prfm, 8, 1, 4, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8, 8, 1, 4, 8, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8_prfm, 8, 1, 4, 8, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4_prfm, 16, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8, 16, 1, 1, 8, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8_prfm, 16, 1, 1, 8, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4, 16, 1, 4, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4_prfm, 16, 1, 4, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8, 16, 1, 4, 8, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8_prfm, 16, 1, 4, 8, 1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4_prfm, 8, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4, 8, 1, 4, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4_prfm, 8, 1, 4, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4_prfm, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4, 16, 1, 4, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4_prfm, 16, 1, 4, 4, 1)

XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x8__avx_u1, 8, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x8__avx_u1_prfm, 8, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x16__avx_u1, 16, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x16__avx_u1_prfm, 16, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x32__avx_u1, 32, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x32__avx_u1_prfm, 32, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x8__avx_u8, 8, 1, 1, 8, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x8__avx_u8_prfm, 8, 1, 1, 8, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x16__avx_u8, 16, 1, 1, 8, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x16__avx_u8_prfm, 16, 1, 1, 8, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x32__avx_u8, 32, 1, 1, 8, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_gio_ukernel_x32__avx_u8_prfm, 32, 1, 1, 8, 1)

XNN_GIO_UKERNEL(xnn_arch_x86_sse4_1, xnn_x32_packw_gemm_gio_ukernel_x4__sse41_u2, 4, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_sse4_1, xnn_x32_packw_gemm_gio_ukernel_x8__sse41_u2, 8, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_sse4_1, xnn_x32_packw_gemm_gio_ukernel_x12__sse41_u2, 12, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_sse4_1, xnn_x32_packw_gemm_gio_ukernel_x16__sse41_u2, 16, 1, 1, 1, 1)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

XNN_GIO_UKERNEL(0, xnn_x32_packw_gemm_gio_ukernel_x4__scalar, 4, 1, 1, 1, 1)
XNN_GIO_UKERNEL(0, xnn_x32_packw_gemm_gio_ukernel_x8__scalar, 8, 1, 1, 1, 1)
XNN_GIO_UKERNEL(0, xnn_x32_packw_gemm_gio_ukernel_x16__scalar, 16, 1, 1, 1, 1)
XNN_GIO_UKERNEL(0, xnn_x32_packw_gemm_gio_ukernel_x32__scalar, 32, 1, 1, 1, 1)

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm, 16, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4, 32, 1, 1, 4, 1)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm, 32, 1, 1, 4, 1)

XNN_GIO_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_gio_ukernel_x16__avx512f_u1, 16, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_gio_ukernel_x16__avx512f_u1_prfm, 16, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_gio_ukernel_x32__avx512f_u1, 32, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_gio_ukernel_x32__avx512f_u1_prfm, 32, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_gio_ukernel_x16__avx512f_u8, 16, 1, 1, 8, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_gio_ukernel_x16__avx512f_u8_prfm, 16, 1, 1, 8, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_gio_ukernel_x32__avx512f_u8, 32, 1, 1, 8, 1)
XNN_GIO_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_gio_ukernel_x32__avx512f_u8_prfm, 32, 1, 1, 8, 1)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86_64 || XNN_ARCH_X86)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4, 2, 4, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_u4, 8, 1, 4, 4, 1)

XNN_GIO_UKERNEL(0, xnn_x32_packw_gemm_gio_ukernel_x4__wasmsimd_u2, 4, 1, 1, 1, 1)
XNN_GIO_UKERNEL(0, xnn_x32_packw_gemm_gio_ukernel_x8__wasmsimd_u2, 8, 1, 1, 1, 1)
XNN_GIO_UKERNEL(0, xnn_x32_packw_gemm_gio_ukernel_x12__wasmsimd_u2, 12, 1, 1, 1, 1)
XNN_GIO_UKERNEL(0, xnn_x32_packw_gemm_gio_ukernel_x16__wasmsimd_u2, 16, 1, 1, 1, 1)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4, 2, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_u4, 2, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_u4, 3, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_u4, 3, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4, 4, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_u4, 4, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_u4, 8, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_u4, 16, 1, 1, 4, 1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_u4, 16, 1, 1, 4, 1)

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u2, 1, 1, 1, 2, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u4, 1, 1, 1, 4, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u8, 1, 1, 1, 8, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u2, 2, 1, 1, 2, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u4, 2, 1, 1, 4, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u8, 2, 1, 1, 8, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u2, 4, 1, 1, 2, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u4, 4, 1, 1, 4, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u8, 4, 1, 1, 8, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u2, 8, 1, 1, 2, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u4, 8, 1, 1, 4, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u8, 8, 1, 1, 8, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV()

#if XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)
XNN_GIO_UKERNEL(xnn_arch_hvx, xnn_x32_packw_gemm_gio_ukernel_x32__hvx_u2, 32, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_hvx, xnn_x32_packw_gemm_gio_ukernel_x64__hvx_u2, 64, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_hvx, xnn_x32_packw_gemm_gio_ukernel_x96__hvx_u2, 96, 1, 1, 1, 1)
XNN_GIO_UKERNEL(xnn_arch_hvx, xnn_x32_packw_gemm_gio_ukernel_x128__hvx_u2, 128, 1, 1, 1, 1)
#endif  // XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

