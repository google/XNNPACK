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
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vadd_minmax_ukernel__neon_ld64_u8, 8, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vadd_minmax_ukernel__neon_ld64_u24, 24, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vadd_minmax_ukernel__neon_ld64_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vadd_minmax_ukernel__neon_ld128_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vadd_minmax_ukernel__neon_ld128_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u8, 8, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u24, 24, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u8, 8, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u24, 24, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vadd_minmax_ukernel__avx_mul16_ld64_u8, 8, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vadd_minmax_ukernel__avx_mul16_ld64_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vadd_minmax_ukernel__avx_mul16_ld64_u24, 24, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vadd_minmax_ukernel__avx_mul16_ld64_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u8, 8, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u24, 24, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u8, 8, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u24, 24, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u8, 8, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u24, 24, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__wasmsimd_u8, 8, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__wasmsimd_u16, 16, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__wasmsimd_u24, 24, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__wasmsimd_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_qs8_vadd_minmax_ukernel__hvx_u32, 32, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_qs8_vadd_minmax_ukernel__hvx_u64, 64, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_qs8_vadd_minmax_ukernel__hvx_u96, 96, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_qs8_vadd_minmax_ukernel__hvx_u128, 128, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
#endif  // XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)

XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__scalar_u1, 1, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__scalar_u2, 2, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_vadd_minmax_ukernel__scalar_u4, 4, false, int8_t, struct xnn_qs8_add_minmax_params, xnn_init_qs8_add_minmax_scalar_params)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
