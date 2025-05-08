// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, channel_tile, vector_tile, datatype_in, datatype_out, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, row_tile, channel_tile, vector_tile, datatype_in, datatype_out)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, row_tile, channel_tile, vector_tile, datatype_in, datatype_out) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, channel_tile, vector_tile, datatype_in, datatype_out, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_rdsum_ukernel_7p7x__scalar_c4, 7, 4, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_rdsum_ukernel_7p7x__neon_c16, 7, 16, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_rdsum_ukernel_7p7x__neon_c32, 7, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_rdsum_ukernel_7p7x__neon_c64, 7, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_rdsum_ukernel_7p7x__sse41_c16, 7, 16, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_rdsum_ukernel_7p7x__sse41_c32, 7, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_rdsum_ukernel_7p7x__sse41_c64, 7, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_rdsum_ukernel_7p7x__avx2_c32, 7, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_rdsum_ukernel_7p7x__avx2_c64, 7, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c64, 7, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c128, 7, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c16, 7, 16, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c32, 7, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c64, 7, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_qs8_rdsum_ukernel_7p7x__rvv_u1v, 7, 1, true, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_qs8_rdsum_ukernel_7p7x__rvv_u2v, 7, 2, true, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
