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


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_rdmin_ukernel_2p2x__neon_c32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdmin_ukernel_2p2x__sse2_c32, 2, 32, false, float, float, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_rdmin_ukernel_2p2x__avx_c32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_rdmin_ukernel_2p2x__hvx_c32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_rdmin_ukernel_2p2x__scalar_c2, 2, 2, false, float, float, void*, NULL)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
