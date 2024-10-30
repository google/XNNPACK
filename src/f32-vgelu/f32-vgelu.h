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

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__scalar_u1, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__scalar_u2, 2, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__scalar_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u1, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u2, 2, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_fma3, xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vgelu_ukernel__avx512f_rational_12_10_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vgelu_ukernel__avx512f_rational_12_10_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vgelu_ukernel__avx512f_rational_12_10_div_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vgelu_ukernel__avx512f_rational_12_10_div_u64, 64, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u64, 64, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_vgelu_ukernel__hvx_rational_12_10_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_vgelu_ukernel__hvx_rational_12_10_div_u64, 64, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_vgelu_ukernel__hvx_rational_12_10_div_u128, 128, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
