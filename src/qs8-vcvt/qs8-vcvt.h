// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_CVT_UKERNEL_WITH_PARAMS
#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out, params_type, init_params) \
    XNN_CVT_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out)
#define XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_CVT_UKERNEL
#define XNN_CVT_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out) \
    XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out, void, /*init_params=*/nullptr)
#define XNN_DEFINED_CVT_UKERNEL
#endif

#ifndef XNN_QUANTIZED
#define XNN_QUANTIZED(T) T
#define XNN_DEFINED_QUANTIZED
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vcvt_ukernel__neon_u8, 8, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vcvt_ukernel__neon_u16, 16, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qs8_vcvt_ukernel__neon_u32, 32, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__sse2_u16, 16, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__sse2_u32, 32, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_ssse3, xnn_qs8_vcvt_ukernel__ssse3_u16, 16, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_ssse3, xnn_qs8_vcvt_ukernel__ssse3_u32, 32, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vcvt_ukernel__avx_u8, 8, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vcvt_ukernel__avx_u16, 16, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qs8_vcvt_ukernel__avx_u32, 32, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vcvt_ukernel__sse41_u8, 8, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vcvt_ukernel__sse41_u16, 16, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qs8_vcvt_ukernel__sse41_u32, 32, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_vcvt_ukernel__avx2_u16, 16, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_vcvt_ukernel__avx2_u32, 32, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qs8_vcvt_ukernel__avx2_u64, 64, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__wasmsimd_u8, 8, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__wasmsimd_u16, 16, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__wasmsimd_u32, 32, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u8, 8, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u16, 16, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u32, 32, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_v6, xnn_qs8_vcvt_ukernel__armsimd32_u4, 4, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_v6, xnn_qs8_vcvt_ukernel__armsimd32_u8, 8, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_ARM

XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__scalar_u1, 1, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__scalar_u2, 2, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_qs8_vcvt_ukernel__scalar_u4, 4, false, XNN_QUANTIZED(int8_t), XNN_QUANTIZED(int8_t), struct xnn_qs8_cvt_params, xnn_init_qs8_cvt_scalar_params)

#ifdef XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#undef XNN_CVT_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_CVT_UKERNEL
#undef XNN_DEFINED_CVT_UKERNEL
#undef XNN_CVT_UKERNEL
#endif

#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif

