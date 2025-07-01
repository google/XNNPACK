// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#ifndef XNN_QUANTIZED
#define XNN_QUANTIZED(T) T
#define XNN_DEFINED_QUANTIZED
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qu8_vcvt_ukernel__neon_u8, 8, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qu8_vcvt_ukernel__neon_u16, 16, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qu8_vcvt_ukernel__neon_u32, 32, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__sse2_u16, 16, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__sse2_u32, 32, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_ssse3, xnn_qu8_vcvt_ukernel__ssse3_u16, 16, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_ssse3, xnn_qu8_vcvt_ukernel__ssse3_u32, 32, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qu8_vcvt_ukernel__avx_u8, 8, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qu8_vcvt_ukernel__avx_u16, 16, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qu8_vcvt_ukernel__avx_u32, 32, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qu8_vcvt_ukernel__sse41_u8, 8, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qu8_vcvt_ukernel__sse41_u16, 16, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qu8_vcvt_ukernel__sse41_u32, 32, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_vcvt_ukernel__avx2_u16, 16, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_vcvt_ukernel__avx2_u32, 32, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_vcvt_ukernel__avx2_u64, 64, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__wasmsimd_u8, 8, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__wasmsimd_u16, 16, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__wasmsimd_u32, 32, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u8, 8, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u16, 16, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u32, 32, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM
XNN_UKERNEL(xnn_arch_arm_v6, xnn_qu8_vcvt_ukernel__armsimd32_u4, 4, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_v6, xnn_qu8_vcvt_ukernel__armsimd32_u8, 8, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
#endif  // XNN_ARCH_ARM

XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__scalar_u1, 1, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__scalar_u2, 2, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vcvt_ukernel__scalar_u4, 4, false, XNN_QUANTIZED(uint8_t), XNN_QUANTIZED(uint8_t), struct xnn_qu8_cvt_params, xnn_init_qu8_cvt_scalar_params)


#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif
