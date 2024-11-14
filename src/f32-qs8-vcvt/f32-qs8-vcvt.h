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
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_qs8_vcvt_ukernel__neon_u8, 8, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_qs8_vcvt_ukernel__neon_u16, 16, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_qs8_vcvt_ukernel__neon_u24, 24, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_qs8_vcvt_ukernel__neon_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_v8, xnn_f32_qs8_vcvt_ukernel__neonv8_u8, 8, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_v8, xnn_f32_qs8_vcvt_ukernel__neonv8_u16, 16, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_v8, xnn_f32_qs8_vcvt_ukernel__neonv8_u24, 24, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_v8, xnn_f32_qs8_vcvt_ukernel__neonv8_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_qs8_vcvt_ukernel__rvv_u1v, 1, true, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_qs8_vcvt_ukernel__rvv_u2v, 2, true, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_qs8_vcvt_ukernel__rvv_u4v, 4, true, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_qs8_vcvt_ukernel__rvv_u8v, 8, true, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__sse2_u8, 8, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__sse2_u16, 16, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__sse2_u24, 24, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__sse2_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_qs8_vcvt_ukernel__sse41_u8, 8, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_qs8_vcvt_ukernel__sse41_u16, 16, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_qs8_vcvt_ukernel__sse41_u24, 24, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_qs8_vcvt_ukernel__sse41_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_qs8_vcvt_ukernel__avx_u8, 8, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_qs8_vcvt_ukernel__avx_u16, 16, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_qs8_vcvt_ukernel__avx_u24, 24, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_qs8_vcvt_ukernel__avx_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_qs8_vcvt_ukernel__avx2_u16, 16, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_qs8_vcvt_ukernel__avx2_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_qs8_vcvt_ukernel__avx2_u48, 48, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_qs8_vcvt_ukernel__avx2_u64, 64, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f32_qs8_vcvt_ukernel__avx512skx_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f32_qs8_vcvt_ukernel__avx512skx_u64, 64, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f32_qs8_vcvt_ukernel__avx512skx_u96, 96, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f32_qs8_vcvt_ukernel__avx512skx_u128, 128, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_u8, 8, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_u16, 16, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_u24, 24, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u8, 8, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u16, 16, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u24, 24, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u1, 1, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u2, 2, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u3, 3, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u4, 4, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_qs8_vcvt_ukernel__hvx_u32, 32, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_qs8_vcvt_ukernel__hvx_u64, 64, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_qs8_vcvt_ukernel__hvx_u96, 96, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_qs8_vcvt_ukernel__hvx_u128, 128, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_f32_qs8_vcvt_ukernel__hvx_u256, 256, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ENABLE_HVX && (XNN_ARCH_HEXAGON)

XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_u1, 1, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_u2, 2, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_u3, 3, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_u4, 4, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u1, 1, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u2, 2, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u3, 3, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u4, 4, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u1, 1, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u2, 2, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u3, 3, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u4, 4, false, float, XNN_QUANTIZED(int8_t), struct xnn_f32_qs8_cvt_params, xnn_init_f32_qs8_cvt_scalar_params)

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

