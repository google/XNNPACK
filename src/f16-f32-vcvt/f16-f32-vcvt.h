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


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int32_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16, xnn_f16_f32_vcvt_ukernel__neonfp16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16, xnn_f16_f32_vcvt_ukernel__neonfp16_u16, 16, false, xnn_float16, float, void, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__sse2_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__sse2_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__sse2_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__sse2_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__sse2_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__sse2_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__sse2_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__sse2_int32_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int32_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int32_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f16_f32_vcvt_ukernel__f16c_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f16_f32_vcvt_ukernel__f16c_u16, 16, false, xnn_float16, float, void, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f16_f32_vcvt_ukernel__avx512skx_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f16_f32_vcvt_ukernel__avx512skx_u32, 32, false, xnn_float16, float, void, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u32, 32, false, xnn_float16, float, void, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u32, 32, false, xnn_float16, float, void, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__scalar_u1, 1, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__scalar_u2, 2, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__scalar_u3, 3, false, xnn_float16, float, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f16_f32_vcvt_ukernel__scalar_u4, 4, false, xnn_float16, float, void, NULL)

#ifdef XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#undef XNN_CVT_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_CVT_UKERNEL
#undef XNN_DEFINED_CVT_UKERNEL
#undef XNN_CVT_UKERNEL
#endif
