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
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_f16_vcvt_ukernel__neon_u8, 8, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_f16_vcvt_ukernel__neon_u16, 16, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_f16_vcvt_ukernel__neon_u24, 24, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_f16_vcvt_ukernel__neon_u32, 32, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16, xnn_f32_f16_vcvt_ukernel__neonfp16_u8, 8, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16, xnn_f32_f16_vcvt_ukernel__neonfp16_u16, 16, false, float, xnn_float16, void, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__sse2_u8, 8, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__sse2_u16, 16, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__sse2_u24, 24, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__sse2_u32, 32, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_f16_vcvt_ukernel__sse41_u8, 8, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_f16_vcvt_ukernel__sse41_u16, 16, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_f16_vcvt_ukernel__sse41_u24, 24, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_f32_f16_vcvt_ukernel__sse41_u32, 32, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_f16_vcvt_ukernel__avx_u8, 8, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_f16_vcvt_ukernel__avx_u16, 16, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_f16_vcvt_ukernel__avx_u24, 24, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_f32_f16_vcvt_ukernel__avx_u32, 32, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f32_f16_vcvt_ukernel__f16c_u8, 8, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f32_f16_vcvt_ukernel__f16c_u16, 16, false, float, xnn_float16, void, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f32_f16_vcvt_ukernel__avx512skx_u16, 16, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f32_f16_vcvt_ukernel__avx512skx_u32, 32, false, float, xnn_float16, void, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__wasmsimd_u8, 8, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__wasmsimd_u16, 16, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__wasmsimd_u24, 24, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__wasmsimd_u32, 32, false, float, xnn_float16, void, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u8, 8, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u16, 16, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24, 24, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u32, 32, false, float, xnn_float16, void, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_FP16_VECTOR
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector_fp16_arith, xnn_f32_f16_vcvt_ukernel__rvvfp16arith_u1v, 1, true, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector_fp16_arith, xnn_f32_f16_vcvt_ukernel__rvvfp16arith_u2v, 2, true, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector_fp16_arith, xnn_f32_f16_vcvt_ukernel__rvvfp16arith_u4v, 4, true, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector_fp16_arith, xnn_f32_f16_vcvt_ukernel__rvvfp16arith_u8v, 8, true, float, xnn_float16, void, NULL)
#endif

XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u1, 1, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u2, 2, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u3, 3, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4, 4, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u1, 1, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2, 2, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3, 3, false, float, xnn_float16, void, NULL)
XNN_CVT_UKERNEL_WITH_PARAMS(0, xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u4, 4, false, float, xnn_float16, void, NULL)

#ifdef XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_CVT_UKERNEL_WITH_PARAMS
#undef XNN_CVT_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_CVT_UKERNEL
#undef XNN_DEFINED_CVT_UKERNEL
#undef XNN_CVT_UKERNEL
#endif
