// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_f32_vcvt_ukernel__neon_int32_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16, xnn_f16_f32_vcvt_ukernel__neonfp16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16, xnn_f16_f32_vcvt_ukernel__neonfp16_u16, 16, false, xnn_float16, float, void, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__sse2_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__sse2_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__sse2_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__sse2_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__sse2_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__sse2_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__sse2_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__sse2_int32_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f16_f32_vcvt_ukernel__sse41_int32_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f16_f32_vcvt_ukernel__avx_int32_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32_vcvt_ukernel__f16c_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_f16c, xnn_f16_f32_vcvt_ukernel__f16c_u16, 16, false, xnn_float16, float, void, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32_vcvt_ukernel__avx512skx_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_f32_vcvt_ukernel__avx512skx_u32, 32, false, xnn_float16, float, void, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u32, 32, false, xnn_float16, float, void, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u32, 32, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u8, 8, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u16, 16, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u24, 24, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u32, 32, false, xnn_float16, float, void, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_FP16_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_f32_vcvt_ukernel__rvvfp16arith_u1v, 1, true, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_f32_vcvt_ukernel__rvvfp16arith_u2v, 2, true, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_f32_vcvt_ukernel__rvvfp16arith_u4v, 4, true, xnn_float16, float, void, NULL)
#endif

XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__scalar_u1, 1, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__scalar_u2, 2, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__scalar_u3, 3, false, xnn_float16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_f32_vcvt_ukernel__scalar_u4, 4, false, xnn_float16, float, void, NULL)

