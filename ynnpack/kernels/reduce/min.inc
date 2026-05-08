// clang-format off

#ifdef YNN_ARCH_ARM_NEON
YNN_REDUCE_K1_KERNEL(arch_flag::neon, min_fp32_k1_neon, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, min_fp32_kn_neon, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::neon, min_bf16_k1_neon, bfloat16, bfloat16)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, min_bf16_kn_neon, bfloat16, bfloat16)
#ifdef YNN_ARCH_ARM64
YNN_REDUCE_K1_KERNEL(arch_flag::neon, min_fp64_k1_neon, double, double)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, min_fp64_kn_neon, double, double)
#endif
YNN_REDUCE_K1_KERNEL(arch_flag::neon, min_fp16_k1_neon, half, half)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, min_fp16_kn_neon, half, half)
YNN_REDUCE_K1_KERNEL(arch_flag::neon, min_int8_k1_neon, int8_t, int8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, min_int8_kn_neon, int8_t, int8_t)
YNN_REDUCE_K1_KERNEL(arch_flag::neon, min_uint8_k1_neon, uint8_t, uint8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, min_uint8_kn_neon, uint8_t, uint8_t)
#endif

#ifdef YNN_ARCH_HEXAGON_HVX
YNN_REDUCE_K1_KERNEL(arch_flag::hvx, min_fp32_k1_hvx, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::hvx, min_fp32_kn_hvx, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::hvx, min_bf16_k1_hvx, bfloat16, bfloat16)
YNN_REDUCE_KN_KERNEL(arch_flag::hvx, min_bf16_kn_hvx, bfloat16, bfloat16)
YNN_REDUCE_K1_KERNEL(arch_flag::hvx, min_fp16_k1_hvx, half, half)
YNN_REDUCE_KN_KERNEL(arch_flag::hvx, min_fp16_kn_hvx, half, half)
YNN_REDUCE_K1_KERNEL(arch_flag::hvx, min_int8_k1_hvx, int8_t, int8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::hvx, min_int8_kn_hvx, int8_t, int8_t)
YNN_REDUCE_K1_KERNEL(arch_flag::hvx, min_uint8_k1_hvx, uint8_t, uint8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::hvx, min_uint8_kn_hvx, uint8_t, uint8_t)
#endif  // YNN_ARCH_HEXAGON_HVX

#ifdef YNN_ARCH_WASM_SIMD128
YNN_REDUCE_K1_KERNEL(arch_flag::simd128, min_fp32_k1_simd128, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::simd128, min_fp32_kn_simd128, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::simd128, min_int8_k1_simd128, int8_t, int8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::simd128, min_int8_kn_simd128, int8_t, int8_t)
YNN_REDUCE_K1_KERNEL(arch_flag::simd128, min_uint8_k1_simd128, uint8_t, uint8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::simd128, min_uint8_kn_simd128, uint8_t, uint8_t)
#endif  // YNN_ARCH_WASM_SIMD128

#ifdef YNN_ARCH_X86_AVX512
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, min_bf16_k1_avx512, bfloat16, bfloat16)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, min_bf16_kn_avx512, bfloat16, bfloat16)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, min_fp16_k1_avx512, half, half)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, min_fp16_kn_avx512, half, half)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, min_int8_k1_avx512, int8_t, int8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, min_int8_kn_avx512, int8_t, int8_t)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, min_uint8_k1_avx512, uint8_t, uint8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, min_uint8_kn_avx512, uint8_t, uint8_t)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, min_fp32_k1_avx512, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, min_fp32_kn_avx512, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, min_fp64_k1_avx512, double, double)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, min_fp64_kn_avx512, double, double)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2
YNN_REDUCE_K1_KERNEL(arch_flag::avx2, min_bf16_k1_avx2, bfloat16, bfloat16)
YNN_REDUCE_KN_KERNEL(arch_flag::avx2, min_bf16_kn_avx2, bfloat16, bfloat16)
YNN_REDUCE_K1_KERNEL(arch_flag::avx2, min_fp16_k1_avx2, half, half)
YNN_REDUCE_KN_KERNEL(arch_flag::avx2, min_fp16_kn_avx2, half, half)
YNN_REDUCE_K1_KERNEL(arch_flag::avx2, min_int8_k1_avx2, int8_t, int8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::avx2, min_int8_kn_avx2, int8_t, int8_t)
YNN_REDUCE_K1_KERNEL(arch_flag::avx2, min_uint8_k1_avx2, uint8_t, uint8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::avx2, min_uint8_kn_avx2, uint8_t, uint8_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_REDUCE_K1_KERNEL(arch_flag::avx, min_fp32_k1_avx, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::avx, min_fp32_kn_avx, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::avx, min_fp64_k1_avx, double, double)
YNN_REDUCE_KN_KERNEL(arch_flag::avx, min_fp64_kn_avx, double, double)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE41
YNN_REDUCE_K1_KERNEL(arch_flag::sse41, min_int8_k1_sse41, int8_t, int8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::sse41, min_int8_kn_sse41, int8_t, int8_t)
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
YNN_REDUCE_K1_KERNEL(arch_flag::sse2, min_fp32_k1_sse2, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::sse2, min_fp32_kn_sse2, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::sse2, min_bf16_k1_sse2, bfloat16, bfloat16)
YNN_REDUCE_KN_KERNEL(arch_flag::sse2, min_bf16_kn_sse2, bfloat16, bfloat16)
YNN_REDUCE_K1_KERNEL(arch_flag::sse2, min_fp16_k1_sse2, half, half)
YNN_REDUCE_KN_KERNEL(arch_flag::sse2, min_fp16_kn_sse2, half, half)
YNN_REDUCE_K1_KERNEL(arch_flag::sse2, min_uint8_k1_sse2, uint8_t, uint8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::sse2, min_uint8_kn_sse2, uint8_t, uint8_t)
#endif  // YNN_ARCH_X86_SSE2

YNN_REDUCE_K1_KERNEL(arch_flag::none, min_fp32_k1, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::none, min_fp32_kn, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::none, min_fp64_k1, double, double)
YNN_REDUCE_KN_KERNEL(arch_flag::none, min_fp64_kn, double, double)
YNN_REDUCE_K1_KERNEL(arch_flag::none, min_fp16_k1, half, half)
YNN_REDUCE_KN_KERNEL(arch_flag::none, min_fp16_kn, half, half)
YNN_REDUCE_K1_KERNEL(arch_flag::none, min_bf16_k1, bfloat16, bfloat16)
YNN_REDUCE_KN_KERNEL(arch_flag::none, min_bf16_kn, bfloat16, bfloat16)
YNN_REDUCE_K1_KERNEL(arch_flag::none, min_int8_k1, int8_t, int8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::none, min_int8_kn, int8_t, int8_t)
YNN_REDUCE_K1_KERNEL(arch_flag::none, min_uint8_k1, uint8_t, uint8_t)
YNN_REDUCE_KN_KERNEL(arch_flag::none, min_uint8_kn, uint8_t, uint8_t)
