// clang-format off

#ifdef YNN_ARCH_ARM_NEON
YNN_REDUCE_KERNEL(arch_flag::neon, min_k1_fp32_neon, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::neon, min_kn_fp32_neon, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::neon, min_k1_xf16_neon, reduce_dim::k1, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::neon, min_kn_xf16_neon, reduce_dim::kn, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::neon, min_k1_xf16_neon, reduce_dim::k1, half, half)
YNN_REDUCE_KERNEL(arch_flag::neon, min_kn_xf16_neon, reduce_dim::kn, half, half)
YNN_REDUCE_KERNEL(arch_flag::neon, min_k1_xf8_neon, reduce_dim::k1, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::neon, min_kn_xf8_neon, reduce_dim::kn, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::neon, min_k1_xf8_neon, reduce_dim::k1, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::neon, min_kn_xf8_neon, reduce_dim::kn, fp8_e4m3, fp8_e4m3)
#ifdef YNN_ARCH_ARM64
YNN_REDUCE_KERNEL(arch_flag::neon, min_k1_fp64_neon, reduce_dim::k1, double, double)
YNN_REDUCE_KERNEL(arch_flag::neon, min_kn_fp64_neon, reduce_dim::kn, double, double)
#endif
YNN_REDUCE_KERNEL(arch_flag::neon, min_k1_int8_neon, reduce_dim::k1, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::neon, min_kn_int8_neon, reduce_dim::kn, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::neon, min_k1_uint8_neon, reduce_dim::k1, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::neon, min_kn_uint8_neon, reduce_dim::kn, uint8_t, uint8_t)
#endif

#ifdef YNN_ARCH_HEXAGON_HVX
YNN_REDUCE_KERNEL(arch_flag::hvx, min_k1_fp32_hvx, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_kn_fp32_hvx, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_k1_xf16_hvx, reduce_dim::k1, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_kn_xf16_hvx, reduce_dim::kn, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_k1_xf16_hvx, reduce_dim::k1, half, half)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_kn_xf16_hvx, reduce_dim::kn, half, half)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_k1_xf8_hvx, reduce_dim::k1, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_kn_xf8_hvx, reduce_dim::kn, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_k1_xf8_hvx, reduce_dim::k1, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_kn_xf8_hvx, reduce_dim::kn, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_k1_int8_hvx, reduce_dim::k1, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_kn_int8_hvx, reduce_dim::kn, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_k1_uint8_hvx, reduce_dim::k1, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::hvx, min_kn_uint8_hvx, reduce_dim::kn, uint8_t, uint8_t)
#endif  // YNN_ARCH_HEXAGON_HVX

#ifdef YNN_ARCH_WASM_SIMD128
YNN_REDUCE_KERNEL(arch_flag::simd128, min_k1_fp32_simd128, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::simd128, min_kn_fp32_simd128, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::simd128, min_k1_int8_simd128, reduce_dim::k1, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::simd128, min_kn_int8_simd128, reduce_dim::kn, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::simd128, min_k1_uint8_simd128, reduce_dim::k1, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::simd128, min_kn_uint8_simd128, reduce_dim::kn, uint8_t, uint8_t)
#endif  // YNN_ARCH_WASM_SIMD128

#ifdef YNN_ARCH_X86_AVX512
YNN_REDUCE_KERNEL(arch_flag::avx512, min_k1_xf16_avx512, reduce_dim::k1, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_kn_xf16_avx512, reduce_dim::kn, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_k1_xf16_avx512, reduce_dim::k1, half, half)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_kn_xf16_avx512, reduce_dim::kn, half, half)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_k1_xf8_avx512, reduce_dim::k1, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_kn_xf8_avx512, reduce_dim::kn, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_k1_xf8_avx512, reduce_dim::k1, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_kn_xf8_avx512, reduce_dim::kn, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_k1_int8_avx512, reduce_dim::k1, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_kn_int8_avx512, reduce_dim::kn, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_k1_uint8_avx512, reduce_dim::k1, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_kn_uint8_avx512, reduce_dim::kn, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_k1_fp32_avx512, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_kn_fp32_avx512, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_k1_fp64_avx512, reduce_dim::k1, double, double)
YNN_REDUCE_KERNEL(arch_flag::avx512, min_kn_fp64_avx512, reduce_dim::kn, double, double)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2
YNN_REDUCE_KERNEL(arch_flag::avx2, min_k1_xf16_avx2, reduce_dim::k1, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_kn_xf16_avx2, reduce_dim::kn, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_k1_xf16_avx2, reduce_dim::k1, half, half)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_kn_xf16_avx2, reduce_dim::kn, half, half)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_k1_xf8_avx2, reduce_dim::k1, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_kn_xf8_avx2, reduce_dim::kn, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_k1_xf8_avx2, reduce_dim::k1, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_kn_xf8_avx2, reduce_dim::kn, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_k1_int8_avx2, reduce_dim::k1, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_kn_int8_avx2, reduce_dim::kn, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_k1_uint8_avx2, reduce_dim::k1, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::avx2, min_kn_uint8_avx2, reduce_dim::kn, uint8_t, uint8_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_REDUCE_KERNEL(arch_flag::avx, min_k1_fp32_avx, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::avx, min_kn_fp32_avx, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::avx, min_k1_fp64_avx, reduce_dim::k1, double, double)
YNN_REDUCE_KERNEL(arch_flag::avx, min_kn_fp64_avx, reduce_dim::kn, double, double)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE41
YNN_REDUCE_KERNEL(arch_flag::sse41, min_k1_int8_sse41, reduce_dim::k1, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::sse41, min_kn_int8_sse41, reduce_dim::kn, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::sse41, min_k1_xf8_sse41, reduce_dim::k1, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::sse41, min_kn_xf8_sse41, reduce_dim::kn, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::sse41, min_k1_xf8_sse41, reduce_dim::k1, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::sse41, min_kn_xf8_sse41, reduce_dim::kn, fp8_e4m3, fp8_e4m3)
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
YNN_REDUCE_KERNEL(arch_flag::sse2, min_k1_fp32_sse2, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::sse2, min_kn_fp32_sse2, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::sse2, min_k1_xf16_sse2, reduce_dim::k1, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::sse2, min_kn_xf16_sse2, reduce_dim::kn, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::sse2, min_k1_xf16_sse2, reduce_dim::k1, half, half)
YNN_REDUCE_KERNEL(arch_flag::sse2, min_kn_xf16_sse2, reduce_dim::kn, half, half)
YNN_REDUCE_KERNEL(arch_flag::sse2, min_k1_uint8_sse2, reduce_dim::k1, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::sse2, min_kn_uint8_sse2, reduce_dim::kn, uint8_t, uint8_t)
#endif  // YNN_ARCH_X86_SSE2

YNN_REDUCE_KERNEL(arch_flag::none, min_k1_fp32, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::none, min_kn_fp32, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::none, min_k1_fp64, reduce_dim::k1, double, double)
YNN_REDUCE_KERNEL(arch_flag::none, min_kn_fp64, reduce_dim::kn, double, double)
YNN_REDUCE_KERNEL(arch_flag::none, min_k1_fp16, reduce_dim::k1, half, half)
YNN_REDUCE_KERNEL(arch_flag::none, min_kn_fp16, reduce_dim::kn, half, half)
YNN_REDUCE_KERNEL(arch_flag::none, min_k1_bf16, reduce_dim::k1, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::none, min_kn_bf16, reduce_dim::kn, bfloat16, bfloat16)
YNN_REDUCE_KERNEL(arch_flag::none, min_k1_int8, reduce_dim::k1, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::none, min_kn_int8, reduce_dim::kn, int8_t, int8_t)
YNN_REDUCE_KERNEL(arch_flag::none, min_k1_uint8, reduce_dim::k1, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::none, min_kn_uint8, reduce_dim::kn, uint8_t, uint8_t)
YNN_REDUCE_KERNEL(arch_flag::none, min_k1_fp8_e5m2, reduce_dim::k1, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::none, min_kn_fp8_e5m2, reduce_dim::kn, fp8_e5m2, fp8_e5m2)
YNN_REDUCE_KERNEL(arch_flag::none, min_k1_fp8_e4m3, reduce_dim::k1, fp8_e4m3, fp8_e4m3)
YNN_REDUCE_KERNEL(arch_flag::none, min_kn_fp8_e4m3, reduce_dim::kn, fp8_e4m3, fp8_e4m3)
