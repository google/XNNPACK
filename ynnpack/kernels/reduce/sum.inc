// clang-format off

#ifdef YNN_ARCH_ARM_NEONBF16
YNN_REDUCE_KERNEL(arch_flag::neonbf16, sum_k1_bf16_fp32_neonbf16, reduce_dim::k1, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::neonbf16, sum_kn_bf16_fp32_neonbf16, reduce_dim::kn, bfloat16, float)
#endif // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM_NEONFP16
YNN_REDUCE_KERNEL(arch_flag::neonfp16, sum_k1_fp16_fp32_neonfp16, reduce_dim::k1, half, float)
YNN_REDUCE_KERNEL(arch_flag::neonfp16, sum_kn_fp16_fp32_neonfp16, reduce_dim::kn, half, float)
#endif  // YNN_ARCH_ARM_NEONFP16
#ifdef YNN_ARCH_ARM_NEONDOT
YNN_REDUCE_KERNEL(arch_flag::neondot, sum_k1_int8_int32_neondot, reduce_dim::k1, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::neondot, sum_kn_int8_int32_neondot, reduce_dim::kn, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::neondot, sum_k1_uint8_int32_neondot, reduce_dim::k1, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::neondot, sum_kn_uint8_int32_neondot, reduce_dim::kn, uint8_t, int32_t)
#endif  // YNN_ARCH_ARM_NEONDOT
#ifdef YNN_ARCH_ARM_NEON
YNN_REDUCE_KERNEL(arch_flag::neon, sum_k1_fp32_neon, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::neon, sum_kn_fp32_neon, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::neon, sum_k1_int32_neon, reduce_dim::k1, int32_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::neon, sum_kn_int32_neon, reduce_dim::kn, int32_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::neon, sum_k1_bf16_fp32_neon, reduce_dim::k1, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::neon, sum_kn_bf16_fp32_neon, reduce_dim::kn, bfloat16, float)
#ifdef YNN_ARCH_ARM64
YNN_REDUCE_KERNEL(arch_flag::neon, sum_k1_fp64_neon, reduce_dim::k1, double, double)
YNN_REDUCE_KERNEL(arch_flag::neon, sum_kn_fp64_neon, reduce_dim::kn, double, double)
#endif
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_HEXAGON_HVX
YNN_REDUCE_KERNEL(arch_flag::hvx, sum_k1_int8_int32_hvx, reduce_dim::k1, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::hvx, sum_kn_int8_int32_hvx, reduce_dim::kn, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::hvx, sum_k1_uint8_int32_hvx, reduce_dim::k1, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::hvx, sum_kn_uint8_int32_hvx, reduce_dim::kn, uint8_t, int32_t)
#endif  // YNN_ARCH_HEXAGON_HVX

#ifdef YNN_ARCH_WASM_SIMD128
YNN_REDUCE_KERNEL(arch_flag::simd128, sum_k1_fp32_simd128, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::simd128, sum_kn_fp32_simd128, reduce_dim::kn, float, float)
#endif  // YNN_ARCH_WASM_SIMD128

#ifdef YNN_ARCH_X86_AVX512BF16
YNN_REDUCE_KERNEL(arch_flag::avx512bf16, sum_k1_bf16_fp32_avx512bf16, reduce_dim::k1, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::avx512bf16, sum_kn_bf16_fp32_avx512bf16, reduce_dim::kn, bfloat16, float)
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_k1_bf16_fp32_avx512, reduce_dim::k1, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_kn_bf16_fp32_avx512, reduce_dim::kn, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_k1_fp16_fp32_avx512, reduce_dim::k1, half, float)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_kn_fp16_fp32_avx512, reduce_dim::kn, half, float)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_k1_uint8_int32_avx512, reduce_dim::k1, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_kn_uint8_int32_avx512, reduce_dim::kn, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_k1_int8_int32_avx512, reduce_dim::k1, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_kn_int8_int32_avx512, reduce_dim::kn, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_k1_int32_avx512, reduce_dim::k1, int32_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_kn_int32_avx512, reduce_dim::kn, int32_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_k1_fp32_avx512, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_kn_fp32_avx512, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_k1_fp64_avx512, reduce_dim::k1, double, double)
YNN_REDUCE_KERNEL(arch_flag::avx512, sum_kn_fp64_avx512, reduce_dim::kn, double, double)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2
YNN_REDUCE_KERNEL(arch_flag::avx2, sum_k1_bf16_fp32_avx2, reduce_dim::k1, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::avx2, sum_kn_bf16_fp32_avx2, reduce_dim::kn, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::avx2, sum_k1_uint8_int32_avx2, reduce_dim::k1, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx2, sum_kn_uint8_int32_avx2, reduce_dim::kn, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx2, sum_k1_int8_int32_avx2, reduce_dim::k1, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx2, sum_kn_int8_int32_avx2, reduce_dim::kn, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx2, sum_k1_int32_avx2, reduce_dim::k1, int32_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::avx2, sum_kn_int32_avx2, reduce_dim::kn, int32_t, int32_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_REDUCE_KERNEL(arch_flag::avx, sum_k1_fp32_avx, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::avx, sum_kn_fp32_avx, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::avx, sum_k1_fp64_avx, reduce_dim::k1, double, double)
YNN_REDUCE_KERNEL(arch_flag::avx, sum_kn_fp64_avx, reduce_dim::kn, double, double)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_F16C
YNN_REDUCE_KERNEL(arch_flag::f16c, sum_k1_fp16_fp32_f16c, reduce_dim::k1, half, float)
YNN_REDUCE_KERNEL(arch_flag::f16c, sum_kn_fp16_fp32_f16c, reduce_dim::kn, half, float)
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_SSE41
YNN_REDUCE_KERNEL(arch_flag::sse41, sum_kn_uint8_int32_sse41, reduce_dim::kn, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::sse41, sum_kn_int8_int32_sse41, reduce_dim::kn, int8_t, int32_t)
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_k1_fp32_sse2, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_kn_fp32_sse2, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_k1_bf16_fp32_sse2, reduce_dim::k1, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_kn_bf16_fp32_sse2, reduce_dim::kn, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_k1_uint8_int32_sse2, reduce_dim::k1, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_kn_uint8_int32_sse2, reduce_dim::kn, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_kn_int8_int32_sse2, reduce_dim::kn, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_k1_int32_sse2, reduce_dim::k1, int32_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::sse2, sum_kn_int32_sse2, reduce_dim::kn, int32_t, int32_t)
#endif  // YNN_ARCH_X86_SSE2
#ifdef YNN_ARCH_X86_SSSE3
YNN_REDUCE_KERNEL(arch_flag::ssse3, sum_k1_uint8_int32_ssse3, reduce_dim::k1, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::ssse3, sum_kn_uint8_int32_ssse3, reduce_dim::kn, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::ssse3, sum_k1_int8_int32_ssse3, reduce_dim::k1, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::ssse3, sum_kn_int8_int32_ssse3, reduce_dim::kn, int8_t, int32_t)
#endif  // YNN_ARCH_X86_SSSE3

#if !defined(YNN_ARCH_X86_SSE2)
YNN_REDUCE_KERNEL(arch_flag::none, sum_k1_fp32, reduce_dim::k1, float, float)
YNN_REDUCE_KERNEL(arch_flag::none, sum_kn_fp32, reduce_dim::kn, float, float)
YNN_REDUCE_KERNEL(arch_flag::none, sum_k1_bf16_fp32, reduce_dim::k1, bfloat16, float)
YNN_REDUCE_KERNEL(arch_flag::none, sum_kn_bf16_fp32, reduce_dim::kn, bfloat16, float)
#endif  // !defined(YNN_ARCH_X86_SSE2)
YNN_REDUCE_KERNEL(arch_flag::none, sum_k1_fp64, reduce_dim::k1, double, double)
YNN_REDUCE_KERNEL(arch_flag::none, sum_kn_fp64, reduce_dim::kn, double, double)
YNN_REDUCE_KERNEL(arch_flag::none, sum_k1_fp16_fp32, reduce_dim::k1, half, float)
YNN_REDUCE_KERNEL(arch_flag::none, sum_kn_fp16_fp32, reduce_dim::kn, half, float)
YNN_REDUCE_KERNEL(arch_flag::none, sum_k1_int8_int32, reduce_dim::k1, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::none, sum_kn_int8_int32, reduce_dim::kn, int8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::none, sum_k1_uint8_int32, reduce_dim::k1, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::none, sum_kn_uint8_int32, reduce_dim::kn, uint8_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::none, sum_k1_int32, reduce_dim::k1, int32_t, int32_t)
YNN_REDUCE_KERNEL(arch_flag::none, sum_kn_int32, reduce_dim::kn, int32_t, int32_t)
