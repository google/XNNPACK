// clang-format off

#ifdef YNN_ARCH_ARM_NEONBF16
YNN_REDUCE_K1_KERNEL(arch_flag::neonbf16, sum_squared_bf16_fp32_k1_neonbf16, bfloat16, float)
YNN_REDUCE_KN_KERNEL(arch_flag::neonbf16, sum_squared_bf16_fp32_kn_neonbf16, bfloat16, float)
#endif // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM_NEONFP16
YNN_REDUCE_K1_KERNEL(arch_flag::neonfp16, sum_squared_fp16_fp32_k1_neonfp16, half, float)
YNN_REDUCE_KN_KERNEL(arch_flag::neonfp16, sum_squared_fp16_fp32_kn_neonfp16, half, float)
#endif  // YNN_ARCH_ARM_NEONFP16
#ifdef YNN_ARCH_ARM_NEONFMA
YNN_REDUCE_K1_KERNEL(arch_flag::neon, sum_squared_fp32_k1_neonfma, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, sum_squared_fp32_kn_neonfma, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::neon, sum_squared_bf16_fp32_k1_neonfma, bfloat16, float)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, sum_squared_bf16_fp32_kn_neonfma, bfloat16, float)
#endif  // YNN_ARCH_ARM_NEONFMA
#ifdef YNN_ARCH_ARM_NEONDOT
YNN_REDUCE_K1_KERNEL(arch_flag::neondot, sum_squared_int8_int32_k1_neondot, int8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::neondot, sum_squared_int8_int32_kn_neondot, int8_t, int32_t)
YNN_REDUCE_K1_KERNEL(arch_flag::neondot, sum_squared_uint8_int32_k1_neondot, uint8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::neondot, sum_squared_uint8_int32_kn_neondot, uint8_t, int32_t)
#endif  // YNN_ARCH_ARM_NEONDOT
#ifdef YNN_ARCH_ARM_NEON
YNN_REDUCE_K1_KERNEL(arch_flag::neon, sum_squared_fp32_k1_neon, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, sum_squared_fp32_kn_neon, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::neon, sum_squared_bf16_fp32_k1_neon, bfloat16, float)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, sum_squared_bf16_fp32_kn_neon, bfloat16, float)
#ifdef YNN_ARCH_ARM64
YNN_REDUCE_K1_KERNEL(arch_flag::neon, sum_squared_fp64_k1_neon, double, double)
YNN_REDUCE_KN_KERNEL(arch_flag::neon, sum_squared_fp64_kn_neon, double, double)
#endif
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_HEXAGON_HVX
YNN_REDUCE_K1_KERNEL(arch_flag::hvx, sum_squared_int8_int32_k1_hvx, int8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::hvx, sum_squared_int8_int32_kn_hvx, int8_t, int32_t)
YNN_REDUCE_K1_KERNEL(arch_flag::hvx, sum_squared_uint8_int32_k1_hvx, uint8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::hvx, sum_squared_uint8_int32_kn_hvx, uint8_t, int32_t)
#endif  // YNN_ARCH_HEXAGON_HVX

#ifdef YNN_ARCH_WASM_SIMD128
YNN_REDUCE_K1_KERNEL(arch_flag::simd128, sum_squared_fp32_k1_simd128, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::simd128, sum_squared_fp32_kn_simd128, float, float)
#endif  // YNN_ARCH_WASM_SIMD128

#ifdef YNN_ARCH_X86_AVX512BF16
YNN_REDUCE_K1_KERNEL(arch_flag::avx512bf16, sum_squared_bf16_fp32_k1_avx512bf16, bfloat16, float)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512bf16, sum_squared_bf16_fp32_kn_avx512bf16, bfloat16, float)
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, sum_squared_bf16_fp32_k1_avx512, bfloat16, float)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, sum_squared_bf16_fp32_kn_avx512, bfloat16, float)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, sum_squared_fp16_fp32_k1_avx512, half, float)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, sum_squared_fp16_fp32_kn_avx512, half, float)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, sum_squared_uint8_int32_k1_avx512, uint8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, sum_squared_uint8_int32_kn_avx512, uint8_t, int32_t)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, sum_squared_int8_int32_k1_avx512, int8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, sum_squared_int8_int32_kn_avx512, int8_t, int32_t)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, sum_squared_fp32_k1_avx512, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, sum_squared_fp32_kn_avx512, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::avx512, sum_squared_fp64_k1_avx512, double, double)
YNN_REDUCE_KN_KERNEL(arch_flag::avx512, sum_squared_fp64_kn_avx512, double, double)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2
YNN_REDUCE_K1_KERNEL(arch_flag::avx2, sum_squared_bf16_fp32_k1_avx2, bfloat16, float)
YNN_REDUCE_KN_KERNEL(arch_flag::avx2, sum_squared_bf16_fp32_kn_avx2, bfloat16, float)
YNN_REDUCE_K1_KERNEL(arch_flag::avx2, sum_squared_uint8_int32_k1_avx2, uint8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::avx2, sum_squared_uint8_int32_kn_avx2, uint8_t, int32_t)
YNN_REDUCE_K1_KERNEL(arch_flag::avx2, sum_squared_int8_int32_k1_avx2, int8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::avx2, sum_squared_int8_int32_kn_avx2, int8_t, int32_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_REDUCE_K1_KERNEL(arch_flag::avx, sum_squared_fp32_k1_avx, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::avx, sum_squared_fp32_kn_avx, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::avx, sum_squared_fp64_k1_avx, double, double)
YNN_REDUCE_KN_KERNEL(arch_flag::avx, sum_squared_fp64_kn_avx, double, double)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_F16C
YNN_REDUCE_K1_KERNEL(arch_flag::f16c, sum_squared_fp16_fp32_k1_f16c, half, float)
YNN_REDUCE_KN_KERNEL(arch_flag::f16c, sum_squared_fp16_fp32_kn_f16c, half, float)
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_SSE41
YNN_REDUCE_K1_KERNEL(arch_flag::sse41, sum_squared_uint8_int32_k1_sse41, uint8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::sse41, sum_squared_uint8_int32_kn_sse41, uint8_t, int32_t)
YNN_REDUCE_K1_KERNEL(arch_flag::sse41, sum_squared_int8_int32_k1_sse41, int8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::sse41, sum_squared_int8_int32_kn_sse41, int8_t, int32_t)
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
YNN_REDUCE_K1_KERNEL(arch_flag::sse2, sum_squared_fp32_k1_sse2, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::sse2, sum_squared_fp32_kn_sse2, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::sse2, sum_squared_bf16_fp32_k1_sse2, bfloat16, float)
YNN_REDUCE_KN_KERNEL(arch_flag::sse2, sum_squared_bf16_fp32_kn_sse2, bfloat16, float)
#endif  // YNN_ARCH_X86_SSE2

// We can assume that SSE2 exists on any x86 CPU we will run on, so we don't
// need to include these kernels. Reasons to do this:
// - Saves a little code size to not include kernels we will never need.
// - consistent_reduce_test will assert that all available kernels produce
//   arithmetically consistent results, which is not true for these kernels,
//   especially bf16, because most bf16 kernels do a pairwise reduction as the
//   "innermost" reduction step.
#if !defined(YNN_ARCH_X86_SSE2)
YNN_REDUCE_K1_KERNEL(arch_flag::none, sum_squared_fp32_k1, float, float)
YNN_REDUCE_KN_KERNEL(arch_flag::none, sum_squared_fp32_kn, float, float)
YNN_REDUCE_K1_KERNEL(arch_flag::none, sum_squared_bf16_fp32_k1, bfloat16, float)
YNN_REDUCE_KN_KERNEL(arch_flag::none, sum_squared_bf16_fp32_kn, bfloat16, float)
#endif  // !defined(YNN_ARCH_X86_SSE2)
YNN_REDUCE_K1_KERNEL(arch_flag::none, sum_squared_fp64_k1, double, double)
YNN_REDUCE_KN_KERNEL(arch_flag::none, sum_squared_fp64_kn, double, double)
YNN_REDUCE_K1_KERNEL(arch_flag::none, sum_squared_fp16_fp32_k1, half, float)
YNN_REDUCE_KN_KERNEL(arch_flag::none, sum_squared_fp16_fp32_kn, half, float)
YNN_REDUCE_K1_KERNEL(arch_flag::none, sum_squared_int8_int32_k1, int8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::none, sum_squared_int8_int32_kn, int8_t, int32_t)
YNN_REDUCE_K1_KERNEL(arch_flag::none, sum_squared_uint8_int32_k1, uint8_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::none, sum_squared_uint8_int32_kn, uint8_t, int32_t)
YNN_REDUCE_K1_KERNEL(arch_flag::none, sum_squared_int32_k1, int32_t, int32_t)
YNN_REDUCE_KN_KERNEL(arch_flag::none, sum_squared_int32_kn, int32_t, int32_t)
