// clang-format off

#ifdef YNN_ARCH_ARM_NEONBF16
YNN_UNARY_REDUCE_KERNEL(arch_flag::neonbf16, sum_squared_bf16_fp32_neonbf16, bfloat16, float)
#endif // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM_NEONFP16
YNN_UNARY_REDUCE_KERNEL(arch_flag::neonfp16, sum_squared_fp16_fp32_neonfp16, half, float)
#endif  // YNN_ARCH_ARM_NEONFP16
#ifdef YNN_ARCH_ARM_NEONFMA
YNN_UNARY_REDUCE_KERNEL(arch_flag::neonfma, sum_squared_fp32_neonfma, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neonfma, sum_squared_bf16_fp32_neonfma, bfloat16, float)
#endif  // YNN_ARCH_ARM_NEONFMA
#ifdef YNN_ARCH_ARM_NEONDOT
YNN_UNARY_REDUCE_KERNEL(arch_flag::neondot, sum_squared_int8_int32_neondot, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neondot, sum_squared_uint8_int32_neondot, uint8_t, int32_t)
#endif  // YNN_ARCH_ARM_NEONDOT
#ifdef YNN_ARCH_ARM_NEON
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, sum_squared_fp32_neon, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, sum_squared_bf16_fp32_neon, bfloat16, float)
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_HEXAGON_HVX
YNN_UNARY_REDUCE_KERNEL(arch_flag::hvx, sum_squared_int8_int32_hvx, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::hvx, sum_squared_uint8_int32_hvx, uint8_t, int32_t)
#endif  // YNN_ARCH_HEXAGON_HVX

#ifdef YNN_ARCH_X86_AVX512BF16
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bf16, sum_squared_bf16_fp32_avx512bf16, bfloat16, float)
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512, sum_squared_bf16_fp32_avx512, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512, sum_squared_fp16_fp32_avx512, half, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512, sum_squared_uint8_int32_avx512, uint8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512, sum_squared_int8_int32_avx512, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512, sum_squared_fp32_avx512, float, float)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2_FMA3
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2 | arch_flag::fma3, sum_squared_bf16_fp32_avx2_fma3, bfloat16, float)
#endif  // YNN_ARCH_X86_AVX2_FMA3
#ifdef YNN_ARCH_X86_AVX2
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_bf16_fp32_avx2, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_int8_int32_avx2, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_uint8_int32_avx2, uint8_t, int32_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_fp32_avx, float, float)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_F16C
YNN_UNARY_REDUCE_KERNEL(arch_flag::f16c, sum_squared_fp16_fp32_f16c, half, float)
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_SSE41
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse41, sum_squared_int8_int32_sse41, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse41, sum_squared_uint8_int32_sse41, uint8_t, int32_t)
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, sum_squared_fp32_sse2, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, sum_squared_bf16_fp32_sse2, bfloat16, float)
#endif  // YNN_ARCH_X86_SSE2

// We can assume that SSE2 exists on any x86 CPU we will run on, so we don't
// need to include these kernels. Reasons to do this:
// - Saves a little code size to not include kernels we will never need.
// - consistent_reduce_test will assert that all available kernels produce
//   arithmetically consistent results, which is not true for these kernels,
//   especially bf16, because most bf16 kernels do a pairwise reduction as the
//   "innermost" reduction step.
#if !defined(YNN_ARCH_X86_SSE2)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_fp32, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_bf16_fp32, bfloat16, float)
#endif  // !defined(YNN_ARCH_X86_SSE2)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_fp16_fp32, half, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_int8_int32, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_uint8_int32, uint8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_int32, int32_t, int32_t)
