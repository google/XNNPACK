// clang-format off

#ifdef YNN_ARCH_ARM_NEONBF16
YNN_UNARY_REDUCE_KERNEL(arch_flag::neonbf16, sum_bf16_fp32_neonbf16, bfloat16, float)
#endif // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM_NEONFP16
YNN_UNARY_REDUCE_KERNEL(arch_flag::neonfp16, sum_fp16_fp32_neonfp16, half, float)
#endif  // YNN_ARCH_ARM_NEONFP16
#ifdef YNN_ARCH_ARM_NEONDOT
YNN_UNARY_REDUCE_KERNEL(arch_flag::neondot, sum_int8_int32_neondot, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neondot, sum_uint8_int32_neondot, uint8_t, int32_t)
#endif  // YNN_ARCH_ARM_NEONDOT
#ifdef YNN_ARCH_ARM_NEON
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, sum_fp32_neon, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, sum_bf16_fp32_neon, bfloat16, float)
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_X86_AVX512BF16
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bf16, sum_bf16_fp32_avx512bf16, bfloat16, float)
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512BW
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, sum_bf16_fp32_avx512bw, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, sum_fp16_fp32_avx512bw, half, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, sum_uint8_int32_avx512bw, uint8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, sum_int8_int32_avx512bw, int8_t, int32_t)
#endif  // YNN_ARCH_X86_AVX512BW
#ifdef YNN_ARCH_X86_AVX512F
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512f, sum_fp32_avx512f, float, float)
#endif  // YNN_ARCH_X86_AVX512F
#ifdef YNN_ARCH_X86_AVX2
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_bf16_fp32_avx2, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_uint8_int32_avx2, uint8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_int8_int32_avx2, int8_t, int32_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_fp32_avx, float, float)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_F16C
YNN_UNARY_REDUCE_KERNEL(arch_flag::f16c, sum_fp16_fp32_f16c, half, float)
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_SSE41
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse41, sum_uint8_int32_sse41, uint8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse41, sum_int8_int32_sse41, int8_t, int32_t)
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, sum_fp32_sse2, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, sum_bf16_fp32_sse2, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, sum_uint8_int32_sse2, uint8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, sum_int8_int32_sse2, int8_t, int32_t)
#endif  // YNN_ARCH_X86_SSE2
#ifdef YNN_ARCH_X86_SSSE3
YNN_UNARY_REDUCE_KERNEL(arch_flag::ssse3, sum_uint8_int32_ssse3, uint8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::ssse3, sum_int8_int32_ssse3, int8_t, int32_t)
#endif  // YNN_ARCH_X86_SSSE3

YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_fp32, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_fp16_fp32, half, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_bf16_fp32, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_int8_int32, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_uint8_int32, uint8_t, int32_t)
