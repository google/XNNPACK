// clang-format off

#ifdef YNN_ARCH_ARM_NEON
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, min_fp32_4x4_neon, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, min_bf16_4x8_neon, bfloat16, bfloat16)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, min_fp16_4x8_neon, half, half)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, min_int8_4x16_neon, int8_t, int8_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::neon, min_uint8_4x16_neon, uint8_t, uint8_t)
#endif

#ifdef YNN_ARCH_X86_AVX512BW
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, min_bf16_4x32_avx512bw, bfloat16, bfloat16)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, min_fp16_4x32_avx512bw, bfloat16, bfloat16)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, min_int8_4x64_avx512bw, int8_t, int8_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, min_uint8_4x64_avx512bw, uint8_t, uint8_t)
#endif  // YNN_ARCH_X86_AVX512BW
#ifdef YNN_ARCH_X86_AVX512F
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512f, min_fp32_4x16_avx512f, float, float)
#endif  // YNN_ARCH_X86_AVX512F
#ifdef YNN_ARCH_X86_AVX2
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, min_bf16_4x16_avx2, bfloat16, bfloat16)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, min_fp16_4x16_avx2, half, half)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, min_int8_4x32_avx2, int8_t, int8_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, min_uint8_4x32_avx2, uint8_t, uint8_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX2
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, min_fp32_4x8_avx, float, float)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_SSE41
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse41, min_int8_4x16_sse41, int8_t, int8_t)
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, min_fp32_4x4_sse2, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, min_bf16_4x8_sse2, bfloat16, bfloat16)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, min_fp16_4x8_sse2, half, half)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, min_uint8_4x16_sse2, uint8_t, uint8_t)
#endif  // YNN_ARCH_X86_SSE2

YNN_UNARY_REDUCE_KERNEL(arch_flag::none, min_fp32, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, min_fp16, half, half)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, min_bf16, bfloat16, bfloat16)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, min_int8, int8_t, int8_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, min_uint8, uint8_t, uint8_t)
