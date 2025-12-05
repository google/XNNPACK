// clang-format off

#ifdef YNN_ARCH_X86_AVX512BF16
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bf16, sum_squared_bf16_fp32_avx512bf16, bfloat16, float)
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512FP16
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512fp16, sum_squared_fp16_fp32_avx512fp16, half, float)
#endif  // YNN_ARCH_X86_AVX512FP16
#ifdef YNN_ARCH_X86_AVX512BW
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, sum_squared_uint8_int32_avx512bw, uint8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512bw, sum_squared_int8_int32_avx512bw, int8_t, int32_t)
#endif  // YNN_ARCH_X86_AVX512BW
#ifdef YNN_ARCH_X86_AVX512F
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx512f, sum_squared_fp32_avx512f, float, float)
#endif  // YNN_ARCH_X86_AVX512F
#ifdef YNN_ARCH_X86_AVX2
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_fp32_avx2, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_int8_int32_avx2, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_uint8_int32_avx2, uint8_t, int32_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_F16C
YNN_UNARY_REDUCE_KERNEL(arch_flag::f16c, sum_squared_fp16_fp32_f16c, half, float)
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_SSE41
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse41, sum_squared_int8_int32_sse41, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse41, sum_squared_uint8_int32_sse41, uint8_t, int32_t)
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
YNN_UNARY_REDUCE_KERNEL(arch_flag::sse2, sum_squared_fp32_sse2, float, float)
#endif  // YNN_ARCH_X86_SSE2

YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_fp32, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_fp16_fp32, half, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_bf16_fp32, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_int8_int32, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_uint8_int32, uint8_t, int32_t)
