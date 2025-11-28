// clang-format off

#ifdef YNN_ARCH_X86_AVX2
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_fp32_avx2, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_int8_int32_avx2, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::avx2, sum_squared_uint8_int32_avx2, uint8_t, int32_t)
#endif  // YNN_ARCH_X86_AVX2

YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_fp32, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_fp16_fp32, half, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_bf16_fp32, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_int8_int32, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_uint8_int32, uint8_t, int32_t)
