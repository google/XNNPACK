// clang-format off

#ifdef YNN_ARCH_X86_AVX512
YNN_REDUCE_KERNEL(arch_flag::avx512, propagate_nan_k1_fp32_avx512, reduce_dim::k1, float, float, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::avx512, propagate_nan_kn_fp32_avx512, reduce_dim::kn, float, float, reduce_flag::none)
#ifdef YNN_ENABLE_FP64
YNN_REDUCE_KERNEL(arch_flag::avx512, propagate_nan_k1_fp64_avx512, reduce_dim::k1, double, double, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::avx512, propagate_nan_kn_fp64_avx512, reduce_dim::kn, double, double, reduce_flag::none)
#endif  // YNN_ENABLE_FP64
YNN_REDUCE_KERNEL(arch_flag::avx512, propagate_nan_k1_bf16_avx512, reduce_dim::k1, bfloat16, bfloat16, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::avx512, propagate_nan_kn_bf16_avx512, reduce_dim::kn, bfloat16, bfloat16, reduce_flag::none)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2
YNN_REDUCE_KERNEL(arch_flag::avx2, propagate_nan_k1_bf16_avx2, reduce_dim::k1, bfloat16, bfloat16, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::avx2, propagate_nan_kn_bf16_avx2, reduce_dim::kn, bfloat16, bfloat16, reduce_flag::none)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_REDUCE_KERNEL(arch_flag::avx, propagate_nan_k1_fp32_avx, reduce_dim::k1, float, float, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::avx, propagate_nan_kn_fp32_avx, reduce_dim::kn, float, float, reduce_flag::none)
#ifdef YNN_ENABLE_FP64
YNN_REDUCE_KERNEL(arch_flag::avx, propagate_nan_k1_fp64_avx, reduce_dim::k1, double, double, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::avx, propagate_nan_kn_fp64_avx, reduce_dim::kn, double, double, reduce_flag::none)
#endif  // YNN_ENABLE_FP64
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE2
YNN_REDUCE_KERNEL(arch_flag::sse2, propagate_nan_k1_fp32_sse2, reduce_dim::k1, float, float, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::sse2, propagate_nan_kn_fp32_sse2, reduce_dim::kn, float, float, reduce_flag::none)
#endif  // YNN_ARCH_X86_SSE2

YNN_REDUCE_KERNEL(arch_flag::none, propagate_nan_k1_fp32, reduce_dim::k1, float, float, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::none, propagate_nan_kn_fp32, reduce_dim::kn, float, float, reduce_flag::none)
#ifdef YNN_ENABLE_FP64
YNN_REDUCE_KERNEL(arch_flag::none, propagate_nan_k1_fp64, reduce_dim::k1, double, double, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::none, propagate_nan_kn_fp64, reduce_dim::kn, double, double, reduce_flag::none)
#endif  // YNN_ENABLE_FP64
YNN_REDUCE_KERNEL(arch_flag::none, propagate_nan_k1_fp16, reduce_dim::k1, half, half, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::none, propagate_nan_kn_fp16, reduce_dim::kn, half, half, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::none, propagate_nan_k1_bf16, reduce_dim::k1, bfloat16, bfloat16, reduce_flag::none)
YNN_REDUCE_KERNEL(arch_flag::none, propagate_nan_kn_bf16, reduce_dim::kn, bfloat16, bfloat16, reduce_flag::none)
