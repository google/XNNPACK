#ifdef YNN_ARCH_X86_AVX512BF16
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx512bf16, dequantize_dot_bf16_avx512bf16, bfloat16)
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx512, dequantize_dot_f32_avx512, float)
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx512, dequantize_dot_bf16_avx512, bfloat16)
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx512, dequantize_dot_fp16_avx512, half)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_F16C
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::f16c, dequantize_dot_fp16_f16c, half)
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_AVX2
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx2, dequantize_dot_f32_avx2, float)
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx2, dequantize_dot_bf16_avx2, bfloat16)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx, dequantize_dot_f32_avx, float)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE41
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::sse41, dequantize_dot_f32_sse41, float)
#endif  // YNN_ARCH_X86_SSE41

#ifdef YNN_ARCH_ARM_NEONBF16
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::neonbf16, dequantize_dot_bf16_neonbf16, bfloat16)
#endif  // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM_NEONFP16
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::neonfp16, dequantize_dot_fp16_neonfp16, half)
#endif  // YNN_ARCH_ARM_NEONFP16
#ifdef YNN_ARCH_ARM_NEON
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::neon, dequantize_dot_f32_neon, float)
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::neon, dequantize_dot_bf16_neon, bfloat16)
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_WASM_SIMD128
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::simd128, dequantize_dot_f32_simd128, float)
#endif  // YNN_ARCH_WASM_SIMD128
