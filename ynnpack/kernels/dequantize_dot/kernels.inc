#ifdef YNN_ARCH_X86_AVX512
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx512, dequantize_dot_f32_1x64_avx512, float)
#endif  // YNN_ARCH_X86_AVX512

#ifdef YNN_ARCH_X86_AVX2
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx2, dequantize_dot_f32_1x32_avx2, float)
#endif  // YNN_ARCH_X86_AVX2

#ifdef YNN_ARCH_X86_AVX
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::avx, dequantize_dot_f32_1x32_avx, float)
#endif  // YNN_ARCH_X86_AVX

#ifdef YNN_ARCH_X86_SSE41
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::sse41, dequantize_dot_f32_1x16_sse41, float)
#endif  // YNN_ARCH_X86_SSE41

#ifdef YNN_ARCH_ARM_NEON
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::neon, dequantize_dot_f32_1x16_neon, float)
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_WASM_SIMD128
YNN_DEQUANTIZE_DOT_KERNEL(arch_flag::simd128, dequantize_dot_f32_1x16_simd128, float)
#endif  // YNN_ARCH_WASM_SIMD128
