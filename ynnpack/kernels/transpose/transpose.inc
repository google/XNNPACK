#ifdef YNN_ARCH_X86_AVX2
YNN_TRANSPOSE_KERNEL(arch_flag::avx2, transpose_x32_avx2, uint32_t)
YNN_TRANSPOSE_KERNEL(arch_flag::avx2, transpose_x64_avx2, uint64_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_TRANSPOSE_KERNEL(arch_flag::avx, transpose_x256_avx, x256_t)
YNN_TRANSPOSE_KERNEL(arch_flag::avx, transpose_x512_avx, x512_t)
YNN_TRANSPOSE_KERNEL(arch_flag::avx, transpose_x1024_avx, x1024_t)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE2
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x4_sse2, uint4x2)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x8_sse2, uint8_t)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x16_sse2, uint16_t)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x32_sse2, uint32_t)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x64_sse2, uint64_t)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x128_sse2, x128_t)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x256_sse2, x256_t)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x512_sse2, x512_t)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x1024_sse2, x1024_t)
#endif  // YNN_ARCH_X86_SSE2

#ifdef YNN_ARCH_ARM_NEON
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x4_neon, uint4x2)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x8_neon, uint8_t)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x16_neon, uint16_t)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x32_neon, uint32_t)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x64_neon, uint64_t)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x128_neon, x128_t)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x256_neon, x256_t)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x512_neon, x512_t)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x1024_neon, x1024_t)
#endif  // YNN_ARCH_ARM_NEON

YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x4, uint4x2)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x8, uint8_t)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x16, uint16_t)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x32, uint32_t)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x64, uint64_t)
