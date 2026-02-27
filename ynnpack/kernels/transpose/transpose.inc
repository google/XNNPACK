#ifdef YNN_ARCH_X86_AVX512
YNN_TRANSPOSE_KERNEL(arch_flag::avx512, transpose_x4_avx512, 4)
YNN_TRANSPOSE_KERNEL(arch_flag::avx512, transpose_x8_avx512, 8)
YNN_TRANSPOSE_KERNEL(arch_flag::avx512, transpose_x16_avx512, 16)
YNN_TRANSPOSE_KERNEL(arch_flag::avx512, transpose_x32_avx512, 32)
YNN_TRANSPOSE_KERNEL(arch_flag::avx512, transpose_x64_avx512, 64)
YNN_TRANSPOSE_KERNEL(arch_flag::avx512, transpose_x128_avx512, 128)
YNN_TRANSPOSE_KERNEL(arch_flag::avx512, transpose_x256_avx512, 256)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2
YNN_TRANSPOSE_KERNEL(arch_flag::avx2, transpose_x32_avx2, 32)
YNN_TRANSPOSE_KERNEL(arch_flag::avx2, transpose_x64_avx2, 64)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
YNN_TRANSPOSE_KERNEL(arch_flag::avx, transpose_x256_avx, 256)
YNN_TRANSPOSE_KERNEL(arch_flag::avx, transpose_x512_avx, 512)
YNN_TRANSPOSE_KERNEL(arch_flag::avx, transpose_x1024_avx, 1024)
YNN_TRANSPOSE_KERNEL(arch_flag::avx, transpose_x2048_avx, 2048)
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE2
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x4_sse2, 4)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x8_sse2, 8)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x16_sse2, 16)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x32_sse2, 32)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x64_sse2, 64)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x128_sse2, 128)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x256_sse2, 256)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x512_sse2, 512)
YNN_TRANSPOSE_KERNEL(arch_flag::sse2, transpose_x1024_sse2, 1024)
#endif  // YNN_ARCH_X86_SSE2

#ifdef YNN_ARCH_ARM64_SVE
YNN_TRANSPOSE_KERNEL(arch_flag::sve, transpose_x4_sve, 4)
YNN_TRANSPOSE_KERNEL(arch_flag::sve, transpose_x8_sve, 8)
YNN_TRANSPOSE_KERNEL(arch_flag::sve, transpose_x16_sve, 16)
YNN_TRANSPOSE_KERNEL(arch_flag::sve, transpose_x32_sve, 32)
YNN_TRANSPOSE_KERNEL(arch_flag::sve, transpose_x64_sve, 64)
#endif  // YNN_ARCH_ARM64_SVE

#ifdef YNN_ARCH_ARM_NEON
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x4_neon, 4)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x8_neon, 8)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x16_neon, 16)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x32_neon, 32)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x64_neon, 64)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x128_neon, 128)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x256_neon, 256)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x512_neon, 512)
YNN_TRANSPOSE_KERNEL(arch_flag::neon, transpose_x1024_neon, 1024)
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_HEXAGON_HVX
YNN_TRANSPOSE_KERNEL(arch_flag::hvx, transpose_x32_hvx, 32)
YNN_TRANSPOSE_KERNEL(arch_flag::hvx, transpose_x64_hvx, 64)
YNN_TRANSPOSE_KERNEL(arch_flag::hvx, transpose_x128_hvx, 128)
YNN_TRANSPOSE_KERNEL(arch_flag::hvx, transpose_x256_hvx, 256)
// TODO(dsharlet): Triggers "Error: sbrk out of memory" in simulator testing.
// YNN_TRANSPOSE_KERNEL(arch_flag::hvx, transpose_x512_hvx, 512)
// YNN_TRANSPOSE_KERNEL(arch_flag::hvx, transpose_x1024_hvx, 1024)
#endif  // YNN_ARCH_HEXAGON_HVX

YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x4, 4)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x8, 8)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x16, 16)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x32, 32)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x64, 64)
YNN_TRANSPOSE_KERNEL(arch_flag::none, transpose_x128, 128)
