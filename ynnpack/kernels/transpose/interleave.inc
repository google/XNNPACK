#ifdef YNN_ARCH_X86_AVX512
YNN_INTERLEAVE_KERNEL(arch_flag::avx512, interleave2_x4_avx512, 2, 4)
YNN_INTERLEAVE_KERNEL(arch_flag::avx512, interleave2_x8_avx512, 2, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::avx512, interleave2_x16_avx512, 2, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::avx512, interleave2_x32_avx512, 2, 32)
YNN_INTERLEAVE_KERNEL(arch_flag::avx512, interleave4_x8_avx512, 4, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::avx512, interleave4_x16_avx512, 4, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::avx512, interleave4_x32_avx512, 4, 32)
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave2_x4_avx2, 2, 4)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave2_x8_avx2, 2, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave2_x16_avx2, 2, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave2_x32_avx2, 2, 32)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave4_x8_avx2, 4, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave4_x16_avx2, 4, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave4_x32_avx2, 4, 32)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_SSE2
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave2_x4_sse2, 2, 4)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave2_x8_sse2, 2, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave2_x16_sse2, 2, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave2_x32_sse2, 2, 32)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave4_x8_sse2, 4, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave4_x16_sse2, 4, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave4_x32_sse2, 4, 32)
#endif  // YNN_ARCH_X86_SSE2

#ifdef YNN_ARCH_ARM64_SVE
YNN_INTERLEAVE_KERNEL(arch_flag::sve, interleave2_x4_sve, 2, 4)
YNN_INTERLEAVE_KERNEL(arch_flag::sve, interleave2_x8_sve, 2, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::sve, interleave2_x16_sve, 2, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::sve, interleave2_x32_sve, 2, 32)
YNN_INTERLEAVE_KERNEL(arch_flag::sve, interleave4_x8_sve, 4, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::sve, interleave4_x16_sve, 4, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::sve, interleave4_x32_sve, 4, 32)
#endif  // YNN_ARCH_ARM64_SVE

#ifdef YNN_ARCH_ARM_NEON
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave2_x4_neon, 2, 4)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave2_x8_neon, 2, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave2_x16_neon, 2, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave2_x32_neon, 2, 32)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave4_x8_neon, 4, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave4_x16_neon, 4, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave4_x32_neon, 4, 32)
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_HEXAGON_HVX
YNN_INTERLEAVE_KERNEL(arch_flag::hvx, interleave2_x8_hvx, 2, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::hvx, interleave2_x16_hvx, 2, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::hvx, interleave2_x32_hvx, 2, 32)
YNN_INTERLEAVE_KERNEL(arch_flag::hvx, interleave4_x8_hvx, 4, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::hvx, interleave4_x16_hvx, 4, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::hvx, interleave4_x32_hvx, 4, 32)
#endif  // YNN_ARCH_HEXAGON_HVX

YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x4, 0, 4)
YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x8, 0, 8)
YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x16, 0, 16)
YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x32, 0, 32)
YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x64, 0, 64)
