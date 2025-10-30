#ifdef YNN_ARCH_X86_AVX2
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave2_x4_avx2, 2, uint4x2)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave2_x8_avx2, 2, uint8_t)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave2_x16_avx2, 2, uint16_t)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave2_x32_avx2, 2, uint32_t)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave4_x8_avx2, 4, uint8_t)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave4_x16_avx2, 4, uint16_t)
YNN_INTERLEAVE_KERNEL(arch_flag::avx2, interleave4_x32_avx2, 4, uint32_t)
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_SSE2
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave2_x4_sse2, 2, uint4x2)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave2_x8_sse2, 2, uint8_t)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave2_x16_sse2, 2, uint16_t)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave2_x32_sse2, 2, uint32_t)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave4_x8_sse2, 4, uint8_t)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave4_x16_sse2, 4, uint16_t)
YNN_INTERLEAVE_KERNEL(arch_flag::sse2, interleave4_x32_sse2, 4, uint32_t)
#endif  // YNN_ARCH_X86_SSE2

#ifdef YNN_ARCH_ARM_NEON
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave2_x4_neon, 2, uint4x2)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave2_x8_neon, 2, uint8_t)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave2_x16_neon, 2, uint16_t)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave2_x32_neon, 2, uint32_t)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave4_x8_neon, 4, uint8_t)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave4_x16_neon, 4, uint16_t)
YNN_INTERLEAVE_KERNEL(arch_flag::neon, interleave4_x32_neon, 4, uint32_t)
#endif  // YNN_ARCH_ARM_NEON

YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x4, 0, uint4x2)
YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x8, 0, uint8_t)
YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x16, 0, uint16_t)
YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x32, 0, uint32_t)
YNN_INTERLEAVE_KERNEL(arch_flag::none, interleave_x64, 0, uint64_t)
