// clang-format off

#ifdef YNN_ARCH_ARM64_NEON
YNN_LUT_KERNEL(arch_flag::neon, lut_u2_u8_arm64_neon, uint2x4, 8)
YNN_LUT_KERNEL(arch_flag::neon, lut_u4_u8_arm64_neon, uint4x2, 8)
YNN_LUT_KERNEL(arch_flag::neon, lut_u4_u16_arm64_neon, uint4x2, 16)
YNN_LUT_KERNEL(arch_flag::neon, lut_u8_u8_arm64_neon, uint8_t, 8)
#endif  // YNN_ARCH_ARM64_NEON
#ifdef YNN_ARCH_ARM_NEON
YNN_LUT_KERNEL(arch_flag::neon, lut_u2_u8_arm_neon, uint2x4, 8)
YNN_LUT_KERNEL(arch_flag::neon, lut_u2_u16_arm_neon, uint2x4, 16)
YNN_LUT_KERNEL(arch_flag::neon, lut_u4_u8_arm_neon, uint4x2, 8)
YNN_LUT_KERNEL(arch_flag::neon, lut_u4_u16_arm_neon, uint4x2, 16)
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_X86_SSSE3
YNN_LUT_KERNEL(arch_flag::ssse3, lut_u2_u8_ssse3, uint2x4, 8)
YNN_LUT_KERNEL(arch_flag::ssse3, lut_u2_u16_ssse3, uint2x4, 16)
YNN_LUT_KERNEL(arch_flag::ssse3, lut_u4_u8_ssse3, uint4x2, 8)
YNN_LUT_KERNEL(arch_flag::ssse3, lut_u4_u16_ssse3, uint4x2, 16)
#endif  // YNN_ARCH_X86_SSSE3

YNN_LUT_KERNEL(arch_flag::none, lut_u2_u8, uint2x4, 8)
YNN_LUT_KERNEL(arch_flag::none, lut_u2_u16, uint2x4, 16)
YNN_LUT_KERNEL(arch_flag::none, lut_u4_u8, uint4x2, 8)
YNN_LUT_KERNEL(arch_flag::none, lut_u4_u16, uint4x2, 16)
YNN_LUT_KERNEL(arch_flag::none, lut_u8_u8, uint8_t, 8)
YNN_LUT_KERNEL(arch_flag::none, lut_u8_u16, uint8_t, 16)
YNN_LUT_KERNEL(arch_flag::none, lut_u8_u32, uint8_t, 32)
YNN_LUT_KERNEL(arch_flag::none, lut_s32_u16, int32_t, 16)
YNN_LUT_KERNEL(arch_flag::none, lut_s32_u32, int32_t, 32)
