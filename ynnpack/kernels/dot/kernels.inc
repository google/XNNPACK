// clang-format off

#ifdef YNN_ARCH_X86_AMXBF16
YNN_DOT_KERNEL(arch_flag::amxbf16, dot_bf16_bf16_fp32_16x64x32_16x16x2_amxbf16, 16, 64, 32, 16, 2, 0, bfloat16, bfloat16, float)
#endif  // YNN_ARCH_X86_AMXBF16
#ifdef YNN_ARCH_X86_AMXFP16
YNN_DOT_KERNEL(arch_flag::amxfp16, dot_fp16_fp16_fp32_16x64x32_16x16x2_amxfp16, 16, 64, 32, 16, 2, 0, half, half, float)
#endif  // YNN_ARCH_X86_AMXFP16
#ifdef YNN_ARCH_X86_AMXINT8
YNN_DOT_KERNEL(arch_flag::amxint8, dot_int8_int8_int32_16x64x64_16x16x4_amxint8, 16, 64, 64, 16, 4, 0, int8_t, int8_t, int32_t)
YNN_DOT_KERNEL(arch_flag::amxint8, dot_uint8_int8_int32_16x64x64_16x16x4_amxint8, 16, 64, 64, 16, 4, 0, uint8_t, int8_t, int32_t)
#endif  // YNN_ARCH_X86_AMXINT8

#ifdef YNN_ARCH_X86_AVX512VNNI
#include "ynnpack/kernels/dot/x86_avx512vnni_uint8_int8_int32.inc"
#include "ynnpack/kernels/dot/x86_avx512vnni_uint8_int8_int32_k16.inc"
#endif  // YNN_ARCH_X86_AVX512VNNI
#ifdef YNN_ARCH_X86_AVX512BW
#include "ynnpack/kernels/dot/x86_avx512bw_int8_int8_int32.inc"
#include "ynnpack/kernels/dot/x86_avx512bw_int8_int8_int32_k16.inc"
#endif  // YNN_ARCH_X86_AVX512BW
#ifdef YNN_ARCH_X86_AVX512BF16
#include "ynnpack/kernels/dot/x86_avx512bf16_bf16_bf16_fp32.inc"
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512F
#include "ynnpack/kernels/dot/x86_avx512f_fp32.inc"
#include "ynnpack/kernels/dot/x86_avx512f_fp32_k2.inc"
#include "ynnpack/kernels/dot/x86_avx512f_fp32_k4.inc"
#include "ynnpack/kernels/dot/x86_avx512f_bf16_bf16_fp32.inc"
#endif  // YNN_ARCH_X86_AVX512F
#ifdef YNN_ARCH_X86_AVX2
#include "ynnpack/kernels/dot/x86_avx2_int8_int8_int32.inc"
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_FMA3
#include "ynnpack/kernels/dot/x86_fma3_fp32.inc"
#endif  // YNN_ARCH_X86_FMA3
#ifdef YNN_ARCH_X86_AVX2_FMA3
#include "ynnpack/kernels/dot/x86_avx2_fma3_bf16_bf16_fp32.inc"
#include "ynnpack/kernels/dot/x86_avx2_fma3_fp32_k2.inc"
#endif  // YNN_ARCH_X86_AVX2_FMA3
#ifdef YNN_ARCH_X86_AVX
#include "ynnpack/kernels/dot/x86_avx_fp32.inc"
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_AVX2
#include "ynnpack/kernels/dot/x86_avx2_fp32_k2.inc"
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_SSE2
#include "ynnpack/kernels/dot/x86_sse2_fp32.inc"
#endif  // YNN_ARCH_X86_SSE2

#ifndef YNN_DISABLE_SME
#ifdef YNN_ARCH_ARM64_SME2
YNN_DOT_KERNEL(arch_flag::sme2, dot_fp32_sme2, sme_vl(float{}),
               4 * sme_vl(float{}), 1, sme_vl(float{}), 1,
               dot_flag::transpose_a, float, float, float)
YNN_DOT_KERNEL(arch_flag::sme2, dot_bf16_bf16_fp32_sme2, sme_vl(float{}),
               4 * sme_vl(float{}), 2, sme_vl(float{}), 2,
               dot_flag::transpose_a, bfloat16, bfloat16, float)
YNN_DOT_KERNEL(arch_flag::sme2, dot_fp16_fp16_fp32_sme2, sme_vl(float{}),
               4 * sme_vl(float{}), 2, sme_vl(float{}), 2,
               dot_flag::transpose_a, half, half, float)
YNN_DOT_KERNEL(arch_flag::sme2, dot_int8_int8_int32_sme2, sme_vl(int32_t{}),
               4 * sme_vl(int32_t{}), 4, sme_vl(int32_t{}), 4,
               dot_flag::transpose_a, int8_t, int8_t, int32_t)
#endif  // YNN_ARCH_ARM64_SME2

#ifdef YNN_ARCH_ARM64_SME
YNN_DOT_KERNEL(arch_flag::sme, dot_fp32_sme, sme_vl(float{}),
               4 * sme_vl(float{}), 1, sme_vl(float{}), 1,
               dot_flag::transpose_a, float, float, float)
YNN_DOT_KERNEL(arch_flag::sme, dot_bf16_bf16_fp32_sme, sme_vl(float{}),
               4 * sme_vl(float{}), 2, sme_vl(float{}), 2,
               dot_flag::transpose_a, bfloat16, bfloat16, float)
YNN_DOT_KERNEL(arch_flag::sme, dot_fp16_fp16_fp32_sme, sme_vl(float{}),
               4 * sme_vl(float{}), 2, sme_vl(float{}), 2,
               dot_flag::transpose_a, half, half, float)
YNN_DOT_KERNEL(arch_flag::sme, dot_int8_int8_int32_sme, sme_vl(int32_t{}),
               4 * sme_vl(int32_t{}), 4, sme_vl(int32_t{}), 4,
               dot_flag::transpose_a, int8_t, int8_t, int32_t)
#endif  // YNN_ARCH_ARM64_SME
#endif  // YNN_DISABLE_SME

#ifdef YNN_ARCH_ARM64_NEONI8MM
#include "ynnpack/kernels/dot/arm64_neoni8mm_int8_int8_int32.inc"
#endif  // YNN_ARCH_ARM64_NEONI8MM
#ifdef YNN_ARCH_ARM64_NEON
#include "ynnpack/kernels/dot/arm64_neon_fp32.inc"
#endif  // YNN_ARCH_ARM64

#ifdef YNN_ARCH_ARM_NEONDOT
#include "ynnpack/kernels/dot/arm_neondot_int8_int8_int32.inc"
#endif  // YNN_ARCH_ARM_NEONDOT
#ifdef YNN_ARCH_ARM_NEON
#include "ynnpack/kernels/dot/arm_neon_int8_int8_int32.inc"
#endif  // YNN_ARCH_ARM_NEON

YNN_DOT_KERNEL(arch_flag::none, dot_fp32, 1, 128, 1, 1, 1, 0, float, float, float)
YNN_DOT_KERNEL(arch_flag::none, dot_fp16_fp16_fp32, 1, 128, 1, 1, 1, 0, half, half, float)
YNN_DOT_KERNEL(arch_flag::none, dot_bf16_bf16_fp32, 1, 128, 1, 1, 1, 0, bfloat16, bfloat16, float)
YNN_DOT_KERNEL(arch_flag::none, dot_int8_int8_int32, 1, 128, 1, 1, 1, 0, int8_t, int8_t, int32_t)
YNN_DOT_KERNEL(arch_flag::none, dot_uint8_int8_int32, 1, 128, 1, 1, 1, 0, uint8_t, int8_t, int32_t)
YNN_DOT_KERNEL(arch_flag::none, dot_int8_int4_int32, 1, 128, 1, 2, 1, 0, uint8_t, int4x2, int32_t)
