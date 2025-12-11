// clang-format off

#ifdef YNN_ARCH_X86_AMXBF16
YNN_DOT_KERNEL(arch_flag::amxbf16, dot_bf16_bf16_fp32_16x64x32_16x16x2_amxbf16,
               16, 64, 32, 1, 16, 2,
               /*flags=*/0, bfloat16, bfloat16, float)
#endif  // YNN_ARCH_X86_AMXBF16
#ifdef YNN_ARCH_X86_AMXFP16
YNN_DOT_KERNEL(arch_flag::amxfp16, dot_fp16_fp16_fp32_16x64x32_16x16x2_amxfp16,
               16, 64, 32, 1, 16, 2,
               /*flags=*/0, half, half, float)
#endif  // YNN_ARCH_X86_AMXFP16
#ifdef YNN_ARCH_X86_AMXINT8
YNN_DOT_KERNEL(arch_flag::amxint8, dot_int8_int8_int32_16x64x64_16x16x4_amxint8,
               16, 64, 64, 1, 16, 4,
               dot_flag::consistent_arithmetic, int8_t, int8_t, int32_t)
YNN_DOT_KERNEL(arch_flag::amxint8, dot_uint8_int8_int32_16x64x64_16x16x4_amxint8,
               16, 64, 64, 1, 16, 4,
               dot_flag::consistent_arithmetic, uint8_t, int8_t, int32_t)
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
#include "ynnpack/kernels/dot/x86_avx512f_fp16_fp16_fp32.inc"
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
#ifdef YNN_ARCH_X86_F16C
#include "ynnpack/kernels/dot/x86_f16c_fp16_fp16_fp32.inc"
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_F16C_FMA3
#include "ynnpack/kernels/dot/x86_f16c_fma3_fp16_fp16_fp32.inc"
#endif  // YNN_ARCH_X86_F16C_FMA3
#ifdef YNN_ARCH_X86_AVX2
#include "ynnpack/kernels/dot/x86_avx2_fp32_k2.inc"
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_SSE2
#include "ynnpack/kernels/dot/x86_sse2_fp32.inc"
#endif  // YNN_ARCH_X86_SSE2

#ifndef YNN_DISABLE_SME
#ifdef YNN_ARCH_ARM64_SME2
YNN_DOT_KERNEL(arch_flag::sme2, dot_fp32_sme2,
               sme_vl(float{}), 4 * sme_vl(float{}), 1, 1, sme_vl(float{}), 1,
               dot_flag::transpose_a | dot_flag::consistent_arithmetic,
               float, float, float)
YNN_DOT_KERNEL(arch_flag::sme2, dot_bf16_bf16_fp32_sme2,
               sme_vl(float{}), 4 * sme_vl(float{}), 2, 1, sme_vl(float{}), 2,
               dot_flag::transpose_a,
               bfloat16, bfloat16, float)
YNN_DOT_KERNEL(arch_flag::sme2, dot_fp16_fp16_fp32_sme2,
               sme_vl(float{}), 4 * sme_vl(float{}), 2, 1, sme_vl(float{}), 2,
               dot_flag::transpose_a,
               half, half, float)
YNN_DOT_KERNEL(arch_flag::sme2, dot_int8_int8_int32_sme2,
               sme_vl(int32_t{}), 4 * sme_vl(int32_t{}), 4, 1, sme_vl(int32_t{}), 4,
               dot_flag::transpose_a | dot_flag::consistent_arithmetic,
               int8_t, int8_t, int32_t)
#endif  // YNN_ARCH_ARM64_SME2

#ifdef YNN_ARCH_ARM64_SME
YNN_DOT_KERNEL(arch_flag::sme, dot_fp32_sme,
               sme_vl(float{}), 4 * sme_vl(float{}), 1, 1, sme_vl(float{}), 1,
               dot_flag::transpose_a | dot_flag::consistent_arithmetic,
               float, float, float)
YNN_DOT_KERNEL(arch_flag::sme, dot_bf16_bf16_fp32_sme,
               sme_vl(float{}), 4 * sme_vl(float{}), 2, 1, sme_vl(float{}), 2,
               dot_flag::transpose_a,
               bfloat16, bfloat16, float)
YNN_DOT_KERNEL(arch_flag::sme, dot_fp16_fp16_fp32_sme,
               sme_vl(float{}), 4 * sme_vl(float{}), 2, 1, sme_vl(float{}), 2,
               dot_flag::transpose_a,
               half, half, float)
YNN_DOT_KERNEL(arch_flag::sme, dot_int8_int8_int32_sme,
               sme_vl(int32_t{}), 4 * sme_vl(int32_t{}), 4, 1, sme_vl(int32_t{}), 4,
               dot_flag::transpose_a | dot_flag::consistent_arithmetic,
               int8_t, int8_t, int32_t)
#endif  // YNN_ARCH_ARM64_SME
#endif  // YNN_DISABLE_SME

#ifdef YNN_ARCH_ARM64_NEONI8MM
#include "ynnpack/kernels/dot/arm64_neoni8mm_int8_int8_int32.inc"
#endif  // YNN_ARCH_ARM64_NEONI8MM
#ifdef YNN_ARCH_ARM_NEONBF16
#include "ynnpack/kernels/dot/arm_neonbf16_bf16_bf16_fp32_k2.inc"
#endif  // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM64_NEON
#include "ynnpack/kernels/dot/arm64_neon_fp32.inc"
#include "ynnpack/kernels/dot/arm64_neon_bf16_bf16_fp32.inc"
#endif  // YNN_ARCH_ARM64

#ifdef YNN_ARCH_ARM_NEONDOT
#include "ynnpack/kernels/dot/arm_neondot_int8_int8_int32.inc"
#endif  // YNN_ARCH_ARM_NEONDOT
#ifdef YNN_ARCH_ARM_NEON
#include "ynnpack/kernels/dot/arm_neon_int8_int8_int32.inc"
#endif  // YNN_ARCH_ARM_NEON

#if defined(YNN_ARCH_X86) || defined(YNN_ARCH_ARM64)
// On x86 and on ARM64, we consider kernels that use FMA to be consistent (and
// not ones that don't).
#define YNN_DOT_KERNEL_CONSISTENT_ARITHMETIC 0
#else
#define YNN_DOT_KERNEL_CONSISTENT_ARITHMETIC dot_flag::consistent_arithmetic
#endif


YNN_DOT_KERNEL(arch_flag::none, dot_fp32_1x128x1_1x1x1,
               1, 128, 1, 1, 1, 1,
               YNN_DOT_KERNEL_CONSISTENT_ARITHMETIC,
               float, float, float)
YNN_DOT_KERNEL(arch_flag::none, dot_fp16_fp16_fp32_1x128x1_1x1x1,
               1, 128, 1, 1, 1, 1,
               dot_flag::consistent_arithmetic, half, half, float)
YNN_DOT_KERNEL(arch_flag::none, dot_bf16_bf16_fp32_1x128x1_1x1x1,
               1, 128, 1, 1, 1, 1,
               YNN_DOT_KERNEL_CONSISTENT_ARITHMETIC,
               bfloat16, bfloat16, float)
YNN_DOT_KERNEL(arch_flag::none, dot_int8_int8_int32_1x128x1_1x1x1,
               1, 128, 1, 1, 1, 1,
               dot_flag::consistent_arithmetic, int8_t, int8_t, int32_t)
YNN_DOT_KERNEL(arch_flag::none, dot_uint8_int8_int32_1x128x1_1x1x1,
               1, 128, 1, 1, 1, 1,
               dot_flag::consistent_arithmetic, uint8_t, int8_t, int32_t)
YNN_DOT_KERNEL(arch_flag::none, dot_int8_int4_int32_1x128x2_1x1x2,
               1, 128, 2, 1, 1, 2,
               dot_flag::consistent_arithmetic, uint8_t, int4x2, int32_t)
