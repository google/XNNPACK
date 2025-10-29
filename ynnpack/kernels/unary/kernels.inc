// clang-format off

#ifdef YNN_ARCH_X86_AVX512F
#include "ynnpack/kernels/unary/x86_avx512f.inc"
#endif  // YNN_ARCH_X86_AVX512F
#ifdef YNN_ARCH_X86_FMA3
#include "ynnpack/kernels/unary/x86_fma3.inc"
#endif  // YNN_ARCH_X86_FMA3
#ifdef YNN_ARCH_X86_F16C
#include "ynnpack/kernels/unary/x86_f16c.inc"
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_AVX2
#include "ynnpack/kernels/unary/x86_avx2.inc"
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
#include "ynnpack/kernels/unary/x86_avx.inc"
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE41
#include "ynnpack/kernels/unary/x86_sse41.inc"
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
#include "ynnpack/kernels/unary/x86_sse2.inc"
#endif  // YNN_ARCH_X86_SSE2

#ifdef YNN_ARCH_ARM_NEON
#include "ynnpack/kernels/unary/arm_neon.inc"
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_ARM_NEONFP16
#include "ynnpack/kernels/unary/arm_neonfp16.inc"
#endif  // YNN_ARCH_ARM_NEONFP16
