// clang-format off

#ifdef YNN_ARCH_X86_AVX512F
#include "ynnpack/kernels/binary/x86_avx512f.inc"
#endif  // YNN_ARCH_X86_AVX512F
#ifdef YNN_ARCH_X86_AVX2
#include "ynnpack/kernels/binary/x86_avx2.inc"
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
#include "ynnpack/kernels/binary/x86_avx.inc"
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE2
#include "ynnpack/kernels/binary/x86_sse2.inc"
#endif  // YNN_ARCH_X86_SSE2

#ifdef YNN_ARCH_ARM_NEON
#include "ynnpack/kernels/binary/arm_neon.inc"
#endif  // YNN_ARCH_ARM_NEON

