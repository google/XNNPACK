// clang-format off

#ifdef YNN_ARCH_X86_AVX512BW
#include "ynnpack/kernels/ternary/x86_avx512bw.inc"
#endif  // YNN_ARCH_X86_AVX512BW
#ifdef YNN_ARCH_X86_AVX512F
#include "ynnpack/kernels/ternary/x86_avx512f.inc"
#endif  // YNN_ARCH_X86_AVX512F
#ifdef YNN_ARCH_X86_AVX2
#include "ynnpack/kernels/ternary/x86_avx2.inc"
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
#include "ynnpack/kernels/ternary/x86_avx.inc"
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE41
#include "ynnpack/kernels/ternary/x86_sse41.inc"
#endif  // YNN_ARCH_X86_SSE41
#ifdef YNN_ARCH_X86_SSE2
#include "ynnpack/kernels/ternary/x86_sse2.inc"
#endif  // YNN_ARCH_X86

#ifdef YNN_ARCH_ARM_NEON
#include "ynnpack/kernels/ternary/arm_neon.inc"
#endif  // YNN_ARCH_ARM_NEON
