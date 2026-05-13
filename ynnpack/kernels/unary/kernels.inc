// clang-format off

#ifdef YNN_ARCH_X86_AVX512BF16
#include "ynnpack/kernels/unary/x86_avx512bf16.inc"
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512
#include "ynnpack/kernels/unary/x86_avx512.inc"
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2_FMA3
#include "ynnpack/kernels/unary/x86_avx2_fma3.inc"
#endif  // YNN_ARCH_X86_AVX2_FMA3
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

#ifdef YNN_ARCH_ARM_NEONBF16
#include "ynnpack/kernels/unary/arm_neonbf16.inc"
#endif  // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM_NEONFP16
#include "ynnpack/kernels/unary/arm_neonfp16.inc"
#endif  // YNN_ARCH_ARM_NEONFP16
#ifdef YNN_ARCH_ARM64_NEON
#include "ynnpack/kernels/unary/arm64_neon.inc"
#endif  // YNN_ARCH_ARM64_NEON
#ifdef YNN_ARCH_ARM_NEON
#include "ynnpack/kernels/unary/arm_neon.inc"
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_WASM_SIMD128
#include "ynnpack/kernels/unary/wasm_simd128.inc"
#endif  // YNN_ARCH_WASM_SIMD128
