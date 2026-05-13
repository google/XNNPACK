// clang-format off

#ifdef YNN_ARCH_X86_AVX512BF16
#include "ynnpack/kernels/binary/x86_avx512bf16.inc"
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512
#include "ynnpack/kernels/binary/x86_avx512.inc"
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_AVX2
#include "ynnpack/kernels/binary/x86_avx2.inc"
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
#include "ynnpack/kernels/binary/x86_avx.inc"
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_SSE2
#include "ynnpack/kernels/binary/x86_sse2.inc"
#endif  // YNN_ARCH_X86_SSE2

#ifdef YNN_ARCH_ARM_NEONBF16
#include "ynnpack/kernels/binary/arm_neonbf16.inc"
#endif  // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM64_NEON
#include "ynnpack/kernels/binary/arm64_neon.inc"
#endif  // YNN_ARCH_ARM64_NEON
#ifdef YNN_ARCH_ARM_NEON
#include "ynnpack/kernels/binary/arm_neon.inc"
#endif  // YNN_ARCH_ARM_NEON
#ifdef YNN_ARCH_WASM_SIMD128
#include "ynnpack/kernels/binary/wasm_simd128.inc"
#endif  // YNN_ARCH_WASM_SIMD128
