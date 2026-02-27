// clang-format off

#ifdef YNN_ARCH_X86_AVX512
#include "ynnpack/kernels/ternary/x86_avx512bw.inc"
#include "ynnpack/kernels/ternary/x86_avx512f.inc"
#endif  // YNN_ARCH_X86_AVX512
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

YNN_ELEMENTWISE_KERNEL(arch_flag::none, quantize_fp32_to_int8, quantize_int8, nullptr, float, float, int32_t, int8_t)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, quantize_fp32_to_uint8, quantize_uint8, nullptr, float, float, int32_t, uint8_t)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, dequantize_int8_to_fp32, dequantize, nullptr, int8_t, int32_t, float, float)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, dequantize_uint8_to_fp32, dequantize, nullptr, uint8_t, int32_t, float, float)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, dequantize_int32_to_fp32, dequantize, nullptr, int32_t, int32_t, float, float)
