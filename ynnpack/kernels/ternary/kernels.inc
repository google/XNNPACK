// clang-format off

#ifdef YNN_ARCH_X86_AVX512
#include "ynnpack/kernels/ternary/x86_avx512.inc"
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

#ifdef YNN_ARCH_ARM64_NEON
#include "ynnpack/kernels/ternary/arm64_neon.inc"
#endif  // YNN_ARCH_ARM64_NEON
#ifdef YNN_ARCH_ARM_NEON
#include "ynnpack/kernels/ternary/arm_neon.inc"
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_WASM_SIMD128
#include "ynnpack/kernels/ternary/wasm_simd128.inc"
#endif  // YNN_ARCH_WASM_SIMD128

YNN_ELEMENTWISE_KERNEL(arch_flag::none, quantize_fp32_to_int8, quantize_int8, float, float, int32_t, int8_t)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, quantize_fp32_to_uint8, quantize_uint8, float, float, int32_t, uint8_t)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, dequantize_int8_to_fp32, dequantize, int8_t, int32_t, float, float)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, dequantize_uint8_to_fp32, dequantize, uint8_t, int32_t, float, float)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, dequantize_int32_to_fp32, dequantize, int32_t, int32_t, float, float)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, dequantize_int4_to_fp32, dequantize, int4x2, int32_t, float, float)
YNN_ELEMENTWISE_KERNEL(arch_flag::none, dequantize_int2_to_fp32, dequantize, int2x4, int32_t, float, float)
