#ifndef XNNPACK_YNNPACK_KERNELS_DOT_COST_MODEL_COST_MODEL_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_COST_MODEL_COST_MODEL_H_

#include <cstdint>

namespace ynn {

// Estimates the cost of a single block block_m x block_n x block_k dot kernel.
struct dot_cost_model {
  float block_overhead = 0.0f;
  float load_a_cost = 5.0f;
  float load_b_cost = 11.0f;
  float output_cost = 9.0f;

  float estimate_block_cost(uint32_t k, uint32_t block_m, uint32_t block_n,
                            uint32_t block_k) const {
    const uint32_t k_aligned = (k + block_k - 1) & ~(block_k - 1);
    return block_overhead +
           (load_a_cost * block_m + load_b_cost * block_n) * k_aligned +
           output_cost * (block_m * block_n);
  }
};

// These cost models likely should be fitted for each CPU microarchitecture they
// could run on.
struct dot_cost_models {
#ifdef YNN_ARCH_X86_AMXBF16
  dot_cost_model x86_amx_bf16;
#endif  // YNN_ARCH_X86_AMXBF16
#ifdef YNN_ARCH_X86_AMXFP16
  dot_cost_model x86_amx_fp16;
#endif  // YNN_ARCH_X86_AMXFP16
#ifdef YNN_ARCH_X86_AMXINT8
  dot_cost_model x86_amx_int8;
  dot_cost_model x86_amx_uint8;
#endif  // YNN_ARCH_X86_AMXINT8
#ifdef YNN_ARCH_X86_AVX512VNNI
  dot_cost_model x86_avx512vnni_uint8_int2_int32;
  dot_cost_model x86_avx512vnni_uint8_int4_int32;
  dot_cost_model x86_avx512vnni_uint8_int8_int32;
  dot_cost_model x86_avx512vnni_uint8_int8_int32_k16;
#endif  // YNN_ARCH_X86_AVX512VNNI
#ifdef YNN_ARCH_X86_AVXVNNI
  dot_cost_model x86_avxvnni_uint8_int2_int32;
  dot_cost_model x86_avxvnni_uint8_int4_int32;
  dot_cost_model x86_avxvnni_uint8_int8_int32;
#endif  // YNN_ARCH_X86_AVXVNNI
#ifdef YNN_ARCH_X86_AVX512BF16
  dot_cost_model x86_avx512bf16_bf16_bf16_fp32;
#endif  // YNN_ARCH_X86_AVX512BF16
#ifdef YNN_ARCH_X86_AVX512
  dot_cost_model x86_avx512_bf16_bf16_fp32;
  dot_cost_model x86_avx512_bf16_bf16_fp32_k1;
  dot_cost_model x86_avx512_fp16_fp16_fp32;
  dot_cost_model x86_avx512_fp32;
  dot_cost_model x86_avx512_fp32_k2;
  dot_cost_model x86_avx512_fp32_k4;
  dot_cost_model x86_avx512_fp64;
  dot_cost_model x86_avx512_int8_int8_int32_symmetric_b;
  dot_cost_model x86_avx512_int8_int8_int32;
  dot_cost_model x86_avx512_int8_int8_int32_k1;
  dot_cost_model x86_avx512_int8_int8_int32_k16;
  dot_cost_model x86_avx512_uint8_int2_int32;
  dot_cost_model x86_avx512_uint8_int4_int32;
#endif  // YNN_ARCH_X86_AVX512
#ifdef YNN_ARCH_X86_FMA3
  dot_cost_model x86_fma3_fp32;
  dot_cost_model x86_fma3_fp64;
#endif  // YNN_ARCH_X86_FMA3
#ifdef YNN_ARCH_X86_AVX2_FMA3
  dot_cost_model x86_avx2_fma3_bf16_bf16_fp32;
  dot_cost_model x86_avx2_fma3_bf16_bf16_fp32_k1;
  dot_cost_model x86_avx2_fma3_fp32_k8;
  dot_cost_model x86_avx2_fma3_fp32_k2;
#endif  // YNN_ARCH_X86_AVX2_FMA3
#ifdef YNN_ARCH_X86_AVX2
  dot_cost_model x86_avx2_int8_int8_int32_symmetric_b;
  dot_cost_model x86_avx2_int8_int8_int32;
  dot_cost_model x86_avx2_int8_int8_int32_k1;
  dot_cost_model x86_avx2_uint8_int2_int32;
  dot_cost_model x86_avx2_uint8_int4_int32;
  dot_cost_model x86_avx2_fp32_k2;
#endif  // YNN_ARCH_X86_AVX2
#ifdef YNN_ARCH_X86_AVX
  dot_cost_model x86_avx_fp32;
  dot_cost_model x86_avx_fp64;
#endif  // YNN_ARCH_X86_AVX
#ifdef YNN_ARCH_X86_F16C_FMA3
  dot_cost_model x86_f16c_fma3_fp16_fp16_fp32;
#endif  // YNN_ARCH_X86_F16C_FMA3
#ifdef YNN_ARCH_X86_F16C
  dot_cost_model x86_f16c_fp16_fp16_fp32;
#endif  // YNN_ARCH_X86_F16C
#ifdef YNN_ARCH_X86_SSE2
  dot_cost_model x86_sse2_fp32;
#endif  // YNN_ARCH_X86_SSE2

#ifndef YNN_DISABLE_SME
#ifdef YNN_ARCH_ARM64_SME2
  dot_cost_model arm64_sme2_fp32;
  dot_cost_model arm64_sme2_bf16;
  dot_cost_model arm64_sme2_fp16;
  dot_cost_model arm64_sme2_int8;
#endif  // YNN_ARCH_ARM64_SME2

#ifdef YNN_ARCH_ARM64_SME
  dot_cost_model arm64_sme_fp32;
  dot_cost_model arm64_sme_bf16;
  dot_cost_model arm64_sme_fp16;
  dot_cost_model arm64_sme_int8;
  dot_cost_model arm64_sme_int4;
  dot_cost_model arm64_sme_int2;
#endif  // YNN_ARCH_ARM64_SME
#endif  // YNN_DISABLE_SME

#ifdef YNN_ARCH_ARM64_NEONFP8DOT4
  // These dot costs are assumed to match int8 neondot. When we have real
  // hardware, we should fit a real model.
  dot_cost_model arm64_neonfp8dot4_fp8_e5m2_fp8_e5m2_fp32;
  dot_cost_model arm64_neonfp8dot4_fp8_e4m3_fp8_e4m3_fp32;
#endif  // YNN_ARCH_ARM64_NEONFP8DOT4
#ifdef YNN_ARCH_ARM64_NEONI8MM
  dot_cost_model arm64_neoni8mm_int8_int2_int32;
  dot_cost_model arm64_neoni8mm_int8_int4_int32;
  dot_cost_model arm64_neoni8mm_int8_int8_int32;
#endif  // YNN_ARCH_ARM64_NEONI8MM
#ifdef YNN_ARCH_ARM_NEONBF16
  dot_cost_model arm_neonbf16_bf16_bf16_fp32_k4;
  dot_cost_model arm_neonbf16_bf16_bf16_fp32_k2;
#endif  // YNN_ARCH_ARM_NEONBF16
#ifdef YNN_ARCH_ARM64_NEON
  dot_cost_model arm64_neon_fp32;
  dot_cost_model arm64_neon_fp64;
  dot_cost_model arm64_neon_bf16_bf16_fp32;
#endif  // YNN_ARCH_ARM64_NEON

#ifdef YNN_ARCH_ARM_NEONDOT
  dot_cost_model arm_neondot_int8_int2_int32;
  dot_cost_model arm_neondot_int8_int4_int32;
  dot_cost_model arm_neondot_int8_int8_int32;
#endif  // YNN_ARCH_ARM_NEONDOT
#ifdef YNN_ARCH_ARM_NEON
  dot_cost_model arm_neon_int8_int8_int32;
#endif  // YNN_ARCH_ARM_NEON

#ifdef YNN_ARCH_WASM_SIMD128
  dot_cost_model wasm_simd128_fp32;
  dot_cost_model wasm_simd128_fp64;
  dot_cost_model wasm_simd128_int8_int8_int32;
#endif  // YNN_ARCH_WASM_SIMD128

#ifdef YNN_ARCH_HEXAGON_HVX
  dot_cost_model hexagon_hvx_fp32_32x1;
#endif  // YNN_ARCH_HEXAGON_HVX
};

static constexpr dot_cost_model dot_cost_avoid = {100.0f, 100.0f, 100.0f,
                                                  100.0f};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_COST_MODEL_COST_MODEL_H_
