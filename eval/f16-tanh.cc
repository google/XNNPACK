// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <vector>

#include <gtest/gtest.h>

#include <fp16.h>

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


constexpr int kBlockSize = 1024;


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1_RR1_P3_DIV, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__aarch64_neonfp16arith_expm1_rr1_p3_div(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1_RR1_P3_DIV, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__aarch64_neonfp16arith_expm1_rr1_p3_div(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1_RR1_P3_DIV, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__aarch64_neonfp16arith_expm1_rr1_p3_div(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1_RR1_P3_DIV, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__aarch64_neonfp16arith_expm1_rr1_p3_div(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1fma(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1fma(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1FMA, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1fma(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1FMA, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1fma(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1recps(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1recps(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1RECPS, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1recps(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1RECPS, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1recps(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_RECPE, DISABLED_positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_recpe(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_RECPE, DISABLED_negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_recpe(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_RECPE, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_recpe(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_RECPE, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_recpe(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1_RR1_P3_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__avx2_expm1_rr1_p3_div(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__avx2_expm1_rr1_p3_div(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_DIV, positive_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__avx2_expm1_rr1_p3_div(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_DIV, negative_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__avx2_expm1_rr1_p3_div(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1_RR1_P3_RCP, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__avx2_expm1_rr1_p3_rcp(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_RCP, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__avx2_expm1_rr1_p3_rcp(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_RCP, positive_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__avx2_expm1_rr1_p3_rcp(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_RCP, negative_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__avx2_expm1_rr1_p3_rcp(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_P17, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__fma3_p17(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__FMA3_P17, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__fma3_p17(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__FMA3_P17, positive_nan) {
    TEST_REQUIRES_X86_FMA3;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__fma3_p17(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__FMA3_P17, negative_nan) {
    TEST_REQUIRES_X86_FMA3;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__fma3_p17(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_P19, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__fma3_p19(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__FMA3_P19, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__fma3_p19(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__FMA3_P19, positive_nan) {
    TEST_REQUIRES_X86_FMA3;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__fma3_p19(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__FMA3_P19, negative_nan) {
    TEST_REQUIRES_X86_FMA3;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__fma3_p19(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__F16C_P19, positive_saturation) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x4482); n <= UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7C00));
      }
      xnn_math_f16_tanh__f16c_p19(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x3C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__F16C_P19, negative_saturation) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0xC482); n <= UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0xFC00));
      }
      xnn_math_f16_tanh__f16c_p19(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xBC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__F16C_P19, positive_nan) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__f16c_p19(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(TANH__F16C_P19, negative_nan) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = UINT16_C(0x8000) | std::min<uint16_t>(UINT16_C(0x7FFF), n + i);
      }
      xnn_math_f16_tanh__f16c_p19(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint16_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE((outputs[i] & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
