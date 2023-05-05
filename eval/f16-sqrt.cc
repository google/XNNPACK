// Copyright 2022 Google LLC
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

#include <fp16/fp16.h>

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


constexpr int kBlockSize = 1024;


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x8000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[0]
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, DISABLED_negative_subnormal) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0001); n <= UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x03FF)) | UINT16_C(0x8000);
      }
      xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x8000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, negative_normal) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n <= UINT16_C(0x7BFF); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7BFF)) | UINT16_C(0x8000);
      }
      xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t abs_output = outputs[i] & UINT16_C(0x7FFF);
        ASSERT_GT(abs_output, UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, negative_infinity) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t abs_output = outputs[0] & UINT16_C(0x7FFF);
    ASSERT_GT(abs_output, UINT16_C(0x7C00))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[0]
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n <= UINT16_C(0x7FFF); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7FFF)) | UINT16_C(0x8000);
      }
      xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t abs_output = outputs[i] & UINT16_C(0x7FFF);
        ASSERT_GT(abs_output, UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x0000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[0]
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, DISABLED_positive_subnormal) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0001); n <= UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x03FF));
      }
      xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x0000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, positive_normal) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n <= UINT16_C(0x7BFF); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x3BFF));
      }
      xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = std::sqrt(fp16_ieee_to_fp32_value(inputs[i]));
        const float reference_output_ulp = 8192.0f /* difference between HP and SP ULP */ *
          (uint32_as_float(float_as_uint32(reference_output) + 1) - reference_output);
        const float ulp_error = std::abs(reference_output - fp16_ieee_to_fp32_value(outputs[i])) / reference_output_ulp;
        ASSERT_LT(ulp_error, 0.5f)
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0')
            << inputs[i] << " (" << fp16_ieee_to_fp32_value(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0')
            << float_as_uint32(reference_output) << " (" << reference_output << ")"
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0')
            << outputs[i] << " (" << fp16_ieee_to_fp32_value(outputs[i]) << ")";
      }
    }
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, positive_infinity) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x7C00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[0]
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(F16_SQRT__AARCH64_NEONFP16ARITH_SQRT, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n <= UINT16_C(0x7FFF); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7FFF));
      }
      xnn_math_f16_sqrt__aarch64_neonfp16arith_sqrt(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t abs_output = outputs[i] & UINT16_C(0x7FFF);
        ASSERT_GT(abs_output, UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x8000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[0]
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, DISABLED_negative_subnormal) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0001); n <= UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x03FF)) | UINT16_C(0x8000);
      }
      xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x8000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, negative_normal) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n <= UINT16_C(0x7BFF); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7BFF)) | UINT16_C(0x8000);
      }
      xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t abs_output = outputs[i] & UINT16_C(0x7FFF);
        ASSERT_GT(abs_output, UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, negative_infinity) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t abs_output = outputs[0] & UINT16_C(0x7FFF);
    ASSERT_GT(abs_output, UINT16_C(0x7C00))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[0]
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n <= UINT16_C(0x7FFF); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7FFF)) | UINT16_C(0x8000);
      }
      xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t abs_output = outputs[i] & UINT16_C(0x7FFF);
        ASSERT_GT(abs_output, UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x0000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[0]
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, DISABLED_positive_subnormal) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0001); n <= UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x03FF));
      }
      xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x0000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, positive_normal) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n <= UINT16_C(0x7BFF); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x3BFF));
      }
      xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = std::sqrt(fp16_ieee_to_fp32_value(inputs[i]));
        const float reference_output_ulp = 8192.0f /* difference between HP and SP ULP */ *
          (uint32_as_float(float_as_uint32(reference_output) + 1) - reference_output);
        const float ulp_error = std::abs(reference_output - fp16_ieee_to_fp32_value(outputs[i])) / reference_output_ulp;
        ASSERT_LT(ulp_error, 0.5f)
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0')
            << inputs[i] << " (" << fp16_ieee_to_fp32_value(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0')
            << float_as_uint32(reference_output) << " (" << reference_output << ")"
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0')
            << outputs[i] << " (" << fp16_ieee_to_fp32_value(outputs[i]) << ")";
      }
    }
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, positive_infinity) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x7C00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[0]
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(F16_SQRT__NEONFP16ARITH_NR1FMA1ADJ, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C01); n <= UINT16_C(0x7FFF); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::min<uint16_t>(n + i, UINT16_C(0x7FFF));
      }
      xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t abs_output = outputs[i] & UINT16_C(0x7FFF);
        ASSERT_GT(abs_output, UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
