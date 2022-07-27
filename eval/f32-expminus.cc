// Copyright 2020 Google LLC
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


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expminus__neonfma_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_expminus__neonfma_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC2AEAC50); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expminus__neonfma_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = UINT32_C(0x00000000);
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expminus__neonfma_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expminus__neonfma_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expminus__neonfma_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_expminus__neonfma_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC2AEAC50); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expminus__neonfma_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = UINT32_C(0x00000000);
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expminus__neonfma_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expminus__neonfma_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPMINUS__NEONFMA_RR2_P5, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expminus__neonfma_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__NEONFMA_RR2_P5, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_expminus__neonfma_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__NEONFMA_RR2_P5, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC2AEAC50); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expminus__neonfma_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = UINT32_C(0x00000000);
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__NEONFMA_RR2_P5, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expminus__neonfma_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__NEONFMA_RR2_P5, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expminus__neonfma_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPMINUS__AVX2_RR2_P5, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expminus__avx2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__AVX2_RR2_P5, positive_zero) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_expminus__avx2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__AVX2_RR2_P5, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC2AEAC50); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expminus__avx2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = UINT32_C(0x00000000);
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__AVX2_RR2_P5, positive_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expminus__avx2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__AVX2_RR2_P5, negative_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expminus__avx2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPMINUS__SSE2_RR2_P5, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expminus__sse2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__SSE2_RR2_P5, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_expminus__sse2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 1.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPMINUS__SSE2_RR2_P5, negative_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC2AEAC50); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expminus__sse2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = UINT32_C(0x00000000);
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__SSE2_RR2_P5, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expminus__sse2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPMINUS__SSE2_RR2_P5, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expminus__sse2_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expminus__scalar_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 1.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, positive_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +0.0f);
  xnn_math_f32_expminus__scalar_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 1.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC2AEAC50); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expminus__scalar_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = UINT32_C(0x00000000);
      ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expminus__scalar_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expminus__scalar_rr2_lut64_p2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}


TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expminus__scalar_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 1.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, positive_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +0.0f);
  xnn_math_f32_expminus__scalar_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 1.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC2AEAC50); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expminus__scalar_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = UINT32_C(0x00000000);
      ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expminus__scalar_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expminus__scalar_rr2_lut2048_p1(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}


TEST(EXPMINUS__SCALAR_RR2_P5, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expminus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 1.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPMINUS__SCALAR_RR2_P5, positive_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +0.0f);
  xnn_math_f32_expminus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 1.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPMINUS__SCALAR_RR2_P5, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC2AEAC50); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expminus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = UINT32_C(0x00000000);
      ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPMINUS__SCALAR_RR2_P5, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expminus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPMINUS__SCALAR_RR2_P5, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expminus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}
