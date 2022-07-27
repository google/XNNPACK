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
  TEST(EXPM1MINUS__NEON_RR2_LUT16_P3, negative_zero) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__neon_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__NEON_RR2_LUT16_P3, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__neon_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__NEON_RR2_LUT16_P3, positive_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__neon_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__NEON_RR2_LUT16_P3, negative_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__neon_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPM1MINUS__NEON_RR2_P6, negative_zero) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__neon_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__NEON_RR2_P6, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__neon_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__NEON_RR2_P6, positive_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__neon_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__NEON_RR2_P6, negative_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__neon_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPM1MINUS__NEONFMA_RR1_LUT16_P3, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_LUT16_P3, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_LUT16_P3, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_LUT16_P3, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPM1MINUS__NEONFMA_RR1_P6, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__neonfma_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_P6, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__neonfma_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_P6, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__neonfma_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_P6, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__neonfma_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX512F_RR1_LUT16_P3_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX512F_RR1_LUT16_P3_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX512F_RR1_LUT16_P3_PERM, positive_nan) {
    TEST_REQUIRES_X86_AVX512F;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX512F_RR1_LUT16_P3_PERM, negative_nan) {
    TEST_REQUIRES_X86_AVX512F;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX512F_RR1_P6, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx512f_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX512F_RR1_P6, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx512f_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX512F_RR1_P6, positive_nan) {
    TEST_REQUIRES_X86_AVX512F;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx512f_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX512F_RR1_P6, negative_nan) {
    TEST_REQUIRES_X86_AVX512F;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx512f_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX2_RR1_LUT4_P4_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT4_P4_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT4_P4_PERM, positive_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT4_P4_PERM, negative_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX2_RR1_LUT8_P4_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT8_P4_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT8_P4_PERM, positive_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT8_P4_PERM, negative_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX2_RR1_LUT16_P3_GATHER, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT16_P3_GATHER, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT16_P3_GATHER, positive_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT16_P3_GATHER, negative_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX2_RR1_P6, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx2_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX2_RR1_P6, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx2_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX2_RR1_P6, positive_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx2_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX2_RR1_P6, negative_nan) {
    TEST_REQUIRES_X86_AVX2;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx2_rr1_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX_RR2_LUT4_P4_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT4_P4_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT4_P4_PERM, positive_nan) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT4_P4_PERM, negative_nan) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX_RR2_LUT16_P3, negative_zero) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT16_P3, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT16_P3, positive_nan) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT16_P3, negative_nan) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX_RR2_P6, negative_zero) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__avx_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__AVX_RR2_P6, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__avx_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX_RR2_P6, positive_nan) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__avx_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__AVX_RR2_P6, negative_nan) {
    TEST_REQUIRES_X86_AVX;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__avx_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__SSE2_RR2_LUT16_P3, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__sse2_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__SSE2_RR2_LUT16_P3, negative_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__sse2_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__SSE2_RR2_LUT16_P3, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__sse2_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__SSE2_RR2_LUT16_P3, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__sse2_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__SSE2_RR2_P6, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__sse2_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__SSE2_RR2_P6, negative_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__sse2_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__SSE2_RR2_P6, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__sse2_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__SSE2_RR2_P6, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__sse2_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_ANDNOT, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_ANDNOT, negative_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_ANDNOT, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_ANDNOT, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_MAX, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_MAX, negative_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_MAX, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_MAX, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_ANDNOT, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_ANDNOT, negative_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_ANDNOT, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_ANDNOT, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_MAX, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const float reference_output = 0.0f;
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_MAX, negative_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const float reference_output = -1.0f;
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_MAX, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_MAX, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
      }
      xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(EXPM1MINUS__SCALAR_RR2_LUT4_P4, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expm1minus__scalar_rr2_lut4_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 0.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT4_P4, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut4_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const float reference_output = -1.0f;
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT4_P4, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut4_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT4_P4, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut4_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}


TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P3, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expm1minus__scalar_rr2_lut8_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 0.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P3, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut8_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const float reference_output = -1.0f;
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P3, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut8_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P3, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut8_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}


TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P4, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expm1minus__scalar_rr2_lut8_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 0.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P4, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut8_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const float reference_output = -1.0f;
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P4, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut8_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P4, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut8_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}


TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P3, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expm1minus__scalar_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 0.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P3, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const float reference_output = -1.0f;
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P3, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P3, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut16_p3(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}


TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P4, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expm1minus__scalar_rr2_lut16_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 0.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P4, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut16_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const float reference_output = -1.0f;
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P4, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut16_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P4, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_lut16_p4(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}


TEST(EXPM1MINUS__SCALAR_RR2_P5, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expm1minus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 0.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPM1MINUS__SCALAR_RR2_P5, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const float reference_output = -1.0f;
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_P5, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expm1minus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_P5, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_p5(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}


TEST(EXPM1MINUS__SCALAR_RR2_P6, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_expm1minus__scalar_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const float reference_output = 0.0f;
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
}

TEST(EXPM1MINUS__SCALAR_RR2_P6, negative_saturation) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC18AA123); n <= UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xFF800000)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const float reference_output = -1.0f;
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(reference_output)
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_P6, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), n + i));
    }
    xnn_math_f32_expm1minus__scalar_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}

TEST(EXPM1MINUS__SCALAR_RR2_P6, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(UINT32_C(0x7FFFFFFF), UINT32_C(0x80000000) | (n + i)));
    }
    xnn_math_f32_expm1minus__scalar_rr2_p6(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
    }
  }
}
