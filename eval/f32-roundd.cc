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

#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


constexpr int kBlockSize = 1024;

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(ROUNDD__SSE_ADDSUB, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE_ADDSUB, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE_ADDSUB, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE_ADDSUB, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE_ADDSUB, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE_ADDSUB, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(ROUNDD__SSE2_CVT, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE2_CVT, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE2_CVT, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE2_CVT, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE2_CVT, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE2_CVT, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse2_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(ROUNDD__SSE41, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE41, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE41, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE41, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__SSE41, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__SSE41, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(ROUNDD__NEON_ADDSUB, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEON_ADDSUB, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEON_ADDSUB, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEON_ADDSUB, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEON_ADDSUB, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_ADDSUB, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neon_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(ROUNDD__NEON_CVT, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEON_CVT, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEON_CVT, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEON_CVT, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEON_CVT, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEON_CVT, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neon_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(ROUNDD__NEONV8, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEONV8, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEONV8, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEONV8, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__NEONV8, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__NEONV8, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD
  TEST(ROUNDD__WASMSIMD_ADDSUB, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_ADDSUB, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_WASMSIMD
  TEST(ROUNDD__WASMSIMD_CVT, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_CVT, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_CVT, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_CVT, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_CVT, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_CVT, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_WASMSIMD
  TEST(ROUNDD__WASMSIMD_NATIVE, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
    xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
    xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, positive_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, negative_integral) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, positive_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, negative_snan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, positive_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDD__WASMSIMD_NATIVE, negative_snan_to_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundd__wasmsimd_native(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD

TEST(ROUNDD__SCALAR_ADDSUB, positive_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
  xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDD__SCALAR_ADDSUB, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
  xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDD__SCALAR_ADDSUB, positive_subnormal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, negative_subnormal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, positive_normal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, negative_normal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, positive_integral) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, negative_integral) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, positive_infinity) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
  xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDD__SCALAR_ADDSUB, negative_infinity) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
  xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDD__SCALAR_ADDSUB, positive_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, negative_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, positive_snan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, negative_snan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, positive_snan_to_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_ADDSUB, negative_snan_to_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundd__scalar_addsub(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, positive_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), UINT32_C(0x00000000));
  xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDD__SCALAR_CVT, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), UINT32_C(0x80000000));
  xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDD__SCALAR_CVT, positive_subnormal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x00800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x00000001)));
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, negative_subnormal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0x80800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x80000001)));
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, positive_normal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x00800000); n < UINT32_C(0x4B800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, negative_normal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x80800000); n < UINT32_C(0xCB800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, positive_integral) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, negative_integral) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, positive_infinity) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
  xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDD__SCALAR_CVT, negative_infinity) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
  xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::floor(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDD__SCALAR_CVT, positive_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, negative_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, positive_snan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, negative_snan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, positive_snan_to_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDD__SCALAR_CVT, negative_snan_to_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundd__scalar_cvt(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::floor(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}
