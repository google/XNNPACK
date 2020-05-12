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
  TEST(ROUNDNE__SSE, small_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, small_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, large_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, large_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, infinite_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x7F800000));
    xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__SSE, infinite_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0xFF800000));
    xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__SSE, qnan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, qnan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, snan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, snan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, snan_positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE, snan_negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(ROUNDNE__SSE2, small_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, small_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, large_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, large_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, infinite_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x7F800000));
    xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__SSE2, infinite_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0xFF800000));
    xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__SSE2, qnan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, qnan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, snan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, snan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, DISABLED_snan_positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE2, DISABLED_snan_negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse2(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(ROUNDNE__SSE41, small_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, small_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, large_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, large_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, infinite_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x7F800000));
    xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__SSE41, infinite_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0xFF800000));
    xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__SSE41, qnan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, qnan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, snan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, snan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, snan_positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__SSE41, snan_negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__sse41(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(ROUNDNE__NEON, small_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, small_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, large_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, large_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, infinite_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x7F800000));
    xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__NEON, infinite_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0xFF800000));
    xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__NEON, qnan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, qnan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, snan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, snan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, snan_positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEON, snan_negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__neon(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(ROUNDNE__NEONV8, small_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, small_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, large_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, large_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, infinite_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x7F800000));
    xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__NEONV8, infinite_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0xFF800000));
    xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__NEONV8, qnan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, qnan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, snan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, snan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, snan_positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__NEONV8, snan_negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__neonv8(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(ROUNDNE__PSIMD, small_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x4B800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, small_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0xCB800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, large_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, large_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, infinite_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0x7F800000));
    xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__PSIMD, infinite_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT32_C(0xFF800000));
    xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
    ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
  }

  TEST(ROUNDNE__PSIMD, qnan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(n + i);
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, qnan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, snan_positive) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, snan_negative) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, DISABLED_snan_positive_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }

  TEST(ROUNDNE__PSIMD, DISABLED_snan_negative_qnan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
      }
      xnn_math_f32_roundne__psimd(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
        ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

TEST(ROUNDNE__SCALAR, small_positive) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x00000000); n < UINT32_C(0x4B800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, small_negative) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x80000000); n < UINT32_C(0xCB800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, large_positive) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x4B800000); n < UINT32_C(0x7F800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, large_negative) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xCB800000); n < UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, infinite_positive) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), UINT32_C(0x7F800000));
  xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDNE__SCALAR, infinite_negative) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), UINT32_C(0xFF800000));
  xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
  const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[0]));
  ASSERT_EQ(reference_output, fp32_to_bits(outputs[0]))
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[0]);
}

TEST(ROUNDNE__SCALAR, qnan_positive) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(n + i);
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, qnan_negative) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7FC00000); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | (n + i));
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, snan_positive) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, snan_negative) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output & UINT32_C(0xFFBFFFFF), fp32_to_bits(outputs[i]) & UINT32_C(0xFFBFFFFF))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, snan_positive_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}

TEST(ROUNDNE__SCALAR, snan_negative_qnan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800000); n < UINT32_C(0x7FC00000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = fp32_from_bits(UINT32_C(0x80000000) | std::max<uint32_t>(n + i, UINT32_C(0x7F800001)));
    }
    xnn_math_f32_roundne__scalar(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t reference_output = fp32_to_bits(std::nearbyint(inputs[i]));
      ASSERT_EQ(reference_output, fp32_to_bits(outputs[i]))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(outputs[i]);
    }
  }
}
