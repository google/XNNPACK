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
  TEST(ROUNDNE__SSE, positive_normal) {
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

  TEST(ROUNDNE__SSE, negative_normal) {
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

  TEST(ROUNDNE__SSE, positive_integral) {
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

  TEST(ROUNDNE__SSE, negative_integral) {
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

  TEST(ROUNDNE__SSE, positive_infinity) {
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

  TEST(ROUNDNE__SSE, negative_infinity) {
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

  TEST(ROUNDNE__SSE, positive_qnan) {
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

  TEST(ROUNDNE__SSE, negative_qnan) {
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

  TEST(ROUNDNE__SSE, positive_snan) {
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

  TEST(ROUNDNE__SSE, negative_snan) {
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

  TEST(ROUNDNE__SSE, positive_snan_to_qnan) {
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

  TEST(ROUNDNE__SSE, negative_snan_to_qnan) {
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
  TEST(ROUNDNE__SSE2, positive_normal) {
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

  TEST(ROUNDNE__SSE2, negative_normal) {
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

  TEST(ROUNDNE__SSE2, positive_integral) {
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

  TEST(ROUNDNE__SSE2, negative_integral) {
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

  TEST(ROUNDNE__SSE2, positive_infinity) {
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

  TEST(ROUNDNE__SSE2, negative_infinity) {
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

  TEST(ROUNDNE__SSE2, positive_qnan) {
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

  TEST(ROUNDNE__SSE2, negative_qnan) {
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

  TEST(ROUNDNE__SSE2, positive_snan) {
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

  TEST(ROUNDNE__SSE2, negative_snan) {
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

  TEST(ROUNDNE__SSE2, DISABLED_positive_snan_to_qnan) {
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

  TEST(ROUNDNE__SSE2, DISABLED_negative_snan_to_qnan) {
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
  TEST(ROUNDNE__SSE41, positive_normal) {
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

  TEST(ROUNDNE__SSE41, negative_normal) {
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

  TEST(ROUNDNE__SSE41, positive_integral) {
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

  TEST(ROUNDNE__SSE41, negative_integral) {
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

  TEST(ROUNDNE__SSE41, positive_infinity) {
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

  TEST(ROUNDNE__SSE41, negative_infinity) {
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

  TEST(ROUNDNE__SSE41, positive_qnan) {
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

  TEST(ROUNDNE__SSE41, negative_qnan) {
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

  TEST(ROUNDNE__SSE41, positive_snan) {
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

  TEST(ROUNDNE__SSE41, negative_snan) {
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

  TEST(ROUNDNE__SSE41, positive_snan_to_qnan) {
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

  TEST(ROUNDNE__SSE41, negative_snan_to_qnan) {
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
  TEST(ROUNDNE__NEON, positive_normal) {
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

  TEST(ROUNDNE__NEON, negative_normal) {
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

  TEST(ROUNDNE__NEON, positive_integral) {
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

  TEST(ROUNDNE__NEON, negative_integral) {
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

  TEST(ROUNDNE__NEON, positive_infinity) {
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

  TEST(ROUNDNE__NEON, negative_infinity) {
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

  TEST(ROUNDNE__NEON, positive_qnan) {
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

  TEST(ROUNDNE__NEON, negative_qnan) {
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

  TEST(ROUNDNE__NEON, positive_snan) {
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

  TEST(ROUNDNE__NEON, negative_snan) {
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

  TEST(ROUNDNE__NEON, positive_snan_to_qnan) {
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

  TEST(ROUNDNE__NEON, negative_snan_to_qnan) {
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
  TEST(ROUNDNE__NEONV8, positive_normal) {
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

  TEST(ROUNDNE__NEONV8, negative_normal) {
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

  TEST(ROUNDNE__NEONV8, positive_integral) {
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

  TEST(ROUNDNE__NEONV8, negative_integral) {
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

  TEST(ROUNDNE__NEONV8, positive_infinity) {
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

  TEST(ROUNDNE__NEONV8, negative_infinity) {
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

  TEST(ROUNDNE__NEONV8, positive_qnan) {
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

  TEST(ROUNDNE__NEONV8, negative_qnan) {
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

  TEST(ROUNDNE__NEONV8, positive_snan) {
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

  TEST(ROUNDNE__NEONV8, negative_snan) {
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

  TEST(ROUNDNE__NEONV8, positive_snan_to_qnan) {
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

  TEST(ROUNDNE__NEONV8, negative_snan_to_qnan) {
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
  TEST(ROUNDNE__PSIMD, positive_normal) {
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

  TEST(ROUNDNE__PSIMD, negative_normal) {
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

  TEST(ROUNDNE__PSIMD, positive_integral) {
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

  TEST(ROUNDNE__PSIMD, negative_integral) {
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

  TEST(ROUNDNE__PSIMD, positive_infinity) {
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

  TEST(ROUNDNE__PSIMD, negative_infinity) {
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

  TEST(ROUNDNE__PSIMD, positive_qnan) {
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

  TEST(ROUNDNE__PSIMD, negative_qnan) {
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

  TEST(ROUNDNE__PSIMD, positive_snan) {
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

  TEST(ROUNDNE__PSIMD, negative_snan) {
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

  TEST(ROUNDNE__PSIMD, positive_snan_to_qnan) {
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

  TEST(ROUNDNE__PSIMD, negative_snan_to_qnan) {
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

TEST(ROUNDNE__SCALAR, positive_normal) {
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

TEST(ROUNDNE__SCALAR, negative_normal) {
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

TEST(ROUNDNE__SCALAR, positive_integral) {
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

TEST(ROUNDNE__SCALAR, negative_integral) {
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

TEST(ROUNDNE__SCALAR, positive_infinity) {
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

TEST(ROUNDNE__SCALAR, negative_infinity) {
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

TEST(ROUNDNE__SCALAR, positive_qnan) {
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

TEST(ROUNDNE__SCALAR, negative_qnan) {
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

TEST(ROUNDNE__SCALAR, positive_snan) {
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

TEST(ROUNDNE__SCALAR, negative_snan) {
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

TEST(ROUNDNE__SCALAR, positive_snan_to_qnan) {
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

TEST(ROUNDNE__SCALAR, negative_snan_to_qnan) {
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
