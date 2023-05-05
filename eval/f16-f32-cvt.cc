// Copyright 2021 Google LLC
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

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(CVT__SSE2_INT16, positive_normal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT16, negative_normal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT16, positive_zero) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE2_INT16, negative_zero) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE2_INT16, positive_subnormal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT16, negative_subnormal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT16, positive_infinity) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE2_INT16, negative_infinity) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE2_INT16, positive_nan) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT16, negative_nan) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__sse2_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(CVT__SSE2_INT32, positive_normal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT32, negative_normal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT32, positive_zero) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE2_INT32, negative_zero) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE2_INT32, positive_subnormal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT32, negative_subnormal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT32, positive_infinity) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE2_INT32, negative_infinity) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE2_INT32, positive_nan) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE2_INT32, negative_nan) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__sse2_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(CVT__SSE41_INT16, positive_normal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT16, negative_normal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT16, positive_zero) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE41_INT16, negative_zero) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE41_INT16, positive_subnormal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT16, negative_subnormal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT16, positive_infinity) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE41_INT16, negative_infinity) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE41_INT16, positive_nan) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT16, negative_nan) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__sse41_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(CVT__SSE41_INT32, positive_normal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT32, negative_normal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT32, positive_zero) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE41_INT32, negative_zero) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE41_INT32, positive_subnormal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT32, negative_subnormal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT32, positive_infinity) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE41_INT32, negative_infinity) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__SSE41_INT32, positive_nan) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__SSE41_INT32, negative_nan) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__sse41_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(CVT__F16C, positive_normal) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__F16C, negative_normal) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__F16C, positive_zero) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__F16C, negative_zero) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__F16C, positive_subnormal) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__F16C, negative_subnormal) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__F16C, positive_infinity) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__F16C, negative_infinity) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__F16C, positive_nan) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__F16C, negative_nan) {
    TEST_REQUIRES_X86_F16C;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__f16c(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CVT__NEON_INT16, positive_normal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT16, negative_normal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT16, positive_zero) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEON_INT16, negative_zero) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEON_INT16, positive_subnormal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT16, negative_subnormal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT16, positive_infinity) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEON_INT16, negative_infinity) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEON_INT16, positive_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT16, negative_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__neon_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CVT__NEON_INT32, positive_normal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT32, negative_normal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT32, positive_zero) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEON_INT32, negative_zero) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEON_INT32, positive_subnormal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT32, negative_subnormal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT32, positive_infinity) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEON_INT32, negative_infinity) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEON_INT32, positive_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEON_INT32, negative_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__neon_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CVT__NEONFP16, positive_normal) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEONFP16, negative_normal) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEONFP16, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEONFP16, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEONFP16, positive_subnormal) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEONFP16, negative_subnormal) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEONFP16, positive_infinity) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEONFP16, negative_infinity) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__NEONFP16, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__NEONFP16, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__neonfp16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(CVT__WASMSIMD_INT16, positive_normal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT16, negative_normal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT16, positive_zero) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__WASMSIMD_INT16, negative_zero) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__WASMSIMD_INT16, positive_subnormal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT16, negative_subnormal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT16, positive_infinity) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__WASMSIMD_INT16, negative_infinity) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__WASMSIMD_INT16, positive_nan) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT16, negative_nan) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__wasmsimd_int16(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(CVT__WASMSIMD_INT32, positive_normal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x0400); n < UINT16_C(0x7C00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT32, negative_normal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8400); n < UINT16_C(0xFC00); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = n + i;
      }
      xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT32, positive_zero) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x0000));
    xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__WASMSIMD_INT32, negative_zero) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x8000));
    xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__WASMSIMD_INT32, positive_subnormal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = 0; n < UINT16_C(0x0400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x0001));
      }
      xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT32, negative_subnormal) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x8000); n < UINT16_C(0x8400); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x8001));
      }
      xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT32, positive_infinity) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0x7C00));
    xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__WASMSIMD_INT32, negative_infinity) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), UINT16_C(0xFC00));
    xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
    const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[0]));
    ASSERT_EQ(reference_output, float_as_uint32(outputs[0]))
      << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[0]);
  }

  TEST(CVT__WASMSIMD_INT32, positive_nan) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(n + i, UINT16_C(0x7C01));
      }
      xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  TEST(CVT__WASMSIMD_INT32, negative_nan) {
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint16_t n = UINT16_C(0x7C00); n < UINT16_C(0x8000); n += kBlockSize) {
      for (uint16_t i = 0; i < kBlockSize; i++) {
        inputs[i] = std::max<uint16_t>(UINT16_C(0x8000) | (n + i), UINT16_C(0xFC01));
      }
      xnn_math_f16_f32_cvt__wasmsimd_int32(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(fp16_ieee_to_fp32_value(inputs[i]));
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
