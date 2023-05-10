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
#include <limits>
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
  TEST(CVT__SSE2, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x387FE000); n < UINT32_C(0x477FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB87FE000); n < UINT32_C(0xC77FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x33000001); n < UINT32_C(0x387FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x387FDFFF)));
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB3000001); n < UINT32_C(0xB87FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xB87FDFFF)));
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, positive_underflow) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000001); n < UINT32_C(0x33000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x0000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, negative_underflow) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000001); n < UINT32_C(0xB3000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x8000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x0000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__SSE2, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x8000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__SSE2, positive_overflow) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x477FF000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x7C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, negative_overflow) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC77FF000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xFC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x7C00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__SSE2, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0xFC00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__SSE2, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
        ASSERT_LT(outputs[i], UINT16_C(0x8000))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE2, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__sse2(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0xFC00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(CVT__SSE41, positive_normal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x387FE000); n < UINT32_C(0x477FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, negative_normal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB87FE000); n < UINT32_C(0xC77FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, positive_subnormal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x33000001); n < UINT32_C(0x387FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x387FDFFF)));
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, negative_subnormal) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB3000001); n < UINT32_C(0xB87FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xB87FDFFF)));
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, positive_underflow) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000001); n < UINT32_C(0x33000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x0000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, negative_underflow) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000001); n < UINT32_C(0xB3000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x8000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, positive_zero) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x0000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__SSE41, negative_zero) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x8000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__SSE41, positive_overflow) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x477FF000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x7C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, negative_overflow) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC77FF000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xFC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, positive_infinity) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x7C00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__SSE41, negative_infinity) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0xFC00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__SSE41, positive_nan) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
        ASSERT_LT(outputs[i], UINT16_C(0x8000))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__SSE41, negative_nan) {
    TEST_REQUIRES_X86_SSE41;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__sse41(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0xFC00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(CVT__F16C, positive_normal) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x387FE000); n < UINT32_C(0x477FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, negative_normal) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB87FE000); n < UINT32_C(0xC77FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, positive_subnormal) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x33000001); n < UINT32_C(0x387FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x387FDFFF)));
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, negative_subnormal) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB3000001); n < UINT32_C(0xB87FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xB87FDFFF)));
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, positive_underflow) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000001); n < UINT32_C(0x33000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x0000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, negative_underflow) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000001); n < UINT32_C(0xB3000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x8000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, positive_zero) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x0000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__F16C, negative_zero) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x8000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__F16C, positive_overflow) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x477FF000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x7C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, negative_overflow) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC77FF000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xFC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, positive_infinity) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x7C00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__F16C, negative_infinity) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0xFC00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__F16C, positive_nan) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
        ASSERT_LT(outputs[i], UINT16_C(0x8000))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__F16C, negative_nan) {
    TEST_REQUIRES_X86_F16C;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__f16c(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0xFC00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CVT__NEON, positive_normal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x387FE000); n < UINT32_C(0x477FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, negative_normal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB87FE000); n < UINT32_C(0xC77FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, positive_subnormal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x33000001); n < UINT32_C(0x387FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x387FDFFF)));
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, negative_subnormal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB3000001); n < UINT32_C(0xB87FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xB87FDFFF)));
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, positive_underflow) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000001); n < UINT32_C(0x33000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x0000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, negative_underflow) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000001); n < UINT32_C(0xB3000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x8000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, positive_zero) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x0000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__NEON, negative_zero) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x8000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__NEON, positive_overflow) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x477FF000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x7C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, negative_overflow) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC77FF000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xFC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, positive_infinity) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x7C00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__NEON, negative_infinity) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0xFC00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__NEON, positive_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
        ASSERT_LT(outputs[i], UINT16_C(0x8000))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEON, negative_nan) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__neon(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0xFC00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CVT__NEONFP16, positive_normal) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x387FE000); n < UINT32_C(0x477FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, negative_normal) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB87FE000); n < UINT32_C(0xC77FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, positive_subnormal) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x33000001); n < UINT32_C(0x387FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x387FDFFF)));
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, negative_subnormal) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB3000001); n < UINT32_C(0xB87FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xB87FDFFF)));
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, positive_underflow) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000001); n < UINT32_C(0x33000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x0000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, negative_underflow) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000001); n < UINT32_C(0xB3000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x8000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x0000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__NEONFP16, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x8000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__NEONFP16, positive_overflow) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x477FF000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x7C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, negative_overflow) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC77FF000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xFC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, positive_infinity) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x7C00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__NEONFP16, negative_infinity) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0xFC00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__NEONFP16, positive_nan) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
        ASSERT_LT(outputs[i], UINT16_C(0x8000))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__NEONFP16, negative_nan) {
    TEST_REQUIRES_ARM_NEON_FP16;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__neonfp16(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0xFC00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(CVT__WASMSIMD, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x387FE000); n < UINT32_C(0x477FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB87FE000); n < UINT32_C(0xC77FF000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, positive_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x33000001); n < UINT32_C(0x387FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x387FDFFF)));
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, negative_subnormal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xB3000001); n < UINT32_C(0xB87FE000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xB87FDFFF)));
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, positive_underflow) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x00000001); n < UINT32_C(0x33000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x0000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, negative_underflow) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x80000001); n < UINT32_C(0xB3000001); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x8000);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, positive_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +0.0f);
    xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x0000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__WASMSIMD, negative_zero) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -0.0f);
    xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x8000);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__WASMSIMD, positive_overflow) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x477FF000); n < UINT32_C(0x7F800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0x7C00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, negative_overflow) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0xC77FF000); n < UINT32_C(0xFF800000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(n + i);
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint16_t reference_output = UINT16_C(0xFC00);
        ASSERT_EQ(reference_output, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, positive_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0x7C00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__WASMSIMD, negative_infinity) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
    xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    const uint16_t reference_output = UINT16_C(0xFC00);
    ASSERT_EQ(reference_output, outputs[0])
      << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
      << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
      << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
  }

  TEST(CVT__WASMSIMD, positive_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0x7C00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
        ASSERT_LT(outputs[i], UINT16_C(0x8000))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }

  TEST(CVT__WASMSIMD, negative_nan) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
    for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
      for (uint32_t i = 0; i < kBlockSize; i++) {
        inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
      }
      xnn_math_f32_f16_cvt__wasmsimd(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_GT(outputs[i], UINT16_C(0xFC00))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

TEST(CVT__SCALAR_BITCAST, positive_normal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x387FE000); n < UINT32_C(0x477FF000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, negative_normal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xB87FE000); n < UINT32_C(0xC77FF000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, positive_subnormal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x33000001); n < UINT32_C(0x387FE000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x387FDFFF)));
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, negative_subnormal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xB3000001); n < UINT32_C(0xB87FE000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xB87FDFFF)));
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, positive_underflow) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x00000001); n < UINT32_C(0x33000001); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = UINT16_C(0x0000);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, negative_underflow) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x80000001); n < UINT32_C(0xB3000001); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = UINT16_C(0x8000);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, positive_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +0.0f);
  xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
  const uint16_t reference_output = UINT16_C(0x0000);
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
}

TEST(CVT__SCALAR_BITCAST, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
  const uint16_t reference_output = UINT16_C(0x8000);
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
}

TEST(CVT__SCALAR_BITCAST, positive_overflow) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x477FF000); n < UINT32_C(0x7F800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = UINT16_C(0x7C00);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, negative_overflow) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC77FF000); n < UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = UINT16_C(0xFC00);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, positive_infinity) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
  xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
  const uint16_t reference_output = UINT16_C(0x7C00);
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
}

TEST(CVT__SCALAR_BITCAST, negative_infinity) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
  xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
  const uint16_t reference_output = UINT16_C(0xFC00);
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
}

TEST(CVT__SCALAR_BITCAST, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_GT(outputs[i], UINT16_C(0x7C00))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      ASSERT_LT(outputs[i], UINT16_C(0x8000))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_BITCAST, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
    }
    xnn_math_f32_f16_cvt__scalar_bitcast(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_GT(outputs[i], UINT16_C(0xFC00))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, positive_normal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x387FE000); n < UINT32_C(0x477FF000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, negative_normal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xB87FE000); n < UINT32_C(0xC77FF000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, positive_subnormal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x33000001); n < UINT32_C(0x387FE000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x387FDFFF)));
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, negative_subnormal) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xB3000001); n < UINT32_C(0xB87FE000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0xB87FDFFF)));
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(inputs[i]);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, positive_underflow) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x00000001); n < UINT32_C(0x33000001); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = UINT16_C(0x0000);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, negative_underflow) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x80000001); n < UINT32_C(0xB3000001); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = UINT16_C(0x8000);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, positive_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +0.0f);
  xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
  const uint16_t reference_output = UINT16_C(0x0000);
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
}

TEST(CVT__SCALAR_FABSF, negative_zero) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -0.0f);
  xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
  const uint16_t reference_output = UINT16_C(0x8000);
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
}

TEST(CVT__SCALAR_FABSF, positive_overflow) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x477FF000); n < UINT32_C(0x7F800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = UINT16_C(0x7C00);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, negative_overflow) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0xC77FF000); n < UINT32_C(0xFF800000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(n + i);
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = UINT16_C(0xFC00);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, positive_infinity) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), +std::numeric_limits<float>::infinity());
  xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
  const uint16_t reference_output = UINT16_C(0x7C00);
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
}

TEST(CVT__SCALAR_FABSF, negative_infinity) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  std::fill(inputs.begin(), inputs.end(), -std::numeric_limits<float>::infinity());
  xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
  const uint16_t reference_output = UINT16_C(0xFC00);
  ASSERT_EQ(reference_output, outputs[0])
    << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[0])
    << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
    << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[0];
}

TEST(CVT__SCALAR_FABSF, positive_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_GT(outputs[i], UINT16_C(0x7C00))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
      ASSERT_LT(outputs[i], UINT16_C(0x8000))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

TEST(CVT__SCALAR_FABSF, negative_nan) {
  std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(0x7F800001); n < UINT32_C(0x80000000); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, UINT32_C(0x7FFFFFFF)));
    }
    xnn_math_f32_f16_cvt__scalar_fabsf(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      ASSERT_GT(outputs[i], UINT16_C(0xFC00))
        << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}
