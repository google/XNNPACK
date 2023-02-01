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

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


constexpr int kBlockSize = 1024;

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CVT__NEON, positive_normal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t max_input = float_as_uint32((float) (std::numeric_limits<int8_t>::max() - zero_point));
      for (uint32_t n = 0; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__neon(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          long reference_output = std::lrintf(inputs[i]) + long(zero_point);
          if (inputs[i] >= float(std::numeric_limits<long>::max())) {
            reference_output = std::numeric_limits<int8_t>::max();
          } else if (inputs[i] <= float(std::numeric_limits<long>::min())) {
            reference_output = std::numeric_limits<int8_t>::min();
          }
          ASSERT_EQ(reference_output, long(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__NEON, negative_normal) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t max_input = float_as_uint32((float) (zero_point - std::numeric_limits<int8_t>::min()));
      for (uint32_t n = 0; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__neon(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          long reference_output = std::lrintf(inputs[i]) + long(zero_point);
          if (inputs[i] >= float(std::numeric_limits<long>::max())) {
            reference_output = std::numeric_limits<int8_t>::max();
          } else if (inputs[i] <= float(std::numeric_limits<long>::min())) {
            reference_output = std::numeric_limits<int8_t>::min();
          }
          ASSERT_EQ(reference_output, long(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__NEON, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t min_input = float_as_uint32((float) (std::numeric_limits<int8_t>::max() - zero_point));
      const uint32_t max_input = UINT32_C(0x7F800000);
      for (uint32_t n = min_input; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__neon(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          const int32_t reference_output = std::numeric_limits<int8_t>::max();
          ASSERT_EQ(reference_output, int32_t(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__NEON, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t min_input = float_as_uint32((float) (zero_point - std::numeric_limits<int8_t>::min()));
      const uint32_t max_input = UINT32_C(0x7F800000);
      for (uint32_t n = min_input; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__neon(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          const int32_t reference_output = std::numeric_limits<int8_t>::min();
          ASSERT_EQ(reference_output, int32_t(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CVT__NEONV8, positive_normal) {
    TEST_REQUIRES_ARM_NEON_V8;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t max_input = float_as_uint32((float) (std::numeric_limits<int8_t>::max() - zero_point));
      for (uint32_t n = 0; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__neonv8(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          long reference_output = std::lrintf(inputs[i]) + long(zero_point);
          if (inputs[i] >= float(std::numeric_limits<long>::max())) {
            reference_output = std::numeric_limits<int8_t>::max();
          } else if (inputs[i] <= float(std::numeric_limits<long>::min())) {
            reference_output = std::numeric_limits<int8_t>::min();
          }
          ASSERT_EQ(reference_output, long(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__NEONV8, negative_normal) {
    TEST_REQUIRES_ARM_NEON_V8;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t max_input = float_as_uint32((float) (zero_point - std::numeric_limits<int8_t>::min()));
      for (uint32_t n = 0; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__neonv8(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          long reference_output = std::lrintf(inputs[i]) + long(zero_point);
          if (inputs[i] >= float(std::numeric_limits<long>::max())) {
            reference_output = std::numeric_limits<int8_t>::max();
          } else if (inputs[i] <= float(std::numeric_limits<long>::min())) {
            reference_output = std::numeric_limits<int8_t>::min();
          }
          ASSERT_EQ(reference_output, long(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__NEONV8, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_V8;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t min_input = float_as_uint32((float) (std::numeric_limits<int8_t>::max() - zero_point));
      const uint32_t max_input = UINT32_C(0x7F800000);
      for (uint32_t n = min_input; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__neonv8(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          const int32_t reference_output = std::numeric_limits<int8_t>::max();
          ASSERT_EQ(reference_output, int32_t(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__NEONV8, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_V8;

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t min_input = float_as_uint32((float) (zero_point - std::numeric_limits<int8_t>::min()));
      const uint32_t max_input = UINT32_C(0x7F800000);
      for (uint32_t n = min_input; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__neonv8(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          const int32_t reference_output = std::numeric_limits<int8_t>::min();
          ASSERT_EQ(reference_output, int32_t(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(CVT__WASMSIMD, positive_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t max_input = float_as_uint32((float) (std::numeric_limits<int8_t>::max() - zero_point));
      for (uint32_t n = 0; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__wasmsimd(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          long reference_output = std::lrintf(inputs[i]) + long(zero_point);
          if (inputs[i] >= float(std::numeric_limits<long>::max())) {
            reference_output = std::numeric_limits<int8_t>::max();
          } else if (inputs[i] <= float(std::numeric_limits<long>::min())) {
            reference_output = std::numeric_limits<int8_t>::min();
          }
          ASSERT_EQ(reference_output, long(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__WASMSIMD, negative_normal) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t max_input = float_as_uint32((float) (zero_point - std::numeric_limits<int8_t>::min()));
      for (uint32_t n = 0; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__wasmsimd(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          long reference_output = std::lrintf(inputs[i]) + long(zero_point);
          if (inputs[i] >= float(std::numeric_limits<long>::max())) {
            reference_output = std::numeric_limits<int8_t>::max();
          } else if (inputs[i] <= float(std::numeric_limits<long>::min())) {
            reference_output = std::numeric_limits<int8_t>::min();
          }
          ASSERT_EQ(reference_output, long(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__WASMSIMD, positive_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t min_input = float_as_uint32((float) (std::numeric_limits<int8_t>::max() - zero_point));
      const uint32_t max_input = UINT32_C(0x7F800000);
      for (uint32_t n = min_input; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__wasmsimd(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          const int32_t reference_output = std::numeric_limits<int8_t>::max();
          ASSERT_EQ(reference_output, int32_t(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }

  TEST(CVT__WASMSIMD, negative_saturation) {
    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> outputs(kBlockSize);
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      const uint32_t min_input = float_as_uint32((float) (zero_point - std::numeric_limits<int8_t>::min()));
      const uint32_t max_input = UINT32_C(0x7F800000);
      for (uint32_t n = min_input; n < max_input; n += kBlockSize) {
        for (uint32_t i = 0; i < kBlockSize; i++) {
          inputs[i] = uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(n + i, max_input));
        }
        xnn_math_f32_qs8_cvt__wasmsimd(kBlockSize * sizeof(int8_t), inputs.data(), outputs.data(), int8_t(zero_point));
        for (uint32_t i = 0; i < kBlockSize; i++) {
          const int32_t reference_output = std::numeric_limits<int8_t>::min();
          ASSERT_EQ(reference_output, int32_t(outputs[i]))
            << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
            << ", reference = " << std::dec << reference_output
            << ", optimized = " << std::dec << int32_t(outputs[i])
            << ", zero point = " << std::dec << zero_point;
        }
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
