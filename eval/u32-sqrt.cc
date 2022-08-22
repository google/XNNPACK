// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


constexpr int kBlockSize = 1024;


TEST(SQRT__SCALAR_BITMANIP, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_bitmanip(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_BITMANIP, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_bitmanip(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_CLZ_BINSEARCH, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_clz_binsearch(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_CLZ_BINSEARCH, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_clz_binsearch(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_CLZ_NEWTON, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_clz_newton(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_CLZ_NEWTON, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_clz_newton(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_CVTI32_SQRT_LRINT, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_cvti32_sqrt_lrint(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_CVTI32_SQRT_LRINT, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_cvti32_sqrt_lrint(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_CVTI64_SQRT_LRINT, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_cvti64_sqrt_lrint(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_CVTI64_SQRT_LRINT, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_cvti64_sqrt_lrint(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_CVTU32_SQRT_LRINT, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_cvtu32_sqrt_lrint(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_LRINT, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_cvtu32_sqrt_lrint(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_CVTI64_SQRTF_LRINTF, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_cvti64_sqrtf_lrintf(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_CVTI64_SQRTF_LRINTF, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_cvti64_sqrtf_lrintf(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_CVTU32_SQRTF_LRINTF, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_cvtu32_sqrtf_lrintf(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRTF_LRINTF, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_cvtu32_sqrtf_lrintf(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_HASHEMIAN, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_hashemian(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_HASHEMIAN, 65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_hashemian(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}


TEST(SQRT__SCALAR_TFLM, uint16_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = 0; n <= UINT32_C(4294901760); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::min<uint32_t>(n + i, UINT32_C(4294901760));
    }
    xnn_math_u32_sqrt__scalar_tflm(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      const int64_t squared_output = int64_t(uint64_t(output) * uint64_t(output));

      const uint32_t prev_output = output - 1;
      const int64_t squared_prev_output = int64_t(uint64_t(prev_output) * uint64_t(prev_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_prev_output - int64_t(input)))
        << "input = " << input << ", output = " << output;

      const uint32_t next_output = output + 1;
      const int64_t squared_next_output = int64_t(uint64_t(next_output) * uint64_t(next_output));
      ASSERT_LT(std::abs(squared_output - int64_t(input)), std::abs(squared_next_output - int64_t(input)))
        << "input = " << input << ", output = " << output;
    }
  }
}

TEST(SQRT__SCALAR_TFLM, DISABLED_65536_output) {
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> inputs(kBlockSize);
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> outputs(kBlockSize);
  for (uint32_t n = UINT32_C(4294901761); n >= UINT32_C(4294901761); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      inputs[i] = std::max<uint32_t>(n + i, UINT32_C(4294901761));
    }
    xnn_math_u32_sqrt__scalar_tflm(kBlockSize * sizeof(uint32_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint32_t input = inputs[i];
      const uint32_t output = outputs[i];
      ASSERT_EQ(output, UINT32_C(0x00010000))
        << "input = " << input << ", output = " << output;
    }
  }
}
