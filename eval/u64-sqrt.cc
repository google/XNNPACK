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


namespace {

uint64_t Sqrt(uint64_t n) {
  if (n == 0) {
    return n;
  }

  uint64_t x0 = n >> 1;
  uint64_t x1 = (x0 + n / x0) >> 1;
  do {
    x0 = x1;
    x1 = (x0 + n / x0) >> 1;
  } while (x1 < x0);

  // x0 is sqrt(n) rounded down, round up if needed
  if (int64_t(x0 * x0 + x0 - n) < 0) {
    x0 += 1;
  }
  return x0;
}

}  // namespace


TEST(SQRT__SCALAR_CVTU32_SQRT_CVTSATU32F64, min_mantissa_exact_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x0010000000000000) << s;
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_CVTSATU32F64, min_mantissa_min_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x001FFFFFFFFFFFFF) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_CVTSATU32F64, min_mantissa_max_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x0020000000000001) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_CVTSATU32F64, max_mantissa_exact_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x001FFFFFFFFFFFFF) << s;
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_CVTSATU32F64, max_mantissa_min_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x003FFFFFFFFFFFFD) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_CVTSATU32F64, max_mantissa_max_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x003FFFFFFFFFFFFF) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_CVTSATU32F64, largest_inputs) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint64_t i = 0; i < kBlockSize; i++) {
    inputs[i] = -i;
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_CVTSATU32F64, double_rounding) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint64_t n = UINT64_C(33554432); n <= UINT64_C(4294967295); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint64_t t = std::min(n + uint64_t(i), UINT64_C(4294967295));
      inputs[i] = t * t + t;
    }
    xnn_math_u64_sqrt__scalar_cvtu32_sqrt_cvtsatu32f64(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint64_t input = inputs[i];
      const uint64_t output = outputs[i];
      const uint64_t reference_output = Sqrt(input);
      ASSERT_EQ(output, reference_output) << "input: " << input;
    }
  }
}


TEST(SQRT__SCALAR_CVTU32_SQRT_LLRINT, min_mantissa_exact_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x0010000000000000) << s;
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_LLRINT, min_mantissa_min_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x001FFFFFFFFFFFFF) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_LLRINT, min_mantissa_max_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x0020000000000001) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_LLRINT, max_mantissa_exact_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x001FFFFFFFFFFFFF) << s;
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_LLRINT, max_mantissa_min_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x003FFFFFFFFFFFFD) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_LLRINT, max_mantissa_max_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x003FFFFFFFFFFFFF) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_LLRINT, largest_inputs) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint64_t i = 0; i < kBlockSize; i++) {
    inputs[i] = -i;
  }
  xnn_math_u64_sqrt__scalar_cvtu32_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU32_SQRT_LLRINT, double_rounding) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint64_t n = UINT64_C(33554432); n <= UINT64_C(4294967295); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint64_t t = std::min(n + uint64_t(i), UINT64_C(4294967295));
      inputs[i] = t * t + t;
    }
    xnn_math_u64_sqrt__scalar_cvtu32_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint64_t input = inputs[i];
      const uint64_t output = outputs[i];
      const uint64_t reference_output = Sqrt(input);
      ASSERT_EQ(output, reference_output) << "input: " << input;
    }
  }
}


TEST(SQRT__SCALAR_CVTU64_SQRT_LLRINT, min_mantissa_exact_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x0010000000000000) << s;
  }
  xnn_math_u64_sqrt__scalar_cvtu64_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU64_SQRT_LLRINT, min_mantissa_min_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x001FFFFFFFFFFFFF) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu64_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU64_SQRT_LLRINT, min_mantissa_max_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x0020000000000001) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu64_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU64_SQRT_LLRINT, max_mantissa_exact_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x001FFFFFFFFFFFFF) << s;
  }
  xnn_math_u64_sqrt__scalar_cvtu64_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU64_SQRT_LLRINT, max_mantissa_min_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x003FFFFFFFFFFFFD) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu64_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU64_SQRT_LLRINT, max_mantissa_max_input) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint32_t s = std::min<uint32_t>(i + 1, 11);
    inputs[i] = UINT64_C(0x003FFFFFFFFFFFFF) << (s - 1);
  }
  xnn_math_u64_sqrt__scalar_cvtu64_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU64_SQRT_LLRINT, largest_inputs) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint64_t i = 0; i < kBlockSize; i++) {
    inputs[i] = -i;
  }
  xnn_math_u64_sqrt__scalar_cvtu64_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
  for (uint32_t i = 0; i < kBlockSize; i++) {
    const uint64_t input = inputs[i];
    const uint64_t output = outputs[i];
    const uint64_t reference_output = Sqrt(input);
    ASSERT_EQ(output, reference_output) << "input: " << input;
  }
}

TEST(SQRT__SCALAR_CVTU64_SQRT_LLRINT, double_rounding) {
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> inputs(kBlockSize);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> outputs(kBlockSize);
  for (uint64_t n = UINT64_C(33554432); n <= UINT64_C(4294967295); n += kBlockSize) {
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint64_t t = std::min(n + uint64_t(i), UINT64_C(4294967295));
      inputs[i] = t * t + t;
    }
    xnn_math_u64_sqrt__scalar_cvtu64_sqrt_llrint(kBlockSize * sizeof(uint64_t), inputs.data(), outputs.data());
    for (uint32_t i = 0; i < kBlockSize; i++) {
      const uint64_t input = inputs[i];
      const uint64_t output = outputs[i];
      const uint64_t reference_output = Sqrt(input);
      ASSERT_EQ(output, reference_output) << "input: " << input;
    }
  }
}
