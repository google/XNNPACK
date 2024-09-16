// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/transpose.h"

using transpose_ukernel =
    std::function<void(const void* input, void* output, size_t input_row_stride,
                       size_t output_row_stride, size_t input_element_stride,
                       size_t output_element_stride, size_t element_size,
                       size_t block_width, size_t block_height)>;

void TestTranspose(transpose_ukernel ukernel, size_t input_stride,
                   size_t output_stride, size_t input_element_stride,
                   size_t output_element_stride, size_t element_size,
                   size_t width, size_t height) {
  std::vector<uint8_t> input(input_stride * height * input_element_stride +
                             XNN_EXTRA_BYTES);
  std::vector<uint8_t> output(output_stride * width * output_element_stride);
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), UINT8_C(0xA5));

  // Call optimized micro-kernel.
  ukernel(input.data(), output.data(), input_stride * input_element_stride,
          output_stride * output_element_stride, input_element_stride,
          output_element_stride, element_size, width, height);

  // Verify results.
  for (size_t c = 0; c < width; c++) {
    for (size_t r = 0; r < height; r++) {
      EXPECT_EQ(
          std::memcmp(&input[input_element_stride * (c + r * input_stride)],
                      &output[output_element_stride * (r + c * output_stride)],
                      element_size),
          0)
          << "at row " << r << " / " << height << ", at column " << c << " / "
          << width;
    }
  }
}

struct TestParams {
  const char* name;
  uint64_t arch_flags;
  transpose_ukernel ukernel;
  size_t element_size;
  size_t block_width;
  size_t block_height;
};

// We want to be able to treat transposev ukernels as transposec ukernels for
// testing purposes, this set of wrapper builders lets us do that.
transpose_ukernel make_ukernel_wrapper(xnn_transposec_ukernel_fn ukernel) {
  return [ukernel = std::move(ukernel)](
             const void* input, void* output, size_t input_row_stride,
             size_t output_row_stride, size_t input_element_stride,
             size_t output_element_stride, size_t element_size,
             size_t block_width, size_t block_height) {
    // Verify we aren't trying to treat a transposec ukernel as a transposev
    // ukernel in an unsupported way.
    assert(input_element_stride == element_size);
    assert(output_element_stride == element_size);
    ukernel(input, output, input_row_stride, output_row_stride, block_width,
            block_height);
  };
}

template <typename UKernelFn>
transpose_ukernel make_ukernel_wrapper(UKernelFn ukernel) {
  return make_ukernel_wrapper(
      reinterpret_cast<xnn_transposec_ukernel_fn>(ukernel));
}

transpose_ukernel make_ukernel_wrapper(xnn_transposev_ukernel_fn ukernel) {
  return ukernel;
}

// This set of test params has all transpose ukernels
TestParams transpose_ukernels[] = {
#define XNN_TRANSPOSE_UKERNEL(arch_flags, ukernel, element_size, element_type, \
                              block_width, block_height)                       \
  {#ukernel,     arch_flags,  make_ukernel_wrapper(ukernel),                   \
   element_size, block_width, block_height},
#include "src/x8-transposec/x8-transposec.h"
#include "src/x16-transposec/x16-transposec.h"
#include "src/x24-transposec/x24-transposec.h"
#include "src/x32-transposec/x32-transposec.h"
#include "src/x64-transposec/x64-transposec.h"
#include "src/xx-transposev/xx-transposev.h"
};
#undef XNN_TRANSPOSE_UKERNEL

// This set of test params has only transposev ukernels.
TestParams transposev_ukernels[] = {
#define XNN_TRANSPOSE_UKERNEL(arch_flags, ukernel, element_size, element_type, \
                              block_width, block_height)                       \
  {#ukernel,     arch_flags,  make_ukernel_wrapper(ukernel),                   \
   element_size, block_width, block_height},
#include "src/xx-transposev/xx-transposev.h"
};
#undef XNN_TRANSPOSE_UKERNEL

class TransposeTest : public testing::TestWithParam<TestParams> {};

TEST_P(TransposeTest, bh_bw) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width * 2;
  const size_t output_stride = block_height * 2;
  const size_t width = block_width;
  const size_t height = block_height;
  TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                element_size, element_size, width, height);
}

TEST_P(TransposeTest, bh_1_bhx2_bw_1_bwx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  for (size_t i = 1; i <= block_height * 2; ++i) {
    for (size_t j = 1; j <= block_width * 2; ++j) {
      const size_t input_stride = j * 3;
      const size_t output_stride = i * 7;
      const size_t width = j;
      const size_t height = i;
      TestTranspose(GetParam().ukernel, input_stride, output_stride,
                    element_size, element_size, element_size, width, height);
    }
  }
}

TEST_P(TransposeTest, bh_bwx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width * 2;
  const size_t output_stride = block_height;
  const size_t width = block_width * 2;
  const size_t height = block_height;
  TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                element_size, element_size, width, height);
}

TEST_P(TransposeTest, bh_bwp1_bwx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  for (size_t i = block_width + 1; i < block_width * 2; ++i) {
    const size_t input_stride = i;
    const size_t output_stride = block_height * 2;
    const size_t width = i;
    const size_t height = block_height;
    TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                  element_size, element_size, width, height);
  }
}

TEST_P(TransposeTest, bhx2_bwp1_bwx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  for (size_t i = block_width + 1; i < block_width * 2; ++i) {
    const size_t input_stride = i;
    const size_t output_stride = block_height * 2;
    const size_t width = i;
    const size_t height = block_height * 2;
    TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                  element_size, element_size, width, height);
  }
}

TEST_P(TransposeTest, bhx2_bw) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width;
  const size_t output_stride = block_height * 3 + 4;
  const size_t width = block_width;
  const size_t height = block_height * 2;
  TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                element_size, element_size, width, height);
}

TEST_P(TransposeTest, bhp1_bhx2_bw) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  for (size_t i = block_height + 1; i < block_height * 2; ++i) {
    const size_t input_stride = block_width + 17;
    const size_t output_stride = i;
    const size_t width = block_width + 3;
    const size_t height = i;
    TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                  element_size, element_size, width, height);
  }
}

TEST_P(TransposeTest, bhp1_bhx2_bwx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  for (size_t i = block_height + 1; i < block_height * 2; ++i) {
    const size_t input_stride = block_width * 2;
    const size_t output_stride = i;
    const size_t width = block_width * 2;
    const size_t height = i;
    TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                  element_size, element_size, width, height);
  }
}

TEST_P(TransposeTest, bhp1_bhx2_bwp1_bwx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  for (size_t i = block_height + 1; i < block_height * 2; ++i) {
    for (size_t j = block_width + 1; j < block_width * 2; ++j) {
      const size_t input_stride = j;
      const size_t output_stride = i;
      const size_t width = j;
      const size_t height = i;
      TestTranspose(GetParam().ukernel, input_stride, output_stride,
                    element_size, element_size, element_size, width, height);
    }
  }
}

TEST_P(TransposeTest, bh_bw_is_bwx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width * 2;
  const size_t output_stride = block_height;
  const size_t width = block_width;
  const size_t height = block_height;
  TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                element_size, element_size, width, height);
}

TEST_P(TransposeTest, bh_bw_os_bhx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width;
  const size_t output_stride = block_height * 2;
  const size_t width = block_width;
  const size_t height = block_height;
  TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                element_size, element_size, width, height);
}

TEST_P(TransposeTest, bh_bw_is_bwx2_os_bhx2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width * 2;
  const size_t output_stride = block_height * 2;
  const size_t width = block_width;
  const size_t height = block_height;
  TestTranspose(GetParam().ukernel, input_stride, output_stride, element_size,
                element_size, element_size, width, height);
}

INSTANTIATE_TEST_SUITE_P(transpose, TransposeTest,
                         ::testing::ValuesIn(transpose_ukernels),
                         [](const auto& info) { return info.param.name; });

class TransposeVTest : public testing::TestWithParam<TestParams> {};

TEST_P(TransposeVTest, bhx17_bwx19_ies_esp11) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width * 19;
  const size_t output_stride = block_height * 17;
  const size_t width = block_width * 19;
  const size_t height = block_height * 17;
  const size_t input_element_stride = element_size + 11;
  const size_t output_element_stride = element_size;
  TestTranspose(GetParam().ukernel, input_stride, output_stride,
                input_element_stride, output_element_stride, element_size,
                width, height);
}

TEST_P(TransposeVTest, bhx3_bwx5_oes_esp11) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width * 5;
  const size_t output_stride = block_height * 3;
  const size_t width = block_width * 5;
  const size_t height = block_height * 3;
  const size_t input_element_stride = element_size;
  const size_t output_element_stride = element_size + 11;
  TestTranspose(GetParam().ukernel, input_stride, output_stride,
                input_element_stride, output_element_stride, element_size,
                width, height);
}

TEST_P(TransposeVTest, bhx7_bwx23_ies_esp17_oes_esp13) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t element_size = GetParam().element_size / 8;
  const size_t block_width = GetParam().block_width;
  const size_t block_height = GetParam().block_height;
  // if (get_batch_scale(element_size) < block_height) GTEST_SKIP();
  const size_t input_stride = block_width * 23 + 5;
  const size_t output_stride = block_height * 7 + 6;
  const size_t width = block_width * 23;
  const size_t height = block_height * 7;
  const size_t input_element_stride = element_size + 17;
  const size_t output_element_stride = element_size + 13;
  TestTranspose(GetParam().ukernel, input_stride, output_stride,
                input_element_stride, output_element_stride, element_size,
                width, height);
}

INSTANTIATE_TEST_SUITE_P(transposev, TransposeVTest,
                         ::testing::ValuesIn(transposev_ukernels),
                         [](const auto& info) { return info.param.name; });