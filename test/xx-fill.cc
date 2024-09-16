// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <ios>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/fill.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

class FillMicrokernelTester {
 public:
  FillMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const { return this->rows_; }

  FillMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const { return this->channels_; }

  FillMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      return this->output_stride_;
    }
  }

  FillMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void Test(xnn_fill_ukernel_fn fill) const {
    ASSERT_GE(output_stride(), channels());

    xnnpack::ReplicableRandomDevice rng;
    auto u8rng = [&rng]() {
      return std::uniform_int_distribution<uint32_t>(
          0, std::numeric_limits<uint8_t>::max())(rng);
    };

    std::vector<uint8_t> output((rows() - 1) * output_stride() + channels());
    std::vector<uint8_t> output_copy(output.size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(output.begin(), output.end(), std::ref(u8rng));
      std::copy(output.cbegin(), output.cend(), output_copy.begin());
      std::array<uint8_t, 4> fill_pattern;
      std::generate(fill_pattern.begin(), fill_pattern.end(), std::ref(u8rng));
      uint32_t fill_value = 0;
      memcpy(&fill_value, fill_pattern.data(), sizeof(fill_value));

      // Call optimized micro-kernel.
      fill(rows(), channels() * sizeof(uint8_t), output.data(),
           output_stride() * sizeof(uint8_t), fill_value);

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(uint32_t(output[i * output_stride() + c]),
                    uint32_t(fill_pattern[c % fill_pattern.size()]))
              << "at row " << i << " / " << rows() << ", channel " << c << " / "
              << channels() << ", fill value 0x" << std::hex << std::setw(8)
              << std::setfill('0') << fill_value << ", output value 0x"
              << std::hex << std::setw(8) << std::setfill('0')
              << output[i * output_stride() + c];
        }
      }
      for (size_t i = 0; i + 1 < rows(); i++) {
        for (size_t c = channels(); c < output_stride(); c++) {
          EXPECT_EQ(uint32_t(output[i * output_stride() + c]),
                    uint32_t(output_copy[i * output_stride() + c]))
              << "at row " << i << " / " << rows() << ", channel " << c << " / "
              << channels() << ", original value 0x" << std::hex << std::setw(8)
              << std::setfill('0') << output_copy[i * output_stride() + c]
              << ", output value 0x" << std::hex << std::setw(8)
              << std::setfill('0') << output[i * output_stride() + c];
        }
      }
    }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  size_t output_stride_{0};
  size_t iterations_{15};
};

struct TestParams {
  const char* name;
  uint64_t arch_flags;
  xnn_fill_ukernel_fn ukernel;
};

#define XNN_FILL_UKERNEL(arch_flags, ukernel) {#ukernel, arch_flags, ukernel},
TestParams test_params[] = {
#include "src/xx-fill/xx-fill.h"
};
#undef XNN_FILL_UKERNEL

class FillTest : public testing::TestWithParam<TestParams> {};

TEST_P(FillTest, channels_eq_64) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  FillMicrokernelTester().channels(64).Test(GetParam().ukernel);
}

TEST_P(FillTest, channels_div_64) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = 128; channels <= 192; channels += 64) {
    FillMicrokernelTester().channels(channels).Test(GetParam().ukernel);
  }
}

TEST_P(FillTest, channels_lt_64) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = 1; channels < 64; channels++) {
    FillMicrokernelTester().channels(channels).Test(GetParam().ukernel);
  }
}

TEST_P(FillTest, channels_gt_64) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = 65; channels < 128; channels++) {
    FillMicrokernelTester().channels(channels).Test(GetParam().ukernel);
  }
}

TEST_P(FillTest, multiple_rows) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < 192; channels += 15) {
      FillMicrokernelTester().channels(channels).rows(rows).Test(
          GetParam().ukernel);
    }
  }
}

TEST_P(FillTest, multiple_rows_with_output_stride) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < 192; channels += 15) {
      FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .output_stride(193)
          .Test(GetParam().ukernel);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(fill, FillTest, ::testing::ValuesIn(test_params),
                         [](const auto& info) { return info.param.name; });
