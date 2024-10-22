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
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/pad.h"
#include "xnnpack/buffer.h"
#include "replicable_random_device.h"

class PadMicrokernelTester {
 public:
  PadMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const { return this->rows_; }

  PadMicrokernelTester& input_channels(size_t input_channels) {
    assert(input_channels != 0);
    this->input_channels_ = input_channels;
    return *this;
  }

  size_t input_channels() const { return this->input_channels_; }

  PadMicrokernelTester& pre_padding(size_t pre_padding) {
    this->pre_padding_ = pre_padding;
    return *this;
  }

  size_t pre_padding() const { return this->pre_padding_; }

  PadMicrokernelTester& post_padding(size_t post_padding) {
    this->post_padding_ = post_padding;
    return *this;
  }

  size_t post_padding() const { return this->post_padding_; }

  size_t output_channels() const {
    return pre_padding() + input_channels() + post_padding();
  }

  PadMicrokernelTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return input_channels();
    } else {
      assert(this->input_stride_ >= input_channels());
      return this->input_stride_;
    }
  }

  PadMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return pre_padding() + input_channels() + post_padding();
    } else {
      assert(this->output_stride_ >=
             pre_padding() + input_channels() + post_padding());
      return this->output_stride_;
    }
  }

  PadMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void Test(xnn_pad_ukernel_fn pad) const {
    xnnpack::ReplicableRandomDevice rng;
    auto u8rng = [&rng]() {
      return std::uniform_int_distribution<uint32_t>(
          0, std::numeric_limits<uint8_t>::max())(rng);
    };

    xnnpack::Buffer<uint8_t> input(input_channels() +
                               (rows() - 1) * input_stride() +
                               XNN_EXTRA_BYTES / sizeof(uint8_t));
    xnnpack::Buffer<uint8_t> output(
        (pre_padding() + input_channels() + post_padding()) +
        (rows() - 1) * output_stride());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng);
      xnnpack::fill_uniform_random_bits(output.data(), output.size(), rng);
      std::array<uint8_t, 4> fill_pattern;
      std::generate(fill_pattern.begin(), fill_pattern.end(), std::ref(u8rng));
      uint32_t fill_value = 0;
      memcpy(&fill_value, fill_pattern.data(), sizeof(fill_value));

      // Call optimized micro-kernel.
      pad(rows(), input_channels() * sizeof(uint8_t),
          pre_padding() * sizeof(uint8_t), post_padding() * sizeof(uint8_t),
          input.data(), input_stride() * sizeof(uint8_t), output.data(),
          output_stride() * sizeof(uint8_t), fill_value);

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t l = 0; l < pre_padding(); l++) {
          ASSERT_EQ(uint32_t(output[i * output_stride() + l]),
                    uint32_t(fill_pattern[l % fill_pattern.size()]))
              << "at row " << i << " / " << rows() << ", channel " << i << " / "
              << output_channels() << " (" << pre_padding() << " + "
              << input_channels() << " + " << post_padding() << ")"
              << ", fill value 0x" << std::hex << std::setw(8)
              << std::setfill('0') << fill_value << ", output value 0x"
              << std::hex << std::setw(2) << std::setfill('0')
              << uint32_t(output[i * output_stride() + l]);
        }
        for (size_t c = 0; c < input_channels(); c++) {
          ASSERT_EQ(uint32_t(output[i * output_stride() + pre_padding() + c]),
                    uint32_t(input[i * input_stride() + c]))
              << "at row " << i << " / " << rows() << ", channel " << i << " / "
              << output_channels() << " (" << pre_padding() << " + "
              << input_channels() << " + " << post_padding() << ")"
              << ", fill value 0x" << std::hex << std::setw(8)
              << std::setfill('0') << fill_value << ", output value 0x"
              << std::hex << std::setw(2) << std::setfill('0')
              << uint32_t(output[i * output_stride() + pre_padding() + c]);
        }
        for (size_t r = 0; r < post_padding(); r++) {
          ASSERT_EQ(uint32_t(output[i * output_stride() + pre_padding() +
                                    input_channels() + r]),
                    uint32_t(fill_pattern[r % fill_pattern.size()]))
              << "at row " << i << " / " << rows() << ", channel " << i << " / "
              << output_channels() << " (" << pre_padding() << " + "
              << input_channels() << " + " << post_padding() << ")"
              << ", fill value 0x" << std::hex << std::setw(8)
              << std::setfill('0') << fill_value << ", output value 0x"
              << std::hex << std::setw(2) << std::setfill('0')
              << uint32_t(output[i * output_stride() + pre_padding() +
                                 input_channels() + r]);
        }
      }
    }
  }

 private:
  size_t rows_{1};
  size_t input_channels_{1};
  size_t pre_padding_{0};
  size_t post_padding_{0};
  size_t input_stride_{0};
  size_t output_stride_{0};
  size_t iterations_{15};
};

struct TestParams {
  const char* name;
  uint64_t arch_flags;
  xnn_pad_ukernel_fn ukernel;
  size_t tile_size;
};

#define XNN_PAD_UKERNEL(arch_flags, ukernel, tile_size) \
  {#ukernel, arch_flags, ukernel, tile_size},
TestParams test_params[] = {
#include "xx-pad/xx-pad.h"
};
#undef XNN_PAD_UKERNEL

class PadTest : public testing::TestWithParam<TestParams> {};

TEST_P(PadTest, fulltile_copy_channels_eq_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  PadMicrokernelTester().rows(1).input_channels(tile_size).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_copy_channels_div_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t channels = tile_size * 2; channels <= tile_size * 3;
       channels += tile_size) {
    PadMicrokernelTester().rows(1).input_channels(channels).Test(
        GetParam().ukernel);
  }
}

TEST_P(PadTest, fulltile_copy_channels_lt_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t channels = 1; channels < tile_size; channels++) {
    PadMicrokernelTester().rows(1).input_channels(channels).Test(
        GetParam().ukernel);
  }
}

TEST_P(PadTest, fulltile_copy_channels_gt_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t channels = 17; channels < tile_size * 2; channels++) {
    PadMicrokernelTester().rows(1).input_channels(channels).Test(
        GetParam().ukernel);
  }
}

TEST_P(PadTest, fulltile_pre_padding_eq_1) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PadMicrokernelTester().rows(1).input_channels(1).pre_padding(1).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_pre_padding_eq_2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PadMicrokernelTester().rows(1).input_channels(1).pre_padding(2).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_pre_padding_eq_4) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PadMicrokernelTester().rows(1).input_channels(1).pre_padding(4).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_pre_padding_eq_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  PadMicrokernelTester().rows(1).input_channels(1).pre_padding(tile_size).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_pre_padding_div_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t pre_padding = tile_size * 2; pre_padding <= tile_size * 3;
       pre_padding += tile_size) {
    PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(GetParam().ukernel);
  }
}

TEST_P(PadTest, fulltile_pre_padding_lt_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t pre_padding = 1; pre_padding < tile_size; pre_padding++) {
    PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(GetParam().ukernel);
  }
}

TEST_P(PadTest, fulltile_pre_padding_gt_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t pre_padding = 17; pre_padding < tile_size * 2; pre_padding++) {
    PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(GetParam().ukernel);
  }
}

TEST_P(PadTest, fulltile_post_padding_eq_1) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PadMicrokernelTester().rows(1).input_channels(1).post_padding(1).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_post_padding_eq_2) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PadMicrokernelTester().rows(1).input_channels(1).post_padding(2).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_post_padding_eq_4) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PadMicrokernelTester().rows(1).input_channels(1).post_padding(4).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_post_padding_eq_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  PadMicrokernelTester().rows(1).input_channels(1).post_padding(tile_size).Test(
      GetParam().ukernel);
}

TEST_P(PadTest, fulltile_post_padding_div_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t post_padding = tile_size * 2; post_padding <= tile_size * 3;
       post_padding += tile_size) {
    PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(GetParam().ukernel);
  }
}

TEST_P(PadTest, fulltile_post_padding_lt_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t post_padding = 1; post_padding < tile_size; post_padding++) {
    PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(GetParam().ukernel);
  }
}

TEST_P(PadTest, fulltile_post_padding_gt_tile_size) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t post_padding = 17; post_padding < tile_size * 2; post_padding++) {
    PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(GetParam().ukernel);
  }
}

TEST_P(PadTest, multitile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < tile_size * 3; channels += 3) {
      PadMicrokernelTester()
          .rows(rows)
          .input_channels(channels)
          .pre_padding(channels)
          .post_padding(channels)
          .Test(GetParam().ukernel);
    }
  }
}

TEST_P(PadTest, multitile_with_input_stride) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < tile_size * 3; channels += 3) {
      PadMicrokernelTester()
          .rows(rows)
          .input_channels(channels)
          .pre_padding(channels)
          .post_padding(channels)
          .input_stride(51)
          .Test(GetParam().ukernel);
    }
  }
}

TEST_P(PadTest, multitile_with_output_stride) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t tile_size = GetParam().tile_size;
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < tile_size * 3; channels += 3) {
      PadMicrokernelTester()
          .rows(rows)
          .input_channels(2 * channels)
          .pre_padding(channels)
          .post_padding(channels)
          .output_stride(193)
          .Test(GetParam().ukernel);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(pad, PadTest, ::testing::ValuesIn(test_params),
                         [](const auto& info) { return info.param.name; });