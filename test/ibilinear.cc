// clang-format off
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/ibilinear.h"
#include "src/xnnpack/isa-checks.h"
#include "test/ibilinear-microkernel-tester.h"
#include "test/next_prime.h"

namespace {


struct XnnTestParam {
  const char *name;
  IBilinearMicrokernelTester::Kernel kernel;
  uint64_t arch_flags;
  size_t channel_tile, pixel_tile;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, pixel_tile, datatype, weight_type, params_type, init_params) \
{ #ukernel, IBilinearMicrokernelTester::Kernel{ukernel}, arch_flags, channel_tile, pixel_tile },

const XnnTestParam xnn_test_params[] = {
#include "src/f16-ibilinear/f16-ibilinear.h"
#include "src/f32-ibilinear/f32-ibilinear.h"
#include "src/s8-ibilinear/s8-ibilinear.h"
#include "src/u8-ibilinear/u8-ibilinear.h"
};

}  // namespace


TEST_P(XnnTest, channels_eq)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  IBilinearMicrokernelTester()
    .pixels(GetParam().pixel_tile)
    .channels(GetParam().channel_tile)
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_div)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().channel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 10; channels += GetParam().channel_tile) { 
    IBilinearMicrokernelTester()
    .pixels(GetParam().pixel_tile)
    .channels(channels)
    .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().channel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t channels = 1; channels < GetParam().channel_tile; channels++) {
    IBilinearMicrokernelTester()
    .pixels(GetParam().pixel_tile)
    .channels(channels)
    .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1; channels < ((GetParam().channel_tile == 1) ? 10 : GetParam().channel_tile * 2); channels++) {
    IBilinearMicrokernelTester()
    .pixels(GetParam().pixel_tile)
    .channels(channels)
    .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, pixels_div)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().pixel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t pixels = GetParam().pixel_tile * 2; pixels < GetParam().pixel_tile * 10; pixels += GetParam().pixel_tile) {
    for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += max(1, (GetParam().channel_tile - 1))) {
      IBilinearMicrokernelTester()
      .pixels(pixels)
      .channels(channels)
      .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, pixels_lt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().pixel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t pixels = 1; pixels < GetParam().pixel_tile; pixels++) {
    for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += max(1, (GetParam().channel_tile - 1))) {
      IBilinearMicrokernelTester()
      .pixels(pixels)
      .channels(channels)
      .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, pixels_gt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pixels = GetParam().pixel_tile + 1; pixels < max((GetParam().pixel_tile * 2), 3); pixels++) {
    for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += max(1, (GetParam().channel_tile - 1))) {
      IBilinearMicrokernelTester()
      .pixels(pixels)
      .channels(channels)
      .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, input_offset)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pixels = 1; pixels < GetParam().pixel_tile * 5; pixels += max(1, (GetParam().pixel_tile - 1))) {               
    for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += max(1, (GetParam().channel_tile - 1))) {
      IBilinearMicrokernelTester()
      .pixels(pixels)
      .channels(channels)
      .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
      .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, output_stride)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pixels = 1; pixels < GetParam().pixel_tile * 5; pixels += max(1, (GetParam().pixel_tile - 1))) {
    for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += max(1, (GetParam().channel_tile - 1))) {
      IBilinearMicrokernelTester()
      .pixels(pixels)
      .channels(channels)
      .output_stride(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
      .Test(GetParam().kernel);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ibilinear,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);
