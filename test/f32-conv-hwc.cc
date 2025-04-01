// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <limits>
#include <string>

#include <gtest/gtest.h>
#include "replicable_random_device.h"
#include "xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/conv.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/pack.h"
#include "test/conv-hwc-microkernel-tester.h"

namespace {


struct XnnTestParam {
  const char *name;
  ConvHWCMicrokernelTester::Kernel kernel;
  uint64_t arch_flags;
  size_t kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile, input_widths;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile, input_widths, datatype, params_type, init_params) \
{ #ukernel, ConvHWCMicrokernelTester::Kernel{ukernel, init_params}, arch_flags, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile, input_widths },
const XnnTestParam xnn_test_params[] = {
#include "src/f32-conv-hwc/f32-conv-hwc.h"
};

}  // namespace

TEST_P(XnnTest, input_width_eq)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  ConvHWCMicrokernelTester()
    .kernel_size(GetParam().kernel_size)
    .subsampling(GetParam().subsampling)
    .input_channels(GetParam().input_channels)
    .output_channels_tile(GetParam().output_channels_tile)
    .output_channels(GetParam().output_channels_tile)
    .input_width(GetParam().input_widths)
    .input_height(GetParam().kernel_size)
    .padding(GetParam().padding_left, GetParam().padding_right)
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, input_width_div)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().input_widths <= 1) {
    GTEST_SKIP();
  }
  for (size_t input_width = GetParam().input_widths * 2; input_width < (GetParam().input_widths * 8);
       input_width += GetParam().input_widths * 3) {
    ConvHWCMicrokernelTester()
      .kernel_size(GetParam().kernel_size)
      .subsampling(GetParam().subsampling)
      .input_channels(GetParam().input_channels)
      .output_channels_tile(GetParam().output_channels_tile)
      .output_channels(GetParam().output_channels_tile)
      .input_width(input_width)
      .input_height(GetParam().kernel_size)
      .padding(GetParam().padding_left, GetParam().padding_right)
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, input_width_lt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);

  for (size_t input_width = (GetParam().padding_left ? 1 : 2); input_width < GetParam().input_widths; input_width++) {
    ConvHWCMicrokernelTester()
      .kernel_size(GetParam().kernel_size)
      .subsampling(GetParam().subsampling)
      .input_channels(GetParam().input_channels)
      .output_channels_tile(GetParam().output_channels_tile)
      .output_channels(GetParam().output_channels_tile)
      .input_width(input_width)
      .input_height(GetParam().kernel_size)
      .padding(GetParam().padding_left, GetParam().padding_right)
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, input_width_gt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);

  for (size_t input_width = GetParam().input_widths + 1; input_width < GetParam().input_widths * 2; input_width++) {
    ConvHWCMicrokernelTester()
      .kernel_size(GetParam().kernel_size)
      .subsampling(GetParam().subsampling)
      .input_channels(GetParam().input_channels)
      .output_channels_tile(GetParam().output_channels_tile)
      .output_channels(GetParam().output_channels_tile)
      .input_width(input_width)
      .input_height(GetParam().kernel_size)
      .padding(GetParam().padding_left, GetParam().padding_right)
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, output_channels_lt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);

  for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile; output_channels++) {
    for (size_t input_width = (GetParam().padding_left ? 1 : 2); input_width < GetParam().input_widths * 8;
         input_width += (GetParam().input_widths * 2 - 1)) {

      ConvHWCMicrokernelTester()
        .kernel_size(GetParam().kernel_size)
        .subsampling(GetParam().subsampling)
        .input_channels(GetParam().input_channels)
        .output_channels_tile(GetParam().output_channels_tile)
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(GetParam().kernel_size)
        .padding(GetParam().padding_left, GetParam().padding_right)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, output_channels_div)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);

  for (size_t output_channels = GetParam().output_channels_tile * 2;
       output_channels <= GetParam().output_channels_tile * 4; output_channels += GetParam().output_channels_tile) {

    for (size_t input_width = (GetParam().padding_left ? 1 : 2); input_width < GetParam().input_widths * 8;
         input_width += (GetParam().input_widths * 2 - 1)) {

      ConvHWCMicrokernelTester()
        .kernel_size(GetParam().kernel_size)
        .subsampling(GetParam().subsampling)
        .input_channels(GetParam().input_channels)
        .output_channels_tile(GetParam().output_channels_tile)
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(GetParam().kernel_size)
        .padding(GetParam().padding_left, GetParam().padding_right)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, output_channels_gt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);

  for (size_t output_channels = GetParam().output_channels_tile + 1;
       output_channels < GetParam().output_channels_tile * 2; output_channels++) {

    for (size_t input_width = (GetParam().padding_left ? 1 : 2); input_width < GetParam().input_widths * 8;
         input_width += (GetParam().input_widths * 2 - 1)) {

      ConvHWCMicrokernelTester()
        .kernel_size(GetParam().kernel_size)
        .subsampling(GetParam().subsampling)
        .input_channels(GetParam().input_channels)
        .output_channels_tile(GetParam().output_channels_tile)
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(GetParam().kernel_size)
        .padding(GetParam().padding_left, GetParam().padding_right)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, input_height_lt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t input_heights = 1; input_heights < 3; input_heights++) {
    for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile * 2;
         output_channels += GetParam().output_channels_tile - 1) {
      for (size_t input_widths_ = (GetParam().padding_left ? 1 : 2); input_widths_ < GetParam().input_widths * 8;
           input_widths_ += (GetParam().input_widths * 2 - 1)) {
        auto tester = ConvHWCMicrokernelTester()
          .kernel_size(GetParam().kernel_size)
          .subsampling(GetParam().subsampling)
          .input_channels(GetParam().input_channels)
          .output_channels_tile(GetParam().output_channels_tile)
          .output_channels(output_channels)
          .input_width(input_widths_)
          .input_height(input_heights);
        if (GetParam().padding_left == 0) {
          tester.padding_right(1);
          tester.padding_height(1);
        } else {
          tester.padding(1);
        }
        tester.Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, input_height_gt)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t input_heights = 4; input_heights <= 9; input_heights++) {
    for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile * 2;
         output_channels += GetParam().output_channels_tile - 1) {
      for (size_t input_widths_ = (GetParam().padding_left ? 1 : 2); input_widths_ < GetParam().input_widths * 8;
           input_widths_ += (GetParam().input_widths * 2 - 1)) {
        ConvHWCMicrokernelTester()
          .kernel_size(GetParam().kernel_size)
          .subsampling(GetParam().subsampling)
          .input_channels(GetParam().input_channels)
          .output_channels_tile(GetParam().output_channels_tile)
          .output_channels(output_channels)
          .input_width(input_widths_)
          .input_height(input_heights)
          .padding(GetParam().padding_left, GetParam().padding_right)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, padding_top)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t padding_tops = 0; padding_tops <= 1; padding_tops++) {
    for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile * 2;
         output_channels += GetParam().output_channels_tile - 1) {
      for (size_t input_widths_ = (GetParam().padding_left ? 1 : 2); input_widths_ < GetParam().input_widths * 8;
           input_widths_ += (GetParam().input_widths * 2 - 1)) {
        ConvHWCMicrokernelTester()
          .kernel_size(GetParam().kernel_size)
          .subsampling(GetParam().subsampling)
          .input_channels(GetParam().input_channels)
          .output_channels_tile(GetParam().output_channels_tile)
          .output_channels(output_channels)
          .input_width(input_widths_)
          .input_height(9)
          .padding(GetParam().padding_left, GetParam().padding_right)
          .padding_top(padding_tops)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, padding_bottom)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t padding_bottoms = 0; padding_bottoms <= 1; padding_bottoms++) {
    for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile * 2;
         output_channels += GetParam().output_channels_tile - 1) {
      for (size_t input_widths_ = (GetParam().padding_left ? 1 : 2); input_widths_ < GetParam().input_widths * 8;
           input_widths_ += (GetParam().input_widths * 2 - 1)) {
        ConvHWCMicrokernelTester()
          .kernel_size(GetParam().kernel_size)
          .subsampling(GetParam().subsampling)
          .input_channels(GetParam().input_channels)
          .output_channels_tile(GetParam().output_channels_tile)
          .output_channels(output_channels)
          .input_width(input_widths_)
          .input_height(9)
          .padding(GetParam().padding_left, GetParam().padding_right)
          .padding_bottom(padding_bottoms)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, output_y_start)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_y_starts = 1; output_y_starts <= 3; output_y_starts++) {
    for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile * 2;
         output_channels += GetParam().output_channels_tile - 1) {
      for (size_t input_widths_ = (GetParam().padding_left ? 1 : 2); input_widths_ < GetParam().input_widths * 8;
           input_widths_ += (GetParam().input_widths * 2 - 1)) {
        ConvHWCMicrokernelTester()
          .kernel_size(GetParam().kernel_size)
          .subsampling(GetParam().subsampling)
          .input_channels(GetParam().input_channels)
          .output_channels_tile(GetParam().output_channels_tile)
          .output_channels(output_channels)
          .input_width(input_widths_)
          .input_height(9)
          .output_y_start(output_y_starts)
          .padding(GetParam().padding_left, GetParam().padding_right)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, output_y_end)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_y_ends = 2; output_y_ends < 5; output_y_ends++) {
    for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile * 2;
         output_channels += GetParam().output_channels_tile - 1) {
      for (size_t input_widths_ = (GetParam().padding_left ? 1 : 2); input_widths_ < GetParam().input_widths * 8;
           input_widths_ += (GetParam().input_widths * 2 - 1)) {
        ConvHWCMicrokernelTester()
          .kernel_size(GetParam().kernel_size)
          .subsampling(GetParam().subsampling)
          .input_channels(GetParam().input_channels)
          .output_channels_tile(GetParam().output_channels_tile)
          .output_channels(output_channels)
          .input_width(input_widths_)
          .input_height(9)
          .output_y_end(output_y_ends)
          .padding(GetParam().padding_left, GetParam().padding_right)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, qmin)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile * 2;
       output_channels += GetParam().output_channels_tile - 1) {
    for (size_t input_widths_ = (GetParam().padding_left ? 1 : 2); input_widths_ < GetParam().input_widths * 8;
         input_widths_ += (GetParam().input_widths * 2 - 1)) {
      ConvHWCMicrokernelTester()
        .kernel_size(GetParam().kernel_size)
        .subsampling(GetParam().subsampling)
        .input_channels(GetParam().input_channels)
        .output_channels_tile(GetParam().output_channels_tile)
        .output_channels(output_channels)
        .input_width(input_widths_)
        .input_height(6)
        .qmin(128)
        .padding(GetParam().padding_left, GetParam().padding_right)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, qmax)
{
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_channels = 1; output_channels < GetParam().output_channels_tile * 2;
       output_channels += GetParam().output_channels_tile - 1) {
    for (size_t input_widths_ = (GetParam().padding_left ? 1 : 2); input_widths_ < GetParam().input_widths * 8;
         input_widths_ += (GetParam().input_widths * 2 - 1)) {
      ConvHWCMicrokernelTester()
        .kernel_size(GetParam().kernel_size)
        .subsampling(GetParam().subsampling)
        .input_channels(GetParam().input_channels)
        .output_channels_tile(GetParam().output_channels_tile)
        .output_channels(output_channels)
        .input_width(input_widths_)
        .input_height(6)
        .qmax(128)
        .padding(GetParam().padding_left, GetParam().padding_right)
        .Test(GetParam().kernel);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(conv_hwc,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);
