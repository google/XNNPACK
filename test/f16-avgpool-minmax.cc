// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pavgpool.h"
#include "avgpool-microkernel-tester.h"

namespace {

using XnnTest = XnnAvgPoolTest<AvgPoolMicrokernelTester::TestF16AvgPoolFns>;
using XnnTestParam = XnnAvgPoolTestParam<AvgPoolMicrokernelTester::TestF16AvgPoolFns>;

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  { #ukernel, {nullptr, ukernel, init_params}, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile },

#define XNN_UKERNEL_UNIPASS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  { #ukernel, {ukernel, nullptr, init_params}, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile },

const XnnTestParam xnn_test_params[] = {
#include "src/f16-avgpool/f16-avgpool-minmax.h"
};

#undef XNN_UKERNEL_MULTIPASS
#undef XNN_UKERNEL_UNIPASS

}  // namespace

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile) {
  XnnTest::channels_eq_channel_tile_unipass_fulltile();
}
TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_input_offset) {
  XnnTest::channels_eq_channel_tile_unipass_fulltile_with_input_offset();
}
TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_zero) {
  XnnTest::channels_eq_channel_tile_unipass_fulltile_with_zero();
}
TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmin) {
  XnnTest::channels_eq_channel_tile_unipass_fulltile_with_qmin();
}
TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmax) {
  XnnTest::channels_eq_channel_tile_unipass_fulltile_with_qmax();
}
TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile) {
  XnnTest::channels_eq_channel_tile_unipass_subtile();
}
TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile_with_input_offset) {
  XnnTest::channels_eq_channel_tile_unipass_subtile_with_input_offset();
}
TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile_with_zero) {
  XnnTest::channels_eq_channel_tile_unipass_subtile_with_zero();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile) {
  XnnTest::channels_div_channel_tile_unipass_fulltile();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_input_offset) {
  XnnTest::channels_div_channel_tile_unipass_fulltile_with_input_offset();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_zero) {
  XnnTest::channels_div_channel_tile_unipass_fulltile_with_zero();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_qmin) {
  XnnTest::channels_div_channel_tile_unipass_fulltile_with_qmin();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_qmax) {
  XnnTest::channels_div_channel_tile_unipass_fulltile_with_qmax();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile) {
  XnnTest::channels_div_channel_tile_unipass_subtile();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile_with_input_offset) {
  XnnTest::channels_div_channel_tile_unipass_subtile_with_input_offset();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile_with_zero) {
  XnnTest::channels_div_channel_tile_unipass_subtile_with_zero();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile) {
  XnnTest::channels_lt_channel_tile_unipass_fulltile();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_input_offset) {
  XnnTest::channels_lt_channel_tile_unipass_fulltile_with_input_offset();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_zero) {
  XnnTest::channels_lt_channel_tile_unipass_fulltile_with_zero();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_qmin) {
  XnnTest::channels_lt_channel_tile_unipass_fulltile_with_qmin();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_qmax) {
  XnnTest::channels_lt_channel_tile_unipass_fulltile_with_qmax();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile) {
  XnnTest::channels_lt_channel_tile_unipass_subtile();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile_with_input_offset) {
  XnnTest::channels_lt_channel_tile_unipass_subtile_with_input_offset();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile_with_zero) {
  XnnTest::channels_lt_channel_tile_unipass_subtile_with_zero();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile) {
  XnnTest::channels_gt_channel_tile_unipass_fulltile();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_input_offset) {
  XnnTest::channels_gt_channel_tile_unipass_fulltile_with_input_offset();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_zero) {
  XnnTest::channels_gt_channel_tile_unipass_fulltile_with_zero();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmin) {
  XnnTest::channels_gt_channel_tile_unipass_fulltile_with_qmin();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmax) {
  XnnTest::channels_gt_channel_tile_unipass_fulltile_with_qmax();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile) {
  XnnTest::channels_gt_channel_tile_unipass_subtile();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile_with_input_offset) {
  XnnTest::channels_gt_channel_tile_unipass_subtile_with_input_offset();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile_with_zero) {
  XnnTest::channels_gt_channel_tile_unipass_subtile_with_zero();
}
TEST_P(XnnTest, few_output_pixels_0) {
  XnnTest::few_output_pixels_0();
}
TEST_P(XnnTest, few_output_pixels_with_input_offset_0) {
  XnnTest::few_output_pixels_with_input_offset_0();
}
TEST_P(XnnTest, few_output_pixels_with_zero_0) {
  XnnTest::few_output_pixels_with_zero_0();
}
TEST_P(XnnTest, few_output_pixels_with_qmin_0) {
  XnnTest::few_output_pixels_with_qmin_0();
}
TEST_P(XnnTest, few_output_pixels_with_qmax_0) {
  XnnTest::few_output_pixels_with_qmax_0();
}
TEST_P(XnnTest, few_output_pixels_with_output_stride_0) {
  XnnTest::few_output_pixels_with_output_stride_0();
}
TEST_P(XnnTest, few_output_pixels_with_step_0) {
  XnnTest::few_output_pixels_with_step_0();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile) {
  XnnTest::channels_eq_channel_tile_twopass_fulltile();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_input_offset) {
  XnnTest::channels_eq_channel_tile_twopass_fulltile_with_input_offset();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_zero) {
  XnnTest::channels_eq_channel_tile_twopass_fulltile_with_zero();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmin) {
  XnnTest::channels_eq_channel_tile_twopass_fulltile_with_qmin();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmax) {
  XnnTest::channels_eq_channel_tile_twopass_fulltile_with_qmax();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile) {
  XnnTest::channels_eq_channel_tile_twopass_subtile();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile_with_input_offset) {
  XnnTest::channels_eq_channel_tile_twopass_subtile_with_input_offset();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile_with_zero) {
  XnnTest::channels_eq_channel_tile_twopass_subtile_with_zero();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile) {
  XnnTest::channels_div_channel_tile_twopass_fulltile();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_input_offset) {
  XnnTest::channels_div_channel_tile_twopass_fulltile_with_input_offset();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_zero) {
  XnnTest::channels_div_channel_tile_twopass_fulltile_with_zero();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_qmin) {
  XnnTest::channels_div_channel_tile_twopass_fulltile_with_qmin();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_qmax) {
  XnnTest::channels_div_channel_tile_twopass_fulltile_with_qmax();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile) {
  XnnTest::channels_div_channel_tile_twopass_subtile();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile_with_input_offset) {
  XnnTest::channels_div_channel_tile_twopass_subtile_with_input_offset();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile_with_zero) {
  XnnTest::channels_div_channel_tile_twopass_subtile_with_zero();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile) {
  XnnTest::channels_lt_channel_tile_twopass_fulltile();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_input_offset) {
  XnnTest::channels_lt_channel_tile_twopass_fulltile_with_input_offset();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_zero) {
  XnnTest::channels_lt_channel_tile_twopass_fulltile_with_zero();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_qmin) {
  XnnTest::channels_lt_channel_tile_twopass_fulltile_with_qmin();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_qmax) {
  XnnTest::channels_lt_channel_tile_twopass_fulltile_with_qmax();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile) {
  XnnTest::channels_lt_channel_tile_twopass_subtile();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile_with_input_offset) {
  XnnTest::channels_lt_channel_tile_twopass_subtile_with_input_offset();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile_with_zero) {
  XnnTest::channels_lt_channel_tile_twopass_subtile_with_zero();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile) {
  XnnTest::channels_gt_channel_tile_twopass_fulltile();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_input_offset) {
  XnnTest::channels_gt_channel_tile_twopass_fulltile_with_input_offset();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_zero) {
  XnnTest::channels_gt_channel_tile_twopass_fulltile_with_zero();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmin) {
  XnnTest::channels_gt_channel_tile_twopass_fulltile_with_qmin();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmax) {
  XnnTest::channels_gt_channel_tile_twopass_fulltile_with_qmax();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile) {
  XnnTest::channels_gt_channel_tile_twopass_subtile();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile_with_input_offset) {
  XnnTest::channels_gt_channel_tile_twopass_subtile_with_input_offset();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile_with_zero) {
  XnnTest::channels_gt_channel_tile_twopass_subtile_with_zero();
}
TEST_P(XnnTest, channels_eq_channel_tile_multipass) {
  XnnTest::channels_eq_channel_tile_multipass();
}
TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_input_offset) {
  XnnTest::channels_eq_channel_tile_multipass_with_input_offset();
}
TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_zero) {
  XnnTest::channels_eq_channel_tile_multipass_with_zero();
}
TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmin) {
  XnnTest::channels_eq_channel_tile_multipass_with_qmin();
}
TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmax) {
  XnnTest::channels_eq_channel_tile_multipass_with_qmax();
}
TEST_P(XnnTest, channels_div_channel_tile_multipass) {
  XnnTest::channels_div_channel_tile_multipass();
}
TEST_P(XnnTest, channels_div_channel_tile_multipass_with_input_offset) {
  XnnTest::channels_div_channel_tile_multipass_with_input_offset();
}
TEST_P(XnnTest, channels_div_channel_tile_multipass_with_zero) {
  XnnTest::channels_div_channel_tile_multipass_with_zero();
}
TEST_P(XnnTest, channels_div_channel_tile_multipass_with_qmin) {
  XnnTest::channels_div_channel_tile_multipass_with_qmin();
}
TEST_P(XnnTest, channels_div_channel_tile_multipass_with_qmax) {
  XnnTest::channels_div_channel_tile_multipass_with_qmax();
}
TEST_P(XnnTest, channels_lt_channel_tile_multipass) {
  XnnTest::channels_lt_channel_tile_multipass();
}
TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_input_offset) {
  XnnTest::channels_lt_channel_tile_multipass_with_input_offset();
}
TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_zero) {
  XnnTest::channels_lt_channel_tile_multipass_with_zero();
}
TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_qmin) {
  XnnTest::channels_lt_channel_tile_multipass_with_qmin();
}
TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_qmax) {
  XnnTest::channels_lt_channel_tile_multipass_with_qmax();
}
TEST_P(XnnTest, channels_gt_channel_tile_multipass) {
  XnnTest::channels_gt_channel_tile_multipass();
}
TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_input_offset) {
  XnnTest::channels_gt_channel_tile_multipass_with_input_offset();
}
TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_zero) {
  XnnTest::channels_gt_channel_tile_multipass_with_zero();
}
TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmin) {
  XnnTest::channels_gt_channel_tile_multipass_with_qmin();
}
TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmax) {
  XnnTest::channels_gt_channel_tile_multipass_with_qmax();
}
TEST_P(XnnTest, few_output_pixels) {
  XnnTest::few_output_pixels();
}
TEST_P(XnnTest, few_output_pixels_with_input_offset) {
  XnnTest::few_output_pixels_with_input_offset();
}
TEST_P(XnnTest, few_output_pixels_with_zero) {
  XnnTest::few_output_pixels_with_zero();
}
TEST_P(XnnTest, few_output_pixels_with_qmin) {
  XnnTest::few_output_pixels_with_qmin();
}
TEST_P(XnnTest, few_output_pixels_with_qmax) {
  XnnTest::few_output_pixels_with_qmax();
}
TEST_P(XnnTest, few_output_pixels_with_output_stride) {
  XnnTest::few_output_pixels_with_output_stride();
}
TEST_P(XnnTest, few_output_pixels_with_step) {
  XnnTest::few_output_pixels_with_step();
}

INSTANTIATE_TEST_SUITE_P(f16_avgpool_minmax,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params, xnn_test_params + sizeof(xnn_test_params)/sizeof(*xnn_test_params)),
                         GetTestName);
// xnn_test_params[] can be empty for some build configurations
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(XnnTest);
