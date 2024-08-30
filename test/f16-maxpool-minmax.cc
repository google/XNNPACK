// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <limits>
#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams-init.h"
#include "maxpool-microkernel-tester.h"
#include "next_prime.h"

namespace {

// _Float16 isn't supported on all targets, but all we really need is to specify
// any type that is neither u8 nor s8 for our purposes here.
using XnnTest = XnnMaxpoolTest<xnn_f16_maxpool_ukernel_fn, xnn_init_f16_minmax_params_fn, xnn_float16>;
using XnnTestParam = XnnMaxpoolTestParam<xnn_f16_maxpool_ukernel_fn, xnn_init_f16_minmax_params_fn>;

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax, params_type, init_params) \
  { #ukernel, ukernel, init_params, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax },

const XnnTestParam xnn_test_params[] = {
#include "src/f16-maxpool/f16-maxpool-minmax.h"
};

#undef XNN_UKERNEL_WITH_PARAMS

}  // namespace

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile) {
	XnnTest::channels_eq_channel_tile_unipass_fulltile();
}
TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_input_offset) {
	XnnTest::channels_eq_channel_tile_unipass_fulltile_with_input_offset();
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
TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile) {
	XnnTest::channels_div_channel_tile_unipass_fulltile();
}
TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_input_offset) {
	XnnTest::channels_div_channel_tile_unipass_fulltile_with_input_offset();
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
TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile) {
	XnnTest::channels_lt_channel_tile_unipass_fulltile();
}
TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_input_offset) {
	XnnTest::channels_lt_channel_tile_unipass_fulltile_with_input_offset();
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
TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile) {
	XnnTest::channels_gt_channel_tile_unipass_fulltile();
}
TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_input_offset) {
	XnnTest::channels_gt_channel_tile_unipass_fulltile_with_input_offset();
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
TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile) {
	XnnTest::channels_eq_channel_tile_twopass_fulltile();
}
TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_input_offset) {
	XnnTest::channels_eq_channel_tile_twopass_fulltile_with_input_offset();
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
TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile) {
	XnnTest::channels_div_channel_tile_twopass_fulltile();
}
TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_input_offset) {
	XnnTest::channels_div_channel_tile_twopass_fulltile_with_input_offset();
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
TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile) {
	XnnTest::channels_lt_channel_tile_twopass_fulltile();
}
TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_input_offset) {
	XnnTest::channels_lt_channel_tile_twopass_fulltile_with_input_offset();
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
TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile) {
	XnnTest::channels_gt_channel_tile_twopass_fulltile();
}
TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_input_offset) {
	XnnTest::channels_gt_channel_tile_twopass_fulltile_with_input_offset();
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
TEST_P(XnnTest, channels_eq_channel_tile_multipass) {
	XnnTest::channels_eq_channel_tile_multipass();
}
TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_input_offset) {
	XnnTest::channels_eq_channel_tile_multipass_with_input_offset();
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

INSTANTIATE_TEST_SUITE_P(f16_maxpool_minmax,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params, xnn_test_params + sizeof(xnn_test_params)/sizeof(*xnn_test_params)),
                         GetTestName);
// xnn_test_params[] can be empty for some build configurations
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(XnnTest);

