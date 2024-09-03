// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packb.h"
#include "packb-microkernel-tester.h"

namespace {

using XnnTest = XnnPackBTest<xnn_x32_packb_gemm_ukernel_fn>;
using XnnTestParam = XnnPackBTestParam<xnn_x32_packb_gemm_ukernel_fn>;

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_subtile, channel_round) \
  { #ukernel, ukernel, arch_flags, channel_tile, channel_subtile, channel_round },

const XnnTestParam xnn_test_params[] = {
#include "src/x32-packb/x32-packb.h"
};

#undef XNN_UKERNEL_WITH_PARAMS

}  // namespace

TEST_P(XnnTest, n_eq_channel_tile) {
	XnnTest::n_eq_channel_tile();
}
TEST_P(XnnTest, n_div_channel_tile) {
	XnnTest::n_div_channel_tile();
}
TEST_P(XnnTest, n_lt_channel_tile) {
	XnnTest::n_lt_channel_tile();
}
TEST_P(XnnTest, n_gt_channel_tile) {
	XnnTest::n_gt_channel_tile();
}
TEST_P(XnnTest, groups_gt_1) {
	XnnTest::groups_gt_1();
}

INSTANTIATE_TEST_SUITE_P(x32_packb,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

