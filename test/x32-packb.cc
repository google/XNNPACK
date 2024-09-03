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
#include "packb-microkernel-test-p.h"  // for XnnPackBTestParam

namespace {

using XnnTestParam = XnnPackBTestParam<xnn_x32_packb_gemm_ukernel_fn>;
class XnnTest : public testing::TestWithParam<XnnTestParam> {};

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

#define XNN_TEST_SUITE_NAME XnnTest
#include "packb-microkernel-test-p.h"
#undef XNN_TEST_SUITE_NAME

INSTANTIATE_TEST_SUITE_P(x32_packb,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

