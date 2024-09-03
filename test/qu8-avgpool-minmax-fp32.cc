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
#include "avgpool-microkernel-test-p.h"

namespace {

using XnnTestParam = XnnAvgPoolTestParam<AvgPoolMicrokernelTester::TestQU8AvgPoolFns>;
class XnnTest : public testing::TestWithParam<XnnTestParam> {};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, requantize, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  { #ukernel, {nullptr, ukernel, init_params, requantize}, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile },

#define XNN_UKERNEL_UNIPASS(arch_flags, ukernel, requantize, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  { #ukernel, {ukernel, nullptr, init_params, requantize}, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile },

const XnnTestParam xnn_test_params[] = {
#include "src/qu8-avgpool/qu8-avgpool-minmax.h"
};

#undef XNN_UKERNEL_MULTIPASS
#undef XNN_UKERNEL_UNIPASS

}  // namespace

#define XNN_TEST_SUITE_NAME XnnTest
#include "avgpool-microkernel-test-p.h"
#undef XNN_TEST_SUITE_NAME

INSTANTIATE_TEST_SUITE_P(qu8_avgpool_minmax,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params, xnn_test_params + sizeof(xnn_test_params)/sizeof(*xnn_test_params)),
                         GetTestName);
// xnn_test_params[] can be empty for some build configurations
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(XnnTest);
