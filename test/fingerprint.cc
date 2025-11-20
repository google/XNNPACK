// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/experimental.h"
#include "include/xnnpack.h"
#include "src/operators/fingerprint_id.h"

namespace {

using ::testing::Eq;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::ValuesIn;

struct FingerprintTest : testing::TestWithParam<xnn_fingerprint_id> {
  xnn_fingerprint_id GetFingerprintId() const { return GetParam(); }
};

TEST_P(FingerprintTest, ComputeFingerprint) {
  xnn_initialize(nullptr);
  // Clear fingerprints to be sure that we actually compute the fingerprint and
  // don't get a fingerprint from a previous test execution.
  xnn_clear_fingerprints();

  // Check fingerprint will compute the fingerprint if it hasn't been computed
  // yet (which we expect). The check itself should fail as we pass in a value
  // of 0.
  xnn_status status =
      xnn_check_fingerprint({/*id=*/GetFingerprintId(), /*value=*/0});
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP() << "Skipping because hardware doesn't support this operation.";
  }
  ASSERT_THAT(status, Eq(xnn_status_invalid_parameter));

  const xnn_fingerprint* fingerprint = xnn_get_fingerprint(GetFingerprintId());
  ASSERT_THAT(fingerprint, NotNull());
  EXPECT_THAT(fingerprint->id, GetFingerprintId());
  EXPECT_THAT(fingerprint->value, Not(0));
}

std::vector<xnn_fingerprint_id> kFingerprintIds = {
#define XNN_FINGERPRINT_ID(operator, in, out, ...) \
  XNN_FINGERPRINT_ID_NAME(operator, in, out, __VA_ARGS__),
#include "src/operators/fingerprint_id.h.inc"
#undef XNN_FINGERPRINT_ID
};

INSTANTIATE_TEST_SUITE_P(
    FingerprintTest, FingerprintTest, ValuesIn(kFingerprintIds),
    [](const testing::TestParamInfo<xnn_fingerprint_id>& info) {
      return xnn_fingerprint_id_to_string(info.param);
    });

}  // namespace
