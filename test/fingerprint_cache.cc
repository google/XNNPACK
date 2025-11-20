// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/operators/fingerprint_cache.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/experimental.h"
#include "include/xnnpack.h"
#include "src/operators/fingerprint_id.h"
#include "src/xnnpack/cache.h"

using ::testing::Eq;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::SizeIs;
using ::testing::StrEq;

TEST(FingerprintIdTest, AllIdsAreUnique) {
  std::vector<xnn_fingerprint_id> ids{
#define XNN_FINGERPRINT_ID(op, i, o, ...) \
  XNN_FINGERPRINT_ID_NAME(op, i, o, __VA_ARGS__),
#include "src/operators/fingerprint_id.h.inc"
#undef XNN_FINGERPRINT_ID
  };
  std::set<xnn_fingerprint_id> unique_ids(ids.begin(), ids.end());
  EXPECT_THAT(unique_ids, SizeIs(ids.size()));
}

TEST(FingerprintIdTest, ComputeFingerprintIdValueWorks) {
  EXPECT_THAT(xnn_compute_fingerprint_id_value(
                  xnn_fingerprint_id_helper_test, xnn_fingerprint_id_helper_f16,
                  xnn_fingerprint_id_helper_f32, xnn_fingerprint_id_helper_qc8w,
                  xnn_fingerprint_id_helper_example_flag, 0),
              xnn_fingerprint_id_test_f16_f32_qc8w_example_flag);
}

TEST(FingerprintIdTest, ToStringIsCorrect) {
  EXPECT_THAT(xnn_fingerprint_id_to_string(
                  xnn_fingerprint_id_test_f16_f32_qc8w_example_flag),
              StrEq("xnn_fingerprint_id_test_f16_f32_qc8w_example_flag"));
}

struct FingerprintCacheTest : testing::Test {
  void SetUp() override {
    xnn_initialize(nullptr);
    xnn_clear_fingerprints();
  }

  void TearDown() override {
    xnn_deinitialize();
    xnn_clear_fingerprints();
  }
};

TEST_F(FingerprintCacheTest, SetAndGetFingerprint) {
  const struct xnn_fingerprint expected{
      /*id=*/xnn_fingerprint_id_test_f16_f32_qc8w_example_flag,
      /*value=*/314};
  xnn_set_fingerprint(expected);
  const struct xnn_fingerprint* fingerprint = xnn_get_fingerprint(expected.id);
  ASSERT_THAT(fingerprint, NotNull());
  EXPECT_THAT(fingerprint->id, Eq(expected.id));
  EXPECT_THAT(fingerprint->value, Eq(expected.value));
}

TEST_F(FingerprintCacheTest, SetGetFingerprintMultipleTimesDoesntDeadlock) {
  constexpr struct xnn_fingerprint finger1{
      /*id=*/xnn_fingerprint_id_test_f16_f32_qc8w_example_flag,
      /*value=*/314};
  constexpr struct xnn_fingerprint finger2{
      /*id=*/xnn_fingerprint_id_test_f16_f32_qc8w_example_flag,
      /*value=*/654};
  static_assert(finger1.id == finger2.id,
                "This test checks that updating a fingerprint value doesn't "
                "deadlock. Do use different ids for the two fingerprints.");
  xnn_set_fingerprint(finger1);
  xnn_set_fingerprint(finger1);
  // Update the value of the fingerprint.
  xnn_set_fingerprint(finger2);
  xnn_set_fingerprint(finger2);

  xnn_get_fingerprint(finger1.id);
  xnn_get_fingerprint(finger2.id);
  xnn_get_fingerprint(finger1.id);
  xnn_get_fingerprint(finger2.id);
}

TEST_F(FingerprintCacheTest, InitializeAndFinalize) {
  struct fingerprint_context context = create_fingerprint_context(
      xnn_fingerprint_id_test_f16_f32_qc8w_example_flag);
  EXPECT_THAT(context.status, Eq(xnn_status_uninitialized));
  EXPECT_THAT(context.fingerprint_id,
              Eq(xnn_fingerprint_id_test_f16_f32_qc8w_example_flag));
  finalize_fingerprint_context(&context);
  EXPECT_THAT(
      xnn_get_fingerprint(xnn_fingerprint_id_test_f16_f32_qc8w_example_flag),
      NotNull());
}

TEST_F(FingerprintCacheTest, ReserveAndWrite) {
  constexpr size_t kBufferSize = 8;
  struct fingerprint_context context = create_fingerprint_context(
      xnn_fingerprint_id_test_f16_f32_qc8w_example_flag);
  uint8_t* buffer = reinterpret_cast<uint8_t*>(
      context.cache.reserve_space(context.cache.context, kBufferSize));
  ASSERT_THAT(buffer, NotNull());
  std::iota(buffer, buffer + kBufferSize, 1);
  // We just need a random key.
  const xnn_weights_cache_look_up_key key = {0};
  context.cache.look_up_or_insert(context.cache.context, &key, buffer,
                                  kBufferSize);
  // The fingerprinting cache never returns a positive look up.
  EXPECT_THAT(context.cache.look_up(context.cache.context, &key),
              Eq(XNN_CACHE_NOT_FOUND));
  finalize_fingerprint_context(&context);
  const xnn_fingerprint* fingerprint =
      xnn_get_fingerprint(xnn_fingerprint_id_test_f16_f32_qc8w_example_flag);
  ASSERT_THAT(fingerprint, NotNull());
  EXPECT_THAT(fingerprint->id,
              Eq(xnn_fingerprint_id_test_f16_f32_qc8w_example_flag));
  EXPECT_THAT(fingerprint->value, Not(Eq(0)));
}
