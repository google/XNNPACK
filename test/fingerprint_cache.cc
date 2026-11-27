// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/operators/fingerprint_cache.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <set>
#include <thread>
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
                  xnn_fingerprint_id_helper_nr2, 0),
              xnn_fingerprint_id_test_f16_f32_qc8w_nr2);
}

TEST(FingerprintIdTest, ToStringIsCorrect) {
  EXPECT_THAT(
      xnn_fingerprint_id_to_string(xnn_fingerprint_id_test_f16_f32_qc8w_nr2),
      StrEq("xnn_fingerprint_id_test_f16_f32_qc8w_nr2"));
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
      /*id=*/xnn_fingerprint_id_test_f16_f32_qc8w_nr2,
      /*value=*/314};
  ASSERT_EQ(xnn_set_fingerprint(expected), xnn_status_success);
  const struct xnn_fingerprint* fingerprint = xnn_get_fingerprint(expected.id);
  ASSERT_THAT(fingerprint, NotNull());
  EXPECT_THAT(fingerprint->id, Eq(expected.id));
  EXPECT_THAT(fingerprint->value, Eq(expected.value));
}

TEST_F(FingerprintCacheTest, NewIdAtCapacityIsRejected) {
  constexpr uint32_t kCapacity = 256;
  for (uint32_t id = 0; id < kCapacity; ++id) {
    ASSERT_EQ(xnn_set_fingerprint({id, id + 1}), xnn_status_success);
  }
  const xnn_fingerprint* last = xnn_get_fingerprint_by_idx(kCapacity - 1);
  ASSERT_THAT(last, NotNull());
  EXPECT_EQ(last->id, kCapacity - 1);

  EXPECT_EQ(xnn_set_fingerprint({kCapacity, 123}), xnn_status_invalid_state);
  EXPECT_EQ(xnn_get_fingerprint_by_idx(kCapacity), nullptr);
  EXPECT_EQ(xnn_get_fingerprint(kCapacity), nullptr);

  EXPECT_EQ(xnn_set_fingerprint({0, 456}), xnn_status_success);
  const xnn_fingerprint* updated = xnn_get_fingerprint(0);
  ASSERT_THAT(updated, NotNull());
  EXPECT_EQ(updated->value, 456);

  xnn_clear_fingerprints();
  EXPECT_EQ(xnn_set_fingerprint({kCapacity, 123}), xnn_status_success);
  ASSERT_THAT(xnn_get_fingerprint(kCapacity), NotNull());
}

TEST_F(FingerprintCacheTest, ConcurrentInsertionsAtCapacity) {
  constexpr uint32_t kCapacity = 256;
  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kAttemptsPerThread = 64;
  std::vector<xnn_status> statuses(kThreads * kAttemptsPerThread);
  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (uint32_t thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      ready.fetch_add(1);
      while (!start.load()) {
        std::this_thread::yield();
      }
      for (uint32_t i = 0; i < kAttemptsPerThread; ++i) {
        const uint32_t id = thread * kAttemptsPerThread + i;
        statuses[id] = xnn_set_fingerprint({id, id + 1});
      }
    });
  }
  while (ready.load() != kThreads) {
    std::this_thread::yield();
  }
  start.store(true);
  for (std::thread& thread : threads) {
    thread.join();
  }

  uint32_t inserted = 0;
  for (uint32_t id = 0; id < statuses.size(); ++id) {
    if (statuses[id] == xnn_status_success) {
      ++inserted;
      const xnn_fingerprint* fingerprint = xnn_get_fingerprint(id);
      ASSERT_THAT(fingerprint, NotNull());
      EXPECT_EQ(fingerprint->value, id + 1);
    } else {
      EXPECT_EQ(statuses[id], xnn_status_invalid_state);
      EXPECT_EQ(xnn_get_fingerprint(id), nullptr);
    }
  }
  EXPECT_EQ(inserted, kCapacity);
  EXPECT_EQ(xnn_get_fingerprint_by_idx(kCapacity), nullptr);
}

TEST_F(FingerprintCacheTest, FinalizeReportsFullCache) {
  constexpr uint32_t kCapacity = 256;
  const uint32_t target_id = xnn_fingerprint_id_test_f16_f32_qc8w_nr2;
  for (uint32_t id = 0, inserted = 0; inserted < kCapacity; ++id) {
    if (id != target_id) {
      ASSERT_EQ(xnn_set_fingerprint({id, id + 1}), xnn_status_success);
      ++inserted;
    }
  }
  struct fingerprint_context context = create_fingerprint_context(
      xnn_fingerprint_id_test_f16_f32_qc8w_nr2);
  ASSERT_EQ(context.status, xnn_status_uninitialized);
  ASSERT_THAT(context.cache.context, NotNull());
  EXPECT_EQ(finalize_fingerprint_context(&context), xnn_status_invalid_state);
  EXPECT_EQ(xnn_get_fingerprint(target_id), nullptr);
}

TEST_F(FingerprintCacheTest, SetGetFingerprintMultipleTimesDoesntDeadlock) {
  constexpr struct xnn_fingerprint finger1{
      /*id=*/xnn_fingerprint_id_test_f16_f32_qc8w_nr2,
      /*value=*/314};
  constexpr struct xnn_fingerprint finger2{
      /*id=*/xnn_fingerprint_id_test_f16_f32_qc8w_nr2,
      /*value=*/654};
  static_assert(finger1.id == finger2.id,
                "This test checks that updating a fingerprint value doesn't "
                "deadlock. Do use different ids for the two fingerprints.");
  EXPECT_EQ(xnn_set_fingerprint(finger1), xnn_status_success);
  EXPECT_EQ(xnn_set_fingerprint(finger1), xnn_status_success);
  // Update the value of the fingerprint.
  EXPECT_EQ(xnn_set_fingerprint(finger2), xnn_status_success);
  EXPECT_EQ(xnn_set_fingerprint(finger2), xnn_status_success);

  xnn_get_fingerprint(finger1.id);
  xnn_get_fingerprint(finger2.id);
  xnn_get_fingerprint(finger1.id);
  xnn_get_fingerprint(finger2.id);
}

TEST_F(FingerprintCacheTest, InitializeAndFinalize) {
  struct fingerprint_context context =
      create_fingerprint_context(xnn_fingerprint_id_test_f16_f32_qc8w_nr2);
  EXPECT_THAT(context.status, Eq(xnn_status_uninitialized));
  EXPECT_THAT(context.fingerprint_id,
              Eq(xnn_fingerprint_id_test_f16_f32_qc8w_nr2));
  EXPECT_EQ(finalize_fingerprint_context(&context), xnn_status_success);
  EXPECT_THAT(xnn_get_fingerprint(xnn_fingerprint_id_test_f16_f32_qc8w_nr2),
              NotNull());
}

TEST_F(FingerprintCacheTest, ReserveAndWrite) {
  constexpr size_t kBufferSize = 8;
  struct fingerprint_context context =
      create_fingerprint_context(xnn_fingerprint_id_test_f16_f32_qc8w_nr2);
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
  EXPECT_EQ(finalize_fingerprint_context(&context), xnn_status_success);
  const xnn_fingerprint* fingerprint =
      xnn_get_fingerprint(xnn_fingerprint_id_test_f16_f32_qc8w_nr2);
  ASSERT_THAT(fingerprint, NotNull());
  EXPECT_THAT(fingerprint->id, Eq(xnn_fingerprint_id_test_f16_f32_qc8w_nr2));
  EXPECT_THAT(fingerprint->value, Not(Eq(0)));
}
