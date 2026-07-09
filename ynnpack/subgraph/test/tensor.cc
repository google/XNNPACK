// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/tensor.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;

inline subgraph_ptr create_subgraph(uint32_t external_value_ids,
                                    uint32_t flags) {
  ynn_subgraph_t subgraph = nullptr;
  ynn_status status = ynn_create_subgraph(external_value_ids, flags, &subgraph);
  if (status != ynn_status_success) {
    return subgraph_ptr(nullptr, ynn_delete_subgraph);
  }
  return subgraph_ptr(subgraph, ynn_delete_subgraph);
}

TEST(tensor, find_static_tensor) {
  subgraph_ptr subgraph = create_subgraph(0, 0);

  // Define an int8 scalar constant with value 0 (passed as int8 0).
  const int8_t zero_i8 = 0;
  uint32_t zero_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(&*subgraph, ynn_type_int8, 0, nullptr,
                              &zero_i8, YNN_VALUE_FLAG_COPY_DATA, &zero_id),
            ynn_status_success);

  // Define another int8 scalar constant with float value 48.0f using
  // YNN_VALUE_FLAG_COPY_DATA_FP32.
  // In IEEE 754 float, 48.0f is 0x42400000 (little-endian: 00 00 40 42).
  // The first byte of 48.0f is 0x00. If find_static_tensor compares raw
  // unconverted float bytes (0x00) against existing int8 zero (0x00) using
  // std::memcmp, these values are incorrectly deduplicated.
  const float val48_f32 = 48.0f;
  uint32_t val48_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      ynn_define_tensor(&*subgraph, ynn_type_int8, 0, nullptr,
                        &val48_f32, YNN_VALUE_FLAG_COPY_DATA_FP32, &val48_id),
      ynn_status_success);

  EXPECT_NE(val48_id, zero_id);

  // Verify that defining another int8 scalar with float 48.0f DOES correctly
  // deduplicate to val48_id once converted.
  uint32_t val48_dup_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(&*subgraph, ynn_type_int8, 0, nullptr,
                              &val48_f32, YNN_VALUE_FLAG_COPY_DATA_FP32,
                              &val48_dup_id),
            ynn_status_success);

  EXPECT_EQ(val48_dup_id, val48_id);
}

}  // namespace
}  // namespace ynn
