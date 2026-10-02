// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/base.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

TEST(SubByteTypesRegression, TiledConvert2Bit) {
  // 131072 * 2 = 262144 elements logical.
  // Cache split threshold is 131072 elements, so we should get loop splits.
  std::vector<size_t> shapes_a = {4, 262144};
  std::vector<size_t> shapes_c = {4, 262144};

  SubgraphBuilder subgraph(3, 0);

  uint32_t a_id = 0;
  uint32_t b_id = 1;
  uint32_t output_id = 2;

  subgraph.AddInput(type_of<ynn::int2x4>(), shapes_a, a_id)
      .AddTensor(type_of<int8_t>(), shapes_a, b_id)
      .AddOutput(type_of<int32_t>(), 2, output_id);

  subgraph.AddConvert(a_id, type_of<int8_t>(), b_id, 0)
      .AddConvert(b_id, type_of<int32_t>(), output_id, 0);

  // We explicitly use the multi-threaded scheduler to enforce loop
  // partitioning.
  TestScheduler scheduler(3);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::mt19937 rng(42);
  Tensor<ynn::int2x4> a(shapes_a);
  Tensor<int32_t> c(shapes_c);

  fill_random(a.data(), a.size(), rng, -2, 1);
  std::fill(c.begin(), c.end(), 0);

  runtime.SetupExternalTensor(a.data(), a_id)
      .SetupExternalTensor(c.data(), output_id)
      .ReshapeRuntime()
      .InvokeRuntime();

  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // Reference comparison
  using A_info = type_info<ynn::int2x4>;
  for (size_t i = 0; i < 4; ++i) {
    const ynn::int2x4* a_i = address_of(a(i, 0));
    for (size_t j = 0; j < 262144; ++j) {
      int32_t expected_val = static_cast<int32_t>(A_info::get(a_i, j));
      ASSERT_EQ(c(i, j), expected_val) << "Mismatch at i=" << i << ", j=" << j;
    }
  }
}

void TestInt4TransposeDequantize(bool kn_params) {
  constexpr size_t kRows = 4;
  constexpr size_t kCols = 4;

  auto value_fn = [](size_t row, size_t col) -> int8_t {
    return static_cast<int8_t>(row * kCols + col) - 8;
  };

  // Exactly the packed size, so that ASan catches reads past its end.
  std::vector<uint8_t> packed(kRows * kCols / 2, 0);
  for (size_t i = 0; i < kRows * kCols; ++i) {
    const uint8_t nibble = static_cast<uint8_t>(value_fn(i / kCols, i % kCols));
    packed[i / 2] |= (nibble & 0xF) << (4 * (i % 2));
  }
  const std::vector<float> scales(kn_params ? kCols * kRows : 1, 1.0f);
  const std::vector<int32_t> zero_points(kCols * kRows, 0);

  ynn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(
      ynn_create_subgraph(/*external_value_ids=*/1, /*flags=*/0, &subgraph),
      ynn_status_success);
  uint32_t out_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_fp32, 2, nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id),
            ynn_status_success);

  const size_t dims[2] = {kRows, kCols};
  uint32_t int4_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_int4, 2, dims, packed.data(),
                              /*flags=*/0, &int4_id),
            ynn_status_success);
  const int32_t perm[2] = {1, 0};
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_static_transpose(subgraph, 2, perm, int4_id,
                                        &transposed_id, /*flags=*/0),
            ynn_status_success);

  const size_t param_dims[2] = {kCols, kRows};
  const size_t param_rank = kn_params ? 2 : 0;
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_fp32, param_rank, param_dims,
                              scales.data(), YNN_VALUE_FLAG_COPY_DATA,
                              &scale_id),
            ynn_status_success);
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  if (kn_params) {
    ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_int32, param_rank,
                                param_dims, zero_points.data(),
                                YNN_VALUE_FLAG_COPY_DATA, &zero_point_id),
              ynn_status_success);
  }
  ASSERT_EQ(ynn_define_dequantize(subgraph, transposed_id, zero_point_id,
                                  scale_id, ynn_type_fp32, &out_id,
                                  /*flags=*/0),
            ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph, /*threadpool=*/nullptr, 0),
            ynn_status_success);
  ynn_runtime_t runtime = nullptr;
  ASSERT_EQ(ynn_create_runtime(subgraph, /*threadpool=*/nullptr, 0, &runtime),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime), ynn_status_success);
  std::vector<float> out(kCols * kRows, -100.0f);
  ASSERT_EQ(ynn_set_external_value_data(runtime, out_id, out.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime), ynn_status_success);
  ynn_delete_runtime(runtime);
  ynn_delete_subgraph(subgraph);

  for (size_t col = 0; col < kCols; ++col) {
    for (size_t row = 0; row < kRows; ++row) {
      EXPECT_EQ(out[col * kRows + row],
                static_cast<float>(value_fn(row, col)));
    }
  }
}

TEST(SubByteTypesRegression, Int4TransposeDequantizeScalarParams) {
  TestInt4TransposeDequantize(/*kn_params=*/false);
}

TEST(SubByteTypesRegression, Int4TransposeDequantizeTensorParams) {
  TestInt4TransposeDequantize(/*kn_params=*/true);
}

}  // namespace ynn
