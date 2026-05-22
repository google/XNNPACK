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

}  // namespace ynn
