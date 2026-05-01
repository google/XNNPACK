// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/base.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

template <typename Rng>
ynn_type random_type(Rng& rng) {
  ynn_type types[] = {
      ynn_type_int2, ynn_type_int4, ynn_type_int8,
      ynn_type_bf16, ynn_type_fp32,  ynn_type_int32,
  };
  std::uniform_int_distribution<> type_dist(0, std::size(types) - 1);
  return types[type_dist(rng)];
}

template <typename T>
void TestImpl(T, size_t rank, bool output_rank, bool reshape_1d) {
  ReplicableRandomDevice rng;

  for (size_t num_axes = 0; num_axes <= rank; ++num_axes) {
    if (output_rank == 0 && !reshape_1d && num_axes != 1) {
      // We can only get a scalar shape if the number of axes is 1.
      continue;
    }

    std::vector<int32_t> all_axes(rank);
    std::iota(all_axes.begin(), all_axes.end(), 0);

    do {
      std::vector<int32_t> axes(all_axes.begin(), all_axes.begin() + num_axes);

      ynn_type in_type = random_type(rng);

      // Define subgraph
      SubgraphBuilder subgraph(2);
      subgraph.AddInput(in_type, rank, 0)
          .AddOutput(type_of<T>(), output_rank, 1)
          .AddGetTensorShape(axes, type_of<T>(), output_rank, 0, 1,
                             reshape_1d ? YNN_NODE_FLAG_RESHAPE_1D : 0);

      Runtime runtime(subgraph.GetSubgraph());
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      for (int reshape = 0; reshape < 2; ++reshape) {
        std::vector<size_t> shape = random_shape(rng, rank);
        // Ensure the innermost dimension is aligned to the element count for
        // sub-byte packed types like int4 and int2.
        int elem_count = type_element_count(in_type);
        if (elem_count > 1 && !shape.empty()) {
          shape.back() =
              std::max<size_t>(1, shape.back() / elem_count) * elem_count;
        }

        std::vector<size_t> expected_shape;
        if (output_rank != 0) {
          expected_shape = {reshape_1d ? 1 : axes.size()};
        }

        // Check reshaped shape is correct
        runtime.ReshapeExternalTensor(shape, nullptr, 0).ReshapeRuntime();
        ASSERT_EQ(runtime.GetExternalTensorShape(1), expected_shape);

        // Run subgraph
        Tensor<T> output(expected_shape);
        runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

        // Verify results.
        if (reshape_1d) {
          size_t expected_extent = 1;
          for (size_t i = 0; i < axes.size(); ++i) {
            expected_extent *= shape[axes[i]];
          }
          ASSERT_EQ(output[0], expected_extent);
        } else {
          for (size_t i = 0; i < axes.size(); ++i) {
            ASSERT_EQ(output[i], shape[axes[i]]);
          }
        }
      }
    } while (std::next_permutation(all_axes.begin(), all_axes.end()));
  }
}

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_fp32:
      return std::forward<F>(f)(float());
    case ynn_type_int32:
      return std::forward<F>(f)(int32_t());
    default:
      YNN_UNREACHABLE;
  }
}

class GetTensorShape
    : public testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(GetTensorShape, Scalar) {
  SwitchType(std::get<0>(GetParam()), [&](auto a_type) {
    TestImpl(a_type, std::get<1>(GetParam()), /*output_rank=*/0,
             /*reshape_1d=*/false);
  });
}

TEST_P(GetTensorShape, ScalarReshape1D) {
  SwitchType(std::get<0>(GetParam()), [&](auto a_type) {
    TestImpl(a_type, std::get<1>(GetParam()), /*output_rank=*/0,
             /*reshape_1d=*/true);
  });
}

TEST_P(GetTensorShape, Vector) {
  SwitchType(std::get<0>(GetParam()), [&](auto a_type) {
    TestImpl(a_type, std::get<1>(GetParam()), /*output_rank=*/1,
             /*reshape_1d=*/false);
  });
}

TEST_P(GetTensorShape, VectorReshape1D) {
  SwitchType(std::get<0>(GetParam()), [&](auto a_type) {
    TestImpl(a_type, std::get<1>(GetParam()), /*output_rank=*/1,
             /*reshape_1d=*/true);
  });
}

// This operation should work for arbitrary rank and this upper bound should
// cover all code paths.
constexpr int max_rank_for_testing = 4;

INSTANTIATE_TEST_SUITE_P(
    Test, GetTensorShape,
    testing::Combine(testing::Values(ynn_type_fp32, ynn_type_int32),
                     testing::Range(1, max_rank_for_testing)),
    [](const testing::TestParamInfo<GetTensorShape::ParamType>& info) {
      return test_param_to_string(info);
    });

}  // namespace ynn
