// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

// Returns {x[i] for i in perm}
template <typename T>
std::vector<T> permute(const std::vector<int>& perm, const std::vector<T>& x) {
  std::vector<T> result(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    result[i] = x[perm[i]];
  }
  return result;
}

// Returns sum(a[i] * b[i])
size_t dot(const std::vector<size_t>& a, const std::vector<size_t>& b) {
  size_t result = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

// Computes the dense (no padding) strides for a shape, where the last dimension
// has stride 1.
std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
  std::vector<size_t> strides(shape.size());
  strides.back() = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

template <typename T>
void TestImpl(T, size_t rank) {
  using T_info = type_info<T>;
  constexpr size_t elem_count = T_info::element_count();

  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> basis_dist(1, 100);

  std::vector<int32_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);

  do {
    quantization_params quantization = random_quantization(type_of<T>(), rng);

    // Define subgraph
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), rank, 0, quantization)
        .AddOutput(type_of<T>(), rank, 1, quantization)
        .AddTranspose(perm, 0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    // We need an algorithm for generating data in a tensor that we can
    // transpose. The data we generate is the dot product of the coordinate and
    // this basis. To generate the expected transposed data, we can just
    // transpose the coordinate and compute the dot product with this basis.
    std::vector<size_t> basis(rank);
    basis[0] = 1;
    for (size_t i = 1; i < rank; ++i) {
      basis[i] = basis[i - 1] * basis_dist(rng);
    }

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, rank);
      // We need both the input and output dense dimension to be aligned to the
      // number of elements in an instance of T.
      input_shape.back() = align_up(input_shape.back(), elem_count);
      input_shape[perm.back()] = align_up(input_shape[perm.back()], elem_count);

      size_t flat_size =
          std::accumulate(input_shape.begin(), input_shape.end(),
                          static_cast<size_t>(1), std::multiplies<size_t>());

      Buffer<T> input(flat_size / elem_count);
      std::vector<size_t> input_strides = compute_strides(input_shape);
      for (const auto& i : EnumerateIndices(input_shape)) {
        size_t flat_index = dot(i, input_strides);
        T_info::set(input.data(), flat_index, dot(i, basis));
      }

      // Check reshaped shape is correct
      std::vector<size_t> output_shape = permute(perm, input_shape);
      runtime.ReshapeExternalTensor(input_shape, input.data(), 0)
          .ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(1), output_shape);

      // Run subgraph
      Buffer<T> output(flat_size / elem_count);
      runtime.SetupExternalTensor(output.data(), 1).InvokeRuntime();

      // Verify results.
      std::vector<size_t> output_strides = compute_strides(output_shape);
      for (const auto& i : EnumerateIndices(input_shape)) {
        // Store the value in an instance of T, to get any truncation/rounding
        // that would have happened.
        T expected;
        T_info::set(&expected, 0, dot(i, basis));
        size_t flat_index = dot(permute(perm, i), output_strides);
        ASSERT_EQ(T_info::get(&expected, 0),
                  T_info::get(output.data(), flat_index));
      }
    }
  } while (std::next_permutation(perm.begin(), perm.end()));
}

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int4:
      return std::forward<F>(f)(int4x2());
    case ynn_type_uint4:
      return std::forward<F>(f)(uint4x2());
    case ynn_type_int8:
      return std::forward<F>(f)(int8_t());
    case ynn_type_uint8:
      return std::forward<F>(f)(uint8_t());
    case ynn_type_int32:
      return std::forward<F>(f)(int32_t());
    case ynn_type_fp16:
      return std::forward<F>(f)(half());
    case ynn_type_bf16:
      return std::forward<F>(f)(bfloat16());
    case ynn_type_fp32:
      return std::forward<F>(f)(float());
    default:
      YNN_UNREACHABLE;
  }
}

class Transpose : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(Transpose, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchType(type, [&](auto type) { TestImpl(type, rank); });
}

// This operation should work for arbitrary rank and this upper bound should
// cover all code paths.
constexpr int max_rank_for_testing = 5;

INSTANTIATE_TEST_SUITE_P(
    Transpose, Transpose,
    testing::Combine(testing::Values(ynn_type_int4, ynn_type_uint4,
                                     ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, max_rank_for_testing)),
    test_param_to_string<Transpose::ParamType>);

}  // namespace ynn
