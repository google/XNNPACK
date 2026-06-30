// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <gmock/gmock.h>
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

// Limit rank of tensors for testing, we have no special case codepaths beyond
// rank 2, so this should be plenty of coverage.
constexpr int max_test_rank = 5;

template <typename T>
void test_transpose_2d(T, size_t m, size_t n) {
  // Define subgraph
  SubgraphBuilder subgraph(2);
  subgraph.AddInput(type_of<T>(), 2, 0)
      .AddOutput(type_of<T>(), 2, 1)
      .AddTranspose({1, 0}, 0, 1);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::vector<size_t> input_shape = {m, n};

  Buffer<T> input(m * n);
  for (size_t i = 0; i < m * n; ++i) {
    input[i] = i;
  }

  std::vector<size_t> output_shape = {n, m};
  runtime.ReshapeExternalTensor(input_shape, input.data(), 0).ReshapeRuntime();
  ASSERT_EQ(runtime.GetExternalTensorShape(1), output_shape);

  // Run subgraph
  Buffer<T> output(m * n);
  runtime.SetupExternalTensor(output.data(), 1).InvokeRuntime();

  // Verify results.
  Buffer<T> expected(m * n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      expected[j * m + i] = i * n + j;
    }
  }
  EXPECT_THAT(output, testing::ElementsAreArray(expected));
}

template <typename T>
void test_slice(T, const std::vector<size_t>& input_shape, int dim) {
  // Define subgraph
  SubgraphBuilder subgraph(2);
  subgraph.AddInput(type_of<T>(), input_shape.size(), 0)
      .AddOutput(type_of<T>(), 1, 1)
      .AddTranspose({dim}, 0, 1);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  size_t input_size =
      std::accumulate(input_shape.begin(), input_shape.end(),
                      static_cast<size_t>(1), std::multiplies<size_t>());
  Buffer<T> input(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input[i] = i;
  }

  std::vector<size_t> output_shape = {input_shape[dim]};
  runtime.ReshapeExternalTensor(input_shape, input.data(), 0).ReshapeRuntime();
  ASSERT_EQ(runtime.GetExternalTensorShape(1), output_shape);

  // Run subgraph
  Buffer<T> output(input_shape[dim]);
  runtime.SetupExternalTensor(output.data(), 1).InvokeRuntime();

  // Verify results.
  size_t stride =
      std::accumulate(input_shape.begin() + dim + 1, input_shape.end(),
                      static_cast<size_t>(1), std::multiplies<size_t>());
  Buffer<T> expected(input_shape[dim]);
  for (size_t i = 0; i < input_shape[dim]; ++i) {
    expected[i] = i * stride;
  }
  EXPECT_THAT(output, testing::ElementsAreArray(expected));
}

// Returns {x[i] for i in perm}
template <typename T>
std::vector<T> permute(const std::vector<int>& perm, const std::vector<T>& x,
                       T default_value) {
  std::vector<T> result(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] >= 0 && static_cast<size_t>(perm[i]) < x.size()) {
      result[i] = x[perm[i]];
    } else {
      result[i] = default_value;
    }
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
  if (shape.empty()) {
    return {};
  }
  std::vector<size_t> strides(shape.size());
  strides.back() = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Align both the input and output trailing dimensions of a transpose to be a
// multiple of `align`.
void align_shape(std::vector<size_t>& shape, const std::vector<int32_t>& perm,
                 size_t align) {
  if (!shape.empty()) {
    shape.back() = align_up(shape.back(), align);
  }
  if (!perm.empty() && static_cast<size_t>(perm.back()) < shape.size()) {
    shape[perm.back()] = align_up(shape[perm.back()], align);
  }
};

template <typename T>
void test_random(T, bool with_copy) {
  using T_info = type_info<T>;
  constexpr size_t elem_count = T_info::element_count();

  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> basis_dist(1, 100);
  std::uniform_int_distribution<int> input_rank_dist(0, max_test_rank);

  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    const size_t input_rank = input_rank_dist(rng);
    std::uniform_int_distribution<int> output_rank_dist(0, input_rank);
    // Generate a random permutation that has some new dimensions in it.
    // This avoids generating permutations that use the same input dimension
    // more than once. This seems like something that maybe should work, but it
    // doesn't currently.
    std::vector<int32_t> perm(input_rank * 2 + 1);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    perm.resize(output_rank_dist(rng));
    while (elem_count > 1 && !perm.empty() && perm.back() >= input_rank) {
      // Don't make a new trailing dimension if the type is not byte aligned.
      perm.pop_back();
    }

    // Define subgraph
    std::vector<size_t> input_template_shape =
        random_shape(rng, input_rank, 0, 9);
    align_shape(input_template_shape, perm, elem_count);
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), input_template_shape, 0)
        .AddOutput(type_of<T>(), perm.size(), 1);

    if (with_copy) {
      // This variation allows the transpose to alias because the output is not
      // an external output.
      uint32_t transpose_id = YNN_INVALID_VALUE_ID;
      subgraph.AddTensor(type_of<T>(), perm.size(), transpose_id);
      subgraph.AddTranspose(perm, 0, transpose_id).AddCopy(transpose_id, 1);
    } else {
      subgraph.AddTranspose(perm, 0, 1);
    }

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    // We need an algorithm for generating data in a tensor that we can
    // transpose. The data we generate is the dot product of the coordinate and
    // this basis. To generate the expected transposed data, we can just
    // transpose the coordinate and compute the dot product with this basis.
    std::vector<size_t> basis(input_rank);
    if (!basis.empty()) {
      basis[0] = 1;
    }
    for (size_t i = 1; i < input_rank; ++i) {
      basis[i] = basis[i - 1] * basis_dist(rng);
    }

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, input_template_shape);
      align_shape(input_shape, perm, elem_count);

      size_t flat_size_in =
          std::accumulate(input_shape.begin(), input_shape.end(),
                          static_cast<size_t>(1), std::multiplies<size_t>());

      Buffer<T> input(flat_size_in);
      std::vector<size_t> input_strides = compute_strides(input_shape);
      for (const auto& i : EnumerateIndices(input_shape)) {
        size_t flat_index = dot(i, input_strides);
        input[flat_index] = dot(i, basis);
      }

      // Check reshaped shape is correct
      std::vector<size_t> output_shape =
          permute(perm, input_shape, static_cast<size_t>(1));
      runtime.ReshapeExternalTensor(input_shape, input.data(), 0)
          .ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(1), output_shape);

      // Run subgraph
      size_t flat_size_out =
          std::accumulate(output_shape.begin(), output_shape.end(),
                          static_cast<size_t>(1), std::multiplies<size_t>());
      Buffer<T> output(flat_size_out);
      runtime.SetupExternalTensor(output.data(), 1).InvokeRuntime();

      // Verify results.
      std::vector<size_t> output_strides = compute_strides(output_shape);
      for (const auto& i_out : EnumerateIndices(output_shape)) {
        std::vector<size_t> i_in(input_rank, 0);
        for (int d = static_cast<int>(perm.size()) - 1; d >= 0; --d) {
          if (perm[d] >= 0 && static_cast<size_t>(perm[d]) < input_rank) {
            i_in[perm[d]] = i_out[d];
          }
        }
        // Store the value in an instance of T, to get any truncation/rounding
        // that would have happened.
        T expected;
        T_info::set(&expected, 0, dot(i_in, basis));
        size_t flat_index = dot(i_out, output_strides);
        ASSERT_EQ(output[flat_index], T_info::get(&expected, 0));
      }
    }
  }
}

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int2:
      return std::forward<F>(f)(int2x4());
    case ynn_type_uint2:
      return std::forward<F>(f)(uint2x4());
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

class Transpose : public ::testing::TestWithParam<ynn_type> {};

TEST_P(Transpose, transpose_2d) {
  SwitchType(GetParam(), [&](auto type) { test_transpose_2d(type, 8, 4); });
}

TEST_P(Transpose, slice_0) {
  SwitchType(GetParam(), [&](auto type) { test_slice(type, {8, 4, 16}, 0); });
}

TEST_P(Transpose, slice_1) {
  SwitchType(GetParam(), [&](auto type) { test_slice(type, {8, 4, 16}, 1); });
}

TEST_P(Transpose, slice_2) {
  SwitchType(GetParam(), [&](auto type) { test_slice(type, {8, 4, 16}, 2); });
}

TEST_P(Transpose, random) {
  SwitchType(GetParam(),
             [&](auto type) { test_random(type, /*with_copy=*/false); });
}

TEST_P(Transpose, random_with_copy) {
  SwitchType(GetParam(),
             [&](auto type) { test_random(type, /*with_copy=*/true); });
}

INSTANTIATE_TEST_SUITE_P(
    Transpose, Transpose,
    testing::Values(ynn_type_int2, ynn_type_uint2, ynn_type_int4,
                    ynn_type_uint4, ynn_type_int8, ynn_type_uint8,
                    ynn_type_fp16, ynn_type_bf16, ynn_type_fp32),
    [](const testing::TestParamInfo<Transpose::ParamType>& info) {
      return to_string(info.param);
    });

}  // namespace ynn
