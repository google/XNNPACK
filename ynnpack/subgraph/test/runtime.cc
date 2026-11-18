// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {
// By default WASM doesn't have any thread support.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
TEST(runtime, dot_concurrency) {}
#else
TEST(runtime, dot_concurrency) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t c_id = 2;
  constexpr uint32_t init_zero = YNN_INVALID_VALUE_ID;

  TestScheduler scheduler(3);

  auto get_concurrency = [&](SubgraphBuilder& builder) -> int32_t {
    Runtime runtime(builder.GetSubgraph(), &scheduler);
    EXPECT_EQ(runtime.Status(), ynn_status_success);
    int32_t concurrency;
    EXPECT_EQ(runtime.Query(ynn_runtime_property_concurrency, &concurrency),
              ynn_status_success);
    return concurrency;
  };

  // We should be able to statically know this graph will not run a parallel
  // loop.
  SubgraphBuilder small(3);
  small.AddInput(type_of<float>(), {8, 8}, a_id);
  small.AddInput(type_of<float>(), {8, 8}, b_id);
  small.AddOutput(type_of<float>(), {8, 8}, c_id);
  small.AddDot(1, a_id, b_id, init_zero, c_id);
  // TODO(b/458542243): This doesn't actually work because we don't simplify
  // away these loops yet.
  // ASSERT_EQ(get_concurrency(small), 1);

  // We should be able to statically know this graph will run parallel loops.
  SubgraphBuilder big(3);
  big.AddInput(type_of<float>(), {800, 800}, a_id);
  big.AddInput(type_of<float>(), {800, 800}, b_id);
  big.AddOutput(type_of<float>(), {800, 800}, c_id);
  big.AddDot(1, a_id, b_id, init_zero, c_id);
  ASSERT_GT(get_concurrency(big), 1);

  // We don't know in this case, we might run a parallel loop if the input is
  // big enough.
  SubgraphBuilder dynamic(3);
  dynamic.AddInput(type_of<float>(), 2, a_id);
  dynamic.AddInput(type_of<float>(), 2, b_id);
  dynamic.AddOutput(type_of<float>(), 2, c_id);
  dynamic.AddDot(1, a_id, b_id, init_zero, c_id);
  ASSERT_GT(get_concurrency(dynamic), 1);
}
#endif

TEST(runtime, cross_runtime_isolation) {
  constexpr uint32_t in_id = 0;
  constexpr uint32_t out_id = 1;
  SubgraphBuilder builder(2);
  builder.AddInput(type_of<float>(), 1, in_id)
      .AddOutput(type_of<float>(), 1, out_id)
      .AddUnary(ynn_unary_negate, in_id, out_id);

  Runtime A(builder.GetSubgraph());
  Runtime B(builder.GetSubgraph());
  ASSERT_EQ(A.Status(), ynn_status_success);
  ASSERT_EQ(B.Status(), ynn_status_success);

  std::vector<float> in_a(100, 1.0f);
  std::vector<float> in_b(10, 2.0f);
  std::vector<float> out_a(100, 0.0f);
  std::vector<float> out_b(10, 0.0f);

  A.ReshapeExternalTensor(TensorShape({100}), in_a.data(), in_id);
  A.ReshapeRuntime();

  // Runtime B's output shape must not be affected by Runtime A's reshape.
  EXPECT_NE(B.GetExternalTensorShape(out_id), std::vector<size_t>{100});

  B.ReshapeExternalTensor(TensorShape({10}), in_b.data(), in_id);
  B.ReshapeRuntime();

  EXPECT_EQ(A.GetExternalTensorShape(out_id), std::vector<size_t>{100});
  EXPECT_EQ(B.GetExternalTensorShape(out_id), std::vector<size_t>{10});

  A.SetupExternalTensor(out_a.data(), out_id).InvokeRuntime();
  B.SetupExternalTensor(out_b.data(), out_id).InvokeRuntime();

  ASSERT_EQ(A.Status(), ynn_status_success);
  ASSERT_EQ(B.Status(), ynn_status_success);

  for (float v : out_a) {
    EXPECT_FLOAT_EQ(v, -1.0f);
  }
  for (float v : out_b) {
    EXPECT_FLOAT_EQ(v, -2.0f);
  }
}

namespace {

std::vector<size_t> GetStrides(ynn_runtime_t runtime, uint32_t id) {
  size_t rank = YNN_MAX_TENSOR_RANK;
  std::vector<size_t> strides(rank);
  EXPECT_EQ(ynn_get_external_value_strides(runtime, id, &rank, strides.data()),
            ynn_status_success);
  strides.resize(rank);
  return strides;
}

}  // namespace

TEST(runtime, external_strides) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 2, a_id)
      .AddInput(type_of<float>(), 2, b_id)
      .AddOutput(type_of<float>(), 2, x_id)
      .AddBinary(ynn_binary_add, a_id, b_id, x_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // `a` and `x` are the first `cols` columns of arrays with wider rows.
  constexpr size_t rows = 5;
  constexpr size_t cols = 7;
  constexpr size_t a_cols = 10;
  constexpr size_t x_cols = 9;
  std::vector<float> a(rows * a_cols);
  std::vector<float> b(rows * cols);
  std::vector<float> x(rows * x_cols, -1.0f);
  std::iota(a.begin(), a.end(), 0.0f);
  std::iota(b.begin(), b.end(), 1000.0f);

  const size_t extents[] = {rows, cols};
  const size_t a_strides[] = {a_cols * sizeof(float), sizeof(float)};
  const size_t x_strides[] = {x_cols * sizeof(float), sizeof(float)};
  const size_t dense_strides[] = {cols * sizeof(float), sizeof(float)};

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, extents),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, extents),
            ynn_status_success);

  // The required strides are not known until the runtime is reshaped.
  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), a_id, a.data(), 2,
                                           a_strides),
            ynn_status_invalid_parameter);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  EXPECT_THAT(GetStrides(runtime.get(), a_id),
              testing::ElementsAreArray(dense_strides));
  EXPECT_THAT(GetStrides(runtime.get(), x_id),
              testing::ElementsAreArray(dense_strides));

  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), a_id, a.data(), 2,
                                           a_strides),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), b_id, b.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), x_id, x.data(), 2,
                                           x_strides),
            ynn_status_success);
  EXPECT_THAT(GetStrides(runtime.get(), a_id),
              testing::ElementsAreArray(a_strides));
  EXPECT_THAT(GetStrides(runtime.get(), x_id),
              testing::ElementsAreArray(x_strides));

  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < x_cols; ++j) {
      const float expected =
          j < cols ? a[i * a_cols + j] + b[i * cols + j] : -1.0f;
      ASSERT_EQ(x[i * x_cols + j], expected) << i << ", " << j;
    }
  }

  // Setting only the data does not change the strides.
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x.data()),
            ynn_status_success);
  EXPECT_THAT(GetStrides(runtime.get(), x_id),
              testing::ElementsAreArray(x_strides));

  // Reshaping resets the strides of outputs, but not inputs.
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  EXPECT_THAT(GetStrides(runtime.get(), a_id),
              testing::ElementsAreArray(a_strides));
  EXPECT_THAT(GetStrides(runtime.get(), x_id),
              testing::ElementsAreArray(dense_strides));

  // Setting the shape resets the strides of an input.
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, extents),
            ynn_status_success);
  EXPECT_THAT(GetStrides(runtime.get(), a_id),
              testing::ElementsAreArray(dense_strides));
}

TEST(runtime, external_strides_dot) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 2, a_id)
      .AddInput(type_of<float>(), 2, b_id)
      .AddOutput(type_of<float>(), 2, x_id)
      .AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, x_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  constexpr size_t m = 5;
  constexpr size_t k = 6;
  constexpr size_t n = 7;
  constexpr size_t a_cols = 8;
  constexpr size_t x_cols = 11;
  std::vector<float> a(m * a_cols);
  std::vector<float> b(k * n);
  std::vector<float> x(m * x_cols, -1.0f);
  std::iota(a.begin(), a.end(), 0.0f);
  std::iota(b.begin(), b.end(), 1.0f);

  const size_t a_extents[] = {m, k};
  const size_t b_extents[] = {k, n};
  const size_t a_strides[] = {a_cols * sizeof(float), sizeof(float)};
  const size_t x_strides[] = {x_cols * sizeof(float), sizeof(float)};

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_extents),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, b_extents),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);

  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), a_id, a.data(), 2,
                                           a_strides),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), b_id, b.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), x_id, x.data(), 2,
                                           x_strides),
            ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < x_cols; ++j) {
      float expected = -1.0f;
      if (j < n) {
        expected = 0.0f;
        for (size_t l = 0; l < k; ++l) {
          expected += a[i * a_cols + l] * b[l * n + j];
        }
      }
      ASSERT_EQ(x[i * x_cols + j], expected) << i << ", " << j;
    }
  }
}

TEST(runtime, external_strides_innermost_dimension) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  builder.AddInput(type_of<float>(), 2, a_id)
      .AddOutput(type_of<float>(), 2, x_id)
      .AddUnary(ynn_unary_negate, a_id, x_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const size_t extents[] = {5, 7};
  const size_t bad_strides[] = {7 * 2 * sizeof(float), 2 * sizeof(float)};
  const size_t dense_strides[] = {7 * sizeof(float), sizeof(float)};

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, extents),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);

  for (uint32_t id : {a_id, x_id}) {
    ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), id, nullptr, 2,
                                             bad_strides),
              ynn_status_invalid_parameter);
    EXPECT_THAT(GetStrides(runtime.get(), id),
                testing::ElementsAreArray(dense_strides));
    ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), id, nullptr, 2,
                                             dense_strides),
              ynn_status_success);
  }
}

TEST(runtime, external_strides_required_by_reshape) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  builder.AddInput(type_of<float>(), {4, 6}, a_id)
      .AddOutput(type_of<float>(), {3, 8}, x_id)
      .AddReshape({3, 8}, a_id, x_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);

  std::vector<float> a(4 * 6);
  std::vector<float> x(3 * 8);
  std::iota(a.begin(), a.end(), 0.0f);

  // Reshaping requires both the input and output to be dense.
  const size_t a_bad_strides[] = {8 * sizeof(float), sizeof(float)};
  const size_t a_strides[] = {6 * sizeof(float), sizeof(float)};
  const size_t x_bad_strides[] = {10 * sizeof(float), sizeof(float)};
  const size_t x_strides[] = {8 * sizeof(float), sizeof(float)};

  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), a_id, a.data(), 2,
                                           a_bad_strides),
            ynn_status_invalid_parameter);
  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), a_id, a.data(), 2,
                                           a_strides),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), x_id, x.data(), 2,
                                           x_bad_strides),
            ynn_status_invalid_parameter);
  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), x_id, x.data(), 2,
                                           x_strides),
            ynn_status_success);

  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);
  ASSERT_THAT(x, testing::ElementsAreArray(a));
}

TEST(runtime, reshape_skips_unchanged_shape) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t x_id = 2;
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 2, a_id)
      .AddInput(type_of<float>(), 2, b_id)
      .AddOutput(type_of<float>(), 2, x_id)
      .AddBinary(ynn_binary_add, a_id, b_id, x_id);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const size_t shape1[] = {3, 5};
  const size_t bad_shape[] = {3, 4};
  const size_t shape2[] = {2, 6};

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, shape1),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, bad_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_invalid_parameter);
  // Retrying reshape after a failure must still fail.
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_invalid_parameter);

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, shape1),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  EXPECT_EQ(runtime.GetExternalTensorShape(x_id), (std::vector<size_t>{3, 5}));

  // Setting an unchanged shape keeps the runtime reshaped, while still
  // resetting custom output strides on reshape.
  std::vector<float> a(3 * 5, 2.0f);
  std::vector<float> b(3 * 5, 3.0f);
  std::vector<float> x(3 * 7, 0.0f);
  const size_t x_strides[] = {7 * sizeof(float), sizeof(float)};
  const size_t dense_strides1[] = {5 * sizeof(float), sizeof(float)};
  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), x_id, x.data(), 2,
                                           x_strides),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, shape1),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, shape1),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  EXPECT_THAT(GetStrides(runtime.get(), x_id),
              testing::ElementsAreArray(dense_strides1));

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id, a.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), b_id, b.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);
  for (size_t i = 0; i < 15; ++i) {
    EXPECT_FLOAT_EQ(x[i], 5.0f);
  }

  // Changing a shape requires a reshape before setting custom strides and
  // updates the output shape.
  const size_t x_strides2[] = {8 * sizeof(float), sizeof(float)};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, shape2),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, shape2),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data_v2(runtime.get(), x_id, x.data(), 2,
                                           x_strides2),
            ynn_status_invalid_parameter);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  EXPECT_EQ(runtime.GetExternalTensorShape(x_id), (std::vector<size_t>{2, 6}));
}

}  // namespace ynn
