// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-utils.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

static const float kMaxAbsInput = 10.0f;
static const float kRMSNormEpsilon = 1.0e-6f;

namespace xnnpack {

template <typename T>
Tensor<T> normalize(enum xnn_norm_type norm_type, Tensor<T> x,
                    Tensor<T> scale = {}) {
  Tensor<T> y(x.extents());
  std::vector<size_t> batch_dims = x.extents();
  size_t channels = x.extents().back();
  batch_dims.pop_back();
  for (std::vector<size_t> i : EnumerateIndices(batch_dims)) {
    i.push_back(0);
    const T* x_i = &x(i);
    T* y_i = &y(i);
    double sum_of_squares = 0.0;
    for (size_t c = 0; c < channels; c++) {
      const double x_i_c = x_i[c];
      sum_of_squares += x_i_c * x_i_c;
    }
    sum_of_squares = std::max(sum_of_squares, 0.0);
    double rms_scale;
    switch (norm_type) {
      case xnn_norm_l2:
        rms_scale = 1.0 / std::sqrt(kRMSNormEpsilon + sum_of_squares);
        break;
      case xnn_norm_rms:
        rms_scale =
            1.0 / std::sqrt(kRMSNormEpsilon + sum_of_squares / channels);
        break;
      default:
        XNN_UNREACHABLE;
    }
    for (size_t c = 0; c < channels; c++) {
      y_i[c] = x_i[c] * rms_scale *
               (scale.empty() ? 1.0f : static_cast<double>(scale[c]));
    }
  }
  return y;
}

template <typename T>
void TestImpl(size_t rank, bool use_scale, enum xnn_norm_type norm_type) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  // Define subgraph
  SubgraphTester subgraph(use_scale ? 3 : 2);
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t scale_id = use_scale ? 2 : XNN_INVALID_VALUE_ID;
  subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), input_id)
      .AddOutputTensor(rank, xnn_datatype_of<T>(), output_id);
  if (use_scale) {
    subgraph.AddInputTensor(1, xnn_datatype_of<T>(), scale_id);
  }
  subgraph.AddRMSNorm(input_id, scale_id, output_id, norm_type,
                      kRMSNormEpsilon);
  xnn_status status = subgraph.CreateRuntime();
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
    return;
  }

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    std::vector<size_t> shape = random_shape(rng, rank);

    // Generate the input.
    Tensor<T> input(shape, xnnpack::XnnExtraBytes);
    DatatypeGenerator<T> generator(-kMaxAbsInput, kMaxAbsInput);
    input.generate([&]() { return generator(rng); });

    // Generate and populate the scale Tensor, if requested.
    Tensor<T> scale;
    if (use_scale) {
      scale = Tensor<T>({shape.back()}, xnnpack::XnnExtraBytes);
      DatatypeGenerator<T> scale_generator(-1.0f, 1.0f);
      scale.generate([&]() { return scale_generator(rng); });
      subgraph.ReshapeExternalTensor(scale.shape(), scale.base(), scale_id);
    }

    Tensor<T> expected = normalize(norm_type, input, scale);

    // Check reshaped shape is correct
    subgraph.ReshapeExternalTensor(shape, input.base(), input_id)
        .ReshapeRuntime();
    ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), expected.extents());

    // Run subgraph
    // RMSNorm reads from the output assuming XNN_EXTRA_BYTES exist.
    Tensor<T> output(expected.extents(), xnnpack::XnnExtraBytes);
    subgraph.SetupExternalTensor(output.base(), output_id)
        .SetupRuntime()
        .InvokeRuntime();

    // Verify results.
    const float tolerance = NumericLimits<T>::epsilon() * kMaxAbsInput *
                            kMaxAbsInput * shape.back() * 2.0;
    ASSERT_THAT(output.template cast<float>(),
                testing::Pointwise(testing::NanSensitiveFloatNear(tolerance),
                                   expected.template cast<float>()));
  }
}

template <typename T>
void TestSubgraphRewrite(size_t rank, bool use_scale,
                         enum xnn_norm_type norm_type) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    // Define subgraph
    SubgraphTester subgraph(2);
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t scale_id = XNN_INVALID_VALUE_ID;
    std::vector<size_t> shape = random_shape(rng, rank);

    // Generate the input.
    Tensor<T> input(shape, xnnpack::XnnExtraBytes);
    DatatypeGenerator<T> generator(-kMaxAbsInput, kMaxAbsInput);
    input.generate([&]() { return generator(rng); });

    subgraph.AddInputTensor(shape, xnn_datatype_of<T>(), input_id);
    subgraph.AddOutputTensor(shape, xnn_datatype_of<T>(), output_id);

    // Generate and populate the scale Tensor, if requested.
    Tensor<T> scale;
    if (use_scale) {
      std::vector<size_t> scale_shape = shape;
      std::fill(scale_shape.begin(), scale_shape.end() - 1, 1);
      scale = Tensor<T>(scale_shape, xnnpack::XnnExtraBytes);
      DatatypeGenerator<T> scale_generator(-1.0f, 1.0f);
      scale.generate([&]() { return scale_generator(rng); });
      subgraph.AddInternalStaticTensor(scale_shape, xnn_datatype_of<T>(),
                                       &scale_id, scale.data());
    }

    // Generate the RMS/L2-Norm nodes, randomly swapping inputs where
    // associative.

    // b = mul(a, a).
    uint32_t squared_id = XNN_INVALID_VALUE_ID;
    subgraph.AddInternalDynamicTensor(shape, xnn_datatype_of<T>(), &squared_id,
                                      /*flags=*/0);
    subgraph.AddMultiply(input_id, input_id, squared_id);

    // c = reduce_sum(b, axis=-1).
    uint32_t sum_of_squares_id = XNN_INVALID_VALUE_ID;
    std::vector<size_t> reduced_shape = shape;
    reduced_shape.back() = 1;
    subgraph.AddInternalDynamicTensor(reduced_shape, xnn_datatype_of<T>(),
                                      &sum_of_squares_id,
                                      /*flags=*/0);
    subgraph.AddReduce(xnn_reduce_sum, {static_cast<int64_t>(rank) - 1},
                       squared_id, sum_of_squares_id, XNN_FLAG_KEEP_DIMS);

    uint32_t scaled_sum_id = sum_of_squares_id;
    const T inv_n = 1.0 / shape.back();
    switch (norm_type) {
      case xnn_norm_rms: {
        // d = mul(c, inv_n).
        uint32_t inv_n_id = XNN_INVALID_VALUE_ID;
        subgraph.AddInternalDynamicTensor(reduced_shape, xnn_datatype_of<T>(),
                                          &scaled_sum_id,
                                          /*flags=*/0);
        subgraph.AddInternalStaticTensor(/*shape=*/{1}, xnn_datatype_of<T>(),
                                         &inv_n_id, &inv_n);
        if (rng() % 2) {
          subgraph.AddMultiply(sum_of_squares_id, inv_n_id, scaled_sum_id);
        } else {
          subgraph.AddMultiply(inv_n_id, sum_of_squares_id, scaled_sum_id);
        }
      } break;
      case xnn_norm_l2:
        break;
      default:
        XNN_UNREACHABLE;
    }

    // Optionally e = add(d, eps).
    uint32_t shifted_scaled_sum_id = scaled_sum_id;
    T epsilon = 0.0;
    if (rng() % 2) {
      uint32_t epsilon_id = XNN_INVALID_VALUE_ID;
      epsilon = kRMSNormEpsilon;
      subgraph.AddInternalDynamicTensor(reduced_shape, xnn_datatype_of<T>(),
                                        &shifted_scaled_sum_id,
                                        /*flags=*/0);
      subgraph.AddInternalStaticTensor(/*shape=*/{1}, xnn_datatype_of<T>(),
                                       &epsilon_id, &epsilon);
      if (rng() % 2) {
        subgraph.AddAddition(scaled_sum_id, epsilon_id, shifted_scaled_sum_id);
      } else {
        subgraph.AddAddition(epsilon_id, scaled_sum_id, shifted_scaled_sum_id);
      }
    }

    // f = sqrt(shifted_rms_id).
    uint32_t sqrt_shifted_scaled_sum_id = XNN_INVALID_VALUE_ID;
    subgraph.AddInternalDynamicTensor(reduced_shape, xnn_datatype_of<T>(),
                                      &sqrt_shifted_scaled_sum_id,
                                      /*flags=*/0);
    subgraph.AddUnary(xnn_unary_square_root, /*params=*/nullptr,
                      shifted_scaled_sum_id, sqrt_shifted_scaled_sum_id);

    // g = div(a, f).
    if (use_scale) {
      uint32_t normalized_id = XNN_INVALID_VALUE_ID;
      subgraph.AddInternalDynamicTensor(shape, xnn_datatype_of<T>(),
                                        &normalized_id,
                                        /*flags=*/0);
      subgraph.AddDivide(input_id, sqrt_shifted_scaled_sum_id, normalized_id);
      if (rng() % 2) {
        subgraph.AddMultiply(normalized_id, scale_id, output_id);
      } else {
        subgraph.AddMultiply(scale_id, normalized_id, output_id);
      }
    } else {
      subgraph.AddBinary(xnn_binary_divide, /*params=*/nullptr, input_id,
                         sqrt_shifted_scaled_sum_id, output_id);
    }

    // Set up the input/output tensors.
    Tensor<T> expected(shape, xnnpack::XnnExtraBytes);
    subgraph.SetupExternalTensor(input.base(), input_id);
    subgraph.SetupExternalTensor(expected.base(), output_id);

    // Evaluate once with `XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC` enabled to
    // prevent the subgraph replacement.
    xnn_status status = subgraph.CreateRuntime(
        /*threadpool=*/nullptr, XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC);
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }
    ASSERT_GT(subgraph.NumNodes(), 1);

    // Run the subgraph.
    subgraph.ReshapeRuntime();
    subgraph.SetupRuntime();
    subgraph.InvokeRuntime();

    // Create the runtime and evaluate again and check that the subgraph was
    // replaced.
    subgraph.Optimize();
    ASSERT_EQ(subgraph.NumNodes(), 1);
    ASSERT_EQ(subgraph.Node(0)->type, xnn_node_type_normalize);

    // Run the subgraph.
    Tensor<T> output(shape, xnnpack::XnnExtraBytes);
    subgraph.SetupExternalTensor(output.base(), output_id);

    // Run the subgraph.
    subgraph.ReshapeRuntime();
    subgraph.SetupRuntime();
    subgraph.InvokeRuntime();

    // Verify results.
    const float tolerance = NumericLimits<T>::epsilon() * kMaxAbsInput *
                            kMaxAbsInput * shape.back();
    ASSERT_THAT(output.template cast<float>(),
                testing::Pointwise(testing::NanSensitiveFloatNear(tolerance),
                                   expected.template cast<float>()));
  }
}

template <typename T>
class NormalizeTest : public ::testing::TestWithParam<
                          std::tuple<size_t, bool, enum xnn_norm_type>> {};
std::string NormalizeTestName(
    const testing::TestParamInfo<std::tuple<size_t, bool, enum xnn_norm_type>>&
        info) {
  auto& params = info.param;
  char buff[100];
  sprintf(buff, "%s_%zu%s", xnn_norm_type_to_string(std::get<2>(params)),
          std::get<0>(params), std::get<1>(params) ? "_scaling" : "");
  return std::string(buff);
}
using NormalizeTestF16 = NormalizeTest<xnn_float16>;
using NormalizeTestF32 = NormalizeTest<float>;

TEST_P(NormalizeTestF16, test) {
  TestImpl<xnn_float16>(/*rank=*/std::get<0>(GetParam()),
                        /*use_scale=*/std::get<1>(GetParam()),
                        /*norm_type=*/std::get<2>(GetParam()));
}
TEST_P(NormalizeTestF32, test) {
  TestImpl<float>(/*rank=*/std::get<0>(GetParam()),
                  /*use_scale=*/std::get<1>(GetParam()),
                  /*norm_type=*/std::get<2>(GetParam()));
}
TEST_P(NormalizeTestF16, subgraph_rewrite) {
  TestSubgraphRewrite<xnn_float16>(/*rank=*/std::get<0>(GetParam()),
                                   /*use_scale=*/std::get<1>(GetParam()),
                                   /*norm_type=*/std::get<2>(GetParam()));
}
TEST_P(NormalizeTestF32, subgraph_rewrite) {
  TestSubgraphRewrite<float>(/*rank=*/std::get<0>(GetParam()),
                             /*use_scale=*/std::get<1>(GetParam()),
                             /*norm_type=*/std::get<2>(GetParam()));
}

auto test_params = testing::Combine(
    testing::Range<size_t>(1, XNN_MAX_TENSOR_DIMS), testing::Bool(),
    testing::Values(xnn_norm_l2, xnn_norm_rms));
INSTANTIATE_TEST_SUITE_P(RMSNorm, NormalizeTestF16, test_params,
                         NormalizeTestName);
INSTANTIATE_TEST_SUITE_P(RMSNorm, NormalizeTestF32, test_params,
                         NormalizeTestName);

}  // namespace xnnpack
