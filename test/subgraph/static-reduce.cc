// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/runtime-flags.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

struct Param {
  using TupleT = std::tuple<xnn_reduce_operator, bool, bool, int>;
  explicit Param(TupleT p)
      : reduce_operator(std::get<0>(p)),
        keep_dims(std::get<1>(p)),
        use_neg_axes(std::get<2>(p)),
        rank(std::get<3>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    switch (reduce_operator) {
      case xnn_reduce_mean:
        sstr << "mean";
        break;
      case xnn_reduce_mean_squared:
        sstr << "mean_squared";
        break;
      case xnn_reduce_sum:
        sstr << "sum";
        break;
      case xnn_reduce_sum_squared:
        sstr << "sum_squared";
        break;
      case xnn_reduce_max:
        sstr << "max";
        break;
      case xnn_reduce_min:
        sstr << "min";
        break;
      case xnn_reduce_invalid:
        sstr << "invalid";
        break;
    }
    if (keep_dims) {
      sstr << "_keep_dims";
    }
    if (use_neg_axes) {
      sstr << "_use_neg_axes";
    }
    sstr << "_" << rank;
    return sstr.str();
  }

  xnn_reduce_operator reduce_operator;
  bool keep_dims;
  bool use_neg_axes;
  int rank;
};

std::vector<int64_t> mask_to_axes(uint32_t mask) {
  std::vector<int64_t> axes;
  for (uint32_t i = 0; i < XNN_MAX_TENSOR_DIMS; ++i) {
    if (mask & (1 << i)) {
      axes.push_back(i);
    }
  }
  return axes;
}

void negate_axes(int64_t rank, std::vector<int64_t>& axes) {
  for (int64_t& axis : axes) {
    axis = axis - rank;
  }
}

template <typename T>
std::string to_string(const std::vector<T>& v) {
  std::stringstream sstr;
  sstr << "{";
  for (const T& t : v) {
    sstr << static_cast<float>(t);
    if (&t != &v.back()) {
      sstr << ", ";
    }
  }
  sstr << "}";
  return sstr.str();
}

std::function<void(float&, float)> get_reference_op(xnn_reduce_operator op) {
  switch (op) {
    case xnn_reduce_sum:
    case xnn_reduce_mean:
      return [](float& output, float input) { output += input; };
    case xnn_reduce_sum_squared:
    case xnn_reduce_mean_squared:
      return [](float& output, float input) { output += input * input; };
    case xnn_reduce_min:
      return
          [](float& output, float input) { output = std::min(output, input); };
    case xnn_reduce_max:
      return
          [](float& output, float input) { output = std::max(output, input); };
    default:
      XNN_UNREACHABLE;
  }
}

template <typename T, typename Accum>
void TestImpl(const Param& p) {
  xnn_datatype datatype = xnn_datatype_of<T>();

  ReplicableRandomDevice rng;

  auto reference_op = get_reference_op(p.reduce_operator);
  const bool is_minmax = (p.reduce_operator == xnn_reduce_min ||
                          p.reduce_operator == xnn_reduce_max);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  for (uint32_t mask = 1; mask < (1 << p.rank); ++mask) {
    std::vector<int64_t> reduction_axes = mask_to_axes(mask);
    if (p.use_neg_axes) {
      negate_axes(p.rank, reduction_axes);
    }

    xnn_quantization_params input_quantization =
        random_quantization(datatype, rng);
    xnn_quantization_params output_quantization =
        random_quantization(datatype, rng);

    if (is_minmax) {
      input_quantization = {0, 1.0f};
      output_quantization = input_quantization;
    }

    // Create a runtime with the reduce op in it.
    SubgraphTester tester(2);
    tester.AddInputTensor(p.rank, datatype, input_quantization, 0)
        .AddOutputTensor(p.keep_dims ? p.rank : p.rank - reduction_axes.size(),
                         datatype, output_quantization, 1)
        .AddReduce(p.reduce_operator, reduction_axes, 0, 1,
                   p.keep_dims ? XNN_FLAG_KEEP_DIMS : 0);

    xnn_status status = tester.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    // Run several times, with different shapes.
    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, p.rank);
      std::vector<size_t> output_shape = input_shape;
      size_t reduced_elements = 1;
      for (size_t i = 0; i < p.rank; ++i) {
        if (mask & (1 << i)) {
          reduced_elements *= input_shape[i];
          output_shape[i] = 1;
        }
      }

      if (std::is_integral<Accum>::value &&
          (p.reduce_operator == xnn_reduce_mean ||
           p.reduce_operator == xnn_reduce_sum)) {
        // Skip reduction tests that might overflow.
        if (reduced_elements >= std::log2(sizeof(Accum) - sizeof(T)) * 8) {
          continue;
        }
      }

      Tensor<T> input(input_shape, xnnpack::XnnExtraBytes);
      // Don't let the data be zero-mean, to avoid numerical issues with sums
      // near 0.
      DatatypeGenerator<T> generator(0.0f, 1.0f, input_quantization);
      input.generate([&]() { return generator(rng); });

      Tensor<T> output(output_shape);

      tester.ReshapeExternalTensor(input_shape, input.data(), 0)
          .ReshapeRuntime()
          .SetupExternalTensor(output.data(), 1)
          .SetupRuntime();

      if (p.keep_dims) {
        ASSERT_EQ(tester.GetExternalTensorShape(1), output_shape);
      } else {
        ASSERT_EQ(squeeze(tester.GetExternalTensorShape(1)),
                  squeeze(output_shape));
      }

      tester.InvokeRuntime();

      // Compute reference results.
      Tensor<float> expected(output_shape);
      expected.fill(get_reduce_identity<float>(p.reduce_operator));
      broadcast_extent_1(expected);
      for (const auto& i : EnumerateIndices(input.extents())) {
        reference_op(expected(i), dequantize(input(i), input_quantization));
      }
      const float scale = (p.reduce_operator == xnn_reduce_mean ||
                           p.reduce_operator == xnn_reduce_mean_squared)
                              ? 1.0f / reduced_elements
                              : 1.0f;

      // Verify the output matches the reference.
      for (const auto& i : EnumerateIndices(output.extents())) {
        const float reference = expected(i) * scale;
        if (xnn_datatype_is_quantized(datatype)) {
          ASSERT_NEAR(output(i), quantize<T>(reference, output_quantization), 1)
              << "input_shape=" << to_string(input_shape)
              << ", reduction_axes=" << to_string(reduction_axes);
        } else {
          ASSERT_NEAR(output(i), reference, 1e-3f * std::abs(reference) + 1e-3f)
              << "input_shape=" << to_string(input_shape)
              << ", reduction_axes=" << to_string(reduction_axes);
        }
      }
    }
  }
}

std::vector<size_t> normalize_shape(std::vector<size_t> shape) {
  return shape.empty() ? std::vector<size_t>({1}) : shape;
}

template <typename T, typename Accum = T>
void TestSubgraphRewrite(const Param& p) {
  if (p.reduce_operator != xnn_reduce_sum_squared &&
      p.reduce_operator != xnn_reduce_mean_squared) {
    GTEST_SKIP();
    return;
  }

  const size_t rank = p.rank;
  const bool use_neg_axes = p.use_neg_axes;
  const bool keep_dims = p.keep_dims;
  const xnn_reduce_operator reduce_operator = p.reduce_operator;
  auto reference_op = get_reference_op(reduce_operator);
  DatatypeGenerator<T> input_generator(-1.0, 1.0);

  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (uint32_t mask = 1; mask < (1 << rank); ++mask) {
    for (uint32_t iter = 0; iter < 10; iter++) {
      std::vector<int64_t> reduction_axes = mask_to_axes(mask);
      if (use_neg_axes) {
        negate_axes(rank, reduction_axes);
      }

      // Define subgraph
      enum external_value_ids : uint32_t {
        input_id = 0,
        output_id,
        num_external_values
      };
      SubgraphTester subgraph(num_external_values);
      std::vector<size_t> input_shape = random_shape(rng, rank);
      std::vector<size_t> output_shape;
      size_t reduced_elements = 1;
      for (size_t i = 0; i < rank; ++i) {
        if (mask & (1 << i)) {
          reduced_elements *= input_shape[i];
          if (keep_dims) {
            output_shape.push_back(1);
          }
        } else {
          output_shape.push_back(input_shape[i]);
        }
      }
      const T inv_n = 1.0 / reduced_elements;

      // Generate the input.
      Tensor<T> input(input_shape, xnnpack::XnnExtraBytes);
      input.generate([&]() { return input_generator(rng); });

      subgraph.AddInputTensor(input_shape, xnn_datatype_of<T>(), input_id);
      subgraph.AddOutputTensor(output_shape.size(), xnn_datatype_of<T>(),
                               output_id);

      // Generate the reduce_sum(sqr(x)) or reduce_sum(mul(x, x)) nodes.

      // b = mul(a, a) or b = sqr(a).
      uint32_t squared_id = XNN_INVALID_VALUE_ID;
      subgraph.AddInternalDynamicTensor(input_shape, xnn_datatype_of<T>(),
                                        &squared_id,
                                        /*flags=*/0);
      if (rng() % 2) {
        subgraph.AddMultiply(input_id, input_id, squared_id);
      } else {
        subgraph.AddUnary(xnn_unary_square, /*params=*/nullptr, input_id,
                          squared_id);
      }

      // c = reduce_sum(b).
      switch (reduce_operator) {
        case xnn_reduce_sum_squared:
          subgraph.AddReduce(xnn_reduce_sum, reduction_axes, squared_id,
                             output_id,
                             /*flags=*/keep_dims ? XNN_FLAG_KEEP_DIMS : 0);
          break;
        case xnn_reduce_mean_squared:
          if (rng() % 2) {
            subgraph.AddReduce(xnn_reduce_mean, reduction_axes, squared_id,
                               output_id,
                               /*flags=*/keep_dims ? XNN_FLAG_KEEP_DIMS : 0);
          } else {
            uint32_t sum_squared_id = XNN_INVALID_VALUE_ID;
            subgraph.AddInternalDynamicTensor(
                output_shape, xnn_datatype_of<T>(), &sum_squared_id,
                /*flags=*/0);
            subgraph.AddReduce(xnn_reduce_sum, reduction_axes, squared_id,
                               sum_squared_id,
                               /*flags=*/keep_dims ? XNN_FLAG_KEEP_DIMS : 0);
            // d = mul(c, inv_n).
            uint32_t inv_n_id = XNN_INVALID_VALUE_ID;
            subgraph.AddInternalStaticTensor(
                /*shape=*/{1}, xnn_datatype_of<T>(), &inv_n_id, &inv_n);
            if (rng() % 2) {
              subgraph.AddMultiply(sum_squared_id, inv_n_id, output_id);
            } else {
              subgraph.AddMultiply(inv_n_id, sum_squared_id, output_id);
            }
          }
          break;
        default:
          XNN_UNREACHABLE;
      }

      // Evaluate once with `XNN_FLAG_NO_OPERATOR_FUSION` enabled to
      // prevent the subgraph replacement.
      xnn_status status = subgraph.CreateRuntime(
          /*threadpool=*/nullptr,
          xnn_test_runtime_flags() | XNN_FLAG_NO_OPERATOR_FUSION);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
        return;
      }
      if (subgraph.NumNodes() == 0) {
        // If there are zero nodes, we aren't using XNNPACK's internal
        // implementation, don't try to use it.
        GTEST_SKIP();
        return;
      }
      ASSERT_GT(subgraph.NumNodes(), 1);

      // Reshape the subgraph.
      subgraph.ReshapeExternalTensor(input_shape, input.data(), input_id);
      subgraph.ReshapeRuntime();

      // Set up the input/output tensors.
      Tensor<T> output_original(output_shape, xnnpack::XnnExtraBytes);
      subgraph.SetupExternalTensor(output_original.base(), output_id);
      subgraph.SetupRuntime();
      ASSERT_EQ(normalize_shape(subgraph.GetExternalTensorShape(output_id)),
                normalize_shape(output_shape))
          << "input_shape=" << to_string(input_shape)
          << ", reduction_axes=" << to_string(reduction_axes);

      // Run the subgraph.
      subgraph.InvokeRuntime();

      // Re-create the runtime and evaluate again and check that the subgraph
      // was replaced.
      ASSERT_EQ(subgraph.CreateRuntime(), xnn_status_success);
      ASSERT_EQ(subgraph.NumNodes(), 1);
      ASSERT_EQ(subgraph.Node(0)->type,
                xnn_reduce_operator_to_node_type(reduce_operator));

      // Reshape the subgraph.
      Tensor<T> output_rewritten(output_shape, xnnpack::XnnExtraBytes);
      subgraph.ReshapeExternalTensor(input_shape, input.data(), input_id);
      subgraph.ReshapeExternalTensor(output_shape.size(),
                                     output_rewritten.data(), output_id);
      subgraph.ReshapeRuntime();

      // Set up the input/output tensors.
      subgraph.SetupRuntime();
      ASSERT_EQ(normalize_shape(subgraph.GetExternalTensorShape(output_id)),
                normalize_shape(output_shape))
          << "input_shape=" << to_string(input_shape)
          << ", reduction_axes=" << to_string(reduction_axes);

      // Run the subgraph.
      subgraph.InvokeRuntime();

      // Verify results, tolerance is computed with 2x the number of elements
      // since we have both a multiply and an add.
      const float tolerance =
          NumericLimits<T>::epsilon() * 2.0f * reduced_elements;
      ASSERT_THAT(output_rewritten,
                  testing::Pointwise(testing::NanSensitiveFloatNear(tolerance),
                                     output_original))
          << "input_shape=" << to_string(input_shape)
          << ", output_shape=" << to_string(output_shape)
          << ", reduction_axes=" << to_string(reduction_axes)
          << ", tolerance=" << tolerance;
    }
  }
}

using Reduce = ::testing::TestWithParam<Param>;

using ReduceQS8 = Reduce;
using ReduceQU8 = Reduce;
using ReduceF16 = Reduce;
using ReduceF32 = Reduce;
using ReduceF16Rewrite = Reduce;
using ReduceF32Rewrite = Reduce;

TEST_P(ReduceQS8, test) { TestImpl<quantized<int8_t>, int32_t>(GetParam()); }
TEST_P(ReduceQU8, test) { TestImpl<quantized<uint8_t>, int32_t>(GetParam()); }
TEST_P(ReduceF16, test) { TestImpl<xnn_float16, float>(GetParam()); }
TEST_P(ReduceF32, test) { TestImpl<float, float>(GetParam()); }
TEST_P(ReduceF16Rewrite, test) {
  TestSubgraphRewrite<xnn_float16, float>(GetParam());
}
TEST_P(ReduceF32Rewrite, test) {
  TestSubgraphRewrite<float, float>(GetParam());
}

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::Values;

auto params = testing::ConvertGenerator<Param::TupleT>(Combine(
    Values(xnn_reduce_sum, xnn_reduce_mean, xnn_reduce_max, xnn_reduce_min),
    Bool(), Bool(), Range(0, XNN_MAX_TENSOR_DIMS)));
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceQS8, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceQU8, params,
                         [](auto p) { return p.param.Name(); });

auto params2 = testing::ConvertGenerator<Param::TupleT>(Combine(
    Values(xnn_reduce_sum, xnn_reduce_mean, xnn_reduce_max, xnn_reduce_min,
           xnn_reduce_mean_squared, xnn_reduce_sum_squared),
    Bool(), Bool(), Range(0, XNN_MAX_TENSOR_DIMS)));
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF16, params2,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF32, params2,
                         [](auto p) { return p.param.Name(); });

auto params3 = testing::ConvertGenerator<Param::TupleT>(
    Combine(Values(xnn_reduce_mean_squared, xnn_reduce_sum_squared), Bool(),
            Bool(), Range(0, XNN_MAX_TENSOR_DIMS)));
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF16Rewrite, params3,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF32Rewrite, params3,
                         [](auto p) { return p.param.Name(); });

}  // namespace xnnpack
