// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
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

using ::testing::Contains;
using ::testing::ResultOf;

namespace xnnpack {

struct Param {
  using TupleT = std::tuple<xnn_reduce_operator, bool, int>;
  explicit Param(TupleT p)
      : reduce_operator(std::get<0>(p)),
        keep_dims(std::get<1>(p)),
        rank(std::get<2>(p)) {}

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
    sstr << "_" << rank;
    return sstr.str();
  }

  xnn_reduce_operator reduce_operator;
  bool keep_dims;
  int rank;
};

std::vector<int64_t> mask_to_axes(uint32_t mask, int32_t rank) {
  std::vector<int64_t> axes;
  for (int32_t i = 0; i < XNN_MAX_TENSOR_DIMS; ++i) {
    if (mask & (1 << i)) {
      axes.push_back(i);
    }
  }
  // It is legal to specify an axis more than once. To test negative axes,
  // we can just specify the same axis in both forms.
  for (int32_t i = 0;
       i < XNN_MAX_TENSOR_DIMS && axes.size() < XNN_MAX_TENSOR_DIMS; ++i) {
    if (mask & (1 << i)) {
      axes.push_back(i - rank);
    }
  }
  return axes;
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
  std::bernoulli_distribution empty_shape_dist(0.01f);

  auto reference_op = get_reference_op(p.reduce_operator);
  const bool is_minmax = (p.reduce_operator == xnn_reduce_min ||
                          p.reduce_operator == xnn_reduce_max);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  for (uint32_t mask = 1; mask < (1 << p.rank); ++mask) {
    const int num_reduction_axes =
        std::bitset<XNN_MAX_TENSOR_DIMS>(mask).count();
    std::vector<int64_t> reduction_axes = mask_to_axes(mask, p.rank);

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
        .AddOutputTensor(p.keep_dims ? p.rank : p.rank - num_reduction_axes,
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
          if (empty_shape_dist(rng)) {
            input_shape[i] = 0;
          }
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
      const bool is_mean = p.reduce_operator == xnn_reduce_mean ||
                           p.reduce_operator == xnn_reduce_mean_squared;
      const float scale = is_mean ? 1.0f / reduced_elements : 1.0f;

      // Verify the output matches the reference.
      for (const auto& i : EnumerateIndices(output.extents())) {
        const float reference = expected(i) * scale;
        if (xnn_datatype_is_quantized(datatype)) {
          if (!is_mean || reduced_elements > 0) {
            ASSERT_NEAR(output(i), quantize<T>(reference, output_quantization),
                        1)
                << "input_shape=" << to_string(input_shape)
                << ", reduction_axes=" << to_string(reduction_axes);
          } else {
            // An empty mean is not well defined (it is 0/0), let's not require
            // a specific value.
          }
        } else if (std::isnan(static_cast<float>(output(i))) &&
                   std::isnan(static_cast<float>(reference))) {
          // Both are NaN, but the below ASSERT_NEAR would fail.
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
  if (p.reduce_operator != xnn_reduce_sum_squared) {
    GTEST_SKIP();
    return;
  }

  const size_t rank = p.rank;
  const bool keep_dims = p.keep_dims;
  const xnn_reduce_operator reduce_operator = p.reduce_operator;
  auto reference_op = get_reference_op(reduce_operator);
  DatatypeGenerator<T> input_generator(-1.0, 1.0);

  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (uint32_t mask = 1; mask < (1 << rank); ++mask) {
    for (uint32_t iter = 0; iter < 10; iter++) {
      std::vector<int64_t> reduction_axes = mask_to_axes(mask, rank);

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

      // Generate the input.
      Tensor<T> input(input_shape, xnnpack::XnnExtraBytes);
      input.generate([&]() { return input_generator(rng); });

      const bool use_multiply = rng() % 2;
      auto build_subgraph = [&](SubgraphTester& tester) {
        tester.AddInputTensor(input_shape, xnn_datatype_of<T>(), input_id);
        tester.AddOutputTensor(output_shape.size(), xnn_datatype_of<T>(),
                               output_id);

        // Generate the reduce_sum(sqr(x)) or reduce_sum(mul(x, x)) nodes.

        // b = mul(a, a) or b = sqr(a).
        uint32_t squared_id = XNN_INVALID_VALUE_ID;
        tester.AddInternalDynamicTensor(input_shape, xnn_datatype_of<T>(),
                                        &squared_id,
                                        /*flags=*/0);
        if (use_multiply) {
          tester.AddMultiply(input_id, input_id, squared_id);
        } else {
          tester.AddUnary(xnn_unary_square, /*params=*/nullptr, input_id,
                          squared_id);
        }

        // c = reduce_sum(b).
        tester.AddReduce(xnn_reduce_sum, reduction_axes, squared_id, output_id,
                         /*flags=*/keep_dims ? XNN_FLAG_KEEP_DIMS : 0);
      };

      SubgraphTester subgraph_original(num_external_values);
      build_subgraph(subgraph_original);

      // Evaluate once with `XNN_FLAG_NO_OPERATOR_FUSION` enabled to
      // prevent the subgraph replacement.
      xnn_status status = subgraph_original.CreateRuntime(
          /*threadpool=*/nullptr,
          xnn_test_runtime_flags() | XNN_FLAG_NO_OPERATOR_FUSION);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
        return;
      }
      if (subgraph_original.NumNodes() == 0) {
        // If there are zero nodes, we aren't using XNNPACK's internal
        // implementation, don't try to use it.
        GTEST_SKIP();
        return;
      }
      ASSERT_GT(subgraph_original.NumNodes(), 1);

      // Reshape the subgraph.
      subgraph_original.ReshapeExternalTensor(input_shape, input.data(),
                                              input_id);
      subgraph_original.ReshapeRuntime();

      // Set up the input/output tensors.
      Tensor<T> output_original(output_shape, xnnpack::XnnExtraBytes);
      subgraph_original.SetupExternalTensor(output_original.base(), output_id);
      subgraph_original.SetupRuntime();
      ASSERT_EQ(
          normalize_shape(subgraph_original.GetExternalTensorShape(output_id)),
          normalize_shape(output_shape))
          << "input_shape=" << to_string(input_shape)
          << ", reduction_axes=" << to_string(reduction_axes);

      // Run the subgraph.
      subgraph_original.InvokeRuntime();

      // Create the runtime on a fresh subgraph with fusion enabled and check
      // that the subgraph was replaced.
      build_subgraph(subgraph);
      ASSERT_EQ(subgraph.CreateRuntime(), xnn_status_success);

      // Some rewrites may introduce other nodes. We check that the fused reduce
      // operator has been added.
      std::vector<int> indices(subgraph.NumNodes());
      std::iota(indices.begin(), indices.end(), 0);
      ASSERT_THAT(
          indices,
          Contains(
              ResultOf([&](int index) { return subgraph.Node(index)->type; },
                       xnn_reduce_operator_to_node_type(reduce_operator)))
              .Times(1));

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
      // A further 2x is needed in support of implementations with more
      // reassociation in the reduction (e.g RVV).
      const float tolerance =
          NumericLimits<T>::epsilon() * 4.0f * reduced_elements;
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
    Bool(), Range(0, XNN_MAX_TENSOR_DIMS)));
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceQS8, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceQU8, params,
                         [](auto p) { return p.param.Name(); });

auto params2 = testing::ConvertGenerator<Param::TupleT>(Combine(
    Values(xnn_reduce_sum, xnn_reduce_mean, xnn_reduce_max, xnn_reduce_min,
           xnn_reduce_mean_squared, xnn_reduce_sum_squared),
    Bool(), Range(0, XNN_MAX_TENSOR_DIMS)));
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF16, params2,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF32, params2,
                         [](auto p) { return p.param.Name(); });

auto params3 = testing::ConvertGenerator<Param::TupleT>(Combine(
    Values(xnn_reduce_sum_squared), Bool(), Range(0, XNN_MAX_TENSOR_DIMS)));
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF16Rewrite, params3,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF32Rewrite, params3,
                         [](auto p) { return p.param.Name(); });

#ifndef XNNPACK_USE_YNNPACK
TEST(StaticReduce, WidenFp16Accumulators) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  const std::vector<size_t> input_shape = {4, 128};
  Tensor<xnn_float16> input(input_shape, xnnpack::XnnExtraBytes);
  // Each element is 32, so sum(sqr(x)) = 128 * 1024 = 131072 > fp16 max
  // (65504), while the scaled mean 1024 fits in fp16.
  for (size_t i = 0; i < input.size(); ++i) {
    input.base()[i] = xnn_float16(32.0f + static_cast<float>(i % 4));
  }

  for (int64_t axis : {int64_t{0}, int64_t{1}}) {
    for (bool keep_dims : {false, true}) {
      for (bool square_input : {false, true}) {
        for (int op_variant = 0; op_variant < 4; ++op_variant) {
          const size_t reduced_elements = input_shape[axis];
          std::vector<size_t> output_shape;
          for (size_t d = 0; d < input_shape.size(); ++d) {
            if (static_cast<int64_t>(d) == axis) {
              if (keep_dims) {
                output_shape.push_back(1);
              }
            } else {
              output_shape.push_back(input_shape[d]);
            }
          }

          const xnn_float16 scale =
              op_variant >= 2
                  ? xnn_float16(static_cast<float>(reduced_elements))
                  : xnn_float16(1.0f / static_cast<float>(reduced_elements));

          SubgraphTester subgraph(2);
          subgraph.AddInputTensor(input_shape.size(), xnn_datatype_fp16, 0)
              .AddOutputTensor(output_shape.size(), xnn_datatype_fp16, 1);
          uint32_t reduce_input_id = 0;
          if (square_input) {
            subgraph.AddInternalDynamicTensor(
                input_shape.size(), xnn_datatype_fp16, &reduce_input_id);
            subgraph.AddMultiply(0, 0, reduce_input_id);
          }
          uint32_t reduced_id = XNN_INVALID_VALUE_ID;
          subgraph.AddInternalDynamicTensor(output_shape.size(),
                                            xnn_datatype_fp16, &reduced_id);
          subgraph.AddReduce(xnn_reduce_sum, {axis}, reduce_input_id,
                             reduced_id, keep_dims ? XNN_FLAG_KEEP_DIMS : 0);
          uint32_t scale_id = XNN_INVALID_VALUE_ID;
          subgraph.AddInternalStaticTensor({1}, xnn_datatype_fp16, &scale_id,
                                           &scale);
          if (op_variant == 0) {
            subgraph.AddMultiply(reduced_id, scale_id, 1);
          } else if (op_variant == 1) {
            subgraph.AddMultiply(scale_id, reduced_id, 1);
          } else if (op_variant == 2) {
            subgraph.AddDivide(reduced_id, scale_id, 1);
          } else {
            subgraph.AddDivide(scale_id, reduced_id, 1);
          }

          const xnn_status status = subgraph.CreateRuntime();
          if (status == xnn_status_unsupported_hardware) {
            GTEST_SKIP();
          }
          ASSERT_EQ(status, xnn_status_success);

          for (const std::vector<size_t>& shape :
               {input_shape, std::vector<size_t>{2, 64}}) {
            std::vector<size_t> cur_out_shape;
            for (size_t d = 0; d < shape.size(); ++d) {
              if (static_cast<int64_t>(d) == axis) {
                if (keep_dims) {
                  cur_out_shape.push_back(1);
                }
              } else {
                cur_out_shape.push_back(shape[d]);
              }
            }
            Tensor<xnn_float16> output(cur_out_shape, xnnpack::XnnExtraBytes);
            subgraph.ReshapeExternalTensor(shape, input.data(), 0)
                .ReshapeRuntime()
                .SetupExternalTensor(output.base(), 1)
                .SetupRuntime()
                .InvokeRuntime();

            const size_t cur_reduced = shape[axis];
            const size_t unreduced = shape[1 - axis];
            for (size_t o = 0; o < unreduced; ++o) {
              float acc = 0.0f;
              for (size_t r = 0; r < cur_reduced; ++r) {
                const size_t idx =
                    axis == 1 ? (o * shape[1] + r) : (r * shape[1] + o);
                const float v = static_cast<float>(input.base()[idx]);
                acc += square_input ? v * v : v;
              }
              float expected;
              if (op_variant < 2) {
                expected = acc * static_cast<float>(scale);
              } else if (op_variant == 2) {
                expected = acc / static_cast<float>(scale);
              } else {
                expected = static_cast<float>(scale) / acc;
              }
              const float actual = static_cast<float>(output.base()[o]);
              ASSERT_NEAR(actual, expected, std::abs(expected) * 2e-3f)
                  << "axis=" << axis << ", keep_dims=" << keep_dims
                  << ", square_input=" << square_input
                  << ", op_variant=" << op_variant << ", o=" << o;
            }
          }
        }
      }
    }
  }
}

TEST(StaticReduce, WidenFp16AccumulatorsForceFp16) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  const std::vector<size_t> input_shape = {4, 128};
  Tensor<float> input(input_shape, xnnpack::XnnExtraBytes);
  for (size_t i = 0; i < input.size(); ++i) {
    input.base()[i] = 32.0f + static_cast<float>(i % 4);
  }

  for (int64_t axis : {int64_t{0}, int64_t{1}}) {
    for (bool keep_dims : {false, true}) {
      for (bool square_input : {false, true}) {
        for (int op_variant = 0; op_variant < 3; ++op_variant) {
          const size_t reduced_elements = input_shape[axis];
          std::vector<size_t> output_shape;
          for (size_t d = 0; d < input_shape.size(); ++d) {
            if (static_cast<int64_t>(d) == axis) {
              if (keep_dims) {
                output_shape.push_back(1);
              }
            } else {
              output_shape.push_back(input_shape[d]);
            }
          }
          // For divide with squared inputs on axis 1, divide by 65536 (> fp16
          // max) to verify the original fp32 scalar is preserved.
          const float divisor = (square_input && axis == 1)
                                    ? 65536.0f
                                    : static_cast<float>(reduced_elements);
          const float scale = op_variant == 2 ? divisor : (1.0f / divisor);

          SubgraphTester subgraph(2);
          subgraph.AddInputTensor(input_shape, xnn_datatype_fp32, 0)
              .AddOutputTensor(output_shape.size(), xnn_datatype_fp32, 1);
          uint32_t reduce_input_id = 0;
          if (square_input) {
            subgraph.AddInternalDynamicTensor(input_shape, xnn_datatype_fp32,
                                              &reduce_input_id);
            subgraph.AddMultiply(0, 0, reduce_input_id);
          }
          uint32_t sum_id = XNN_INVALID_VALUE_ID;
          subgraph.AddInternalDynamicTensor(output_shape, xnn_datatype_fp32,
                                            &sum_id);
          subgraph.AddReduce(xnn_reduce_sum, {axis}, reduce_input_id, sum_id,
                             keep_dims ? XNN_FLAG_KEEP_DIMS : 0);
          uint32_t scale_id = XNN_INVALID_VALUE_ID;
          subgraph.AddInternalStaticTensor({1}, xnn_datatype_fp32, &scale_id,
                                           &scale);
          if (op_variant == 0) {
            subgraph.AddMultiply(sum_id, scale_id, 1);
          } else if (op_variant == 1) {
            subgraph.AddMultiply(scale_id, sum_id, 1);
          } else {
            subgraph.AddDivide(sum_id, scale_id, 1);
          }

          const xnn_status status = subgraph.CreateRuntime(
              /*threadpool=*/nullptr,
              xnn_test_runtime_flags() | XNN_FLAG_FORCE_FP16_INFERENCE);
          if (status == xnn_status_unsupported_hardware) {
            GTEST_SKIP();
          }
          ASSERT_EQ(status, xnn_status_success);

          Tensor<float> output(output_shape, xnnpack::XnnExtraBytes);
          subgraph.ReshapeExternalTensor(input_shape, input.data(), 0)
              .ReshapeRuntime()
              .SetupExternalTensor(output.base(), 1)
              .SetupRuntime()
              .InvokeRuntime();

          const size_t unreduced = input_shape[1 - axis];
          for (size_t o = 0; o < unreduced; ++o) {
            float acc = 0.0f;
            for (size_t r = 0; r < reduced_elements; ++r) {
              const float v = axis == 1 ? input({o, r}) : input({r, o});
              acc += square_input ? v * v : v;
            }
            const float expected = acc / divisor;
            ASSERT_NEAR(output.base()[o], expected, expected * 2e-3f)
                << "axis=" << axis << ", keep_dims=" << keep_dims
                << ", square_input=" << square_input
                << ", op_variant=" << op_variant << ", o=" << o;
          }
        }
      }
    }
  }

  // Verify subgraph cleanup when optimized without creating a runtime.
  const float scale = 1.0f / 128.0f;
  SubgraphTester unrun(2);
  unrun.AddInputTensor(input_shape, xnn_datatype_fp32, 0)
      .AddOutputTensor(2, xnn_datatype_fp32, 1);
  uint32_t sum_id = XNN_INVALID_VALUE_ID;
  unrun.AddInternalDynamicTensor({4, 1}, xnn_datatype_fp32, &sum_id);
  unrun.AddReduce(xnn_reduce_sum, {1}, 0, sum_id, XNN_FLAG_KEEP_DIMS);
  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  unrun.AddInternalStaticTensor({1}, xnn_datatype_fp32, &scale_id, &scale);
  unrun.AddMultiply(sum_id, scale_id, 1);
  unrun.Optimize(xnn_test_runtime_flags() | XNN_FLAG_FORCE_FP16_INFERENCE);
}

TEST(StaticReduce, WidenFp16AccumulatorsSharedScalarAndExternalSum) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  const std::vector<size_t> shape = {2, 128};
  Tensor<xnn_float16> input_a(shape, xnnpack::XnnExtraBytes);
  Tensor<xnn_float16> input_b(shape, xnnpack::XnnExtraBytes);
  input_a.fill(xnn_float16(600.0f));  // sum = 76800 > fp16 max (65504)
  input_b.fill(xnn_float16(800.0f));  // sum = 102400 > fp16 max (65504)
  const xnn_float16 scale(1.0f / 128.0f);

  // Two reduce_sum->mul ops (one with a downstream unary op) and a non-reduce
  // mul sharing a single static scalar constant.
  {
    SubgraphTester subgraph(5);
    subgraph.AddInputTensor(shape, xnn_datatype_fp16, 0)
        .AddInputTensor(shape, xnn_datatype_fp16, 1)
        .AddOutputTensor(2, xnn_datatype_fp16, 2)
        .AddOutputTensor(2, xnn_datatype_fp16, 3)
        .AddOutputTensor(2, xnn_datatype_fp16, 4);
    uint32_t scale_id = XNN_INVALID_VALUE_ID;
    subgraph.AddInternalStaticTensor({1}, xnn_datatype_fp16, &scale_id, &scale);
    uint32_t sum_a_id = XNN_INVALID_VALUE_ID;
    uint32_t mean_a_id = XNN_INVALID_VALUE_ID;
    uint32_t sum_b_id = XNN_INVALID_VALUE_ID;
    subgraph.AddInternalDynamicTensor({2, 1}, xnn_datatype_fp16, &sum_a_id)
        .AddInternalDynamicTensor({2, 1}, xnn_datatype_fp16, &mean_a_id)
        .AddInternalDynamicTensor({2, 1}, xnn_datatype_fp16, &sum_b_id)
        .AddReduce(xnn_reduce_sum, {1}, 0, sum_a_id, XNN_FLAG_KEEP_DIMS)
        .AddMultiply(sum_a_id, scale_id, mean_a_id)
        .AddUnary(xnn_unary_square_root, /*params=*/nullptr, mean_a_id, 2)
        .AddReduce(xnn_reduce_sum, {1}, 1, sum_b_id, XNN_FLAG_KEEP_DIMS)
        .AddMultiply(scale_id, sum_b_id, 3)
        .AddMultiply(0, scale_id, 4);

    const xnn_status status = subgraph.CreateRuntime(
        /*threadpool=*/nullptr, xnn_test_runtime_flags());
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
    }
    ASSERT_EQ(status, xnn_status_success);

    Tensor<xnn_float16> out_sqrt_mean_a({2, 1}, xnnpack::XnnExtraBytes);
    Tensor<xnn_float16> out_mean_b({2, 1}, xnnpack::XnnExtraBytes);
    Tensor<xnn_float16> out_scaled_a(shape, xnnpack::XnnExtraBytes);
    subgraph.ReshapeExternalTensor(shape, input_a.data(), 0)
        .ReshapeExternalTensor(shape, input_b.data(), 1)
        .ReshapeRuntime()
        .SetupExternalTensor(out_sqrt_mean_a.base(), 2)
        .SetupExternalTensor(out_mean_b.base(), 3)
        .SetupExternalTensor(out_scaled_a.base(), 4)
        .SetupRuntime()
        .InvokeRuntime();

    for (size_t i = 0; i < 2; ++i) {
      ASSERT_NEAR(static_cast<float>(out_sqrt_mean_a.base()[i]),
                  std::sqrt(600.0f), 5e-2f);
      ASSERT_NEAR(static_cast<float>(out_mean_b.base()[i]), 800.0f, 1.0f);
    }
    ASSERT_NEAR(static_cast<float>(out_scaled_a.base()[0]), 600.0f / 128.0f,
                1e-2f);
  }

  // External reduce_sum output must keep its fp16 datatype and buffer size.
  {
    Tensor<xnn_float16> small_input(shape, xnnpack::XnnExtraBytes);
    small_input.fill(xnn_float16(2.0f));
    SubgraphTester subgraph(3);
    subgraph.AddInputTensor(shape, xnn_datatype_fp16, 0)
        .AddOutputTensor(2, xnn_datatype_fp16, 1)
        .AddOutputTensor(2, xnn_datatype_fp16, 2);
    uint32_t scale_id = XNN_INVALID_VALUE_ID;
    subgraph.AddInternalStaticTensor({1}, xnn_datatype_fp16, &scale_id, &scale);
    subgraph.AddReduce(xnn_reduce_sum, {1}, 0, 1, XNN_FLAG_KEEP_DIMS)
        .AddMultiply(1, scale_id, 2);

    const xnn_status status = subgraph.CreateRuntime(
        /*threadpool=*/nullptr, xnn_test_runtime_flags());
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
    }
    ASSERT_EQ(status, xnn_status_success);

    Tensor<xnn_float16> out_sum({2, 1}, xnnpack::XnnExtraBytes);
    Tensor<xnn_float16> out_mean({2, 1}, xnnpack::XnnExtraBytes);
    subgraph.ReshapeExternalTensor(shape, small_input.data(), 0)
        .ReshapeRuntime()
        .SetupExternalTensor(out_sum.base(), 1)
        .SetupExternalTensor(out_mean.base(), 2)
        .SetupRuntime()
        .InvokeRuntime();

    for (size_t i = 0; i < 2; ++i) {
      ASSERT_NEAR(static_cast<float>(out_sum.base()[i]), 256.0f, 0.5f);
      ASSERT_NEAR(static_cast<float>(out_mean.base()[i]), 2.0f, 1e-2f);
    }
  }
}

TEST(StaticReduce, DefineRejectsMismatchedOutputDatatype) {
  const int64_t reduction_axes[1] = {1};
  for (xnn_reduce_operator op : {xnn_reduce_min, xnn_reduce_max}) {
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(2, xnn_datatype_fp16, 0)
        .AddOutputTensor(2, xnn_datatype_fp32, 1);
    EXPECT_EQ(xnn_define_static_reduce_v2(subgraph.Subgraph(), op,
                                          /*num_reduction_axes=*/1,
                                          reduction_axes, /*input_id=*/0,
                                          /*output_id=*/1, /*flags=*/0),
              xnn_status_invalid_parameter);
  }
  SubgraphTester subgraph(2);
  subgraph.AddInputTensor(2, xnn_datatype_fp32, 0)
      .AddOutputTensor(2, xnn_datatype_fp16, 1);
  EXPECT_EQ(xnn_define_static_reduce_v2(subgraph.Subgraph(), xnn_reduce_sum,
                                        /*num_reduction_axes=*/1,
                                        reduction_axes, /*input_id=*/0,
                                        /*output_id=*/1, /*flags=*/0),
            xnn_status_invalid_parameter);
}

TEST(StaticReduce, DefineRejectsUnsupportedInputDatatype) {
  SubgraphTester subgraph(2);
  subgraph.AddInputTensor(2, xnn_datatype_int32, 0)
      .AddOutputTensor(2, xnn_datatype_int32, 1);
  const int64_t reduction_axes[1] = {1};
  EXPECT_EQ(xnn_define_static_reduce_v2(subgraph.Subgraph(), xnn_reduce_sum,
                                        /*num_reduction_axes=*/1,
                                        reduction_axes, /*input_id=*/0,
                                        /*output_id=*/1, /*flags=*/0),
            xnn_status_unsupported_parameter);
}

TEST(StaticReduce, DefineRejectsUnsupportedQuantizedSumSquared) {
  const xnn_quantization_params quantization = {0, 1.0f};
  const int64_t reduction_axes[1] = {1};
  for (xnn_datatype datatype : {xnn_datatype_qint8, xnn_datatype_quint8}) {
    for (xnn_reduce_operator op :
         {xnn_reduce_sum_squared, xnn_reduce_mean_squared}) {
      SubgraphTester subgraph(2);
      subgraph.AddInputTensor(2, datatype, quantization, 0)
          .AddOutputTensor(2, datatype, quantization, 1);
      EXPECT_EQ(xnn_define_static_reduce_v2(subgraph.Subgraph(), op,
                                            /*num_reduction_axes=*/1,
                                            reduction_axes, /*input_id=*/0,
                                            /*output_id=*/1, /*flags=*/0),
                xnn_status_unsupported_parameter);
    }
  }
}

TEST(StaticReduce, DefineRejectsTooManyReductionAxes) {
  SubgraphTester subgraph(2);
  subgraph.AddInputTensor(2, xnn_datatype_fp32, 0)
      .AddOutputTensor(2, xnn_datatype_fp32, 1);
  const int64_t reduction_axes[XNN_MAX_TENSOR_DIMS + 1] = {};
  EXPECT_EQ(xnn_define_static_reduce_v2(subgraph.Subgraph(), xnn_reduce_sum,
                                        XNN_MAX_TENSOR_DIMS + 1, reduction_axes,
                                        /*input_id=*/0, /*output_id=*/1,
                                        /*flags=*/0),
            xnn_status_unsupported_parameter);
  const size_t unsigned_reduction_axes[XNN_MAX_TENSOR_DIMS + 1] = {};
  EXPECT_EQ(xnn_define_static_reduce(
                subgraph.Subgraph(), xnn_reduce_sum, XNN_MAX_TENSOR_DIMS + 1,
                unsigned_reduction_axes, /*input_id=*/0, /*output_id=*/1,
                /*flags=*/0),
            xnn_status_unsupported_parameter);
}

TEST(StaticReduce, DefineRejectsZeroReductionAxes) {
  SubgraphTester subgraph(2);
  subgraph.AddInputTensor(2, xnn_datatype_fp32, 0)
      .AddOutputTensor(2, xnn_datatype_fp32, 1);
  EXPECT_EQ(xnn_define_static_reduce_v2(subgraph.Subgraph(), xnn_reduce_sum,
                                        /*num_reduction_axes=*/0,
                                        /*reduction_axes=*/nullptr,
                                        /*input_id=*/0, /*output_id=*/1,
                                        /*flags=*/0),
            xnn_status_invalid_parameter);
}

TEST(StaticReduce, DefineRejectsQuantizedMinMaxRequantization) {
  const xnn_quantization_params input_quantization = {1, 0.5f};
  const int64_t reduction_axes[1] = {1};
  for (xnn_datatype datatype : {xnn_datatype_qint8, xnn_datatype_quint8}) {
    for (xnn_reduce_operator op : {xnn_reduce_min, xnn_reduce_max}) {
      for (xnn_quantization_params output_quantization :
           {xnn_quantization_params{2, 0.5f},
            xnn_quantization_params{1, 0.25f}}) {
        SubgraphTester subgraph(2);
        subgraph.AddInputTensor(2, datatype, input_quantization, 0)
            .AddOutputTensor(2, datatype, output_quantization, 1);
        EXPECT_EQ(xnn_define_static_reduce_v2(subgraph.Subgraph(), op,
                                              /*num_reduction_axes=*/1,
                                              reduction_axes, /*input_id=*/0,
                                              /*output_id=*/1, /*flags=*/0),
                  xnn_status_unsupported_parameter);
      }

      SubgraphTester subgraph(2);
      subgraph.AddInputTensor(2, datatype, input_quantization, 0)
          .AddOutputTensor(2, datatype, input_quantization, 1);
      EXPECT_EQ(xnn_define_static_reduce_v2(subgraph.Subgraph(), op,
                                            /*num_reduction_axes=*/1,
                                            reduction_axes, /*input_id=*/0,
                                            /*output_id=*/1, /*flags=*/0),
                xnn_status_success);
    }
  }
}
#endif  // XNNPACK_USE_YNNPACK

}  // namespace xnnpack
