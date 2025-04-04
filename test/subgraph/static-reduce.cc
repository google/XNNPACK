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

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
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
      case xnn_reduce_sum:
        sstr << "sum";
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
    sstr << t;
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

      Tensor<T> input(input_shape, PaddingBytes{XNN_EXTRA_BYTES});
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
      const float scale =
          p.reduce_operator == xnn_reduce_mean ? 1.0f / reduced_elements : 1.0f;

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

template <typename T>
class Reduce : public ::testing::TestWithParam<Param> {};

using ReduceQS8 = Reduce<quantized<int8_t>>;
using ReduceQU8 = Reduce<quantized<uint8_t>>;
using ReduceF16 = Reduce<xnn_float16>;
using ReduceF32 = Reduce<float>;

TEST_P(ReduceQS8, test) { TestImpl<quantized<int8_t>, int32_t>(GetParam()); }
TEST_P(ReduceQU8, test) { TestImpl<quantized<uint8_t>, int32_t>(GetParam()); }
TEST_P(ReduceF16, test) { TestImpl<xnn_float16, float>(GetParam()); }
TEST_P(ReduceF32, test) { TestImpl<float, float>(GetParam()); }

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
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF16, params,
                         [](auto p) { return p.param.Name(); });
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceF32, params,
                         [](auto p) { return p.param.Name(); });

}  // namespace xnnpack
