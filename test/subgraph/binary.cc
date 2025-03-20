// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/reference-utils.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

using ::testing::Combine;
using ::testing::ValuesIn;

namespace xnnpack {

std::vector<size_t> input_shape(size_t rank, std::vector<size_t> dims) {
  // Remove the leading dimensions to reduce to `rank` dimensions.
  std::reverse(dims.begin(), dims.end());
  dims.resize(rank);
  std::reverse(dims.begin(), dims.end());
  return dims;
}

float compute_float(xnn_binary_operator op, float a, float b) {
  switch (op) {
    case xnn_binary_add:
      return a + b;
    case xnn_binary_copysign:
      return std::copysign(a, b);
    case xnn_binary_divide:
      return a / b;
    case xnn_binary_maximum:
      return std::max(a, b);
    case xnn_binary_minimum:
      return std::min(a, b);
    case xnn_binary_multiply:
      return a * b;
    case xnn_binary_subtract:
      return a - b;
    case xnn_binary_squared_difference:
      return (a - b) * (a - b);
    case xnn_binary_prelu:
      return a < 0 ? a * b : a;
    case xnn_binary_modulus:
      return std::fmod(a, b);
    case xnn_binary_atan2:
      return std::atan2(a, b);
    case xnn_binary_pow:
      return std::pow(a, b);
    case xnn_binary_bitwise_and:
    case xnn_binary_bitwise_or:
    case xnn_binary_bitwise_xor:
    case xnn_binary_shift_left:
    case xnn_binary_shift_right_logical:
    case xnn_binary_shift_right_arithmetic:
    case xnn_binary_invalid:
      break;
  }
  XNN_UNREACHABLE;
  return 0.0;
}

int32_t compute_integral(xnn_binary_operator op, int32_t a, int32_t b) {
  switch (op) {
    case xnn_binary_add:
      return widen(a) + widen(b);
    case xnn_binary_copysign:
      return std::copysign(a, b);
    case xnn_binary_divide:
      return euclidean_div(a, b);
    case xnn_binary_maximum:
      return std::max(a, b);
    case xnn_binary_minimum:
      return std::min(a, b);
    case xnn_binary_multiply:
      return widen(a) * widen(b);
    case xnn_binary_subtract:
      return widen(a) - widen(b);
    case xnn_binary_modulus:
      return euclidean_mod(a, b);
    case xnn_binary_pow:
      return integer_pow(a, b);
    case xnn_binary_bitwise_and:
      return a & b;
    case xnn_binary_bitwise_or:
      return a | b;
    case xnn_binary_bitwise_xor:
      return a ^ b;
    case xnn_binary_shift_left:
      return a << (b & 31);
    case xnn_binary_shift_right_logical:
      return static_cast<uint32_t>(a) >> (b & 31);
    case xnn_binary_shift_right_arithmetic:
      return a >> (b & 31);
    case xnn_binary_squared_difference:
    case xnn_binary_atan2:
    case xnn_binary_prelu:
    case xnn_binary_invalid:
      break;
  }
  XNN_UNREACHABLE;
  return 0;
}

float compute_tolerance(xnn_datatype datatype, float output_ref) {
  return (std::abs(output_ref) + 1.0f) * 3.0f * epsilon(datatype);
}

struct Param {
  using TupleT = std::tuple<xnn_datatype, xnn_binary_operator, int>;
  explicit Param(TupleT p)
      : datatype(std::get<0>(p)), op(std::get<1>(p)), rank(std::get<2>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    sstr << xnn_datatype_to_string(datatype) << "_"
         << xnn_binary_operator_to_string(op) << "_rank" << rank;
    return sstr.str();
  }

  xnn_datatype datatype;
  xnn_binary_operator op;
  size_t rank;
};

template <typename T>
void TestImpl(const Param& p) {
  ReplicableRandomDevice rng;

  // We want the total number of elements to be reasonable, so choose max_dim
  // such that a random shape of rank `p.rank` produces this max size.
  constexpr size_t max_size = 1024;
  const size_t max_dim = static_cast<size_t>(std::ceil(
      std::pow(static_cast<double>(max_size),
               1.0 / static_cast<double>(std::max<size_t>(1, p.rank)))));
  for (size_t input_rank = 0; input_rank <= p.rank; input_rank++) {
    for (std::pair<size_t, size_t> input_ranks :
         {std::make_pair(input_rank, p.rank),
          std::make_pair(p.rank, input_rank)}) {
      xnn_quantization_params a_quantization =
          random_quantization(p.datatype, rng);
      xnn_quantization_params b_quantization =
          random_quantization(p.datatype, rng);
      xnn_quantization_params output_quantization =
          random_quantization(p.datatype, rng);

      DatatypeGenerator<T> output_gen(output_quantization);
      xnn_binary_params params = {
          dequantize(output_gen(rng), output_quantization),
          dequantize(output_gen(rng), output_quantization),
      };
      if (params.output_min > params.output_max) {
        std::swap(params.output_min, params.output_max);
      }

      SubgraphTester subgraph(3);
      subgraph.AddInputTensor(input_ranks.first, p.datatype, a_quantization, 0)
          .AddInputTensor(input_ranks.second, p.datatype, b_quantization, 1)
          .AddOutputTensor(p.rank, p.datatype, output_quantization, 2)
          .AddBinary(p.op, &params, 0, 1, 2);
      xnn_status status = subgraph.CreateRuntime();
      ASSERT_EQ(status, xnn_status_success);

      for (int reshape = 0; reshape < 2; ++reshape) {
        std::vector<size_t> output_shape =
            random_shape(rng, p.rank, 1, max_dim);
        std::vector<size_t> a_shape =
            input_shape(input_ranks.first, output_shape);
        std::vector<size_t> b_shape =
            input_shape(input_ranks.second, output_shape);

        Tensor<T> a(a_shape, {XNN_EXTRA_BYTES});
        Tensor<T> b(b_shape, {XNN_EXTRA_BYTES});
        Tensor<T> output(output_shape);

        DatatypeGenerator<T> a_gen(a_quantization);
        DatatypeGenerator<T> b_gen(b_quantization);
        a.generate([&]() { return a_gen(rng); });
        b.generate([&]() { return b_gen(rng); });

        subgraph.ReshapeExternalTensor(a_shape, a.data(), 0)
            .ReshapeExternalTensor(b_shape, b.data(), 1)
            .ReshapeRuntime();

        ASSERT_EQ(subgraph.GetExternalTensorShape(2), output_shape);

        subgraph.SetupExternalTensor(output.data(), 2)
            .SetupRuntime()
            .InvokeRuntime();

        for (const auto& i : EnumerateIndices(output.extents())) {
          if (std::is_integral<T>::value) {
            const int32_t expected = std::min<int32_t>(
                std::max<int32_t>(compute_integral(p.op, a(i), b(i)),
                                  params.output_min),
                params.output_max);
            ASSERT_EQ(expected, output(i))
                << "i = " << index_to_string(i)
                << ", a_shape=" << index_to_string(a_shape)
                << ", b_shape=" << index_to_string(b_shape)
                << ", a(i) = " << static_cast<int32_t>(a(i))
                << ", b(i) = " << static_cast<int32_t>(b(i));
          } else if (is_quantized<T>()) {
            const float a_i = dequantize(a(i), a_quantization);
            const float b_i = dequantize(b(i), b_quantization);
            float expected = compute_float(p.op, a_i, b_i);
            expected = std::max<float>(expected, params.output_min);
            expected = std::min<float>(expected, params.output_max);
            expected = fake_quantize(expected, output_quantization);
            expected = std::max<float>(expected, NumericLimits<T>::min());
            expected = std::min<float>(expected, NumericLimits<T>::max());
            if (std::isnan(expected)) {
              // We don't know how to represent NaN for quantized datatypes.
            } else {
              ASSERT_NEAR(expected, output(i), 1)
                  << "i = " << index_to_string(i)
                  << ", a_shape=" << index_to_string(a_shape)
                  << ", b_shape=" << index_to_string(b_shape)
                  << ", a(i) = " << a_i << " (" << static_cast<int32_t>(a(i))
                  << ")"
                  << ", b(i) = " << b_i << " (" << static_cast<int32_t>(b(i))
                  << ")"
                  << ", output(i) = " << static_cast<int32_t>(output(i));
            }
          } else {
            float expected = static_cast<T>(compute_float(p.op, a(i), b(i)));
            expected = std::max<float>(expected, params.output_min);
            expected = std::min<float>(expected, params.output_max);
            if (std::isnan(expected)) {
              // Checking the output is NaN could make sense, but it fails in
              // a variety of cases.
            } else {
              ASSERT_NEAR(expected, output(i),
                          compute_tolerance(p.datatype, expected))
                  << "i = " << index_to_string(i)
                  << ", a_shape=" << index_to_string(a_shape)
                  << ", b_shape=" << index_to_string(b_shape)
                  << ", a(i) = " << static_cast<float>(a(i))
                  << ", b(i) = " << static_cast<float>(b(i));
            }
          }
        }
      }
    }
  }
}

class IntegerOps : public testing::TestWithParam<Param> {};
class RealOps : public testing::TestWithParam<Param> {};

TEST_P(IntegerOps, test) {
  switch (GetParam().datatype) {
    case xnn_datatype_int32:
      TestImpl<int>(GetParam());
      break;
    default:
      XNN_UNREACHABLE;
  }
}

TEST_P(RealOps, test) {
  switch (GetParam().datatype) {
    case xnn_datatype_qint8:
      TestImpl<quantized<int8_t>>(GetParam());
      break;
    case xnn_datatype_quint8:
      TestImpl<quantized<uint8_t>>(GetParam());
      break;
    case xnn_datatype_fp16:
      TestImpl<xnn_float16>(GetParam());
      break;
    case xnn_datatype_bf16:
      TestImpl<xnn_bfloat16>(GetParam());
      break;
    case xnn_datatype_fp32:
      TestImpl<float>(GetParam());
      break;
    default:
      XNN_UNREACHABLE;
  }
}

// clang-format off
const xnn_datatype all_integer_datatypes[] = {
    xnn_datatype_int32,
};

const xnn_datatype all_real_datatypes[] = {
    xnn_datatype_quint8,
    xnn_datatype_qint8,
#ifndef XNN_EXCLUDE_F16_TESTS
    xnn_datatype_fp16,
#endif
    xnn_datatype_bf16,
    xnn_datatype_fp32,
};

const xnn_binary_operator all_integer_ops[] = {
    xnn_binary_add,
    xnn_binary_copysign,
    xnn_binary_divide,
    xnn_binary_maximum,
    xnn_binary_minimum,
    xnn_binary_multiply,
    xnn_binary_subtract,
    xnn_binary_modulus,
    xnn_binary_pow,
    xnn_binary_bitwise_and,
    xnn_binary_bitwise_or,
    xnn_binary_bitwise_xor,
    xnn_binary_shift_left,
    xnn_binary_shift_right_logical,
    xnn_binary_shift_right_arithmetic,
};

const xnn_binary_operator all_real_ops[] = {
    xnn_binary_add,
    xnn_binary_atan2,
    xnn_binary_copysign,
    xnn_binary_divide,
    xnn_binary_maximum,
    xnn_binary_minimum,
    xnn_binary_modulus,
    xnn_binary_multiply,
    xnn_binary_pow,
    xnn_binary_prelu,
    xnn_binary_squared_difference,
    xnn_binary_subtract,
};
// clang-format on

auto all_ranks = testing::Range(0, XNN_MAX_TENSOR_DIMS);

INSTANTIATE_TEST_SUITE_P(BinaryTest, IntegerOps,
                         testing::ConvertGenerator<Param::TupleT>(
                             Combine(ValuesIn(all_integer_datatypes),
                                     ValuesIn(all_integer_ops), all_ranks)),
                         [](const auto& info) { return info.param.Name(); });

INSTANTIATE_TEST_SUITE_P(BinaryTest, RealOps,
                         testing::ConvertGenerator<Param::TupleT>(
                             Combine(ValuesIn(all_real_datatypes),
                                     ValuesIn(all_real_ops), all_ranks)),
                         [](const auto& info) { return info.param.Name(); });

}  // namespace xnnpack