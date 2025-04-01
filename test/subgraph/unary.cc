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
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/reference-utils.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"
#include "test/unary-ops.h"

using ::testing::Combine;
using ::testing::ValuesIn;

namespace xnnpack {

struct Param {
  using TupleT = std::tuple<xnn_datatype, xnn_unary_operator, int>;
  explicit Param(TupleT p)
      : datatype(std::get<0>(p)), op(std::get<1>(p)), rank(std::get<2>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    sstr << xnn_datatype_to_string(datatype) << "_"
         << xnn_unary_operator_to_string(op) << "_rank" << rank;
    return sstr.str();
  }

  xnn_datatype datatype;
  xnn_unary_operator op;
  size_t rank;
};

struct ConvertParam {
  using TupleT = std::tuple<xnn_datatype, xnn_datatype, int>;
  explicit ConvertParam(TupleT p)
      : in(std::get<0>(p)), out(std::get<1>(p)), rank(std::get<2>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    sstr << xnn_datatype_to_string(in) << "_to_" << xnn_datatype_to_string(out)
         << "_rank" << rank;
    return sstr.str();
  }

  xnn_datatype in;
  xnn_datatype out;
  size_t rank;
};

template <typename In, typename Out = In>
void TestImpl(size_t rank, xnn_unary_operator op) {
  const xnn_datatype datatype_in = xnn_datatype_of<In>();
  const xnn_datatype datatype_out = xnn_datatype_of<Out>();

  ReplicableRandomDevice rng;

  // We want the total number of elements to be reasonable, so choose max_dim
  // such that a random shape of rank `rank` produces this max size.
  constexpr size_t max_size = 1024;
  const size_t max_dim = static_cast<size_t>(std::ceil(
      std::pow(static_cast<double>(max_size),
               1.0 / static_cast<double>(std::max<size_t>(1, rank)))));

  const UnaryOpInfo* op_info = GetUnaryOpInfo(op);
  ASSERT_NE(op_info, nullptr);

  xnn_quantization_params input_quantization =
      op_info->InputQuantizationParams(datatype_in);
  xnn_quantization_params output_quantization =
      op_info->OutputQuantizationParams(datatype_out);

  Interval domain = op_info->Domain(datatype_in);
  xnn_unary_params params = op_info->DefaultParams();

  SubgraphTester subgraph(3);
  subgraph.AddInputTensor(rank, datatype_in, input_quantization, 0)
      .AddOutputTensor(rank, datatype_out, output_quantization, 1)
      .AddUnary(op, &params, 0, 1);
  xnn_status status = subgraph.CreateRuntime();
  ASSERT_EQ(status, xnn_status_success);

  for (int reshape = 0; reshape < 2; ++reshape) {
    std::vector<size_t> shape = random_shape(rng, rank, 1, max_dim);

    Tensor<In> input(shape, {XNN_EXTRA_BYTES});
    Tensor<Out> output(shape);

    // TODO(b/397863125): This is a workaround for intrinsics unhandled by msan.
    std::fill_n(input.data(), input.size() + XNN_EXTRA_BYTES / sizeof(In), 0);

    DatatypeGenerator<In> gen(domain.min, domain.max, input_quantization);
    input.generate([&]() { return gen(rng); });

    subgraph.ReshapeExternalTensor(shape, input.data(), 0).ReshapeRuntime();

    ASSERT_EQ(subgraph.GetExternalTensorShape(1), shape);

    subgraph.SetupExternalTensor(output.data(), 1)
        .SetupRuntime()
        .InvokeRuntime();

    for (const auto& i : EnumerateIndices(output.extents())) {
      if (std::is_integral<Out>::value) {
        if (std::is_integral<In>::value) {
          const int32_t expected =
              op_info->ReferenceImpl(static_cast<int32_t>(input(i)), params);
          ASSERT_EQ(expected, output(i))
              << "i = " << index_to_string(i)
              << ", input(i) = " << static_cast<int32_t>(input(i));
        } else {
          // Integral output, non-integral input. We need to potentially
          // dequantize the input, and avoid UB when converting to int.
          const float input_i = dequantize(input(i), input_quantization);
          const int32_t expected =
              round_float_to_int<Out>(op_info->ReferenceImpl(input_i, params));
          ASSERT_EQ(expected, output(i))
              << "i = " << index_to_string(i) << ", input(i) = " << input_i
              << " (" << static_cast<float>(input(i)) << ")";
        }
      } else if (is_quantized<Out>()) {
        const float input_i = dequantize(input(i), input_quantization);
        float expected = op_info->ReferenceImpl(input_i, params);
        expected = fake_quantize(expected, output_quantization);
        expected = std::max<float>(expected, NumericLimits<Out>::min());
        expected = std::min<float>(expected, NumericLimits<Out>::max());
        if (std::isnan(expected)) {
          // This is expected to overflow.
        } else {
          ASSERT_NEAR(expected, output(i), 1)
              << "i = " << index_to_string(i) << ", input(i) = " << input_i
              << " (" << static_cast<float>(input(i)) << ")"
              << ", output(i) = " << static_cast<int32_t>(output(i));
        }
      } else {
        const float input_i = dequantize(input(i), input_quantization);
        float expected = op_info->ReferenceImpl(input_i, params);
        // Force overflow to infinity if that is what should happen.
        expected = static_cast<float>(static_cast<Out>(expected));
        if (std::abs(expected) < NumericLimits<Out>::smallest_normal()) {
          // Flush denormals to 0
          expected = 0.0f;
        }
        if (op_info->IsInSupportedRange(expected)) {
          if (std::isnan(static_cast<float>(expected))) {
            ASSERT_TRUE(std::isnan(static_cast<float>(output(i))));
          } else {
            ASSERT_NEAR(expected, output(i),
                        op_info->Tolerance(expected, datatype_out))
                << "i = " << index_to_string(i) << ", input(i) = " << input_i
                << " (" << static_cast<float>(input(i)) << ")";
          }
        }
      }
    }
  }
}

class IntegerOps : public testing::TestWithParam<Param> {};
class RealOps : public testing::TestWithParam<Param> {};
class Convert : public testing::TestWithParam<ConvertParam> {};

TEST_P(IntegerOps, test) {
  switch (GetParam().datatype) {
    case xnn_datatype_int32:
      TestImpl<int>(GetParam().rank, GetParam().op);
      break;
    default:
      XNN_UNREACHABLE;
  }
}

TEST_P(RealOps, test) {
  switch (GetParam().datatype) {
    case xnn_datatype_qint8:
      TestImpl<quantized<int8_t>>(GetParam().rank, GetParam().op);
      break;
    case xnn_datatype_quint8:
      TestImpl<quantized<uint8_t>>(GetParam().rank, GetParam().op);
      break;
    case xnn_datatype_fp16:
      TestImpl<xnn_float16>(GetParam().rank, GetParam().op);
      break;
    case xnn_datatype_bf16:
      TestImpl<xnn_bfloat16>(GetParam().rank, GetParam().op);
      break;
    case xnn_datatype_fp32:
      TestImpl<float>(GetParam().rank, GetParam().op);
      break;
    default:
      XNN_UNREACHABLE;
  }
}

template <typename Out>
void ConvertImpl(size_t rank, xnn_datatype in) {
  switch (in) {
    case xnn_datatype_int32:
      TestImpl<int32_t, Out>(rank, xnn_unary_convert);
      break;
    case xnn_datatype_qint8:
      TestImpl<quantized<int8_t>, Out>(rank, xnn_unary_convert);
      break;
    case xnn_datatype_quint8:
      TestImpl<quantized<uint8_t>, Out>(rank, xnn_unary_convert);
      break;
    case xnn_datatype_fp16:
      TestImpl<xnn_float16, Out>(rank, xnn_unary_convert);
      break;
    case xnn_datatype_bf16:
      TestImpl<xnn_bfloat16, Out>(rank, xnn_unary_convert);
      break;
    case xnn_datatype_fp32:
      TestImpl<float, Out>(rank, xnn_unary_convert);
      break;
    default:
      XNN_UNREACHABLE;
  }
}

TEST_P(Convert, test) {
  switch (GetParam().out) {
    case xnn_datatype_int32:
      ConvertImpl<int32_t>(GetParam().rank, GetParam().in);
      break;
    case xnn_datatype_qint8:
      ConvertImpl<quantized<int8_t>>(GetParam().rank, GetParam().in);
      break;
    case xnn_datatype_quint8:
      ConvertImpl<quantized<uint8_t>>(GetParam().rank, GetParam().in);
      break;
    case xnn_datatype_fp16:
      ConvertImpl<xnn_float16>(GetParam().rank, GetParam().in);
      break;
    case xnn_datatype_bf16:
      ConvertImpl<xnn_bfloat16>(GetParam().rank, GetParam().in);
      break;
    case xnn_datatype_fp32:
      ConvertImpl<float>(GetParam().rank, GetParam().in);
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

const xnn_datatype all_datatypes[] = {
    xnn_datatype_int32,
    xnn_datatype_quint8,
    xnn_datatype_qint8,
#ifndef XNN_EXCLUDE_F16_TESTS
    xnn_datatype_fp16,
#endif
    xnn_datatype_bf16,
    xnn_datatype_fp32,
};

const xnn_unary_operator all_integer_ops[] = {
    xnn_unary_clamp,
    xnn_unary_abs,
    xnn_unary_negate,
    xnn_unary_square,
    xnn_unary_count_leading_zeros,
    xnn_unary_bitwise_not,
    xnn_unary_popcount,
    xnn_unary_sign,
};

const xnn_unary_operator all_real_ops[] = {
    xnn_unary_clamp,
    xnn_unary_abs,
    xnn_unary_bankers_rounding,
    xnn_unary_ceiling,
    xnn_unary_elu,
    xnn_unary_exp,
    xnn_unary_floor,
    xnn_unary_gelu,
    xnn_unary_hardswish,
    xnn_unary_leaky_relu,
    xnn_unary_log,
    xnn_unary_negate,
    xnn_unary_sigmoid,
    xnn_unary_square,
    xnn_unary_square_root,
    xnn_unary_reciprocal_square_root,
    xnn_unary_tanh,
    xnn_unary_cube_root,
    xnn_unary_cosine,
    xnn_unary_sine,
    xnn_unary_sign,
};
// clang-format on

auto all_ranks = testing::Range(0, XNN_MAX_TENSOR_DIMS);

INSTANTIATE_TEST_SUITE_P(UnaryTest, IntegerOps,
                         testing::ConvertGenerator<Param::TupleT>(
                             Combine(ValuesIn(all_integer_datatypes),
                                     ValuesIn(all_integer_ops), all_ranks)),
                         [](const auto& info) { return info.param.Name(); });

INSTANTIATE_TEST_SUITE_P(UnaryTest, RealOps,
                         testing::ConvertGenerator<Param::TupleT>(
                             Combine(ValuesIn(all_real_datatypes),
                                     ValuesIn(all_real_ops), all_ranks)),
                         [](const auto& info) { return info.param.Name(); });

INSTANTIATE_TEST_SUITE_P(UnaryTest, Convert,
                         testing::ConvertGenerator<ConvertParam::TupleT>(
                             Combine(ValuesIn(all_datatypes),
                                     ValuesIn(all_datatypes), all_ranks)),
                         [](const auto& info) { return info.param.Name(); });

}  // namespace xnnpack
