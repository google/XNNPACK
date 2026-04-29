/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "litert/tensor/backends/xnnpack/arithmetic.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "include/xnnpack.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "litert/tensor/arithmetic_graph.h"
#include "litert/tensor/backends/xnnpack/utils.h"  // IWYU pragma: keep
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/utils/macros.h"

namespace litert::tensor::graph {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

template <Type... Types>
absl::Status ValidateTensorType(const graph::Tensor& tensor,
                                absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  if (!((info.type == Types) || ...)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s only supports %v tensors. Got type id %v.", op_name,
                        absl::StrJoin({Types...}, ", "), info.type));
  }
  return absl::OkStatus();
}

// TODO: b/493560478 - Decide if this needs to be removed.
[[maybe_unused]]
absl::Status ValidateFp32OrQuantizedConstantWeights(const graph::Tensor& tensor,
                                                    absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  if (info.type == Type::kFP32) {
    return absl::OkStatus();
  }
  if (info.type != Type::kI8) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s only supports FP32 weights or quantized INT8 "
                        "constant weights. Got type id %d.",
                        op_name, static_cast<int>(info.type)));
  }
  if (info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s INT8 weights must be constant tensors.", op_name));
  }
  if (info.quantization == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s INT8 weights require quantization metadata.", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& quantization,
      info.quantization->As<const PerChannelAffineQuantization>());
  if (quantization.scales.empty() || quantization.zero_points.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s INT8 weights require non-empty scales and "
                        "zero-points.",
                        op_name));
  }
  return absl::OkStatus();
}

struct BinaryIOIds {
  uint32_t lhs;
  uint32_t rhs;
  uint32_t output;
};

struct ActivationBounds {
  float output_min;
  float output_max;
};

absl::StatusOr<ActivationBounds> GetActivationBounds(
    FusedActivation activation, absl::string_view op_name) {
  ActivationBounds b;
  switch (activation) {
    case kActNone:
      b.output_min = -kInf;
      b.output_max = kInf;
      break;
    case kActRelu:
      b.output_min = 0.0f;
      b.output_max = kInf;
      break;
    case kActRelu6:
      b.output_min = 0.0f;
      b.output_max = 6.0f;
      break;
    case kActReluN1To1:
      b.output_min = -1.0f;
      b.output_max = 1.0f;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("%s: fused activation %d not supported in XNNPACK",
                          op_name, static_cast<int>(activation)));
  }
  return b;
}

absl::StatusOr<BinaryIOIds> PrepareBinaryIO(const Operation& op,
                                            XnnpackBuildContext& ctx,
                                            absl::string_view op_name) {
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects two inputs", op_name));
  }
  const graph::Tensor& lhs = op.inputs[0];
  const graph::Tensor& rhs = op.inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t lhs_id, ctx.DefineValue(lhs));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t rhs_id, ctx.DefineValue(rhs));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  return BinaryIOIds{lhs_id, rhs_id, output_id};
}

absl::StatusOr<xnn_binary_params> BuildBinaryParams(FusedActivation activation,
                                                    absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bounds,
                              GetActivationBounds(activation, op_name));
  xnn_binary_params params;
  params.output_min = bounds.output_min;
  params.output_max = bounds.output_max;
  return params;
}

struct PaddingValues {
  uint32_t top;
  uint32_t right;
  uint32_t bottom;
  uint32_t left;
};

PaddingValues ComputePadding(Padding padding, int input_h, int input_w,
                             int filter_h, int filter_w, int stride_h,
                             int stride_w, int dilation_h, int dilation_w) {
  PaddingValues p{0, 0, 0, 0};
  if (padding == kPaddingSame) {
    const int eff_filter_h = (filter_h - 1) * dilation_h + 1;
    const int eff_filter_w = (filter_w - 1) * dilation_w + 1;
    const int out_h = static_cast<int>(
        std::ceil(static_cast<float>(input_h) / static_cast<float>(stride_h)));
    const int out_w = static_cast<int>(
        std::ceil(static_cast<float>(input_w) / static_cast<float>(stride_w)));
    const int pad_h =
        std::max(0, (out_h - 1) * stride_h + eff_filter_h - input_h);
    const int pad_w =
        std::max(0, (out_w - 1) * stride_w + eff_filter_w - input_w);
    p.top = pad_h / 2;
    p.bottom = pad_h - p.top;
    p.left = pad_w / 2;
    p.right = pad_w - p.left;
  }
  return p;
}

struct TransposeConvPaddingValues {
  uint32_t top;
  uint32_t right;
  uint32_t bottom;
  uint32_t left;
  uint32_t adj_h;
  uint32_t adj_w;
};

absl::StatusOr<TransposeConvPaddingValues> ComputeTransposeConvPadding(
    Padding padding, int input_h, int input_w, int filter_h, int filter_w,
    int stride_h, int stride_w, int output_h, int output_w) {
  if (input_h <= 0 || input_w <= 0 || filter_h <= 0 || filter_w <= 0 ||
      stride_h <= 0 || stride_w <= 0 || output_h <= 0 || output_w <= 0) {
    return absl::InvalidArgumentError(
        "TransposeConv expects positive input/filter/stride/output sizes");
  }

  auto compute_dim = [&](int input, int filter, int stride, int output,
                         uint32_t* pad_before, uint32_t* pad_after,
                         uint32_t* adj,
                         absl::string_view dim_name) -> absl::Status {
    const int base = (input - 1) * stride + filter;
    int pad_total = 0;
    int adj_local = 0;
    if (output >= base) {
      adj_local = output - base;
    } else {
      pad_total = base - output;
    }
    if (adj_local >= stride) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TransposeConv %s adjustment (%d) must be < stride (%d)", dim_name,
          adj_local, stride));
    }
    if (padding == kPaddingValid && pad_total != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TransposeConv %s padding (%d) not allowed for VALID padding",
          dim_name, pad_total));
    }
    const int pad_before_local = pad_total / 2;
    const int pad_after_local = pad_total - pad_before_local;
    *pad_before = static_cast<uint32_t>(pad_before_local);
    *pad_after = static_cast<uint32_t>(pad_after_local);
    *adj = static_cast<uint32_t>(adj_local);
    return absl::OkStatus();
  };

  TransposeConvPaddingValues p{0, 0, 0, 0, 0, 0};
  LRT_TENSOR_RETURN_IF_ERROR(compute_dim(input_h, filter_h, stride_h, output_h,
                                         &p.top, &p.bottom, &p.adj_h,
                                         "height"));
  LRT_TENSOR_RETURN_IF_ERROR(compute_dim(input_w, filter_w, stride_w, output_w,
                                         &p.left, &p.right, &p.adj_w, "width"));
  return p;
}

absl::Status AddBinaryNode(xnn_binary_operator op_type, const BinaryIOIds& io,
                           const xnn_binary_params& params,
                           XnnpackBuildContext& ctx,
                           absl::string_view op_name) {
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_binary(ctx.subgraph(), op_type, &params,
                                               io.lhs, io.rhs, io.output,
                                               /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

struct UnaryIOIds {
  uint32_t input;
  uint32_t output;
};

absl::StatusOr<UnaryIOIds> PrepareUnaryIO(const Operation& op,
                                          XnnpackBuildContext& ctx,
                                          absl::string_view op_name) {
  if (op.inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects one input", op_name));
  }
  const graph::Tensor& input = op.inputs[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  return UnaryIOIds{input_id, output_id};
}

absl::Status AddUnaryNode(xnn_unary_operator op_type, const UnaryIOIds& io,
                          XnnpackBuildContext& ctx, absl::string_view op_name,
                          const xnn_unary_params* params = nullptr) {
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_unary(ctx.subgraph(), op_type, params,
                                              io.input, io.output,
                                              /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

}  // namespace

absl::Status OpMixin<AddOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();

  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareBinaryIO(*this, ctx, op_name));
  FusedActivation activation = kActNone;
  if (const auto* op =
          dynamic_cast<const AddOperation<XnnpackMixinTag>*>(this)) {
    activation = op->activation;
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto params,
                              BuildBinaryParams(activation, op_name));
  return AddBinaryNode(xnn_binary_add, io, params, ctx, op_name);
}

absl::Status OpMixin<MulOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareBinaryIO(*this, ctx, op_name));
  FusedActivation activation = kActNone;
  if (const auto* op =
          dynamic_cast<const MulOperation<XnnpackMixinTag>*>(this)) {
    activation = op->activation;
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto params,
                              BuildBinaryParams(activation, op_name));
  return AddBinaryNode(xnn_binary_multiply, io, params, ctx, op_name);
}

absl::Status OpMixin<SubOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareBinaryIO(*this, ctx, op_name));
  FusedActivation activation = kActNone;
  if (const auto* op =
          dynamic_cast<const SubOperation<XnnpackMixinTag>*>(this)) {
    activation = op->activation;
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto params,
                              BuildBinaryParams(activation, op_name));
  return AddBinaryNode(xnn_binary_subtract, io, params, ctx, op_name);
}

absl::Status OpMixin<DivOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareBinaryIO(*this, ctx, op_name));
  FusedActivation activation = kActNone;
  if (const auto* op =
          dynamic_cast<const DivOperation<XnnpackMixinTag>*>(this)) {
    activation = op->activation;
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto params,
                              BuildBinaryParams(activation, op_name));
  return AddBinaryNode(xnn_binary_divide, io, params, ctx, op_name);
}

absl::Status OpMixin<MaximumOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareBinaryIO(*this, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto params,
                              BuildBinaryParams(kActNone, op_name));
  return AddBinaryNode(xnn_binary_maximum, io, params, ctx, op_name);
}

absl::Status OpMixin<MinimumOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareBinaryIO(*this, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto params,
                              BuildBinaryParams(kActNone, op_name));
  return AddBinaryNode(xnn_binary_minimum, io, params, ctx, op_name);
}

absl::Status OpMixin<PowOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareBinaryIO(*this, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto params,
                              BuildBinaryParams(kActNone, op_name));
  return AddBinaryNode(xnn_binary_pow, io, params, ctx, op_name);
}

absl::Status OpMixin<AbsOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_abs, io, ctx, op_name);
}

absl::Status OpMixin<ReluOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  xnn_unary_params params;
  params.clamp.min = 0.0f;
  params.clamp.max = std::numeric_limits<float>::infinity();
  return AddUnaryNode(xnn_unary_clamp, io, ctx, op_name, &params);
}

absl::Status OpMixin<Relu6OperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  xnn_unary_params params;
  params.clamp.min = 0.0f;
  params.clamp.max = 6.0f;
  return AddUnaryNode(xnn_unary_clamp, io, ctx, op_name, &params);
}

absl::Status OpMixin<LeakyReluOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  auto data = dynamic_cast<const LeakyReluOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  xnn_unary_params params;
  params.leaky_relu.negative_slope = data->alpha;
  return AddUnaryNode(xnn_unary_leaky_relu, io, ctx, op_name, &params);
}

absl::Status OpMixin<EluOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  xnn_unary_params params;
  params.elu.alpha = 1.0f;
  return AddUnaryNode(xnn_unary_elu, io, ctx, op_name, &params);
}

absl::Status OpMixin<HardSwishOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_hardswish, io, ctx, op_name);
}

absl::Status OpMixin<PReluOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareBinaryIO(*this, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto params,
                              BuildBinaryParams(kActNone, op_name));
  return AddBinaryNode(xnn_binary_prelu, io, params, ctx, op_name);
}

absl::Status OpMixin<L2NormalizationOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "L2Normalization is not supported in XNNPACK.");
}

absl::Status OpMixin<SquareOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_square, io, ctx, op_name);
}

absl::Status OpMixin<RsqrtOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_reciprocal_square_root, io, ctx, op_name);
}

absl::Status OpMixin<SqrtOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_square_root, io, ctx, op_name);
}

absl::Status OpMixin<ExpOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_exp, io, ctx, op_name);
}

absl::Status OpMixin<LogOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_log, io, ctx, op_name);
}

absl::Status OpMixin<CeilOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_ceiling, io, ctx, op_name);
}

absl::Status OpMixin<FloorOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_floor, io, ctx, op_name);
}

absl::Status OpMixin<SignOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_sign, io, ctx, op_name);
}

absl::Status OpMixin<RoundOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_bankers_rounding, io, ctx, op_name);
}

absl::Status OpMixin<NegOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_negate, io, ctx, op_name);
}

absl::Status OpMixin<TanhOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_tanh, io, ctx, op_name);
}

absl::Status OpMixin<LogisticOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_sigmoid, io, ctx, op_name);
}

absl::Status OpMixin<CosOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_cosine, io, ctx, op_name);
}

absl::Status OpMixin<CastOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_convert, io, ctx, op_name);
}

absl::Status OpMixin<SinOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  return AddUnaryNode(xnn_unary_sine, io, ctx, op_name);
}

absl::Status OpMixin<GeluOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  bool approximate = false;
  if (const auto* op =
          dynamic_cast<const GeluOperation<XnnpackMixinTag>*>(this)) {
    approximate = op->approximate;
  }
  const xnn_unary_operator op_type =
      approximate ? xnn_unary_approxgelu : xnn_unary_gelu;
  return AddUnaryNode(op_type, io, ctx, op_name);
}

absl::Status OpMixin<SoftmaxOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  float beta_value = 1.0f;
  if (const auto* op =
          dynamic_cast<const SoftmaxOperation<XnnpackMixinTag>*>(this)) {
    beta_value = op->beta;
  }
  if (beta_value != 1.0f) {
    return absl::UnimplementedError(absl::StrFormat(
        "%s: XNNPACK softmax only supports beta == 1", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto io, PrepareUnaryIO(*this, ctx, op_name));
  LRT_TENSOR_RETURN_IF_ERROR(
      xnn_define_softmax(ctx.subgraph(), io.input, io.output, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<AveragePool2DOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "AveragePool2D";
  const auto* op_data =
      dynamic_cast<const AveragePool2DOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("AveragePool2D op missing parameters");
  }
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 input", op_name));
  }
  const graph::Tensor& input = inputs[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = op_data->filter_height;
  const int filter_w = op_data->filter_width;
  const int stride_h = op_data->stride_h;
  const int stride_w = op_data->stride_w;

  const PaddingValues pad =
      ComputePadding(op_data->padding, input_h, input_w, filter_h, filter_w,
                     stride_h, stride_w, /*dilation_h=*/1, /*dilation_w=*/1);

  FusedActivation activation = op_data->activation;
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bounds,
                              GetActivationBounds(activation, op_name));

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_average_pooling_2d(
      ctx.subgraph(), pad.top, pad.right, pad.bottom, pad.left, filter_h,
      filter_w, stride_h, stride_w, bounds.output_min, bounds.output_max,
      input_id, output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<MaxPool2DOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "MaxPool2D";
  const auto* op_data =
      dynamic_cast<const MaxPool2DOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("MaxPool2D op missing parameters");
  }
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 input", op_name));
  }
  const graph::Tensor& input = inputs[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = op_data->filter_height;
  const int filter_w = op_data->filter_width;
  const int stride_h = op_data->stride_h;
  const int stride_w = op_data->stride_w;

  const PaddingValues pad =
      ComputePadding(op_data->padding, input_h, input_w, filter_h, filter_w,
                     stride_h, stride_w, /*dilation_h=*/1, /*dilation_w=*/1);

  FusedActivation activation = op_data->activation;
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bounds,
                              GetActivationBounds(activation, op_name));

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_max_pooling_2d(
      ctx.subgraph(), pad.top, pad.right, pad.bottom, pad.left, filter_h,
      filter_w, stride_h, stride_w, /*dilation_height=*/1, /*dilation_width=*/1,
      bounds.output_min, bounds.output_max, input_id, output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<Conv2DOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Conv2D";
  const auto* op_data =
      dynamic_cast<const Conv2DOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("Conv2D op missing parameters");
  }
  if (inputs.size() < 2 || inputs.size() > 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 2 or 3 inputs (input, filter[, bias])", op_name));
  }
  const graph::Tensor& input = inputs[0];
  const graph::Tensor& filter = inputs[1];
  const graph::Tensor* bias = inputs.size() == 3 ? &inputs[2] : nullptr;

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t filter_id, ctx.DefineValue(filter));
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (bias != nullptr) {
    LRT_TENSOR_ASSIGN_OR_RETURN(bias_id, ctx.DefineValue(*bias));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& filter_info, graph::GetInfo(filter));

  // The 0th dimension is always the batch dimension in the tensor API (BHWC).
  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = filter_info.shape[1];
  const int filter_w = filter_info.shape[2];
  const int stride_h = op_data->stride_h;
  const int stride_w = op_data->stride_w;
  const int dilation_h = op_data->dilation_h_factor;
  const int dilation_w = op_data->dilation_w_factor;

  const PaddingValues pad =
      ComputePadding(op_data->padding, input_h, input_w, filter_h, filter_w,
                     stride_h, stride_w, dilation_h, dilation_w);

  const size_t group_input_channels = filter_info.shape[3];
  const size_t group_output_channels = filter_info.shape[0];

  FusedActivation activation = op_data->activation;
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bounds,
                              GetActivationBounds(activation, op_name));

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_convolution_2d(
      ctx.subgraph(), pad.top, pad.right, pad.bottom, pad.left, filter_h,
      filter_w, stride_h, stride_w, dilation_h, dilation_w,
      /*groups=*/1, group_input_channels, group_output_channels,
      bounds.output_min, bounds.output_max, input_id, filter_id, bias_id,
      output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<DepthwiseConv2DOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "DepthwiseConv2D";
  const auto* op_data =
      dynamic_cast<const DepthwiseConv2DOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("DepthwiseConv2D op missing parameters");
  }
  if (inputs.size() < 2 || inputs.size() > 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 2 or 3 inputs (input, filter[, bias])", op_name));
  }
  const graph::Tensor& input = inputs[0];
  const graph::Tensor& filter = inputs[1];
  const graph::Tensor* bias = inputs.size() == 3 ? &inputs[2] : nullptr;

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t filter_id, ctx.DefineValue(filter));
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (bias != nullptr) {
    LRT_TENSOR_ASSIGN_OR_RETURN(bias_id, ctx.DefineValue(*bias));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& filter_info, graph::GetInfo(filter));

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = filter_info.shape[1];
  const int filter_w = filter_info.shape[2];
  const int stride_h = op_data->stride_h;
  const int stride_w = op_data->stride_w;
  const int dilation_h = op_data->dilation_h_factor;
  const int dilation_w = op_data->dilation_w_factor;

  const PaddingValues pad =
      ComputePadding(op_data->padding, input_h, input_w, filter_h, filter_w,
                     stride_h, stride_w, dilation_h, dilation_w);

  const size_t input_channels = input_info.shape[3];
  const int depth_multiplier = op_data->depth_multiplier;

  if (filter_info.shape[3] != input_channels * depth_multiplier) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s filter channels (%d) != input_channels (%d) * "
        "depth_multiplier (%d)",
        op_name, filter_info.shape[3], input_channels, depth_multiplier));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto bounds, GetActivationBounds(op_data->activation, op_name));

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_depthwise_convolution_2d(
      ctx.subgraph(), pad.top, pad.right, pad.bottom, pad.left, filter_h,
      filter_w, stride_h, stride_w, dilation_h, dilation_w, depth_multiplier,
      input_channels, bounds.output_min, bounds.output_max, input_id, filter_id,
      bias_id, output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<FullyConnectedOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  const absl::string_view op_name = GetName();
  const auto* op_data =
      dynamic_cast<const FullyConnectedOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("FullyConnected op missing parameters");
  }
  if (inputs.size() < 2 || inputs.size() > 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 2 or 3 inputs (input, weights[, bias])", op_name));
  }
  const graph::Tensor& input = inputs[0];
  const graph::Tensor& weights = inputs[1];
  const graph::Tensor* bias = inputs.size() == 3 ? &inputs[2] : nullptr;

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));
  uint32_t fc_input_id = input_id;

  auto weights_info_or = graph::GetInfo(weights);
  auto input_info_or = graph::GetInfo(input);
  bool weights_are_quantized = false;
  if (weights_info_or.ok() && weights_info_or->type == Type::kI8) {
    if (weights_info_or->quantization) {
      if (auto pcq_or =
              weights_info_or->quantization->As<PerChannelAffineQuantization>();
          pcq_or.ok()) {
        bool all_zeros = true;
        for (int64_t zp : pcq_or->zero_points) {
          if (zp != 0) {
            all_zeros = false;
            break;
          }
        }
        weights_are_quantized = all_zeros;
      } else {
        weights_are_quantized = true;
      }
    }
  }

  if (weights_are_quantized && input_info_or.ok() &&
      input_info_or->type == Type::kFP32) {
    uint32_t qd_id = XNN_INVALID_VALUE_ID;
    std::vector<size_t> dims(input_info_or->shape.begin(),
                             input_info_or->shape.end());
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_dynamically_quantized_tensor_value(
        ctx.subgraph(), xnn_datatype_qdint8, dims.size(),
        /*num_nonbatch_dims=*/1, dims.data(), XNN_INVALID_VALUE_ID, 0, &qd_id))
        << "Could not define dynamically quantized tensor.";

    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_unary(
        ctx.subgraph(), xnn_unary_convert, /*params=*/nullptr, input_id, qd_id,
        /*flags=*/0))
        << "Could not define convert node for dynamic quantization.";

    fc_input_id = qd_id;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t weights_id, ctx.DefineValue(weights));
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (bias != nullptr) {
    LRT_TENSOR_ASSIGN_OR_RETURN(bias_id, ctx.DefineValue(*bias));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));
  FusedActivation activation = op_data->activation;
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bounds,
                              GetActivationBounds(activation, op_name));

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_fully_connected(
      ctx.subgraph(), bounds.output_min, bounds.output_max, fc_input_id,
      weights_id, bias_id, output_id,
      /*flags=*/0))
      << "xnn_define_fully_connected failed";
  return absl::OkStatus();
}

absl::Status OpMixin<BatchMatMulOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "BatchMatMul";
  const auto* op_data =
      dynamic_cast<const BatchMatMulOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("BatchMatMul op missing parameters");
  }

  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }

  const graph::Tensor& input1 = inputs[0];
  const graph::Tensor& input2 = inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input1_id, ctx.DefineValue(input1));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input2_id, ctx.DefineValue(input2));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  uint32_t flags = 0;
  if (op_data->adj_x) {
    flags |= XNN_FLAG_TRANSPOSE_A;
  }
  if (op_data->adj_y) {
    flags |= XNN_FLAG_TRANSPOSE_B;
  }

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_batch_matrix_multiply(
      ctx.subgraph(), input1_id, input2_id, output_id, flags))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<TransposeOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Transpose";

  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, perm)", op_name));
  }

  const graph::Tensor& input = inputs[0];
  const graph::Tensor& perm_tensor = inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  // Get permutation from const tensor
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& perm_info,
                              graph::GetInfo(perm_tensor));
  if (perm_info.type != Type::kI32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: permutation must be int32. Got type id %d.",
                        op_name, static_cast<int>(perm_info.type)));
  }
  if (perm_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: permutation must be a constant tensor", op_name));
  }
  auto locked = perm_info.buffer->Lock();
  const int32_t* perm_data = reinterpret_cast<const int32_t*>(locked.data());
  if (locked.size() % sizeof(int32_t) != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: permutation buffer size (%d) is not a multiple of int32 size",
        op_name, locked.size()));
  }
  size_t num_dims = locked.size() / sizeof(int32_t);

  std::vector<size_t> perm(num_dims);
  std::copy(perm_data, perm_data + num_dims, perm.begin());

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_transpose(
      ctx.subgraph(), num_dims, perm.data(), input_id, output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<MeanOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Mean";
  const auto* op_data =
      dynamic_cast<const MeanOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("Mean op missing parameters");
  }

  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, axes)", op_name));
  }

  const graph::Tensor& input = inputs[0];
  const graph::Tensor& axes_tensor = inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  // Get reduction axes from const tensor
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axes_info,
                              graph::GetInfo(axes_tensor));
  if (axes_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axes must be a constant tensor", op_name));
  }
  LockedBufferSpan<const int32_t> axes_data =
      axes_info.buffer->Lock().As<const int32_t>();
  const size_t num_axes = axes_info.shape[0];
  const std::vector<int64_t> axes(axes_data.begin(), axes_data.end());
  // Set XNN_FLAG_KEEP_DIMS if keep_dims is true
  const uint32_t flags = op_data->keep_dims ? XNN_FLAG_KEEP_DIMS : 0;

  LRT_TENSOR_RETURN_IF_ERROR(
      xnn_define_static_reduce_v2(ctx.subgraph(), xnn_reduce_mean, num_axes,
                                  axes.data(), input_id, output_id, flags))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SliceOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Slice";

  if (inputs.size() != 3) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 3 inputs (input, begin, size)", op_name));
  }

  const graph::Tensor& input = inputs[0];
  const graph::Tensor& begin_tensor = inputs[1];
  const graph::Tensor& size_tensor = inputs[2];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  // Get begin offsets from const tensor
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& begin_info,
                              graph::GetInfo(begin_tensor));
  if (begin_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: begin must be a constant tensor", op_name));
  }
  auto begin_locked = begin_info.buffer->Lock();
  const int32_t* begin_data =
      reinterpret_cast<const int32_t*>(begin_locked.data());
  size_t num_dims = begin_info.shape[0];

  // Get sizes from const tensor
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& size_info,
                              graph::GetInfo(size_tensor));
  if (size_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must be a constant tensor", op_name));
  }
  auto size_locked = size_info.buffer->Lock();
  const int32_t* size_data =
      reinterpret_cast<const int32_t*>(size_locked.data());

  std::vector<size_t> offsets(num_dims);
  std::vector<size_t> sizes(num_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    offsets[i] = static_cast<size_t>(begin_data[i]);
    sizes[i] = static_cast<size_t>(size_data[i]);
  }

  LRT_TENSOR_RETURN_IF_ERROR(
      xnn_define_static_slice(ctx.subgraph(), num_dims, offsets.data(),
                              sizes.data(), input_id, output_id,
                              /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ConcatenationOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Concatenation";
  const auto* op_data =
      dynamic_cast<const ConcatenationOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("Concatenation op missing parameters");
  }

  if (inputs.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects at least 1 input", op_name));
  }

  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto& input : inputs) {
    LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));
    input_ids.push_back(input_id);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_RETURN_IF_ERROR(
      xnn_define_concatenate(ctx.subgraph(), op_data->axis, input_ids.size(),
                             input_ids.data(), output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SqueezeOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Squeeze";

  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 input", op_name));
  }

  const graph::Tensor& input = inputs[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info, graph::GetInfo(output));

  std::vector<size_t> new_shape;
  new_shape.reserve(output_info.shape.size());
  for (int dim : output_info.shape) {
    new_shape.push_back(static_cast<size_t>(dim));
  }

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_reshape(
      ctx.subgraph(), new_shape.size(), new_shape.data(), input_id, output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ExpandDimsOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "ExpandDims";

  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }

  const graph::Tensor& input = inputs[0];
  const graph::Tensor& axis_tensor = inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axis_info,
                              graph::GetInfo(axis_tensor));
  if (axis_info.type != Type::kI32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis must be int32. Got type id %d.", op_name,
                        static_cast<int>(axis_info.type)));
  }
  if (axis_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis must be a constant tensor", op_name));
  }
  auto locked_axis = axis_info.buffer->Lock().As<const int32_t>();
  int real_axis = locked_axis.data()[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  std::vector<size_t> new_shape;
  new_shape.reserve(input_info.shape.size() + 1);
  for (int dim : input_info.shape) {
    new_shape.push_back(static_cast<size_t>(dim));
  }

  if (real_axis < 0) {
    real_axis += input_info.shape.size() + 1;
  }
  new_shape.insert(new_shape.begin() + real_axis, 1);

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_reshape(
      ctx.subgraph(), new_shape.size(), new_shape.data(), input_id, output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ReshapeOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Reshape";
  const auto* op_data =
      dynamic_cast<const ReshapeOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("Reshape op missing parameters");
  }

  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 input", op_name));
  }

  const graph::Tensor& input = inputs[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  std::vector<size_t> new_shape;
  new_shape.reserve(op_data->new_shape.size());
  for (int dim : op_data->new_shape) {
    new_shape.push_back(static_cast<size_t>(dim));
  }

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_reshape(
      ctx.subgraph(), new_shape.size(), new_shape.data(), input_id, output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<TileOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Tile";

  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, multiples)", op_name));
  }

  const graph::Tensor& input = inputs[0];
  const graph::Tensor& multiples_tensor = inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  // Get multiples from const tensor
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& multiples_info,
                              graph::GetInfo(multiples_tensor));
  if (multiples_info.type != Type::kI32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: multiples must be int32. Got type id %d.", op_name,
                        static_cast<int>(multiples_info.type)));
  }
  if (multiples_info.shape.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: multiples must be a 1D tensor, but has rank %d",
                        op_name, multiples_info.shape.size()));
  }
  if (multiples_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: multiples must be a constant tensor", op_name));
  }
  auto locked = multiples_info.buffer->Lock();
  const int32_t* multiples_data =
      reinterpret_cast<const int32_t*>(locked.data());
  if (locked.size() % sizeof(int32_t) != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: multiples buffer size (%d) is not a multiple of int32 size",
        op_name, locked.size()));
  }
  size_t num_dims = locked.size() / sizeof(int32_t);

  // Get input shape and compute output shape
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  if (input_info.shape.size() != num_dims) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: input rank (%d) != multiples size (%d)", op_name,
                        input_info.shape.size(), num_dims));
  }

  std::vector<size_t> new_shape(num_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    int mult = multiples_data[i];
    int dim = input_info.shape[i];
    if (mult > 1 && dim != 1) {
      return absl::UnimplementedError(absl::StrFormat(
          "%s: XNNPACK broadcast can only tile dimensions of size 1. "
          "Dimension %d has size %d but multiples[%d]=%d",
          op_name, i, dim, i, mult));
    }
    new_shape[i] = static_cast<size_t>(dim * mult);
  }

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_broadcast(
      ctx.subgraph(), num_dims, new_shape.data(), input_id, output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ResizeBilinearOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "ResizeBilinear";
  const auto* op_data =
      dynamic_cast<const ResizeBilinearOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("ResizeBilinear op missing parameters");
  }
  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, size)", op_name));
  }

  const graph::Tensor& input = inputs[0];
  const graph::Tensor& size_tensor = inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& size_info,
                              graph::GetInfo(size_tensor));
  if (size_info.type != Type::kI32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must be int32. Got type id %d.", op_name,
                        static_cast<int>(size_info.type)));
  }
  if (size_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must be a constant tensor", op_name));
  }
  auto size_data = size_info.buffer->Lock().As<const int32_t>();
  if (size_data.size() < 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must have at least 2 elements", op_name));
  }
  const int new_height = size_data.data()[0];
  const int new_width = size_data.data()[1];
  if (new_height <= 0 || new_width <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must be positive. Got %d x %d", op_name,
                        new_height, new_width));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  if (input_info.shape.size() != 4) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: input must be 4D (BHWC). Got rank %d", op_name,
                        input_info.shape.size()));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info, graph::GetInfo(output));
  if (output_info.shape.size() != 4 || output_info.shape[1] != new_height ||
      output_info.shape[2] != new_width) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: output shape does not match size (%d x %d)",
                        op_name, new_height, new_width));
  }

  if (op_data->align_corners && op_data->half_pixel_centers) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: align_corners and half_pixel_centers are mutually "
                        "exclusive",
                        op_name));
  }
  uint32_t flags = 0;
  if (op_data->align_corners) {
    flags |= XNN_FLAG_ALIGN_CORNERS;
  } else if (!op_data->half_pixel_centers) {
    flags |= XNN_FLAG_TENSORFLOW_LEGACY_MODE;
  }

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_resize_bilinear_2d(
      ctx.subgraph(), static_cast<size_t>(new_height),
      static_cast<size_t>(new_width), input_id, output_id, flags))
      << op_name;
  return absl::OkStatus();
}

absl::Status
OpMixin<ResizeNearestNeighborOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "ResizeNearestNeighbor";
  const auto* op_data =
      dynamic_cast<const ResizeNearestNeighborOperation<XnnpackMixinTag>*>(
          this);
  if (op_data == nullptr) {
    return absl::InternalError("ResizeNearestNeighbor op missing parameters");
  }
  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, size)", op_name));
  }

  if (op_data->align_corners || op_data->half_pixel_centers) {
    return absl::UnimplementedError(
        absl::StrFormat("%s: XNNPACK only supports align_corners=false and "
                        "half_pixel_centers=false",
                        op_name));
  }

  const graph::Tensor& input = inputs[0];
  const graph::Tensor& size_tensor = inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& size_info,
                              graph::GetInfo(size_tensor));
  if (size_info.type != Type::kI32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must be int32. Got type id %d.", op_name,
                        static_cast<int>(size_info.type)));
  }
  if (size_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must be a constant tensor", op_name));
  }
  auto size_data = size_info.buffer->Lock().As<const int32_t>();
  if (size_data.size() < 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must have at least 2 elements", op_name));
  }
  const int new_height = size_data.data()[0];
  const int new_width = size_data.data()[1];
  if (new_height <= 0 || new_width <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: size must be positive. Got %d x %d", op_name,
                        new_height, new_width));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  if (input_info.shape.size() != 4) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: input must be 4D (BHWC). Got rank %d", op_name,
                        input_info.shape.size()));
  }
  const int batch = input_info.shape[0];
  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int channels = input_info.shape[3];

  if (new_height % input_h != 0 || new_width % input_w != 0) {
    return absl::UnimplementedError(absl::StrFormat(
        "%s: only integer scaling supported. Input %dx%d, output %dx%d",
        op_name, input_h, input_w, new_height, new_width));
  }
  const int scale_h = new_height / input_h;
  const int scale_w = new_width / input_w;
  if (scale_h <= 0 || scale_w <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: invalid scale %d x %d", op_name, scale_h, scale_w));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info, graph::GetInfo(output));
  if (output_info.shape.size() != 4 || output_info.shape[1] != new_height ||
      output_info.shape[2] != new_width) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: output shape does not match size (%d x %d)",
                        op_name, new_height, new_width));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  const std::array<size_t, 6> reshape_dims = {
      static_cast<size_t>(batch), static_cast<size_t>(input_h),
      static_cast<size_t>(1),     static_cast<size_t>(input_w),
      static_cast<size_t>(1),     static_cast<size_t>(channels)};
  const std::array<size_t, 6> broadcast_dims = {
      static_cast<size_t>(batch),   static_cast<size_t>(input_h),
      static_cast<size_t>(scale_h), static_cast<size_t>(input_w),
      static_cast<size_t>(scale_w), static_cast<size_t>(channels)};

  uint32_t reshape_id = XNN_INVALID_VALUE_ID;
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
      ctx.subgraph(), xnn_datatype_fp32, reshape_dims.size(),
      reshape_dims.data(), /*data=*/nullptr, XNN_INVALID_VALUE_ID,
      /*flags=*/0, &reshape_id))
      << op_name;

  uint32_t broadcast_id = XNN_INVALID_VALUE_ID;
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
      ctx.subgraph(), xnn_datatype_fp32, broadcast_dims.size(),
      broadcast_dims.data(), /*data=*/nullptr, XNN_INVALID_VALUE_ID,
      /*flags=*/0, &broadcast_id))
      << op_name;

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_reshape(
      ctx.subgraph(), reshape_dims.size(), reshape_dims.data(), input_id,
      reshape_id, /*flags=*/0))
      << op_name;

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_broadcast(
      ctx.subgraph(), broadcast_dims.size(), broadcast_dims.data(), reshape_id,
      broadcast_id, /*flags=*/0))
      << op_name;

  const std::array<size_t, 4> output_dims = {
      static_cast<size_t>(batch), static_cast<size_t>(new_height),
      static_cast<size_t>(new_width), static_cast<size_t>(channels)};
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_static_reshape(
      ctx.subgraph(), output_dims.size(), output_dims.data(), broadcast_id,
      output_id, /*flags=*/0))
      << op_name;

  return absl::OkStatus();
}

absl::Status OpMixin<GatherOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("Gather is not supported in XNNPACK.");
}

absl::Status OpMixin<SpaceToDepthOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "SpaceToDepth";

  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 input", op_name));
  }

  const graph::Tensor& input = inputs[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  auto data = dynamic_cast<const SpaceToDepthOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_space_to_depth_2d(
      ctx.subgraph(), data->block_size, input_id, output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<DepthToSpaceOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "DepthToSpace";

  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 input", op_name));
  }

  const graph::Tensor& input = inputs[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  auto data = dynamic_cast<const DepthToSpaceOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_depth_to_space_2d(
      ctx.subgraph(), data->block_size, input_id, output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<TransposeConvOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "TransposeConv";
  const auto* op_data =
      dynamic_cast<const TransposeConvOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("TransposeConv op missing parameters");
  }
  if (inputs.size() != 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 3 inputs (output_shape, filter, input)", op_name));
  }

  const graph::Tensor& output_shape = inputs[0];
  const graph::Tensor& filter = inputs[1];
  const graph::Tensor& input = inputs[2];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t filter_id, ctx.DefineValue(filter));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& filter_info, graph::GetInfo(filter));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info, graph::GetInfo(output));

  if (input_info.shape.size() != 4 || filter_info.shape.size() != 4 ||
      output_info.shape.size() != 4) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 4D tensors for input, filter, and output", op_name));
  }

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = filter_info.shape[1];
  const int filter_w = filter_info.shape[2];
  const int output_h = output_info.shape[1];
  const int output_w = output_info.shape[2];
  const int stride_h = op_data->stride_h;
  const int stride_w = op_data->stride_w;

  const size_t group_input_channels = filter_info.shape[3];
  const size_t group_output_channels = filter_info.shape[0];
  if (input_info.shape[3] != static_cast<int>(group_input_channels)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s input channels (%d) != filter input channels (%d)",
                        op_name, input_info.shape[3], group_input_channels));
  }
  if (output_info.shape[3] != static_cast<int>(group_output_channels)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s output channels (%d) != filter output channels (%d)", op_name,
        output_info.shape[3], group_output_channels));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_shape_info,
                              graph::GetInfo(output_shape));
  if (output_shape_info.buffer == nullptr ||
      output_shape_info.type != Type::kI32) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: output_shape must be a constant int32 tensor", op_name));
  }
  auto output_shape_data = output_shape_info.buffer->Lock().As<const int32_t>();
  if (output_shape_data.size() >= 4) {
    if (output_shape_data.data()[1] != output_h ||
        output_shape_data.data()[2] != output_w ||
        output_shape_data.data()[3] != output_info.shape[3]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s: output_shape does not match output tensor shape", op_name));
    }
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& padding,
      ComputeTransposeConvPadding(op_data->padding, input_h, input_w, filter_h,
                                  filter_w, stride_h, stride_w, output_h,
                                  output_w));

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_deconvolution_2d(
      ctx.subgraph(), padding.top, padding.right, padding.bottom, padding.left,
      padding.adj_h, padding.adj_w, filter_h, filter_w, stride_h, stride_w,
      /*dilation_height=*/1, /*dilation_width=*/1, /*groups=*/1,
      group_input_channels, group_output_channels, /*output_min=*/-kInf,
      /*output_max=*/kInf, input_id, filter_id, XNN_INVALID_VALUE_ID, output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<TransposeConv2DOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "TransposeConv2D";
  const auto* op_data =
      dynamic_cast<const TransposeConv2DOperation<XnnpackMixinTag>*>(this);
  if (op_data == nullptr) {
    return absl::InternalError("TransposeConv2D op missing parameters");
  }
  if (inputs.size() < 3) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 3 inputs, got %d", op_name, inputs.size()));
  }

  const graph::Tensor& filter = inputs[1];
  const graph::Tensor& input = inputs[2];
  const graph::Tensor* bias = inputs.size() > 3 ? &inputs[3] : nullptr;

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& filter_info, graph::GetInfo(filter));

  if (filter_info.buffer == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s requires constant weights in XNNPACK backend.", op_name));
  }
  if (bias != nullptr) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& bias_info, graph::GetInfo(*bias));
    if (bias_info.buffer == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s requires constant bias in XNNPACK backend.", op_name));
    }
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  const graph::Tensor& output = outputs.front();
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(output));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info, graph::GetInfo(output));

  const size_t output_channels = filter_info.shape[0];
  const size_t input_channels = filter_info.shape[3];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t filter_id, ctx.DefineValue(filter));
  uint32_t bias_id = XNN_INVALID_VALUE_ID;

  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& padding,
      ComputeTransposeConvPadding(
          op_data->padding, input_info.shape[1], input_info.shape[2],
          filter_info.shape[1], filter_info.shape[2], op_data->stride_h,
          op_data->stride_w, output_info.shape[1], output_info.shape[2]));

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_deconvolution_2d(
      ctx.subgraph(), padding.top, padding.right, padding.bottom, padding.left,
      padding.adj_h, padding.adj_w, filter_info.shape[1], filter_info.shape[2],
      op_data->stride_h, op_data->stride_w, /*dilation_height=*/1,
      /*dilation_width=*/1, /*groups=*/1, input_channels, output_channels,
      /*output_min=*/-kInf, /*output_max=*/kInf, input_id, filter_id, bias_id,
      output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SplitOperationTag, XnnpackMixinTag>::ToXnnpack(
    XnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Split";

  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }

  const graph::Tensor& input = inputs[0];
  const graph::Tensor& axis_tensor = inputs[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));

  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*this));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }

  auto data = dynamic_cast<const SplitOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  if (outputs.size() != data->num_splits) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects %d outputs, but got %d", op_name,
                        data->num_splits, outputs.size()));
  }

  std::vector<uint32_t> output_ids(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    LRT_TENSOR_ASSIGN_OR_RETURN(output_ids[i], ctx.DefineValue(outputs[i]));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axis_info,
                              graph::GetInfo(axis_tensor));
  if (axis_info.type != Type::kI32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis must be int32. Got type id %d.", op_name,
                        static_cast<int>(axis_info.type)));
  }
  if (axis_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis must be a constant tensor", op_name));
  }
  auto locked_axis = axis_info.buffer->Lock().As<const int32_t>();
  int real_axis = locked_axis.data()[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, graph::GetInfo(input));
  if (real_axis < 0) {
    real_axis += input_info.shape.size();
  }

  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_even_split(
      ctx.subgraph(), real_axis, input_id, outputs.size(), output_ids.data(),
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}
}  // namespace litert::tensor::graph
