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
#include "litert/tensor/internal/graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/source_location.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/internal/type_id.h"
#include "litert/tensor/utils/matchers.h"

namespace litert::tensor::graph {
namespace {

using testing::ElementsAreArray;
using testing::Not;
using testing::SizeIs;
using testing::StrEq;

MATCHER_P(IsLocation, loc, "") {
  return testing::ExplainMatchResult(testing::Eq(loc.line()), arg.line(),
                                     result_listener) &&
         testing::ExplainMatchResult(testing::StrEq(loc.file_name()),
                                     arg.file_name(), result_listener);
}

TEST(TensorTest, NewTensorGroupWorks) {
  const auto loc = absl::SourceLocation::current();
  const std::shared_ptr<const TensorGroup> g = NewTensorGroup(3, loc);
  EXPECT_NE(g, nullptr);
  EXPECT_THAT(g->tensor_infos, SizeIs(3));
  EXPECT_THAT(g->status, IsOk());
  EXPECT_EQ(g->producer, nullptr);
  EXPECT_THAT(g->loc, IsLocation(loc));
}

TEST(TensorTest, NewTensorCreatesAValidTensor) {
  const auto loc = absl::SourceLocation::current();
  Tensor a = NewTensor(loc);
  EXPECT_EQ(a.index, 0);
  ASSERT_NE(a.group, nullptr);
  EXPECT_THAT(GetStatus(a), IsOk());
  EXPECT_THAT(GetLocation(a), IsLocation(loc));
}

TEST(TensorTest, SetAndGetTensorName) {
  Tensor a = NewTensor(absl::SourceLocation::current());
  EXPECT_THAT(SetName(a, "my tensor name"), IsOk());
  EXPECT_THAT(GetName(a), IsOkAndHolds(StrEq("my tensor name")));
}

TEST(TensorTest, SetAndGetBuffer) {
  Tensor a = NewTensor(absl::SourceLocation::current());
  auto buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  EXPECT_THAT(SetBuffer(a, buffer), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & retrieved_buffer, GetBuffer(a));
  EXPECT_THAT(retrieved_buffer.Lock(), ElementsAreArray(buffer->Lock()));
}

TEST(TensorTest, DefaultTensorIsInvalid) {
  Tensor a;
  EXPECT_THAT(GetStatus(a), Not(IsOk()));
  EXPECT_THAT(GetLocation(a), IsLocation(absl::SourceLocation()));
  EXPECT_THAT(GetProducer(a), Not(IsOk()));
  EXPECT_THAT(GetConsumers(a), Not(IsOk()));
}

TEST(TensorTest, TensorWithWringIndexIsInvalid) {
  Tensor a = NewTensor(absl::SourceLocation::current());
  a.index = 3;
  EXPECT_THAT(GetStatus(a), Not(IsOk()));
}

TEST(QuantizationTest, GetTypeId) {
  PerChannelAffineQuantization quant({1.0f}, {0}, 0);
  EXPECT_EQ(quant.GetTypeId(),
            internal::TypeId::Get<PerChannelAffineQuantization>());
}

TEST(QuantizationTest, IsA) {
  PerChannelAffineQuantization quant({1.0f}, {0}, 0);
  EXPECT_TRUE(quant.IsA(internal::TypeId::Get<PerChannelAffineQuantization>()));
}

struct DummyQuantization : Quantization {
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<DummyQuantization>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<DummyQuantization>();
  }
};

TEST(QuantizationTest, As) {
  PerChannelAffineQuantization quant({1.0f}, {0}, 0);
  Quantization& quant_ref = quant;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(PerChannelAffineQuantization & quant_as,
                                  quant_ref.As<PerChannelAffineQuantization>());
  EXPECT_EQ(&quant_as, &quant);
  EXPECT_THAT(quant_ref.As<DummyQuantization>(), Not(IsOk()));

  const Quantization& const_quant_ref = quant;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const PerChannelAffineQuantization& const_quant_as,
      const_quant_ref.As<PerChannelAffineQuantization>());
  EXPECT_EQ(&const_quant_as, &quant);
  EXPECT_THAT(const_quant_ref.As<DummyQuantization>(), Not(IsOk()));
}

struct DummyOp : Operation {
  absl::string_view GetName() const override { return "DummyOp"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

TEST(ScopeTest, NestedScopeHierarchyAndAttributesAndBoundaries) {
  EXPECT_EQ(CurrentScope(), nullptr);

  Tensor leaf_input = NewTensor(absl::SourceLocation::current());
  Tensor leaf_weight = NewTensor(absl::SourceLocation::current());

  std::shared_ptr<DummyOp> inner_op;
  Tensor inner_output;
  {
    ScopeGuard layer_scope("transformer_layer", "layer_3");
    layer_scope.Attr("layer_idx", 3).Bind("input", leaf_input);

    {
      ScopeGuard attn_scope("gated_attention", "attn");
      attn_scope.Attr("num_heads", 8)
          .Attr("scale", 0.0625)
          .Attr("mode", "gqa")
          .Bind("input", leaf_input)
          .Bind("q_weight", leaf_weight);

      inner_op = std::make_shared<DummyOp>();
      auto group = NewTensorGroup(1, absl::SourceLocation::current());
      inner_op->outputs_group = group;
      group->producer = inner_op;
      inner_output = GetTensor(0, group);
      attn_scope.Bind("output", inner_output);
    }

    EXPECT_EQ(CurrentScope(), layer_scope.scope());
  }
  EXPECT_EQ(CurrentScope(), nullptr);

  ASSERT_NE(inner_op->scope, nullptr);
  EXPECT_EQ(inner_op->scope->kind, "gated_attention");
  EXPECT_EQ(inner_op->scope->name, "attn");
  EXPECT_EQ(inner_op->scope->FullPath(), "layer_3/attn");
  EXPECT_EQ(inner_op->scope->GetIntAttr("num_heads"), 8);
  EXPECT_DOUBLE_EQ(*inner_op->scope->GetFloatAttr("scale"), 0.0625);
  EXPECT_EQ(inner_op->scope->GetStringAttr("mode"), "gqa");

  ASSERT_NE(inner_op->scope->parent, nullptr);
  EXPECT_EQ(inner_op->scope->parent->kind, "transformer_layer");
  EXPECT_EQ(inner_op->scope->parent->name, "layer_3");
  EXPECT_EQ(inner_op->scope->parent->GetIntAttr("layer_idx"), 3);
  EXPECT_EQ(inner_op->scope->FindAncestor("transformer_layer"),
            inner_op->scope->parent.get());

  auto bound_in = inner_op->scope->GetBoundaryTensor("input");
  ASSERT_TRUE(bound_in.has_value());
  EXPECT_EQ(*bound_in, leaf_input);

  auto bound_w = inner_op->scope->GetBoundaryTensor("q_weight");
  ASSERT_TRUE(bound_w.has_value());
  EXPECT_EQ(*bound_w, leaf_weight);

  auto bound_out = inner_op->scope->GetBoundaryTensor("output");
  ASSERT_TRUE(bound_out.has_value());
  EXPECT_EQ(*bound_out, inner_output);
}

TEST(ScopeTest, BindOptionalTensorWrapper) {
  struct FakeTensorWrapper {
    Tensor raw;
    const Tensor& GetRaw() const { return raw; }
  };

  FakeTensorWrapper present{NewTensor(absl::SourceLocation::current())};
  std::optional<FakeTensorWrapper> opt_present = present;
  std::optional<FakeTensorWrapper> opt_empty = std::nullopt;

  std::shared_ptr<Scope> captured_scope;
  {
    ScopeGuard scope("test_scope");
    scope.Bind("present", opt_present).Bind("empty", opt_empty);
    captured_scope = scope.scope();
  }

  ASSERT_NE(captured_scope, nullptr);
  auto bound_present = captured_scope->GetBoundaryTensor("present");
  ASSERT_TRUE(bound_present.has_value());
  EXPECT_EQ(*bound_present, present.raw);
  EXPECT_FALSE(captured_scope->GetBoundaryTensor("empty").has_value());
}

TEST(QuantizationTest, BlockwiseFloatZeroPointsAndPredicates) {
  BlockwiseQuantization sym({0.25f, 0.5f}, {}, /*block_size=*/32);
  EXPECT_FALSE(sym.HasNonZeroFloatZeroPoints());
  EXPECT_FALSE(sym.HasNonZeroZeroPoints());
  EXPECT_FLOAT_EQ(sym.ZeroPointAt(0), 0.0f);
  EXPECT_FLOAT_EQ(sym.ZeroPointAt(1), 0.0f);

  BlockwiseQuantization all_zero_fzp({0.25f, 0.5f}, {}, /*block_size=*/32,
                                     /*quantized_dimension=*/0, {0.0f, 0.0f});
  EXPECT_FALSE(all_zero_fzp.HasNonZeroFloatZeroPoints());
  EXPECT_FALSE(all_zero_fzp.HasNonZeroZeroPoints());

  BlockwiseQuantization broadcast_fzp({0.25f, 0.5f}, {}, /*block_size=*/32,
                                      /*quantized_dimension=*/0, {-2.5f});
  EXPECT_TRUE(broadcast_fzp.HasNonZeroFloatZeroPoints());
  EXPECT_TRUE(broadcast_fzp.HasNonZeroZeroPoints());
  EXPECT_FLOAT_EQ(broadcast_fzp.ZeroPointAt(0), -2.5f);
  EXPECT_FLOAT_EQ(broadcast_fzp.ZeroPointAt(7), -2.5f);

  BlockwiseQuantization per_block_fzp({0.25f, 0.5f}, {}, /*block_size=*/32,
                                      /*quantized_dimension=*/0,
                                      {0.0f, -6.25f});
  EXPECT_TRUE(per_block_fzp.HasNonZeroFloatZeroPoints());
  EXPECT_TRUE(per_block_fzp.HasNonZeroZeroPoints());
  EXPECT_FLOAT_EQ(per_block_fzp.ZeroPointAt(0), 0.0f);
  EXPECT_FLOAT_EQ(per_block_fzp.ZeroPointAt(1), -6.25f);
  EXPECT_FLOAT_EQ(per_block_fzp.ZeroPointAt(2), 0.0f);

  BlockwiseQuantization int_zp({0.25f, 0.5f}, {0, -3}, /*block_size=*/32);
  EXPECT_FALSE(int_zp.HasNonZeroFloatZeroPoints());
  EXPECT_TRUE(int_zp.HasNonZeroZeroPoints());
  EXPECT_FLOAT_EQ(int_zp.ZeroPointAt(0), 0.0f);
  EXPECT_FLOAT_EQ(int_zp.ZeroPointAt(1), -3.0f);
}

TEST(QuantizationTest,
     BlockwiseCpuSymmetricFallbackI8AndI4AndCacheInvalidation) {
  constexpr int kRows = 2;
  constexpr int kCols = 64;
  constexpr int kBlockSize = 32;

  // 1. Blockwise INT8 (`Q8_0`) -> per-channel symmetric `qcint8`.
  std::vector<int8_t> raw_i8(kRows * kCols);
  for (int i = 0; i < kRows * kCols; ++i) {
    raw_i8[i] = static_cast<int8_t>((i % 25) - 12);
  }
  BlockwiseQuantization bq_i8({0.125f, 0.5f, 0.25f, 0.0625f}, {}, kBlockSize);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto fb_i8,
      bq_i8.GetOrCreateCpuSymmetricFallback(
          Type::kI8, kRows, kCols, raw_i8.data(), raw_i8.size()));
  ASSERT_NE(fb_i8, nullptr);
  EXPECT_EQ(fb_i8->type, Type::kI8);
  ASSERT_EQ(fb_i8->scales.size(), static_cast<size_t>(kRows));
  EXPECT_EQ(fb_i8->bf16_scales, nullptr);

  // Subsequent call returns the exact cached pointer.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto fb_i8_cached,
      bq_i8.GetOrCreateCpuSymmetricFallback(
          Type::kI8, kRows, kCols, raw_i8.data(), raw_i8.size()));
  EXPECT_EQ(fb_i8_cached.get(), fb_i8.get());

  const auto sym_i8 = fb_i8->data->Span<const int8_t>();
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const int blk = r * (kCols / kBlockSize) + (c / kBlockSize);
      const float ref =
          bq_i8.scales[blk] * static_cast<float>(raw_i8[r * kCols + c]);
      const float deq =
          fb_i8->scales[r] * static_cast<float>(sym_i8[r * kCols + c]);
      EXPECT_NEAR(deq, ref, 0.05f);
    }
  }

  bq_i8.InvalidateBf16ScalesCache();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto fb_i8_rebuilt,
      bq_i8.GetOrCreateCpuSymmetricFallback(
          Type::kI8, kRows, kCols, raw_i8.data(), raw_i8.size()));
  EXPECT_NE(fb_i8_rebuilt.get(), fb_i8.get());

  // 2. Affine blockwise INT4 (`Q4_1` / `Q4_K`) -> symmetric `qbint4`.
  auto buf_i4 = OwningCpuBuffer::Allocate<Type::kI4>(kRows * kCols);
  int4_t* packed_i4 = buf_i4->Span<int4_t>().data();
  for (int idx = 0; idx < (kRows * kCols) / 2; ++idx) {
    packed_i4[idx].a = static_cast<int8_t>((idx % 15) - 7);
    packed_i4[idx].b = static_cast<int8_t>(((idx * 3 + 5) % 15) - 7);
  }
  BlockwiseQuantization bq_i4({0.25f, 0.125f, 0.5f, 0.0625f}, {}, kBlockSize,
                              /*quantized_dimension=*/0,
                              {-2.0f, 1.5f, 0.0f, -3.5f});
  auto raw_i4_lock = buf_i4->Lock();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto fb_i4,
      bq_i4.GetOrCreateCpuSymmetricFallback(
          Type::kI4, kRows, kCols, raw_i4_lock.data(), raw_i4_lock.size()));
  ASSERT_NE(fb_i4, nullptr);
  EXPECT_EQ(fb_i4->type, Type::kI4);
  ASSERT_EQ(fb_i4->scales.size(), 4u);
  ASSERT_NE(fb_i4->bf16_scales, nullptr);
  EXPECT_EQ(fb_i4->bf16_scales->size(), 4u);

  const auto sym_i4 = fb_i4->data->Span<const int4_t>();
  for (int blk = 0; blk < 4; ++blk) {
    const float max_abs = 10.0f * bq_i4.scales[blk];
    for (int j = 0; j < kBlockSize; j += 2) {
      const int idx = (blk * kBlockSize + j) / 2;
      const float ref_a =
          bq_i4.scales[blk] *
          (static_cast<float>(static_cast<int>(packed_i4[idx].a)) -
           bq_i4.ZeroPointAt(blk));
      const float ref_b =
          bq_i4.scales[blk] *
          (static_cast<float>(static_cast<int>(packed_i4[idx].b)) -
           bq_i4.ZeroPointAt(blk));
      const float deq_a =
          static_cast<float>((*fb_i4->bf16_scales)[blk]) *
          static_cast<float>(static_cast<int>(sym_i4[idx].a));
      const float deq_b =
          static_cast<float>((*fb_i4->bf16_scales)[blk]) *
          static_cast<float>(static_cast<int>(sym_i4[idx].b));
      // Symmetric 4-bit re-quantization step is at most max(|w|) / 8.
      EXPECT_NEAR(deq_a, ref_a, max_abs / 7.0f + 0.05f);
      EXPECT_NEAR(deq_b, ref_b, max_abs / 7.0f + 0.05f);
    }
  }

  // Error validation checks.
  EXPECT_THAT(bq_i4.GetOrCreateCpuSymmetricFallback(
                  Type::kI4, kRows, /*cols=*/48, raw_i4_lock.data(),
                  raw_i4_lock.size()),
              Not(IsOk()));
  EXPECT_THAT(bq_i4.GetOrCreateCpuSymmetricFallback(
                  Type::kI4, kRows, kCols, raw_i4_lock.data(), /*raw_bytes=*/4),
              Not(IsOk()));
  EXPECT_THAT(bq_i4.GetOrCreateCpuSymmetricFallback(
                  Type::kFP32, kRows, kCols, raw_i4_lock.data(),
                  raw_i4_lock.size()),
              Not(IsOk()));
}

}  // namespace
}  // namespace litert::tensor::graph
