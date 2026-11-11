/* Copyright 2026 Google LLC.

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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/tensor/arithmetic.h"
#include "litert/tensor/backends/testing/numerical_test_bridge.h"
#include "litert/tensor/backends/testing/numerical_test_suite.h"
#include "litert/tensor/backends/xnnpack/arithmetic.h"
#include "litert/tensor/backends/xnnpack/conversion.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/runners/xnnpack/runner.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/macros.h"
#include "litert/tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

class XnnpackTestBackendBridge : public TestBackendBridge {
 public:
  ~XnnpackTestBackendBridge() override = default;

  absl::Status Initialize() override {
    if (xnn_initialize(nullptr) != xnn_status_success) {
      return absl::InternalError("Failed to initialize XNNPACK");
    }
    return absl::OkStatus();
  }

  absl::Status BuildGraph(absl::Span<const TensorHandle> inputs,
                          absl::Span<const TensorHandle> outputs) override {
    std::vector<TensorHandle> output_handles(outputs.begin(), outputs.end());
    auto runner_or = XnnpackRunner::Create(output_handles);
    if (!runner_or.ok()) {
      absl::Status s = runner_or.status();
      if (s.code() == absl::StatusCode::kInvalidArgument &&
          absl::StrContains(s.message(), "does not implement")) {
        return absl::UnimplementedError(s.message());
      }
      return s;
    }
    runner_ = std::make_unique<XnnpackRunner>(std::move(*runner_or));
    return absl::OkStatus();
  }

  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data) override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    return runner_->SetInput(tensor, data);
  }

  absl::Status Execute() override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    return runner_->Run();
  }

  absl::Status GetOutput(const TensorHandle& tensor,
                         absl::Span<std::byte> data) override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto output_bytes, runner_->ReadOutput(tensor));
    if (output_bytes.size() != data.size()) {
      return absl::InvalidArgumentError("Output buffer size mismatch");
    }
    std::memcpy(data.data(), output_bytes.data(), data.size());
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<XnnpackRunner> runner_;
};

struct XnnpackTraits {
  using Tag = XnnpackMixinTag;
  // TODO(b/574438217): Enable these tests once the XNNPACK backend is fixed.
  static constexpr absl::string_view kDisabledTests[] = {
      // Blocks of 16 fail when the runtime is created.
      "FullyConnectedQuantizedWeightsOp",
      // Fully connected ops reject per-tensor int8 weights.
      "FullyConnectedPerTensorAndAsymmetricOp",
      // Batch matmuls reject int4 right-hand sides.
      "BatchMatMulQuantizedRhsOp",
      // Transposes reject int4 and blockwise values.
      "TransposedQuantizedWeightsIntoBatchMatMulOp",
      // Transposes reject int4 values, and asymmetric int4 weights fail to
      // dequantize.
      "TransposedQuantizedWeightsIntoAddOp",
      // Dequantizing int4 values aborts in an XNNPACK assertion.
      "QuantizedWeightsSharedByDotAndAddOp",
      // Per-channel zero points of tensors that aren't constants are
      // dequantized as if they were constants, which fails.
      "DequantizePerChannelOp",
  };
  static std::unique_ptr<TestBackendBridge> CreateBridge() {
    return std::make_unique<XnnpackTestBackendBridge>();
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Xnnpack, NumericalTestSuite, XnnpackTraits);

TEST(XnnpackQuantizedWeightsFallbackTest,
     FullyConnectedAndBatchMatMulSupportBlockwiseQ8_0AndAffineInt4) {
  constexpr int kOut = 6;
  constexpr int kIn = 64;
  constexpr int kBlockSize = 32;
  constexpr int kBlocksPerRow = kIn / kBlockSize;

  // Build blockwise INT8 (`Q8_0`) weights and reference FP32 weights.
  std::vector<int8_t> q8_data(kOut * kIn);
  std::vector<float> q8_scales(kOut * kBlocksPerRow);
  std::vector<float> q8_ref(kOut * kIn);
  for (int r = 0; r < kOut; ++r) {
    for (int b = 0; b < kBlocksPerRow; ++b) {
      q8_scales[r * kBlocksPerRow + b] =
          0.02f * static_cast<float>((r * kBlocksPerRow + b) + 1);
      for (int j = 0; j < kBlockSize; ++j) {
        const int idx = r * kIn + b * kBlockSize + j;
        q8_data[idx] = static_cast<int8_t>((idx * 7) % 41 - 20);
        q8_ref[idx] =
            q8_scales[r * kBlocksPerRow + b] * static_cast<float>(q8_data[idx]);
      }
    }
  }
  Tensor<XnnpackMixinTag> w_q8(
      {.name = "w_q8_0",
       .type = Type::kI8,
       .shape = {kOut, kIn},
       .buffer = q8_data,
       .quantization = std::make_shared<BlockwiseQuantization>(
           q8_scales, std::vector<int64_t>{}, kBlockSize,
           /*quantized_dimension=*/0)});

  // Build affine blockwise INT4 (`Q4_1` / `Q4_K`) weights and reference FP32.
  auto q4_buf = OwningCpuBuffer::Allocate<Type::kI4>(kOut * kIn);
  int4_t* q4_packed = q4_buf->Span<int4_t>().data();
  std::vector<float> q4_scales(kOut * kBlocksPerRow);
  std::vector<float> q4_zps(kOut * kBlocksPerRow);
  std::vector<float> q4_ref(kOut * kIn);
  for (int r = 0; r < kOut; ++r) {
    for (int b = 0; b < kBlocksPerRow; ++b) {
      const int blk = r * kBlocksPerRow + b;
      q4_scales[blk] = 0.05f * static_cast<float>((blk % 5) + 1);
      q4_zps[blk] = (blk % 2 == 0) ? -2.0f : 1.5f;
      for (int j = 0; j < kBlockSize; j += 2) {
        const int elem = r * kIn + b * kBlockSize + j;
        const int qa = ((elem * 3 + 1) % 15) - 7;
        const int qb = ((elem * 5 + 4) % 15) - 7;
        q4_packed[elem / 2].a = static_cast<int8_t>(qa);
        q4_packed[elem / 2].b = static_cast<int8_t>(qb);
        q4_ref[elem] =
            q4_scales[blk] * (static_cast<float>(qa) - q4_zps[blk]);
        q4_ref[elem + 1] =
            q4_scales[blk] * (static_cast<float>(qb) - q4_zps[blk]);
      }
    }
  }
  Tensor<XnnpackMixinTag> w_q4_affine(
      {.name = "w_q4_k",
       .type = Type::kI4,
       .shape = {kOut, kIn},
       .buffer = q4_buf,
       .quantization = std::make_shared<BlockwiseQuantization>(
           q4_scales, std::vector<int64_t>{}, kBlockSize,
           /*quantized_dimension=*/0, q4_zps)});

  for (int seq_len : {1, 8}) {
    std::vector<float> x_vals(seq_len * kIn);
    for (int i = 0; i < seq_len * kIn; ++i) {
      x_vals[i] = 0.05f * static_cast<float>((i * 11) % 17 - 8);
    }

    // 1. FullyConnected + BatchMatMul(adj_y=true) with Q8_0 weights.
    {
      Tensor<XnnpackMixinTag> x(
          {.name = "x", .type = Type::kFP32, .shape = {seq_len, kIn}});
      Tensor<XnnpackMixinTag> y_fc = FullyConnected(x, w_q8);
      Tensor<XnnpackMixinTag> y_bmm =
          BatchMatMul(x, w_q8, /*adj_x=*/false, /*adj_y=*/true);
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          XnnpackRunner runner,
          XnnpackRunner::Create({TensorHandle(y_fc), TensorHandle(y_bmm)}));
      ASSERT_THAT(runner.SetInput(TensorHandle(x), x_vals), IsOk());
      ASSERT_THAT(runner.Run(), IsOk());
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          auto actual_fc, runner.ReadOutputAs<float>(TensorHandle(y_fc)));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          auto actual_bmm, runner.ReadOutputAs<float>(TensorHandle(y_bmm)));

      std::vector<float> expected(seq_len * kOut, 0.0f);
      for (int t = 0; t < seq_len; ++t) {
        for (int o = 0; o < kOut; ++o) {
          for (int i = 0; i < kIn; ++i) {
            expected[t * kOut + o] += x_vals[t * kIn + i] * q8_ref[o * kIn + i];
          }
        }
      }
      EXPECT_THAT(actual_fc, testing::Pointwise(testing::FloatNear(0.15f),
                                                expected));
      EXPECT_THAT(actual_bmm, testing::Pointwise(testing::FloatNear(0.15f),
                                                 expected));
    }

    // 2. FullyConnected with affine blockwise INT4 (`Q4_1`/`Q4_K`) weights.
    {
      Tensor<XnnpackMixinTag> x(
          {.name = "x", .type = Type::kFP32, .shape = {seq_len, kIn}});
      Tensor<XnnpackMixinTag> y_fc = FullyConnected(x, w_q4_affine);
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          XnnpackRunner runner, XnnpackRunner::Create({TensorHandle(y_fc)}));
      ASSERT_THAT(runner.SetInput(TensorHandle(x), x_vals), IsOk());
      ASSERT_THAT(runner.Run(), IsOk());
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          auto actual_fc, runner.ReadOutputAs<float>(TensorHandle(y_fc)));

      std::vector<float> expected(seq_len * kOut, 0.0f);
      for (int t = 0; t < seq_len; ++t) {
        for (int o = 0; o < kOut; ++o) {
          for (int i = 0; i < kIn; ++i) {
            expected[t * kOut + o] += x_vals[t * kIn + i] * q4_ref[o * kIn + i];
          }
        }
      }
      // Symmetric 4-bit fallback re-quantizes asymmetric [-8..7] - zp into
      // symmetric [-8..7], so allow the expected 4-bit re-quantization margin.
      EXPECT_THAT(actual_fc, testing::Pointwise(testing::FloatNear(0.45f),
                                                expected));
    }
  }
}

}  // namespace
}  // namespace litert::tensor
