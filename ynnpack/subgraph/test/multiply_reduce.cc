// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

struct MultiplyReduceParams {
  size_t batch;
  size_t num_blocks;
  size_t m;
  size_t n;
  bool dynamic_a;
  bool static_b;
  int threads;
};

// Reproduces the runtime multiply + reduce_sum steps from blockwise dot
// (e.g. QD8F32QB4WFullyConnected):
//   mul = multiply(a[batch, num_blocks, m, n] (int32),
//                  b[num_blocks, 1, n] (fp32))
//   out = reduce_sum(mul, axis=1) -> fp32[batch, m, n]
void TestMultiplyReduceInt32Fp32(const MultiplyReduceParams& p) {
  const std::vector<size_t> a_shape = {p.batch, p.num_blocks, p.m, p.n};
  const std::vector<size_t> b_shape = {p.num_blocks, 1, p.n};
  const std::vector<size_t> out_shape = {p.batch, p.m, p.n};

  Tensor<int32_t> a(a_shape);
  for (size_t batch = 0; batch < p.batch; ++batch) {
    for (size_t blk = 0; blk < p.num_blocks; ++blk) {
      for (size_t m = 0; m < p.m; ++m) {
        for (size_t n = 0; n < p.n; ++n) {
          a(batch, blk, m, n) = static_cast<int32_t>(
              ((batch * 3 + blk * 7 + m * 5 + n) % 13) - 6);
        }
      }
    }
  }

  Tensor<float> b(b_shape);
  for (size_t blk = 0; blk < p.num_blocks; ++blk) {
    for (size_t n = 0; n < p.n; ++n) {
      b(blk, 0, n) =
          0.25f * static_cast<float>(static_cast<int>((blk * 3 + n) % 9) - 4);
    }
  }

  SubgraphBuilder subgraph(p.static_b ? 2 : 3);
  const uint32_t a_id = 0;
  const uint32_t out_id = 1;
  const uint32_t b_id = 2;

  subgraph
      .AddInput(type_of<int32_t>(),
                p.dynamic_a ? TensorShape(4) : TensorShape(a_shape), a_id)
      .AddOutput(type_of<float>(),
                 p.dynamic_a ? TensorShape(3) : TensorShape(out_shape), out_id);

  if (p.static_b) {
    uint32_t static_b_id = YNN_INVALID_VALUE_ID;
    subgraph.AddTensor(type_of<float>(), b_shape, static_b_id, b.data(),
                       YNN_VALUE_FLAG_COPY_DATA);
    uint32_t mul_id = YNN_INVALID_VALUE_ID;
    subgraph.AddTensor(type_of<float>(), 4, mul_id);
    subgraph.AddBinary(ynn_binary_multiply, a_id, static_b_id, mul_id)
        .AddReduce(ynn_reduce_sum, /*reduce_axes=*/{1}, mul_id,
                   YNN_INVALID_VALUE_ID, out_id);
  } else {
    subgraph.AddInput(type_of<float>(), b_shape, b_id);
    uint32_t mul_id = YNN_INVALID_VALUE_ID;
    subgraph.AddTensor(type_of<float>(), 4, mul_id);
    subgraph.AddBinary(ynn_binary_multiply, a_id, b_id, mul_id)
        .AddReduce(ynn_reduce_sum, /*reduce_axes=*/{1}, mul_id,
                   YNN_INVALID_VALUE_ID, out_id);
  }

  TestScheduler scheduler(p.threads - 1);
  Runtime runtime(subgraph.GetSubgraph(), p.threads > 1 ? &scheduler : nullptr);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(a_shape, a.data(), a_id);
  if (!p.static_b) {
    runtime.ReshapeExternalTensor(b_shape, b.data(), b_id);
  }
  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(out_id), out_shape);

  Tensor<float> out(out_shape);
  out.fill(-12345.0f);
  runtime.SetupExternalTensor(out.data(), out_id).InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t batch = 0; batch < p.batch; ++batch) {
    for (size_t m = 0; m < p.m; ++m) {
      for (size_t n = 0; n < p.n; ++n) {
        float expected = 0.0f;
        for (size_t blk = 0; blk < p.num_blocks; ++blk) {
          expected += static_cast<float>(a(batch, blk, m, n)) * b(blk, 0, n);
        }
        ASSERT_EQ(out(batch, m, n), expected)
            << "at (" << batch << ", " << m << ", " << n << ")";
      }
    }
  }
}

class MultiplyReduceTest
    : public testing::TestWithParam<std::tuple<size_t, bool, bool, int>> {};

TEST_P(MultiplyReduceTest, BlockwiseScaleAndSum) {
  const auto [n, dynamic_a, static_b, threads] = GetParam();
  TestMultiplyReduceInt32Fp32({
      /*batch=*/1,
      /*num_blocks=*/32,
      /*m=*/512,
      /*n=*/n,
      /*dynamic_a=*/dynamic_a,
      /*static_b=*/static_b,
      /*threads=*/threads,
  });
}

INSTANTIATE_TEST_SUITE_P(
    QD8F32QB4WShapes, MultiplyReduceTest,
    testing::Combine(testing::Values<size_t>(1, 2, 4, 8, 16, 32, 64, 128, 256,
                                             512),
                     testing::Bool(), testing::Bool(), testing::Values(1, 4)),
    [](const testing::TestParamInfo<MultiplyReduceTest::ParamType>& info) {
      return "N_" + std::to_string(std::get<0>(info.param)) + "_" +
             (std::get<1>(info.param) ? "dynamic" : "static") + "_" +
             (std::get<2>(info.param) ? "static_b" : "external_b") + "_t" +
             std::to_string(std::get<3>(info.param));
    });

// Reproduces the constant-folding multiply + reduce_sum step for the zero-point
// column-sum correction:
//   mul = multiply(a[num_blocks, 1, n] (fp32), b[num_blocks, 1, n] (fp32))
//   out = reduce_sum(mul, axis=0) -> fp32[1, n]
TEST(MultiplyReduceFp32Test, ColsumScaleAndSum) {
  constexpr size_t kNumBlocks = 32;
  for (size_t n : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
    const std::vector<size_t> shape = {kNumBlocks, 1, n};
    const std::vector<size_t> out_shape = {1, n};

    Tensor<float> a(shape);
    Tensor<float> b(shape);
    for (size_t blk = 0; blk < kNumBlocks; ++blk) {
      for (size_t j = 0; j < n; ++j) {
        a(blk, 0, j) =
            static_cast<float>(static_cast<int>((blk * 5 + j) % 11) - 5);
        b(blk, 0, j) =
            0.25f * static_cast<float>(static_cast<int>((blk * 3 + j) % 7) - 3);
      }
    }

    SubgraphBuilder subgraph(3);
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t out_id = 2;
    uint32_t mul_id = YNN_INVALID_VALUE_ID;
    subgraph.AddInput(type_of<float>(), shape, a_id)
        .AddInput(type_of<float>(), shape, b_id)
        .AddOutput(type_of<float>(), out_shape, out_id)
        .AddTensor(type_of<float>(), shape, mul_id);
    subgraph.AddBinary(ynn_binary_multiply, a_id, b_id, mul_id)
        .AddReduce(ynn_reduce_sum, /*reduce_axes=*/{0}, mul_id,
                   YNN_INVALID_VALUE_ID, out_id);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    Tensor<float> out(out_shape);
    runtime.ReshapeExternalTensor(shape, a.data(), a_id)
        .ReshapeExternalTensor(shape, b.data(), b_id)
        .ReshapeRuntime()
        .SetupExternalTensor(out.data(), out_id)
        .InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (size_t j = 0; j < n; ++j) {
      float expected = 0.0f;
      for (size_t blk = 0; blk < kNumBlocks; ++blk) {
        expected += a(blk, 0, j) * b(blk, 0, j);
      }
      ASSERT_EQ(out(0, j), expected) << "at n=" << n << ", j=" << j;
    }
  }
}

}  // namespace
}  // namespace ynn
