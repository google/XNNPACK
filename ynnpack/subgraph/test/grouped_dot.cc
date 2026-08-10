// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

// Reference implementation of grouped_dot.
void ReferenceGroupedDot(
    size_t E, const int32_t* expert_counts, const int32_t* offsets,
    const float* input_a,
    const float* input_b,  // Unpacked weights [E, D_in, D_out]
    float* output, size_t D_in, size_t D_out) {
  for (size_t e = 0; e < E; ++e) {
    int32_t count = expert_counts[e];
    if (count == 0) continue;

    int32_t offset = offsets[e];
    const float* A_e = input_a + offset * D_in;
    const float* B_e = input_b + e * D_in * D_out;
    float* C_e = output + offset * D_out;

    for (int i = 0; i < count; ++i) {
      for (size_t d_out = 0; d_out < D_out; ++d_out) {
        float sum = 0.0f;
        for (size_t d_in = 0; d_in < D_in; ++d_in) {
          sum += A_e[i * D_in + d_in] * B_e[d_in * D_out + d_out];
        }
        C_e[i * D_out + d_out] = sum;
      }
    }
  }
}

void RunGroupedDotTest(size_t E, size_t NK, size_t D_in, size_t D_out,
                       bool empty_some_experts = false,
                       int num_threads = 0) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // 1. Generate routing.
  std::vector<int32_t> expert_indices(NK);
  std::uniform_int_distribution<int32_t> exp_dist(
      0, empty_some_experts ? (E / 2 - 1) : (E - 1));
  for (size_t i = 0; i < NK; ++i) {
    expert_indices[i] = exp_dist(rng);
  }

  // Compute counts and offsets.
  std::vector<int32_t> expert_counts_data(E, 0);
  for (size_t i = 0; i < NK; ++i) {
    expert_counts_data[expert_indices[i]]++;
  }
  std::vector<int32_t> offsets_data(E + 1, 0);
  for (size_t e = 0; e < E; ++e) {
    offsets_data[e + 1] = offsets_data[e] + expert_counts_data[e];
  }

  // 2. Generate inputs.
  Tensor<float> input_a({NK, D_in});
  for (size_t i = 0; i < input_a.size(); ++i) {
    input_a.data()[i] = dist(rng);
  }

  Tensor<float> input_b({E, D_in, D_out});
  for (size_t i = 0; i < input_b.size(); ++i) {
    input_b.data()[i] = dist(rng);
  }

  Tensor<int32_t> expert_counts({E});
  std::copy(expert_counts_data.begin(), expert_counts_data.end(),
            expert_counts.data());

  Tensor<int32_t> expert_offsets({E + 1});
  std::copy(offsets_data.begin(), offsets_data.end(), expert_offsets.data());

  // 3. Run reference.
  Tensor<float> ref_output({NK, D_out});
  ref_output.fill(0.0f);
  ReferenceGroupedDot(E, expert_counts.data(), expert_offsets.data(),
                      input_a.data(), input_b.data(), ref_output.data(),
                      D_in, D_out);

  // 4. Build subgraph.
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t counts_id = 2;
  const uint32_t offsets_id = 3;
  uint32_t output_id = 4;

  SubgraphBuilder subgraph(5);
  subgraph.AddInput(ynn_type_fp32, {NK, D_in}, a_id)
      .AddInput(ynn_type_fp32, {E, D_in, D_out}, b_id)
      .AddInput(ynn_type_int32, {E}, counts_id)
      .AddInput(ynn_type_int32, {E + 1}, offsets_id)
      .AddOutput(ynn_type_fp32, {NK, D_out}, output_id);

  ynn_status status = ynn_define_grouped_dot(
      subgraph.GetSubgraph(), a_id, b_id, counts_id, offsets_id, &output_id,
      /*flags=*/0);
  ASSERT_EQ(status, ynn_status_success);

  // 5. Create runtime.
  std::unique_ptr<TestScheduler> scheduler;
  if (num_threads > 1) {
    scheduler = std::make_unique<TestScheduler>(num_threads - 1);
  }
  Runtime runtime(subgraph.GetSubgraph(), scheduler.get());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> ynn_output({NK, D_out});
  ynn_output.fill(0.0f);

  runtime.SetupExternalTensor(input_a.data(), a_id);
  runtime.SetupExternalTensor(input_b.data(), b_id);
  runtime.SetupExternalTensor(expert_counts.data(), counts_id);
  runtime.SetupExternalTensor(expert_offsets.data(), offsets_id);
  runtime.SetupExternalTensor(ynn_output.data(), output_id);

  // 6. Invoke runtime.
  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // 7. Compare.
  for (size_t i = 0; i < NK * D_out; ++i) {
    EXPECT_NEAR(ynn_output.data()[i], ref_output.data()[i], 1e-4f)
        << "At index " << i;
  }
}

TEST(GroupedDotTest, Basic) {
  RunGroupedDotTest(/*E=*/8, /*NK=*/64, /*D_in=*/64, /*D_out=*/128);
}

TEST(GroupedDotTest, MultiThreaded) {
  RunGroupedDotTest(/*E=*/8, /*NK=*/64, /*D_in=*/64, /*D_out=*/128,
                    /*empty_some_experts=*/false, /*num_threads=*/4);
}

TEST(GroupedDotTest, SingleToken) {
  RunGroupedDotTest(/*E=*/8, /*NK=*/1, /*D_in=*/64, /*D_out=*/128,
                    /*empty_some_experts=*/false, /*num_threads=*/4);
}

TEST(GroupedDotTest, TwoTokens) {
  RunGroupedDotTest(/*E=*/8, /*NK=*/2, /*D_in=*/64, /*D_out=*/128,
                    /*empty_some_experts=*/false, /*num_threads=*/4);
}

TEST(GroupedDotTest, PrefillTokens) {
  RunGroupedDotTest(/*E=*/8, /*NK=*/18, /*D_in=*/64, /*D_out=*/128,
                    /*empty_some_experts=*/false, /*num_threads=*/4);
}

TEST(GroupedDotTest, MoEShape) {
  RunGroupedDotTest(/*E=*/4, /*NK=*/18, /*D_in=*/256, /*D_out=*/512,
                    /*empty_some_experts=*/false, /*num_threads=*/4);
}

TEST(GroupedDotTest, EmptyExperts) {
  RunGroupedDotTest(/*E=*/8, /*NK=*/32, /*D_in=*/64, /*D_out=*/128,
                    /*empty_some_experts=*/true);
}

TEST(GroupedDotTest, DifferentSizes) {
  RunGroupedDotTest(/*E=*/4, /*NK=*/16, /*D_in=*/32, /*D_out=*/64);
}

TEST(GroupedDotTest, InvalidShapeMismatch) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t counts_id = 2;
  const uint32_t offsets_id = 3;
  uint32_t output_id = 4;

  SubgraphBuilder subgraph(5);
  // D_in is 32 for A, but 64 for B.
  subgraph.AddInput(ynn_type_fp32, {16, 32}, a_id)
      .AddInput(ynn_type_fp32, {4, 64, 128}, b_id)
      .AddInput(ynn_type_int32, {4}, counts_id)
      .AddInput(ynn_type_int32, {5}, offsets_id)
      .AddOutput(ynn_type_fp32, {16, 128}, output_id);

  ynn_status status = ynn_define_grouped_dot(
      subgraph.GetSubgraph(), a_id, b_id, counts_id, offsets_id, &output_id,
      /*flags=*/0);
  EXPECT_EQ(status, ynn_status_invalid_parameter);
}

TEST(GroupedDotTest, InvalidType) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t counts_id = 2;
  const uint32_t offsets_id = 3;
  uint32_t output_id = 4;

  SubgraphBuilder subgraph(5);
  // input_a is int32 instead of fp32.
  subgraph.AddInput(ynn_type_int32, {16, 64}, a_id)
      .AddInput(ynn_type_fp32, {4, 64, 128}, b_id)
      .AddInput(ynn_type_int32, {4}, counts_id)
      .AddInput(ynn_type_int32, {5}, offsets_id)
      .AddOutput(ynn_type_fp32, {16, 128}, output_id);

  ynn_status status = ynn_define_grouped_dot(
      subgraph.GetSubgraph(), a_id, b_id, counts_id, offsets_id, &output_id,
      /*flags=*/0);
  EXPECT_EQ(status, ynn_status_unsupported_parameter);
}

TEST(GroupedDotTest, InvalidRank) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t counts_id = 2;
  const uint32_t offsets_id = 3;
  uint32_t output_id = 4;

  SubgraphBuilder subgraph(5);
  // input_b is rank 2 instead of rank 3.
  subgraph.AddInput(ynn_type_fp32, {16, 64}, a_id)
      .AddInput(ynn_type_fp32, {64, 128}, b_id)
      .AddInput(ynn_type_int32, {4}, counts_id)
      .AddInput(ynn_type_int32, {5}, offsets_id)
      .AddOutput(ynn_type_fp32, {16, 128}, output_id);

  ynn_status status = ynn_define_grouped_dot(
      subgraph.GetSubgraph(), a_id, b_id, counts_id, offsets_id, &output_id,
      /*flags=*/0);
  EXPECT_EQ(status, ynn_status_invalid_parameter);
}

TEST(GroupedDotTest, InvalidOffsetsLength) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t counts_id = 2;
  const uint32_t offsets_id = 3;
  uint32_t output_id = 4;

  SubgraphBuilder subgraph(5);
  // expert_offsets is size 4 instead of E+1 (5).
  subgraph.AddInput(ynn_type_fp32, {16, 64}, a_id)
      .AddInput(ynn_type_fp32, {4, 64, 128}, b_id)
      .AddInput(ynn_type_int32, {4}, counts_id)
      .AddInput(ynn_type_int32, {4}, offsets_id)
      .AddOutput(ynn_type_fp32, {16, 128}, output_id);

  ynn_status status = ynn_define_grouped_dot(
      subgraph.GetSubgraph(), a_id, b_id, counts_id, offsets_id, &output_id,
      /*flags=*/0);
  EXPECT_EQ(status, ynn_status_invalid_parameter);
}

TEST(GroupedDotTest, UnalignedDIn) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t counts_id = 2;
  const uint32_t offsets_id = 3;
  uint32_t output_id = 4;

  SubgraphBuilder subgraph(5);
  // D_in = 3 with fp16 has tile_k = 2, so 3 % 2 != 0.
  subgraph.AddInput(ynn_type_fp16, {16, 3}, a_id)
      .AddInput(ynn_type_fp16, {4, 3, 128}, b_id)
      .AddInput(ynn_type_int32, {4}, counts_id)
      .AddInput(ynn_type_int32, {5}, offsets_id)
      .AddOutput(ynn_type_fp16, {16, 128}, output_id);

  ynn_status status = ynn_define_grouped_dot(
      subgraph.GetSubgraph(), a_id, b_id, counts_id, offsets_id, &output_id,
      /*flags=*/0);
  EXPECT_EQ(status, ynn_status_unsupported_parameter);
}

}  // namespace
}  // namespace ynn
