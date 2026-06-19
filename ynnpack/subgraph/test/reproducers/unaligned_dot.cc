// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

TEST(Dot, UnalignedSplit) {
  // This is an unfortunately complicated test that exposed a bug:
  // - A dot is packed, so the split of the loop over n needs to be aligned
  // - We start out thinking we want to split by larger than n, so we don't set
  // a split, without "step_is_required = true".
  // - Later, the scheduler decides to split that dimension to something
  // smaller, producing an unaligned split of n.
  ReplicableRandomDevice rng;

  const float max_abs_value = 1.0f;

  // Inputs
  Tensor<float> in0({32});
  Tensor<float> in1({64, 32});
  Tensor<float> in2({64, 1, 64, 32});
  Tensor<float> in3({64, 1, 64, 32});
  Tensor<float> in4({64, 1, 64, 1});
  Tensor<float> out5({64, 1, 64, 32});

  fill_random(in0.data(), in0.size(), rng, -max_abs_value, max_abs_value);
  fill_random(in1.data(), in1.size(), rng, -max_abs_value, max_abs_value);
  fill_random(in2.data(), in2.size(), rng, -max_abs_value, max_abs_value);
  fill_random(in3.data(), in3.size(), rng, -max_abs_value, max_abs_value);
  fill_random(in4.data(), in4.size(), rng, -max_abs_value, max_abs_value);

  // Reference computation
  Tensor<float> val_8_ref({64, 1, 64, 32});
  for (size_t i0 = 0; i0 < 64; ++i0) {
    for (size_t i1 = 0; i1 < 1; ++i1) {
      for (size_t i2 = 0; i2 < 64; ++i2) {
        for (size_t i3 = 0; i3 < 32; ++i3) {
          val_8_ref(i0, i1, i2, i3) = in0(i3);
        }
      }
    }
  }

  Tensor<float> val_10_ref({64, 1, 64, 32});
  for (size_t i = 0; i < val_10_ref.size(); ++i) {
    val_10_ref.data()[i] = in2.data()[i] + val_8_ref.data()[i];
  }

  Tensor<float> val_19_ref({1, 64, 1, 32});
  for (size_t i0 = 0; i0 < 1; ++i0) {
    for (size_t i1 = 0; i1 < 64; ++i1) {
      for (size_t i2 = 0; i2 < 1; ++i2) {
        for (size_t i3 = 0; i3 < 32; ++i3) {
          val_19_ref(i0, i1, i2, i3) = in1(i1, i3);
        }
      }
    }
  }

  Tensor<float> val_21_ref({1, 64, 32, 64});
  for (size_t i0 = 0; i0 < 1; ++i0) {
    for (size_t i1 = 0; i1 < 64; ++i1) {
      for (size_t i2 = 0; i2 < 32; ++i2) {
        for (size_t i3 = 0; i3 < 64; ++i3) {
          val_21_ref(i0, i1, i2, i3) = val_10_ref(i3, i0, i1, i2);
        }
      }
    }
  }

  Tensor<float> val_22_ref({1, 64, 1, 64});
  val_22_ref.fill(0.0f);
  for (size_t b0 = 0; b0 < 1; ++b0) {
    for (size_t b1 = 0; b1 < 64; ++b1) {
      for (size_t m = 0; m < 1; ++m) {
        for (size_t n = 0; n < 64; ++n) {
          float sum = 0.0f;
          for (size_t k = 0; k < 32; ++k) {
            sum += val_19_ref(b0, b1, m, k) * val_21_ref(b0, b1, k, n);
          }
          val_22_ref(b0, b1, m, n) = sum;
        }
      }
    }
  }

  Tensor<float> val_14_ref({64, 1, 64, 1});
  for (size_t i0 = 0; i0 < 64; ++i0) {
    for (size_t i1 = 0; i1 < 1; ++i1) {
      for (size_t i2 = 0; i2 < 64; ++i2) {
        for (size_t i3 = 0; i3 < 1; ++i3) {
          val_14_ref(i0, i1, i2, i3) = val_22_ref(i1, i2, 0, i0);
        }
      }
    }
  }

  Tensor<float> val_15_ref({64, 1, 64, 1});
  for (size_t i = 0; i < val_15_ref.size(); ++i) {
    val_15_ref.data()[i] = val_14_ref.data()[i] + in4.data()[i];
  }

  Tensor<float> val_18_ref({64, 1, 64, 1});
  for (size_t i0 = 0; i0 < 64; ++i0) {
    for (size_t i1 = 0; i1 < 1; ++i1) {
      for (size_t i2 = 0; i2 < 64; ++i2) {
        for (size_t i3 = 0; i3 < 1; ++i3) {
          val_18_ref(i0, i1, i2, i3) = val_15_ref(i0, 0, i2, 0);
        }
      }
    }
  }

  Tensor<float> val_17_ref({64, 1, 64, 32});
  for (size_t i0 = 0; i0 < 64; ++i0) {
    for (size_t i1 = 0; i1 < 1; ++i1) {
      for (size_t i2 = 0; i2 < 64; ++i2) {
        for (size_t i3 = 0; i3 < 32; ++i3) {
          val_17_ref(i0, i1, i2, i3) = val_18_ref(i0, i1, i2, 0);
        }
      }
    }
  }

  Tensor<float> expected({64, 1, 64, 32});
  for (size_t i = 0; i < expected.size(); ++i) {
    expected.data()[i] = val_17_ref.data()[i] + in3.data()[i];
  }

  // Build Subgraph
  SubgraphBuilder subgraph(6, 0);

  const uint32_t in0_id = 0;
  const uint32_t in1_id = 1;
  const uint32_t in2_id = 2;
  const uint32_t in3_id = 3;
  const uint32_t in4_id = 4;
  const uint32_t out5_id = 5;

  subgraph.AddInput(ynn_type_fp32, {32}, in0_id)
      .AddInput(ynn_type_fp32, {64, 32}, in1_id)
      .AddInput(ynn_type_fp32, {64, 1, 64, 32}, in2_id)
      .AddInput(ynn_type_fp32, {64, 1, 64, 32}, in3_id)
      .AddInput(ynn_type_fp32, {64, 1, 64, 1}, in4_id)
      .AddOutput(ynn_type_fp32, {64, 1, 64, 32}, out5_id);

  uint32_t val_8_id = YNN_INVALID_VALUE_ID;
  uint32_t val_10_id = YNN_INVALID_VALUE_ID;
  uint32_t val_19_id = YNN_INVALID_VALUE_ID;
  uint32_t val_21_id = YNN_INVALID_VALUE_ID;
  uint32_t val_22_id = YNN_INVALID_VALUE_ID;
  uint32_t val_14_id = YNN_INVALID_VALUE_ID;
  uint32_t val_15_id = YNN_INVALID_VALUE_ID;
  uint32_t val_18_id = YNN_INVALID_VALUE_ID;
  uint32_t val_17_id = YNN_INVALID_VALUE_ID;

  subgraph.AddTensor(ynn_type_fp32, {64, 1, 64, 32}, val_8_id)
      .AddTensor(ynn_type_fp32, {64, 1, 64, 32}, val_10_id)
      .AddTensor(ynn_type_fp32, {1, 64, 1, 32}, val_19_id)
      .AddTensor(ynn_type_fp32, {1, 64, 32, 64}, val_21_id)
      .AddTensor(ynn_type_fp32, {1, 64, 1, 64}, val_22_id)
      .AddTensor(ynn_type_fp32, {64, 1, 64, 1}, val_14_id)
      .AddTensor(ynn_type_fp32, {64, 1, 64, 1}, val_15_id)
      .AddTensor(ynn_type_fp32, {64, 1, 64, 1}, val_18_id)
      .AddTensor(ynn_type_fp32, {64, 1, 64, 32}, val_17_id);

  subgraph.AddStaticBroadcast({64, 1, 64, 32}, in0_id, val_8_id)
      .AddBinary(ynn_binary_add, in2_id, val_8_id, val_10_id)
      .AddTranspose({2, 0, 2, 1}, in1_id, val_19_id)
      .AddTranspose({1, 2, 3, 0}, val_10_id, val_21_id)
      .AddDot(1, val_19_id, val_21_id, YNN_INVALID_VALUE_ID, val_22_id)
      .AddTranspose({3, 0, 1, 4}, val_22_id, val_14_id)
      .AddBinary(ynn_binary_add, val_14_id, in4_id, val_15_id)
      .AddTranspose({0, 4, 2, 4}, val_15_id, val_18_id)
      .AddStaticBroadcast({64, 1, 64, 32}, val_18_id, val_17_id)
      .AddBinary(ynn_binary_add, val_17_id, in3_id, out5_id);

  // Run the subgraph
  TestScheduler scheduler(6);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor({32}, in0.data(), in0_id);
  runtime.ReshapeExternalTensor({64, 32}, in1.data(), in1_id);
  runtime.ReshapeExternalTensor({64, 1, 64, 32}, in2.data(), in2_id);
  runtime.ReshapeExternalTensor({64, 1, 64, 32}, in3.data(), in3_id);
  runtime.ReshapeExternalTensor({64, 1, 64, 1}, in4.data(), in4_id);

  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(out5_id),
            std::vector<size_t>({64, 1, 64, 32}));

  runtime.SetupExternalTensor(out5.data(), out5_id).InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // Compare results
  for (const auto& i : EnumerateIndices(expected.extents())) {
    ASSERT_NEAR(out5(i), expected(i), 1e-4) << "i=" << index_to_string(i);
  }
}

}  // namespace ynn
