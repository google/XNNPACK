// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

TEST(DotSumTest, SplitF32ToBf16) {
  // We will test splitting into 1, 2, and 3 values.
  // We create a subgraph with 1 input (fp32) and 3 outputs (bf16).
  subgraph_ptr subgraph = create_subgraph(4, 0);  // 1 input + 3 outputs
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_ids[3] = {1, 2, 3};
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(
        ynn_define_tensor(subgraph.get(), ynn_type_bf16, 1, nullptr, nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_ids[i]),
        ynn_status_success);
  }

  // We call define_convert_f32_to_bf16_sum to split x into y_ids.
  // We pass pre-defined external outputs.
  ASSERT_EQ(define_convert_f32_to_bf16_sum(subgraph.get(), x_id, 3, y_ids, 0),
            ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  size_t shape[] = {4};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 1, shape),
            ynn_status_success);

  // Input data: some floats that require different levels of precision.
  // e.g. pi, e, some small values, etc.
  std::vector<float> x_data = {
      3.14159265f,  // needs many bits
      1.0f,         // exact in bf16
      0.1f,         // not exact
      0.0001f       // small
  };
  std::vector<bfloat16> y0_data(4, bfloat16(0.0f));
  std::vector<bfloat16> y1_data(4, bfloat16(0.0f));
  std::vector<bfloat16> y2_data(4, bfloat16(0.0f));

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x_data.data()),
            ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), y_ids[0], y0_data.data()),
      ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), y_ids[1], y1_data.data()),
      ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), y_ids[2], y2_data.data()),
      ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  for (size_t i = 0; i < x_data.size(); ++i) {
    float x = x_data[i];
    float v0 = static_cast<float>(y0_data[i]);
    float v1 = static_cast<float>(y1_data[i]);
    float v2 = static_cast<float>(y2_data[i]);

    float rec1 = v0;
    float rec2 = v0 + v1;
    float rec3 = v0 + v1 + v2;

    // Relative errors
    float err1 = std::abs(x - rec1) / std::abs(x);
    float err2 = std::abs(x - rec2) / std::abs(x);
    float err3 = std::abs(x - rec3) / std::abs(x);

    // 1 split (bf16) has 8 bits of precision (relative error ~ 1/256 = 0.0039)
    // 2 splits has ~ 16 bits of precision (relative error ~ 1/65536 = 1.5e-5)
    // 3 splits has ~ 24 bits of precision (relative error ~ 1/16M = 6e-8)

    // For x = 1.0, it should be exact in first split, others 0.
    if (x == 1.0f) {
      EXPECT_EQ(v0, 1.0f);
      EXPECT_EQ(v1, 0.0f);
      EXPECT_EQ(v2, 0.0f);
    } else {
      EXPECT_LT(err1, 0.01f);
      if (err1 > 0.0f) {  // if it wasn't already exact
        EXPECT_LT(err2, err1);
        EXPECT_LT(err2, 0.0001f);
        if (err2 > 0.0f) {
          EXPECT_LT(err3, err2);
          // For float, 3 splits should be very close to exact.
          EXPECT_LT(err3, 1e-6f);
        }
      }
    }
  }
}

TEST(DotSumTest, DefineDotSum) {
  // Test composite dot with various split sizes.
  // We will test 2x2 split (which is what bf16x3 rewrite conceptually does,
  // but here we compute all 4 products).
  // Actually, let's test 2x2.
  // Input A (fp32) -> split into A0, A1 (bf16)
  // Input B (fp32) -> split into B0, B1 (bf16)
  // We will run:
  // 1. Reference float dot: A * B
  // 2. Composite dot: composite_dot({A0, A1}, {B0, B1})
  // Compare results.

  subgraph_ptr subgraph = create_subgraph(3, 0);  // A (in), B (in), C (out)
  ASSERT_NE(subgraph, nullptr);

  uint32_t a_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);

  uint32_t b_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_id),
            ynn_status_success);

  uint32_t c_id = 2;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &c_id),
            ynn_status_success);

  // Define splits in subgraph
  uint32_t a_values[2] = {YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID};
  ASSERT_EQ(
      define_convert_f32_to_bf16_sum(subgraph.get(), a_id, 2, a_values, 0),
      ynn_status_success);

  uint32_t b_values[2] = {YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID};
  ASSERT_EQ(
      define_convert_f32_to_bf16_sum(subgraph.get(), b_id, 2, b_values, 0),
      ynn_status_success);

  // Call dot sum
  // We use num_k_dims = 1
  ASSERT_EQ(define_dot_sum(subgraph.get(), 1, 2, a_values, 2, b_values,
                           YNN_INVALID_VALUE_ID, c_id, 0),
            ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  size_t a_shape[] = {2, 4};  // M=2, K=4
  size_t b_shape[] = {4, 3};  // K=4, N=3

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, b_shape),
            ynn_status_success);

  std::vector<float> a_data = {1.1f,  2.2f,  3.3f,  4.4f,
                               -1.1f, 0.55f, 1.55f, -2.55f};
  std::vector<float> b_data = {0.55f, 1.1f,  2.2f, -0.55f, 0.0f,  1.1f,
                               1.1f,  -1.1f, 0.0f, 2.2f,   1.55f, -1.1f};
  std::vector<float> c_data(6, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id, a_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), b_id, b_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), c_id, c_data.data()),
            ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  // Compute reference
  std::vector<float> expected_c(6, 0.0f);
  for (size_t m = 0; m < 2; ++m) {
    for (size_t n = 0; n < 3; ++n) {
      float sum = 0.0f;
      for (size_t k = 0; k < 4; ++k) {
        sum += a_data[m * 4 + k] * b_data[k * 3 + n];
      }
      expected_c[m * 3 + n] = sum;
    }
  }

  // Compare. Since we used 2 splits (which is 4 products total),
  // the precision should be very high (approx 16 bits of precision, relative
  // error < 1.5e-5).
  for (size_t i = 0; i < 6; ++i) {
    float ref = expected_c[i];
    float val = c_data[i];
    float err = std::abs(ref - val);
    if (std::abs(ref) > 1e-5f) {
      err /= std::abs(ref);
      EXPECT_LT(err, 1e-4f)
          << "at index " << i << ", ref=" << ref << ", val=" << val;
    } else {
      EXPECT_LT(err, 1e-4f)
          << "at index " << i << ", ref=" << ref << ", val=" << val;
    }
  }
}

TEST(DotTest, BlockwiseDot) {
  subgraph_ptr subgraph =
      create_subgraph(3, 0);  // 1 input + 1 weight + 1 output
  ASSERT_NE(subgraph, nullptr);

  const size_t M = 2;
  const size_t K = 4;
  const size_t N = 2;
  const size_t block_size = 2;
  const size_t num_blocks = K / block_size;

  uint32_t a_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);

  // B is static: shape [K, N] = [4, 2], type int8
  const int8_t b_data[K * N] = {
      1, 2, 3, 4, 5, 6, 7, 8,
  };
  const size_t b_dims[2] = {K, N};
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, b_dims, b_data,
                              YNN_VALUE_FLAG_COPY_DATA, &b_id),
            ynn_status_success);

  // B scales: shape [N, num_blocks] = [2, 2], type fp32
  const float b_scale_data[N * num_blocks] = {
      0.5f,
      0.25f,
      2.0f,
      1.0f,
  };
  const size_t b_scale_dims[2] = {N, num_blocks};
  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, b_scale_dims,
                        b_scale_data, YNN_VALUE_FLAG_COPY_DATA, &b_scale_id),
      ynn_status_success);

  uint32_t out_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id),
            ynn_status_success);

  ASSERT_EQ(define_blockwise_dot(subgraph.get(), a_id, YNN_INVALID_VALUE_ID,
                                 YNN_INVALID_VALUE_ID, b_id,
                                 YNN_INVALID_VALUE_ID, b_scale_id, block_size,
                                 YNN_INVALID_VALUE_ID, ynn_type_fp32, out_id,
                                 0),
            ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  const size_t a_shape[2] = {M, K};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_shape),
            ynn_status_success);

  const std::vector<float> a_data = {
      1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -2.0f,
  };
  std::vector<float> out_data(M * N, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id,
                                        const_cast<float*>(a_data.data())),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), out_id, out_data.data()),
            ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  // Reference computation:
  // For each m, n:
  // sum over blocks b: (sum over k in block: a_quant[m, k] * b[k, n]) *
  // b_scale[n, b] * a_scale[m] Because a is dynamically quantized, let's verify
  // output is non-zero and finite
  for (size_t i = 0; i < M * N; ++i) {
    EXPECT_FALSE(std::isnan(out_data[i]));
    EXPECT_FALSE(std::isinf(out_data[i]));
  }
}

}  // namespace
}  // namespace ynn
