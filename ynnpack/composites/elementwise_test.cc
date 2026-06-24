// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

float gelu(float x) {
  return x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

float approxgelu(float x) {
  const float c1 = static_cast<float>(std::sqrt(2.0f / M_PI));
  const float c3 = 0.044715f;
  return x * 0.5f * (1.0f + std::tanh(c1 * x * (1.0f + c3 * x * x)));
}

float elu(float x, float alpha) { return x < 0.0f ? alpha * std::expm1(x) : x; }

float leaky_relu(float x, float alpha) { return x < 0.0f ? alpha * x : x; }

float hardswish(float x) {
  return x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

template <typename Define, typename Ref>
void VerifyUnaryComposite(Define define_fn, Ref ref_fn) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  if (ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, nullptr, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_INPUT,
                        &x_id) != ynn_status_success) {
    ADD_FAILURE() << "Failed to define input tensor";
    return;
  }

  uint32_t y_id = 1;
  if (ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, nullptr, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                        &y_id) != ynn_status_success) {
    ADD_FAILURE() << "Failed to define output tensor";
    return;
  }

  if (define_fn(subgraph.get(), x_id, y_id) != ynn_status_success) {
    ADD_FAILURE() << "Failed to define composite operation";
    return;
  }

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  const size_t n = 100;
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i) / 10.0f - 5.0f;
  }

  size_t shape[] = {n};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 1, shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);

  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(y[i], ref_fn(x[i]), 1e-5f);
  }
}

TEST(Composites, GeluExact) {
  VerifyUnaryComposite([](ynn_subgraph_t g, uint32_t in,
                          uint32_t& out) { return define_gelu(g, in, out); },
                       gelu);
}

TEST(Composites, GeluApproximate) {
  VerifyUnaryComposite(
      [](ynn_subgraph_t g, uint32_t in, uint32_t& out) {
        return define_approx_gelu(g, in, out);
      },
      approxgelu);
}

TEST(Composites, Elu) {
  const float alpha = 1.0f;
  VerifyUnaryComposite(
      [alpha](ynn_subgraph_t g, uint32_t in, uint32_t& out) {
        return define_elu(g, in, alpha, out);
      },
      [alpha](float x) { return elu(x, alpha); });
}

TEST(Composites, LeakyRelu) {
  const float alpha = 0.2f;
  VerifyUnaryComposite(
      [alpha](ynn_subgraph_t g, uint32_t in, uint32_t& out) {
        return define_leaky_relu(g, in, alpha, out);
      },
      [alpha](float x) { return leaky_relu(x, alpha); });
}

TEST(Composites, Hardswish) {
  VerifyUnaryComposite(
      [](ynn_subgraph_t g, uint32_t in, uint32_t& out) {
        return define_hardswish(g, in, out);
      },
      hardswish);
}

}  // namespace
}  // namespace ynn
