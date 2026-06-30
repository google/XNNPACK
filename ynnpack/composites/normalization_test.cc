// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/type.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

std::vector<float> reference_softmax(const std::vector<float>& x, float beta) {
  size_t n = x.size();
  std::vector<float> y(n);
  float max_val = *std::max_element(x.begin(), x.end());
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    y[i] = std::exp(beta * (x[i] - max_val));
    sum += y[i];
  }
  for (size_t i = 0; i < n; ++i) {
    y[i] /= sum;
  }
  return y;
}

std::vector<float> reference_log_softmax(const std::vector<float>& x) {
  size_t n = x.size();
  std::vector<float> y(n);
  float max_val = *std::max_element(x.begin(), x.end());
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum += std::exp(x[i] - max_val);
  }
  float log_sum = std::log(sum);
  for (size_t i = 0; i < n; ++i) {
    y[i] = x[i] - max_val - log_sum;
  }
  return y;
}

template <typename T, typename Define, typename Ref>
void VerifyNormalizationComposite(Define define_fn, Ref ref_fn) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  ynn_type tensor_type = type_of<T>();

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), tensor_type, 1, nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), tensor_type, 1, nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id),
            ynn_status_success);

  ASSERT_EQ(define_fn(subgraph.get(), x_id, y_id), ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), /*threadpool=*/nullptr,
                                  /*flags=*/0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  const size_t n = 10;
  std::vector<float> x_fp32(n);
  for (size_t i = 0; i < n; ++i) {
    x_fp32[i] = static_cast<float>(i) - 5.0f;
  }

  size_t shape[] = {n};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 1, shape),
            ynn_status_success);

  std::vector<T> x(n);
  std::copy_n(x_fp32.begin(), n, x.begin());
  std::vector<T> y(n);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y.data()),
            ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  std::vector<float> expected = ref_fn(x_fp32);
  std::vector<float> y_fp32(n);
  std::copy_n(y.begin(), n, y_fp32.begin());

  float tolerance = 1e-5f;
  if constexpr (std::is_same_v<T, half>) {
    tolerance = 1e-2f;
  } else if constexpr (std::is_same_v<T, bfloat16>) {
    tolerance = 5e-2f;
  }

  EXPECT_THAT(y_fp32,
              ::testing::Pointwise(::testing::FloatNear(tolerance), expected));
}

template <typename Define, typename Ref>
void VerifyNormalizationComposite(ynn_type tensor_type, Define define_fn,
                                  Ref ref_fn) {
  if (tensor_type == ynn_type_fp32) {
    VerifyNormalizationComposite<float>(define_fn, ref_fn);
  } else if (tensor_type == ynn_type_fp16) {
    VerifyNormalizationComposite<half>(define_fn, ref_fn);
  } else if (tensor_type == ynn_type_bf16) {
    VerifyNormalizationComposite<bfloat16>(define_fn, ref_fn);
  } else {
    FAIL() << "Unsupported tensor type";
  }
}

class SoftmaxTest : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {
};

TEST_P(SoftmaxTest, Verify) {
  ynn_type tensor_type = std::get<0>(GetParam());
  float beta = std::get<1>(GetParam());

  VerifyNormalizationComposite(
      tensor_type,
      [beta](ynn_subgraph_t g, uint32_t in, uint32_t& out) {
        return define_softmax(g, in, beta, out);
      },
      [beta](const std::vector<float>& x) {
        return reference_softmax(x, beta);
      });
}

INSTANTIATE_TEST_SUITE_P(
    Softmax, SoftmaxTest,
    ::testing::Combine(::testing::Values(ynn_type_fp32, ynn_type_fp16,
                                         ynn_type_bf16),
                       ::testing::Values(1, 2)),
    [](const testing::TestParamInfo<SoftmaxTest::ParamType>& info) {
      ynn_type tensor_type = std::get<0>(info.param);
      int beta = std::get<1>(info.param);
      return std::string(to_string(tensor_type)) + "_beta_" +
             std::to_string(beta);
    });

class LogSoftmaxTest : public ::testing::TestWithParam<ynn_type> {};

TEST_P(LogSoftmaxTest, Verify) {
  ynn_type tensor_type = GetParam();

  VerifyNormalizationComposite(
      tensor_type,
      [](ynn_subgraph_t g, uint32_t in, uint32_t& out) {
        return define_log_softmax(g, in, out);
      },
      reference_log_softmax);
}

INSTANTIATE_TEST_SUITE_P(
    LogSoftmax, LogSoftmaxTest,
    ::testing::Values(ynn_type_fp32, ynn_type_fp16, ynn_type_bf16),
    [](const testing::TestParamInfo<LogSoftmaxTest::ParamType>& info) {
      return to_string(info.param);
    });

}  // namespace
}  // namespace ynn
