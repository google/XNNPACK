// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "unary-operator-tester.h"
#include "pthreadpool.h"

namespace xnnpack {

class LeakyReLUOperatorTester : public UnaryOperatorTester {
 public:
  LeakyReLUOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-20.0f, 20.0f};
    range_f16_ = {-25.0f, 25.0f};
  }

  inline LeakyReLUOperatorTester& alpha(float alpha) {
    assert(alpha > 0.0f);
    assert(alpha < 1.0f);
    this->alpha_ = alpha;
    return *this;
  }

  inline float alpha() const { return this->alpha_; }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override {
    return std::signbit(x) ? std::expm1(x) * alpha() : x;
  }

  // Computes the absolute tolerance for a reference value `y_ref`. Tests will
  // fail when `std::abs(y - y_ref) > AbsTol32(y_ref)`. Note that for `fp16`
  // tests, both `y` and `y_ref` will be converted to `float` for the tolerance
  // evaluation.
  float AbsTolF32(float) const override { return 5e-6f; }
  float AbsTolF16(float y_ref) const override {
    return std::max(1.0e-4f, std::abs(y_ref) * 5.0e-3f);
  }
  float AbsTolQS8(float) const override { return 0.6f; };
  float AbsTolQU8(float) const override { return 0.6f; };

  xnn_status CreateOpF32(uint32_t flags,
                         xnn_operator_t* op_out) const override {
    return xnn_create_elu_nc_f32(alpha(), flags, op_out);
  }
  CREATE_OP_RESHAPE_OVERRIDE_F32(elu);
  CREATE_OP_SETUP_OVERRIDE_F32(elu);
  xnn_status RunOpF32(size_t channels, size_t input_stride,
                      size_t output_stride, size_t batch_size,
                      const float* input, float* output, uint32_t flags,
                      pthreadpool_t threadpool) const override {
    return xnn_run_elu_nc_f32(channels, input_stride, output_stride, batch_size,
                              input, output, alpha(), flags, threadpool);
  }

  xnn_status CreateOpF16(uint32_t flags,
                         xnn_operator_t* op_out) const override {
    return xnn_create_elu_nc_f16(alpha(), flags, op_out);
  }
  CREATE_OP_RESHAPE_OVERRIDE_F16(elu);
  CREATE_OP_SETUP_OVERRIDE_F16(elu);

  xnn_status CreateOpQS8(int8_t input_zero_point, float input_scale,
                         int8_t output_zero_point, float output_scale,
                         int8_t output_min, int8_t output_max, uint32_t flags,
                         xnn_operator_t* op_out) const override {
    return xnn_create_elu_nc_qs8(alpha(), input_zero_point, input_scale,
                                 output_zero_point, output_scale, output_min,
                                 output_max, flags, op_out);
  }
  CREATE_OP_RESHAPE_OVERRIDE_QS8(elu);
  CREATE_OP_SETUP_OVERRIDE_QS8(elu);

 private:
  float alpha_ = 0.5f;
};

CREATE_UNARY_FLOAT_TESTS(F32, LeakyReLUOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, LeakyReLUOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, LeakyReLUOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

CREATE_UNARY_QUANTIZED_TESTS(QS8, LeakyReLUOperatorTester);

#ifndef XNN_EXCLUDE_F16_TESTS
TEST(ELU_NC_F16, small_batch_with_alpha) {
  for (size_t batch_size = 1; batch_size <= 3; batch_size += 2) {
    for (size_t channels = 1; channels < 100; channels += 15) {
      for (float alpha = 1.0e-4f; alpha < 1.0f; alpha *= 3.14159265f) {
        LeakyReLUOperatorTester()
            .alpha(alpha)
            .batch_size(3)
            .channels(channels)
            .iterations(1)
            .TestF16();
      }
    }
  }
}
#endif  // XNN_EXCLUDE_F16_TESTS

TEST(ELU_NC_F32, small_batch_with_alpha) {
  for (size_t batch_size = 1; batch_size <= 3; batch_size += 2) {
    for (size_t channels = 1; channels < 100; channels += 15) {
      for (float alpha = 1.0e-4f; alpha < 1.0f; alpha *= 3.14159265f) {
        LeakyReLUOperatorTester()
            .alpha(alpha)
            .batch_size(3)
            .channels(channels)
            .iterations(1)
            .TestF32();
      }
    }
  }
}

TEST(ELU_NC_QS8, small_batch_with_alpha) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float alpha = 1.0e-4f; alpha < 1.0f; alpha *= 3.14159265f) {
      LeakyReLUOperatorTester()
          .alpha(alpha)
          .batch_size(3)
          .channels(channels)
          .iterations(1)
          .TestQS8();
    }
  }
}

TEST(ELU_NC_QS8, strided_batch_with_alpha) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float alpha = 1.0e-4f; alpha < 1.0f; alpha *= 3.14159265f) {
      LeakyReLUOperatorTester()
          .alpha(alpha)
          .batch_size(3)
          .channels(channels)
          .input_stride(129)
          .output_stride(117)
          .iterations(1)
          .TestQS8();
    }
  }
}
};  // namespace xnnpack
