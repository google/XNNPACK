// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include "unary-operator-tester.h"
#include "pthreadpool.h"
namespace xnnpack {

class LeakyReLUOperatorTester : public UnaryOperatorTester {
 public:
  LeakyReLUOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-20.0f, 20.0f};
    range_f16_ = {-25.0f, 25.0f};
    input_scale(1.25f);
    input_zero_point(41);
    output_scale(0.75f);
    output_zero_point(53);
  }

  LeakyReLUOperatorTester& negative_slope(float negative_slope) {
    assert(std::isnormal(negative_slope));
    this->negative_slope_ = negative_slope;
    return *this;
  }

  inline float negative_slope() const { return this->negative_slope_; }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override {
    return std::signbit(x) ? x * negative_slope() : x;
  }

  // Computes the absolute tolerance for a reference value `y_ref`. Tests will
  // fail when `std::abs(y - y_ref) > AbsTol32(y_ref)`. Note that for `fp16`
  // tests, both `y` and `y_ref` will be converted to `float` for the tolerance
  // evaluation.
  float AbsTolF32(float) const override { return 5e-6f; }
  float AbsTolF16(float y_ref) const override {
    return std::max(1.0e-4f, std::abs(y_ref) * 5.0e-3f);
  }
  float AbsTolQS8(float) const override { return 0.9f; };
  float AbsTolQU8(float) const override { return 0.9f; };

  xnn_status CreateOpF32(uint32_t flags,
                         xnn_operator_t* op_out) const override {
    return xnn_create_leaky_relu_nc_f32(negative_slope(), flags, op_out);
  }
  CREATE_OP_RESHAPE_OVERRIDE_F32(leaky_relu);
  CREATE_OP_SETUP_OVERRIDE_F32(leaky_relu);
  xnn_status RunOpF32(size_t channels, size_t input_stride,
                      size_t output_stride, size_t batch_size,
                      const float* input, float* output, uint32_t flags,
                      pthreadpool_t threadpool) const override {
    return xnn_run_leaky_relu_nc_f32(channels, input_stride, output_stride,
                                     batch_size, input, output,
                                     negative_slope(), flags, threadpool);
  }

  xnn_status CreateOpF16(uint32_t flags,
                         xnn_operator_t* op_out) const override {
    return xnn_create_leaky_relu_nc_f16(negative_slope(), flags, op_out);
  }
  CREATE_OP_RESHAPE_OVERRIDE_F16(leaky_relu);
  CREATE_OP_SETUP_OVERRIDE_F16(leaky_relu);

  xnn_status CreateOpQS8(int8_t input_zero_point, float input_scale,
                         int8_t output_zero_point, float output_scale,
                         int8_t output_min, int8_t output_max, uint32_t flags,
                         xnn_operator_t* op_out) const override {
    return xnn_create_leaky_relu_nc_qs8(negative_slope(), input_zero_point,
                                        input_scale, output_zero_point,
                                        output_scale, flags, op_out);
  }
  CREATE_OP_RESHAPE_OVERRIDE_QS8(leaky_relu);
  CREATE_OP_SETUP_OVERRIDE_QS8(leaky_relu);

  xnn_status CreateOpQU8(uint8_t input_zero_point, float input_scale,
                         uint8_t output_zero_point, float output_scale,
                         uint8_t output_min, uint8_t output_max, uint32_t flags,
                         xnn_operator_t* op_out) const override {
    return xnn_create_leaky_relu_nc_qu8(negative_slope(), input_zero_point,
                                        input_scale, output_zero_point,
                                        output_scale, flags, op_out);
  }
  CREATE_OP_RESHAPE_OVERRIDE_QU8(leaky_relu);
  CREATE_OP_SETUP_OVERRIDE_QU8(leaky_relu);

 private:
  float negative_slope_ = 0.3f;
};

CREATE_UNARY_FLOAT_TESTS(F32, LeakyReLUOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, LeakyReLUOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, LeakyReLUOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

CREATE_UNARY_QUANTIZED_TESTS_NO_QMIN(QS8, LeakyReLUOperatorTester);
CREATE_UNARY_QUANTIZED_TESTS_NO_QMIN(QU8, LeakyReLUOperatorTester);

#ifndef XNN_EXCLUDE_F16_TESTS
TEST(LEAKY_RELU_NC_F16, small_batch_with_negative_slope) {
  for (size_t batch_size = 1; batch_size <= 3; batch_size += 2) {
    for (size_t channels = 1; channels < 100; channels += 15) {
      for (float negative_slope :
           std::vector<float>({-10.0f, -1.0f, -0.1f, 0.1f, 10.0f})) {
        LeakyReLUOperatorTester()
            .negative_slope(negative_slope)
            .batch_size(3)
            .channels(channels)
            .iterations(1)
            .TestF16();
      }
    }
  }
}
#endif  // XNN_EXCLUDE_F16_TESTS

TEST(LEAKY_RELU_NC_F32, small_batch_with_negative_slope) {
  for (size_t batch_size = 1; batch_size <= 3; batch_size += 2) {
    for (size_t channels = 1; channels < 100; channels += 15) {
      for (float negative_slope :
           std::vector<float>({-10.0f, -1.0f, -0.1f, 0.1f, 10.0f})) {
        LeakyReLUOperatorTester()
            .negative_slope(negative_slope)
            .batch_size(3)
            .channels(channels)
            .iterations(1)
            .TestF32();
      }
    }
  }
}

TEST(LEAKY_RELU_NC_QS8, unit_batch_with_negative_slope) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float negative_slope :
         std::vector<float>({-10.0f, -1.0f, -0.1f, 0.1f, 10.0f})) {
      LeakyReLUOperatorTester()
          .negative_slope(negative_slope)
          .batch_size(1)
          .channels(channels)
          .iterations(1)
          .TestQS8();
    }
  }
}

TEST(LEAKY_RELU_NC_QS8, unit_batch_with_output_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float output_scale = 1.0e-2f; output_scale < 1.0e+2f;
         output_scale *= 3.14159265f) {
      LeakyReLUOperatorTester()
          .output_scale(output_scale)
          .batch_size(1)
          .channels(channels)
          .iterations(1)
          .TestQS8();
    }
  }
}

TEST(LEAKY_RELU_NC_QS8, unit_batch_with_output_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int16_t output_zero_point = 0; output_zero_point <= 255;
         output_zero_point += 51) {
      LeakyReLUOperatorTester()
          .output_zero_point(output_zero_point)
          .batch_size(1)
          .channels(channels)
          .iterations(1)
          .TestQS8();
    }
  }
}

TEST(LEAKY_RELU_NC_QU8, unit_batch_with_negative_slope) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float negative_slope :
         std::vector<float>({-10.0f, -1.0f, -0.1f, 0.1f, 10.0f})) {
      LeakyReLUOperatorTester()
          .negative_slope(negative_slope)
          .batch_size(1)
          .channels(channels)
          .iterations(1)
          .TestQU8();
    }
  }
}

TEST(LEAKY_RELU_NC_QU8, unit_batch_with_output_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float output_scale = 1.0e-2f; output_scale < 1.0e+2f;
         output_scale *= 3.14159265f) {
      LeakyReLUOperatorTester()
          .output_scale(output_scale)
          .batch_size(1)
          .channels(channels)
          .iterations(1)
          .TestQU8();
    }
  }
}

TEST(LEAKY_RELU_NC_QU8, unit_batch_with_output_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int16_t output_zero_point = 0; output_zero_point <= 255;
         output_zero_point += 51) {
      LeakyReLUOperatorTester()
          .output_zero_point(output_zero_point)
          .batch_size(1)
          .channels(channels)
          .iterations(1)
          .TestQU8();
    }
  }
}

};  // namespace xnnpack
