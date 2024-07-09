// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "unary-operator-tester.h"
#include "pthreadpool.h"

namespace xnnpack {

#define xnn_reshape_clamp_nc_qs8 xnn_reshape_clamp_nc_s8
#define xnn_reshape_clamp_nc_qu8 xnn_reshape_clamp_nc_u8
#define xnn_setup_clamp_nc_qs8 xnn_setup_clamp_nc_s8
#define xnn_setup_clamp_nc_qu8 xnn_setup_clamp_nc_u8

class ClampOperatorTester : public UnaryOperatorTester {
 public:
  ClampOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-10.0f, 10.0f};
    range_f16_ = {-10.0f, 10.0f};
    input_scale(1.0f);
    input_zero_point(128);
    output_scale(1.0f);
    output_zero_point(128);
  }

  ClampOperatorTester& relu_activation(bool relu_activation) {
    relu_activation_ = relu_activation;
    return *this;
  }
  ClampOperatorTester& clamp_low(bool clamp_low) {
    clamp_low_ = clamp_low;
    return *this;
  }
  ClampOperatorTester& clamp_high(bool clamp_high) {
    clamp_high_ = clamp_high;
    return *this;
  }

  bool relu_activation() const { return relu_activation_; }
  float clamp_low() const { return clamp_low_; }
  float clamp_high() const { return clamp_high_; }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override {
    return relu_activation() ? std::max(x, 0.f)
                             : std::min(std::max(x, clamp_low()), clamp_high());
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
    const float output_min = relu_activation() ? 0.0f : clamp_low();
    const float output_max = relu_activation()
                                 ? std::numeric_limits<float>::infinity()
                                 : clamp_high();
    return xnn_create_clamp_nc_f32(output_min, output_max, 0, op_out);
  }

  xnn_status RunOpF32(size_t channels, size_t input_stride,
                      size_t output_stride, size_t batch_size,
                      const float* input, float* output, uint32_t flags,
                      pthreadpool_t threadpool) const override {
    const float output_min = relu_activation() ? 0.0f : clamp_low();
    const float output_max = relu_activation()
                                 ? std::numeric_limits<float>::infinity()
                                 : clamp_high();

    return xnn_run_clamp_nc_f32(channels, input_stride, output_stride,
                                batch_size, input, output, output_min,
                                output_max, flags, threadpool);
  }
  xnn_status CreateOpF16(uint32_t flags,
                         xnn_operator_t* op_out) const override {
    const float output_min = relu_activation() ? 0.0f : clamp_low();
    const float output_max = relu_activation()
                                 ? std::numeric_limits<float>::infinity()
                                 : clamp_high();
    return xnn_create_clamp_nc_f16(output_min, output_max, 0, op_out);
  }
  xnn_status CreateOpQS8(int8_t input_zero_point, float input_scale,
                         int8_t output_zero_point, float output_scale,
                         int8_t output_min, int8_t output_max, uint32_t flags,
                         xnn_operator_t* op_out) const override {
    int8_t q_low =
        static_cast<int8_t>(clamp_low() / output_scale + output_zero_point);
    int8_t q_high =
        static_cast<int8_t>(clamp_high() / output_scale + output_zero_point);
    return xnn_create_clamp_nc_s8(q_low, q_high, 0, op_out);
  }
  xnn_status CreateOpQU8(uint8_t input_zero_point, float input_scale,
                         uint8_t output_zero_point, float output_scale,
                         uint8_t output_min, uint8_t output_max, uint32_t flags,
                         xnn_operator_t* op_out) const override {
    uint8_t q_low =
        static_cast<uint8_t>(clamp_low() / output_scale + output_zero_point);
    uint8_t q_high =
        static_cast<uint8_t>(clamp_high() / output_scale + output_zero_point);
    return xnn_create_clamp_nc_u8(q_low, q_high, 0, op_out);
  }

  CREATE_OP_RESHAPE_OVERRIDE_F32(clamp);
  CREATE_OP_SETUP_OVERRIDE_F32(clamp);

  CREATE_OP_RESHAPE_OVERRIDE_F16(clamp);
  CREATE_OP_SETUP_OVERRIDE_F16(clamp);

  CREATE_OP_RESHAPE_OVERRIDE_QS8(clamp);
  CREATE_OP_SETUP_OVERRIDE_QS8(clamp);

  CREATE_OP_RESHAPE_OVERRIDE_QU8(clamp);
  CREATE_OP_SETUP_OVERRIDE_QU8(clamp);

 private:
  bool relu_activation_ = false;
  float clamp_low_ = -5.0f;
  float clamp_high_ = 5.0f;
};

CREATE_UNARY_FLOAT_TESTS(F32, ClampOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, ClampOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, ClampOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

CREATE_UNARY_TEST(QS8, ClampOperatorTester)
INSTANTIATE_TEST_SUITE_P(
    datatype, ClampOperatorTesterQS8,
    testing::ValuesIn<UnaryOpTestParams>({
        UnaryOpTestParams::UnitBatch(),
        UnaryOpTestParams::SmallBatch(),
        UnaryOpTestParams::SmallBatch().InputStride(129),
        UnaryOpTestParams::SmallBatch().OutputStride(117),
    }),
    [](const testing::TestParamInfo<ClampOperatorTesterQS8::ParamType>& info) {
      return info.param.ToString();
    });

CREATE_UNARY_TEST(QU8, ClampOperatorTester)
INSTANTIATE_TEST_SUITE_P(
    datatype, ClampOperatorTesterQU8,
    testing::ValuesIn<UnaryOpTestParams>({
        UnaryOpTestParams::UnitBatch(),
        UnaryOpTestParams::SmallBatch(),
        UnaryOpTestParams::SmallBatch().InputStride(129),
        UnaryOpTestParams::SmallBatch().OutputStride(117),
    }),
    [](const testing::TestParamInfo<ClampOperatorTesterQU8::ParamType>& info) {
      return info.param.ToString();
    });

#ifndef XNN_EXCLUDE_F16_TESTS
TEST(CLAMP_NC_F16, unit_batch_with_clamp_min) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t clamp_low = std::numeric_limits<int16_t>::min() + 16;
         clamp_low < std::numeric_limits<int16_t>::max() - 16;
         clamp_low += 257) {
      ClampOperatorTester()
          .clamp_low(clamp_low)
          .batch_size(1)
          .channels(channels)
          .iterations(3)
          .TestF16();
    }
  }
}
TEST(CLAMP_NC_F16, unit_batch_with_clamp_max) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t clamp_high = std::numeric_limits<int16_t>::min() + 16;
         clamp_high < std::numeric_limits<int16_t>::max() - 16;
         clamp_high += 257) {
      ClampOperatorTester()
          .clamp_high(clamp_high)
          .batch_size(1)
          .channels(channels)
          .iterations(3)
          .TestF16();
    }
  }
}
#endif  // XNN_EXCLUDE_F16_TESTS

TEST(CLAMP_NC_F32, unit_batch_with_clamp_low) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t clamp_low = std::numeric_limits<int16_t>::min() + 1;
         clamp_low < std::numeric_limits<int16_t>::max(); clamp_low += 257) {
      ClampOperatorTester()
          .clamp_low(clamp_low)
          .batch_size(1)
          .channels(channels)
          .iterations(3)
          .TestF32();
    }
  }
}
TEST(CLAMP_NC_F32, unit_batch_with_clamp_high) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t clamp_high = std::numeric_limits<int16_t>::min() + 1;
         clamp_high < std::numeric_limits<int16_t>::max(); clamp_high += 257) {
      ClampOperatorTester()
          .clamp_high(clamp_high)
          .batch_size(1)
          .channels(channels)
          .iterations(3)
          .TestF32();
    }
  }
}
TEST(CLAMP_NC_F32, unit_batch_with_relu) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
        .relu_activation(true)
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestF32();
  }
}

TEST(CLAMP_NC_S8, unit_batch_with_clamp_low) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t clamp_low = std::numeric_limits<int8_t>::min() + 1;
         clamp_low < std::numeric_limits<int8_t>::max(); clamp_low++) {
      ClampOperatorTester()
          .clamp_low(clamp_low)
          .batch_size(1)
          .channels(channels)
          .iterations(3)
          .TestQS8();
    }
  }
}
TEST(CLAMP_NC_S8, unit_batch_with_clamp_high) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t clamp_high = std::numeric_limits<int8_t>::min() + 1;
         clamp_high < std::numeric_limits<int8_t>::max(); clamp_high++) {
      ClampOperatorTester()
          .clamp_high(clamp_high)
          .batch_size(1)
          .channels(channels)
          .iterations(3)
          .TestQS8();
    }
  }
}

TEST(CLAMP_NC_U8, unit_batch_with_clamp_low) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t clamp_low = std::numeric_limits<uint8_t>::min() + 1;
         clamp_low < std::numeric_limits<uint8_t>::max(); clamp_low++) {
      ClampOperatorTester()
          .clamp_low(clamp_low)
          .batch_size(1)
          .channels(channels)
          .iterations(3)
          .TestQU8();
    }
  }
}
TEST(CLAMP_NC_U8, unit_batch_with_clamp_high) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t clamp_high = 1; clamp_high < 255; clamp_high++) {
      ClampOperatorTester()
          .clamp_high(clamp_high)
          .batch_size(1)
          .channels(channels)
          .iterations(3)
          .TestQU8();
    }
  }
}

};  // namespace xnnpack
