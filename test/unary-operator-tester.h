// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_TEST_UNARY_OPERATOR_TESTER_H_
#define __XNNPACK_TEST_UNARY_OPERATOR_TESTER_H_

#include <sys/types.h>
#include <xnnpack.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "pthreadpool.h"

namespace xnnpack {

class UnaryOperatorTester {
 public:
  virtual ~UnaryOperatorTester() = default;

  UnaryOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    channels_ = channels;
    return *this;
  }

  UnaryOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    input_stride_ = input_stride;
    return *this;
  }

  UnaryOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    output_stride_ = output_stride;
    return *this;
  }

  UnaryOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    batch_size_ = batch_size;
    return *this;
  }

  UnaryOperatorTester& iterations(size_t iterations) {
    iterations_ = iterations;
    return *this;
  }

  UnaryOperatorTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    input_scale_ = input_scale;
    return *this;
  }

  UnaryOperatorTester& input_zero_point(int16_t input_zero_point) {
    input_zero_point_ = input_zero_point;
    return *this;
  }

  UnaryOperatorTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    output_scale_ = output_scale;
    return *this;
  }

  UnaryOperatorTester& output_zero_point(int16_t output_zero_point) {
    output_zero_point_ = output_zero_point;
    return *this;
  }

  UnaryOperatorTester& qmin(int16_t qmin) {
    qmin_ = qmin;
    return *this;
  }

  UnaryOperatorTester& qmax(int16_t qmax) {
    qmax_ = qmax;
    return *this;
  }

  size_t channels() const { return channels_; }

  size_t input_stride() const {
    if (input_stride_ == 0) {
      return channels_;
    } else {
      assert(input_stride_ >= channels_);
      return input_stride_;
    }
  }

  size_t output_stride() const {
    if (output_stride_ == 0) {
      return channels_;
    } else {
      assert(output_stride_ >= channels_);
      return output_stride_;
    }
  }

  size_t batch_size() const { return batch_size_; }

  size_t iterations() const { return iterations_; }

  float input_scale() const { return input_scale_; }

  int16_t input_zero_point() const { return input_zero_point_; }

  float output_scale() const { return output_scale_; }

  int16_t output_zero_point() const { return output_zero_point_; }

  int16_t qmin() const { return qmin_; }

  int16_t qmax() const { return qmax_; }

  // Converters between float and quantized types.
  float FloatFromInputQS8(int8_t x) const {
    return input_scale() * (static_cast<int32_t>(x) -
                            static_cast<int32_t>(input_zero_point() - 0x80));
  }
  float FloatFromInputQU8(uint8_t x) const {
    return input_scale() *
           (static_cast<int32_t>(x) - static_cast<int32_t>(input_zero_point()));
  }
  float QuantizeAsFloatQS8(float x) const {
    float y =
        x / output_scale() + static_cast<int32_t>(output_zero_point() - 0x80);
    y = std::min<float>(y, qmax() - 0x80);
    y = std::max<float>(y, qmin() - 0x80);
    return y;
  }
  float QuantizeAsFloatQU8(float x) const {
    float y = x / output_scale() + static_cast<int32_t>(output_zero_point());
    y = std::min<float>(y, qmax());
    y = std::max<float>(y, qmin());
    return y;
  }

  virtual void TestF16();
  virtual void TestF32();
  virtual void TestRunF32();
  virtual void TestQS8();
  virtual void TestQU8();

 protected:
  UnaryOperatorTester() = default;

  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  virtual float RefFunc(float x) const = 0;

  // Computes the absolute tolerance for a reference value `y_ref`. Tests will
  // fail when `std::abs(y - y_ref) > AbsTol32(y_ref)`.
  // Note that for `fp16` tests, both `y` and `y_ref` will be converted to
  // `float` for the tolerance evaluation.
  virtual float AbsTolF32(float y_ref) const { return 0.0f; };
  virtual float AbsTolF16(float y_ref) const { return 0.0f; };

  // For the `QSU` and `QU8` tests, `y_ref` is the reference value transformed
  // to the quantization range, e.g. `[qmin(), qmax()]` for `QU8` (see
  // `QuantizeAsFloatQS8` and `QuantizeAsFloatQU8`).
  virtual float AbsTolQS8(float y_ref) const { return 0.0f; };
  virtual float AbsTolQU8(float y_ref) const { return 0.0f; };

  // Check the results for each datatype. Override these functions to perform
  // additional checks.
  virtual void CheckResultF32(float y, float y_ref, size_t batch,
                              size_t channel, float input) const {
    EXPECT_NEAR(y_ref, y, AbsTolF32(y_ref))
        << "at batch " << batch << " / " << batch_size() << ", channel "
        << channel << " / " << channels() << ", input " << input;
  }
  virtual void CheckResultF16(uint16_t y, float y_ref, size_t batch,
                              size_t channel, uint16_t input) const {
    EXPECT_NEAR(y_ref, fp16_ieee_to_fp32_value(y), AbsTolF16(y_ref))
        << "at batch " << batch << " / " << batch_size() << ", channel "
        << channel << " / " << channels() << ", input "
        << fp16_ieee_to_fp32_value(input);
  }
  virtual void CheckResultQS8(int8_t y, float y_ref, size_t batch,
                              size_t channel, int8_t input) const {
    EXPECT_NEAR(y_ref, static_cast<float>(y), AbsTolQS8(y_ref))
        << "at batch " << batch << " / " << batch_size() << ", channel "
        << channel << " / " << channels() << ", input "
        << static_cast<int32_t>(input) << " (" << FloatFromInputQS8(input)
        << ")";
  }
  virtual void CheckResultQU8(uint8_t y, float y_ref, size_t batch,
                              size_t channel, uint8_t input) const {
    EXPECT_NEAR(y_ref, static_cast<float>(y), AbsTolQU8(y_ref))
        << "at batch " << batch << " / " << batch_size() << ", channel "
        << channel << " / " << channels() << ", input "
        << static_cast<int32_t>(input) << " (" << FloatFromInputQU8(input)
        << ")";
  }

  // Wrappers for the create/reshape/setup/run functions of the underlying `f32`
  // op, override these with calls to the actual op functions, e.g. using the
  // `CREATE_OP_OVERRIDES_F32` macro defined below.
  virtual xnn_status CreateOpF32(uint32_t flags, xnn_operator_t* op_out) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status ReshapeOpF32(xnn_operator_t op, size_t batch_size,
                                  size_t channels, size_t input_stride,
                                  size_t output_stride,
                                  pthreadpool_t threadpool) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status SetupOpF32(xnn_operator_t op, const float* input,
                                float* output) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status RunOpF32(size_t channels, size_t input_stride,
                              size_t output_stride, size_t batch_size,
                              const float* input, float* output, uint32_t flags,
                              pthreadpool_t threadpool) const {
    return xnn_status_invalid_parameter;
  }

  // Wrappers for the create/reshape/setup functions of the underlying `f16`
  // op, override these with calls to the actual op functions, e.g. using the
  // `CREATE_OP_OVERRIDES_F16` macro defined below.
  virtual xnn_status CreateOpF16(uint32_t flags, xnn_operator_t* op_out) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status ReshapeOpF16(xnn_operator_t op, size_t batch_size,
                                  size_t channels, size_t input_stride,
                                  size_t output_stride,
                                  pthreadpool_t threadpool) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status SetupOpF16(xnn_operator_t op, const void* input,
                                void* output) const {
    return xnn_status_invalid_parameter;
  }

  // Wrappers for the create/reshape/setup functions of the underlying `qs8`
  // op, override these with calls to the actual op functions, e.g. using the
  // `CREATE_OP_OVERRIDES_QS8` macro defined below.
  virtual xnn_status CreateOpQS8(int8_t input_zero_point, float input_scale,
                                 int8_t output_zero_point, float output_scale,
                                 int8_t output_min, int8_t output_max,
                                 uint32_t flags, xnn_operator_t* op_out) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status ReshapeOpQS8(xnn_operator_t op, size_t batch_size,
                                  size_t channels, size_t input_stride,
                                  size_t output_stride,
                                  pthreadpool_t threadpool) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status SetupOpQS8(xnn_operator_t op, const int8_t* input,
                                int8_t* output) const {
    return xnn_status_invalid_parameter;
  }

  // Wrappers for the create/reshape/setup functions of the underlying `qu8`
  // op, override these with calls to the actual op functions, e.g. using the
  // `CREATE_OP_OVERRIDES_QU8` macro defined below.
  virtual xnn_status CreateOpQU8(uint8_t input_zero_point, float input_scale,
                                 uint8_t output_zero_point, float output_scale,
                                 uint8_t output_min, uint8_t output_max,
                                 uint32_t flags, xnn_operator_t* op_out) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status ReshapeOpQU8(xnn_operator_t op, size_t batch_size,
                                  size_t channels, size_t input_stride,
                                  size_t output_stride,
                                  pthreadpool_t threadpool) const {
    return xnn_status_invalid_parameter;
  }
  virtual xnn_status SetupOpQU8(xnn_operator_t op, const uint8_t* input,
                                uint8_t* output) const {
    return xnn_status_invalid_parameter;
  }

  // Input ranges for the different type-dependent tests.
  std::pair<float, float> range_f32_ = {-10.0f, 10.0f};
  std::pair<float, float> range_f16_ = {-10.0f, 10.0f};
  std::pair<float, int32_t> range_qs8_ = {std::numeric_limits<int8_t>::min(),
                                          std::numeric_limits<int8_t>::max()};
  std::pair<float, int32_t> range_qu8_ = {0,
                                          std::numeric_limits<uint8_t>::max()};

 private:
  size_t batch_size_ = 1;
  size_t channels_ = 1;
  size_t input_stride_ = 0;
  size_t output_stride_ = 0;
  float input_scale_ = 0.75f;
  int16_t input_zero_point_ = 121;
  float output_scale_ = 1.0f / 128.0f;
  int16_t output_zero_point_ = 128;
  int16_t qmin_ = 0;
  int16_t qmax_ = 255;
  size_t iterations_ = 15;
};

#define CREATE_OP_CREATE_OVERRIDE_F32(op_name)                   \
  xnn_status CreateOpF32(uint32_t flags, xnn_operator_t* op_out) \
      const override {                                           \
    return xnn_create_##op_name##_nc_f32(flags, op_out);         \
  }

#define CREATE_OP_RESHAPE_OVERRIDE_F32(op_name)                             \
  xnn_status ReshapeOpF32(xnn_operator_t op, size_t batch_size,             \
                          size_t channels, size_t input_stride,             \
                          size_t output_stride, pthreadpool_t threadpool)   \
      const override {                                                      \
    return xnn_reshape_##op_name##_nc_f32(                                  \
        op, batch_size, channels, input_stride, output_stride, threadpool); \
  }

#define CREATE_OP_SETUP_OVERRIDE_F32(op_name)                                 \
  xnn_status SetupOpF32(xnn_operator_t op, const float* input, float* output) \
      const override {                                                        \
    return xnn_setup_##op_name##_nc_f32(op, input, output);                   \
  }

#define CREATE_OP_RUN_OVERRIDE_F32(op_name)                                  \
  xnn_status RunOpF32(size_t channels, size_t input_stride,                  \
                      size_t output_stride, size_t batch_size,               \
                      const float* input, float* output, uint32_t flags,     \
                      pthreadpool_t threadpool) const override {             \
    return xnn_run_##op_name##_nc_f32(channels, input_stride, output_stride, \
                                      batch_size, input, output, flags,      \
                                      threadpool);                           \
  }

#define CREATE_OP_OVERRIDES_F32(op_name)   \
  CREATE_OP_CREATE_OVERRIDE_F32(op_name);  \
  CREATE_OP_RESHAPE_OVERRIDE_F32(op_name); \
  CREATE_OP_SETUP_OVERRIDE_F32(op_name);   \
  CREATE_OP_RUN_OVERRIDE_F32(op_name);

#define CREATE_OP_CREATE_OVERRIDE_F16(op_name)                   \
  xnn_status CreateOpF16(uint32_t flags, xnn_operator_t* op_out) \
      const override {                                           \
    return xnn_create_##op_name##_nc_f16(flags, op_out);         \
  }

#define CREATE_OP_RESHAPE_OVERRIDE_F16(op_name)                             \
  xnn_status ReshapeOpF16(xnn_operator_t op, size_t batch_size,             \
                          size_t channels, size_t input_stride,             \
                          size_t output_stride, pthreadpool_t threadpool)   \
      const override {                                                      \
    return xnn_reshape_##op_name##_nc_f16(                                  \
        op, batch_size, channels, input_stride, output_stride, threadpool); \
  }

#define CREATE_OP_SETUP_OVERRIDE_F16(op_name)                               \
  xnn_status SetupOpF16(xnn_operator_t op, const void* input, void* output) \
      const override {                                                      \
    return xnn_setup_##op_name##_nc_f16(op, input, output);                 \
  }

#define CREATE_OP_OVERRIDES_F16(op_name)   \
  CREATE_OP_CREATE_OVERRIDE_F16(op_name);  \
  CREATE_OP_RESHAPE_OVERRIDE_F16(op_name); \
  CREATE_OP_SETUP_OVERRIDE_F16(op_name);

#define CREATE_OP_CREATE_OVERRIDE_QS8(op_name)                                 \
  xnn_status CreateOpQS8(int8_t input_zero_point, float input_scale,           \
                         int8_t output_zero_point, float output_scale,         \
                         int8_t output_min, int8_t output_max, uint32_t flags, \
                         xnn_operator_t* op_out) const override {              \
    return xnn_create_##op_name##_nc_qs8(                                      \
        input_zero_point, input_scale, output_zero_point, output_scale,        \
        output_min, output_max, flags, op_out);                                \
  }

#define CREATE_OP_RESHAPE_OVERRIDE_QS8(op_name)                             \
  xnn_status ReshapeOpQS8(xnn_operator_t op, size_t batch_size,             \
                          size_t channels, size_t input_stride,             \
                          size_t output_stride, pthreadpool_t threadpool)   \
      const override {                                                      \
    return xnn_reshape_##op_name##_nc_qs8(                                  \
        op, batch_size, channels, input_stride, output_stride, threadpool); \
  }

#define CREATE_OP_SETUP_OVERRIDE_QS8(op_name)                   \
  xnn_status SetupOpQS8(xnn_operator_t op, const int8_t* input, \
                        int8_t* output) const override {        \
    return xnn_setup_##op_name##_nc_qs8(op, input, output);     \
  }

#define CREATE_OP_OVERRIDES_QS8(op_name)   \
  CREATE_OP_CREATE_OVERRIDE_QS8(op_name);  \
  CREATE_OP_RESHAPE_OVERRIDE_QS8(op_name); \
  CREATE_OP_SETUP_OVERRIDE_QS8(op_name);

#define CREATE_OP_CREATE_OVERRIDE_QU8(op_name)                                \
  xnn_status CreateOpQU8(                                                     \
      uint8_t input_zero_point, float input_scale, uint8_t output_zero_point, \
      float output_scale, uint8_t output_min, uint8_t output_max,             \
      uint32_t flags, xnn_operator_t* op_out) const override {                \
    return xnn_create_##op_name##_nc_qu8(                                     \
        input_zero_point, input_scale, output_zero_point, output_scale,       \
        output_min, output_max, flags, op_out);                               \
  }

#define CREATE_OP_RESHAPE_OVERRIDE_QU8(op_name)                             \
  xnn_status ReshapeOpQU8(xnn_operator_t op, size_t batch_size,             \
                          size_t channels, size_t input_stride,             \
                          size_t output_stride, pthreadpool_t threadpool)   \
      const override {                                                      \
    return xnn_reshape_##op_name##_nc_qu8(                                  \
        op, batch_size, channels, input_stride, output_stride, threadpool); \
  }

#define CREATE_OP_SETUP_OVERRIDE_QU8(op_name)                    \
  xnn_status SetupOpQU8(xnn_operator_t op, const uint8_t* input, \
                        uint8_t* output) const override {        \
    return xnn_setup_##op_name##_nc_qu8(op, input, output);      \
  }

#define CREATE_OP_OVERRIDES_QU8(op_name)   \
  CREATE_OP_CREATE_OVERRIDE_QU8(op_name);  \
  CREATE_OP_RESHAPE_OVERRIDE_QU8(op_name); \
  CREATE_OP_SETUP_OVERRIDE_QU8(op_name);

template <typename T>
struct LoopLimits {
  T min;
  T stride;
  T max;
  std::string ToString() const {
    return "[" + std::to_string(min) + ":" + std::to_string(stride) + ":" +
           std::to_string(max) + "]";
  }
};

// Mimics the behaviour of `std::optional`, which is only available as of C++17.
template <typename T>
class Optional {
 public:
  Optional() = default;
  explicit Optional(T value) : has_value_(true), value_(value) {}
  Optional& operator=(const T& other) {
    has_value_ = true;
    value_ = other;
    return *this;
  }

  // Clears the value if it was set.
  void reset() {
    if (has_value_) {
      value_ = T();
      has_value_ = false;
    }
  }

  // Accessors to check whether a value has been set.
  bool has_value() const { return has_value_; }
  explicit operator bool() const { return has_value_; }

  // Accessors to access the value.
  T& value() { return value_; }
  const T& value() const { return value_; }
  T& operator*() { return value_; }
  const T& operator*() const { return value_; }
  T* operator->() { return &value_; }
  const T* operator->() const { return &value_; }

 private:
  bool has_value_ = false;
  T value_;
};

struct UnaryOpTestParams {
  UnaryOpTestParams(std::string test_name_, size_t batch_size_,
                    LoopLimits<size_t> channels_)
      : test_name(test_name_), batch_size(batch_size_), channels(channels_) {}

  static UnaryOpTestParams UnitBatch() {
    return UnaryOpTestParams("unit_batch", 1, LoopLimits<size_t>{1, 100, 15});
  }
  static UnaryOpTestParams SmallBatch() {
    return UnaryOpTestParams("small_batch", 3, LoopLimits<size_t>{1, 100, 15});
  }
  static UnaryOpTestParams StridedBatch() {
    return UnaryOpTestParams("strided_batch", 3, LoopLimits<size_t>{1, 100, 15})
        .InputStride(129)
        .OutputStride(117);
  }
  UnaryOpTestParams& BatchSize(size_t batch_size) {
    this->batch_size = batch_size;
    return *this;
  }
  UnaryOpTestParams& Channels(LoopLimits<size_t> channels) {
    this->channels = channels;
    return *this;
  }
  UnaryOpTestParams& Iterations(size_t iterations) {
    this->iterations = iterations;
    return *this;
  }
  UnaryOpTestParams& InputStride(size_t input_stride) {
    this->input_stride = input_stride;
    return *this;
  }
  UnaryOpTestParams& OutputStride(size_t output_stride) {
    this->output_stride = output_stride;
    return *this;
  }
  UnaryOpTestParams& Qmin(uint8_t qmin) {
    this->qmin = qmin;
    return *this;
  }
  UnaryOpTestParams& Qmax(uint8_t qmax) {
    this->qmax = qmax;
    return *this;
  }
  UnaryOpTestParams& InputScale(LoopLimits<float> input_scale) {
    this->input_scale = input_scale;
    return *this;
  }
  UnaryOpTestParams& InputZeroPoint(LoopLimits<int32_t> input_zero_point) {
    this->input_zero_point = input_zero_point;
    return *this;
  }

  std::string ToString() const {
    std::string result = test_name;
    if (input_stride && !output_stride) {
      result += "_with_input_stride";
    } else if (!input_stride && output_stride) {
      result += "_with_output_stride";
    }
    if (qmin) {
      result += "_with_qmin";
    }
    if (qmax) {
      result += "_with_qmax";
    }
    if (input_zero_point) {
      result += "_with_input_zero_point";
    }
    if (input_scale) {
      result += "_with_input_scale";
    }
    return result;
  }

  std::string test_name;
  size_t batch_size;
  size_t iterations = 3;
  LoopLimits<size_t> channels;
  Optional<size_t> input_stride;
  Optional<size_t> output_stride;
  Optional<uint8_t> qmin;
  Optional<uint8_t> qmax;
  Optional<LoopLimits<float>> input_scale;
  Optional<LoopLimits<int32_t>> input_zero_point;
};

inline std::ostream& operator<<(std::ostream& os, UnaryOpTestParams params) {
  os << "{test_name: '" << params.test_name
     << "', batch_size: " << params.batch_size;
  if (params.input_stride) {
    os << ", input_stride: " << *params.input_stride;
  }
  if (params.output_stride) {
    os << ", output_stride: " << *params.output_stride;
  }
  if (params.qmin) {
    os << ", qmin: " << *params.qmin;
  }
  if (params.qmax) {
    os << ", qmax: " << *params.qmax;
  }
  if (params.input_scale) {
    os << ", input_scale: " << params.input_scale->ToString();
  }
  if (params.input_zero_point) {
    os << ", input_zero_point: " << params.input_zero_point->ToString();
  }
  return os << "}";
}

#define CREATE_UNARY_TEST(datatype, Tester)                           \
  using Tester##datatype = testing::TestWithParam<UnaryOpTestParams>; \
  TEST_P(Tester##datatype, Test##datatype) {                          \
    const UnaryOpTestParams& test_case = GetParam();                  \
    for (size_t channels = test_case.channels.min;                    \
         channels < test_case.channels.max;                           \
         channels += test_case.channels.stride) {                     \
      LoopLimits<float> input_scale_limits{1, 2, 2};                  \
      if (test_case.input_scale) {                                    \
        input_scale_limits = *test_case.input_scale;                  \
      }                                                               \
      for (float input_scale = input_scale_limits.min;                \
           input_scale < input_scale_limits.max;                      \
           input_scale *= input_scale_limits.stride) {                \
        LoopLimits<int32_t> input_zero_point_limits{0, 1, 1};         \
        if (test_case.input_zero_point) {                             \
          input_zero_point_limits = *test_case.input_zero_point;      \
        }                                                             \
        for (int32_t input_zero_point = input_zero_point_limits.min;  \
             input_zero_point <= input_zero_point_limits.max;         \
             input_zero_point += input_zero_point_limits.stride) {    \
          Tester tester;                                              \
          tester.batch_size(test_case.batch_size)                     \
              .channels(channels)                                     \
              .iterations(test_case.iterations);                      \
          if (test_case.input_stride) {                               \
            tester.input_stride(*test_case.input_stride);             \
          }                                                           \
          if (test_case.output_stride) {                              \
            tester.output_stride(*test_case.output_stride);           \
          }                                                           \
          if (test_case.qmin) {                                       \
            tester.qmin(*test_case.qmin);                             \
          }                                                           \
          if (test_case.qmax) {                                       \
            tester.qmax(*test_case.qmax);                             \
          }                                                           \
          if (test_case.input_scale) {                                \
            tester.input_scale(input_scale);                          \
          }                                                           \
          if (test_case.input_zero_point) {                           \
            tester.input_zero_point(input_zero_point);                \
          }                                                           \
          tester.Test##datatype();                                    \
        }                                                             \
      }                                                               \
    }                                                                 \
  }

#define CREATE_UNARY_FLOAT_TESTS(datatype, Tester)                          \
  CREATE_UNARY_TEST(datatype, Tester)                                       \
  INSTANTIATE_TEST_SUITE_P(                                                 \
      datatype, Tester##datatype,                                           \
      testing::ValuesIn<UnaryOpTestParams>({                                \
          UnaryOpTestParams::UnitBatch(),                                   \
          UnaryOpTestParams::SmallBatch(),                                  \
          UnaryOpTestParams::SmallBatch().InputStride(129),                 \
          UnaryOpTestParams::SmallBatch().OutputStride(117),                \
          UnaryOpTestParams::StridedBatch(),                                \
      }),                                                                   \
      [](const testing::TestParamInfo<Tester##datatype::ParamType>& info) { \
        return info.param.ToString();                                       \
      });

#define CREATE_UNARY_QUANTIZED_TESTS(datatype, Tester)                         \
  CREATE_UNARY_TEST(datatype, Tester)                                          \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      datatype, Tester##datatype,                                              \
      testing::ValuesIn<UnaryOpTestParams>({                                   \
          UnaryOpTestParams::UnitBatch(),                                      \
          UnaryOpTestParams::UnitBatch().Qmin(128),                            \
          UnaryOpTestParams::UnitBatch().Qmax(128),                            \
          UnaryOpTestParams::UnitBatch().InputScale({1.0e-2f, 1.0e2f, 10.0f}), \
          UnaryOpTestParams::UnitBatch().InputZeroPoint({0, 255, 51}),         \
          UnaryOpTestParams::SmallBatch(),                                     \
          UnaryOpTestParams::SmallBatch().InputStride(129),                    \
          UnaryOpTestParams::SmallBatch().OutputStride(117),                   \
          UnaryOpTestParams::SmallBatch().Qmin(128),                           \
          UnaryOpTestParams::SmallBatch().Qmax(128),                           \
          UnaryOpTestParams::SmallBatch().InputScale(                          \
              {1.0e-2f, 1.0e2f, 10.0f}),                                       \
          UnaryOpTestParams::SmallBatch().InputZeroPoint({0, 255, 51}),        \
          UnaryOpTestParams::StridedBatch(),                                   \
          UnaryOpTestParams::StridedBatch().Qmin(128),                         \
          UnaryOpTestParams::StridedBatch().Qmax(128),                         \
          UnaryOpTestParams::StridedBatch().InputScale(                        \
              {1.0e-2f, 1.0e2f, 10.0f}),                                       \
          UnaryOpTestParams::StridedBatch().InputZeroPoint({0, 255, 51}),      \
      }),                                                                      \
      [](const testing::TestParamInfo<Tester##datatype::ParamType>& info) {    \
        return info.param.ToString();                                          \
      });

#define CREATE_UNARY_QUANTIZED_TESTS_NO_QMIN(datatype, Tester)                 \
  CREATE_UNARY_TEST(datatype, Tester)                                          \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      datatype, Tester##datatype,                                              \
      testing::ValuesIn<UnaryOpTestParams>({                                   \
          UnaryOpTestParams::UnitBatch(),                                      \
          UnaryOpTestParams::UnitBatch().InputScale({1.0e-2f, 1.0e2f, 10.0f}), \
          UnaryOpTestParams::UnitBatch().InputZeroPoint({0, 255, 51}),         \
          UnaryOpTestParams::SmallBatch(),                                     \
          UnaryOpTestParams::SmallBatch().InputStride(129),                    \
          UnaryOpTestParams::SmallBatch().OutputStride(117),                   \
          UnaryOpTestParams::SmallBatch().InputScale(                          \
              {1.0e-2f, 1.0e2f, 10.0f}),                                       \
          UnaryOpTestParams::SmallBatch().InputZeroPoint({0, 255, 51}),        \
          UnaryOpTestParams::StridedBatch(),                                   \
          UnaryOpTestParams::StridedBatch().InputScale(                        \
              {1.0e-2f, 1.0e2f, 10.0f}),                                       \
          UnaryOpTestParams::StridedBatch().InputZeroPoint({0, 255, 51}),      \
      }),                                                                      \
      [](const testing::TestParamInfo<Tester##datatype::ParamType>& info) {    \
        return info.param.ToString();                                          \
      });

};  // namespace xnnpack

#endif  // __XNNPACK_TEST_UNARY_OPERATOR_TESTER_H_
