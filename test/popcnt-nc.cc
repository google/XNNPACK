// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdlib>
#include <limits>

#include "unary-operator-tester.h"
#include "xnnpack.h"
#include "replicable_random_device.h"

namespace xnnpack {
class PopCountOperatorTester : public UnaryOperatorTester{
 public:
  PopCountOperatorTester() : UnaryOperatorTester() {

  }
  void TestS32() override {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> s32dist(range_s32_.first,
                                                   range_s32_.second);

    std::vector<int32_t> input(XNN_EXTRA_BYTES / sizeof(int32_t) +
                            (batch_size() - 1) * input_stride() + channels());
    std::vector<int32_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<int32_t> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return s32dist(rng); });
      std::fill(output.begin(), output.end(), 0);
      
      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] = popcount_ref(input[i * input_stride() + c]);
        }
      }

      // Create, setup, run, and destroy Square operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t op = nullptr;

      ASSERT_EQ(xnn_status_success, CreateOpS32(0, &op));
      ASSERT_NE(nullptr, op);

      // Smart pointer to automatically delete op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
                ReshapeOpS32(op, batch_size(), channels(), input_stride(),
                             output_stride(), /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success, SetupOpS32(op, input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const int32_t y = output[i * output_stride() + c];
          const int32_t y_ref = output_ref[i * channels() + c];
          CheckResultS32(y, y_ref, i, c, input[i * input_stride() + c]);
        }
      }
    }
  }

  void TestRunS32() override {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> s32dist(range_s32_.first,
                                                   range_s32_.second);

    std::vector<int32_t> input(XNN_EXTRA_BYTES / sizeof(int32_t) +
                               (batch_size() - 1) * input_stride() + channels());
    std::vector<int32_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<int32_t> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return s32dist(rng); });
      std::fill(output.begin(), output.end(), 0);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] = popcount_ref(input[i * input_stride() + c]);
        }
      }

      // Initialize and run Square Root operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

      ASSERT_EQ(
        xnn_status_success,
        RunOpS32(channels(), input_stride(), output_stride(), batch_size(),
                 input.data(), output.data(), 0, /*threadpool=*/nullptr));
        // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float y = output[i * output_stride() + c];
          const float y_ref = output_ref[i * channels() + c];
          CheckResultS32(y, y_ref, i, c, input[i * input_stride() + c]);
        }
      }
    }
  }

 protected:
  float RefFunc(float x) const override { return 0.0f; }
  int32_t popcount_ref(uint32_t a) const {
    int count = 0;
    while (a) {
        count += a & 1;
        a >>= 1;
    }
    return count;
  }

  xnn_status CreateOpS32(uint32_t flags, xnn_operator_t* op_out) const override {
    return xnn_create_pop_count_nc_s32(flags, op_out);       
  }

  xnn_status ReshapeOpS32(xnn_operator_t op, size_t batch_size,
                          size_t channels, size_t input_stride,
                          size_t output_stride, pthreadpool_t threadpool) const override {
    return xnn_reshape_pop_count_nc_s32(op, batch_size, channels, input_stride, output_stride, threadpool); 
  }

  xnn_status SetupOpS32(xnn_operator_t op, const int32_t* input, int32_t* output) const override {
    return xnn_setup_pop_count_nc_s32(op, input, output);                   
  }

  xnn_status RunOpS32(size_t channels, size_t input_stride,
                      size_t output_stride, size_t batch_size,
                      const int32_t* input, int32_t* output, uint32_t flags,
                      pthreadpool_t threadpool) const override {
    return xnn_run_pop_count_nc_s32(channels, input_stride, output_stride,batch_size, input, output, flags,threadpool);
  }
};
#define CREATE_UNARY_POPCNT_TESTS(datatype, Tester)                       \
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
CREATE_UNARY_POPCNT_TESTS(S32,PopCountOperatorTester)
CREATE_UNARY_POPCNT_TESTS(RunS32,PopCountOperatorTester)
};
