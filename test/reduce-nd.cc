// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/requantization.h"
#include "replicable_random_device.h"
#include "pthreadpool.h"

class ReduceOperatorTester {
 public:
  ReduceOperatorTester& input_shape(std::initializer_list<size_t> input_shape) {
    assert(input_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input_shape_ = std::vector<size_t>(input_shape);
    return *this;
  }

  ReduceOperatorTester& input_shape(const std::vector<size_t>& input_shape) {
    assert(input_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input_shape_ = std::vector<size_t>(input_shape);
    return *this;
  }

  const std::vector<size_t>& input_shape() const {
    return this->input_shape_;
  }

  size_t num_input_dims() const {
    return this->input_shape_.size();
  }

  size_t num_input_elements() const {
    return std::accumulate(
      this->input_shape_.begin(), this->input_shape_.end(), size_t(1), std::multiplies<size_t>());
  }

  ReduceOperatorTester& reduction_axes(std::initializer_list<size_t> reduction_axes) {
    assert(reduction_axes.size() <= XNN_MAX_TENSOR_DIMS);
    this->reduction_axes_ = std::vector<size_t>(reduction_axes);
    return *this;
  }

  ReduceOperatorTester& reduction_axes(const std::vector<size_t> reduction_axes) {
    assert(reduction_axes.size() <= XNN_MAX_TENSOR_DIMS);
    this->reduction_axes_ = reduction_axes;
    return *this;
  }

  const std::vector<size_t>& reduction_axes() const {
    return this->reduction_axes_;
  }

  size_t num_reduction_axes() const {
    return this->reduction_axes_.size();
  }

  ReduceOperatorTester& multithreaded(size_t multithreaded) {
    this->multithreaded_ = multithreaded;
    return *this;
  }

  size_t multithreaded() const {
    return this->multithreaded_;
  }

  size_t num_threads() const {
    // Do not spin up excessive number of threads for tests.
    return multithreaded() ? 5 : 1;
  }

  ReduceOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  ReduceOperatorTester& operation(enum xnn_reduce_operator operation) {
    this->reduce_operator_ = operation;
    return *this;
  }

   enum xnn_reduce_operator operation() const {
    return this->reduce_operator_;
  }

   struct QuantizationConfig {
     // Zero means no quantization.
     float input_scale = 0;
     float output_scale;
     int32_t input_zero_point;
     int32_t output_zero_point;
     int32_t quantized_output_min;
     int32_t quantized_output_max;

     bool IsQuantized() const { return input_scale != 0; }
     static QuantizationConfig Invalid() { return {0}; }
   };

  struct QS8Config {
    using StorageType = int8_t;
    using AccumulatorType = int32_t;

    static double GetTolerance() { return 0; }
    static xnn_datatype GetXNNDatatype() { return xnn_datatype_qint8; };

    static std::uniform_int_distribution<int32_t> BuildRngDistribution() {
      return std::uniform_int_distribution<int32_t>(
          std::numeric_limits<StorageType>::min(),
          std::numeric_limits<StorageType>::max());
    }

    static QuantizationConfig GenerateQuantization(xnnpack::ReplicableRandomDevice& rng,
                                                   std::uniform_int_distribution<int32_t>& dist) {
      QuantizationConfig q{
          /*input_scale=*/0.5,
          /*output_scale=*/0.75,
          /*input_zero_point=*/dist(rng),
          /*output_zero_point=*/dist(rng),
      };
      q.quantized_output_min = xnn_qs8_quantize(-INFINITY, q.output_scale, q.output_zero_point);
      q.quantized_output_max = xnn_qs8_quantize(INFINITY, q.output_scale, q.output_zero_point);
      return q;
    }
  };

  struct QU8Config {
    using StorageType = uint8_t;
    using AccumulatorType = uint32_t;

    static double GetTolerance() { return 0; }
    static xnn_datatype GetXNNDatatype() { return xnn_datatype_quint8; };

    static std::uniform_int_distribution<int32_t> BuildRngDistribution() {
      return std::uniform_int_distribution<int32_t>(
          std::numeric_limits<StorageType>::min(),
          std::numeric_limits<StorageType>::max());
    }

    static QuantizationConfig GenerateQuantization(xnnpack::ReplicableRandomDevice& rng,
                                                   std::uniform_int_distribution<int32_t>& dist) {
      QuantizationConfig q{
          /*input_scale=*/0.5,
          /*output_scale=*/0.75,
          /*input_zero_point=*/dist(rng),
          /*output_zero_point=*/dist(rng),
      };
      q.quantized_output_min = xnn_qu8_quantize(-INFINITY, q.output_scale, q.output_zero_point);
      q.quantized_output_max = xnn_qu8_quantize(INFINITY, q.output_scale, q.output_zero_point);
      return q;
    }
  };

  struct F16Config {
    using StorageType = xnn_float16;
    using AccumulatorType = float;

    static double GetTolerance() { return 3e-2; }
    static xnn_datatype GetXNNDatatype() { return xnn_datatype_fp16; };

    static std::uniform_real_distribution<float> BuildRngDistribution() {
      return std::uniform_real_distribution<float>(0.01, 1.0);
    }

    static QuantizationConfig GenerateQuantization(xnnpack::ReplicableRandomDevice& rng,
                                                   std::uniform_real_distribution<float>& dist) {
      return QuantizationConfig::Invalid();
    }
  };

  struct F32Config {
    using StorageType = float;
    using AccumulatorType = double;

    static double GetTolerance() { return 3e-6; }
    static xnn_datatype GetXNNDatatype() { return xnn_datatype_fp32; };

    static std::uniform_real_distribution<float> BuildRngDistribution() {
      return std::uniform_real_distribution<float>(0.01, 1.0);
    }

    static QuantizationConfig GenerateQuantization(xnnpack::ReplicableRandomDevice& rng,
                                                   std::uniform_real_distribution<float>& dist) {
      return QuantizationConfig::Invalid();
    }
  };

  template <class Config>
  void Test() const {
    using StorageType = typename Config::StorageType;

    xnnpack::ReplicableRandomDevice rng;
    auto dist = Config::BuildRngDistribution();

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    std::copy(input_shape().cbegin(), input_shape().cend(), input_dims.end() - num_input_dims());
    std::copy(input_dims.cbegin(), input_dims.cend(), output_dims.begin());
    for (size_t axis : reduction_axes()) {
      (output_dims.end() - num_input_dims())[axis] = 1;
    }
    const size_t num_output_elements =
      std::accumulate(output_dims.begin(), output_dims.end(), size_t(1), std::multiplies<size_t>());

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_dims[i - 1] == 1 ? 0 : output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<StorageType> input(XNN_EXTRA_BYTES / sizeof(StorageType) + num_input_elements());
    std::vector<StorageType> output(num_output_elements);
    std::vector<double> output_ref(num_output_elements);
    std::vector<typename Config::AccumulatorType> accumulator(num_output_elements);

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>
          auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));
      std::fill(output_ref.begin(), output_ref.end(), 0);
      std::fill(accumulator.begin(), accumulator.end(), 0);

      const int32_t num_reduced_elements = num_input_elements() / num_output_elements;
      const float reduce_scale =
          operation() == xnn_reduce_mean
              ? static_cast<double>(1.0f) / num_reduced_elements
              : 1;
      const QuantizationConfig q = Config::GenerateQuantization(rng, dist);


      // Compute reference results.
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  size_t input_idx =
                      i * input_strides[0] + j * input_strides[1] +
                      k * input_strides[2] + l * input_strides[3] +
                      m * input_strides[4] + n * input_strides[5];
                  size_t output_idx =
                      i * output_strides[0] + j * output_strides[1] +
                      k * output_strides[2] + l * output_strides[3] +
                      m * output_strides[4] + n * output_strides[5];
                  accumulator[output_idx] +=
                      static_cast<typename Config::AccumulatorType>(input[input_idx]);
                }
              }
            }
          }
        }
      }

      if (q.IsQuantized()) {
        for (size_t idx = 0; idx < output_ref.size(); ++idx) {
          // Shift by input zero point.
          output_ref[idx] =
              static_cast<float>(static_cast<int64_t>(accumulator[idx]) -
                                 q.input_zero_point * num_reduced_elements);
          // Apply scaling & clamp.
          output_ref[idx] *= q.input_scale * reduce_scale / q.output_scale;
          output_ref[idx] = std::min<double>(
              output_ref[idx], q.quantized_output_max - q.output_zero_point);
          output_ref[idx] = std::max<double>(
              output_ref[idx], q.quantized_output_min - q.output_zero_point);
          // Shift by output zero point.
          output_ref[idx] = static_cast<StorageType>(
              std::lrintf(output_ref[idx]) + q.output_zero_point);
        }
      } else {
        for (size_t i = 0; i < accumulator.size(); ++i) {
          output_ref[i] = accumulator[i] * reduce_scale;
        }
      }

      // Create, setup, run, and destroy a reduce operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t reduce_op = nullptr;
      const float scale = q.IsQuantized() ? q.input_scale / q.output_scale : 1;
      const xnn_status status =
          xnn_create_reduce_nd(operation(), Config::GetXNNDatatype(), scale,
                               q.input_zero_point, q.output_zero_point,
                               /*flags=*/0, &reduce_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, reduce_op);

      // Smart pointer to automatically delete reduce_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_reduce_op(reduce_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;

      size_t* workspace_size_ptr = nullptr;
      size_t* workspace_alignment_ptr = nullptr;
      if(Config::GetXNNDatatype() != xnn_datatype_fp32) {
        workspace_size_ptr = &workspace_size;
        workspace_alignment_ptr = &workspace_alignment;
      }

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_reduce_nd(
          reduce_op,
          Config::GetXNNDatatype(),
          num_reduction_axes(),
          reduction_axes().data(),
          num_input_dims(),
          input_shape().data(),
          workspace_size_ptr, workspace_alignment_ptr,
          auto_threadpool.get()));

      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace;
      void* workspace_ptr = nullptr;
      if(Config::GetXNNDatatype() != xnn_datatype_fp32) {
        ASSERT_NE(workspace_size, SIZE_MAX);
        ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        workspace.resize(workspace_size);
        workspace_ptr = workspace.data();
      }

      ASSERT_EQ(xnn_status_success,
        xnn_setup_reduce_nd(
          reduce_op,
          workspace_ptr,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
                xnn_run_operator(reduce_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  ASSERT_NEAR(output[index], output_ref[index], Config::GetTolerance() * std::abs(output_ref[index]))
                    << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")";
                }
              }
            }
          }
        }
      }
    }
  }


 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> reduction_axes_;
  bool multithreaded_{false};
  size_t iterations_{3};
  enum xnn_reduce_operator reduce_operator_;
};

constexpr size_t kDim1 = 2;
constexpr size_t kDim2 = 3;
constexpr size_t kDim3 = 5;
constexpr size_t kDim4 = 7;
constexpr size_t kDim5 = 11;
constexpr size_t kDim6 = 13;

struct TestParam {
  enum xnn_reduce_operator operation;
  enum xnn_datatype datatype;
  int dims;
  int reduction_axes;
  bool multithreaded;

  static std::string GetName(const testing::TestParamInfo<TestParam>& info) {
    std::stringstream sstr;
    const TestParam& param = info.param;
    switch(param.operation) {
      case xnn_reduce_mean:
        sstr << "mean";
        break;
      case xnn_reduce_sum:
        sstr << "sum";
        break;
      case xnn_reduce_invalid:
        sstr << "invalid";
        break;
    }
    sstr << "_" << xnn_datatype_to_string(param.datatype);
    sstr << "_" << param.dims << "d";
    if(param.reduction_axes == (1 << param.dims) - 1) {
      sstr << "_reduce_all";
    } else {
      sstr << "_axes";
      sstr << ((param.reduction_axes & (uint32_t(1) << 0)) != 0 ? "_1" : "");
      sstr << ((param.reduction_axes & (uint32_t(1) << 1)) != 0 ? "_2" : "");
      sstr << ((param.reduction_axes & (uint32_t(1) << 2)) != 0 ? "_3" : "");
      sstr << ((param.reduction_axes & (uint32_t(1) << 3)) != 0 ? "_4" : "");
      sstr << ((param.reduction_axes & (uint32_t(1) << 4)) != 0 ? "_5" : "");
      sstr << ((param.reduction_axes & (uint32_t(1) << 5)) != 0 ? "_6" : "");
    }
    if(param.multithreaded) {
      sstr << "_multithreaded";
    }
    return sstr.str();
  }
};

class ReduceNDTest : public testing::TestWithParam<TestParam> {
 public:
  std::vector<size_t> GetInputShape(const TestParam& params) {
    return std::vector<size_t>(reference_shape.begin(),
                               reference_shape.begin() + params.dims);
  }

  std::vector<size_t> GetReductionAxes(const TestParam& param) {
    const bool reduce_dims[6] = {
      (param.reduction_axes & (uint32_t(1) << 0)) != 0,
      (param.reduction_axes & (uint32_t(1) << 1)) != 0,
      (param.reduction_axes & (uint32_t(1) << 2)) != 0,
      (param.reduction_axes & (uint32_t(1) << 3)) != 0,
      (param.reduction_axes & (uint32_t(1) << 4)) != 0,
      (param.reduction_axes & (uint32_t(1) << 5)) != 0
    };

    std::vector<size_t> reduction_axes;
    for(int i = 0; i < param.dims; ++i) {
      if(reduce_dims[i]) {
        reduction_axes.push_back(i);
      }
    }
    return reduction_axes;
  }

 protected:
  static constexpr std::array<size_t, 6> reference_shape{kDim1, kDim2, kDim3,
                                                         kDim4, kDim5, kDim6};
};

// If you are confused by this, read https://stackoverflow.com/a/28846608
// TLDR: This is needed before C++17.
constexpr std::array<size_t, 6> ReduceNDTest::reference_shape;

TEST_P(ReduceNDTest, reduce) {
  TestParam param(GetParam());
    const std::vector<size_t> input_shape = GetInputShape(param);
    const std::vector<size_t> reduction_axes = GetReductionAxes(param);
    ASSERT_FALSE(input_shape.empty());
    ASSERT_FALSE(reduction_axes.empty());

    ReduceOperatorTester tester;
    tester.operation(param.operation)
          .input_shape(input_shape)
          .reduction_axes(reduction_axes);
    switch(param.datatype) {
      case xnn_datatype_fp16:
        tester.Test<ReduceOperatorTester::F16Config>();
        break;
      case xnn_datatype_fp32:
        tester.Test<ReduceOperatorTester::F32Config>();
        break;
      case xnn_datatype_qint8:
        tester.Test<ReduceOperatorTester::QS8Config>();
        break;
      case xnn_datatype_quint8:
        tester.Test<ReduceOperatorTester::QU8Config>();
        break;
      default:
        FAIL() << "Unsupported datatype";
    }
}

std::vector<TestParam> GenerateTests() {
  std::vector<TestParam> params;
  for(enum xnn_reduce_operator operation : {xnn_reduce_sum, xnn_reduce_mean}) {
    for(enum xnn_datatype datatype : {xnn_datatype_fp16, xnn_datatype_fp32,
                                      xnn_datatype_qint8, xnn_datatype_quint8}) {
      for(int dims = 1; dims <= 6; ++dims) {
        for(int reduction_axes = 1; reduction_axes < (1 << dims); ++reduction_axes) {
          for(bool multithreaded : {false, true}) {
            params.push_back(TestParam{
              operation, datatype, dims, reduction_axes, multithreaded
            });
            if(dims != 6 || reduction_axes != (1 << dims)-1) {
              break; // Only do the multithreaded test when we have 6 dims and
                     // reduce over all the axes.
            }
          }
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(
    ND, ReduceNDTest,
    testing::ValuesIn(GenerateTests()),
    TestParam::GetName);
