// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <ios>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "test/replicable_random_device.h"
#include "test/unary-ops.h"

class VUnaryMicrokernelTester {
 public:
  VUnaryMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const { return this->batch_size_; }

  VUnaryMicrokernelTester& input_quantization(
      const xnn_quantization_params& quantization) {
    this->input_quantization_ = quantization;
    return *this;
  }

  const xnn_quantization_params& input_quantization() const {
    return this->input_quantization_;
  }

  VUnaryMicrokernelTester& output_quantization(
      const xnn_quantization_params& quantization) {
    this->output_quantization_ = quantization;
    return *this;
  }

  const xnn_quantization_params& output_quantization() const {
    return this->output_quantization_;
  }

  VUnaryMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  // Generic test function for `vunary` kernels.
  //
  // The function is templated on the type of the kernel parameters and takes
  // the following arguments:
  //
  //  * `T`: The datatype to test. Should be implicitly convertible to and from
  //    `float`.
  //  * `init_params`: A function that populates a given parameters data
  //    structure or returns `nullptr` if there is no default initialization.
  template <typename TestInfo, typename In, typename Out,
            typename UKernelParamsType>
  void Test(void (*ukernel)(size_t,
                            const typename xnnpack::unwrap_quantized<In>::type*,
                            typename xnnpack::unwrap_quantized<Out>::type*,
                            const UKernelParamsType*),
            xnn_init_unary_uparams_fn init_params,
            const xnn_unary_params& params) const {
    using InKernel = typename xnnpack::unwrap_quantized<In>::type;
    using OutKernel = typename xnnpack::unwrap_quantized<Out>::type;

    TestInfo test_info;
    auto domain = test_info.Domain(xnn_datatype_of<In>());
    xnnpack::ReplicableRandomDevice rng;

    xnnpack::Buffer<In> x(batch_size(), xnnpack::XnnExtraBytes);
    xnnpack::Buffer<Out> y(batch_size());
    xnnpack::Buffer<Out> y_ref(batch_size());
    xnnpack::DatatypeGenerator<In> input_generator(domain.min, domain.max,
                                                   input_quantization_);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(x.data(), x.size(),
                      [&]() { return input_generator(rng); });

      // Compute reference results.
      UnaryReferenceImpl(x.data(), batch_size(), y_ref.data(), test_info,
                         input_quantization_, output_quantization_, params);

      // Initialize the params.
      xnn_unary_uparams uparams;
      if (init_params) {
        init_params(&uparams, &params, &input_quantization_,
                    &output_quantization_);
      }

      // Call optimized micro-kernel.
      ukernel(batch_size() * sizeof(In), (const InKernel*)x.data(),
              (OutKernel*)y.data(), (UKernelParamsType*)&uparams);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        if (test_info.IsInSupportedRange(y_ref[i])) {
          if (std::isnan(static_cast<float>(y_ref[i]))) {
            ASSERT_TRUE(std::isnan(static_cast<float>(y[i])));
          } else {
            ASSERT_NEAR(y[i], y_ref[i],
                        test_info.Tolerance(y_ref[i], xnn_datatype_of<Out>()))
                << "at " << i << " / " << batch_size() << ", x[" << i
                << "] = " << std::scientific << (float)x[i];
          }
        }
      }
    }
  }

  template <typename TestInfo, typename In, typename Out,
            typename UKernelParamsType>
  void TestInPlace(
      void (*ukernel)(size_t,
                      const typename xnnpack::unwrap_quantized<In>::type*,
                      typename xnnpack::unwrap_quantized<Out>::type*,
                      const UKernelParamsType*),
      xnn_init_unary_uparams_fn init_params,
      const xnn_unary_params& params) const {
    using InKernel = typename xnnpack::unwrap_quantized<In>::type;
    using OutKernel = typename xnnpack::unwrap_quantized<Out>::type;
    static_assert(sizeof(InKernel) == sizeof(OutKernel), "");

    TestInfo test_info;
    auto domain = test_info.Domain(xnn_datatype_of<In>());
    xnnpack::ReplicableRandomDevice rng;

    xnnpack::Buffer<In> x(batch_size(), xnnpack::XnnExtraBytes);
    Out* y = reinterpret_cast<Out*>(x.data());
    xnnpack::Buffer<Out> y_ref(batch_size());
    xnnpack::DatatypeGenerator<In> input_generator(domain.min, domain.max,
                                                   input_quantization_);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(x.data(), x.size(),
                      [&]() { return input_generator(rng); });

      // Make a copy of the original input data for debugging output.
      xnnpack::Buffer<In> x_orig(x.size());
      std::copy(x.begin(), x.end(), x_orig.begin());

      // Compute reference results.
      UnaryReferenceImpl(x.data(), batch_size(), y_ref.data(), test_info,
                         input_quantization_, output_quantization_, params);

      // Initialize the params.
      xnn_unary_uparams uparams;
      if (init_params) {
        init_params(&uparams, &params, &input_quantization_,
                    &output_quantization_);
      }

      // Call optimized micro-kernel.
      ukernel(batch_size() * sizeof(In), (const InKernel*)x.data(),
              (OutKernel*)y, (UKernelParamsType*)&uparams);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        if (test_info.IsInSupportedRange(y_ref[i])) {
          if (std::isnan(static_cast<float>(y_ref[i]))) {
            ASSERT_TRUE(std::isnan(static_cast<float>(x[i])));
          } else {
            ASSERT_NEAR(x[i], y_ref[i],
                        test_info.Tolerance(y_ref[i], xnn_datatype_of<Out>()))
                << "at " << i << " / " << batch_size() << ", x[" << i
                << "] = " << std::scientific << (float)x_orig[i];
          }
        }
      }
    }
  }

  template <typename TestInfo, typename In, typename Out,
            typename UKernelParamsType>
  void Test(void (*ukernel)(size_t,
                            const typename xnnpack::unwrap_quantized<In>::type*,
                            typename xnnpack::unwrap_quantized<Out>::type*,
                            const UKernelParamsType*),
            xnn_init_unary_uparams_fn init_params) const {
    Test<TestInfo, In, Out>(ukernel, init_params, TestInfo().DefaultParams());
  }

  template <typename TestInfo, typename In, typename Out,
            typename UKernelParamsType>
  void TestInPlace(
      void (*ukernel)(size_t,
                      const typename xnnpack::unwrap_quantized<In>::type*,
                      typename xnnpack::unwrap_quantized<Out>::type*,
                      const UKernelParamsType*),
      xnn_init_unary_uparams_fn init_params) const {
    TestInPlace<TestInfo, In, Out>(ukernel, init_params,
                                   TestInfo().DefaultParams());
  }

  template <typename TestInfo, typename In, typename Out,
            typename UKernelParamsType>
  void Test(void (*ukernel)(size_t, const In*, Out*, const UKernelParamsType*),
            xnn_init_unary_uparams_fn init_params,
            const xnn_unary_params& params, std::vector<In> inputs,
            const std::vector<Out>& expected, int tolerance_ulp) const {
    std::vector<Out> outputs(inputs.size());
    inputs.resize(inputs.size() + XNN_EXTRA_BYTES / sizeof(In));
    xnn_unary_uparams uparams;
    if (init_params) {
      init_params(&uparams, &params, nullptr, nullptr);
    }
    ukernel(outputs.size() * sizeof(In), inputs.data(), outputs.data(),
            (UKernelParamsType*)&uparams);
    for (size_t i = 0; i < outputs.size(); i++) {
      if (std::isfinite(expected[i])) {
        ASSERT_NEAR(expected[i], outputs[i],
                    tolerance_ulp * std::abs(expected[i]) *
                        std::numeric_limits<float>::epsilon())
            << "for input " << inputs[i];
      } else {
        EXPECT_EQ(std::fpclassify(expected[i]), std::fpclassify(outputs[i]))
            << "for input " << inputs[i] << " and output " << outputs[i]
            << " (FP_INFINITE=" << FP_INFINITE << ", FP_NAN=" << FP_NAN
            << ", FP_NORMAL=" << FP_NORMAL << ", FP_SUBNORMAL=" << FP_SUBNORMAL
            << ", FP_ZERO=" << FP_ZERO << ")";
      }
    }
  }

  template <typename TestInfo, typename In, typename Out,
            typename UKernelParamsType>
  void Test(void (*ukernel)(size_t, const In*, Out*, const UKernelParamsType*),
            xnn_init_unary_uparams_fn init_params, std::vector<In> inputs,
            const std::vector<Out>& expected, int tolerance_ulp) const {
    Test<TestInfo, In, Out>(ukernel, init_params, TestInfo().DefaultParams(),
                            inputs, expected, tolerance_ulp);
  }

 private:
  size_t batch_size_ = 1;
  xnn_quantization_params input_quantization_ = {0, 1.0f};
  xnn_quantization_params output_quantization_ = {0, 1.0f};
  size_t iterations_ = 15;
};

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestBatchEq(uint64_t arch_flags, size_t batch_tile, UKernelFn ukernel,
                 xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const size_t batch_scale = get_batch_scale<In>();
  VUnaryMicrokernelTester()
      .batch_size(batch_tile * batch_scale)
      .Test<TestInfo, In, Out>(ukernel, init_params, args...);
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestBatchDiv(uint64_t arch_flags, size_t batch_tile, UKernelFn ukernel,
                  xnn_init_unary_uparams_fn init_params, Args... args) {
  if (batch_tile == 1) return;
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const size_t batch_scale = get_batch_scale<In>();
  const size_t batch_step = batch_tile * batch_scale;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step;
       batch_size += batch_step) {
    VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test<TestInfo, In, Out>(ukernel, init_params, args...);
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestBatchLT(uint64_t arch_flags, size_t batch_tile, UKernelFn ukernel,
                 xnn_init_unary_uparams_fn init_params, Args... args) {
  if (batch_tile == 1) return;
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const size_t batch_scale = get_batch_scale<In>();
  const size_t batch_end = batch_tile * batch_scale;
  for (size_t batch_size = 1; batch_size < batch_end; batch_size++) {
    VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test<TestInfo, In, Out>(ukernel, init_params, args...);
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestBatchGT(uint64_t arch_flags, size_t batch_tile, UKernelFn ukernel,
                 xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const size_t batch_scale = get_batch_scale<In>();
  const size_t batch_step = batch_tile * batch_scale;
  const size_t batch_end = batch_tile == 1 ? 10 : 2 * batch_step;
  for (size_t batch_size = batch_step + 1; batch_size < batch_end;
       batch_size++) {
    VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test<TestInfo, In, Out>(ukernel, init_params, args...);
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestInPlace(uint64_t arch_flags, size_t batch_tile, UKernelFn ukernel,
                 xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const size_t batch_scale = get_batch_scale<In>();
  const size_t batch_end = batch_tile * batch_scale;
  const size_t batch_step = std::max<size_t>(1, batch_tile - 1);
  for (size_t batch_size = 1; batch_size <= batch_end;
       batch_size += batch_step) {
    VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestInPlace<TestInfo, In, Out>(ukernel, init_params, args...);
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestInputScale(uint64_t arch_flags, size_t batch_tile, UKernelFn ukernel,
                    xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (float input_scale : {4.0f, 16.0f, 64.0f}) {
      xnn_quantization_params input_quantization =
          TestInfo().InputQuantizationParams(xnn_datatype_of<In>());
      xnn_quantization_params output_quantization =
          TestInfo().InputQuantizationParams(xnn_datatype_of<Out>());
      input_quantization.scale = input_scale;
      VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .input_quantization(input_quantization)
          .output_quantization(output_quantization)
          .Test<TestInfo, In, Out>(ukernel, init_params, args...);
    }
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestOutputScale(uint64_t arch_flags, size_t batch_tile, UKernelFn ukernel,
                     xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (float output_scale : {4.0f, 16.0f, 64.0f}) {
      xnn_quantization_params input_quantization =
          TestInfo().InputQuantizationParams(xnn_datatype_of<In>());
      xnn_quantization_params output_quantization =
          TestInfo().InputQuantizationParams(xnn_datatype_of<Out>());
      output_quantization.scale = output_scale;
      VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .input_quantization(input_quantization)
          .output_quantization(output_quantization)
          .Test<TestInfo, In, Out>(ukernel, init_params, args...);
    }
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestInputZeroPoint(uint64_t arch_flags, size_t batch_tile,
                        UKernelFn ukernel,
                        xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  for (int16_t input_zero_point = 2; input_zero_point < 10;
       input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      xnn_quantization_params input_quantization =
          TestInfo().InputQuantizationParams(xnn_datatype_of<In>());
      xnn_quantization_params output_quantization =
          TestInfo().InputQuantizationParams(xnn_datatype_of<Out>());
      input_quantization.zero_point = input_zero_point;
      VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .input_quantization(input_quantization)
          .output_quantization(output_quantization)
          .Test<TestInfo, In, Out>(ukernel, init_params, args...);
    }
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestOutputZeroPoint(uint64_t arch_flags, size_t batch_tile,
                         UKernelFn ukernel,
                         xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  for (int16_t output_zero_point = 2; output_zero_point < 10;
       output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      xnn_quantization_params input_quantization =
          TestInfo().InputQuantizationParams(xnn_datatype_of<In>());
      xnn_quantization_params output_quantization =
          TestInfo().InputQuantizationParams(xnn_datatype_of<Out>());
      output_quantization.zero_point = output_zero_point;
      VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .input_quantization(input_quantization)
          .output_quantization(output_quantization)
          .Test<TestInfo, In, Out>(ukernel, init_params, args...);
    }
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestOutputSaturation(uint64_t arch_flags, size_t batch_tile,
                          UKernelFn ukernel,
                          xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const size_t batch_scale = get_batch_scale<In>();
  const size_t batch_end = batch_tile * batch_scale * 5;
  const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;
  for (size_t batch_size = 1; batch_size <= batch_end;
       batch_size += batch_step) {
    VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .output_quantization({0, 500.0f})
        .Test<TestInfo, In, Out>(ukernel, init_params, args...);
  }
}

template <typename TestInfo, typename In, typename Out, typename UKernelFn,
          typename... Args>
void TestOutputOverflow(uint64_t arch_flags, size_t batch_tile,
                        UKernelFn ukernel,
                        xnn_init_unary_uparams_fn init_params, Args... args) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const size_t batch_scale = get_batch_scale<In>();
  const size_t batch_end = batch_tile * batch_scale * 5;
  const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;
  for (size_t batch_size = 1; batch_size <= batch_end;
       batch_size += batch_step) {
    VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .output_quantization({0, 4294967296.0f})
        .Test<TestInfo, In, Out>(ukernel, init_params, args...);
  }
}
