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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/datatype.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/buffer.h"
#include "replicable_random_device.h"

// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"
#include "unary-ops.h"

class VUnaryMicrokernelTester {
 public:
  VUnaryMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const { return this->batch_size_; }

  VUnaryMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const { return this->inplace_; }

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
  void Test(void (*ukernel)(size_t, const In*, Out*, const UKernelParamsType*),
            xnn_init_unary_uparams_fn init_params,
            const xnn_unary_params& params) const {
    TestInfo test_info;
    auto domain = test_info.Domain(xnn_datatype_of<In>());
    xnnpack::ReplicableRandomDevice rng;

    xnnpack::Buffer<In> x(batch_size() + XNN_EXTRA_BYTES / sizeof(In));
    xnnpack::Buffer<Out> y(batch_size() +
                           (inplace() ? XNN_EXTRA_BYTES / sizeof(Out) : 0));
    xnnpack::Buffer<Out> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      // This should only fill batch_size() elements, but some kernels trigger
      // msan errors if we don't initialize the XNN_EXTRA_BYTES.
      FillRandom(rng, x.data(), x.size(), domain, input_quantization_);
      if (inplace()) {
        std::copy(x.begin(), x.end(), y.begin());
      }
      const In* x_data = inplace() ? (const In*)y.data() : x.data();

      // Compute reference results.
      UnaryReferenceImpl(x_data, batch_size(), y_ref.data(), test_info,
                         input_quantization_, output_quantization_, params);

      // Initialize the params.
      xnn_unary_uparams uparams;
      if (init_params) {
        init_params(&uparams, &params, &input_quantization_,
                    &output_quantization_);
      }

      // Call optimized micro-kernel.
      ukernel(batch_size() * sizeof(In), x_data, y.data(),
              (UKernelParamsType*)&uparams);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i],
                    test_info.Tolerance(y_ref[i], xnn_datatype_of<Out>()))
            << "at " << i << " / " << batch_size() << ", x[" << i
            << "] = " << std::scientific << (float)x[i];
      }
    }
  }

  template <typename TestInfo, typename In, typename Out,
            typename UKernelParamsType>
  void Test(void (*ukernel)(size_t, const In*, Out*, const UKernelParamsType*),
            xnn_init_unary_uparams_fn init_params) const {
    Test<TestInfo>(ukernel, init_params, TestInfo().DefaultParams());
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
        EXPECT_NEAR(expected[i], outputs[i],
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
    Test<TestInfo>(ukernel, init_params, TestInfo().DefaultParams(), inputs,
                   expected, tolerance_ulp);
  }

 private:
  size_t batch_size_ = 1;
  bool inplace_ = false;
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
      .Test<TestInfo>(ukernel, init_params, args...);
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
        .Test<TestInfo>(ukernel, init_params, args...);
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
        .Test<TestInfo>(ukernel, init_params, args...);
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
        .Test<TestInfo>(ukernel, init_params, args...);
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
        .inplace(true)
        .Test<TestInfo>(ukernel, init_params, args...);
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
          .Test<TestInfo>(ukernel, init_params, args...);
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
          .Test<TestInfo>(ukernel, init_params, args...);
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
          .Test<TestInfo>(ukernel, init_params, args...);
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
          .Test<TestInfo>(ukernel, init_params, args...);
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
        .Test<TestInfo>(ukernel, init_params, args...);
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
        .Test<TestInfo>(ukernel, init_params, args...);
  }
}
