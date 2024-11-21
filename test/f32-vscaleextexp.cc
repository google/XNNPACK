// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "replicable_random_device.h"
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/vscaleextexp.h"

class VScaleExtExpMicrokernelTester {
 public:
  VScaleExtExpMicrokernelTester& elements(size_t elements) {
    assert(elements != 0);
    this->elements_ = elements;
    return *this;
  }

  size_t elements() const {
    return this->elements_;
  }

  VScaleExtExpMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_vscaleextexp_ukernel_fn vscaleextexp) const {
    xnnpack::ReplicableRandomDevice rng;
    // Choose such range that expf(x[i]) overflows, but double-precision exp doesn't overflow.
    auto f32rng = [&rng]() {
      return std::uniform_real_distribution<float>(90.0f, 100.0f)(rng);
    };

    xnnpack::Buffer<float> x(elements() + XNN_EXTRA_BYTES / sizeof(float));
    xnnpack::Buffer<float> y(elements());
    xnnpack::Buffer<double> y_ref(elements());
    // TODO(b/372792254): This is hiding a possible msan bug in the microkernels tested here.
    std::fill(y.begin(), y.end(), 0.0f);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));

      // Compute scale parameters.
      double sum = 0.0;
      for (size_t i = 0; i < elements(); i++) {
        sum += std::exp(double(x[i]));
      }
      int sum_exponent;
      const double sum_mantissa = std::frexp(sum, &sum_exponent);
      const float scale_mantissa = float(1.0 / sum_mantissa);
      const float scale_exponent = -float(sum_exponent);

      // Compute reference results.
      for (size_t i = 0; i < elements(); i++) {
        y_ref[i] = std::exp(double(x[i])) / sum;
      }

      // Call optimized micro-kernel.
      vscaleextexp(elements() * sizeof(float), x.data(), y.data(), scale_mantissa, scale_exponent);

      // Verify results.
      for (size_t i = 0; i < elements(); i++) {
        EXPECT_NEAR(y_ref[i], y[i], std::abs(y_ref[i]) * 1.0e-6)
          << "elements = " << elements() << ", scale:mantissa = " << scale_mantissa << ", scale:exponent = " << scale_exponent;
      }
    }
  }

 private:
  size_t elements_{1};
  size_t iterations_{15};
};

#define XNN_TEST_VSCALEEXTEXP_ELEMENT_EQ(ukernel, arch_flags, element_tile, ...)                                       \
  TEST(ukernel, element_eq)                                                                                            \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    VScaleExtExpMicrokernelTester().elements(element_tile).Test(ukernel);                                              \
  }
#define XNN_TEST_VSCALEEXTEXP_ELEMENT_GT(ukernel, arch_flags, element_tile, ...)                                       \
  TEST(ukernel, element_gt)                                                                                            \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    for (size_t element_size = element_tile + 1; element_size < ((element_tile == 1) ? 10 : element_tile * 2);         \
         element_size++) {                                                                                             \
      VScaleExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                            \
    }                                                                                                                  \
  }
#define XNN_TEST_VSCALEEXTEXP_ELEMENT_LT(ukernel, arch_flags, element_tile, ...)                                       \
  TEST(ukernel, element_lt)                                                                                            \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    for (size_t element_size = 1; element_size < element_tile; element_size++) {                                       \
      VScaleExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                            \
    }                                                                                                                  \
  }
#define XNN_TEST_VSCALEEXTEXP_ELEMENT_DIV(ukernel, arch_flags, element_tile, ...)                                      \
  TEST(ukernel, element_div)                                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    for (size_t element_size = 2 * element_tile; element_size < 10 * element_tile; element_size += element_tile) {     \
      VScaleExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                            \
    }                                                                                                                  \
  }

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params)                 \
  XNN_TEST_VSCALEEXTEXP_ELEMENT_EQ(ukernel, arch_flags, element_tile, init_params);                                    \
  XNN_TEST_VSCALEEXTEXP_ELEMENT_DIV(ukernel, arch_flags, element_tile, init_params);                                   \
  XNN_TEST_VSCALEEXTEXP_ELEMENT_LT(ukernel, arch_flags, element_tile, init_params);                                    \
  XNN_TEST_VSCALEEXTEXP_ELEMENT_GT(ukernel, arch_flags, element_tile, init_params);
#include "f32-vscaleextexp/f32-vscaleextexp.h"
#undef XNN_UKERNEL_WITH_PARAMS
