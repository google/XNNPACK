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
#include "xnnpack/raddextexp.h"

class RAddExtExpMicrokernelTester {
 public:
  RAddExtExpMicrokernelTester& elements(size_t elements) {
    assert(elements != 0);
    this->elements_ = elements;
    return *this;
  }

  size_t elements() const {
    return this->elements_;
  }

  RAddExtExpMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_raddextexp_ukernel_fn raddextexp) const {
    xnnpack::ReplicableRandomDevice rng;
    // Choose such range that expf(x[i]) overflows, but double-precision exp doesn't overflow.
    auto f32rng = [&rng]() {
      return std::uniform_real_distribution<float>(90.0f, 100.0f)(rng);
    };

    xnnpack::Buffer<float> x(elements() + XNN_EXTRA_BYTES / sizeof(float));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));

      // Compute reference results.
      double sum_ref = 0.0f;
      for (size_t i = 0; i < elements(); i++) {
        sum_ref += exp(double(x[i]));
      }

      // Call optimized micro-kernel.
      float sum[2];
      raddextexp(elements() * sizeof(float), x.data(), sum);

      // Verify results.
      ASSERT_NEAR(sum_ref, exp2(double(sum[1])) * double(sum[0]), std::abs(sum_ref) * 1.0e-6)
        << "elements = " << elements() << ", y:value = " << sum[0] << ", y:exponent = " << sum[1];
    }
  }

 private:
  size_t elements_{1};
  size_t iterations_{15};
};

#define XNN_TEST_RADDEXTEXP_ELEMENT_EQ(ukernel, arch_flags, element_tile, ...)                                         \
  TEST(ukernel, element_eq)                                                                                            \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    RAddExtExpMicrokernelTester().elements(element_tile).Test(ukernel);                                                \
  }
#define XNN_TEST_RADDEXTEXP_ELEMENT_DIV(ukernel, arch_flags, element_tile, ...)                                        \
  TEST(ukernel, element_gt)                                                                                            \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    for (size_t element_size = element_tile * 2; element_size < element_tile * 10; element_size += element_tile) {     \
      RAddExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                              \
    }                                                                                                                  \
  }
#define XNN_TEST_RADDEXTEXP_ELEMENT_LT(ukernel, arch_flags, element_tile, ...)                                         \
  TEST(ukernel, element_lt)                                                                                            \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    for (size_t element_size = 1; element_size < element_tile; element_size++) {                                       \
      RAddExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                              \
    }                                                                                                                  \
  }
#define XNN_TEST_RADDEXTEXP_ELEMENT_GT(ukernel, arch_flags, element_tile, ...)                                         \
  TEST(ukernel, element_div)                                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    for (size_t element_size = element_tile + 1; element_size < (element_tile == 1 ? 10 : element_tile * 2);           \
         element_size++) {                                                                                             \
      RAddExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                              \
    }                                                                                                                  \
  }

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params)                 \
  XNN_TEST_RADDEXTEXP_ELEMENT_EQ(ukernel, arch_flags, element_tile, init_params);                                      \
  XNN_TEST_RADDEXTEXP_ELEMENT_DIV(ukernel, arch_flags, element_tile, init_params);                                     \
  XNN_TEST_RADDEXTEXP_ELEMENT_LT(ukernel, arch_flags, element_tile, init_params);                                      \
  XNN_TEST_RADDEXTEXP_ELEMENT_GT(ukernel, arch_flags, element_tile, init_params);
#include "f32-raddextexp/f32-raddextexp.h"
#undef XNN_UKERNEL_WITH_PARAMS
