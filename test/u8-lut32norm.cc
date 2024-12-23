// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/lut.h"
#include "src/xnnpack/lut.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/buffer.h"
#include "test/replicable_random_device.h"

class LUTNormMicrokernelTester {
 public:
  LUTNormMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  size_t n() const {
    return this->n_;
  }

  LUTNormMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const {
    return this->inplace_;
  }

  LUTNormMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_u8_lut32norm_ukernel_fn lutnorm) const {
    xnnpack::ReplicableRandomDevice rng;
    auto u32rng = [&]() {
      return std::uniform_int_distribution<uint32_t>(
          1, std::numeric_limits<uint32_t>::max() / (257 * n()))(rng);
    };

    xnnpack::Buffer<uint8_t> x(n());
    xnnpack::Buffer<uint32_t> t(256);
    xnnpack::Buffer<uint8_t> y(n());
    xnnpack::Buffer<float> y_ref(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      xnnpack::fill_uniform_random_bits(x.data(), x.size(), rng);
      std::generate(t.begin(), t.end(), std::ref(u32rng));
      if (inplace()) {
        xnnpack::fill_uniform_random_bits(y.data(), y.size(), rng);
      }
      const uint8_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      uint32_t sum = 0;
      for (size_t i = 0; i < n(); i++) {
        sum += t[x_data[i]];
      }
      for (size_t i = 0; i < n(); i++) {
        y_ref[i] = 256.0f * float(t[x_data[i]]) / float(sum);
        y_ref[i] = std::min(y_ref[i], 255.0f);
      }

      // Call optimized micro-kernel.
      lutnorm(n(), x_data, t.data(), y.data());

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        EXPECT_NEAR(y_ref[i], float(y[i]), 0.5f)
          << "at position " << i << ", n = " << n() << ", sum = " << sum;
      }
    }
  }

 private:
  size_t n_{1};
  bool inplace_{false};
  size_t iterations_{15};
};

#define XNN_TEST_LUTNORM_N_EQ_1(ukernel, arch_flags, ...)                                                              \
  TEST(ukernel, n_eq_1)                                                                                                \
  {                                                                                                                    \
    LUTNormMicrokernelTester().n(1).Test(ukernel);                                                                     \
  }
#define XNN_TEST_LUTNORM_SMALL_N(ukernel, arch_flags, ...)                                                             \
  TEST(ukernel, small_n)                                                                                               \
  {                                                                                                                    \
    for (size_t i = 2; i <= 16; i++) {                                                                                 \
      LUTNormMicrokernelTester().n(i).Test(ukernel);                                                                   \
    }                                                                                                                  \
  }
#define XNN_TEST_LUTNORM_LARGE_N(ukernel, arch_flags, ...)                                                             \
  TEST(ukernel, large_n)                                                                                               \
  {                                                                                                                    \
    for (size_t i = 16; i <= 128; i += 2) {                                                                            \
      LUTNormMicrokernelTester().n(i).Test(ukernel);                                                                   \
    }                                                                                                                  \
  }
#define XNN_TEST_LUTNORM_N_EQ_1_INPLACE(ukernel, arch_flags, ...)                                                      \
  TEST(ukernel, n_eq_1_inplace)                                                                                        \
  {                                                                                                                    \
    LUTNormMicrokernelTester().n(1).inplace(true).Test(ukernel);                                                       \
  }
#define XNN_TEST_LUTNORM_SMALL_N_INPLACE(ukernel, arch_flags, ...)                                                     \
  TEST(ukernel, small_n_inplace)                                                                                       \
  {                                                                                                                    \
    for (size_t i = 2; i <= 16; i++) {                                                                                 \
      LUTNormMicrokernelTester().n(i).inplace(true).Test(ukernel);                                                     \
    }                                                                                                                  \
  }
#define XNN_TEST_LUTNORM_LARGE_N_INPLACE(ukernel, arch_flags, ...)                                                     \
  TEST(ukernel, large_n_inplace)                                                                                       \
  {                                                                                                                    \
    for (size_t i = 16; i <= 128; i += 2) {                                                                            \
      LUTNormMicrokernelTester().n(i).inplace(true).Test(ukernel);                                                     \
    }                                                                                                                  \
  }

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, datatype, params_type, init_params)                               \
  XNN_TEST_LUTNORM_N_EQ_1(ukernel, arch_flags, init_params);                                                           \
  XNN_TEST_LUTNORM_SMALL_N(ukernel, arch_flags, init_params);                                                          \
  XNN_TEST_LUTNORM_LARGE_N(ukernel, arch_flags, init_params);                                                          \
  XNN_TEST_LUTNORM_N_EQ_1_INPLACE(ukernel, arch_flags, init_params);                                                   \
  XNN_TEST_LUTNORM_SMALL_N_INPLACE(ukernel, arch_flags, init_params);                                                  \
  XNN_TEST_LUTNORM_LARGE_N_INPLACE(ukernel, arch_flags, init_params);
#include "src/u8-lut32norm/u8-lut32norm.h"
#undef XNN_UKERNEL_WITH_PARAMS
