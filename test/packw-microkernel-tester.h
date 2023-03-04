// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/pack.h>
#include <xnnpack/packw.h>

class PackWMicrokernelTester {
 public:

  inline PackWMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline PackWMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline size_t packed_n() const {
    return round_up(n(), nr());
  }

  inline PackWMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline PackWMicrokernelTester& nullbias(bool nullbias) {
    this->nullbias_ = nullbias;
    return *this;
  }

  inline bool nullbias() const {
    return this->nullbias_;
  }

  inline PackWMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_x32_packw_gemm_goi_ukernel_fn packw) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    std::vector<uint32_t> weights(n() * k());
    std::vector<uint32_t> bias(n());
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> packed_w(packed_n() * k() + packed_n());
    std::vector<uint32_t> packed_w_ref(packed_n() * k() + packed_n());


    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(weights.begin(), weights.end(), std::ref(u32rng));
      std::generate(bias.begin(), bias.end(), std::ref(u32rng));
      std::fill(packed_w.begin(), packed_w.end(), UINT32_C(0x12345678));
      std::fill(packed_w_ref.begin(), packed_w_ref.end(), UINT32_C(0xDEADBEEF));

      const uint32_t* bias_data = nullbias() ? nullptr : bias.data();

      // Compute reference results.
      xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), 1 /* kr */, 1 /* sr */,
        reinterpret_cast<const float *>(weights.data()), reinterpret_cast<const float *>(bias_data), reinterpret_cast<float *>(packed_w_ref.data()), 0, nullptr);

      // Call optimized micro-kernel.
      packw(1, n(), k(), nr(), 1 /* kr */, 1 /* sr */, weights.data(), bias_data, packed_w.data(), 0, nullptr);

      // Verify results.
      for (size_t i = 0; i < (packed_n() * k() + packed_n()); i++) {
        if (packed_w_ref[i] !=  UINT32_C(0xDEADBEEF)) {  // Allow pad to differ
          EXPECT_EQ(packed_w[i], packed_w_ref[i])
              << "at n " << i << n();
        }
      }
    }
  }

 private:
  size_t n_{1};
  size_t nr_{1};
  size_t k_{100};
  bool nullbias_{false};
  size_t iterations_{15};

};
