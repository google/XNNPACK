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
#include <numeric>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/pack.h>
#include <xnnpack/packw.h>

class PackWMicrokernelTester {
 public:

  inline PackWMicrokernelTester& g(size_t g) {
    this->g_ = g;
    return *this;
  }

  inline size_t g() const {
    return this->g_;
  }

  inline PackWMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline PackWMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  inline size_t kr() const {
    return this->kr_;
  }

  inline PackWMicrokernelTester& sr(size_t sr) {
    this->sr_ = sr;
    return *this;
  }

  inline size_t sr() const {
    return this->sr_;
  }

  inline PackWMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline size_t packed_k() const {
    return round_up_po2(k(), kr() * sr());
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

  void Test(xnn_x8_packw_gemm_goi_ukernel_fn packw) const {
    std::vector<int8_t> weights(n() * k());
    std::vector<uint32_t> bias(n());
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(packed_n() * k() + packed_n() * sizeof(uint32_t));
    std::vector<int8_t> packed_w_ref(packed_n() * k() + packed_n() * sizeof(uint32_t));

    std::iota(weights.begin(), weights.end(), 0);
    std::iota(bias.begin(), bias.end(), UINT32_C(0));
    std::fill(packed_w.begin(), packed_w.end(), INT8_C(0x12));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), INT8_C(0x7B));

    const uint32_t* bias_data = nullbias() ? nullptr : bias.data();
    const xnn_qs8_packing_params packing_params = { 127 };

    // Compute reference results.
    xnn_pack_f32_qs8w_gemm_goi_w(/*g=*/1, n(), k(), nr(), /*kr=*/1, sr(),
      reinterpret_cast<const int8_t *>(weights.data()),
      reinterpret_cast<const float *>(bias_data),
      /*scale=*/nullptr,
      reinterpret_cast<void *>(packed_w_ref.data()),
      /*extra_bytes=*/0, &packing_params);

    // Call optimized micro-kernel.
    packw(/*g=*/1, n(), k(), nr(), /*kr=*/1, sr(),
      weights.data(), bias_data, /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0, &packing_params);

    // Verify results.
    for (size_t i = 0; i < (packed_n() * k() + packed_n() * sizeof(int32_t)); i++) {
      if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
        EXPECT_EQ((int32_t) packed_w[i], (int32_t) packed_w_ref[i])
            << "at n " << i << " of " << (int32_t) (packed_n() * k() + packed_n());
      }
    }
  }

  void Test(xnn_x16_packw_gemm_goi_ukernel_fn packw) const {
    std::vector<uint16_t> weights(g() * n() * k());
    std::vector<uint16_t> padded_weights(g() * n() * packed_k());
    std::vector<uint16_t> bias(g() * n());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_w(g() * (packed_n() * packed_k() + packed_n()));
    std::vector<uint16_t> packed_w_ref(g() * (packed_n() * packed_k() + packed_n()));

    const uint16_t pad_value = std::max(sr(), kr()) == 1 ? UINT16_C(0xDEAD) : 0;
    std::iota(weights.begin(), weights.end(), UINT16_C(0x0001));
    std::iota(bias.begin(), bias.end(), UINT16_C(0x8000));
    std::fill(packed_w.begin(), packed_w.end(), UINT16_C(0xBEEF));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), pad_value);

    // Mandate zero-padding of weights to packed_k() in K dimension.
    std::fill(padded_weights.begin(), padded_weights.end(), 0);
    for (size_t gid = 0; gid < g(); gid++) {
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < k(); j++) {
          padded_weights[(gid * n() + i) * packed_k() + j] = weights[(gid * n() + i) * k() + j];
        }
      }
    }

    const uint16_t* bias_data = nullbias() ? nullptr : bias.data();

    // Compute reference results.
    xnn_pack_f16_gemm_goi_w(g(), n(), packed_k(), nr(), kr(), sr(),
      reinterpret_cast<const uint16_t*>(padded_weights.data()),
      reinterpret_cast<const uint16_t*>(bias_data),
      /*scale=*/nullptr,
      reinterpret_cast<uint16_t*>(packed_w_ref.data()),
      /*extra_bytes=*/0, /*params=*/nullptr);

    // Call optimized micro-kernel.
    packw(g(), n(), k(), nr(), kr(), sr(),
      weights.data(), bias_data, /*scale=*/nullptr, packed_w.data(),
      /*extra_bytes=*/0, /*params=*/nullptr);

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      // Ignore padding in N dimension.
      if (packed_w_ref[i] != pad_value) {
        ASSERT_EQ(packed_w[i], packed_w_ref[i])
            << "at position " << i << " / " << packed_w.size()
            << ", n " << n() << ", k " << k();
      }
    }
  }

  void Test(xnn_x32_packw_gemm_goi_ukernel_fn packw) const {
    std::vector<uint32_t> weights(g() * n() * k());
    std::vector<uint32_t> padded_weights(g() * n() * packed_k());
    std::vector<uint32_t> bias(g() * n());
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> packed_w(g() * (packed_n() * packed_k() + packed_n()));
    std::vector<uint32_t> packed_w_ref(g() * (packed_n() * packed_k() + packed_n()));

    const uint32_t pad_value = UINT32_C(0xDEADBEEF);
    std::iota(weights.begin(), weights.end(), UINT32_C(0x00000001));
    std::iota(bias.begin(), bias.end(), UINT32_C(0x80000000));
    std::fill(packed_w.begin(), packed_w.end(), UINT32_C(0x12345678));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), pad_value);

    // Mandate zero-padding of weights to packed_k() in K dimension.
    std::fill(padded_weights.begin(), padded_weights.end(), 0);
    for (size_t gid = 0; gid < g(); gid++) {
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < k(); j++) {
          padded_weights[(gid * n() + i) * packed_k() + j] = weights[(gid * n() + i) * k() + j];
        }
      }
    }

    const uint32_t* bias_data = nullbias() ? nullptr : bias.data();

    // Compute reference results.
    xnn_pack_f32_gemm_goi_w(g(), n(), packed_k(), nr(), kr(), sr(),
      reinterpret_cast<const float*>(padded_weights.data()),
      reinterpret_cast<const float*>(bias_data),
      /*scale=*/nullptr,
      reinterpret_cast<float*>(packed_w_ref.data()),
      /*extra_bytes=*/0, /*params=*/nullptr);

    // Call optimized micro-kernel.
    packw(g(), n(), k(), nr(), kr(), sr(),
      weights.data(), bias_data, /*scale=*/nullptr, packed_w.data(),
      /*extra_bytes=*/0, /*params=*/nullptr);

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      // Ignore padding in N dimension.
      if (packed_w_ref[i] != pad_value) {
        ASSERT_EQ(packed_w[i], packed_w_ref[i])
            << "at position " << i << " / " << packed_w.size()
            << ", n " << n() << ", k " << k();
      }
    }
  }

 private:
  size_t g_{1};
  size_t n_{1};
  size_t nr_{1};
  size_t kr_{1};
  size_t sr_{1};
  size_t k_{1};
  bool nullbias_{false};
};
