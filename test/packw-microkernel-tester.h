// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pack.h"
#include "test/replicable_random_device.h"

class PackWMicrokernelTester {
 public:
  PackWMicrokernelTester& g(size_t g) {
    this->g_ = g;
    return *this;
  }

  size_t g() const { return this->g_; }

  PackWMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  size_t nr() const { return this->nr_; }

  PackWMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  size_t kr() const { return this->kr_; }

  PackWMicrokernelTester& sr(size_t sr) {
    this->sr_ = sr;
    return *this;
  }

  size_t sr() const { return this->sr_; }

  PackWMicrokernelTester& izp(size_t izp) {
    this->izp_ = izp;
    return *this;
  }

  size_t izp() const { return this->izp_; }

  PackWMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  size_t n() const { return this->n_; }

  size_t packed_k() const { return round_up_po2(k(), kr() * sr()); }

  size_t packed_n() const { return round_up(n(), nr()); }

  PackWMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  size_t k() const { return this->k_; }

  PackWMicrokernelTester& nullbias(bool nullbias) {
    this->nullbias_ = nullbias;
    return *this;
  }

  bool nullbias() const { return this->nullbias_; }

  void Test(xnn_qs8_packw_gemm_goi_ukernel_fn packw) const {
    xnnpack::Buffer<int8_t> weights(n() * k());
    xnnpack::Buffer<int32_t> bias(n());
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w_ref(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));

    std::iota(weights.begin(), weights.end(), 0);
    std::iota(bias.begin(), bias.end(), UINT32_C(0));
    std::fill(packed_w.begin(), packed_w.end(), INT8_C(0));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), INT8_C(0x7B));

    const int32_t* bias_data = nullbias() ? nullptr : bias.data();
    const xnn_qs8_packing_params packing_params = {0};

    // Compute reference results.
    auto* pack_function =
        izp() == 128 ? xnn_pack_qs8_to_qu8_gemm_goi_w : xnn_pack_qs8_gemm_goi_w;
    pack_function(/*g=*/1, n(), k(), nr(), kr(), sr(),
                  reinterpret_cast<const int8_t*>(weights.data()), bias_data,
                  /*scale=*/nullptr,
                  reinterpret_cast<void*>(packed_w_ref.data()),
                  /*extra_bytes=*/0, &packing_params);

    // Call optimized micro-kernel.
    packw(/*g=*/1, n(), k(), nr(), kr(), sr(), weights.data(), bias_data,
          /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
          &packing_params);

    // Verify bias results.
    for (size_t i = 0; i < packed_n() * sizeof(int32_t); i++) {
      if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
        EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i]);
      }
    }

    // Verify weights results.
    // NOTE remainder KC is different so k() is used instead of packed_k() for
    // loop
    for (size_t ki = 0; ki < k(); ki++) {
      for (size_t ni = 0; ni < (n()); ni++) {
        const size_t i = packed_n() * sizeof(int32_t) + ki * packed_n() + ni;
        if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
          EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i])
              << "kr " << kr() << " of kc " << k() << " packed_k " << packed_k()
              << "\n"
              << "nr " << nr() << " of nc " << n() << " packed_n " << packed_n()
              << "\n"
              << "at n " << i << " of "
              << (int32_t)(packed_n() * packed_k() +
                           packed_n() * sizeof(int32_t));
        }
      }
    }
  }

  void Test(xnn_qs8_packw_gemm_gio_ukernel_fn packw) const {
    xnnpack::Buffer<int8_t> weights(n() * k());
    xnnpack::Buffer<int32_t> bias(n());
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w_ref(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));

    std::iota(weights.begin(), weights.end(), 0);
    std::iota(bias.begin(), bias.end(), UINT32_C(0));
    std::fill(packed_w.begin(), packed_w.end(), INT8_C(0));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), INT8_C(0x7B));

    const int32_t* bias_data = nullbias() ? nullptr : bias.data();
    const xnn_qs8_packing_params packing_params = {0};

    // Compute reference results.
    auto* pack_function =
        izp() == 128 ? xnn_pack_qs8_to_qu8_gemm_gio_w : xnn_pack_qs8_gemm_gio_w;
    pack_function(/*g=*/1, n(), k(), nr(), kr(), sr(), n(),
                  reinterpret_cast<const int8_t*>(weights.data()), bias_data,
                  /*scale=*/nullptr,
                  reinterpret_cast<void*>(packed_w_ref.data()),
                  /*extra_bytes=*/0, &packing_params);

    // Call optimized micro-kernel.
    packw(/*g=*/1, n(), k(), nr(), kr(), sr(), n(), weights.data(), bias_data,
          /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
          &packing_params);

    // Verify bias results.
    for (size_t i = 0; i < packed_n() * sizeof(int32_t); i++) {
      if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
        EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i]);
      }
    }

    // Verify weights results.
    // NOTE remainder KC is different so k() is used instead of packed_k() for
    // loop
    for (size_t ki = 0; ki < k(); ki++) {
      for (size_t ni = 0; ni < (n()); ni++) {
        const size_t i = packed_n() * sizeof(int32_t) + ki * packed_n() + ni;
        if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
          EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i])
              << "kr " << kr() << " of kc " << k() << " packed_k " << packed_k()
              << "\n"
              << "nr " << nr() << " of nc " << n() << " packed_n " << packed_n()
              << "\n"
              << "at n " << i << " of "
              << (int32_t)(packed_n() * packed_k() +
                           packed_n() * sizeof(int32_t));
        }
      }
    }
  }

  void Test(xnn_x8_packw_gemm_goi_ukernel_fn packw) const {
    xnnpack::Buffer<int8_t> weights(n() * k());
    xnnpack::Buffer<uint32_t> bias(n());
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w_ref(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));
    std::iota(weights.begin(), weights.end(), 0);
    std::iota(bias.begin(), bias.end(), UINT32_C(0));
    std::fill(packed_w.begin(), packed_w.end(), INT8_C(0x12));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), INT8_C(0x7B));

    const uint32_t* bias_data = nullbias() ? nullptr : bias.data();
    const xnn_qs8_packing_params packing_params = {127};

    // Compute reference results.
    xnn_pack_f32_qs8w_gemm_goi_w(
        /*g=*/1, n(), k(), nr(), kr(), sr(),
        reinterpret_cast<const int8_t*>(weights.data()),
        reinterpret_cast<const float*>(bias_data),
        /*scale=*/nullptr, reinterpret_cast<void*>(packed_w_ref.data()),
        /*extra_bytes=*/0, &packing_params);

    // Call optimized micro-kernel.
    packw(/*g=*/1, n(), k(), nr(), kr(), sr(), weights.data(), bias_data,
          /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
          &packing_params);

    // Verify results.
    for (size_t i = 0; i < (packed_n() * k() + packed_n() * sizeof(int32_t));
         i++) {
      if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
        EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i])
            << "at n " << i << " of "
            << (int32_t)(packed_n() * k() + packed_n());
      }
    }
  }

  void Test(xnn_x8_packw_gemm_gio_ukernel_fn packw) const {
    xnnpack::Buffer<int8_t> weights(n() * k());
    xnnpack::Buffer<uint32_t> bias(n());
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w_ref(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));
    std::iota(weights.begin(), weights.end(), 0);
    std::iota(bias.begin(), bias.end(), UINT32_C(0));
    std::fill(packed_w.begin(), packed_w.end(), INT8_C(0x12));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), INT8_C(0x7B));

    const uint32_t* bias_data = nullbias() ? nullptr : bias.data();
    const xnn_qs8_packing_params packing_params = {127};

    // Compute reference results.
    xnn_pack_f32_qs8w_gemm_gio_w(
        /*g=*/1, n(), k(), nr(), kr(), sr(), n(),
        reinterpret_cast<const int8_t*>(weights.data()),
        reinterpret_cast<const float*>(bias_data),
        /*scale=*/nullptr, reinterpret_cast<void*>(packed_w_ref.data()),
        /*extra_bytes=*/0, &packing_params);

    // Call optimized micro-kernel.
    packw(/*g=*/1, n(), k(), nr(), kr(), sr(), n(), weights.data(), bias_data,
          /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
          &packing_params);

    // Verify results.
    for (size_t i = 0; i < (packed_n() * k() + packed_n() * sizeof(int32_t));
         i++) {
      if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
        EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i])
            << "at n " << i << " of "
            << (int32_t)(packed_n() * k() + packed_n());
      }
    }
  }

  void Test(xnn_qs8_qc4w_packw_gemm_goi_ukernel_fn packw) const {
    xnnpack::ReplicableRandomDevice rng;
    auto i32rng = std::bind(
        std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));

    const size_t k2 = round_up_po2(k(), 2);  // Round up to byte aligned rows

    xnnpack::Buffer<uint8_t> weights(n() * k2 / 2);
    xnnpack::Buffer<int32_t> bias(n());
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w_ref(
        packed_n() * packed_k() + packed_n() * sizeof(uint32_t));

    xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
    std::generate(bias.begin(), bias.end(), std::ref(i32rng));
    std::fill(packed_w.begin(), packed_w.end(), INT8_C(0));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), INT8_C(0x7B));

    const int32_t* bias_data = nullbias() ? nullptr : bias.data();
    const xnn_qs8_qc4w_packing_params packing_params = {0};

    // Compute reference results.
    xnn_pack_qs8_qc4w_gemm_goi_w(
        /*g=*/1, n(), k2, nr(), kr(), sr(), weights.data(), bias_data,
        /*scale=*/nullptr, reinterpret_cast<void*>(packed_w_ref.data()),
        /*extra_bytes=*/0, &packing_params);

    // Call optimized micro-kernel.
    packw(/*g=*/1, n(), k2, nr(), kr(), sr(), weights.data(), bias_data,
          /*scale=*/nullptr, packed_w.data(), /*extra_bytes=*/0,
          &packing_params);

    // Verify bias results.
    for (size_t i = 0; i < packed_n() * sizeof(int32_t); i++) {
      if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
        EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i]);
      }
    }

    // Verify weights results.
    // NOTE remainder KC is different so k2 is used instead of packed_k() for
    // loop
    for (size_t ki = 0; ki < k2; ki++) {
      for (size_t ni = 0; ni < (n()); ni++) {
        const size_t i = packed_n() * sizeof(int32_t) + ki * packed_n() + ni;
        if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
          EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i])
              << "kr " << kr() << " of kc " << k2 << " packed_k " << packed_k()
              << "\n"
              << "nr " << nr() << " of nc " << n() << " packed_n " << packed_n()
              << "\n"
              << "at n " << i << " of "
              << (int32_t)(packed_n() * packed_k() +
                           packed_n() * sizeof(int32_t));
        }
      }
    }
  }

  void Test(xnn_x16_packw_gemm_goi_ukernel_fn packw) const {
    xnnpack::Buffer<xnn_float16> weights(g() * n() * k());
    xnnpack::Buffer<xnn_float16> padded_weights(g() * n() * packed_k());
    xnnpack::Buffer<xnn_float16> bias(g() * n());
    xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_w(
        g() * (packed_n() * packed_k() + packed_n()));
    xnnpack::Buffer<xnn_float16> packed_w_ref(
        g() * (packed_n() * packed_k() + packed_n()));

    const xnn_float16 pad_value =
        xnn_float16_from_bits(std::max(sr(), kr()) == 1 ? UINT16_C(0xDEAD) : 0);
    std::iota(weights.begin(), weights.end(), 1.0f);
    std::iota(bias.begin(), bias.end(), 0.5f);
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), pad_value);

    // Mandate zero-padding of weights to packed_k() in K dimension.
    std::fill(padded_weights.begin(), padded_weights.end(), 0.0f);
    for (size_t gid = 0; gid < g(); gid++) {
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < k(); j++) {
          padded_weights[(gid * n() + i) * packed_k() + j] =
              weights[(gid * n() + i) * k() + j];
        }
      }
    }

    const xnn_float16* bias_data = nullbias() ? nullptr : bias.data();

    // Compute reference results.
    xnn_pack_f16_gemm_goi_w(
        g(), n(), packed_k(), nr(), kr(), sr(),
        reinterpret_cast<const uint16_t*>(padded_weights.data()),
        reinterpret_cast<const uint16_t*>(bias_data),
        /*scale=*/nullptr, reinterpret_cast<uint16_t*>(packed_w_ref.data()),
        /*extra_bytes=*/0, /*params=*/nullptr);

    // Call optimized micro-kernel.
    packw(g(), n(), k(), nr(), kr(), sr(),
          reinterpret_cast<const uint16_t*>(weights.data()),
          reinterpret_cast<const uint16_t*>(bias_data), /*scale=*/nullptr,
          reinterpret_cast<uint16_t*>(packed_w.data()),
          /*extra_bytes=*/0, /*params=*/nullptr);

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      // Ignore padding in N dimension.
      if (packed_w_ref[i] != pad_value) {
        ASSERT_EQ(packed_w[i], packed_w_ref[i])
            << "at position " << i << " / " << packed_w.size() << ", n " << n()
            << ", k " << k();
      }
    }
  }

  void Test(xnn_x16_x32_packw_gemm_gio_ukernel_fn packw) const {
    xnnpack::Buffer<xnn_bfloat16> weights(g() * n() * k());
    xnnpack::Buffer<float> bias(g() * n());
    xnnpack::Buffer<xnn_bfloat16, XNN_ALLOCATION_ALIGNMENT> packed_w(
        g() * (packed_n() * packed_k() + 2 * packed_n()));
    xnnpack::Buffer<xnn_bfloat16> packed_w_ref(
        g() * (packed_n() * packed_k() + 2 * packed_n()));

    const xnn_bfloat16 pad_value = xnn_bfloat16_from_bits(
        std::max(sr(), kr()) == 1 ? UINT16_C(0xDEAD) : 0);
    std::iota(weights.begin(), weights.end(), 1.0f);

    std::iota(bias.begin(), bias.end(), 0.5f);
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), pad_value);

    const float* bias_data = nullbias() ? nullptr : bias.data();

    // Compute reference results.
    xnn_pack_bf16_f32_gemm_gio_w(g(), n(), k(), nr(), kr(), sr(), n(),
                                 weights.data(), bias_data,
                                 /*scale=*/nullptr, packed_w_ref.data(),
                                 /*extra_bytes=*/0, /*params=*/nullptr);

    // Call optimized micro-kernel.
    packw(g(), n(), k(), nr(), kr(), sr(), n(),
          reinterpret_cast<const uint16_t*>(weights.data()),
          reinterpret_cast<const uint32_t*>(bias_data), /*scale=*/nullptr,
          reinterpret_cast<uint16_t*>(packed_w.data()),
          /*extra_bytes=*/0, /*params=*/nullptr);

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      // Ignore padding in N dimension.
      if (packed_w_ref[i] != pad_value) {
        ASSERT_EQ(packed_w[i], packed_w_ref[i])
            << "at position " << i << " / " << packed_w.size() << ", n " << n()
            << ", k " << k();
      }
    }
  }

  void Test(xnn_x16_x32_packw_gemm_goi_ukernel_fn packw) const {
    xnnpack::Buffer<xnn_bfloat16> weights(g() * n() * k());
    xnnpack::Buffer<float> bias(g() * n());
    xnnpack::Buffer<xnn_bfloat16, XNN_ALLOCATION_ALIGNMENT> packed_w(
        g() * (packed_n() * packed_k() + 2 * packed_n()));
    xnnpack::Buffer<xnn_bfloat16> packed_w_ref(
        g() * (packed_n() * packed_k() + 2 * packed_n()));

    const xnn_bfloat16 pad_value = xnn_bfloat16_from_bits(
        std::max(sr(), kr()) == 1 ? UINT16_C(0xDEAD) : 0);
    std::iota(weights.begin(), weights.end(), 1.0f);
    std::iota(bias.begin(), bias.end(), 0.5f);
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), pad_value);

    const float* bias_data = nullbias() ? nullptr : bias.data();

    // Compute reference results.
    xnn_pack_bf16_f32_gemm_goi_w(g(), n(), k(), nr(), kr(), sr(),
                                 weights.data(), bias_data,
                                 /*scale=*/nullptr, packed_w_ref.data(),
                                 /*extra_bytes=*/0, /*params=*/nullptr);

    // Call optimized micro-kernel.
    packw(g(), n(), k(), nr(), kr(), sr(),
          reinterpret_cast<const uint16_t*>(weights.data()),
          reinterpret_cast<const uint32_t*>(bias_data), /*scale=*/nullptr,
          reinterpret_cast<uint16_t*>(packed_w.data()),
          /*extra_bytes=*/0, /*params=*/nullptr);

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      // Ignore padding in N dimension.
      if (packed_w_ref[i] != pad_value) {
        ASSERT_EQ(packed_w[i], packed_w_ref[i])
            << "at position " << i << " / " << packed_w.size() << ", n " << n()
            << ", k " << k();
      }
    }
  }

  void Test(xnn_x32_packw_gemm_goi_ukernel_fn packw) const {
    xnnpack::Buffer<uint32_t> weights(g() * n() * k());
    xnnpack::Buffer<uint32_t> padded_weights(g() * n() * packed_k());
    xnnpack::Buffer<uint32_t> bias(g() * n());
    xnnpack::Buffer<uint32_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
        g() * (packed_n() * packed_k() + packed_n()));
    xnnpack::Buffer<uint32_t> packed_w_ref(
        g() * (packed_n() * packed_k() + packed_n()));

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
          padded_weights[(gid * n() + i) * packed_k() + j] =
              weights[(gid * n() + i) * k() + j];
        }
      }
    }

    const uint32_t* bias_data = nullbias() ? nullptr : bias.data();

    // Compute reference results.
    xnn_pack_f32_gemm_goi_w(
        g(), n(), packed_k(), nr(), kr(), sr(),
        reinterpret_cast<const float*>(padded_weights.data()),
        reinterpret_cast<const float*>(bias_data),
        /*scale=*/nullptr, reinterpret_cast<float*>(packed_w_ref.data()),
        /*extra_bytes=*/0, /*params=*/nullptr);

    // Call optimized micro-kernel.
    packw(g(), n(), k(), nr(), kr(), sr(), weights.data(), bias_data,
          /*scale=*/nullptr, packed_w.data(),
          /*extra_bytes=*/0, /*params=*/nullptr);

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      // Ignore padding in N dimension.
      if (packed_w_ref[i] != pad_value) {
        ASSERT_EQ(packed_w[i], packed_w_ref[i])
            << "at position " << i << " / " << packed_w.size() << ", n " << n()
            << ", k " << k();
      }
    }
  }

  void Test(xnn_x32_packw_gemm_gio_ukernel_fn packw) const {
    xnnpack::Buffer<uint32_t> weights(g() * n() * k());
    xnnpack::Buffer<uint32_t> padded_weights(g() * n() * packed_k());
    xnnpack::Buffer<uint32_t> bias(g() * n());
    xnnpack::Buffer<uint32_t, XNN_ALLOCATION_ALIGNMENT> packed_w(
        g() * (packed_n() * packed_k() + packed_n()));
    xnnpack::Buffer<uint32_t> packed_w_ref(
        g() * (packed_n() * packed_k() + packed_n()));

    const uint32_t pad_value = UINT32_C(0xDEADBEEF);
    std::iota(weights.begin(), weights.end(), UINT32_C(0x00000003));
    std::iota(bias.begin(), bias.end(), UINT32_C(0x80000000));
    std::fill(packed_w.begin(), packed_w.end(), UINT32_C(0x12345678));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), pad_value);

    // Mandate zero-padding of weights to packed_k() in K dimension.
    std::fill(padded_weights.begin(), padded_weights.end(), 0);
    for (size_t gid = 0; gid < g(); gid++) {
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < k(); j++) {
          padded_weights[(gid * n() + i) * packed_k() + j] =
              weights[(gid * n() + i) * k() + j];
        }
      }
    }

    const uint32_t* bias_data = nullbias() ? nullptr : bias.data();

    // Compute reference results.
    xnn_pack_f32_gemm_gio_w(
        g(), n(), packed_k(), nr(), kr(), sr(), n(),
        reinterpret_cast<const float*>(padded_weights.data()),
        reinterpret_cast<const float*>(bias_data),
        /*scale=*/nullptr, reinterpret_cast<float*>(packed_w_ref.data()),
        /*extra_bytes=*/0, /*params=*/nullptr);

    // Call optimized micro-kernel.
    packw(g(), n(), k(), nr(), kr(), sr(), n(), weights.data(), bias_data,
          /*scale=*/nullptr, packed_w.data(),
          /*extra_bytes=*/0, /*params=*/nullptr);

    // Verify bias results.
    for (size_t i = 0; i < packed_n(); i++) {
      if (packed_w_ref[i] != pad_value) {  // Allow pad to differ
        EXPECT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i]);
      }
    }

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      // Ignore padding in N dimension.
      if (packed_w_ref[i] != pad_value) {
        ASSERT_EQ(packed_w[i], packed_w_ref[i])
            << "kr " << kr() << " of kc " << k() << " packed_k " << packed_k()
            << "\n"
            << "nr " << nr() << " of nc " << n() << " packed_n " << packed_n()
            << "\n"
            << "at n " << i << " of " << packed_w.size();
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
  size_t izp_{0};
};
