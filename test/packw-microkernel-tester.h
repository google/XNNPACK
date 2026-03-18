// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_PACKW_MICROKERNEL_TESTER_H_
#define XNNPACK_TEST_PACKW_MICROKERNEL_TESTER_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
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

  PackWMicrokernelTester& bl(size_t bl) {
    this->bl_ = bl;
    return *this;
  }

  size_t bl() const { return this->bl_; }

  PackWMicrokernelTester& kzp(size_t kzp) {
    this->kzp_ = kzp;
    return *this;
  }

  size_t kzp() const { return this->kzp_; }

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

  void Test(xnn_qb4_packw_gemm_goi_ukernel_fn packw) const {
    size_t packed_k2 = round_up_po2(k(), kr() * sr() * 2);
    size_t k_num_blocks = packed_k2 / bl();
    size_t packed_k_bytes = packed_k2 / 2;
    size_t packed_weight_size = packed_n() * (
      packed_k_bytes + sizeof(float) + k_num_blocks * sizeof(uint16_t) + sizeof(float)
    );

    xnnpack::Buffer<uint8_t> weights(XNN_EXTRA_BYTES / sizeof(int8_t) +
                                     n() * k());
    xnnpack::Buffer<int32_t> bias(packed_n());
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w(packed_weight_size);
    xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_w_ref(packed_weight_size);
    xnnpack::Buffer<xnn_bfloat16, XNN_ALLOCATION_ALIGNMENT> bf16_scales(
        n() * k_num_blocks
    );

    std::iota(weights.begin(), weights.end(), 0);
    std::iota(bias.begin(), bias.end(), UINT32_C(15));
    std::fill(packed_w.begin(), packed_w.end(), INT8_C(0));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), INT8_C(0));
    std::iota(bf16_scales.begin(), bf16_scales.end(), 3.75);

    const int32_t* bias_data = nullbias() ? nullptr : bias.data();
    const xnn_bfloat16* scale_data = bf16_scales.data();
    const xnn_qs8_qc4w_packing_params packing_params = {1, static_cast<uint8_t>(kzp())};

    // Compute reference results.
    xnn_pack_qs8_qb4w_gemm_goi_w(
        /*g=*/1, n(), k(), nr(), kr(), sr(), bl(), weights.data(), nullptr,
        /*scale=*/scale_data, packed_w_ref.data(), sizeof(uint16_t) * nr(),
        /*extra_bytes=*/sizeof(float) * nr(), &packing_params);

    // fill in scale as second step (reference)
    size_t stride = nr() * (packed_k_bytes + k_num_blocks * sizeof(uint16_t) + sizeof(float) + sizeof(float));
    size_t block_stride = (bl() /2 + sizeof(uint16_t)) * nr();
    size_t start_offset = nr() * (packed_k_bytes / k_num_blocks + sizeof(float));
    xnn_init_blockwise_scale_bf16_params(
        /*channels=*/n(),
        /*channels_tile=*/nr(),
        /*stride=*/stride,
        /*num_blocks=*/k_num_blocks,
        /*block_stride=*/block_stride,
        /*scale=*/scale_data,
        /*packed_w=*/packed_w_ref.data() + start_offset);

    void* bias_start =
        (void*)((uintptr_t)packed_w_ref.data() + stride - nr() * sizeof(float));

    if (!nullbias()) {
      xnn_init_qs8_qc8w_scale_fp32_params(n(), nr(), stride, (float*)bias_data,
                                          bias_start);
    }

    // Call optimized micro-kernel.
    packw(/*g=*/1, n(), k(), nr(), kr(), sr(), bl(), weights.data(), bias_data,
          /*scale=*/scale_data, packed_w.data(), sizeof(uint16_t) * nr(),
          /*extra_bytes=*/sizeof(float) * nr(), &packing_params);

    const uint8_t* packed_data = (uint8_t*)packed_w.data();
    const uint8_t* packed_ref_data = (uint8_t*)packed_w_ref.data();

    // Compare Packed Tensors.
    for (size_t n_block_start = 0; n_block_start < packed_n();
         n_block_start += nr()) {
      // Number of output channels in this block
      size_t n_remainder = min(nr(), n() - n_block_start);
      // Check KScaledSums
      float* kscale_sum_start = (float*)packed_data;
      float* kscale_sum_ref_start = (float*)packed_ref_data;
      for (size_t ni = 0; ni < n_remainder; ni++) {
        EXPECT_EQ((float)kscale_sum_start[ni], (float)kscale_sum_ref_start[ni])
            << "kscaled sum at index: " << ni
            << " of n_block_start: " << n_block_start << "\n";
      }

      packed_data += nr() * sizeof(float);
      packed_ref_data += nr() * sizeof(float);

      for (size_t bl_start = 0; bl_start < k(); bl_start += bl()) {
        // Check nibbles
        size_t num_planes_block = bl() / (2 * kr());
        for (size_t pi = 0; pi < num_planes_block; pi += 1) {
          for (size_t ni = 0; ni < n_remainder; ni++) {
            for (size_t ki = 0; ki < 2 * kr(); ki++) {
              size_t i = (2 * kr()) * (nr() * pi + ni) + ki;
              uint8_t val_ref = ((i & 1) ? (uint8_t)packed_ref_data[i >> 1] >> 4
                                         : packed_ref_data[i >> 1] & 0xF);
              uint8_t val = ((i & 1) ? (uint8_t)packed_data[i >> 1] >> 4
                                     : packed_data[i >> 1] & 0xF);
              EXPECT_EQ(val_ref, val)
                  << " nibbles do not match location at \n"
                  << "nr_block_start: " << n_block_start << ", plane: " << pi
                  << "\n"
                  << " ni: " << ni << " ki: " << ki << " i: " << i << "\n";
            }
          }
        }
        packed_data += ((bl() * nr()) >> 1) * sizeof(uint8_t);
        packed_ref_data += ((bl() * nr()) >> 1) * sizeof(uint8_t);
        // check scales
        uint16_t* scales_start = (uint16_t*)packed_data;
        uint16_t* scales_ref_start = (uint16_t*)packed_ref_data;
        for (size_t ni = 0; ni < n_remainder; ni++) {
          // Packing divides the scales by 16, multiplying back is a bit easier
          // for readability
          EXPECT_EQ(math_cvt_fp32_bf16(scales_start[ni]) * 16,
                    math_cvt_fp32_bf16(scales_ref_start[ni]) * 16)
              << "n_block_start " << n_block_start << " ni " << ni;
        }

        packed_data += nr() * sizeof(uint16_t);
        packed_ref_data += nr() * sizeof(uint16_t);
      }
      // check bias
      uint32_t* bias_start = (uint32_t*)packed_data;
      uint32_t* bias_ref_start = (uint32_t*)packed_ref_data;
      for (size_t ni = 0; ni < n_remainder; ni++) {
        EXPECT_EQ(bias_start[ni], bias_ref_start[ni])
            << "n_block_start " << n_block_start << " ni " << ni;
      }
      packed_ref_data += nr() * sizeof(uint32_t);
      packed_data += nr() * sizeof(uint32_t);
    }
  }

 private:
  size_t g_{1};
  size_t n_{1};
  size_t nr_{1};
  size_t kr_{1};
  size_t sr_{1};
  size_t k_{1};
  size_t bl_{1};
  bool nullbias_{false};
  size_t izp_{0};
  size_t kzp_{8};
};

#endif  // XNNPACK_TEST_PACKW_MICROKERNEL_TESTER_H_
