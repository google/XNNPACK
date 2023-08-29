// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2023 Google LLC
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

class GemmGoiMicrokernelTester {
 public:
  inline GemmGoiMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline GemmGoiMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }


  inline GemmGoiMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  inline size_t kr() const {
    return this->kr_;
  }

  inline GemmGoiMicrokernelTester& sr(size_t sr) {
    this->sr_ = sr;
    return *this;
  }

  inline size_t sr() const {
    return this->sr_;
  }

  inline GemmGoiMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GemmGoiMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GemmGoiMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline GemmGoiMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  inline size_t ks() const {
    return this->ks_;
  }

  inline GemmGoiMicrokernelTester& a_stride(size_t a_stride) {
    this->a_stride_ = a_stride;
    return *this;
  }

  inline size_t a_stride() const {
    return this->a_stride_ == 0 ? k() : this->a_stride_;
  }

  inline GemmGoiMicrokernelTester& cm_stride(size_t cm_stride) {
    this->cm_stride_ = cm_stride;
    return *this;
  }

  inline size_t cm_stride() const {
    return this->cm_stride_ == 0 ? cn_stride() * ((n() - 1) / nr()) + (n() - 1) % nr() + 1 : this->cm_stride_;
  }

  inline GemmGoiMicrokernelTester& cn_stride(size_t cn_stride) {
    this->cn_stride_ = cn_stride;
    return *this;
  }

  inline size_t cn_stride() const {
    return this->cn_stride_ == 0 ? nr() : this->cn_stride_;
  }

  inline GemmGoiMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  inline uint8_t a_zero_point() const {
    return this->a_zero_point_;
  }

  inline GemmGoiMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  inline uint8_t b_zero_point() const {
    return this->b_zero_point_;
  }

  inline GemmGoiMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GemmGoiMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GemmGoiMicrokernelTester& a_offset(size_t a_offset) {
    this->a_offset_ = a_offset;
    return *this;
  }

  inline size_t a_offset() const {
    return this->a_offset_;
  }

  inline GemmGoiMicrokernelTester& zero_index(size_t zero_index) {
    this->zero_index_ = zero_index;
    return *this;
  }

  inline size_t zero_index() const {
    return this->zero_index_;
  }

  inline GemmGoiMicrokernelTester& extended_weights(bool extended_weights) {
    this->extended_weights_ = extended_weights;
    return *this;
  }

  inline bool extended_weights() const {
    return this->extended_weights_;
  }

  inline GemmGoiMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(
    xnn_f32_gemm_goi_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_minmax_params_fn init_params) const
  {
    ASSERT_LE(m(), mr());
    ASSERT_GE(a_stride(), k());
    ASSERT_GE(cm_stride(), n());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(n() * k());
    std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
      std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);

      // todo: remove packw
      //std::fill(packed_w.begin(), packed_w.end(), 0.0f);
      //xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data(), 0, nullptr);

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LT(m_index * n() + n_index, c_ref.size());
            c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] *
              b[n_index * k() + k_index];
          }
        }
      }

      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min =
          qmin() == std::numeric_limits<uint8_t>::min() ? -std::numeric_limits<float>::infinity()
                      : accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float c_max =
          qmax() == std::numeric_limits<uint8_t>::max() ? +std::numeric_limits<float>::infinity()
                        : accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, c_min, c_max);

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
        }
      }

      gemm_minmax(m(), n(), k() * sizeof(float),
        a.data(), a_stride() * sizeof(float),
        b.data(),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
        &params);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          EXPECT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
          EXPECT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
          EXPECT_NEAR(
              c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
              c_ref[i * n() + j],
              std::max(1.0e-5f, std::abs(c_ref[i * n() + j]) * 1.0e-6f))
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t kr_{1};
  size_t sr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t a_stride_{0};
  size_t cm_stride_{0};
  size_t cn_stride_{0};
  uint8_t a_zero_point_{127};
  uint8_t b_zero_point_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t a_offset_{0};
  size_t zero_index_{SIZE_MAX};
  bool extended_weights_{false};
  size_t iterations_{15};
};
