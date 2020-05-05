// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
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
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>


class GemmMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline GemmMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline GemmMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }


  inline GemmMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  inline size_t kr() const {
    return this->kr_;
  }

  inline GemmMicrokernelTester& sr(size_t sr) {
    this->sr_ = sr;
    return *this;
  }

  inline size_t sr() const {
    return this->sr_;
  }

  inline GemmMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GemmMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GemmMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline GemmMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  inline size_t ks() const {
    return this->ks_;
  }

  inline size_t packed_k() const {
    return k() % kr() == 0 ? k() : (k() / kr() + 1) * kr();
  }

  inline size_t packed_n() const {
    return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
  }

  inline size_t bias_n() const {
    return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
  }

  inline GemmMicrokernelTester& a_stride(size_t a_stride) {
    this->a_stride_ = a_stride;
    return *this;
  }

  inline size_t a_stride() const {
    return this->a_stride_ == 0 ? k() : this->a_stride_;
  }

  inline GemmMicrokernelTester& cm_stride(size_t cm_stride) {
    this->cm_stride_ = cm_stride;
    return *this;
  }

  inline size_t cm_stride() const {
    return this->cm_stride_ == 0 ? cn_stride() * ((n() - 1) / nr()) + (n() - 1) % nr() + 1 : this->cm_stride_;
  }

  inline GemmMicrokernelTester& cn_stride(size_t cn_stride) {
    this->cn_stride_ = cn_stride;
    return *this;
  }

  inline size_t cn_stride() const {
    return this->cn_stride_ == 0 ? nr() : this->cn_stride_;
  }

  inline GemmMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  inline uint8_t a_zero_point() const {
    return this->a_zero_point_;
  }

  inline GemmMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  inline uint8_t b_zero_point() const {
    return this->b_zero_point_;
  }

  inline GemmMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GemmMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GemmMicrokernelTester& a_offset(size_t a_offset) {
    this->a_offset_ = a_offset;
    return *this;
  }

  inline size_t a_offset() const {
    return this->a_offset_;
  }

  inline GemmMicrokernelTester& zero_index(size_t zero_index) {
    this->zero_index_ = zero_index;
    return *this;
  }

  inline size_t zero_index() const {
    return this->zero_index_;
  }

  inline GemmMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_q8_gemm_ukernel_function gemm, Variant variant = Variant::Native) const {
    ASSERT_LE(m(), mr());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> b(n() * k());
    std::vector<int32_t> bias(n());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_w(packed_n() * packed_k() + bias_n() * sizeof(uint32_t) / sizeof(uint8_t));
    std::vector<uint8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<int32_t> acc(m() * n());
    std::vector<uint8_t> c_ref(m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(a.begin(), a.end(), std::ref(u8rng));
      } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
      do {
        std::generate(b.begin(), b.end(), std::ref(u8rng));
      } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0xA5);

      std::fill(packed_w.begin(), packed_w.end(), b_zero_point());
      xnn_pack_q8_gemm_goi_w(1, n(), k(), nr(), kr(),
        a_zero_point(), b_zero_point(),
        b.data(), bias.data(), packed_w.data());

      // Compute 32-bit results and output quantization arguments.
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            acc[m_index * n() + n_index] +=
                (int32_t(a[m_index * a_stride() + k_index]) - int32_t(a_zero_point())) *
                (int32_t(b[n_index * k() + k_index]) - int32_t(b_zero_point()));
          }
          acc[m_index * n() + n_index] += bias[n_index];
        }
      }

      const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
      const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
      const double c_scale = uint32_t(accumulated_max - accumulated_min) >= 256 ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0 : 1.00001;
      const uint8_t c_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / c_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      const float requantization_scale = 1.0f / float(c_scale);
      union xnn_q8_gemm_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_gemm_params(
            a_zero_point(), b_zero_point(),
            requantization_scale, c_zero_point, qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_gemm_params(
            a_zero_point(), b_zero_point(),
            requantization_scale, c_zero_point, qmin(), qmax());
          break;
      }
      const union xnn_q31_requantization_params scalar_requantization_params =
        xnn_init_scalar_requantization_params(
          requantization_scale, c_zero_point, qmin(), qmax());

      gemm(
        m(), n(), k(),
        a.data(), a_stride() * sizeof(uint8_t),
        packed_w.data(),
        c.data(), cm_stride() * sizeof(uint8_t), cn_stride() * sizeof(uint8_t),
        &quantization_params);

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          c_ref[m_index * n() + n_index] = xnn_q31_requantize(acc[m_index * n() + n_index], scalar_requantization_params);
        }
      }

      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_LE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmin()));
          ASSERT_EQ(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(c_ref[i * n() + j]))
              << "at " << i << ", " << j << ": reference = " << (uint32_t) c_ref[i * n() + j]
              << " (accumulator = " << acc[i * n() + j]
              << "), optimized = " << (uint32_t) c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
              << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
              << ", requantization scale = " << requantization_scale << ", output zero point = " << int32_t(c_zero_point);
        }
      }
    }
  }

  void Test(xnn_q8_igemm_ukernel_function igemm, Variant variant = Variant::Native) const {
    ASSERT_LE(m(), mr());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> b(n() * ks() * k());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_w(ks() * packed_n() * packed_k() + bias_n() * sizeof(uint32_t) / sizeof(uint8_t));
    std::vector<int32_t> bias(n());
    std::vector<uint8_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<int32_t> acc(m() * n());
    std::vector<uint8_t> c_ref(m() * n());
    std::vector<uint8_t> junk(k() + 8);
    std::vector<const uint8_t*> im2col(mr() * ks());

    std::fill(junk.begin(), junk.end(), 0xA5);

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(a.begin(), a.end(), std::ref(u8rng));
      } while (a.size() > 1 && *std::max_element(a.cbegin(), a.cend()) == *std::min_element(a.cbegin(), a.cend()));
      do {
        std::generate(b.begin(), b.end(), std::ref(u8rng));
      } while (b.size() > 1 && *std::max_element(b.cbegin(), b.cend()) == *std::min_element(b.cbegin(), b.cend()));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0xA5);

      std::fill(packed_w.begin(), packed_w.end(), b_zero_point());
      xnn_pack_q8_conv_goki_w(
        1, n(), ks(), k(), nr(), kr(),
        a_zero_point(), b_zero_point(),
        b.data(), bias.data(), packed_w.data());

      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        for (size_t m_index = 0; m_index < mr(); m_index++) {
          im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
        }

      }
      std::shuffle(im2col.begin(), im2col.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          im2col[ks_index * mr() + zero_index()] = a.data();
        }
      }
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        for (size_t m_index = m(); m_index < mr(); m_index++) {
          im2col[ks_index * mr() + m_index] = junk.data();
        }
      }

      // Compute 32-bit results and output quantization arguments.
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
            for (size_t k_index = 0; k_index < k(); k_index++) {
              if (im2col[ks_index * mr() + m_index] == a.data()) {
                acc[m_index * n() + n_index] +=
                  (int32_t(im2col[ks_index * mr() + m_index][k_index]) - int32_t(a_zero_point())) *
                  (int32_t(b[(n_index * ks() + ks_index) * k() + k_index]) - int32_t(b_zero_point()));
              } else {
                acc[m_index * n() + n_index] +=
                  (int32_t(im2col[ks_index * mr() + m_index][k_index + a_offset()]) - int32_t(a_zero_point())) *
                  (int32_t(b[(n_index * ks() + ks_index) * k() + k_index]) - int32_t(b_zero_point()));
              }
            }
          }
          acc[m_index * n() + n_index] += bias[n_index];
        }
      }

      const int32_t accumulated_min = *std::min_element(acc.cbegin(), acc.cend());
      const int32_t accumulated_max = *std::max_element(acc.cbegin(), acc.cend());
      const double c_scale = uint32_t(accumulated_max - accumulated_min) >= 256 ? double(uint32_t(accumulated_max - accumulated_min)) / 255.0 : 1.00001;
      const uint8_t c_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / c_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      const float requantization_scale = 1.0f / float(c_scale);
      union xnn_q8_gemm_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_gemm_params(
            a_zero_point(), b_zero_point(),
            requantization_scale, c_zero_point, qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_gemm_params(
            a_zero_point(), b_zero_point(),
            requantization_scale, c_zero_point, qmin(), qmax());
          break;
      }
      const union xnn_q31_requantization_params scalar_requantization_params =
        xnn_init_scalar_requantization_params(
          requantization_scale, c_zero_point, qmin(), qmax());

      const uint8_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

      igemm(
        m(), n(), k(), ks() * mr() * sizeof(void*),
        im2col.data(), packed_w.data(),
        c.data(), cm_stride() * sizeof(uint8_t), cn_stride() * sizeof(uint8_t),
        a_offset() * sizeof(uint8_t), zero_pointer,
        &quantization_params);

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          c_ref[m_index * n() + n_index] = xnn_q31_requantize(acc[m_index * n() + n_index], scalar_requantization_params);
        }
      }

      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_LE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(qmin()));
          ASSERT_EQ(uint32_t(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), uint32_t(c_ref[i * n() + j]))
              << "at " << i << ", " << j << ": reference = " << uint32_t(c_ref[i * n() + j])
              << " (accumulator = " << acc[i * n() + j]
              << "), optimized = " << (uint32_t) c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x "
              << nr() << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k()
              << ", requantization scale = " << requantization_scale << ", output zero point = " << int32_t(c_zero_point);
        }
      }
    }
  }

  void Test(xnn_f16_gemm_minmax_ukernel_function gemm_minmax, Variant variant = Variant::Native) const
  {
    ASSERT_LE(m(), mr());
    ASSERT_GE(a_stride(), k());
    ASSERT_GE(cm_stride(), n());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> b(n() * k());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_w(packed_n() * packed_k() + bias_n());
    std::vector<uint16_t> bias(n());
    std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f16rng));
      std::generate(b.begin(), b.end(), std::ref(f16rng));
      std::generate(bias.begin(), bias.end(), std::ref(f16rng));
      std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);

      std::fill(packed_w.begin(), packed_w.end(), 0);
      xnn_pack_f16_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data());

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LE(n(), packed_n());
            ASSERT_LT(m_index * n() + n_index, c_ref.size());
            ASSERT_LT(m_index * k() + k_index, a.size());
            c_ref[m_index * n() + n_index] +=
              fp16_ieee_to_fp32_value(a[m_index * a_stride() + k_index]) *
              fp16_ieee_to_fp32_value(b[n_index * k() + k_index]);
          }
          c_ref[m_index * n() + n_index] += fp16_ieee_to_fp32_value(bias[n_index]);
        }
      }

      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin())));
      const float c_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax())));

      // Prepare minmax parameters.
      xnn_f16_scaleminmax_params params;
      params = xnn_init_f16_scaleminmax_params(
        UINT16_C(0x3C00) /* 1.0 */,
        fp16_ieee_from_fp32_value(c_min),
        fp16_ieee_from_fp32_value(c_max));

      for (float& c_value : c_ref) {
        c_value = std::max(std::min(c_value, c_max), c_min);
      }

      gemm_minmax(m(), n(), k() * sizeof(uint16_t),
        a.data(), a_stride() * sizeof(uint16_t),
        packed_w.data(),
        c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t),
        &params);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_NEAR(
              fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]),
              c_ref[i * n() + j],
              std::abs(c_ref[i * n() + j]) * 1.0e-2f)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void Test(xnn_f16_igemm_minmax_ukernel_function igemm_minmax, Variant variant = Variant::Native) const {
    ASSERT_LE(m(), mr());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> b(n() * ks() * k());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_w(ks() * packed_k() * packed_n() + bias_n());
    std::vector<uint16_t> bias(n());
    std::vector<uint16_t> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());
    std::vector<uint16_t> junk(k() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<const uint16_t*> im2col(mr() * ks());
    std::fill(junk.begin(), junk.end(), UINT16_C(0x7E00) /* NaN */);

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f16rng));
      std::generate(b.begin(), b.end(), std::ref(f16rng));
      std::generate(bias.begin(), bias.end(), std::ref(f16rng));
      std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);
      std::fill(c_ref.begin(), c_ref.end(), 0);

      std::fill(packed_w.begin(), packed_w.end(), 0);
      xnn_pack_f16_conv_goki_w(
        1, n(), ks(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_w.data());

      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        for (size_t m_index = 0; m_index < mr(); m_index++) {
          im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
        }
      }
      std::shuffle(im2col.begin(), im2col.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          im2col[ks_index * mr() + zero_index()] = a.data();
        }
      }
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        for (size_t m_index = m(); m_index < mr(); m_index++) {
          im2col[ks_index * mr() + m_index] = junk.data();
        }
      }

      std::fill(c_ref.begin(), c_ref.end(), 0.0);
      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
            for (size_t k_index = 0; k_index < k(); k_index++) {
              ASSERT_LT(ks_index * mr() + m_index, im2col.size());
              ASSERT_LT(k_index, k());
              ASSERT_LT(k_index, a_stride());
              if (im2col[ks_index * mr() + m_index] == a.data()) {
                c_ref[m_index * n() + n_index] +=
                  fp16_ieee_to_fp32_value(im2col[ks_index * mr() + m_index][k_index]) *
                  fp16_ieee_to_fp32_value(b[(n_index * ks() + ks_index) * k() + k_index]);
              } else {
                c_ref[m_index * n() + n_index] +=
                  fp16_ieee_to_fp32_value(im2col[ks_index * mr() + m_index][k_index + a_offset()]) *
                  fp16_ieee_to_fp32_value(b[(n_index * ks() + ks_index) * k() + k_index]);
              }
            }
          }
          c_ref[m_index * n() + n_index] += fp16_ieee_to_fp32_value(bias[n_index]);
        }
      }

      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + (accumulated_max - accumulated_min) / 255.0f * uint16_t(qmin())));
      const float c_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - (accumulated_max - accumulated_min) / 255.0f * uint16_t(255 - qmax())));
      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          c_ref[m_index * n() + n_index] = std::min(c_ref[m_index * n() + n_index], c_max);
          c_ref[m_index * n() + n_index] = std::max(c_ref[m_index * n() + n_index], c_min);
        }
      }

      // Prepare minmax parameters.
      xnn_f16_scaleminmax_params params;
      params = xnn_init_f16_scaleminmax_params(
        UINT16_C(0x3C00) /* 1.0 */,
        fp16_ieee_from_fp32_value(c_min),
        fp16_ieee_from_fp32_value(c_max));

      for (float& c_value : c_ref) {
        c_value = std::max(std::min(c_value, c_max), c_min);
      }

      const uint16_t* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

      igemm_minmax(
        m(), n(), k() * sizeof(uint16_t), ks() * mr() * sizeof(void*),
        reinterpret_cast<const void**>(im2col.data()), packed_w.data(),
        c.data(), cm_stride() * sizeof(uint16_t), cn_stride() * sizeof(uint16_t),
        a_offset() * sizeof(uint16_t), zero_pointer,
        &params);

      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_LE(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_max)
              << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
          ASSERT_GE(fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]), c_min)
              << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
          ASSERT_NEAR(
              fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]),
              c_ref[i * n() + j],
              std::abs(c_ref[i * n() + j]) * 1.0e-1f)
              << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << fp16_ieee_to_fp32_value(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()]) << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        }
      }
    }
  }

  void Test(xnn_f32_ppmm_minmax_ukernel_function ppmm, Variant variant = Variant::Native) const {
    ASSERT_LE(m(), mr());
    ASSERT_GE(cm_stride(), n());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> a(packed_k() * mr());
    std::vector<float> b(n() * k());
    std::vector<float> bias(n());
    std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k() + bias_n());
    std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);

      std::fill(packed_w.begin(), packed_w.end(), 0.0f);
      xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data());

      for (size_t i = m(); i < mr(); i++) {
        for (size_t l = 0; l < k(); l++) {
          a[l * mr() + i] = a[l * mr() + m() - 1];
        }
      }

      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          for (size_t l = 0; l < k(); l++) {
            c_ref[i * n() + j] +=
              a[l * mr() + i] *
              b[j * k() + l];
          }
          c_ref[i * n() + j] += bias[j];
        }
      }

      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Prepare output parameters.
      xnn_f32_minmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_minmax_params(c_min, c_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_minmax_params(c_min, c_max);
          break;
      }

      for (float& c_value : c_ref) {
        c_value = std::max(std::min(c_value, c_max), c_min);
      }

      ppmm(m(), n(), k() * sizeof(float),
        a.data(), packed_w.data(),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
        &params);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_NEAR(
              c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
              c_ref[i * n() + j],
              std::abs(c_ref[i * n() + j]) * 1.0e-6f)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void Test(xnn_f32_gemm_ukernel_function gemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_GE(a_stride(), k());
    ASSERT_GE(cm_stride(), n());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(n() * k());
    std::vector<float> bias(n());
    std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k() + bias_n());
    std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);

      std::fill(packed_w.begin(), packed_w.end(), 0.0f);
      xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data());

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LE(n(), packed_n());
            ASSERT_LT(m_index * n() + n_index, c_ref.size());
            c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] *
              b[n_index * k() + k_index];
          }
          c_ref[m_index * n() + n_index] += bias[n_index];
        }
      }

      gemm(m(), n(), k() * sizeof(float),
        a.data(), a_stride() * sizeof(float),
        packed_w.data(),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
        nullptr);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_NEAR(
              c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
              c_ref[i * n() + j],
              std::abs(c_ref[i * n() + j]) * 1.0e-6f)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void Test(xnn_f32_gemm_minmax_ukernel_function gemm_minmax, Variant variant = Variant::Native) const {
    ASSERT_LE(m(), mr());
    ASSERT_GE(a_stride(), k());
    ASSERT_GE(cm_stride(), n());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(n() * k());
    std::vector<float> bias(n());
    std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k() + bias_n());
    std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);

      std::fill(packed_w.begin(), packed_w.end(), 0.0f);
      xnn_pack_f32_gemm_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), bias.data(), packed_w.data());

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LE(n(), packed_n());
            ASSERT_LT(m_index * n() + n_index, c_ref.size());
            c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] *
              b[n_index * k() + k_index];
          }
          c_ref[m_index * n() + n_index] += bias[n_index];
        }
      }

      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Prepare minmax parameters.
      xnn_f32_minmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_minmax_params(c_min, c_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_minmax_params(c_min, c_max);
          break;
      }

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
        }
      }

      gemm_minmax(m(), n(), k() * sizeof(float),
        a.data(), a_stride() * sizeof(float),
        packed_w.data(),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
        &params);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
          ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
          ASSERT_NEAR(
              c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
              c_ref[i * n() + j],
              std::abs(c_ref[i * n() + j]) * 1.0e-6f)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void Test(xnn_f32_gemminc_minmax_ukernel_function gemminc, Variant variant = Variant::Native) const {
    ASSERT_LE(m(), mr());
    ASSERT_GE(a_stride(), k());
    ASSERT_GE(cm_stride(), n());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> a((m() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(n() * k());
    std::vector<float> bias(n());
    std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_n() * packed_k());  // no bias_n()
    std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());
    std::vector<float, AlignedAllocator<float, 64>> acc(mr() * packed_n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);
      std::generate(acc.begin(), acc.end(), std::ref(f32rng));

      std::fill(packed_w.begin(), packed_w.end(), 0.0f);
      xnn_pack_f32_gemminc_goi_w(1, n(), k(), nr(), kr(), sr(), b.data(), packed_w.data());

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t k_index = 0; k_index < k(); k_index++) {
            ASSERT_LE(n(), packed_n());
            ASSERT_LT(m_index * n() + n_index, c_ref.size());
            c_ref[m_index * n() + n_index] +=
              a[m_index * a_stride() + k_index] *
              b[n_index * k() + k_index];
          }
          c_ref[m_index * n() + n_index] += acc[n_index / nr() * nr() * mr() + m_index % mr() * nr() + n_index % nr()];
        }
      }

      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Prepare output parameters.
      xnn_f32_minmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_minmax_params(c_min, c_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_minmax_params(c_min, c_max);
          break;
      }

      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          c_ref[m_index * n() + n_index] = std::max(std::min(c_ref[m_index * n() + n_index], c_max), c_min);
        }
      }

      gemminc(m(), n(), k() * sizeof(float),
        a.data(), a_stride() * sizeof(float),
        packed_w.data(),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
        acc.data(),
        &params);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
          ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
          ASSERT_NEAR(
              c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
              c_ref[i * n() + j],
              std::abs(c_ref[i * n() + j]) * 1.0e-6f)
              << "at " << i << ", " << j << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void Test(xnn_f32_igemm_ukernel_function igemm) const {
    ASSERT_LE(m(), mr());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(n() * ks() * k());
    std::vector<float, AlignedAllocator<float, 64>> packed_w(ks() * packed_k() * packed_n() + bias_n());
    std::vector<float> bias(n());
    std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());
    std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<const float*> im2col(mr() * ks());
    std::fill(junk.begin(), junk.end(), nanf(""));

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);

      std::fill(packed_w.begin(), packed_w.end(), 0.0f);
      xnn_pack_f32_conv_goki_w(
        1, n(), ks(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_w.data());

      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        for (size_t m_index = 0; m_index < mr(); m_index++) {
          im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
        }
      }
      std::shuffle(im2col.begin(), im2col.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          im2col[ks_index * mr() + zero_index()] = a.data();
        }
      }
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        for (size_t m_index = m(); m_index < mr(); m_index++) {
          im2col[ks_index * mr() + m_index] = junk.data();
        }
      }

      std::fill(c_ref.begin(), c_ref.end(), 0.0);
      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
            for (size_t k_index = 0; k_index < k(); k_index++) {
              ASSERT_LT(ks_index * mr() + m_index, im2col.size());
              ASSERT_LT(k_index, k());
              ASSERT_LT(k_index, a_stride());
              if (im2col[ks_index * mr() + m_index] == a.data()) {
                c_ref[m_index * n() + n_index] +=
                  (im2col[ks_index * mr() + m_index][k_index]) *
                  (b[(n_index * ks() + ks_index) * k() + k_index]);
              } else {
                c_ref[m_index * n() + n_index] +=
                  (im2col[ks_index * mr() + m_index][k_index + a_offset()]) *
                  (b[(n_index * ks() + ks_index) * k() + k_index]);
              }
            }
          }
          c_ref[m_index * n() + n_index] += bias[n_index];
        }
      }

      const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

      igemm(
        m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
        im2col.data(), packed_w.data(),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
        a_offset() * sizeof(float), zero_pointer,
        nullptr);

      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_NEAR(
              c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
              c_ref[i * n() + j],
              std::abs(c_ref[i * n() + j]) * 1.0e-6f)
              << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
        }
      }
    }
  }

  void Test(xnn_f32_igemm_minmax_ukernel_function igemm_minmax, Variant variant = Variant::Native) const {
    ASSERT_LE(m(), mr());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> a((mr() - 1) * a_stride() + k() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(n() * ks() * k());
    std::vector<float, AlignedAllocator<float, 64>> packed_w(ks() * packed_k() * packed_n() + bias_n());
    std::vector<float> bias(n());
    std::vector<float> c((mr() - 1) * cm_stride() + ((n() - 1) / nr()) * cn_stride() + (n() - 1) % nr() + 1);
    std::vector<float> c_ref(m() * n());
    std::vector<float> junk(k() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<const float*> im2col(mr() * ks());
    std::fill(junk.begin(), junk.end(), nanf(""));

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);

      std::fill(packed_w.begin(), packed_w.end(), 0.0f);
      xnn_pack_f32_conv_goki_w(
        1, n(), ks(), k(), nr(), kr(), sr(),
        b.data(), bias.data(), packed_w.data());

      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        for (size_t m_index = 0; m_index < mr(); m_index++) {
          im2col[ks_index * mr() + m_index] = a.data() + a_stride() * m_index - a_offset();
        }
      }
      std::shuffle(im2col.begin(), im2col.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
          im2col[ks_index * mr() + zero_index()] = a.data();
        }
      }
      for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
        for (size_t m_index = m(); m_index < mr(); m_index++) {
          im2col[ks_index * mr() + m_index] = junk.data();
        }
      }

      std::fill(c_ref.begin(), c_ref.end(), 0.0);
      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          for (size_t ks_index = 0; ks_index < ks(); ks_index++) {
            for (size_t k_index = 0; k_index < k(); k_index++) {
              ASSERT_LT(ks_index * mr() + m_index, im2col.size());
              ASSERT_LT(k_index, k());
              ASSERT_LT(k_index, a_stride());
              if (im2col[ks_index * mr() + m_index] == a.data()) {
                c_ref[m_index * n() + n_index] +=
                  (im2col[ks_index * mr() + m_index][k_index]) *
                  (b[(n_index * ks() + ks_index) * k() + k_index]);
              } else {
                c_ref[m_index * n() + n_index] +=
                  (im2col[ks_index * mr() + m_index][k_index + a_offset()]) *
                  (b[(n_index * ks() + ks_index) * k() + k_index]);
              }
            }
          }
          c_ref[m_index * n() + n_index] += bias[n_index];
        }
      }

      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());
      for (size_t m_index = 0; m_index < m(); m_index++) {
        for (size_t n_index = 0; n_index < n(); n_index++) {
          c_ref[m_index * n() + n_index] = std::min(c_ref[m_index * n() + n_index], c_max);
          c_ref[m_index * n() + n_index] = std::max(c_ref[m_index * n() + n_index], c_min);
        }
      }

      // Prepare output parameters.
      xnn_f32_minmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_minmax_params(c_min, c_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_minmax_params(c_min, c_max);
          break;
      }

      const float* zero_pointer = (zero_index() != SIZE_MAX) ? a.data() : NULL;

      igemm_minmax(
        m(), n(), k() * sizeof(float), ks() * mr() * sizeof(void*),
        im2col.data(), packed_w.data(),
        c.data(), cm_stride() * sizeof(float), cn_stride() * sizeof(float),
        a_offset() * sizeof(float), zero_pointer,
        &params);

      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_LE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_max)
              << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
          ASSERT_GE(c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()], c_min)
              << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
          ASSERT_NEAR(
              c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()],
              c_ref[i * n() + j],
              std::abs(c_ref[i * n() + j]) * 1.0e-6f)
              << "at " << i << ", " << i << ": reference = " << c_ref[i * n() + j]
              << ", optimized = " << c[i * cm_stride() + (j / nr()) * cn_stride() + j % nr()] << ", Mr x Nr x Kr = " << mr() << " x " << nr()
              << " x " << kr() << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k() << " x " << ks();
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
  size_t iterations_{15};
};
