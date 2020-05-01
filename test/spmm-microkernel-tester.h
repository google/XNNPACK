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
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static inline bool is_fp16_zero(uint16_t x) {
  const uint32_t ext_x = x;
  const uint32_t two_x = ext_x + ext_x;
  return (uint16_t) two_x == 0;
}

class SpMMMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline SpMMMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline SpMMMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline SpMMMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline SpMMMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline SpMMMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline SpMMMicrokernelTester& sparsity(float sparsity) {
    this->sparsity_ = sparsity;
    return *this;
  }

  inline float sparsity() const {
    return this->sparsity_;
  }

  inline SpMMMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline SpMMMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline SpMMMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_spmm_minmax_ukernel_function spmm, Variant variant = Variant::Native) const {
    ASSERT_GE(m(), 1);
    ASSERT_GE(n(), 1);
    ASSERT_GE(k(), 1);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);
    auto prng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float, AlignedAllocator<float, 64>> a(k() * m());
    // Think of b as (n/nr + n % nr) x k, expansion happens later.
    const size_t ncols = n() / nr() + n() % nr();
    std::vector<float> b(ncols * k());
    std::vector<float> bias(n());
    // Number of non-zero weights per N (output channel).
    std::vector<uint32_t> nmap(n());
    // Mapping from index of non-zero weight to increment of K (input channel) following this index.
    std::vector<int32_t> dmap(n() * k());
    std::vector<float> w(n() * k() + n());
    std::vector<float> c(n() * m());
    std::vector<float> c_ref(n() * m());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);
      std::fill(nmap.begin(), nmap.end(), 0);
      std::fill(dmap.begin(), dmap.end(), 0);
      std::fill(w.begin(), w.end(), 0.0f);

      for (float& b_value : b) {
        if (prng() <= sparsity()) {
          b_value = 0.0f;
        }
      }

      uint32_t nnz = 0;
      uint32_t wcnt = 0;
      size_t last_kk = 0;
      bool first_nzz = true;
      size_t first_kk = 0;
      for (size_t nn = 0; nn < n() / nr(); nn++) {
        for (size_t i = 0; i < nr(); ++i)
          w[wcnt++] = bias[nr() * nn + i];
        for (size_t kk = 0; kk < k(); kk++) {
          if (b[nn * k() + kk] != 0.0f) {
            // Every non-zero actually corresponds to nr adjacent non-zeros.
            for (size_t i = 0; i < nr(); ++i)
              w[wcnt++] = b[nn * k() + kk] + static_cast<float>(i);
            // Skip the very first non-zero weight as we record only the difference.
            if (first_nzz) {
              first_kk = kk;
            } else {
              const int32_t increment = int32_t(kk - last_kk) * int32_t(m() * sizeof(float));
              dmap[nnz++] = increment;
            }
            last_kk = kk;
            first_nzz = false;
            nmap[nn] += 1;
          }
        }
      }

      // now we've constructed the matrix for the blocked part and switch to the
      // leftovers, which we do as nr=1 always.
      for (size_t nn = n() / nr(); nn < ncols; nn++) {
        w[wcnt++] = bias[(n() / nr()) * nr() + (nn - n() / nr())];
        for (size_t kk = 0; kk < k(); kk++) {
          if (b[nn * k() + kk] != 0.0f) {
            // Every non-zero actually corresponds to nr adjacent non-zeros.
            w[wcnt++] = b[nn * k() + kk];
            // Skip the very first non-zero weight as we record only the difference.
            if (first_nzz) {
              first_kk = kk;
            } else {
              const int32_t increment = int32_t(kk - last_kk) * int32_t(m() * sizeof(float));
              dmap[nnz++] = increment;
            }
            last_kk = kk;
            first_nzz = false;
            nmap[nn] += 1;
          }
        }
      }
      // In the end, we must return input pointer to the initial value.
      const int64_t increment = int32_t(first_kk - last_kk) * int32_t(m() * sizeof(float));
      dmap[nnz++] = increment;

      // Generate expanded b which will be used in reference calculation.
      // Everywhere there is a non-zero in the original we copy it and add an
      // adjacent non-zero with incremented weight value.
      std::vector<float> b_full(n() * k());
      if (nr() == 1) {
         b_full = b;
      }
      else {
        for (size_t nn = 0; nn < n() / nr(); nn++) {
          for (size_t kk = 0; kk < k(); kk++) {
            if (b[nn * k() + kk] != 0.0f) {
              for (size_t i = 0; i < nr(); ++i)
                b_full[nr() * nn * k() + i * k() + kk] = b[nn * k() + kk] + static_cast<float>(i);
            }
          }
        }
        for (size_t nn = n() / nr(); nn < ncols; nn++) {
          for (size_t kk = 0; kk < k(); kk++) {
            if (b[nn * k() + kk] != 0.0f) {
              b_full[nr() * (n() / nr()) * k() + (nn - n() / nr()) * k() + kk] = b[nn * k() + kk];
            }
          }
        }
      }

      for (size_t oc = 0; oc < n(); oc++) {
        for (size_t pxb = 0; pxb < m(); pxb++) {
          c_ref[oc * m() + pxb] = bias[oc];
          for (size_t ic = 0; ic < k(); ic++) {
            c_ref[oc * m() + pxb] += a[ic * m() + pxb] * b_full[oc * k() + ic];
          }
        }
      }

      // Micro-kernel can access one element beyond w and dmap for software pipelining.
      w.resize(wcnt + 1);
      dmap.resize(nnz + 1);

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& c_value : c_ref) {
        c_value = std::min(std::max(c_value, c_min), c_max);
      }

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

      spmm(m(), n(),
        a.data() + first_kk * m(), w.data(), dmap.data(), nmap.data(), c.data(),
        &params);

      // Validate micro-kernel outputs.
      for (size_t pxb = 0; pxb < n(); pxb++) {
        for (size_t oc = 0; oc < m(); oc++) {
          ASSERT_NEAR(
            c[pxb * m() + oc],
            c_ref[pxb * m() + oc],
            std::abs(c_ref[pxb * m() + oc]) * 1.0e-6f)
            << "at " << pxb << ", " << oc
            << ": Mr x Nr x Kr = " << mr() << " x " << nr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void Test(xnn_f16_spmm_minmax_ukernel_function spmm) const {
    ASSERT_GE(m(), 1);
    ASSERT_GE(n(), 1);
    ASSERT_GE(k(), 1);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);
    auto prng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> a(k() * m());
    // Think of b as (n/nr + n % nr) x k, expansion happens later.
    const size_t ncols = n() / nr() + n() % nr();
    std::vector<uint16_t> b(ncols * k());
    std::vector<uint16_t> bias(n());
    // Number of non-zero weights per N (output channel).
    std::vector<uint32_t> nmap(n());
    // Mapping from index of non-zero weight to increment of K (input channel) following this index.
    std::vector<int32_t> dmap(n() * k());
    std::vector<uint16_t> w(n() * k() + n());
    std::vector<uint16_t> c(n() * m());
    std::vector<float> c_ref(n() * m());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f16rng));
      std::generate(b.begin(), b.end(), std::ref(f16rng));
      std::generate(bias.begin(), bias.end(), std::ref(f16rng));
      std::fill(c.begin(), c.end(), 0xC000);
      std::fill(c_ref.begin(), c_ref.end(), 0.0f);
      std::fill(nmap.begin(), nmap.end(), 0);
      std::fill(dmap.begin(), dmap.end(), 0);
      std::fill(w.begin(), w.end(), 0);

      for (uint16_t& b_value : b) {
        if (prng() <= sparsity()) {
          b_value = 0;
        }
      }

      uint32_t nnz = 0;
      uint32_t wcnt = 0;
      size_t last_kk = 0;
      bool first_nzz = true;
      size_t first_kk = 0;
      for (size_t nn = 0; nn < n() / nr(); nn++) {
        for (size_t i = 0; i < nr(); ++i)
          w[wcnt++] = bias[nr() * nn + i];
        for (size_t kk = 0; kk < k(); kk++) {
          if (!is_fp16_zero(b[nn * k() + kk])) {
            // Every non-zero actually corresponds to nr adjacent non-zeros.
            for (size_t i = 0; i < nr(); ++i)
              w[wcnt++] = fp16_ieee_from_fp32_value(fp16_ieee_to_fp32_value(b[nn * k() + kk]) + static_cast<float>(i));
            // Skip the very first non-zero weight as we record only the difference.
            if (first_nzz) {
              first_kk = kk;
            } else {
              const int32_t increment = int32_t(kk - last_kk) * int32_t(m() * sizeof(uint16_t));
              dmap[nnz++] = increment;
            }
            last_kk = kk;
            first_nzz = false;
            nmap[nn] += 1;
          }
        }
      }

      // now we've constructed the matrix for the blocked part and switch to the
      // leftovers, which we do as nr=1 always.
      for (size_t nn = n() / nr(); nn < ncols; nn++) {
        w[wcnt++] = bias[(n() / nr()) * nr() + (nn - n() / nr())];
        for (size_t kk = 0; kk < k(); kk++) {
          if (!is_fp16_zero(b[nn * k() + kk])) {
            // Every non-zero actually corresponds to nr adjacent non-zeros.
            w[wcnt++] = b[nn * k() + kk];
            // Skip the very first non-zero weight as we record only the difference.
            if (first_nzz) {
              first_kk = kk;
            } else {
              const int32_t increment = int32_t(kk - last_kk) * int32_t(m() * sizeof(uint16_t));
              dmap[nnz++] = increment;
            }
            last_kk = kk;
            first_nzz = false;
            nmap[nn] += 1;
          }
        }
      }
      // In the end, we must return input pointer to the initial value.
      const int64_t increment = int32_t(first_kk - last_kk) * int32_t(m() * sizeof(uint16_t));
      dmap[nnz++] = increment;

      // Generate expanded b which will be used in reference calculation.
      // Everywhere there is a non-zero in the original we copy it and add an
      // adjacent non-zero with incremented weight value.
      std::vector<uint16_t> b_full(n() * k());
      if (nr() == 1) {
         b_full = b;
      }
      else {
        for (size_t nn = 0; nn < n() / nr(); nn++) {
          for (size_t kk = 0; kk < k(); kk++) {
            if (b[nn * k() + kk] != 0.0f) {
              for (size_t i = 0; i < nr(); ++i)
                b_full[nr() * nn * k() + i * k() + kk] = fp16_ieee_from_fp32_value(
                  fp16_ieee_to_fp32_value(b[nn * k() + kk]) + static_cast<float>(i));
            }
          }
        }
        for (size_t nn = n() / nr(); nn < ncols; nn++) {
          for (size_t kk = 0; kk < k(); kk++) {
            if (b[nn * k() + kk] != 0.0f) {
              b_full[nr() * (n() / nr()) * k() + (nn - n() / nr()) * k() + kk] = b[nn * k() + kk];
            }
          }
        }
      }

      for (size_t oc = 0; oc < n(); oc++) {
        for (size_t pxb = 0; pxb < m(); pxb++) {
          c_ref[oc * m() + pxb] = fp16_ieee_to_fp32_value(bias[oc]);
          for (size_t ic = 0; ic < k(); ic++) {
            c_ref[oc * m() + pxb] += fp16_ieee_to_fp32_value(a[ic * m() + pxb]) * fp16_ieee_to_fp32_value(b_full[oc * k() + ic]);
          }
        }
      }

      // Micro-kernel can access one element beyond w and dmap for software pipelining.
      w.resize(wcnt + 1);
      dmap.resize(nnz + 1);

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(c_ref.cbegin(), c_ref.cend());
      const float accumulated_max = *std::max_element(c_ref.cbegin(), c_ref.cend());
      const float c_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float c_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& c_value : c_ref) {
        c_value = std::min(std::max(c_value, c_min), c_max);
      }

      // Prepare scaleminmax parameters.
      xnn_f16_scaleminmax_params params;
      params.scale = UINT16_C(0x3C00) /* 1.0 */;
      params.max = fp16_ieee_from_fp32_value(c_max);
      params.min = fp16_ieee_from_fp32_value(c_min);

      spmm(m(), n(),
        a.data() + first_kk * m(), w.data(), dmap.data(), nmap.data(), c.data(),
        &params);

      // Validate micro-kernel outputs.
      for (size_t pxb = 0; pxb < n(); pxb++) {
        for (size_t oc = 0; oc < m(); oc++) {
          ASSERT_NEAR(
            fp16_ieee_to_fp32_value(c[pxb * m() + oc]),
            c_ref[pxb * m() + oc],
            std::abs(c_ref[pxb * m() + oc]) * 1.0e-2f)
            << "at " << pxb << ", " << oc
            << ": Mr x Nr x Kr = " << mr() << " x " << nr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  float sparsity_{0.5f};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
