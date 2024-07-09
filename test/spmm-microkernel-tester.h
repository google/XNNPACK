// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

static inline bool is_fp16_zero(uint16_t x) {
  const uint16_t two_x = x + x;
  return two_x == 0;
}

class SpMMMicrokernelTester {
 public:
  SpMMMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  size_t mr() const {
    return this->mr_;
  }

  SpMMMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  size_t nr() const {
    return this->nr_;
  }

  SpMMMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  size_t m() const {
    return this->m_;
  }

  SpMMMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  size_t n() const {
    return this->n_;
  }

  SpMMMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  size_t k() const {
    return this->k_;
  }

  SpMMMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return m();
    } else {
      assert(this->output_stride_ >= m());
      return this->output_stride_;
    }
  }

  SpMMMicrokernelTester& sparsity(float sparsity) {
    this->sparsity_ = sparsity;
    return *this;
  }

  float sparsity() const {
    return this->sparsity_;
  }

  SpMMMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  SpMMMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  SpMMMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_spmm_minmax_ukernel_fn spmm, xnn_init_f32_minmax_params_fn init_params) const {
    ASSERT_GE(m(), 1);
    ASSERT_GE(n(), 1);
    ASSERT_GE(k(), 1);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> pdist;

    std::vector<float, AlignedAllocator<float, 64>> input(k() * m());
    // Think of b as (n/nr + n % nr) x k, expansion happens later.
    const size_t ncols = n() / nr() + n() % nr();
    std::vector<float> b(ncols * k());
    std::vector<float> bias(n());
    // Number of non-zero weights per N (output channel).
    std::vector<uint32_t> nmap(n());
    // Mapping from index of non-zero weight to increment of K (input channel) following this index.
    std::vector<int32_t> dmap(n() * k());
    std::vector<float> w(n() * k() + n());
    std::vector<float> output((n() - 1) * output_stride() + m());
    std::vector<float> output_ref(n() * m());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      std::fill(nmap.begin(), nmap.end(), 0);
      std::fill(dmap.begin(), dmap.end(), 0);
      std::fill(w.begin(), w.end(), 0.0f);

      for (float& b_value : b) {
        if (pdist(rng) <= sparsity()) {
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
      // Everywhere there is input non-zero in the original we copy it and add an
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
          output_ref[oc * m() + pxb] = bias[oc];
          for (size_t ic = 0; ic < k(); ic++) {
            output_ref[oc * m() + pxb] += input[ic * m() + pxb] * b_full[oc * k() + ic];
          }
        }
      }

      // Micro-kernel can access one element beyond w and dmap for software pipelining.
      w.resize(wcnt + 1);
      dmap.resize(nnz + 1);

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float output_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::min(std::max(output_value, output_min), output_max);
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, output_min, output_max);

      spmm(m() * sizeof(float), n(),
        input.data() + first_kk * m(),
        w.data(), dmap.data(), nmap.data(),
        output.data(), output_stride() * sizeof(float),
        &params);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_NEAR(
              output[j * output_stride() + i],
              output_ref[j * m() + i],
              std::abs(output_ref[j * m() + i]) * 1.0e-6f)
            << "at M index " << i << " / " << m() << " (tile " << mr() << ")"
            << ", N index " << j << " / " << n() << " (tile " << nr() << ")"
            << ", K = " << k();
        }
      }
    }
  }

  void Test(xnn_f16_spmm_minmax_ukernel_fn spmm, xnn_init_f16_minmax_params_fn init_params) const {
    ASSERT_GE(m(), 1);
    ASSERT_GE(n(), 1);
    ASSERT_GE(k(), 1);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> pdist;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> input(k() * m());
    // Think of b as (n/nr + n % nr) x k, expansion happens later.
    const size_t ncols = n() / nr() + n() % nr();
    std::vector<uint16_t> b(ncols * k());
    std::vector<uint16_t> bias(n());
    // Number of non-zero weights per N (output channel).
    std::vector<uint32_t> nmap(n());
    // Mapping from index of non-zero weight to increment of K (input channel) following this index.
    std::vector<int32_t> dmap(n() * k());
    std::vector<uint16_t> w(n() * k() + n());
    std::vector<uint16_t> output((n() - 1) * output_stride() + m());
    std::vector<float> output_ref(n() * m());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(b.begin(), b.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      std::fill(nmap.begin(), nmap.end(), 0);
      std::fill(dmap.begin(), dmap.end(), 0);
      std::fill(w.begin(), w.end(), 0);

      for (uint16_t& b_value : b) {
        if (pdist(rng) <= sparsity()) {
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
      // Everywhere there is input non-zero in the original we copy it and add an
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
          output_ref[oc * m() + pxb] = fp16_ieee_to_fp32_value(bias[oc]);
          for (size_t ic = 0; ic < k(); ic++) {
            output_ref[oc * m() + pxb] += fp16_ieee_to_fp32_value(input[ic * m() + pxb]) * fp16_ieee_to_fp32_value(b_full[oc * k() + ic]);
          }
        }
      }

      // Micro-kernel can access one element beyond w and dmap for software pipelining.
      w.resize(wcnt + 1);
      dmap.resize(nnz + 1);

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float output_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::min(std::max(output_value, output_min), output_max);
      }

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params,
        fp16_ieee_from_fp32_value(output_min), fp16_ieee_from_fp32_value(output_max));

      spmm(m() * sizeof(uint16_t), n(),
        input.data() + first_kk * m(),
        w.data(), dmap.data(), nmap.data(),
        output.data(), output_stride() * sizeof(uint16_t),
        &params);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_NEAR(
              fp16_ieee_to_fp32_value(output[j * output_stride() + i]),
              output_ref[j * m() + i],
              std::max(1.0e-4f, std::abs(output_ref[j * m() + i]) * 1.0e-2f))
            << "at M index " << i << " / " << m() << " (tile " << mr() << ")"
            << ", N index " << j << " / " << n() << " (tile " << nr() << ")"
            << ", K = " << k();
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
  size_t output_stride_{0};
  float sparsity_{0.5f};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
