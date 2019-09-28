// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>


class ArgmaxPoolMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline ArgmaxPoolMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline ArgmaxPoolMicrokernelTester& s(size_t s) {
    assert(s != 0);
    this->s_ = s;
    return *this;
  }

  inline size_t s() const {
    return this->s_;
  }

  inline ArgmaxPoolMicrokernelTester& kh(size_t kh) {
    assert(kh != 0);
    this->kh_ = kh;
    return *this;
  }

  inline size_t kh() const {
    return this->kh_;
  }

  inline ArgmaxPoolMicrokernelTester& kw(size_t kw) {
    assert(kw != 0);
    this->kw_ = kw;
    return *this;
  }

  inline size_t kw() const {
    return this->kw_;
  }

  inline size_t ks() const {
    return kh() * kw();
  }

  inline size_t packed_ks() const {
    if (ks() <= mr()) {
      return mr();
    } else {
      return (ks() - mr()) % qr() == 0 ? ks() : ((ks() - mr()) / qr() + 1) * qr() + mr();
    }
  }

  inline ArgmaxPoolMicrokernelTester& mr(size_t mr) {
    assert(mr != 0);
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline ArgmaxPoolMicrokernelTester& qr(size_t qr) {
    assert(qr != 0);
    this->qr_ = qr;
    return *this;
  }

  inline size_t qr() const {
    return this->qr_;
  }

  inline ArgmaxPoolMicrokernelTester& kc(size_t kc) {
    assert(kc != 0);
    this->kc_ = kc;
    return *this;
  }

  inline size_t kc() const {
    return this->kc_;
  }

  inline ArgmaxPoolMicrokernelTester& x_stride(size_t x_stride) {
    assert(x_stride != 0);
    this->x_stride_ = x_stride;
    return *this;
  }

  inline size_t x_stride() const {
    if (this->x_stride_ == 0) {
      return kc();
    } else {
      assert(this->x_stride_ >= kc());
      return this->x_stride_;
    }
  }

  inline ArgmaxPoolMicrokernelTester& y_stride(size_t y_stride) {
    assert(y_stride != 0);
    this->y_stride_ = y_stride;
    return *this;
  }

  inline size_t y_stride() const {
    if (this->y_stride_ == 0) {
      return kc();
    } else {
      assert(this->y_stride_ >= kc());
      return this->y_stride_;
    }
  }

  inline ArgmaxPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline ArgmaxPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ArgmaxPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_argmaxpool_up_ukernel_function argmaxpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<const float*> indirect_x(packed_ks() + (n() * s() - 1) * kh());
    std::vector<float> x((indirect_x.size() - 1) * x_stride() + kc() + XNN_EXTRA_BYTES / sizeof(float));

    std::vector<float> y((n() - 1) * y_stride() + kc());
    std::vector<uint32_t> i(n() * kc());
    std::vector<float> y_ref(n() * kc());
    std::vector<uint32_t> i_ref(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      std::fill(y.begin(), y.end(), nanf(""));

      for (size_t p = 0; p < indirect_x.size(); p++) {
        indirect_x[p] = x.data() + p * x_stride();
      }
      std::shuffle(indirect_x.begin(), indirect_x.end(), rng);

      // Compute reference results, without clamping.
      for (size_t p = 0; p < n(); p++) {
        for (size_t k = 0; k < kc(); k++) {
          float max_value = indirect_x[p * s() * kh()][k];
          uint32_t max_index = 0;
          for (size_t j = 1; j < ks(); j++) {
            const float value = indirect_x[p * s() * kh() + j][k];
            if (value > max_value) {
              max_value = value;
              max_index = j;
            }
          }
          y_ref[p * kc() + k] = max_value;
          i_ref[p * kc() + k] = max_index;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float y_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Prepare output parameters.
      xnn_f32_output_params output_params = { };
      switch (variant) {
        case Variant::Native:
          output_params = xnn_compute_f32_output_params(y_min, y_max);
          break;
        case Variant::Scalar:
          output_params = xnn_compute_scalar_f32_output_params(y_min, y_max);
          break;
      }

      // Clamp reference results.
      for (float& y_value : y_ref) {
        y_value = std::max(std::min(y_value, y_max), y_min);
      }

      // Call optimized micro-kernel.
      argmaxpool(n(), ks(), kc(),
        indirect_x.data(), y.data(), i.data(),
        kh() * s() * sizeof(void*),
        (y_stride() - kc()) * sizeof(float),
        &output_params);

      // Verify results.
      for (size_t p = 0; p < n(); p++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_GE(y[p * y_stride() + k], y_min)
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
          ASSERT_LE(y[p * y_stride() + k], y_max)
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
          ASSERT_EQ(y_ref[p * kc() + k], y[p * y_stride() + k])
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
          ASSERT_EQ(indirect_x[p * s() * kh() + i_ref[p * kc() + k]][k], indirect_x[p * s() * kh() + i[p * kc() + k]][k])
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
          ASSERT_EQ(i_ref[p * kc() + k], i[p * kc() + k])
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
        }
      }
    }
  }

  void Test(xnn_f32_argmaxpool_mp_ukernel_function argmaxpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<const float*> indirect_x(packed_ks() + (n() * s() - 1) * kh());
    std::vector<float> x((indirect_x.size() - 1) * x_stride() + kc() + XNN_EXTRA_BYTES / sizeof(float));

    std::vector<float> y((n() - 1) * y_stride() + kc());
    std::vector<uint32_t> i(n() * kc());
    std::vector<uint32_t, AlignedAllocator<uint32_t, XNN_EXTRA_BYTES>> ib(kc() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<float, AlignedAllocator<float, XNN_EXTRA_BYTES>> yb(kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y_ref(n() * kc());
    std::vector<uint32_t> i_ref(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      std::fill(y.begin(), y.end(), nanf(""));

      for (size_t p = 0; p < indirect_x.size(); p++) {
        indirect_x[p] = x.data() + p * x_stride();
      }
      std::shuffle(indirect_x.begin(), indirect_x.end(), rng);

      // Compute reference results, without clamping.
      for (size_t p = 0; p < n(); p++) {
        for (size_t k = 0; k < kc(); k++) {
          float max_value = indirect_x[p * s() * kh()][k];
          uint32_t max_index = 0;
          for (size_t j = 1; j < ks(); j++) {
            const float value = indirect_x[p * s() * kh() + j][k];
            if (value > max_value) {
              max_value = value;
              max_index = j;
            }
          }
          y_ref[p * kc() + k] = max_value;
          i_ref[p * kc() + k] = max_index;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float y_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Prepare output parameters.
      xnn_f32_output_params output_params = { };
      switch (variant) {
        case Variant::Native:
          output_params = xnn_compute_f32_output_params(y_min, y_max);
          break;
        case Variant::Scalar:
          output_params = xnn_compute_scalar_f32_output_params(y_min, y_max);
          break;
      }

      // Clamp reference results.
      for (float& y_value : y_ref) {
        y_value = std::max(std::min(y_value, y_max), y_min);
      }

      // Call optimized micro-kernel.
      argmaxpool(n(), ks(), kc(),
        indirect_x.data(), yb.data(), ib.data(), y.data(), i.data(),
        (kh() * s() - (packed_ks() - qr())) * sizeof(void*),
        (y_stride() - kc()) * sizeof(float),
        &output_params);

      // Verify results.
      for (size_t p = 0; p < n(); p++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_GE(y[p * y_stride() + k], y_min)
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
          ASSERT_LE(y[p * y_stride() + k], y_max)
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
          ASSERT_EQ(y_ref[p * kc() + k], y[p * y_stride() + k])
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
          ASSERT_EQ(indirect_x[p * s() * kh() + i_ref[p * kc() + k]][k], indirect_x[p * s() * kh() + i[p * kc() + k]][k])
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
          ASSERT_EQ(i_ref[p * kc() + k], i[p * kc() + k])
            << "at pixel " << p << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
        }
      }
    }
  }

 private:
  size_t n_{1};
  size_t s_{1};
  size_t kh_{1};
  size_t kw_{1};
  size_t mr_{1};
  size_t qr_{1};
  size_t kc_{1};
  size_t x_stride_{0};
  size_t y_stride_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
