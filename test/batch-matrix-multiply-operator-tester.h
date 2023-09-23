// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>

#include <gtest/gtest.h>


class BatchMatMulOperatorTester {
 public:
  inline BatchMatMulOperatorTester& m(size_t m) {
    assert(m >= 1);
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline BatchMatMulOperatorTester& k(size_t k) {
    assert(k >= 1);
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline BatchMatMulOperatorTester& n(size_t n) {
    assert(n >= 1);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline BatchMatMulOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline BatchMatMulOperatorTester& transpose_b(bool transpose_b) {
    this->transpose_b_ = transpose_b;
    return *this;
  }

  inline bool transpose_b() const {
    return this->transpose_b_;
  }

  inline BatchMatMulOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  inline uint32_t flags() const {
    if (transpose_b()) {
      return XNN_FLAG_TRANSPOSE_B;
    } else {
      return 0;
    }
  }

  void TestF16() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<uint16_t> lhs(XNN_EXTRA_BYTES / sizeof(uint16_t) + batch_size() * m() * k());
    std::vector<uint16_t> rhs(XNN_EXTRA_BYTES / sizeof(uint16_t) + batch_size() * k() * n());
    std::vector<uint16_t> output(batch_size() * m() * n());
    std::vector<float> output_ref(batch_size() * m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(lhs.begin(), lhs.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(rhs.begin(), rhs.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);


      // Compute reference results.
      if (transpose_b()) {
        // lhs is B*M*K, rhs is B*N*K
        for (size_t b = 0; b < batch_size(); b++) {
          for (size_t mi = 0; mi < m(); mi++) {
            for (size_t ni = 0; ni < n(); ni++) {
              for (size_t ki = 0; ki < k(); ki++) {
                output_ref[b * m() * n() + mi * n() + ni] +=
                    fp16_ieee_to_fp32_value(lhs[b * m() * k() + mi * k() + ki]) *
                    fp16_ieee_to_fp32_value(rhs[b * n() * k() + ni * k() + ki]);
              }
            }
          }
        }
      } else {
        // lhs is B*M*K, rhs is B*K*N
        for (size_t b = 0; b < batch_size(); b++) {
          for (size_t mi = 0; mi < m(); mi++) {
            for (size_t ni = 0; ni < n(); ni++) {
              for (size_t ki = 0; ki < k(); ki++) {
                output_ref[b * m() * n() + mi * n() + ni] +=
                    fp16_ieee_to_fp32_value(lhs[b * m() * k() + mi * k() + ki]) *
                    fp16_ieee_to_fp32_value(rhs[b * k() * n() + ki * n() + ni]);
              }
            }
          }
        }
      }


      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t batch_matrix_multiply_op = nullptr;

      const xnn_status status = xnn_create_batch_matrix_multiply_nc_f16(flags(), &batch_matrix_multiply_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, batch_matrix_multiply_op);

      // Smart pointer to automatically delete batch_matrix_multiply_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_batch_matrix_multiply_op(
        batch_matrix_multiply_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(
        xnn_status_success,
        xnn_reshape_batch_matrix_multiply_nc_f16(
          batch_matrix_multiply_op, batch_size(), m(), k(), n(),
          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
      ASSERT_NE(workspace_size, 0);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_batch_matrix_multiply_nc_f16(
          batch_matrix_multiply_op,
          workspace.data(), lhs.data(), rhs.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(batch_matrix_multiply_op, /*threadpool=*/nullptr));

      VerifyF16(output, output_ref);
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<float> lhs(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * m() * k());
    std::vector<float> rhs(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * k() * n());
    std::vector<float> output(batch_size() * m() * n());
    std::vector<float> output_ref(batch_size() * m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(lhs.begin(), lhs.end(), [&]() { return f32dist(rng); });
      std::generate(rhs.begin(), rhs.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);

      // Compute reference results.
      if (transpose_b()) {
        // lhs is B*M*K, rhs is B*N*K
        for (size_t b = 0; b < batch_size(); b++) {
          for (size_t mi = 0; mi < m(); mi++) {
            for (size_t ni = 0; ni < n(); ni++) {
              for (size_t ki = 0; ki < k(); ki++) {
                output_ref[b * m() * n() + mi * n() + ni] +=
                    lhs[b * m() * k() + mi * k() + ki] *
                    rhs[b * n() * k() + ni * k() + ki];
              }
            }
          }
        }
      } else {
        // lhs is B*M*K, rhs is B*K*N
        for (size_t b = 0; b < batch_size(); b++) {
          for (size_t mi = 0; mi < m(); mi++) {
            for (size_t ni = 0; ni < n(); ni++) {
              for (size_t ki = 0; ki < k(); ki++) {
                output_ref[b * m() * n() + mi * n() + ni] +=
                    lhs[b * m() * k() + mi * k() + ki] *
                    rhs[b * k() * n() + ki * n() + ni];
              }
            }
          }
        }
      }


      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t batch_matrix_multiply_op = nullptr;

      const xnn_status status = xnn_create_batch_matrix_multiply_nc_f32(flags(), &batch_matrix_multiply_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, batch_matrix_multiply_op);

      // Smart pointer to automatically delete batch_matrix_multiply_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_batch_matrix_multiply_op(
        batch_matrix_multiply_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(
        xnn_status_success,
        xnn_reshape_batch_matrix_multiply_nc_f32(
          batch_matrix_multiply_op, batch_size(), m(), k(), n(),
          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
      ASSERT_NE(workspace_size, 0);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_batch_matrix_multiply_nc_f32(
          batch_matrix_multiply_op,
          workspace.data(), lhs.data(), rhs.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(batch_matrix_multiply_op, /*threadpool=*/nullptr));

      VerifyF32(output, output_ref);
    }
  }

  void VerifyF16(const std::vector<uint16_t>& output,
                 const std::vector<float>& output_ref) const {
    for (size_t bi = 0; bi < batch_size(); bi++) {
      for (size_t mi = 0; mi < m(); mi++) {
        for (size_t ni = 0; ni < n(); ni++) {
          EXPECT_NEAR(
            output_ref[bi * m() * n() + mi * n() + ni], fp16_ieee_to_fp32_value(output[bi * m() * n() + mi * n() + ni]),
            1.0e-2f * std::abs(output_ref[bi * m() * n() + mi * n() + ni]))
            << "batch = " << bi << " / " << batch_size() << ", m = " << mi << " / " << m() << ", n = " << ni << " / "
            << n();
        }
      }
    }
  }

  void VerifyF32(const std::vector<float>& output,
                 const std::vector<float>& output_ref) const
  {
    // Verify results.
    for (size_t bi = 0; bi < batch_size(); bi++) {
      for (size_t mi = 0; mi < m(); mi++) {
        for (size_t ni = 0; ni < n(); ni++) {
          EXPECT_NEAR(
            output_ref[bi * m() * n() + mi * n() + ni], output[bi * m() * n() + mi * n() + ni],
            1.0e-4f * std::abs(output_ref[bi * m() * n() + mi * n() + ni]))
            << "batch = " << bi << " / " << batch_size() << ", m = " << mi << " / " << m() << ", n = " << ni << " / "
            << n();
        }
      }
    }
  }

 private:
  // TODO(zhin): support flags for transpose lhs.
  size_t m_{1};
  size_t k_{1};
  size_t n_{1};
  size_t batch_size_{1};
  bool transpose_b_{false};
  size_t iterations_{1};
};
