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
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>


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

  inline BatchMatMulOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
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

      // lhs is B*M*K, rhs is B*N*K
      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      for (size_t b = 0; b < batch_size(); b++) {
        for (size_t mi = 0; mi < m(); mi++) {
          for (size_t ni = 0; ni < n(); ni++) {
            for (size_t ki = 0; ki < k(); ki++) {
              output_ref[b * m() * n() + mi * n() + ni] +=
                  lhs[b * m() * k() + mi * k() + ki] *
                  rhs[b * k() * n() + ni * k() + ki];
            }
          }
        }
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t batch_matrix_multiply_op = nullptr;

      const xnn_status status = xnn_create_batch_matrix_multiply_nc_f32(/*flags=*/0, &batch_matrix_multiply_op);
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
        xnn_run_operator(batch_matrix_multiply_op, nullptr /* thread pool */));

      VerifyF32(output, output_ref);
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
  // TODO(zhin): support flags for transpose lhs and rhs.
  size_t m_{1};
  size_t k_{1};
  size_t n_{1};
  size_t batch_size_{1};
  size_t iterations_{1};
};
