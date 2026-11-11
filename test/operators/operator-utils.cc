// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/operator-utils.h"

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "src/xnnpack/config.h"

TEST(COMPUTE_CONVOLUTION_OUTPUT_DIMENSION, compute) {
  ASSERT_EQ(xnn_compute_convolution_output_dimension(5, 3, 1, 1), 3);
  ASSERT_EQ(xnn_compute_convolution_output_dimension(10, 3, 2, 1), 6);
  ASSERT_EQ(xnn_compute_convolution_output_dimension(5, 3, 1, 2), 2);
}

namespace {
// A dummy, nop microkernel for testing.
void dummy_gemm(size_t mr, size_t nr, size_t k, const void *a, size_t a_stride,
                const void *w, void *c, size_t cm_stride, size_t cn_stride,
                const void *params) {}
xnn_hmp_gemm_ukernel empty_gemm_ukernel = xnn_init_hmp_gemm_ukernel(nullptr);
xnn_hmp_gemm_ukernel dummy_gemm_ukernel = xnn_init_hmp_gemm_ukernel(dummy_gemm);

void dummy_igemm(size_t mr, size_t nr, size_t kc, size_t ks, const void **a,
                 const void *w, void *c, size_t cm_stride, size_t cn_stride,
                 size_t a_offset, const void *zero, const void *params) {}
xnn_hmp_igemm_ukernel empty_igemm_ukernel = xnn_init_hmp_igemm_ukernel(nullptr);
xnn_hmp_igemm_ukernel dummy_igemm_ukernel =
    xnn_init_hmp_igemm_ukernel(dummy_igemm);
}  // namespace

TEST(HEURISTIC_MR, batch_size_same_as_mr) {
  xnn_gemm_config params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = dummy_igemm_ukernel;
  params.mr = 2;
  params.nr = 8;

  ASSERT_EQ(2, xnn_get_heuristic_mr_gemm(2, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(2, xnn_get_heuristic_mr_igemm(2, params.mr, params.nr,
                                          params.minmax.igemm));

  params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = dummy_gemm_ukernel;
  params.minmax.gemm[2] = empty_gemm_ukernel;
  params.minmax.gemm[3] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = dummy_igemm_ukernel;
  params.minmax.igemm[2] = empty_igemm_ukernel;
  params.minmax.igemm[3] = dummy_igemm_ukernel;
  params.mr = 4;
  params.nr = 8;

  ASSERT_EQ(4, xnn_get_heuristic_mr_gemm(4, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(4, xnn_get_heuristic_mr_igemm(4, params.mr, params.nr,
                                          params.minmax.igemm));
}

TEST(HEURISTIC_MR, batch_size_smaller_than_mr) {
  xnn_gemm_config params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = dummy_gemm_ukernel;
  params.minmax.gemm[2] = dummy_gemm_ukernel;
  params.minmax.gemm[3] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = dummy_igemm_ukernel;
  params.minmax.igemm[2] = dummy_igemm_ukernel;
  params.minmax.igemm[3] = dummy_igemm_ukernel;
  params.mr = 4;
  params.nr = 8;

  // batch size == 3 < mr == 4, pick smallest available kernel to minimize
  // clamps.
  ASSERT_EQ(3, xnn_get_heuristic_mr_gemm(3, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(3, xnn_get_heuristic_mr_igemm(3, params.mr, params.nr,
                                          params.minmax.igemm));

  params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = empty_gemm_ukernel;
  params.minmax.gemm[2] = empty_gemm_ukernel;
  params.minmax.gemm[3] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = empty_igemm_ukernel;
  params.minmax.igemm[2] = empty_igemm_ukernel;
  params.minmax.igemm[3] = dummy_igemm_ukernel;
  params.mr = 4;
  params.nr = 8;

  // The only kernel with mr < 2 is mr == 1, which is too inefficient for this
  // batch size 2.
  ASSERT_EQ(4, xnn_get_heuristic_mr_gemm(2, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(4, xnn_get_heuristic_mr_igemm(2, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(1, xnn_get_heuristic_mr_gemm(1, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(1, xnn_get_heuristic_mr_igemm(1, params.mr, params.nr,
                                          params.minmax.igemm));

  params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = empty_gemm_ukernel;
  params.minmax.gemm[2] = empty_gemm_ukernel;
  params.minmax.gemm[3] = dummy_gemm_ukernel;
  params.minmax.gemm[4] = dummy_gemm_ukernel;
  params.minmax.gemm[5] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = empty_igemm_ukernel;
  params.minmax.igemm[2] = empty_igemm_ukernel;
  params.minmax.igemm[3] = dummy_igemm_ukernel;
  params.minmax.igemm[4] = dummy_igemm_ukernel;
  params.minmax.igemm[5] = dummy_igemm_ukernel;
  params.mr = 6;
  params.nr = 8;

  ASSERT_EQ(5, xnn_get_heuristic_mr_gemm(5, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(5, xnn_get_heuristic_mr_igemm(5, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(4, xnn_get_heuristic_mr_gemm(4, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(4, xnn_get_heuristic_mr_igemm(4, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(4, xnn_get_heuristic_mr_gemm(2, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(4, xnn_get_heuristic_mr_igemm(2, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(1, xnn_get_heuristic_mr_gemm(1, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(1, xnn_get_heuristic_mr_igemm(1, params.mr, params.nr,
                                          params.minmax.igemm));
}

TEST(HEURISTIC_MR, batch_size_larger_than_mr) {
  xnn_gemm_config params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = empty_gemm_ukernel;
  params.minmax.gemm[2] = dummy_gemm_ukernel;
  params.minmax.gemm[3] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = empty_igemm_ukernel;
  params.minmax.igemm[2] = dummy_igemm_ukernel;
  params.minmax.igemm[3] = dummy_igemm_ukernel;
  params.mr = 4;
  params.nr = 8;

  ASSERT_EQ(3, xnn_get_heuristic_mr_gemm(5, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(3, xnn_get_heuristic_mr_igemm(5, params.mr, params.nr,
                                          params.minmax.igemm));

  params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = dummy_gemm_ukernel;
  params.minmax.gemm[2] = dummy_gemm_ukernel;
  params.minmax.gemm[3] = dummy_gemm_ukernel;
  params.minmax.gemm[4] = dummy_gemm_ukernel;
  params.minmax.gemm[5] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = dummy_igemm_ukernel;
  params.minmax.igemm[2] = dummy_igemm_ukernel;
  params.minmax.igemm[3] = dummy_igemm_ukernel;
  params.minmax.igemm[4] = dummy_igemm_ukernel;
  params.minmax.igemm[5] = dummy_igemm_ukernel;
  params.mr = 6;
  params.nr = 8;

  ASSERT_EQ(4, xnn_get_heuristic_mr_gemm(7, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(4, xnn_get_heuristic_mr_igemm(7, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_gemm(11, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_igemm(11, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_gemm(22, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_igemm(22, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(5, xnn_get_heuristic_mr_gemm(50, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(5, xnn_get_heuristic_mr_igemm(50, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(5, xnn_get_heuristic_mr_gemm(50, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(5, xnn_get_heuristic_mr_igemm(50, params.mr, params.nr,
                                          params.minmax.igemm));
  // Tests some MobiletNet params.
  ASSERT_EQ(6, xnn_get_heuristic_mr_gemm(112 * 112, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_igemm(112 * 112, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_gemm(56 * 56, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_igemm(56 * 56, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_gemm(14 * 14, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(6, xnn_get_heuristic_mr_igemm(14 * 14, params.mr, params.nr,
                                          params.minmax.igemm));
  ASSERT_EQ(5, xnn_get_heuristic_mr_gemm(7 * 7, params.mr, params.nr,
                                         params.minmax.gemm));
  ASSERT_EQ(5, xnn_get_heuristic_mr_igemm(7 * 7, params.mr, params.nr,
                                          params.minmax.igemm));
}

TEST(HEURISTIC_MR, max_mr_without_mr1_kernel) {
  xnn_gemm_config params = {};
  params.minmax.gemm[0] = empty_gemm_ukernel;
  params.minmax.gemm[1] = empty_gemm_ukernel;
  params.minmax.gemm[2] = empty_gemm_ukernel;
  params.minmax.gemm[3] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = empty_igemm_ukernel;
  params.minmax.igemm[1] = empty_igemm_ukernel;
  params.minmax.igemm[2] = empty_igemm_ukernel;
  params.minmax.igemm[3] = dummy_igemm_ukernel;
  params.mr = 4;
  params.nr = 8;

  // batch size == 3 < mr == 4, pick smallest available kernel to minimize
  // clamps.
  ASSERT_EQ(4, xnn_get_heuristic_mr_gemm(3, params.mr, params.nr,
                                         params.minmax.gemm));
}

TEST(HEURISTIC_MR, batch_size_zero) {
  xnn_gemm_config params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = dummy_igemm_ukernel;
  params.mr = 2;
  params.nr = 8;

  // batch_size == 0 must not cause OOB read via mr_is_available_*(0, cases)
  // which would access cases[-1].
  ASSERT_EQ(params.mr, xnn_get_heuristic_mr_gemm(0, params.mr, params.nr,
                                                  params.minmax.gemm));
  ASSERT_EQ(params.mr, xnn_get_heuristic_mr_igemm(0, params.mr, params.nr,
                                                   params.minmax.igemm));
}

TEST(COMPUTE_CONVOLUTION_OUTPUT_DIMENSION, overflow_and_invalid) {
  // subsampling == 0 must return 0 instead of division by zero.
  EXPECT_EQ(0, xnn_compute_convolution_output_dimension(5, 3, 1, 0));
  // kernel == 0 must return 0 instead of underflow.
  EXPECT_EQ(0, xnn_compute_convolution_output_dimension(5, 0, 1, 1));
  // overflow in effective kernel returns 1 (clamped output dim).
  EXPECT_EQ(1, xnn_compute_convolution_output_dimension(5, SIZE_MAX, 2, 1));
}

TEST(COMPUTE_DECONVOLUTION_OUTPUT_DIMENSION, compute_and_overflow) {
  EXPECT_EQ(5, xnn_compute_deconvolution_output_dimension(2, 0, 0, 3, 1, 2));
  // input_dim == 0 must return 0 instead of underflow.
  EXPECT_EQ(0, xnn_compute_deconvolution_output_dimension(0, 0, 0, 3, 1, 2));
  // kernel_dim == 0 must return 0 instead of underflow.
  EXPECT_EQ(0, xnn_compute_deconvolution_output_dimension(2, 0, 0, 0, 1, 2));
  // overflow in stride * (input - 1) returns 0.
  EXPECT_EQ(0, xnn_compute_deconvolution_output_dimension(
                   2, 0, 0, 3, 1, SIZE_MAX));
}

TEST(COMPUTE_UNPOOLING_OUTPUT_DIMENSION, compute_and_overflow) {
  EXPECT_EQ(4, xnn_compute_unpooling_output_dimension(2, 0, 2));
  EXPECT_EQ(0, xnn_compute_unpooling_output_dimension(0, 0, 2));
  EXPECT_EQ(0, xnn_compute_unpooling_output_dimension(2, 0, 0));
}

TEST(HEURISTIC_MR, batch_size_overflow) {
  xnn_gemm_config params = {};
  params.minmax.gemm[0] = dummy_gemm_ukernel;
  params.minmax.gemm[1] = dummy_gemm_ukernel;
  params.minmax.igemm[0] = dummy_igemm_ukernel;
  params.minmax.igemm[1] = dummy_igemm_ukernel;
  params.mr = 2;
  params.nr = 8;

  // Huge batch_size must not wrap microkernel cost to 0 or crash.
  EXPECT_NE(0, xnn_get_heuristic_mr_gemm(SIZE_MAX, params.mr, params.nr,
                                         params.minmax.gemm));
  EXPECT_NE(0, xnn_get_heuristic_mr_igemm(SIZE_MAX, params.mr, params.nr,
                                          params.minmax.igemm));
}

TEST(ALLOCATE_EXTRA_PARAMS, overflow) {
  struct xnn_operator op = {};
  EXPECT_EQ(xnn_status_out_of_memory,
            xnn_allocate_extra_params(&op, SIZE_MAX));
  EXPECT_EQ(xnn_status_out_of_memory,
            xnn_allocate_extra_params(
                &op, SIZE_MAX / sizeof(union xnn_params) + 1));
}
