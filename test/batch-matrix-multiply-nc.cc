// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <gtest/gtest.h>

#include "batch-matrix-multiply-operator-tester.h"


TEST(BATCH_MATRIX_MULTIPLY_NC_F32, unit_batch) {
  BatchMatMulOperatorTester()
    .batch_size(1)
    .m(17)
    .k(23)
    .n(19)
    .iterations(3)
    .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, small_batch) {
  for (size_t batch_size = 2; batch_size < 13; batch_size++) {
    BatchMatMulOperatorTester()
      .batch_size(batch_size)
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
  }
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, unit_batch_bigger_matrices) {
  BatchMatMulOperatorTester()
    .batch_size(1)
    .m(37)
    .k(101)
    .n(71)
    .iterations(3)
    .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, small_batch_bigger_matrices) {
  for (size_t batch_size = 2; batch_size < 13; batch_size++) {
    BatchMatMulOperatorTester()
      .batch_size(batch_size)
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF32();
  }
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, various_batch_and_small_matrices) {
  for (size_t b = 1; b < 11; b++) {
    for (size_t m = 3; m < 27; m += 3) {
      for (size_t k = 5; k < 30; k += 5) {
        for (size_t n = 7; n < 49; n += 7) {
          BatchMatMulOperatorTester()
            .batch_size(b)
            .m(m)
            .k(k)
            .n(n)
            .iterations(3)
            .TestF32();
        }
      }
    }
  }
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, unit_batch_transpose_b) {
  BatchMatMulOperatorTester()
    .transpose_b(true)
    .batch_size(1)
    .m(17)
    .k(23)
    .n(17)
    .iterations(3)
    .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, small_batch_tranpose_b) {
  for (size_t batch_size = 2; batch_size < 13; batch_size++) {
    BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_size(batch_size)
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
  }
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, unit_batch_bigger_matrices_transpose_b) {
  BatchMatMulOperatorTester()
    .transpose_b(true)
    .batch_size(1)
    .m(37)
    .k(101)
    .n(71)
    .iterations(3)
    .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, small_batch_bigger_matrices_transpose_b) {
  for (size_t batch_size = 2; batch_size < 13; batch_size++) {
    BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_size(batch_size)
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF32();
  }
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, various_batch_and_small_matrices_transpose_b) {
  for (size_t b = 1; b < 11; b++) {
    for (size_t m = 3; m < 27; m += 3) {
      for (size_t k = 5; k < 30; k += 5) {
        for (size_t n = 7; n < 49; n += 7) {
          BatchMatMulOperatorTester()
            .transpose_b(true)
            .batch_size(b)
            .m(m)
            .k(k)
            .n(n)
            .iterations(3)
            .TestF32();
        }
      }
    }
  }
}