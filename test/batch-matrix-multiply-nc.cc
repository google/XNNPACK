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
  BatchMatMulOperatorTester()
    .batch_size(5)
    .m(17)
    .k(23)
    .n(19)
    .iterations(3)
    .TestF32();
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
  BatchMatMulOperatorTester()
    .batch_size(5)
    .m(37)
    .k(101)
    .n(71)
    .iterations(3)
    .TestF32();
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
  BatchMatMulOperatorTester()
    .transpose_b(true)
    .batch_size(5)
    .m(17)
    .k(23)
    .n(19)
    .iterations(3)
    .TestF32();
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
  BatchMatMulOperatorTester()
    .transpose_b(true)
    .batch_size(5)
    .m(37)
    .k(101)
    .n(71)
    .iterations(3)
    .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, unit_batch) {
  BatchMatMulOperatorTester()
    .batch_size(1)
    .m(17)
    .k(23)
    .n(19)
    .iterations(3)
    .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, small_batch) {
  BatchMatMulOperatorTester()
    .batch_size(5)
    .m(17)
    .k(23)
    .n(19)
    .iterations(3)
    .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, unit_batch_bigger_matrices) {
  BatchMatMulOperatorTester()
    .batch_size(1)
    .m(37)
    .k(101)
    .n(71)
    .iterations(3)
    .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, small_batch_bigger_matrices) {
  BatchMatMulOperatorTester()
    .batch_size(5)
    .m(37)
    .k(101)
    .n(71)
    .iterations(3)
    .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, unit_batch_transpose_b) {
  BatchMatMulOperatorTester()
    .transpose_b(true)
    .batch_size(1)
    .m(17)
    .k(23)
    .n(17)
    .iterations(3)
    .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, small_batch_tranpose_b) {
  BatchMatMulOperatorTester()
    .transpose_b(true)
    .batch_size(5)
    .m(17)
    .k(23)
    .n(19)
    .iterations(3)
    .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, unit_batch_bigger_matrices_transpose_b) {
  BatchMatMulOperatorTester()
    .transpose_b(true)
    .batch_size(1)
    .m(37)
    .k(101)
    .n(71)
    .iterations(3)
    .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, small_batch_bigger_matrices_transpose_b) {
  BatchMatMulOperatorTester()
    .transpose_b(true)
    .batch_size(5)
    .m(37)
    .k(101)
    .n(71)
    .iterations(3)
    .TestF16();
}
