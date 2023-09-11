// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "binary-elementwise-operator-tester.h"

constexpr size_t kDim1 = 2;
constexpr size_t kDim2 = 3;
constexpr size_t kDim3 = 4;
constexpr size_t kDim4 = 5;
constexpr size_t kDim5 = 6;
constexpr size_t kDim6 = 7;


#ifndef XNN_EXCLUDE_F16_TESTS
TEST(MULTIPLY_ND_F16, multiply_0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
    .TestF16();
}

TEST(MULTIPLY_ND_F16, multiply_1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_2d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_1d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_2d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_3d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_3d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_3d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_0d_x_4d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_1d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_2d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_3d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_4d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_4d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_4d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_4d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_4d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_0d_x_5d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_1d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_2d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_3d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_4d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_5d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_5d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_5d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_5d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_5d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_5d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_0d_x_6d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
    const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_1d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_2d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_3d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_4d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_5d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_6d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MULTIPLY_ND_F16, multiply_6d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_6d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_6d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_6d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_6d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, multiply_6d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(MULTIPLY_ND_F16, qmin) {
  for (int32_t qmin = std::numeric_limits<int16_t>::max() - 1000; qmin > std::numeric_limits<int16_t>::min(); qmin -= 5000) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(qmin)
          .TestF16();
      }
    }
  }
}

TEST(MULTIPLY_ND_F16, qmax) {
  for (int32_t qmax = std::numeric_limits<int16_t>::min() + 1000; qmax < std::numeric_limits<int16_t>::max(); qmax += 5000) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmax(qmax)
          .TestF16();
      }
    }
  }
}
#endif  // XNN_EXCLUDE_F16_TESTS


TEST(MULTIPLY_ND_F32, multiply_0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
    .TestF32();
}

TEST(MULTIPLY_ND_F32, multiply_1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_2d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_1d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_2d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_3d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_3d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_3d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_0d_x_4d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_1d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_2d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_3d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_4d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_4d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_4d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_4d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_4d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_0d_x_5d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_1d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_2d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_3d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_4d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_5d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_5d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_5d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_5d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_5d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_5d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_0d_x_6d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
    const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_1d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_2d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_3d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_4d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_5d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_6d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MULTIPLY_ND_F32, multiply_6d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_6d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_6d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_6d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_6d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, multiply_6d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(MULTIPLY_ND_F32, qmin) {
  for (int32_t qmin = std::numeric_limits<int16_t>::max() - 1000; qmin > std::numeric_limits<int16_t>::min(); qmin -= 5000) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(qmin)
          .TestF32();
      }
    }
  }
}

TEST(MULTIPLY_ND_F32, qmax) {
  for (int32_t qmax = std::numeric_limits<int16_t>::min() + 1000; qmax < std::numeric_limits<int16_t>::max(); qmax += 5000) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmax(qmax)
          .TestF32();
      }
    }
  }
}


TEST(MULTIPLY_ND_QS8, multiply_0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestQS8();
}

TEST(MULTIPLY_ND_QS8, multiply_1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_2d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_1d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_2d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_3d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_3d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_3d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_0d_x_4d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_1d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_2d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_3d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_4d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_4d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_4d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_4d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_4d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_0d_x_5d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_1d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_2d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_3d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_4d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_5d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_5d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_5d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_5d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_5d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_5d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_0d_x_6d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
    const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_1d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_2d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_3d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_4d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_5d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_6d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestQS8();
  }
}

TEST(MULTIPLY_ND_QS8, multiply_6d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_6d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_6d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_6d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_6d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, multiply_6d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestQS8();
    }
  }
}

TEST(MULTIPLY_ND_QS8, input1_scale) {
  for (float input1_scale = 0.1f; input1_scale <= 10.0f; input1_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_scale(input1_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .TestQS8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QS8, input1_zero_point) {
  for (int16_t input1_zero_point = std::numeric_limits<int8_t>::min();
       input1_zero_point <= std::numeric_limits<int8_t>::max();
       input1_zero_point += 51)
  {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_zero_point(input1_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .TestQS8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QS8, input2_scale) {
  for (float input2_scale = 0.1f; input2_scale <= 10.0f; input2_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_scale(input2_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .TestQS8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QS8, input2_zero_point) {
  for (int16_t input2_zero_point = std::numeric_limits<int8_t>::min();
       input2_zero_point <= std::numeric_limits<int8_t>::max();
       input2_zero_point += 51)
  {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input2_zero_point(input2_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .TestQS8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QS8, output_scale) {
  for (float output_scale = 0.1f; output_scale <= 10.0f; output_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_scale(output_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .TestQS8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QS8, output_zero_point) {
  for (int16_t output_zero_point = std::numeric_limits<int8_t>::min();
       output_zero_point <= std::numeric_limits<int8_t>::max();
       output_zero_point += 51)
  {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .output_zero_point(output_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .TestQS8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QS8, qmin) {
  for (int16_t qmin = std::numeric_limits<int8_t>::max() - 1; qmin > std::numeric_limits<int8_t>::min(); qmin -= 50) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .TestQS8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QS8, qmax) {
  for (int16_t qmax = std::numeric_limits<int8_t>::min() + 1; qmax < std::numeric_limits<int8_t>::max(); qmax += 50) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .TestQS8();
      }
    }
  }
}


TEST(MULTIPLY_ND_QU8, multiply_0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .TestQU8();
}

TEST(MULTIPLY_ND_QU8, multiply_1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_2d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_1d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_2d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_3d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_3d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_3d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_0d_x_4d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_1d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_2d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_3d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_4d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_4d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_4d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_4d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_4d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_0d_x_5d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_1d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_2d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_3d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_4d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_5d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_5d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_5d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_5d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_5d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_5d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_0d_x_6d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
    const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
    const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_1d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_2d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_3d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_4d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_5d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_6d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .TestQU8();
  }
}

TEST(MULTIPLY_ND_QU8, multiply_6d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_6d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_6d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_6d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_6d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, multiply_6d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
      const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
      const bool input1_broadcast_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
      const bool input1_broadcast_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
      const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
      const bool input2_broadcast_dim5 = (bm2 & (uint32_t(1) << 4)) != 0;
      const bool input2_broadcast_dim6 = (bm2 & (uint32_t(1) << 5)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .TestQU8();
    }
  }
}

TEST(MULTIPLY_ND_QU8, input1_scale) {
  for (float input1_scale = 0.1f; input1_scale <= 10.0f; input1_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_scale(input1_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .TestQU8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QU8, input1_zero_point) {
  for (int16_t input1_zero_point = std::numeric_limits<uint8_t>::min();
       input1_zero_point <= std::numeric_limits<uint8_t>::max();
       input1_zero_point += 51)
  {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_zero_point(input1_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .TestQU8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QU8, input2_scale) {
  for (float input2_scale = 0.1f; input2_scale <= 10.0f; input2_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_scale(input2_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .TestQU8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QU8, input2_zero_point) {
  for (int16_t input2_zero_point = std::numeric_limits<uint8_t>::min();
       input2_zero_point <= std::numeric_limits<uint8_t>::max();
       input2_zero_point += 51)
  {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input2_zero_point(input2_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .TestQU8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QU8, output_scale) {
  for (float output_scale = 0.1f; output_scale <= 10.0f; output_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_scale(output_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .TestQU8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QU8, output_zero_point) {
  for (int16_t output_zero_point = std::numeric_limits<uint8_t>::min();
       output_zero_point <= std::numeric_limits<uint8_t>::max();
       output_zero_point += 51)
  {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .output_zero_point(output_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .TestQU8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QU8, qmin) {
  for (int16_t qmin = std::numeric_limits<uint8_t>::max() - 1; qmin > std::numeric_limits<uint8_t>::min(); qmin -= 50) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .TestQU8();
      }
    }
  }
}

TEST(MULTIPLY_ND_QU8, qmax) {
  for (int16_t qmax = std::numeric_limits<uint8_t>::min() + 1; qmax < std::numeric_limits<uint8_t>::max(); qmax += 50) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
        const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
        const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
        const bool input1_broadcast_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
        const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
        const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
        const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
        const bool input2_broadcast_dim4 = (bm2 & (uint32_t(1) << 3)) != 0;
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Multiply)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .TestQU8();
      }
    }
  }
}
