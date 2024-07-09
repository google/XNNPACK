// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "binary-elementwise-operator-tester.h"

constexpr size_t kDim1 = 2;
constexpr size_t kDim2 = 3;
constexpr size_t kDim3 = 4;
constexpr size_t kDim4 = 5;
constexpr size_t kDim5 = 6;
constexpr size_t kDim6 = 7;


#ifndef XNN_EXCLUDE_F16_TESTS
TEST(MAXIMUM_ND_F16, maximum_0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
    .TestF16();
}

TEST(MAXIMUM_ND_F16, maximum_1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_2d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_1d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_2d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_3d_x_1d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_3d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_3d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_0d_x_4d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_1d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_2d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_3d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_4d_x_0d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_4d_x_1d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_4d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_4d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_4d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_0d_x_5d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_1d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_2d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_3d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_4d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_5d_x_0d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_5d_x_1d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_5d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_5d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_5d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_5d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_0d_x_6d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_1d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_2d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_3d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_4d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_5d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_6d_x_0d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(MAXIMUM_ND_F16, maximum_6d_x_1d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_6d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_6d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_6d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_6d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(MAXIMUM_ND_F16, maximum_6d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}
#endif  // XNN_EXCLUDE_F16_TESTS

TEST(MAXIMUM_ND_F32, maximum_0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
    .TestF32();
}

TEST(MAXIMUM_ND_F32, maximum_1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
      const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
      const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_2d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = (bm2 & (uint32_t(1) << 0)) != 0;
    const bool input2_broadcast_dim2 = (bm2 & (uint32_t(1) << 1)) != 0;
    const bool input2_broadcast_dim3 = (bm2 & (uint32_t(1) << 2)) != 0;
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_1d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_2d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool input1_broadcast_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool input1_broadcast_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_3d_x_1d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_3d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_3d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_0d_x_4d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_1d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_2d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_3d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_4d_x_0d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_4d_x_1d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_4d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_4d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_4d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_0d_x_5d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_1d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_2d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_3d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_4d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_5d_x_0d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_5d_x_1d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_5d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_5d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_5d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_5d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_0d_x_6d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_1d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_2d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_3d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_4d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_5d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_6d_x_0d) {
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
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(MAXIMUM_ND_F32, maximum_6d_x_1d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_6d_x_2d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_6d_x_3d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_6d_x_4d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_6d_x_5d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(MAXIMUM_ND_F32, maximum_6d_x_6d) {
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Maximum)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}
