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


TEST(ADD_ND_QS8, 0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
    .TestQS8();
}

TEST(ADD_ND_QS8, 1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim2, input2_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim2, input1_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 2d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 1d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 2d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 3d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 3d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 3d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 0d_x_4d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 1d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 2d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 3d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 4d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 4d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 4d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 4d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 4d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 0d_x_5d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 1d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 2d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 3d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 4d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 5d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 5d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 5d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 5d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 5d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 5d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 0d_x_6d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
    const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 1d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 2d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 3d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 4d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 5d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 6d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
    const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestQS8();
  }
}

TEST(ADD_ND_QS8, 6d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 6d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 6d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 6d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 6d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, 6d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestQS8();
    }
  }
}

TEST(ADD_ND_QS8, input1_scale) {
  for (float input1_scale = 0.1f; input1_scale <= 10.0f; input1_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input1_scale(input1_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQS8();
      }
    }
  }
}

TEST(ADD_ND_QS8, input1_zero_point) {
  for (int32_t input1_zero_point = -128; input1_zero_point <= 127; input1_zero_point += 51) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input1_zero_point(input1_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQS8();
      }
    }
  }
}

TEST(ADD_ND_QS8, input2_scale) {
  for (float input2_scale = 0.1f; input2_scale <= 10.0f; input2_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input1_scale(input2_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQS8();
      }
    }
  }
}

TEST(ADD_ND_QS8, input2_zero_point) {
  for (int32_t input2_zero_point = -128; input2_zero_point <= 127; input2_zero_point += 51) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input2_zero_point(input2_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQS8();
      }
    }
  }
}

TEST(ADD_ND_QS8, output_scale) {
  for (float output_scale = 0.1f; output_scale <= 10.0f; output_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input1_scale(output_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQS8();
      }
    }
  }
}

TEST(ADD_ND_QS8, output_zero_point) {
  for (int32_t output_zero_point = -128; output_zero_point <= 127; output_zero_point += 51) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .output_zero_point(output_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQS8();
      }
    }
  }
}

TEST(ADD_ND_QU8, 0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
    .TestQU8();
}

TEST(ADD_ND_QU8, 1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim2, input2_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim2, input1_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 2d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 1d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 2d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 3d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 3d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 3d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 0d_x_4d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 1d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 2d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 3d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 4d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 4d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 4d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 4d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 4d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 0d_x_5d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 1d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 2d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 3d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 4d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 5d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 5d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 5d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 5d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 5d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 5d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 0d_x_6d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
    const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 1d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 2d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 3d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 4d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 5d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 6d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
    const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestQU8();
  }
}

TEST(ADD_ND_QU8, 6d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 6d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 6d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 6d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 6d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, 6d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(ADD_ND_QU8, input1_scale) {
  for (float input1_scale = 0.1f; input1_scale <= 10.0f; input1_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input1_scale(input1_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQU8();
      }
    }
  }
}

TEST(ADD_ND_QU8, input1_zero_point) {
  for (int32_t input1_zero_point = 0; input1_zero_point <= 255; input1_zero_point += 51) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input1_zero_point(input1_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQU8();
      }
    }
  }
}

TEST(ADD_ND_QU8, input2_scale) {
  for (float input2_scale = 0.1f; input2_scale <= 10.0f; input2_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input1_scale(input2_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQU8();
      }
    }
  }
}

TEST(ADD_ND_QU8, input2_zero_point) {
  for (int32_t input2_zero_point = 0; input2_zero_point <= 255; input2_zero_point += 51) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input2_zero_point(input2_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQU8();
      }
    }
  }
}

TEST(ADD_ND_QU8, output_scale) {
  for (float output_scale = 0.1f; output_scale <= 10.0f; output_scale *= 3.14f) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .input1_scale(output_scale)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQU8();
      }
    }
  }
}

TEST(ADD_ND_QU8, output_zero_point) {
  for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
    for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
      for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
        const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
        const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
        const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
        const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
        const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
        const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
        const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
        const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
        const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
        const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
        const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
        const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
        const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
        const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
        const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
        const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
        BinaryElementwiseOperatorTester()
          .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
          .output_zero_point(output_zero_point)
          .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
          .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
          .TestQU8();
      }
    }
  }
}

TEST(ADD_ND_F16, 0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
    .TestF16();
}

TEST(ADD_ND_F16, 1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 2d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 1d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 2d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 3d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 3d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 3d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 0d_x_4d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 1d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 2d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 3d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 4d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 4d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 4d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 4d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 4d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 0d_x_5d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 1d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 2d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 3d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 4d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 5d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 5d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 5d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 5d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 5d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 5d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 0d_x_6d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
    const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 1d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 2d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 3d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 4d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 5d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 6d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
    const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF16();
  }
}

TEST(ADD_ND_F16, 6d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 6d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 6d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 6d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 6d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(ADD_ND_F16, 6d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF16();
    }
  }
}

TEST(ADD_ND_F32, 0d_x_0d) {
  BinaryElementwiseOperatorTester()
    .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
    .TestF32();
}

TEST(ADD_ND_F32, 1d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 0d_x_1d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 1d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 0d_x_2d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 1d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 2d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 2d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 2d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 0d_x_3d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 1d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 2d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 3d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 3d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 3d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 3d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 0d_x_4d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 1d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 2d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 3d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 4d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 4d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 4d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 4d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 4d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 0d_x_5d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 1d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 2d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 3d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 4d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 5d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 5d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 5d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 5d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 5d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 5d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 0d_x_6d) {
  for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
    const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
    const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
    const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
    const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
    const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
    const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
    const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
    const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
    const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
    const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
    const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
    const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 1d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 1); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 2d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 2); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      const size_t input2_dim3 = input2_broadcast_dim3 ? 1 : kDim3;
      const size_t input2_dim4 = input2_broadcast_dim4 ? 1 : kDim4;
      const size_t input2_dim5 = input2_broadcast_dim5 ? 1 : kDim5;
      const size_t input2_dim6 = input2_broadcast_dim6 ? 1 : kDim6;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 3d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 3); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 4d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 4); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 5d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 5); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 6d_x_0d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
    const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
    const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
    const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
    const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
    const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
    const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
    const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
    const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
    const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
    const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
    const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
    BinaryElementwiseOperatorTester()
      .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
      .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
      .TestF32();
  }
}

TEST(ADD_ND_F32, 6d_x_1d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 1); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 6d_x_2d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 2); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const size_t input1_dim1 = input1_broadcast_dim1 ? 1 : kDim1;
      const size_t input1_dim2 = input1_broadcast_dim2 ? 1 : kDim2;
      const size_t input1_dim3 = input1_broadcast_dim3 ? 1 : kDim3;
      const size_t input1_dim4 = input1_broadcast_dim4 ? 1 : kDim4;
      const size_t input1_dim5 = input1_broadcast_dim5 ? 1 : kDim5;
      const size_t input1_dim6 = input1_broadcast_dim6 ? 1 : kDim6;
      const size_t input2_dim1 = input2_broadcast_dim1 ? 1 : kDim1;
      const size_t input2_dim2 = input2_broadcast_dim2 ? 1 : kDim2;
      BinaryElementwiseOperatorTester()
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 6d_x_3d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 3); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 6d_x_4d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 4); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 6d_x_5d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 5); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}

TEST(ADD_ND_F32, 6d_x_6d) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << 6); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << 6); bm2++) {
      const bool input1_broadcast_dim1 = bm1 & (uint32_t(1) << 0);
      const bool input1_broadcast_dim2 = bm1 & (uint32_t(1) << 1);
      const bool input1_broadcast_dim3 = bm1 & (uint32_t(1) << 2);
      const bool input1_broadcast_dim4 = bm1 & (uint32_t(1) << 3);
      const bool input1_broadcast_dim5 = bm1 & (uint32_t(1) << 4);
      const bool input1_broadcast_dim6 = bm1 & (uint32_t(1) << 5);
      const bool input2_broadcast_dim1 = bm2 & (uint32_t(1) << 0);
      const bool input2_broadcast_dim2 = bm2 & (uint32_t(1) << 1);
      const bool input2_broadcast_dim3 = bm2 & (uint32_t(1) << 2);
      const bool input2_broadcast_dim4 = bm2 & (uint32_t(1) << 3);
      const bool input2_broadcast_dim5 = bm2 & (uint32_t(1) << 4);
      const bool input2_broadcast_dim6 = bm2 & (uint32_t(1) << 5);
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
        .operation_type(BinaryElementwiseOperatorTester::OperationType::Add)
        .input1_shape({input1_dim6, input1_dim5, input1_dim4, input1_dim3, input1_dim2, input1_dim1})
        .input2_shape({input2_dim6, input2_dim5, input2_dim4, input2_dim3, input2_dim2, input2_dim1})
        .iterations(1)
        .TestF32();
    }
  }
}
