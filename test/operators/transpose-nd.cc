// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "test/operators/transpose-operator-tester.h"

TEST(TRANSPOSE_ND_X8, transpose_1D) {
  TransposeOperatorTester().num_dims(1).shape({713}).perm({0}).TestX8();
}

TEST(TRANSPOSE_ND_X8, transpose_2D) {
  std::vector<size_t> perm{0, 1};
  do {
    TransposeOperatorTester().num_dims(2).shape({37, 113}).perm(perm).TestX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, transpose_3D) {
  std::vector<size_t> perm{0, 1, 2};
  do {
    TransposeOperatorTester().num_dims(3).shape({5, 7, 11}).perm(perm).TestX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, transpose_4D) {
  std::vector<size_t> perm{0, 1, 2, 3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5, 7, 11, 13})
        .perm(perm)
        .TestX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, transpose_5D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3, 5, 7, 11, 13})
        .perm(perm)
        .TestX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, transpose_6D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4, 5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2, 3, 5, 7, 11, 13})
        .perm(perm)
        .TestX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, transpose_6D_X24) {
  std::vector<size_t> perm{0, 1, 2, 4, 3, 5};
  do {
    // Prevent merging of the final two dimensions.
    if (perm[4] == 4) {
      continue;
    }
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2, 4, 5, 6, 7, 3})
        .perm(perm)
        .TestX8();
    // Force the element size to always be 24 bits.
  } while (std::next_permutation(perm.begin(), perm.end() - 1));
}

TEST(TRANSPOSE_ND_X16, transpose_1D) {
  TransposeOperatorTester().num_dims(1).shape({713}).perm({0}).TestX16();
}

TEST(TRANSPOSE_ND_X16, transpose_2D) {
  std::vector<size_t> perm{0, 1};
  do {
    TransposeOperatorTester().num_dims(2).shape({37, 113}).perm(perm).TestX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, transpose_3D) {
  std::vector<size_t> perm{0, 1, 2};
  do {
    TransposeOperatorTester()
        .num_dims(3)
        .shape({5, 7, 11})
        .perm(perm)
        .TestX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, transpose_4D) {
  std::vector<size_t> perm{0, 1, 2, 3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5, 7, 11, 13})
        .perm(perm)
        .TestX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, transpose_5D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3, 5, 7, 11, 13})
        .perm(perm)
        .TestX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, Run5D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3, 5, 7, 11, 13})
        .perm(perm)
        .TestX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, transpose_6D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4, 5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2, 3, 5, 7, 11, 13})
        .perm(perm)
        .TestX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, transpose_4D_copy) {
  TransposeOperatorTester()
      .num_dims(4)
      .shape({2, 2, 1, 1})
      .perm({0, 2, 3, 1})
      .TestX32();
}

TEST(TRANSPOSE_ND_X32, Zero_dim) {
  TransposeOperatorTester().num_dims(2).shape({7, 0}).perm({1, 0}).TestX32();
}

TEST(TRANSPOSE_ND_X32_2, transpose_1D_redundant_dim) {
  TransposeOperatorTester().num_dims(1).shape({1}).perm({0}).TestX32();
}

TEST(TRANSPOSE_ND_X32, transpose_1D) {
  TransposeOperatorTester().num_dims(1).shape({713}).perm({0}).TestX32();
}

TEST(TRANSPOSE_ND_X32, transpose_2D_all_dimensions_redundant) {
  TransposeOperatorTester().num_dims(2).shape({1, 1}).perm({1, 0}).TestX32();
}

TEST(TRANSPOSE_ND_X32, transpose_2D) {
  std::vector<size_t> perm{0, 1};
  do {
    TransposeOperatorTester().num_dims(2).shape({37, 113}).perm(perm).TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, transpose_3D_redundant_dimension) {
  TransposeOperatorTester()
      .num_dims(3)
      .shape({2, 1, 3})
      .perm({0, 2, 1})
      .TestX32();
}

TEST(TRANSPOSE_ND_X32, transpose_3D) {
  std::vector<size_t> perm{0, 1, 2};
  do {
    TransposeOperatorTester()
        .num_dims(3)
        .shape({5, 7, 11})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, transpose_4D) {
  std::vector<size_t> perm{0, 1, 2, 3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5, 7, 11, 13})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, transpose_5D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3, 5, 7, 11, 13})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, transpose_6D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4, 5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2, 3, 5, 7, 11, 13})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, transpose_6D_DIMS_1) {
  std::vector<size_t> perm{0, 1, 2, 3, 4, 5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({1, 1, 1, 2, 3, 4})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, transpose_1D) {
  TransposeOperatorTester().num_dims(1).shape({713}).perm({0}).TestX64();
}

TEST(TRANSPOSE_ND_X64, transpose_2D) {
  std::vector<size_t> perm{0, 1};
  do {
    TransposeOperatorTester().num_dims(2).shape({37, 113}).perm(perm).TestX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, transpose_3D) {
  std::vector<size_t> perm{0, 1, 2};
  do {
    TransposeOperatorTester()
        .num_dims(3)
        .shape({5, 7, 11})
        .perm(perm)
        .TestX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, transpose_4D) {
  std::vector<size_t> perm{0, 1, 2, 3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5, 7, 11, 13})
        .perm(perm)
        .TestX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, transpose_5D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3, 5, 7, 11, 13})
        .perm(perm)
        .TestX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, transpose_6D) {
  std::vector<size_t> perm{0, 1, 2, 3, 4, 5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2, 3, 5, 7, 11, 13})
        .perm(perm)
        .TestX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_HARDENING, null_op_out) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  EXPECT_EQ(xnn_create_transpose_nd_x32(0, nullptr),
            xnn_status_invalid_parameter);
}

TEST(TRANSPOSE_ND_HARDENING, invalid_reshape_params) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_create_transpose_nd_x32(0, &op), xnn_status_success);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t shape[2] = {2, 3};
  size_t perm[2] = {1, 0};

  // num_dims == 0
  EXPECT_EQ(xnn_reshape_transpose_nd_x32(op, 0, shape, perm, nullptr),
            xnn_status_invalid_parameter);
  // num_dims > XNN_MAX_TENSOR_DIMS
  EXPECT_EQ(xnn_reshape_transpose_nd_x32(op, XNN_MAX_TENSOR_DIMS + 1, shape,
                                         perm, nullptr),
            xnn_status_invalid_parameter);
  // shape == nullptr
  EXPECT_EQ(xnn_reshape_transpose_nd_x32(op, 2, nullptr, perm, nullptr),
            xnn_status_invalid_parameter);
  // perm == nullptr
  EXPECT_EQ(xnn_reshape_transpose_nd_x32(op, 2, shape, nullptr, nullptr),
            xnn_status_invalid_parameter);

  // overflow in total elements
  size_t overflow_shape[2] = {SIZE_MAX / 2 + 1, 3};
  EXPECT_EQ(xnn_reshape_transpose_nd_x32(op, 2, overflow_shape, perm, nullptr),
            xnn_status_out_of_memory);

  // duplicate perm
  size_t dup_perm[2] = {0, 0};
  EXPECT_EQ(xnn_reshape_transpose_nd_x32(op, 2, shape, dup_perm, nullptr),
            xnn_status_invalid_parameter);

  // perm >= num_dims
  size_t oob_perm[2] = {0, 2};
  EXPECT_EQ(xnn_reshape_transpose_nd_x32(op, 2, shape, oob_perm, nullptr),
            xnn_status_invalid_parameter);
}

TEST(TRANSPOSE_ND_HARDENING, invalid_setup_params) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_create_transpose_nd_x32(0, &op), xnn_status_success);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  float input[6] = {0.0f};
  float output[6] = {0.0f};

  // setup unreshaped operator
  EXPECT_EQ(xnn_setup_transpose_nd_x32(op, input, output),
            xnn_status_invalid_state);

  size_t shape[2] = {2, 3};
  size_t perm[2] = {1, 0};
  ASSERT_EQ(xnn_reshape_transpose_nd_x32(op, 2, shape, perm, nullptr),
            xnn_status_success);

  // null input/output
  EXPECT_EQ(xnn_setup_transpose_nd_x32(op, nullptr, output),
            xnn_status_invalid_parameter);
  EXPECT_EQ(xnn_setup_transpose_nd_x32(op, input, nullptr),
            xnn_status_invalid_parameter);
}
