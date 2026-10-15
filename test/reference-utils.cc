// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/reference-config.h"
#include "src/xnnpack/reference-utils.h"

namespace xnnpack {

TEST(ReferenceUtils, EuclideanDivBasic) {
  EXPECT_EQ(euclidean_div(7, 3), 2);
  EXPECT_EQ(euclidean_div(-7, 3), -3);
  EXPECT_EQ(euclidean_div(7, -3), -2);
  EXPECT_EQ(euclidean_div(-7, -3), 3);
}

TEST(ReferenceUtils, EuclideanDivZeroDivisor) {
  EXPECT_EQ(euclidean_div(10, 0), 0);
  EXPECT_EQ(euclidean_div(-10, 0), 0);
  EXPECT_EQ(euclidean_div(0, 0), 0);
  EXPECT_EQ(euclidean_div(10u, 0u), 0u);
}

TEST(ReferenceUtils, EuclideanDivUnitDivisors) {
  EXPECT_EQ(euclidean_div(10, 1), 10);
  EXPECT_EQ(euclidean_div(-10, 1), -10);
  EXPECT_EQ(euclidean_div(10, -1), -10);
  EXPECT_EQ(euclidean_div(-10, -1), 10);
}

TEST(ReferenceUtils, EuclideanDivOverflowEdge) {
  // INT32_MIN / -1 must not trigger hardware divide overflow (#DE / SIGFPE).
  const int32_t min32 = std::numeric_limits<int32_t>::min();
  EXPECT_EQ(euclidean_div(min32, -1), min32);

  const int16_t min16 = std::numeric_limits<int16_t>::min();
  EXPECT_EQ(euclidean_div(min16, static_cast<int16_t>(-1)), min16);

  const int8_t min8 = std::numeric_limits<int8_t>::min();
  EXPECT_EQ(euclidean_div(min8, static_cast<int8_t>(-1)), min8);
}

TEST(ReferenceUtils, EuclideanDivUnsigned) {
  EXPECT_EQ(euclidean_div(10u, 3u), 3u);
  EXPECT_EQ(euclidean_div(0u, 5u), 0u);
}

TEST(ReferenceUtils, EuclideanModBasic) {
  EXPECT_EQ(euclidean_mod(7, 3), 1);
  EXPECT_EQ(euclidean_mod(-7, 3), 2);
  EXPECT_EQ(euclidean_mod(7, -3), 1);
  EXPECT_EQ(euclidean_mod(-7, -3), 2);
}

TEST(ReferenceUtils, EuclideanModZeroDivisor) {
  EXPECT_EQ(euclidean_mod(10, 0), 0);
  EXPECT_EQ(euclidean_mod(-10, 0), 0);
  EXPECT_EQ(euclidean_mod(0, 0), 0);
  EXPECT_EQ(euclidean_mod(10u, 0u), 0u);
}

TEST(ReferenceUtils, EuclideanModUnitDivisors) {
  EXPECT_EQ(euclidean_mod(10, 1), 0);
  EXPECT_EQ(euclidean_mod(-10, 1), 0);
  EXPECT_EQ(euclidean_mod(10, -1), 0);
  EXPECT_EQ(euclidean_mod(-10, -1), 0);
}

TEST(ReferenceUtils, EuclideanModOverflowEdge) {
  // INT32_MIN % -1 must not trigger hardware divide overflow (#DE / SIGFPE).
  const int32_t min32 = std::numeric_limits<int32_t>::min();
  EXPECT_EQ(euclidean_mod(min32, -1), 0);

  const int16_t min16 = std::numeric_limits<int16_t>::min();
  EXPECT_EQ(euclidean_mod(min16, static_cast<int16_t>(-1)), 0);

  const int8_t min8 = std::numeric_limits<int8_t>::min();
  EXPECT_EQ(euclidean_mod(min8, static_cast<int8_t>(-1)), 0);
}

TEST(ReferenceUtils, EuclideanModUnsigned) {
  EXPECT_EQ(euclidean_mod(10u, 3u), 1u);
  EXPECT_EQ(euclidean_mod(0u, 5u), 0u);
}

TEST(ReferenceUtils, IntegerPowPositive) {
  EXPECT_EQ(integer_pow(2, 0), 1);
  EXPECT_EQ(integer_pow(2, 3), 8);
  EXPECT_EQ(integer_pow(-3, 3), -27);
  EXPECT_EQ(integer_pow(-3, 2), 9);
  EXPECT_EQ(integer_pow(0, 5), 0);
}

TEST(ReferenceUtils, IntegerPowNegativeExponents) {
  EXPECT_EQ(integer_pow(1, -1), 1);
  EXPECT_EQ(integer_pow(1, -50), 1);
  EXPECT_EQ(integer_pow(1, std::numeric_limits<int32_t>::min()), 1);

  EXPECT_EQ(integer_pow(-1, -1), -1);
  EXPECT_EQ(integer_pow(-1, -2), 1);
  EXPECT_EQ(integer_pow(-1, -3), -1);
  EXPECT_EQ(integer_pow(-1, std::numeric_limits<int32_t>::min()), 1);

  EXPECT_EQ(integer_pow(2, -1), 0);
  EXPECT_EQ(integer_pow(2, -10), 0);
  EXPECT_EQ(integer_pow(2, std::numeric_limits<int32_t>::min()), 0);

  EXPECT_EQ(integer_pow(-2, -1), 0);
  EXPECT_EQ(integer_pow(-2, std::numeric_limits<int32_t>::min()), 0);

  EXPECT_EQ(integer_pow(0, -5), 0);
}

TEST(ReferenceUtils, ReferenceUnaryAbsInt32Min) {
  const struct xnn_unary_elementwise_config* config =
      xnn_init_unary_reference_config(xnn_unary_abs, xnn_datatype_int32,
                                      xnn_datatype_int32);
  ASSERT_NE(config, nullptr);

  const int32_t input[3] = {-5, 10, std::numeric_limits<int32_t>::min()};
  int32_t output[3] = {0, 0, 0};
  config->ukernel(3 * sizeof(int32_t), input, output, /*params=*/nullptr);

  EXPECT_EQ(output[0], 5);
  EXPECT_EQ(output[1], 10);
  EXPECT_EQ(output[2], std::numeric_limits<int32_t>::min());
}

TEST(ReferenceUtils, ReferenceUnaryNegateInt32Min) {
  const struct xnn_unary_elementwise_config* config =
      xnn_init_unary_reference_config(xnn_unary_negate, xnn_datatype_int32,
                                      xnn_datatype_int32);
  ASSERT_NE(config, nullptr);

  const int32_t input[3] = {-5, 10, std::numeric_limits<int32_t>::min()};
  int32_t output[3] = {0, 0, 0};
  config->ukernel(3 * sizeof(int32_t), input, output, /*params=*/nullptr);

  EXPECT_EQ(output[0], 5);
  EXPECT_EQ(output[1], -10);
  EXPECT_EQ(output[2], std::numeric_limits<int32_t>::min());
}

TEST(ReferenceUtils, ReferenceBinaryShiftLeftNegativeOperands) {
  const struct xnn_binary_elementwise_config* config =
      xnn_init_binary_reference_config(xnn_binary_shift_left,
                                       xnn_datatype_int32);
  ASSERT_NE(config, nullptr);

  const int32_t a[3] = {-1, -10, std::numeric_limits<int32_t>::min()};
  const int32_t b[3] = {4, 2, 1};
  int32_t output[3] = {0, 0, 0};
  config->op_ukernel(3 * sizeof(int32_t), a, b, output, /*params=*/nullptr);

  EXPECT_EQ(output[0], static_cast<int32_t>(static_cast<uint32_t>(-1) << 4));
  EXPECT_EQ(output[1], static_cast<int32_t>(static_cast<uint32_t>(-10) << 2));
  EXPECT_EQ(output[2], 0);
}

TEST(ReferenceUtils, ReferenceBinaryDivideOverflow) {
  const struct xnn_binary_elementwise_config* config =
      xnn_init_binary_reference_config(xnn_binary_divide,
                                       xnn_datatype_int32);
  ASSERT_NE(config, nullptr);

  const int32_t a[2] = {10, std::numeric_limits<int32_t>::min()};
  const int32_t b[2] = {0, -1};
  int32_t output[2] = {0, 0};
  config->op_ukernel(2 * sizeof(int32_t), a, b, output, /*params=*/nullptr);

  EXPECT_EQ(output[0], 0);
  EXPECT_EQ(output[1], std::numeric_limits<int32_t>::min());
}

TEST(ReferenceUtils, ReferenceBinaryModulusOverflow) {
  const struct xnn_binary_elementwise_config* config =
      xnn_init_binary_reference_config(xnn_binary_modulus,
                                       xnn_datatype_int32);
  ASSERT_NE(config, nullptr);

  const int32_t a[2] = {10, std::numeric_limits<int32_t>::min()};
  const int32_t b[2] = {0, -1};
  int32_t output[2] = {0, 0};
  config->op_ukernel(2 * sizeof(int32_t), a, b, output, /*params=*/nullptr);

  EXPECT_EQ(output[0], 0);
  EXPECT_EQ(output[1], 0);
}

}  // namespace xnnpack
