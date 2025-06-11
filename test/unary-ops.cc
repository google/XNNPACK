// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "test/unary-ops.h"

#include "include/xnnpack.h"

const UnaryOpInfo* GetUnaryOpInfo(xnn_unary_operator op) {
  static Abs abs;
  static Clamp clamp;
  static Convert convert;
  static ELU elu;
  static Exp exp;
  static ApproxGELU approxgelu;
  static GELU gelu;
  static HardSwish hardswish;
  static LeakyReLU leaky_relu;
  static Log log;
  static Negate negate;
  static ReciprocalSquareRoot reciprocal_square_root;
  static Sigmoid sigmoid;
  static Square square;
  static SquareRoot square_root;
  static TanH tanh;
  static RoundToNearestEven bankers_rounding;
  static RoundUp ceiling;
  static RoundDown floor;
  static CubeRoot cube_root;
  static Cosine cosine;
  static Sine sine;
  static CountLeadingZeros count_leading_zeros;
  static BitwiseNot bitwise_not;
  static Popcount popcount;
  static Sign sign;

  switch (op) {
    case xnn_unary_abs:
      return &abs;
    case xnn_unary_approxgelu:
      return &approxgelu;
    case xnn_unary_bankers_rounding:
      return &bankers_rounding;
    case xnn_unary_ceiling:
      return &ceiling;
    case xnn_unary_clamp:
      return &clamp;
    case xnn_unary_convert:
      return &convert;
    case xnn_unary_elu:
      return &elu;
    case xnn_unary_exp:
      return &exp;
    case xnn_unary_floor:
      return &floor;
    case xnn_unary_gelu:
      return &gelu;
    case xnn_unary_hardswish:
      return &hardswish;
    case xnn_unary_leaky_relu:
      return &leaky_relu;
    case xnn_unary_log:
      return &log;
    case xnn_unary_negate:
      return &negate;
    case xnn_unary_reciprocal_square_root:
      return &reciprocal_square_root;
    case xnn_unary_sigmoid:
      return &sigmoid;
    case xnn_unary_square:
      return &square;
    case xnn_unary_square_root:
      return &square_root;
    case xnn_unary_tanh:
      return &tanh;
    case xnn_unary_cube_root:
      return &cube_root;
    case xnn_unary_cosine:
      return &cosine;
    case xnn_unary_sine:
      return &sine;
    case xnn_unary_count_leading_zeros:
      return &count_leading_zeros;
    case xnn_unary_bitwise_not:
      return &bitwise_not;
    case xnn_unary_popcount:
      return &popcount;
    case xnn_unary_sign:
      return &sign;
    case xnn_unary_invalid:
      return nullptr;
  }
  return nullptr;
}