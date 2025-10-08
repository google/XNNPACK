// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/unary/reference.h"

#include "ynnpack/include/ynnpack.h"

namespace ynn {

const unary_op_info* get_unary_op_info(ynn_unary_operator op) {
  static abs abs;
  static convert convert;
  static exp exp;
  static expm1 expm1;
  static erf erf;
  static log log;
  static log1p log1p;
  static negate negate;
  static reciprocal_square_root reciprocal_square_root;
  static sigmoid sigmoid;
  static square square;
  static square_root square_root;
  static tanh tanh;
  static round round;
  static ceil ceil;
  static floor floor;
  static cube_root cube_root;
  static sign sign;
  static sine sine;
  static cosine cosine;
  static hardswish hardswish;

  switch (op) {
    case ynn_unary_abs:
      return &abs;
    case ynn_unary_round:
      return &round;
    case ynn_unary_ceil:
      return &ceil;
    case ynn_unary_convert:
      return &convert;
    case ynn_unary_exp:
      return &exp;
    case ynn_unary_expm1:
      return &expm1;
    case ynn_unary_erf:
      return &erf;
    case ynn_unary_floor:
      return &floor;
    case ynn_unary_log:
      return &log;
    case ynn_unary_log1p:
      return &log1p;
    case ynn_unary_negate:
      return &negate;
    case ynn_unary_reciprocal_square_root:
      return &reciprocal_square_root;
    case ynn_unary_square:
      return &square;
    case ynn_unary_square_root:
      return &square_root;
    case ynn_unary_tanh:
      return &tanh;
    case ynn_unary_cube_root:
      return &cube_root;
    case ynn_unary_sign:
      return &sign;
    case ynn_unary_sine:
      return &sine;
    case ynn_unary_cosine:
      return &cosine;
    case ynn_unary_sigmoid:
      return &sigmoid;
    case ynn_unary_hardswish:
      return &hardswish;
    case ynn_unary_invalid:
      return nullptr;
  }
  return nullptr;
}

}  // namespace ynn
