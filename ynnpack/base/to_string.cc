// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/to_string.h"

#include "ynnpack/base/base.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

const char* to_string(enum ynn_unary_operator op) {
  switch (op) {
    case ynn_unary_abs:
      return "abs";
    case ynn_unary_round:
      return "round";
    case ynn_unary_ceil:
      return "ceil";
    case ynn_unary_convert:
      return "convert";
    case ynn_unary_exp:
      return "exp";
    case ynn_unary_expm1:
      return "expm1";
    case ynn_unary_erf:
      return "erf";
    case ynn_unary_floor:
      return "floor";
    case ynn_unary_log:
      return "log";
    case ynn_unary_log1p:
      return "log1p";
    case ynn_unary_negate:
      return "negate";
    case ynn_unary_reciprocal_square_root:
      return "reciprocal_square_root";
    case ynn_unary_square:
      return "square";
    case ynn_unary_square_root:
      return "square_root";
    case ynn_unary_tanh:
      return "tanh";
    case ynn_unary_cube_root:
      return "cube_root";
    case ynn_unary_sign:
      return "sign";
    case ynn_unary_sine:
      return "sine";
    case ynn_unary_cosine:
      return "cosine";
    case ynn_unary_sigmoid:
      return "sigmoid";
    case ynn_unary_hardswish:
      return "hardswish";
    case ynn_unary_invalid:
      return "invalid";
  }
  YNN_UNREACHABLE;
  return "unknown";
}

const char* to_string(enum ynn_binary_operator op) {
  switch (op) {
    case ynn_binary_add:
      return "add";
    case ynn_binary_divide:
      return "divide";
    case ynn_binary_multiply:
      return "multiply";
    case ynn_binary_subtract:
      return "subtract";
    case ynn_binary_copysign:
      return "copysign";
    case ynn_binary_squared_difference:
      return "squared_difference";
    case ynn_binary_min:
      return "minimum";
    case ynn_binary_max:
      return "maximum";
    case ynn_binary_pow:
      return "pow";
    case ynn_binary_leaky_relu:
      return "leaky_relu";
    case ynn_binary_invalid:
      return "invalid";
  }
  YNN_UNREACHABLE;
  return "unknown";
}

const char* to_string(enum ynn_reduce_operator op) {
  switch (op) {
    case ynn_reduce_sum:
      return "reduce_sum";
    case ynn_reduce_sum_squared:
      return "reduce_sum_squared";
    case ynn_reduce_max:
      return "reduce_max";
    case ynn_reduce_min:
      return "reduce_min";
    case ynn_reduce_min_max:
      return "reduce_min_max";
    case ynn_reduce_invalid:
      return "invalid";
  }
  YNN_UNREACHABLE;
  return "unknown";
}

}  // namespace ynn
