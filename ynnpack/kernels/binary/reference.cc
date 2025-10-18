// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/binary/reference.h"

#include "ynnpack/include/ynnpack.h"

namespace ynn {

const binary_op_info* get_binary_op_info(ynn_binary_operator op) {
  static add add;
  static copysign copysign;
  static divide div;
  static max max;
  static min min;
  static multiply mul;
  static pow pow;
  static squared_difference squared_difference;
  static subtract sub;
  static leaky_relu leaky_relu;

  switch (op) {
    case ynn_binary_add:
      return &add;
    case ynn_binary_copysign:
      return &copysign;
    case ynn_binary_divide:
      return &div;
    case ynn_binary_max:
      return &max;
    case ynn_binary_min:
      return &min;
    case ynn_binary_multiply:
      return &mul;
    case ynn_binary_pow:
      return &pow;
    case ynn_binary_squared_difference:
      return &squared_difference;
    case ynn_binary_subtract:
      return &sub;
    case ynn_binary_leaky_relu:
      return &leaky_relu;
    case ynn_binary_invalid:
      return nullptr;
  }
  return nullptr;
}

}  // namespace ynn
