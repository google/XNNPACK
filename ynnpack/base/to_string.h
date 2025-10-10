// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_TO_STRING_H_
#define XNNPACK_YNNPACK_BASE_TO_STRING_H_

#include <iostream>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

const char* to_string(ynn_unary_operator op);
const char* to_string(ynn_binary_operator op);
const char* to_string(ynn_reduce_operator op);

inline std::ostream& operator<<(std::ostream& os, ynn_unary_operator op) {
  return os << to_string(op);
}
inline std::ostream& operator<<(std::ostream& os, ynn_binary_operator op) {
  return os << to_string(op);
}
inline std::ostream& operator<<(std::ostream& os, ynn_reduce_operator op) {
  return os << to_string(op);
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TO_STRING_H_
