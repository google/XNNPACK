// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
#define XNNPACK_YNNPACK_SUBGRAPH_DOT_H_

#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Returns true if dots of type uint8 x `b_type` are faster than dots of type
// int8 x `b_type`.
bool prefer_uint8_dot(ynn_type b_type);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
