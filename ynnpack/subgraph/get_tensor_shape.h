// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_GET_TENSOR_SHAPE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_GET_TENSOR_SHAPE_H_

#include <cstdint>

#include "ynnpack/base/span.h"
#include "ynnpack/include/ynnpack.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace slinky {
class eval_context;
}  // namespace slinky

namespace ynn {

void implement_get_tensor_shape(ynn::span<const slinky::expr> extents,
                                ynn::span<const int32_t> axes, bool reshape_1d,
                                ynn_type type, const slinky::raw_buffer& output,
                                slinky::eval_context* ctx = nullptr);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_GET_TENSOR_SHAPE_H_
