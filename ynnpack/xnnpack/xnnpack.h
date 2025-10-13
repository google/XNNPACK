// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_XNNPACK_XNNPACK_H_
#define XNNPACK_YNNPACK_XNNPACK_XNNPACK_H_

#include <cstddef>
#include <cstdint>
#include <map>

#include "ynnpack/include/ynnpack.h"

// We could just cast `xnn_subgraph_t` to `ynn_subgraph_t`, but this approach is
// both type safe and gives us a place to put some XNNPACK-specific metadata
// when constructing a YNNPACK subgraph.
struct xnn_subgraph {
  ynn_subgraph_t ynn;

  // When implementing `xnn_define_dynamically_quantized_tensor_value`, we don't
  // have anywhere to put this, so store it here.
  std::map<uint32_t, size_t> num_nonbatch_axes;
};

#endif  // XNNPACK_YNNPACK_XNNPACK_XNNPACK_H_
