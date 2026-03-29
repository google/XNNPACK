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
#include <pthreadpool.h>

// We could just cast `xnn_subgraph_t` to `ynn_subgraph_t`, but this approach is
// both type safe and gives us a place to put some XNNPACK-specific metadata
// when constructing a YNNPACK subgraph.
struct xnn_subgraph {
  ynn_subgraph_t ynn;

  // When implementing `xnn_define_dynamically_quantized_tensor_value`, we don't
  // have anywhere to put this, so store it here.
  std::map<uint32_t, size_t> num_nonbatch_axes;
};

// We could just cast `xnn_runtime_t` to `ynn_runtime_t`, but this approach is
// both type safe and gives us a place to put a pthreadpool, which we can't
// easily adapt to a `ynn_threadpool_t`.
struct xnn_runtime {
  ynn_runtime_t ynn;

  pthreadpool_t pthreadpool = nullptr;
};

#endif  // XNNPACK_YNNPACK_XNNPACK_XNNPACK_H_
