// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_TENSOR_H_
#define XNNPACK_YNNPACK_SUBGRAPH_TENSOR_H_

#include <cassert>
#include <cstddef>

#include "ynnpack/include/ynnpack.h"
#include "slinky/runtime/buffer.h"

namespace ynn {

// Storing sub-byte datatypes is tricky. To deal with it, we store all types in
// byte-aligned types, the "physical" storage, that may contain more than one
// element. This is separate from the "logical" shape, which is the number of
// elements. For example, int4 data has a physical shape that has half as many
// indices as the same logical shape. This function converts from a logical
// shape to a physical shape. The physical shape is never exposed to the public
// API, which only deals with logical shapes.
ynn_status to_physical_shape(ynn_type type, size_t rank,
                             const size_t* logical_dims, size_t* physical_dims);

// Initialize a raw_buffer to point to the same memory as xnn_runtime_value.
void init_buffer_strides(slinky::raw_buffer& buffer);

// Initialize a raw_buffer to point to the same memory as xnn_runtime_value.
void init_buffer(slinky::raw_buffer& buffer, size_t elem_size, size_t num_dims,
                 const size_t* dims, const void* data);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_TENSOR_H_
