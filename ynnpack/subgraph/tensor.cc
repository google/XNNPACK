// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/tensor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace ynn {

ynn_status to_physical_shape(ynn_type type, size_t rank,
                             const size_t* logical_dims,
                             size_t* physical_dims) {
  const size_t element_count = type_element_count(type);

  size_t dense_dim = rank == 0 ? 1 : logical_dims[rank - 1];
  if (dense_dim % element_count != 0) {
    // The logical size of the dense dimension must be a multiple of the number
    // of elements in an instance of the type.
    return ynn_status_invalid_parameter;
  }

  std::copy_n(logical_dims, rank, physical_dims);
  if (rank > 0) {
    physical_dims[rank - 1] /= element_count;
  }

  return ynn_status_success;
}

// Initialize a raw_buffer to point to the same memory as xnn_runtime_value.
void init_buffer_strides(slinky::raw_buffer& buffer) {
  slinky::index_t stride = buffer.elem_size;
  for (size_t i = 0; i < buffer.rank; ++i) {
    if (buffer.dim(i) != slinky::dim::broadcast()) {
      buffer.dim(i).set_stride(stride);
      stride *= buffer.dim(i).extent();
    }
  }
}

// Initialize a raw_buffer to point to the same memory as xnn_runtime_value.
void init_buffer(slinky::raw_buffer& buffer, size_t elem_size, size_t num_dims,
                 const size_t* dims, const void* data) {
  assert(buffer.rank >= num_dims);
  buffer.rank = num_dims;
  buffer.elem_size = elem_size;
  buffer.base = const_cast<void*>(data);
  if (dims) {
    for (size_t i = 0; i < num_dims; ++i) {
      buffer.dim(i).set_min_extent(0, dims[num_dims - i - 1]);
      buffer.dim(i).set_fold_factor(slinky::dim::unfolded);
    }
    init_buffer_strides(buffer);
  }
}

extern "C" {

ynn_status ynn_define_tensor_value(ynn_subgraph_t subgraph, enum ynn_type type,
                                   size_t rank, const size_t* dims,
                                   const void* data, uint32_t zero_point_id,
                                   uint32_t scale_id, uint32_t flags,
                                   uint32_t* id_out) {
  ynn_value* value;
  if (*id_out != YNN_INVALID_VALUE_ID) {
    value = &subgraph->value(*id_out);
    assert(value->id == *id_out);
  } else {
    value = &subgraph->new_internal_value();
  }
  value->type = type;
  value->flags = flags;
  value->scale_id = scale_id;
  value->zero_point_id = zero_point_id;

  *id_out = value->id;
  if (!(data || value->is_external())) {
    // We don't care about the shape of this value, it will be inferred.
    return ynn_status_success;
  }

  size_t physical_dims[YNN_MAX_TENSOR_RANK];
  if (dims) {
    if (value->is_external_output()) {
      // We want to infer this later.
      dims = nullptr;
    } else {
      ynn_status status = to_physical_shape(type, rank, dims, physical_dims);
      if (status != ynn_status_success) {
        return status;
      }
      for (size_t d = 0; d < rank; ++d) {
        // Any (logical) extent 1 dimensions of static values may be implicitly
        // broadcasted.
        const slinky::index_t logical = dims[rank - 1 - d];
        const slinky::index_t physical = physical_dims[rank - 1 - d];
        value->extents.push_back(logical == 1 ? slinky::expr{} : physical);
      }
    }
  }

  value->data = slinky::raw_buffer::make(rank);
  init_buffer(*value->data, ynn::type_size_bytes(type), rank,
              dims ? physical_dims : nullptr, data);
  if (data) {
    if (flags & YNN_VALUE_FLAG_COPY_DATA) {
      // TODO: This makes an extra heap allocation of the raw_buffer structure.
      // It's small, but this is wasteful.
      value->data = slinky::raw_buffer::make_copy(*value->data);
    }
    // Don't allow static values to be interpreted as inputs/outputs.
    value->flags &=
        ~(YNN_VALUE_FLAG_EXTERNAL_INPUT | YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  } else if (value->is_external_input()) {
    value->symbol = subgraph->symbols.insert_unique(value->name());
    value->extents.resize(rank);
    // Replace any constant 0 dimensions with dynamic extents.
    for (size_t d = 0; d < rank; ++d) {
      if (!dims || physical_dims[rank - 1 - d] == 0) {
        value->extents[d] = buffer_max(value->symbol, d) + 1;
      }
    }
  }

  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
