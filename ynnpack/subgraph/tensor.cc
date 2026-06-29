// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/tensor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "ynnpack/base/algorithm.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/log.h"
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
      buffer.mutable_dim(i).set_stride(stride);
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
      buffer.mutable_dim(i).set_min_extent(0, dims[num_dims - i - 1]);
      buffer.mutable_dim(i).set_fold_factor(slinky::dim::unfolded);
    }
    init_buffer_strides(buffer);
  }
}

namespace {

bool find_static_tensor(ynn_subgraph_t subgraph, ynn_type type, size_t rank,
                        const size_t* physical_dims, const void* data,
                        uint32_t* id_out) {
  for (const ynn_value& value : subgraph->values) {
    if (!value.is_valid()) continue;
    if (!value.is_static()) continue;
    if (value.type != type) continue;
    if (value.data->rank != rank) continue;
    if (!ynn::all_n(rank, [&](size_t i) {
          return value.data->dim(i).extent() == physical_dims[i];
        })) {
      continue;
    }
    // We don't check very large tensors for duplicates. We assume that callers
    // will not define duplicates of large tensors, and it may be expensive to
    // check them.
    constexpr size_t max_search_size_bytes = 1024 * 1024;
    const size_t size_bytes = value.data->size_bytes();
    if (size_bytes > max_search_size_bytes) {
      continue;
    }
    if (std::memcmp(value.data->base, data, size_bytes) != 0) {
      continue;
    }
    *id_out = value.id;
    return true;
  }
  return false;
}

}  // namespace

extern "C" {

ynn_status ynn_define_tensor(ynn_subgraph_t subgraph, enum ynn_type type,
                             size_t rank, const size_t* dims, const void* data,
                             uint32_t flags, uint32_t* id_out) {
  YNN_RETURN_IF_ERROR(validate_subgraph("define_tensor", subgraph));
  if (rank > YNN_MAX_TENSOR_RANK) {
    YNN_LOG_ERROR() << "rank " << rank << " exceeds YNN_MAX_TENSOR_RANK "
                    << YNN_MAX_TENSOR_RANK;
    return ynn_status_unsupported_parameter;
  }
  if (!id_out) {
    YNN_LOG_ERROR() << "id_out must be non-null";
    return ynn_status_invalid_parameter;
  }
  const bool is_external_input = (flags & YNN_VALUE_FLAG_EXTERNAL_INPUT) != 0;
  const bool is_external_output = (flags & YNN_VALUE_FLAG_EXTERNAL_OUTPUT) != 0;
  if (data) {
    if (is_external_input || is_external_output) {
      YNN_LOG_ERROR() << "data must be null for external tensors";
      return ynn_status_invalid_parameter;
    }
    if (rank > 0 && !dims) {
      YNN_LOG_ERROR()
          << "dims must be non-null for non-scalar external tensors";
      return ynn_status_invalid_parameter;
    }
  }

  if (!data && !is_external_input) {
    // We only use dims for static or input tensors.
    dims = nullptr;
  }

  size_t physical_dims[YNN_MAX_TENSOR_RANK];
  if (dims) {
    YNN_RETURN_IF_ERROR(to_physical_shape(type, rank, dims, physical_dims));
  }

  if (*id_out == YNN_INVALID_VALUE_ID && data) {
    assert(rank == 0 || dims);
    if (find_static_tensor(subgraph, type, rank, physical_dims, data, id_out)) {
      return ynn_status_success;
    }
  }

  ynn_value* value;
  if (*id_out != YNN_INVALID_VALUE_ID) {
    if (*id_out >= subgraph->external_value_ids) {
      YNN_LOG_ERROR() << "tensor ID " << *id_out
                      << " must be an external tensor ID";
      return ynn_status_invalid_parameter;
    }
    value = &subgraph->value(*id_out);
  } else {
    value = &subgraph->new_internal_value();
  }
  value->type = type;
  value->flags = flags;

  *id_out = value->id;
  if (!(data || value->is_external())) {
    // We don't care about the shape of this value, it will be inferred.
    return ynn_status_success;
  }

  if (dims) {
    for (size_t d = 0; d < rank; ++d) {
      // Any (logical) extent 1 dimensions of static values may be implicitly
      // broadcasted.
      const slinky::index_t logical = dims[rank - 1 - d];
      value->extents.push_back(logical == 1 ? slinky::expr{} : logical);
    }
  }

  if (data) {
    const bool copy_data = (flags & YNN_VALUE_FLAG_COPY_DATA) != 0;
    const bool copy_data_fp32 = (flags & YNN_VALUE_FLAG_COPY_DATA_FP32) != 0;

    if (copy_data || copy_data_fp32) {
      // Initialize a buffer just to get the dims.
      slinky::buffer<char, YNN_MAX_TENSOR_RANK> dims_buf(rank);
      init_buffer(dims_buf, ynn::type_size_bytes(type), rank,
                  dims ? physical_dims : nullptr, nullptr);

      value->data = slinky::raw_buffer::make(
          rank, ynn::type_size_bytes(type), dims_buf.dims,
          YNN_ALLOCATION_ALIGNMENT);

      if (copy_data_fp32 && type != ynn_type_fp32) {
        ynn::convert_n(static_cast<const float*>(data),
                       value->data->elem_count(), type, value->data->base);
      } else {
        std::memcpy(value->data->base, data, value->data->size_bytes());
      }
    } else {
      value->data = slinky::raw_buffer::make(rank);
      init_buffer(*value->data, ynn::type_size_bytes(type), rank,
                  dims ? physical_dims : nullptr, data);
    }
  } else {
    value->data = slinky::raw_buffer::make(rank);
    init_buffer(*value->data, ynn::type_size_bytes(type), rank,
                dims ? physical_dims : nullptr, nullptr);
    if (is_external_input) {
      value->symbol = subgraph->globals.symbols.insert_unique(value->name());
      value->extents.resize(rank);
      // Replace any constant 0 dimensions with dynamic extents.
      for (size_t d = 0; d < rank; ++d) {
        if (!dims || physical_dims[rank - 1 - d] == 0) {
          slinky::expr extent_d = buffer_max(value->symbol, d) + 1;
          if (d == 0) {
            int elem_count = ynn::type_element_count(type);
            if (elem_count != 1) {
              extent_d *= elem_count;
            }
          }
          value->extents[d] = extent_d;
        }
      }
    }
  }

  for (size_t d = 0; d < value->extents.size(); ++d) {
    if (!value->extents[d].defined() ||
        slinky::is_constant(value->extents[d], 1)) {
      value->data->mutable_dim(d) = slinky::dim::broadcast();
    }
  }

  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
