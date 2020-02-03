// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


enum xnn_status xnn_define_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    size_t num_dims,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error("failed to create Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  switch (datatype) {
    case xnn_datatype_fp32:
    case xnn_datatype_fp16:
      break;
    default:
      xnn_log_error("failed to create Dense Tensor value: invalid data type (%d)", datatype);
      return xnn_status_unsupported_parameter;
  }

  struct xnn_value* value = subgraph->values + external_id;
  if (external_id == XNN_INVALID_VALUE_ID) {
    value = xnn_subgraph_new_internal_value(subgraph);
    if (value == NULL) {
      return xnn_status_out_of_memory;
    }
  }
  value->type = xnn_value_type_dense_tensor;
  value->datatype = datatype;
  value->shape.num_dims = num_dims;
  memcpy(value->shape.dim, dims, num_dims * sizeof(size_t));
  value->flags = flags;
  value->data = data;

  *id_out = value->id;
  return xnn_status_success;
}

size_t xnn_tensor_get_size(
  xnn_subgraph_t subgraph,
  uint32_t value_id)
{
  assert(value_id < subgraph->num_values);

  const struct xnn_value* value = subgraph->values + value_id;
  assert(value->type == xnn_value_type_dense_tensor);
  assert(value->datatype != xnn_datatype_invalid);

  size_t size = 0;
  switch (value->datatype) {
    case xnn_datatype_fp16:
      size = 2;
      break;
    case xnn_datatype_fp32:
      size = 4;
      break;
    case xnn_datatype_invalid:
      XNN_UNREACHABLE;
  }

  for (size_t i = 0; i < value->shape.num_dims; i++) {
    size *= value->shape.dim[i];
  }

  return size;
}
