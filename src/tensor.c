// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>

static void set_allocation_type(struct xnn_value* value)
{
  if (value->data != NULL) {
    value->allocation_type = xnn_allocation_type_static;
  } else if ((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0) {
    value->allocation_type = xnn_allocation_type_external;
  } else if ((value->flags & XNN_VALUE_FLAG_PERSISTENT) != 0) {
    value->allocation_type = xnn_allocation_type_persistent;
  } else {
    value->allocation_type = xnn_allocation_type_workspace;
  }
}

static void set_shape(struct xnn_value* value, size_t num_dims, const size_t* dims)
{
  value->shape.num_dims = num_dims;
  if (num_dims != 0) {
    memcpy(value->shape.dim, dims, num_dims * sizeof(size_t));
  }
  for (size_t i = 0; i < num_dims; i++) {
    const size_t original_dim = value->shape.dim[i];
    if (original_dim == 0) {
      // Dimension of 0 implies an unknown dimension.
      value->shape.minimum_dim[i] = 0;
      value->shape.maximum_dim[i] = SIZE_MAX;
    } else {
      value->shape.minimum_dim[i] = original_dim;
      value->shape.maximum_dim[i] = original_dim;
    }
  }
}

static enum xnn_status check_zero_point(
  enum xnn_datatype datatype,
  int32_t zero_point)
{
  switch (datatype) {
    case xnn_datatype_qcint4:
      if (zero_point < 0 || zero_point > 15) {
        xnn_log_error(
          "failed to create Quantized Dense Tensor value: invalid zero point %" PRId32" outside the [0, 15] range",
          zero_point);
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_qcint8:
    case xnn_datatype_qint8:
      if ((int32_t) (int8_t) zero_point != zero_point) {
        xnn_log_error(
          "failed to create Quantized Dense Tensor value: invalid zero point %" PRId32" outside the [-128, 127] range",
          zero_point);
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_quint8:
      if ((int32_t) (uint8_t) zero_point != zero_point) {
        xnn_log_error(
          "failed to create Quantized Dense Tensor value: invalid zero point %" PRId32" outside the [0, 255] range",
          zero_point);
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_qcint32:
    case xnn_datatype_qint32:
      if (zero_point != 0) {
        xnn_log_error(
          "failed to create Quantized Dense Tensor value: invalid non-zero zero point %" PRId32,
          zero_point);
        return xnn_status_invalid_parameter;
      }
      break;
    default:
      xnn_log_error("failed to create Quantized Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
      return xnn_status_unsupported_parameter;
  }

  return xnn_status_success;
}

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
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
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
      xnn_log_error("failed to create Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
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
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->flags = flags;
  value->data = (void*) (uintptr_t) data;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

enum xnn_status xnn_define_quantized_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    int32_t zero_point,
    float scale,
    size_t num_dims,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create Quantized Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Quantized Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  enum xnn_status status = xnn_validate_quantized_tensor(
      datatype,
      zero_point,
      scale,
      num_dims,
      dims);
  if (status != xnn_status_success) {
    return status;
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
  value->quantization.zero_point = zero_point;
  value->quantization.scale = scale;
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->flags = flags;
  value->data = (void*) (uintptr_t) data;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

enum xnn_status xnn_define_dynamically_quantized_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    size_t num_dims,
    size_t num_nonbatch_dims,
    const size_t* dims,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create Dynamically Quantized Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Dynamically Quantized Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create Dynamically Quantized Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  if (num_nonbatch_dims > num_dims) {
    xnn_log_error(
      "failed to create Dynamically Quantized Dense Tensor value: "
      "non batch dimensions %zu is greater than number of dimensions %zu",
      num_nonbatch_dims, num_dims);
    return xnn_status_invalid_parameter;
  }

  switch (datatype) {
    case xnn_datatype_qdint8:
      break;
    default:
      xnn_log_error("failed to create Dynamically Quantized Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
      return xnn_status_unsupported_parameter;
  }

  if ((flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0) {
    xnn_log_error(
        "failed to create Dynamically Quantized Dense Tensor value: "
        "external dynamically quantized tensors are not supported.");
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
  value->quantization.num_nonbatch_dims = num_nonbatch_dims;
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->flags = flags;
  value->data = NULL;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

enum xnn_status xnn_define_channelwise_quantized_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    const float* scale,
    size_t num_dims,
    size_t channel_dim,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  return xnn_define_channelwise_quantized_tensor_value_v2(
    subgraph, datatype,
    /*zero_point=*/0, scale,
    num_dims, channel_dim, dims, data,
    external_id, flags,
    id_out);
}

enum xnn_status xnn_validate_quantized_tensor(
    enum xnn_datatype datatype,
    int32_t zero_point,
    float scale,
    size_t num_dims,
    const size_t* dims)
{
  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create Quantized Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  enum xnn_status status = check_zero_point(datatype, zero_point);
  if (status != xnn_status_success) {
    return status;
  }

  if (scale <= 0.0f || !isnormal(scale)) {
    xnn_log_error(
      "failed to create Quantized Dense Tensor value with %.7g scale: scale must be finite, normalized, and positive",
      scale);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_validate_channelwise_quantized_tensor(
    enum xnn_datatype datatype,
    int32_t zero_point,
    const float* scale,
    size_t num_dims,
    size_t channel_dim,
    const size_t* dims)
{
  if (num_dims == 0) {
    xnn_log_error(
      "failed to create Channelwise Quantized Dense Tensor value: no channel dimension exists");
    return xnn_status_invalid_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create Channelwise Quantized Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  if (channel_dim >= num_dims) {
    xnn_log_error(
      "failed to create Channelwise Quantized Dense Tensor value: "
      "channel dimension index %zu is out of range for %zu-dimensional tensor",
      channel_dim, num_dims);
    return xnn_status_invalid_parameter;
  }

  enum xnn_status status = check_zero_point(datatype, zero_point);
  if (status != xnn_status_success) {
    return status;
  }

  switch (datatype) {
    case xnn_datatype_qcint4:
    case xnn_datatype_qcint8:
    case xnn_datatype_qcint32:
      break;
    default:
      xnn_log_error("failed to create Channelwise Quantized Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
      return xnn_status_unsupported_parameter;
  }

  const size_t channels = dims[channel_dim];
  for (size_t channel = 0; channel < channels; channel++) {
    if (scale[channel] <= 0.0f || !isnormal(scale[channel])) {
      xnn_log_error(
        "failed to create Channelwise Quantized Dense Tensor value with %.7g scale in channel #%zu: "
        "scale must be finite, normalized, and positive",
        scale[channel], channel);
      return xnn_status_invalid_parameter;
    }
  }
  return xnn_status_success;
}

enum xnn_status xnn_define_channelwise_quantized_tensor_value_v2(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    int32_t zero_point,
    const float* scale,
    size_t num_dims,
    size_t channel_dim,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create Channelwise Quantized Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Channelwise Quantized Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  enum xnn_status status = xnn_validate_channelwise_quantized_tensor(
      datatype,
      zero_point,
      scale,
      num_dims,
      channel_dim,
      dims);
  if (status != xnn_status_success) {
    return status;
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
  value->quantization.zero_point = zero_point;
  value->quantization.channelwise_scale = scale;
  value->quantization.channel_dimension = channel_dim;
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->flags = flags;
  value->data = (void*) (uintptr_t) data;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

size_t xnn_shape_multiply_all_dims(
  const struct xnn_shape shape[restrict XNN_MIN_ELEMENTS(1)])
{
  size_t batch_size = 1;
  for (size_t i = 0; i < shape->num_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

size_t xnn_shape_multiply_batch_dims(
  const struct xnn_shape shape[restrict XNN_MIN_ELEMENTS(1)],
  size_t num_nonbatch_dims)
{
  size_t batch_size = 1;
  for (size_t i = 0; i + num_nonbatch_dims < shape->num_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

size_t xnn_shape_multiply_non_channel_dims(
  const struct xnn_shape shape[restrict XNN_MIN_ELEMENTS(1)])
{
  size_t batch_size = 1;
  for (size_t i = 0; i + 1 < shape->num_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

size_t xnn_shape_multiply_leading_dims(
  const struct xnn_shape shape[restrict XNN_MIN_ELEMENTS(1)],
  size_t num_leading_dims)
{
  size_t batch_size = 1;
  for (size_t i = 0; i < num_leading_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

size_t xnn_shape_multiply_trailing_dims(
  const struct xnn_shape shape[1],
  size_t start_dim)
{
  size_t product = 1;
  for (size_t i = start_dim; i < shape->num_dims; i++) {
    product *= shape->dim[i];
  }
  return product;
}

size_t xnn_tensor_get_size(const struct xnn_value* value)
{
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
    case xnn_datatype_qcint4:
    case xnn_datatype_qdint8:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qcint8:
      size = 1;
      break;
    case xnn_datatype_qint32:
    case xnn_datatype_qcint32:
      size = 4;
      break;
    case xnn_datatype_invalid:
      XNN_UNREACHABLE;
  }

  size *= xnn_shape_multiply_all_dims(&value->shape);

  // Adjustments for nibbles, assume that we can't have sizes are byte-aligned (rounded up).
  if (value->datatype == xnn_datatype_qcint4) {
    size = round_up_po2(size, 2) >> 1;
  }

  return size;
}

size_t xnn_tensor_get_size_by_id(xnn_subgraph_t subgraph, uint32_t value_id)
{
  assert(value_id < subgraph->num_values);

  const struct xnn_value* value = subgraph->values + value_id;
  return xnn_tensor_get_size(value);
}

static bool tensor_dim_is_static(const struct xnn_value* value, uint32_t dim_index)
{
  return (value->shape.dim[dim_index] == value->shape.minimum_dim[dim_index] &&
          value->shape.dim[dim_index] == value->shape.maximum_dim[dim_index]);
}

bool xnn_tensor_shape_is_static(const struct xnn_value* value)
{
  for (size_t i = 0; i < value->shape.num_dims; i++) {
    if (!tensor_dim_is_static(value, i)) {
      return false;
    }
  }
  return true;
}

enum xnn_shape_inference_status xnn_tensor_propagate_dimension(
  struct xnn_value* to,
  uint32_t to_dim,
  size_t inferred_dim)
{
  assert(to_dim < to->shape.num_dims);

  // If inferred_dim is dynamic, then we don't have useful information to propagate.
  if (to->shape.dim[to_dim] == inferred_dim || inferred_dim == 0) {
    return xnn_shape_inference_status_no_change;
  }

  to->shape.dim[to_dim] = inferred_dim;
  if (inferred_dim > to->shape.maximum_dim[to_dim]) {
    to->shape.maximum_dim[to_dim] = inferred_dim;
  }
  return xnn_shape_inference_status_changed;
}
