// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "include/experimental.h"
#include "include/xnnpack.h"
#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/xnnpack/utils.h"
#include "ynnpack/xnnpack/xnnpack.h"

extern "C" {

xnn_status xnn_define_tensor_value(xnn_subgraph_t subgraph,
                                   xnn_datatype datatype, size_t num_dims,
                                   const size_t* dims, const void* data,
                                   uint32_t external_id, uint32_t flags,
                                   uint32_t* id_out) {
  *id_out =
      external_id == XNN_INVALID_VALUE_ID ? YNN_INVALID_VALUE_ID : external_id;
  // YNNPACK interprets non-null dims for non-constant values to be static
  // shapes, so we can't pass them here unless the shape really is static.
  const size_t* xnn_dims =
      data || (flags & XNN_FLAG_SLINKY_STATIC_BOUNDS) != 0 ? dims : nullptr;
  ynn_status status = ynn_define_tensor_value(
      subgraph->ynn, ynn::type_from_xnn(datatype), num_dims, xnn_dims, data,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, ynn::value_flags_from_xnn(flags),
      id_out);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  ynn_value& value = subgraph->ynn->values[*id_out];
  if (value.is_external_input() && dims && !xnn_dims) {
    // XNNPACK does not require a `xnn_set_external_value_shape` call, so we may
    // never learn the shape if we removed it above. To work around this, we
    // pass null dims here, and make the first call to set the external shape
    // below.
    status = value.set_external_shape(num_dims, dims);
  }
  return ynn::xnn_status_from_ynn(status);
}

xnn_status xnn_define_quantized_tensor_value(
    xnn_subgraph_t subgraph, xnn_datatype datatype, int32_t zero_point,
    float scale, size_t num_dims, const size_t* dims, const void* data,
    uint32_t external_id, uint32_t flags, uint32_t* id_out) {
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  if (zero_point != 0) {
    ynn_status status = ynn_define_tensor_value(
        subgraph->ynn, ynn_type_int32, 0, nullptr, &zero_point,
        /*zero_point_id=*/YNN_INVALID_VALUE_ID,
        /*scale_id=*/YNN_INVALID_VALUE_ID, YNN_VALUE_FLAG_COPY_DATA,
        &zero_point_id);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
  }

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  if (scale != 1.0f) {
    ynn_status status = ynn_define_tensor_value(
        subgraph->ynn, ynn_type_fp32, 0, nullptr, &scale,
        /*zero_point_id=*/YNN_INVALID_VALUE_ID,
        /*scale_id=*/YNN_INVALID_VALUE_ID, YNN_VALUE_FLAG_COPY_DATA, &scale_id);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
  }

  *id_out =
      external_id == XNN_INVALID_VALUE_ID ? YNN_INVALID_VALUE_ID : external_id;
  // YNNPACK interprets non-null dims for non-constant values to be static
  // shapes, so we can't pass them here unless the shape really is static.
  const size_t* xnn_dims =
      data || (flags & XNN_FLAG_SLINKY_STATIC_BOUNDS) != 0 ? dims : nullptr;
  ynn_status status = ynn_define_tensor_value(
      subgraph->ynn, ynn::type_from_xnn(datatype), num_dims, xnn_dims, data,
      zero_point_id, scale_id, flags, id_out);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  ynn_value& value = subgraph->ynn->values[*id_out];
  if (value.is_external_input() && dims && !xnn_dims) {
    // XNNPACK does not require a `xnn_set_external_value_shape` call, so we may
    // never learn the shape if we removed it above. To work around this, we
    // pass null dims here, and make the first call to set the external shape
    // below.
    status = value.set_external_shape(num_dims, dims);
  }
  return ynn::xnn_status_from_ynn(status);
}

xnn_status xnn_define_channelwise_quantized_tensor_value(
    xnn_subgraph_t subgraph, xnn_datatype datatype, const float* scale,
    size_t num_dims, size_t channel_dim, const size_t* dims, const void* data,
    uint32_t external_id, uint32_t flags, uint32_t* id_out) {
  return xnn_define_channelwise_quantized_tensor_value_v2(
      subgraph, datatype,
      /*zero_point=*/0, scale, num_dims, channel_dim, dims, data, external_id,
      flags, id_out);
}

xnn_status xnn_validate_quantized_tensor(xnn_datatype datatype,
                                         int32_t zero_point, float scale,
                                         size_t num_dims, const size_t* dims) {
  return xnn_status_success;
}

xnn_status xnn_validate_channelwise_quantized_tensor(
    xnn_datatype datatype, int32_t zero_point, const float* scale,
    size_t num_dims, size_t channel_dim, const size_t* dims) {
  return xnn_status_success;
}

xnn_status xnn_define_channelwise_quantized_tensor_value_v2(
    xnn_subgraph_t subgraph, xnn_datatype datatype, int32_t zero_point,
    const float* scale, size_t num_dims, size_t channel_dim, const size_t* dims,
    const void* data, uint32_t external_id, uint32_t flags, uint32_t* id_out) {
  assert(data);
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  if (zero_point != 0) {
    ynn_status status = ynn_define_tensor_value(
        subgraph->ynn, ynn_type_int32, 0, nullptr, &zero_point,
        /*zero_point_id=*/YNN_INVALID_VALUE_ID,
        /*scale_id=*/YNN_INVALID_VALUE_ID, YNN_VALUE_FLAG_COPY_DATA,
        &zero_point_id);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
  }

  // It seems that this API is used in XNNPACK in two different ways:
  // - For batch matrix multiply filters, this is expected to be a broadcast
  // only for the non-channel dimension of the "matrix" argument.
  // - For convolution/depthwise filters, this is expected to be a broadcast in
  // every dimension except the channel dimension.
  // For convolutions/depthwise, the channel dimension is 0. So it seems like
  // maybe a consistent rule here is that every dimension after the channel
  // dimension is a broadcast, and every dimension before that is elementwise...
  size_t quantization_dims[YNN_MAX_TENSOR_RANK];
  std::copy_n(dims, channel_dim + 1, quantization_dims);
  if (channel_dim > 0) {
    // ... *except* for the one dimension before the channel dimension, which is
    // a broadcast?!?! This might only be true for XNN_FLAG_TRANSPOSE_B (or not)
    // which would be a headache to support here.
    quantization_dims[channel_dim - 1] = 1;
  }

  // XNNPACK copies the scale data from the caller, do the same here.
  const uint32_t scale_flags = YNN_VALUE_FLAG_COPY_DATA;
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn_define_tensor_value(
      subgraph->ynn, ynn_type_fp32, channel_dim + 1, quantization_dims, scale,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, scale_flags, &scale_id);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  *id_out =
      external_id == XNN_INVALID_VALUE_ID ? YNN_INVALID_VALUE_ID : external_id;
  return ynn::xnn_status_from_ynn(ynn_define_tensor_value(
      subgraph->ynn, ynn::type_from_xnn(datatype), num_dims, dims, data,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID, scale_id,
      ynn::value_flags_from_xnn(flags), id_out));
}

xnn_status xnn_define_blockwise_quantized_tensor_value_v2(
    xnn_subgraph_t subgraph, xnn_datatype datatype, int32_t zero_point,
    const void* scale, size_t num_dims, size_t channel_dim, size_t block_size,
    const size_t* dims, const void* data, uint32_t external_id, uint32_t flags,
    xnn_datatype scale_type, uint32_t* id_out) {
  // TODO: This is similar to xnn_define_channelwise_quantized_tensor_value_v2,
  // but with an extra `ynn_define_static_broadcast` and then
  // `ynn_define_static_reshape` of the scales.
  YNN_LOG_ERROR()
      << "Unsupported xnn_define_blockwise_quantized_tensor_value_v2";
  return xnn_status_deprecated;
}

xnn_status xnn_define_blockwise_quantized_tensor_value(
    xnn_subgraph_t subgraph, xnn_datatype datatype, int32_t zero_point,
    const uint16_t* scale, size_t num_dims, size_t channel_dim,
    size_t block_size, const size_t* dims, const void* data,
    uint32_t external_id, uint32_t flags, uint32_t* id_out) {
  return xnn_define_blockwise_quantized_tensor_value_v2(
      subgraph, datatype, zero_point, scale, num_dims, channel_dim, block_size,
      dims, data, external_id, flags, xnn_datatype_bf16, id_out);
}

xnn_status xnn_define_dynamically_quantized_tensor_value(
    xnn_subgraph_t subgraph, xnn_datatype datatype, size_t num_dims,
    size_t num_nonbatch_dims, const size_t* dims, uint32_t external_id,
    uint32_t flags, uint32_t* id_out) {
  dims = nullptr;
  // A dynamically quantized tensor value in XNNPACK is a tensor with a scale
  // and zero point id with rank `num_nonbatch_dims` and no data (not static).
  // The quantization data is computed later, when a convert node with this
  // tensor as an output is defined.
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn_define_tensor_value(
      subgraph->ynn, ynn_type_int32, num_nonbatch_dims,
      /*dims=*/nullptr, /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, /*flags=*/0, &zero_point_id);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_tensor_value(
      subgraph->ynn, ynn_type_fp32, num_nonbatch_dims,
      /*dims=*/nullptr, /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, /*flags=*/0, &scale_id);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  *id_out =
      external_id == XNN_INVALID_VALUE_ID ? YNN_INVALID_VALUE_ID : external_id;
  status = ynn_define_tensor_value(
      subgraph->ynn, ynn::type_from_xnn(datatype), num_dims, dims,
      /*data=*/nullptr, zero_point_id, scale_id, flags, id_out);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Store the number of non-batch dimensions where we can find it again.
  subgraph->num_nonbatch_axes[*id_out] = num_nonbatch_dims;

  return xnn_status_success;
}

}  // extern "C"
