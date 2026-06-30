// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include <cstdint>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

ynn_status define_constant_of_type(ynn_subgraph_t subgraph, float value,
                                   ynn_type type, uint32_t& id) {
  uint32_t fp32_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, value, fp32_id));
  if (type == ynn_type_fp32) {
    id = fp32_id;
    return ynn_status_success;
  }
  return ynn_define_convert_v2(subgraph, fp32_id, type, &id, 0);
}

}  // namespace

ynn_status define_average_pool_2d(ynn_subgraph_t subgraph, uint32_t input_id,
                                  ynn_type type, bool padding_same,
                                  size_t filter_height, size_t filter_width,
                                  size_t stride_height, size_t stride_width,
                                  uint32_t& output_id) {
  uint32_t norm_id = YNN_INVALID_VALUE_ID;

  const int32_t stencil_axes[] = {1, 2};
  const int32_t new_axes[] = {-3, -2};
  const size_t stencil_dims[] = {filter_height, filter_width};
  const size_t stencil_strides[] = {stride_height, stride_width};
  const size_t stencil_dilations[] = {1, 1};

  if (padding_same) {
    // The normalization we need to divide by varies when we hit the padding.
    // To handle this, we compute a pooled sum of 1s, padded by 0s.
    float one_val = 1.0f;
    const size_t ones_dims[] = {1, 1, 1, 1};
    uint32_t ones_fp32_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_tensor(subgraph, ynn_type_fp32, 4, ones_dims,
                                          &one_val, YNN_VALUE_FLAG_COPY_DATA,
                                          &ones_fp32_id));

    uint32_t ones_id = ones_fp32_id;
    if (type != ynn_type_fp32) {
      ones_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(ynn_define_convert_v2(subgraph, ones_fp32_id, type,
                                                &ones_id, 0));
    }

    uint32_t ones_broadcasted_id = YNN_INVALID_VALUE_ID;
    const int32_t xy[] = {1, 2};
    YNN_RETURN_IF_ERROR(ynn_define_broadcast_like(
        subgraph, 2, xy, ones_id, input_id, &ones_broadcasted_id, 0));

    uint32_t ones_stencil_id = YNN_INVALID_VALUE_ID;
    uint32_t ones_padding_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(
        define_constant_of_type(subgraph, 0.0f, type, ones_padding_id));

    YNN_RETURN_IF_ERROR(ynn_define_stencil_copy(
        subgraph, /*num_stencils=*/2, stencil_axes, new_axes, stencil_dims,
        stencil_strides, stencil_dilations, ones_broadcasted_id,
        ones_padding_id, &ones_stencil_id, /*flags=*/0));

    const int32_t reduce_axes[] = {3, 4};
    YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, 2,
                                          reduce_axes, ones_stencil_id,
                                          YNN_INVALID_VALUE_ID, &norm_id, 0));
  } else {
    float norm_val = static_cast<float>(filter_height * filter_width);
    YNN_RETURN_IF_ERROR(
        define_constant_of_type(subgraph, norm_val, type, norm_id));
  }

  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  uint32_t padding_id = YNN_INVALID_VALUE_ID;
  if (padding_same) {
    YNN_RETURN_IF_ERROR(
        define_constant_of_type(subgraph, 0.0f, type, padding_id));
  }

  YNN_RETURN_IF_ERROR(ynn_define_stencil_copy(
      subgraph, /*num_stencils=*/2, stencil_axes, new_axes, stencil_dims,
      stencil_strides, stencil_dilations, input_id, padding_id, &stencil_id,
      /*flags=*/0));

  uint32_t sum_id = YNN_INVALID_VALUE_ID;
  const int32_t reduce_axes[] = {3, 4};
  YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, 2,
                                        reduce_axes, stencil_id,
                                        YNN_INVALID_VALUE_ID, &sum_id, 0));

  uint32_t div_output_id = output_id;
  if (type != ynn_type_fp32) {
    div_output_id = YNN_INVALID_VALUE_ID;
  }

  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_divide, sum_id,
                                        norm_id, &div_output_id, 0));

  if (div_output_id != output_id) {
    YNN_RETURN_IF_ERROR(
        ynn_define_convert_v2(subgraph, div_output_id, type, &output_id, 0));
  }

  return ynn_status_success;
}

}  // namespace ynn
