// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/subgraph.h"

enum xnn_status xnn_define_global_average_pooling_1d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];

  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];

  reduction_axes[0] = input_value->shape.num_dims - 2;

  return xnn_define_static_reduce(
    subgraph, xnn_reduce_mean, 1, reduction_axes, input_id,
    output_id, flags);
}

enum xnn_status xnn_define_global_average_pooling_2d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];

  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];

  reduction_axes[0] = input_value->shape.num_dims - 3;
  reduction_axes[1] = input_value->shape.num_dims - 2;

  return xnn_define_static_reduce(
    subgraph, xnn_reduce_mean, 2, reduction_axes, input_id,
    output_id, flags);
}
