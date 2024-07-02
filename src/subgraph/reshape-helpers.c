// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/reshape-helpers.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

enum xnn_status resize_unary_elementwise_output_tensor(
  const struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  size_t old_workspace_size,
  pthreadpool_t threadpool)
{
  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = &values[output_id];

  // Propagate input shape to output.
  const struct xnn_value* input = &values[opdata->inputs[0]];
  output->shape.num_dims = input->shape.num_dims;
  memcpy(&output->shape.dim[0], &input->shape.dim[0], input->shape.num_dims * sizeof(size_t));
  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || opdata->workspace_size > old_workspace_size) {
    output->size = new_size;
    if (output->datatype == xnn_datatype_qdint8) {
      // reallocation will use this to adjust memory needed for dynamic quant params
      output->quantization.dynamic_params_size = xnn_tensor_get_dynamic_quant_param_size(output);
    }
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

enum xnn_status resize_binary_elementwise_output_tensor(
  const struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  size_t old_workspace_size,
  pthreadpool_t threadpool)
{
  const uint32_t input0_id = opdata->inputs[0];
  const uint32_t input1_id = opdata->inputs[1];
  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = &values[output_id];


  const struct xnn_value* input0 = &values[input0_id];
  const struct xnn_value* input1 = &values[input1_id];
  const size_t dims0 = input0->shape.num_dims;
  const size_t dims1 = input1->shape.num_dims;
  const size_t out_dims = max(dims0, dims1);

  output->shape.num_dims = max(input0->shape.num_dims, input1->shape.num_dims);
  if (dims0 == 0) {
    output->shape.num_dims = input1->shape.num_dims;
    memcpy(&output->shape.dim[0], &input1->shape.dim[0], sizeof(size_t) * input1->shape.num_dims);
  } else if (dims1 == 0) {
    output->shape.num_dims = input0->shape.num_dims;
    memcpy(&output->shape.dim[0], &input0->shape.dim[0], sizeof(size_t) * input0->shape.num_dims);
  } else {
    for (size_t i = 0; i < out_dims; ++i) {
      const size_t d0 = i >= dims0 ? 1 : input0->shape.dim[dims0 - i - 1];
      const size_t d1 = i >= dims1 ? 1 : input1->shape.dim[dims1 - i - 1];
      if (!(d0 == d1 || d0 == 1 || d1 == 1)) {
        xnn_log_error("Input dimensions %zu and %zu are not broadcastable.", d0, d1);
        return xnn_status_invalid_parameter;
      }
      size_t output_dim = d0;
      size_t cur_dim = out_dims - i - 1;
      if (d0 == 0 || d1 == 0) {
        output_dim = 0;
      } else {
        output_dim = max(d0, d1);
      }
      output->shape.dim[cur_dim] = output_dim;
    }
  }

  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || old_workspace_size > opdata->workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}
