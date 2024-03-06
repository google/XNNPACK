#include <xnnpack/log.h>
#include <xnnpack/reshape-helpers.h>

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
  bool changed = opdata->workspace_size > old_workspace_size;
  const struct xnn_value* input = &values[opdata->inputs[0]];
  output->shape.num_dims = input->shape.num_dims;
  for (size_t cur_dim = 0; cur_dim < input->shape.num_dims; cur_dim++) {
    const enum xnn_shape_inference_status shape_status = xnn_tensor_propagate_dimension(output, cur_dim, input->shape.dim[cur_dim]);
    if (shape_status == xnn_shape_inference_status_error) {
      return xnn_status_invalid_parameter;
    }
    if (shape_status == xnn_shape_inference_status_changed) {
      changed |= true;
    }
  }
  if (!changed) {
    return xnn_status_success;
  }
  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || opdata->workspace_size > old_workspace_size) {
    output->size = new_size;
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
  for (size_t i = 0; i < out_dims; ++i) {
    const size_t d0 = i >= dims0 ? 1 : input0->shape.dim[dims0 - i - 1];
    const size_t d1 = i >= dims1 ? 1 : input1->shape.dim[dims1 - i - 1];
    if (!(d0 == d1 || d0 == 1 || d1 == 1)) {
      xnn_log_error("Input dimensions %zu and %zu are not broadcastable.", d0, d1);
      return xnn_status_invalid_parameter;
    }

    size_t output_dim = d0;
    if (d1 > d0) {
      output_dim = d1;
    }
    size_t cur_dim = out_dims - i - 1;
    const enum xnn_shape_inference_status shape_status = xnn_tensor_propagate_dimension(output, cur_dim, output_dim);
    if (shape_status == xnn_shape_inference_status_error) {
      return xnn_status_invalid_parameter;
    }
  }

  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || old_workspace_size > opdata->workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}
