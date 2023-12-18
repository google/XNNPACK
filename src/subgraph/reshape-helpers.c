#include <xnnpack/reshape-helpers.h>

enum xnn_status reshape_binary_elementwise_op(
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

  // Propagate input shape to output.
  const struct xnn_value* input0 = &values[input0_id];
  const struct xnn_value* input1 = &values[input1_id];
  bool changed = !(opdata->workspace_size > old_workspace_size);
  for (size_t cur_dim = 0; cur_dim < input0->shape.num_dims; cur_dim++) {
    const struct xnn_value* shape_value = input0;
    // handle broadcasting
    if (input0->shape.dim[cur_dim] == 1) {
      shape_value = input1;
    }
    const enum xnn_shape_inference_status shape_status = xnn_tensor_propagate_dimension(output, cur_dim, shape_value, cur_dim);
    if (shape_status == xnn_shape_inference_status_error) {
      return xnn_status_invalid_parameter;
    }
    changed |= true;
  }
  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || old_workspace_size > opdata->workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}
