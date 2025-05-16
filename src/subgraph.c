// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/subgraph.h"

#include <alloca.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/fp16.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"

#ifndef XNN_ENABLE_SPARSE
  #error "XNN_ENABLE_SPARSE not defined"
#endif

enum xnn_status xnn_insert_clamp_node(xnn_subgraph_t subgraph, float output_min, float output_max, struct xnn_node *node) {
  uint32_t output_id = node->outputs[0];
  struct xnn_value* output_value = &subgraph->values[output_id];
  uint32_t new_id = XNN_INVALID_VALUE_ID;
  enum xnn_status status;
  size_t num_dims = output_value->shape.num_dims;
  size_t dims[XNN_MAX_TENSOR_DIMS];
  memcpy(dims, output_value->shape.dim, num_dims * sizeof(size_t));
  switch (output_value->datatype) {
    case xnn_datatype_quint8:
      status = xnn_define_quantized_tensor_value(
          subgraph, xnn_datatype_quint8, output_value->quantization.zero_point,
          output_value->quantization.scale, num_dims, dims, NULL,
          /*external_id=*/XNN_INVALID_VALUE_ID, /*flags=*/0, &new_id);
      break;
    case xnn_datatype_qint8:
      status = xnn_define_quantized_tensor_value(
          subgraph, xnn_datatype_qint8, output_value->quantization.zero_point,
          output_value->quantization.scale, num_dims, dims, NULL,
          /*external_id=*/XNN_INVALID_VALUE_ID, /*flags=*/0, &new_id);
      break;
    default:
      status = xnn_define_tensor_value(
          subgraph, output_value->datatype, num_dims, dims, NULL,
          /*external_id=*/XNN_INVALID_VALUE_ID, /*flags=*/0, &new_id);
      break;
  }
  if (status != xnn_status_success) {
    return status;
  }
  struct xnn_value* new_value = &subgraph->values[new_id];
  new_value->size = 0;
  node->outputs[0] = new_id;
  node->activation.output_min = -INFINITY;
  node->activation.output_max = INFINITY;
  union xnn_unary_params params;
  params.clamp.min = output_min;
  params.clamp.max = output_max;
  return xnn_define_unary(subgraph, xnn_unary_clamp, &params, new_id, output_id,
                          /*flags=*/0);
}

enum xnn_status xnn_insert_pack_lh_node(xnn_subgraph_t subgraph, uint32_t input_id, uint32_t *new_id) {
  const struct xnn_value *input = &subgraph->values[input_id];
  enum xnn_status status = xnn_status_uninitialized;
  switch (input->datatype) {
    case xnn_datatype_qint8: {
      struct xnn_shape input_shape = input->shape;
      status = xnn_define_quantized_tensor_value(
          subgraph, input->datatype, input->quantization.zero_point,
          input->quantization.scale, input_shape.num_dims, input_shape.dim,
          /*data=*/input->data,
          /*external_id=*/XNN_INVALID_VALUE_ID, /*flags=*/0, new_id);
      break;
    }
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      status = xnn_define_tensor_value(
          subgraph, input->datatype, 0, NULL, NULL,
          /*external_id=*/XNN_INVALID_VALUE_ID, /*flags=*/0, new_id);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }

  return xnn_define_pack_lh(subgraph, input_id, *new_id, /*flags=*/0);
}

enum xnn_status xnn_create_subgraph(
    uint32_t external_value_ids,
    uint32_t flags,
    xnn_subgraph_t* subgraph_out)
{
  struct xnn_subgraph* subgraph = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create subgraph: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_out_of_memory;

  subgraph = xnn_allocate_zero_memory(sizeof(struct xnn_subgraph));
  if (subgraph == NULL) {
    xnn_log_error("failed to allocate %zu bytes for subgraph descriptor", sizeof(struct xnn_subgraph));
    goto error;
  }

  subgraph->external_value_ids = external_value_ids;

  subgraph->values = xnn_allocate_zero_memory(external_value_ids * sizeof(struct xnn_value));
  if (subgraph->values == NULL) {
    xnn_log_error("failed to allocate %zu bytes for subgraph values",
      (size_t) external_value_ids * sizeof(struct xnn_value));
    goto error;
  }
  for (size_t i = 0; i < external_value_ids; i++) {
    subgraph->values[i].id = i;
  }
  subgraph->num_values = external_value_ids;
  subgraph->num_reserved_values = external_value_ids;

  // If we're using Slinky, don't inline packing functions as Slinky does that
  // automatically.
  if (flags & XNN_FLAG_SLINKY_ENABLED) {
    flags |= XNN_FLAG_DONT_INLINE_LHS_PACKING;
  }
  subgraph->flags = flags;

  *subgraph_out = subgraph;
  return xnn_status_success;

error:
  xnn_delete_subgraph(subgraph);
  return status;
}


struct xnn_value* xnn_subgraph_new_internal_value(xnn_subgraph_t subgraph)
{
  struct xnn_value* values = subgraph->values;
  const size_t size = subgraph->num_values;
  const size_t capacity = subgraph->num_reserved_values;
  if (capacity < size + 1) {
    const size_t new_capacity = max(min(capacity * 2, capacity + 512), capacity + 64);
    assert(new_capacity >= size + 1);
    values = xnn_reallocate_memory(values, new_capacity * sizeof(struct xnn_value));
    if (values == NULL) {
      xnn_log_error("failed to allocate %zu bytes for subgraph values",
        capacity * sizeof(struct xnn_value));
      return values;
    }

    memset(values + size, 0, (new_capacity - size) * sizeof(struct xnn_value));
    subgraph->num_reserved_values = new_capacity;
    subgraph->values = values;
  }
  subgraph->num_values = size + 1;
  struct xnn_value* new_value = values + size;
  new_value->id = size;
  return new_value;
}

void xnn_node_clear(struct xnn_node* node) {
  assert(node != NULL);
  memset(node, 0, sizeof(struct xnn_node));
}

void xnn_value_clear(struct xnn_value* value) {
  assert(value != NULL);
  memset(value, 0, sizeof(struct xnn_value));
}

void xnn_value_copy(
  struct xnn_value* dst_value,
  const struct xnn_value* src_value)
{
  // Note: Value ID stays unchanged

  dst_value->type = src_value->type;
  dst_value->datatype = src_value->datatype;
  dst_value->quantization = src_value->quantization;
  dst_value->shape = src_value->shape;
  dst_value->size = src_value->size;
  dst_value->allocation_type = src_value->allocation_type;
  dst_value->flags = src_value->flags;
  dst_value->data = src_value->data;
  dst_value->producer = src_value->producer;
  dst_value->first_consumer = src_value->first_consumer;
  dst_value->all_consumers_types_same = src_value->all_consumers_types_same;
  dst_value->num_consumers = src_value->num_consumers;
  dst_value->num_nchw_compatible_consumers = src_value->num_nchw_compatible_consumers;
  dst_value->layout = src_value->layout;
  dst_value->fp16_compatible = src_value->fp16_compatible;
  dst_value->fp16_id = src_value->fp16_id;
  dst_value->fp32_id = src_value->fp32_id;
  dst_value->fp16_temp_data = src_value->fp16_temp_data;
  dst_value->fp32_data = src_value->fp32_data;
  dst_value->gemm_config = src_value->gemm_config;
}

struct xnn_node* xnn_subgraph_new_node(xnn_subgraph_t subgraph)
{
  struct xnn_node* nodes = subgraph->nodes;
  const size_t size = subgraph->num_nodes;
  const size_t capacity = subgraph->num_reserved_nodes;

  if (capacity < size + 1) {
    const size_t new_capacity = max(min(capacity * 2, capacity + 512), capacity + 64);
    assert(new_capacity >= size + 1);
    nodes = xnn_reallocate_memory(nodes, new_capacity * sizeof(struct xnn_node));
    if (nodes == NULL) {
      xnn_log_error("failed to allocate %zu bytes for subgraph nodes",
        capacity * sizeof(struct xnn_node));
      return nodes;
    }

    memset(nodes + size, 0, (new_capacity - size) * sizeof(struct xnn_node));
    subgraph->num_reserved_nodes = new_capacity;
    subgraph->nodes = nodes;
  }
  subgraph->num_nodes = size + 1;
  struct xnn_node* new_node = nodes + size;
  new_node->id = size;
  return new_node;
}

enum xnn_status xnn_subgraph_add_nodes(xnn_subgraph_t subgraph, size_t num_nodes)
{
  struct xnn_node* nodes = subgraph->nodes;
  const size_t size = subgraph->num_nodes;
  const size_t capacity = subgraph->num_reserved_nodes;

  if (capacity < size + num_nodes) {
    const size_t new_capacity = max(min(capacity * 2, capacity + 512), capacity + max(num_nodes, 64));
    assert(new_capacity >= size + num_nodes);
    nodes = xnn_reallocate_memory(nodes, new_capacity * sizeof(struct xnn_node));
    if (nodes == NULL) {
      xnn_log_error("failed to allocate %zu bytes for subgraph nodes",
        capacity * sizeof(struct xnn_node));
      return xnn_status_out_of_memory;
    }

    subgraph->num_reserved_nodes = new_capacity;
    subgraph->nodes = nodes;
  }
  memset(subgraph->nodes + size, 0,
         (subgraph->num_reserved_nodes - size) * sizeof(struct xnn_node));
  subgraph->num_nodes = size + num_nodes;
  struct xnn_node* new_nodes = nodes + size;
  for (size_t i = 0; i < num_nodes; i++) {
    new_nodes[i].id = size + i;
  }

  return xnn_status_success;
}

void xnn_subgraph_analyze_consumers_and_producers(xnn_subgraph_t subgraph)
{
  xnn_log_debug("Called.");
  // Initialize producer/consumer fields to safe defaults.
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    value->producer = XNN_INVALID_NODE_ID;
    value->first_consumer = XNN_INVALID_NODE_ID;
    value->num_consumers = 0;
  }

  // Analyse Nodes' inputs and output and update Values' producer/consumer fields
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];

    if (node->type == xnn_node_type_invalid) {
      continue;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const uint32_t input_id = node->inputs[i];
      assert(input_id < subgraph->num_values);

      if (subgraph->values[input_id].num_consumers++ == 0) {
        assert(subgraph->values[input_id].first_consumer == XNN_INVALID_NODE_ID);
        subgraph->values[input_id].first_consumer = n;
        subgraph->values[input_id].all_consumers_types_same = true;
      } else {
        enum xnn_node_type first_consumer_type = subgraph->nodes[subgraph->values[input_id].first_consumer].type;
        subgraph->values[input_id].all_consumers_types_same &= (first_consumer_type == node->type);
      }
    }

    for (uint32_t o = 0; o < node->num_outputs; o++) {
      const uint32_t output_id = node->outputs[o];
      assert(output_id < subgraph->num_values);

      // Persistent values can be produced by multiple nodes, e.g. copy nodes writing to the same persistent value.
      assert(xnn_value_is_persistent(&subgraph->values[output_id]) ||
             subgraph->values[output_id].producer == XNN_INVALID_NODE_ID);
      subgraph->values[output_id].producer = n;
    }
  }

  // Count extra consumer for Values which are external outputs.
  // Remove unreferenced values.
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    if (xnn_value_is_external_output(value)) {
      value->num_consumers += 1;
    }
  }
}

#define XNN_LAYOUT_FLAG_COMPATIBLE_NCHW      1
#define XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW 2
#define XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC 4
#define XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER 8

static bool all_values_fp(xnn_subgraph_t subgraph, const struct xnn_node* node) {
  for (uint32_t i = 0; i < node->num_inputs; i++) {
    if (subgraph->values[node->inputs[i]].datatype != xnn_datatype_fp16 &&
        subgraph->values[node->inputs[i]].datatype != xnn_datatype_fp32) {
      return false;
    }
  }
  for (uint32_t i = 0; i < node->num_outputs; i++) {
    if (subgraph->values[node->outputs[i]].datatype != xnn_datatype_fp16 &&
        subgraph->values[node->outputs[i]].datatype != xnn_datatype_fp32) {
      return false;
    }
  }
  return true;
}

uint32_t xnn_check_nchw_compatibility(xnn_subgraph_t subgraph, struct xnn_node* node) {
  if (!all_values_fp(subgraph, node)) {
    if (node->type != xnn_node_type_invalid) {
      xnn_log_info(
          "Node %s compute type is incompatible with sparse inference",
          xnn_node_type_to_string(node->type));
    }
    return 0;
  }

  switch (node->type) {
    case xnn_node_type_fully_connected:
      return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
    case xnn_node_type_convolution_2d:
      // Supported cases:
      // - 1x1 convolution (no stride, no dilation, no padding, no groups)
      // - 3x3 stride-2 convolution (no dilation, padding 1 on each side, no groups, 3 input channels)
      if (node->params.convolution_2d.groups != 1) {
        xnn_log_info("Node %s groups (%" PRIu32 ") "
                     "is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.convolution_2d.groups);
        return 0;
      }
      if ((node->params.convolution_2d.dilation_height | node->params.convolution_2d.dilation_width) != 1) {
        xnn_log_info("Node %s dilation (height=%" PRIu32 ", width=%" PRIu32 ") "
                     "is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.convolution_2d.dilation_height,
                     node->params.convolution_2d.dilation_width);
        return 0;
      }
      if ((node->params.convolution_2d.kernel_height | node->params.convolution_2d.kernel_width) == 1) {
        if ((node->params.convolution_2d.input_padding_top | node->params.convolution_2d.input_padding_right |
             node->params.convolution_2d.input_padding_bottom | node->params.convolution_2d.input_padding_left) != 0) {
          xnn_log_info("Node %s (1x1 kernel) padding (top=%" PRIu32 ", right=%" PRIu32", bottom=%" PRIu32 ", left=%" PRIu32") "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.input_padding_top,
                       node->params.convolution_2d.input_padding_right,
                       node->params.convolution_2d.input_padding_bottom,
                       node->params.convolution_2d.input_padding_left);
          return 0;
        }
        if ((node->params.convolution_2d.subsampling_height | node->params.convolution_2d.subsampling_width) != 1) {
          xnn_log_info("Node %s (1x1 kernel) subsampling (height=%" PRIu32 ", width=%" PRIu32 ") "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.subsampling_height,
                       node->params.convolution_2d.subsampling_width);
          return 0;
        }
        return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
      } else if (node->params.convolution_2d.kernel_height == 3 && node->params.convolution_2d.kernel_width == 3) {
        if (node->params.convolution_2d.input_padding_top != 1 || node->params.convolution_2d.input_padding_right != 1 ||
            node->params.convolution_2d.input_padding_bottom != 1 || node->params.convolution_2d.input_padding_left != 1) {
          xnn_log_info("Node %s (3x3 kernel) padding (top=%" PRIu32 ", right=%" PRIu32 ", bottom=%" PRIu32 ", left=%" PRIu32 ") "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.input_padding_top,
                       node->params.convolution_2d.input_padding_right,
                       node->params.convolution_2d.input_padding_bottom,
                       node->params.convolution_2d.input_padding_left);
          return 0;
        }
        if ((node->params.convolution_2d.subsampling_height | node->params.convolution_2d.subsampling_width) != 2) {
          xnn_log_info("Node %s (3x3 kernel) subsampling (height=%" PRIu32 ", width=%" PRIu32 ") "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.subsampling_height,
                       node->params.convolution_2d.subsampling_width);
          return 0;
        }
        if (node->params.convolution_2d.group_input_channels != 3) {
          xnn_log_info("Node %s (3x3 kernel) input channels (%zu) "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.group_input_channels);
          return 0;
        }
        return XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW;
      }
      return 0;
    case xnn_node_type_depthwise_convolution_2d:
      // Supported cases:
      // - 3x3 stride-1 convolution (no dilation, padding 1 on each side)
      // - 3x3 stride-2 convolution (no dilation, padding 1 on each side)
      // - 5x5 stride-1 convolution (no dilation, padding 2 on each side)
      // - 5x5 stride-2 convolution (no dilation, padding 2 on each side)
      if ((node->params.depthwise_convolution_2d.dilation_height | node->params.depthwise_convolution_2d.dilation_width) != 1) {
        xnn_log_info("Node %s dilation (height=%" PRIu32 ", width=%" PRIu32 ") "
                     "is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.convolution_2d.dilation_height,
                     node->params.convolution_2d.dilation_width);
        return 0;
      }
      if (node->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
        xnn_log_info("Node %s flags (%" PRIu32 ") has padding incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->flags);
        return 0;
      }
      if (node->params.depthwise_convolution_2d.depth_multiplier != 1) {
        xnn_log_info("Node %s depth_multiplier (%" PRIu32 ") is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.depthwise_convolution_2d.depth_multiplier);
        return 0;
      }
      if (node->params.depthwise_convolution_2d.subsampling_height != node->params.depthwise_convolution_2d.subsampling_width) {
        xnn_log_info("Node %s subsampling (height=%" PRIu32 ", width=%" PRIu32 ") "
                     "is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.depthwise_convolution_2d.subsampling_height,
                     node->params.depthwise_convolution_2d.subsampling_width);
        return 0;
      }
      switch (node->params.depthwise_convolution_2d.subsampling_height) {
        case 1:
        case 2:
          break;
        default:
          xnn_log_info("Node %s subsampling_height (%" PRIu32 ") "
                       "is incompatible with sparse inference",
                        xnn_node_type_to_string(node->type),
                        node->params.depthwise_convolution_2d.subsampling_height);
          return 0;
      }
      if (node->params.depthwise_convolution_2d.kernel_height != node->params.depthwise_convolution_2d.kernel_width) {
         xnn_log_info("Node %s kernel (height=%" PRIu32 ", width=%" PRIu32 ") "
                      "is incompatible with sparse inference",
                      xnn_node_type_to_string(node->type),
                      node->params.depthwise_convolution_2d.kernel_height,
                      node->params.depthwise_convolution_2d.kernel_width);
        return 0;
      }
      switch (node->params.depthwise_convolution_2d.kernel_height) {
        case 3:
          if (node->params.depthwise_convolution_2d.input_padding_top == 1 &&
              node->params.depthwise_convolution_2d.input_padding_right == 1 &&
              node->params.depthwise_convolution_2d.input_padding_bottom == 1 &&
              node->params.depthwise_convolution_2d.input_padding_left == 1) {
            return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
          } else {
            xnn_log_info("Node %s (3x3 kernel) padding "
                         "(top=%" PRIu32 ", right=%" PRIu32 ", bottom=%" PRIu32 ", left=%" PRIu32 ") "
                         "is incompatible with sparse inference",
                         xnn_node_type_to_string(node->type),
                         node->params.depthwise_convolution_2d.input_padding_top,
                         node->params.depthwise_convolution_2d.input_padding_right,
                         node->params.depthwise_convolution_2d.input_padding_bottom,
                         node->params.depthwise_convolution_2d.input_padding_left);
            return 0;
          }
        case 5:
          if (node->params.depthwise_convolution_2d.input_padding_top == 2 &&
              node->params.depthwise_convolution_2d.input_padding_right == 2 &&
              node->params.depthwise_convolution_2d.input_padding_bottom == 2 &&
              node->params.depthwise_convolution_2d.input_padding_left == 2) {
            return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
          } else {
            xnn_log_info("Node %s (5x5 kernel) padding "
                         "(top=%" PRIu32 ", right=%" PRIu32 ", bottom=%" PRIu32 ", left=%" PRIu32 ") "
                         "is incompatible with sparse inference",
                         xnn_node_type_to_string(node->type),
                         node->params.depthwise_convolution_2d.input_padding_top,
                         node->params.depthwise_convolution_2d.input_padding_right,
                         node->params.depthwise_convolution_2d.input_padding_bottom,
                         node->params.depthwise_convolution_2d.input_padding_left);
            return 0;
          }
        default:
          return 0;
      }
    case xnn_node_type_depth_to_space_2d:
      return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
    case xnn_node_type_global_average_pooling_2d:
      return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
    case xnn_node_type_binary_elementwise:
      if (node->binary_operator != xnn_binary_add &&
          node->binary_operator != xnn_binary_multiply) {
        // TODO: We can probably handle any binary operator here?
        return false;
      }
      assert(node->num_inputs == 2);
      assert(node->num_outputs == 1);
      if (subgraph->values[node->inputs[0]].shape.num_dims != 4 ||
          subgraph->values[node->inputs[1]].shape.num_dims != 4)
      {
        xnn_log_info("Node %s inputs shape is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type));
        return 0;
      }

      if (subgraph->values[node->inputs[0]].data != NULL) {
        // Check that the first input is representable as either a scalar, or a vector
        size_t num_nonunit_dims = 0;
        for (uint32_t i = 0; i < subgraph->values[node->inputs[0]].shape.num_dims; i++) {
          if (subgraph->values[node->inputs[0]].shape.dim[i] != 1) {
            num_nonunit_dims += 1;
          }
        }
        if (num_nonunit_dims > 1) {
          return 0;
        }
      }

      if (subgraph->values[node->inputs[1]].data != NULL) {
        // Check that the second input is representable as either a scalar, or a vector
        size_t num_nonunit_dims = 0;
        for (uint32_t i = 0; i < subgraph->values[node->inputs[0]].shape.num_dims; i++) {
          if (subgraph->values[node->inputs[0]].shape.dim[i] != 1) {
            num_nonunit_dims += 1;
          }
        }
        if (num_nonunit_dims > 1) {
          return 0;
        }
      }

      return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
    case xnn_node_type_static_resize_bilinear_2d:
      if (subgraph->values[node->inputs[0]].shape.dim[1] > 1 &&
          subgraph->values[node->inputs[0]].shape.dim[2] > 1) {
        return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
      } else {
        xnn_log_info("Node %s inputs shape is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type));
        return 0;
      }
    case xnn_node_type_unary_elementwise:
      assert(node->num_inputs == 1);
      assert(node->num_outputs == 1);
      if (subgraph->values[node->inputs[0]].shape.num_dims == 4) {
        return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
      } else {
        xnn_log_info("Node %s inputs shape is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type));
        return 0;
      }
    case xnn_node_type_static_mean:
    case xnn_node_type_static_sum:
      if (subgraph->values[node->inputs[0]].shape.num_dims == 4) {
        return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
      } else {
        xnn_log_info("Node %s inputs shape is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type));
        return 0;
      }
    default:
      return false;
  }
}

void xnn_subgraph_rewrite_for_nchw(xnn_subgraph_t subgraph)
{
  // Convert parts of the subgraph to NCHW for sparse inference
  // Step 1: detect NCHW-compatible Nodes
  // Step 2: detect NCHW-compatible clusters (run connected components graph algorithm)
  // Step 3: check that all NCHW-compatible Values are consumed only by NCHW-compatible Nodes
  // Step 4: switch Values' layout to NCHW
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    node->layout_flags = xnn_check_nchw_compatibility(subgraph, node);
    xnn_log_debug("Node #%" PRIu32 ": %s (NCHW: %s, NHWC->NCHW: %s, NCHW->NHWC: %s)",
      n, xnn_node_type_to_string(node->type),
      node->layout_flags & XNN_LAYOUT_FLAG_COMPATIBLE_NCHW ? "yes" : "no",
      node->layout_flags & XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW ? "yes" : "no",
      node->layout_flags & XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC ? "yes" : "no");
  }

  // Run Shiloach-Vishkin connected components algorithm i.e. find all
  // XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC nodes and set them as cluster leaders
  // to all the producer nodes
  bool update = false;
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    node->cluster_leader = n;
    if (node->layout_flags & XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC) {
      for (uint32_t i = 0; i < node->num_inputs; i++) {
        const struct xnn_value* value = &subgraph->values[node->inputs[i]];
        if (value->data != NULL) {
          // Static data, skip this input value. Compatibility of this static input with NCHW layout was validated
          // during the initial NCHW compatibility check for the Node.
          continue;
        }
        if (xnn_value_is_external(value)) {
          // External value, invalid cluster
          node->layout_flags |= XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
          continue;
        }
        const uint32_t producer_id = value->producer;
        assert(producer_id != XNN_INVALID_NODE_ID);
        assert(producer_id < n);
        struct xnn_node* producer_node = &subgraph->nodes[producer_id];
        if ((producer_node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) != 0 &&
            (producer_node->layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) == 0)
        {
          producer_node->layout_flags &= ~XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
          if (producer_node->cluster_leader != node->cluster_leader) {
            producer_node->cluster_leader = node->cluster_leader = math_max_u32(producer_node->cluster_leader, node->cluster_leader);
            update = true;
          }
        } else {
          node->layout_flags |= XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
        }
      }
    }
  }
  // No NCHW2NHWC compatible nodes have been found thus the graph rewriting
  // practically cannot happen.
  if (!update) {
    return;
  }
  // Propagate the cluster leader to other nodes in the graph until all the
  // nodes in the cluster is not updated
  while (update) {
    update = false;
    for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
      struct xnn_node* node = &subgraph->nodes[n];
      if (node->layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) {
        continue;
      }

      if ((node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NCHW | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC)) == 0) {
        continue;
      }

      for (uint32_t i = 0; i < node->num_inputs; i++) {
        const struct xnn_value* value = &subgraph->values[node->inputs[i]];
        if (value->data != NULL) {
          // Static data, skip this input value. Compatibility of this static input with NCHW layout was validated
          // during the initial NCHW compatibility check for the Node.
          continue;
        }
        if (xnn_value_is_external(value)) {
          // External value, invalid cluster
          node->layout_flags |= XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
          continue;
        }
        const uint32_t producer_id = value->producer;
        assert(producer_id != XNN_INVALID_NODE_ID);
        assert(producer_id < n);
        struct xnn_node* producer_node = &subgraph->nodes[producer_id];
        if ((producer_node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) != 0 &&
            (producer_node->layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) == 0)
        {
          producer_node->layout_flags &= ~XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
          if (producer_node->cluster_leader != node->cluster_leader) {
            producer_node->cluster_leader = node->cluster_leader = math_max_u32(producer_node->cluster_leader, node->cluster_leader);
            update = true;
          }
        } else {
          node->layout_flags |= XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
        }
      }
    }
  }
  // Propagate XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER flags up to the cluster leaders
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    subgraph->nodes[node->cluster_leader].layout_flags |= node->layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
  }
  // Check that all Values consumed by NCHW-compatible cluster don't have NCHW-incompatible consumers
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if ((subgraph->nodes[node->cluster_leader].layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) != 0) {
      continue;
    }

    if ((node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) == 0) {
      continue;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      struct xnn_value* value = &subgraph->values[node->inputs[i]];
      if (value->data != NULL) {
        // Static data, skip this input value because it doesn't have a producer Node.
        continue;
      }
      assert(!xnn_value_is_external(value));
      value->num_nchw_compatible_consumers += 1;
    }
  }
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if ((subgraph->nodes[node->cluster_leader].layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) != 0) {
      continue;
    }

    if ((node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) == 0) {
      continue;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const struct xnn_value* value = &subgraph->values[node->inputs[i]];
      if (value->data != NULL) {
        // Static data, skip this input value because it doesn't have a producer Node.
        continue;
      }
      assert(!xnn_value_is_external(value));
      assert(value->num_nchw_compatible_consumers > 0);
      if (value->num_nchw_compatible_consumers != value->num_consumers) {
        subgraph->nodes[node->cluster_leader].layout_flags |= XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
      }
    }
  }
  // Evaluate if it is profitable to run the model as sparse:
  // - Compute the number of parameters and zeroes in 1x1 Convolution weights
  // - Disable sparse rewriting for clusters without 1x1 Convolutions (num_params == 0)
  //   or with less than 2/3rd of zeroes in 1x1 Convolution filters
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if ((subgraph->nodes[node->cluster_leader].layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) != 0) {
      continue;
    }

    if ((node->type == xnn_node_type_convolution_2d &&
        max(node->params.convolution_2d.kernel_height, node->params.convolution_2d.kernel_width) == 1)
        || node->type == xnn_node_type_fully_connected)
    {
      assert(node->num_inputs >= 2);

      const struct xnn_value* filter = &subgraph->values[node->inputs[1]];
      assert(filter->data != NULL);

      const size_t num_params = filter->shape.dim[0] * filter->shape.dim[filter->shape.num_dims - 1];
      subgraph->nodes[node->cluster_leader].num_params += num_params;

      size_t num_zeroes = 0;
      switch (filter->datatype) {
        case xnn_datatype_fp32: {
          const float* data = (const float*)filter->data;
          for (size_t i = 0; i < num_params; i++) {
            num_zeroes += (size_t)(data[i] == 0.0f);
          }
          break;
        }
        case xnn_datatype_fp16: {
          const xnn_float16* data = (const xnn_float16*)filter->data;
          for (size_t i = 0; i < num_params; i++) {
            num_zeroes += (size_t)(xnn_float16_is_zero(data[i]));
          }
          break;
        }
        default:
          XNN_UNREACHABLE;
      }
      xnn_log_debug("1x1 Convolution 2D Node #%" PRIu32 ": %zu / %zu sparsity", n, num_zeroes, num_params);
      subgraph->nodes[node->cluster_leader].num_zeroes += num_zeroes;
    }
  }
  bool use_nchw_layout = false;
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if ((subgraph->nodes[node->cluster_leader].layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) != 0) {
      continue;
    }

    if ((node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) == 0) {
      continue;
    }

    if (subgraph->nodes[node->cluster_leader].num_zeroes * 3 <= subgraph->nodes[node->cluster_leader].num_params * 2) {
      xnn_log_info("Node #%" PRIu32 ": sparse inference disabled: 1x1 Convolutions contain %zu / %zu zero weights",
        n, subgraph->nodes[node->cluster_leader].num_zeroes, subgraph->nodes[node->cluster_leader].num_params);
      continue;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      struct xnn_value* value = &subgraph->values[node->inputs[i]];
      if (value->data != NULL) {
        // Static data, skip this input value because it doesn't have a producer Node.
        continue;
      }
      assert(!xnn_value_is_external(value));
      assert(value->num_nchw_compatible_consumers > 0);
      assert(value->num_nchw_compatible_consumers == value->num_consumers);
      if (value->layout != xnn_layout_type_nchw) {
        value->layout = xnn_layout_type_nchw;
        xnn_log_info("set Value #%"PRIu32" layout to NCHW", node->inputs[i]);
        use_nchw_layout = true;
      }
    }
  }
  if (use_nchw_layout) {
    xnn_log_info("XNNPACK has switched to sparse inference mode!");
  }
}

static bool any_values_fp32(xnn_subgraph_t subgraph, const struct xnn_node* node) {
  for (uint32_t i = 0; i < node->num_inputs; i++) {
    if (subgraph->values[node->inputs[i]].datatype == xnn_datatype_fp32) {
      return true;
    }
  }
  for (uint32_t i = 0; i < node->num_outputs; i++) {
    if (subgraph->values[node->outputs[i]].datatype == xnn_datatype_fp32) {
      return true;
    }
  }
  return false;
}

static bool all_values_fp32_or_pfp32(xnn_subgraph_t subgraph,
                                     const struct xnn_node* node) {
  for (uint32_t i = 0; i < node->num_inputs; i++) {
    if (subgraph->values[node->inputs[i]].datatype != xnn_datatype_fp32 &&
        subgraph->values[node->inputs[i]].datatype != xnn_datatype_pfp32) {
      return false;
    }
  }
  for (uint32_t i = 0; i < node->num_outputs; i++) {
    if (subgraph->values[node->outputs[i]].datatype != xnn_datatype_fp32 &&
        subgraph->values[node->outputs[i]].datatype != xnn_datatype_pfp32) {
      return false;
    }
  }
  return true;
}

bool xnn_subgraph_rewrite_for_fp16(xnn_subgraph_t subgraph)
{
  xnn_log_info("Analyzing subgraph for FP16 compatibility");

  // Convert tensors and operators in the subgraph to FP16
  // 1. Check that all operators in the subgraph are supported in FP16.
  // 2. Indicate values that must be converted to FP16.
  // 3. Replace FP32 Values with FP16 Values as Nodes' inputs/outputs.
  // 4. Insert FP32->FP16 Convert Nodes for external FP32 inputs and FP16->FP32 Convert Nodes for external outputs.

  const uint32_t num_original_values = subgraph->num_values;

  // Check that all operators in the subgraph are supported in FP16, bail out on any unsupported one.
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if (node->type == xnn_node_type_invalid) {
      // Node was fused away, skip.
      continue;
    }

    if (!any_values_fp32(subgraph, node)) {
      xnn_log_warning("FP16 rewrite aborted: node #%" PRIu32 " (%s) is not FP32", n, xnn_node_type_to_string(node->type));
      return false;
    }
    switch (node->type) {
      case xnn_node_type_binary_elementwise:
      case xnn_node_type_unary_elementwise:
      case xnn_node_type_batch_matrix_multiply:
      case xnn_node_type_concatenate:
      case xnn_node_type_convert:
      case xnn_node_type_average_pooling_2d:
      case xnn_node_type_copy:
      case xnn_node_type_convolution_2d:
      case xnn_node_type_deconvolution_2d:
      case xnn_node_type_depthwise_convolution_2d:
      case xnn_node_type_depth_to_space_2d:
      case xnn_node_type_even_split:
      case xnn_node_type_fully_connected:
      case xnn_node_type_global_average_pooling_2d:
      case xnn_node_type_global_sum_pooling_2d:
      case xnn_node_type_max_pooling_2d:
      case xnn_node_type_softmax:
      case xnn_node_type_space_to_depth_2d:
      case xnn_node_type_static_constant_pad:
      case xnn_node_type_static_mean:
      case xnn_node_type_static_slice:
      case xnn_node_type_static_sum:
      case xnn_node_type_static_reshape:
      case xnn_node_type_static_resize_bilinear_2d:
      case xnn_node_type_static_transpose:
      case xnn_node_type_rope:
        break;
      case xnn_node_type_pack_lh:
        if (xnn_init_x16_pack_lh_config() != NULL) {
          break;
        }
      default:
        xnn_log_warning("FP16 rewrite aborted: node #%" PRIu32 " (%s) is not supported for FP16 inference",
          n, xnn_node_type_to_string(node->type));
        return false;
    }
  }

  // Annotate Values to be converted to FP16 as FP16-compatible.
  // Note that static weights in [Depthwise] Convolution, Fully Connected Nodes remain FP32,
  // they will be converted to FP16 during weight repacking when the operator is created.
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    switch (node->type) {
      case xnn_node_type_deconvolution_2d:
      case xnn_node_type_depthwise_convolution_2d:
        subgraph->values[node->inputs[0]].fp16_compatible = true;
        subgraph->values[node->outputs[0]].fp16_compatible = true;
        break;
      case xnn_node_type_convolution_2d:
        if (subgraph->values[node->inputs[0]].datatype == xnn_datatype_qdint8) {
          subgraph->values[node->outputs[0]].fp16_compatible = true;
        } else {
          subgraph->values[node->inputs[0]].fp16_compatible = true;
          subgraph->values[node->outputs[0]].fp16_compatible = true;
        }
        break;
      case xnn_node_type_fully_connected:
        if (subgraph->values[node->inputs[0]].datatype == xnn_datatype_qdint8 ||
            subgraph->values[node->inputs[0]].datatype ==
                xnn_datatype_qduint8 ||
            subgraph->values[node->inputs[0]].datatype == xnn_datatype_qpint8) {
          subgraph->values[node->outputs[0]].fp16_compatible = true;
        } else if ((subgraph->values[node->inputs[0]].datatype ==
                        xnn_datatype_fp32 ||
                    subgraph->values[node->inputs[0]].datatype ==
                        xnn_datatype_pfp32) &&
                   subgraph->values[node->outputs[0]].datatype ==
                       xnn_datatype_fp32) {
          subgraph->values[node->inputs[0]].fp16_compatible = true;
          subgraph->values[node->outputs[0]].fp16_compatible = true;
          if (subgraph->values[node->inputs[1]].datatype == xnn_datatype_fp32) {
            subgraph->values[node->inputs[1]].fp16_compatible = true;
          }
          if (node->num_inputs > 2 &&
              subgraph->values[node->inputs[2]].datatype == xnn_datatype_fp32) {
            subgraph->values[node->inputs[2]].fp16_compatible = true;
          }
        } else if (subgraph->values[node->inputs[0]].datatype ==
                       xnn_datatype_pfp32 &&
                   subgraph->values[node->inputs[1]].datatype ==
                       xnn_datatype_fp32 &&
                   subgraph->values[node->outputs[0]].datatype ==
                       xnn_datatype_fp32) {
          subgraph->values[node->inputs[0]].fp16_compatible = true;
          subgraph->values[node->inputs[1]].fp16_compatible = true;
          subgraph->values[node->outputs[0]].fp16_compatible = true;
          if (node->num_inputs > 2 &&
              subgraph->values[node->inputs[2]].datatype == xnn_datatype_fp32) {
            subgraph->values[node->inputs[2]].fp16_compatible = true;
          }
        } else if (all_values_fp32_or_pfp32(subgraph, node)) {
          subgraph->values[node->inputs[0]].fp16_compatible = true;
          subgraph->values[node->outputs[0]].fp16_compatible = true;
        } else {
          xnn_log_warning(
              "FP16 rewrite aborted: node #%" PRIu32
              " (%s). Invalid compute type (input=%s, weights=%s, output=%s)",
              n, xnn_node_type_to_string(node->type),
              xnn_datatype_to_string(
                  subgraph->values[node->inputs[0]].datatype),
              xnn_datatype_to_string(
                  subgraph->values[node->inputs[1]].datatype),
              xnn_datatype_to_string(
                  subgraph->values[node->outputs[0]].datatype));
          return false;
        }
        break;
      case xnn_node_type_convert:
        if (subgraph->values[node->inputs[0]].datatype == xnn_datatype_fp32) {
          subgraph->values[node->inputs[0]].fp16_compatible = true;
        }
        if (subgraph->values[node->outputs[0]].datatype == xnn_datatype_fp32) {
          subgraph->values[node->outputs[0]].fp16_compatible = true;
        }
        break;
      case xnn_node_type_pack_lh:
        if (subgraph->values[node->inputs[0]].datatype == xnn_datatype_fp32) {
          subgraph->values[node->inputs[0]].fp16_compatible = true;
        }
        break;
      default:
        for (uint32_t i = 0; i < node->num_inputs; i++) {
          switch (subgraph->values[node->inputs[i]].datatype) {
            case xnn_datatype_fp32:
            case xnn_datatype_pfp32:
              subgraph->values[node->inputs[i]].fp16_compatible = true;
              break;
            default:
              break;
          }
        }
        for (uint32_t o = 0; o < node->num_outputs; o++) {
          switch (subgraph->values[node->outputs[o]].datatype) {
            case xnn_datatype_fp32:
            case xnn_datatype_pfp32:
              subgraph->values[node->outputs[o]].fp16_compatible = true;
              break;
            default:
              break;
          }
        }
        break;
    }
  }

  // Attempt to allocate memory for static values and external input/outputs.
  // The FP16 rewrite is cleanly aborted on failure.
  for (uint32_t n = 0; n < num_original_values; n++) {
    struct xnn_value* value = &subgraph->values[n];
    value->fp16_id = XNN_INVALID_VALUE_ID;
    value->fp32_id = XNN_INVALID_VALUE_ID;
    if (value->fp16_compatible) {
      assert(value->datatype == xnn_datatype_fp32 ||
             value->datatype == xnn_datatype_pfp32);
      if (xnn_value_is_static(value)) {
        assert(value->producer == XNN_INVALID_NODE_ID);
        const size_t fp16_size = xnn_tensor_get_size_by_id(subgraph, n) / 2 + XNN_EXTRA_BYTES;
        value->fp16_temp_data = xnn_allocate_zero_memory(fp16_size);
        if (value->fp16_temp_data == NULL) {
          xnn_log_error("failed to allocate %zu bytes for fp16 tensor data", (size_t)fp16_size);
          goto error;
        }
      } else if (xnn_value_is_external(value)) {
        struct xnn_value* fp16_value = xnn_subgraph_new_internal_value(subgraph);
        if (fp16_value == NULL) {
          xnn_log_error("FP16 rewrite aborted: failed to allocate value for external input/output");
          goto error;
        } else {
          // Recompute value due to potential reallocation in xnn_subgraph_new_internal_value
          value = &subgraph->values[n];
          xnn_value_copy(fp16_value, value);
          switch (value->datatype) {
            case xnn_datatype_fp32:
              fp16_value->datatype = xnn_datatype_fp16;
              break;
            case xnn_datatype_pfp32:
              fp16_value->datatype = xnn_datatype_pfp16;
              break;
            default:
              XNN_UNREACHABLE;
          }
          // Clear external input/output flags
          fp16_value->flags = 0;
          fp16_value->fp16_id = XNN_INVALID_VALUE_ID;
          fp16_value->fp32_id = value->id;
          fp16_value->allocation_type = xnn_allocation_type_workspace;
          value->fp16_id = fp16_value->id;
        }
      } else if (xnn_value_is_internal(value)) {
        // fp16 tensors only need half the memory of fp32 tensors.
        value->size /= 2;
      }
    }
  }

  // Count the number of external inputs and outputs which require Convert nodes
  uint32_t num_external_inputs = 0;
  uint32_t num_external_outputs = 0;
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    const struct xnn_node* node = &subgraph->nodes[n];
    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const struct xnn_value* value = &subgraph->values[node->inputs[i]];
      if (value->fp16_id != XNN_INVALID_VALUE_ID &&
          value->first_consumer == n) {
        assert(value->data == NULL);
        assert(value->datatype == xnn_datatype_fp32);
        assert(subgraph->values[value->fp16_id].datatype == xnn_datatype_fp16);
        // This value isn't always an external input, it could be an external output of the current subgraph (due to
        // partition), and be simultaneously consumed by the current node.
        if (xnn_value_is_external_input(value)) {
          num_external_inputs += 1;
        }
      }
    }
    for (uint32_t o = 0; o < node->num_outputs; o++) {
      const struct xnn_value* value = &subgraph->values[node->outputs[o]];
      if (value->fp16_id != XNN_INVALID_VALUE_ID) {
        assert(value->datatype == xnn_datatype_fp32);
        assert(subgraph->values[value->fp16_id].datatype == xnn_datatype_fp16);
        assert(xnn_value_is_external_output(value));
        num_external_outputs += 1;
      }
    }
  }
  xnn_log_debug("Discovered %"PRIu32" external inputs and %"PRIu32" external outputs",
    num_external_inputs, num_external_outputs);

  // Attempt to allocate memory for the Convert nodes.
  const uint32_t num_original_nodes = subgraph->num_nodes;
  if (xnn_subgraph_add_nodes(subgraph, num_external_inputs + num_external_outputs) != xnn_status_success) {
    xnn_log_error("FP16 rewrite aborted: failed to allocate node for external input/output");
    goto error;
  }

  // From this point the subgraph and tensor data get mutated, clean failure is
  // no longer an option.

  // Replace FP32 Values in Nodes' inputs/outputs with FP16 Values.
  // - FP32 values of static tensors get converted in a new data buffer.
  // - For external inputs and outputs we create same-shaped FP16 Values and use those instead.
  // - Values that are neither static nor external are converted to FP16 in-place
  for (uint32_t n = 0; n < num_original_values; n++) {
    struct xnn_value* value = &subgraph->values[n];
    if (value->fp16_compatible) {
      if (xnn_value_is_static(value)) {
        assert(value->datatype == xnn_datatype_fp32);
        const size_t num_elements = xnn_shape_multiply_all_dims(&value->shape);
        xnn_run_unary_elementwise_nc(
            xnn_unary_convert, xnn_datatype_fp32, xnn_datatype_fp16,
            /*params=*/NULL, /*input_quantization=*/NULL,
            /*output_quantization=*/NULL, 0, num_elements, 1, 1, 1, NULL,
            value->data, value->fp16_temp_data);
        // Remember pointer to the original fp32 data, nodes like convolution need fp32 weights/biases.
        value->fp32_data = value->data;
        value->data = value->fp16_temp_data;
        value->fp16_temp_data = NULL;
        value->datatype = xnn_datatype_fp16;
        xnn_log_debug("FP16 rewrite: converted static FP32 tensor #%" PRIu32 " to FP16 in new buffer", n);
      } else if (xnn_value_is_external(value)) {
        assert(value->datatype == xnn_datatype_fp32);
        assert(value->fp16_id != XNN_INVALID_VALUE_ID);
        value->producer = XNN_INVALID_NODE_ID;
        value->num_consumers = 0;
        value->first_consumer = XNN_INVALID_NODE_ID;
        xnn_log_debug("FP16 rewrite: created FP16 tensor #%" PRIu32
                      " for FP32 tensor #%" PRIu32,
                      subgraph->values[value->fp16_id].id, n);
      } else {
        switch (value->datatype) {
          case xnn_datatype_fp32:
            xnn_log_debug(
                "FP16 rewrite: converted FP32 tensor #%" PRIu32 " to FP16", n);
            value->datatype = xnn_datatype_fp16;
            break;
          case xnn_datatype_pfp32:
            xnn_log_debug("FP16 rewrite: converted PFP32 tensor #%" PRIu32
                          " to PFP16",
                          n);
            value->datatype = xnn_datatype_pfp16;
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }
  }
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if (node->type == xnn_node_type_invalid) {
      // Node was fused away, skip.
      continue;
    }

    if (node->type == xnn_node_type_static_constant_pad) {
      node->params.static_pad.padding_value =
        fp16_ieee_from_fp32_value(uint32_as_float(node->params.static_pad.padding_value));
    } else if (node->type == xnn_node_type_fully_connected) {
      // Patch up any LHS packing of fully-connected nodes, if needed.
      if (node->flags & XNN_FLAG_INLINE_LHS_PACKING) {
        switch (node->params.fully_connected.assumed_input_datatype) {
          case xnn_datatype_pfp32:
            // Switch from packed `fp32` to packed `fp16`.
            node->params.fully_connected.assumed_input_datatype =
                xnn_datatype_pfp16;
            break;
          case xnn_datatype_qpint8:
            // Convert from `qpint8` back to `qdint8` since we don't have a
            // `qpint8` packing function for `f16` inputs.
            node->params.fully_connected.assumed_input_datatype =
                xnn_datatype_qdint8;
            break;
          default:
            break;
        }
      } else if (subgraph->values[node->inputs[0]].datatype ==
                 xnn_datatype_qpint8) {
        subgraph->values[node->inputs[0]].datatype = xnn_datatype_qdint8;
      }
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const uint32_t fp16_id = subgraph->values[node->inputs[i]].fp16_id;
      if (fp16_id != XNN_INVALID_VALUE_ID) {
        assert(subgraph->values[fp16_id].fp32_id == node->inputs[i]);
        node->inputs[i] = fp16_id;
      }
    }
    for (uint32_t o = 0; o < node->num_outputs; o++) {
      const uint32_t fp16_id = subgraph->values[node->outputs[o]].fp16_id;
      if (fp16_id != XNN_INVALID_VALUE_ID) {
        assert(subgraph->values[fp16_id].fp32_id == node->outputs[o]);
        node->outputs[o] = fp16_id;
      }
    }
  }

  struct xnn_node* output_node = &subgraph->nodes[subgraph->num_nodes - 1];
  for (uint32_t n = num_original_nodes; n != 0; n--) {
    const struct xnn_node* node = &subgraph->nodes[n - 1];
    // Insert Convert nodes for outputs
    for (uint32_t o = 0; o < node->num_outputs; o++) {
      const struct xnn_value* value = &subgraph->values[node->outputs[o]];
      if (value->fp32_id != XNN_INVALID_VALUE_ID) {
        xnn_log_debug("Inserted FP16->FP32 Convert Node from tensor #%"PRIu32" to tensor #%"PRIu32,
          value->id, value->fp32_id);
        const uint32_t output_node_id = output_node->id;
        assert(output_node >= subgraph->nodes);
        xnn_node_clear(output_node);
        output_node->id = output_node_id;
        xnn_init_convert_node(output_node, value->id, value->fp32_id, 0 /* flags */);
        output_node -= 1;
      }
    }
    // Move the Node to the new location
    if (output_node != node) {
      const uint32_t output_node_id = output_node->id;
      assert(output_node >= subgraph->nodes);
      memcpy(output_node, node, sizeof(struct xnn_node));
      output_node->id = output_node_id;
      output_node -= 1;
    }
    // Insert Convert nodes for inputs
    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const struct xnn_value* value = &subgraph->values[node->inputs[i]];
      if (value->fp32_id != XNN_INVALID_VALUE_ID && value->first_consumer == n - 1) {
        // Only insert convert nodes if the value actually is an external input. This value could be an external output,
        // if that's the case, we have already inserted a convert node in loop above for outputs.
        if (xnn_value_is_external_input(&subgraph->values[value->fp32_id])) {
          xnn_log_debug("Inserted FP32->FP16 Convert Node from tensor #%"PRIu32" to tensor #%"PRIu32,
                        value->fp32_id, value->id);
          const uint32_t output_node_id = output_node->id;
          assert(output_node >= subgraph->nodes);
          xnn_node_clear(output_node);
          output_node->id = output_node_id;
          xnn_init_convert_node(output_node, value->fp32_id, value->id, 0 /* flags */);
          output_node -= 1;
        }
      }
    }
  }

  xnn_log_info("XNNPACK has switched to FP16 inference mode!");
  // subgraph_dump(subgraph, stderr);

  return true;

error:
  for (uint32_t n = 0; n < subgraph->num_values; n++) {
    struct xnn_value* value = &subgraph->values[n];
    // Deallocate extra memory used during static tensor rewrite.
    if (value->fp16_temp_data != NULL) {
      xnn_release_memory(value->fp16_temp_data);
    }
    // Revert marking values as FP16-compatible, as xnn_delete_subgraph() may assume ownership of those that are.
    value->fp16_compatible = false;
  }

  // Clear the fp16 values created for external inputs and outputs.
  for (uint32_t n = num_original_values; n < subgraph->num_values; n++) {
    xnn_value_clear(&subgraph->values[n]);
  }

  return false;
}

static void xnn_node_replace_output(struct xnn_node* node, uint32_t old_output_id, uint32_t new_output_id)
{
  for (size_t i = 0; i < node->num_outputs; i++) {
    if (node->outputs[i] == old_output_id) {
      node->outputs[i] = new_output_id;
    }
  }
}

static bool is_clamp(const struct xnn_node* node)
{
  return node->type == xnn_node_type_unary_elementwise && node->unary_operator == xnn_unary_clamp;
}

static bool has_clamp(const struct xnn_node* node)
{
  if (is_clamp(node)) {
    return true;
  }
  switch (node->type) {
    case xnn_node_type_average_pooling_2d:
    case xnn_node_type_convolution_2d:
    case xnn_node_type_deconvolution_2d:
    case xnn_node_type_depthwise_convolution_2d:
    case xnn_node_type_fully_connected:
    case xnn_node_type_max_pooling_2d:
      return true;
    default:
      return false;
  }
}

// Can we reorder the use of a value from the producer to the consumer?
// We can if no nodes between the producer and the consumer use the value.
static bool can_reorder_use(xnn_subgraph_t subgraph, uint32_t value_id,
                            uint32_t producer_id, uint32_t consumer_id) {
  assert(producer_id < consumer_id);
  for (uint32_t i = producer_id + 1; i < consumer_id; i++) {
    const struct xnn_node* node = &subgraph->nodes[i];
    for (uint32_t j = 0; j < node->num_inputs; j++) {
      if (node->inputs[j] == value_id) return false;
    }
    for (uint32_t j = 0; j < node->num_outputs; j++) {
      if (node->outputs[j] == value_id) return false;
    }
  }
  return true;
}

enum xnn_status xnn_subgraph_fusion(
    xnn_subgraph_t subgraph)
{
  // Fuse Nodes where possible
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    if (value->num_consumers == 1) {
      const uint32_t producer_id = value->producer;
      if (producer_id == XNN_INVALID_NODE_ID) {
        continue;
      }
      assert(producer_id < subgraph->num_nodes);

      const uint32_t consumer_id = value->first_consumer;
      if (consumer_id == XNN_INVALID_NODE_ID) {
        continue;
      }
      assert(consumer_id < subgraph->num_nodes);

      struct xnn_node* producer = &subgraph->nodes[producer_id];
      assert(producer->type != xnn_node_type_invalid);
      struct xnn_node* consumer = &subgraph->nodes[consumer_id];
      if (consumer->type == xnn_node_type_invalid) {
        xnn_log_fatal(
            "Node %u (produced by %s node %u) has no consumers. Should an "
            "external output have been set?",
            consumer_id, xnn_node_type_to_string(producer->type), producer_id);
        return xnn_status_invalid_state;
      }

      // Try to fuse Clamp Node upstream into producer Node
      if (is_clamp(consumer) && has_clamp(producer)) {
        xnn_log_info("fuse Clamp Node #%"PRIu32" into upstream Node #%"PRIu32, consumer_id, producer_id);
        assert(producer->num_outputs == 1);
        assert(consumer->num_inputs == 1);
        assert(consumer->num_outputs == 1);

        const uint32_t fused_output_id = consumer->outputs[0];
        assert(fused_output_id < subgraph->num_values);
        subgraph->values[fused_output_id].producer = producer_id;
        producer->outputs[0] = fused_output_id;

        producer->activation.output_min =
          math_max_f32(producer->activation.output_min, consumer->activation.output_min);
        producer->activation.output_max =
          math_min_f32(producer->activation.output_max, consumer->activation.output_max);
        producer->params.unary.clamp.min =
            math_max_f32(producer->params.unary.clamp.min,
                          consumer->params.unary.clamp.min);
        producer->params.unary.clamp.max =
            math_min_f32(producer->params.unary.clamp.max,
                          consumer->params.unary.clamp.max);

        xnn_node_clear(consumer);
        xnn_value_clear(value);
      }
      // Try to fuse Constant Pad node downstream into [Depthwise] Convolution 2D Node
      if (producer->type == xnn_node_type_static_constant_pad) {
        assert(producer->num_inputs == 1);
        assert(producer->num_outputs == 1);
        const bool is_spatial_2d_padding = value->shape.num_dims == 4 &&
          (producer->params.static_pad.pre_paddings[0] | producer->params.static_pad.post_paddings[0] |
           producer->params.static_pad.pre_paddings[3] | producer->params.static_pad.post_paddings[3]) == 0;
        const enum xnn_datatype padding_datatype = subgraph->values[producer->outputs[0]].datatype;
        const uint32_t padding_value = producer->params.static_pad.padding_value;
        const bool is_zero_padding =
          (padding_datatype == xnn_datatype_fp32 && padding_value == 0) ||
          ((padding_datatype == xnn_datatype_qint8 || padding_datatype == xnn_datatype_quint8) &&
          padding_value == (uint32_t) (uint8_t) subgraph->values[producer->outputs[0]].quantization.zero_point);
        switch (consumer->type) {
          case xnn_node_type_convolution_2d:
            if (is_spatial_2d_padding && is_zero_padding && !(consumer->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING)) {
              xnn_log_info("fuse Constant Pad Node #%"PRIu32" into Convolution 2D Node #%"PRIu32,
                consumer_id, producer_id);
              assert(consumer->num_inputs >= 1);
              assert(consumer->inputs[0] == producer->outputs[0]);

              consumer->params.convolution_2d.input_padding_top    += producer->params.static_pad.pre_paddings[1];
              consumer->params.convolution_2d.input_padding_right  += producer->params.static_pad.post_paddings[2];
              consumer->params.convolution_2d.input_padding_bottom += producer->params.static_pad.post_paddings[1];
              consumer->params.convolution_2d.input_padding_left   += producer->params.static_pad.pre_paddings[2];

              consumer->inputs[0] = producer->inputs[0];

              const uint32_t fused_input_id = producer->inputs[0];
              assert(fused_input_id < subgraph->num_values);
              if (subgraph->values[fused_input_id].first_consumer == producer_id) {
                subgraph->values[fused_input_id].first_consumer = consumer_id;
              }

              xnn_node_clear(producer);
              xnn_value_clear(value);
            }
            break;
          case xnn_node_type_depthwise_convolution_2d:
            if (is_spatial_2d_padding && is_zero_padding && !(consumer->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING)) {
              xnn_log_info("fuse Constant Pad Node #%"PRIu32" into Depthwise Convolution 2D Node #%"PRIu32,
                consumer_id, producer_id);
              assert(consumer->num_inputs >= 1);
              assert(consumer->inputs[0] == producer->outputs[0]);

              consumer->params.depthwise_convolution_2d.input_padding_top +=
                producer->params.static_pad.pre_paddings[1];
              consumer->params.depthwise_convolution_2d.input_padding_right +=
                producer->params.static_pad.post_paddings[2];
              consumer->params.depthwise_convolution_2d.input_padding_bottom +=
                producer->params.static_pad.post_paddings[1];
              consumer->params.depthwise_convolution_2d.input_padding_left +=
                producer->params.static_pad.pre_paddings[2];

              consumer->inputs[0] = producer->inputs[0];

              const uint32_t fused_input_id = producer->inputs[0];
              assert(fused_input_id < subgraph->num_values);
              if (subgraph->values[fused_input_id].first_consumer == producer_id) {
                subgraph->values[fused_input_id].first_consumer = consumer_id;
              }

              xnn_node_clear(producer);
              xnn_value_clear(value);
            }
            break;
          default:
            break;
        }
      }

      // Try to fuse copy upstream. Copy can be fused upstream as long as this value is internal.
      // E.g. ---> (N1) --- value ---> (Copy) ---> v1
      // If value is persistent or external, fusing copy upstream into N1 will skip the write to value, N1 will write to
      // v1 instead, which is wrong.
      if (consumer->type == xnn_node_type_copy && xnn_value_is_valid(value) && xnn_value_is_internal(value) &&
          can_reorder_use(subgraph, consumer->outputs[0], producer_id, consumer_id)) {
        xnn_log_info(
          "value %d fuse Copy Node #%" PRIu32 " into upstream %s Node #%" PRIu32, value->id, consumer->id,
          xnn_node_type_to_string(producer->type), producer->id);
        assert(consumer->num_inputs == 1);
        assert(consumer->num_outputs == 1);
        const uint32_t fused_output_id = consumer->outputs[0];
        assert(fused_output_id < subgraph->num_values);
        subgraph->values[fused_output_id].producer = producer_id;
        xnn_node_replace_output(producer, value->id, fused_output_id);
        xnn_node_clear(consumer);
        xnn_value_clear(value);
      }

      // Try to fuse copy downstream.
      // E.g. --- v1 ---> (copy) --- value ---> (n2)
      // If value is external or persistent, we cannot simply remove the copy, since we need to write to value.
      if (producer->type == xnn_node_type_copy && xnn_value_is_valid(value) && xnn_value_is_internal(value) &&
          can_reorder_use(subgraph, producer->inputs[0], producer_id, consumer_id)) {
        // We need to check that value is valid here because value could have been cleared by a previous optimization,
        // this can happen if we have a chain of Copy(s), e.g.:
        // ---v1--> (Copy1) ---v2--> (Copy2) ---v3--> (Copy3) ---v4-->
        // v2 could have been cleared when we fused Copy2 upstream into Copy1, so v2 isn't valid anymore, but since v2's
        // producer is also a Copy, we will incorrectly try to fuse Copy1 downstream into Copy2 (again).
        xnn_log_info(
          "value %d fuse Copy Node #%" PRIu32 " into downstream %s Node #%" PRIu32, value->id, producer->id,
          xnn_node_type_to_string(consumer->type), consumer->id);
        assert(producer->num_outputs == 1);
        assert(producer->num_inputs == 1);
        const uint32_t copy_input_id = producer->inputs[0];
        const uint32_t copy_output_id = producer->outputs[0];
        bool found_consumer_input = false;
        for (size_t i = 0; i < consumer->num_inputs; i++) {
          if (consumer->inputs[i] == copy_output_id) {
            consumer->inputs[i] = copy_input_id;;
            found_consumer_input = true;
            // TODO(b/254734644): A consumer can only consume this value once, since we asserted earlier that value has
            // only 1 consumer, so we can break here as there will be no other consumer inputs that has the same id.
            break;
          }
        }
        (void) found_consumer_input;  // Silence unused variable warning in non-debug.
        assert(found_consumer_input);

        if (subgraph->values[copy_input_id].first_consumer == producer_id) {
          subgraph->values[copy_input_id].first_consumer = consumer_id;
        }
        xnn_node_clear(producer);
        xnn_value_clear(value);
      }
    }
  }

  return xnn_status_success;
}

void xnn_subgraph_clean_up(xnn_subgraph_t subgraph) {
  // Count the number of consumers for each value.
  xnn_subgraph_analyze_consumers_and_producers(subgraph);

  // Remove unreferenced values.
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    if (value->type == xnn_value_type_invalid) {
      continue;
    }
    if (value->num_consumers == 0 && !xnn_value_is_external_input(value) &&
        !xnn_value_is_persistent(value)) {
      if (value->producer != XNN_INVALID_NODE_ID) {
        struct xnn_node* producer = &subgraph->nodes[value->producer];
        if (producer->num_outputs == 1) {
          xnn_node_clear(&subgraph->nodes[value->producer]);
        }
      }
      xnn_value_clear(value);
    }
  }

  // Compact the nodes and sort them hierarchically (stably), if needed.
  bool changes = false;
  uint32_t* nodes_map = alloca(sizeof(uint32_t) * subgraph->num_nodes);
  bool* values_ready = alloca(sizeof(bool) * subgraph->num_values);
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    values_ready[i] = value->producer == XNN_INVALID_NODE_ID ||
                      xnn_value_is_external_input(value) ||
                      xnn_value_is_persistent(value);
  }
  uint32_t left = 0;
  uint32_t right = subgraph->num_nodes;
  while (left < right) {
    for (uint32_t i = left; i < right; i++) {
      // Move invalid nodes to the right.
      while (i < right && subgraph->nodes[i].type == xnn_node_type_invalid) {
        for (uint32_t j = i + 1; j < right; j++) {
          changes = true;
          subgraph->nodes[j - 1] = subgraph->nodes[j];
        }
        right--;
      }
      if (right <= i) {
        break;
      }

      // Check whether all inputs to this node have been produced.
      struct xnn_node* node = &subgraph->nodes[i];
      bool all_values_avail = true;
      for (uint32_t j = 0; all_values_avail && j < node->num_inputs; j++) {
        all_values_avail = values_ready[node->inputs[j]];
      }

      // If so, bubble this node down to the left end of the list of nodes.
      if (all_values_avail) {
        nodes_map[node->id] = left;
        node->id = left;
        for (uint32_t j = 0; j < node->num_outputs; j++) {
          values_ready[node->outputs[j]] = true;
        }
        if (left < i) {
          changes = true;
          struct xnn_node tmp_node = *node;
          for (uint32_t j = i; left < j; j--) {
            subgraph->nodes[j] = subgraph->nodes[j - 1];
          }
          subgraph->nodes[left] = tmp_node;
        }
        left++;
      }
    }
  }

  // Update the node IDs in the subgraph values if they have changed..
  if (changes) {
    for (uint32_t i = 0; i < subgraph->num_values; i++) {
      struct xnn_value* value = &subgraph->values[i];
      if (value->producer != XNN_INVALID_NODE_ID) {
        value->producer = nodes_map[value->producer];
      }
      if (value->first_consumer != XNN_INVALID_NODE_ID) {
        value->first_consumer = nodes_map[value->first_consumer];
      }
    }
    subgraph->num_nodes = right;
  }
}

enum xnn_status xnn_subgraph_optimize_packed_lhs(xnn_subgraph_t subgraph) {
  // Count the number of changes made.
  size_t changes = 0;

  // Loop over the nodes in the subgraph.
  for (uint32_t node_id = 0; node_id < subgraph->num_nodes; node_id++) {
    struct xnn_node* node = &subgraph->nodes[node_id];

    // Skip anything that is not a fully-connected node.
    if (node->type != xnn_node_type_fully_connected) {
      continue;
    }

    // Get a handle on the inputs/outputs.
    const uint32_t input_id = node->inputs[0];
    struct xnn_value* input_value = &subgraph->values[input_id];
    struct xnn_value* kernel_value = &subgraph->values[node->inputs[1]];
    struct xnn_value* output_value = &subgraph->values[node->outputs[0]];
    const enum xnn_datatype input_datatype = input_value->datatype;
    const enum xnn_datatype kernel_datatype = kernel_value->datatype;
    const enum xnn_datatype output_datatype = output_value->datatype;

    // Check if we have a packed GEMM config for the combination of
    // input/kernel/output.
    const struct xnn_gemm_config* gemm_config = NULL;
    enum xnn_datatype assumed_datatype = xnn_datatype_invalid;
    switch (input_datatype) {
      case xnn_datatype_fp16:
        if (input_datatype == output_datatype &&
            kernel_datatype == xnn_datatype_fp16) {
          if ((gemm_config = xnn_init_pf16_gemm_config())) {
            assumed_datatype = xnn_datatype_pfp16;
          }
        }
        break;
      case xnn_datatype_fp32:
        if (input_datatype == output_datatype &&
            kernel_datatype == xnn_datatype_fp32) {
          if ((gemm_config = xnn_init_pf32_gemm_config())) {
            assumed_datatype = xnn_datatype_pfp32;
          }
        }
        break;
      case xnn_datatype_qint8:
        if (input_datatype == output_datatype &&
            kernel_datatype == xnn_datatype_qcint8) {
          if ((gemm_config = xnn_init_pqs8_qc8w_gemm_config())) {
            assumed_datatype = xnn_datatype_pqint8;
          }
        }
        break;
      case xnn_datatype_qdint8:
        // We may inline the `qdint8` packing regardless of whether we have a
        // specialized `qpint8` kernel or not.
        assumed_datatype = xnn_datatype_qdint8;
        if (output_datatype == xnn_datatype_fp32) {
          switch (kernel_datatype) {
            case xnn_datatype_qbint4:
              if ((gemm_config = xnn_init_qp8_f32_qb4w_gemm_config())) {
                assumed_datatype = xnn_datatype_qpint8;
              }
              break;
            case xnn_datatype_qcint4:
              if ((gemm_config = xnn_init_qp8_f32_qc4w_gemm_config())) {
                assumed_datatype = xnn_datatype_qpint8;
              }
              break;
            case xnn_datatype_qcint8:
              if ((gemm_config = xnn_init_qp8_f32_qc8w_gemm_config())) {
                assumed_datatype = xnn_datatype_qpint8;
              }
              break;
            default:
              break;
          }
        }
        break;
      default:
        // If none of the above happened, do nothing for this node.
        continue;
    }

    if (assumed_datatype != xnn_datatype_invalid) {
      if (subgraph->flags & XNN_FLAG_DONT_INLINE_LHS_PACKING) {
        if (assumed_datatype == xnn_datatype_qdint8) {
          // If the input is already `qdint8`, don't do anything different.
          continue;
        } else if (assumed_datatype == xnn_datatype_qpint8) {
          xnn_log_debug(
              // `qpint8` inputs are generated by modifying the `convert` op
              // that generated the `qdint8` input, so we only have to add a
              // `pack-lh` op for the other input types.
              "Coercing type of input ID #%" PRIu32
              " of %s node from `%s` to `%s`.",
              input_id, xnn_node_type_to_string(xnn_node_type_convert),
              xnn_datatype_to_string(input_datatype),
              xnn_datatype_to_string(xnn_datatype_qpint8));
          subgraph->values[input_id].datatype = assumed_datatype;
          subgraph->values[input_id].gemm_config = gemm_config;
        } else {
          // Insert a node to pack the LHS.
          xnn_log_debug("Adding %s node for input ID #%" PRIu32
                        " of type `%s` for %s node.",
                        xnn_node_type_to_string(xnn_node_type_pack_lh),
                        input_id,
                        xnn_node_type_to_string(xnn_node_type_fully_connected),
                        xnn_datatype_to_string(xnn_datatype_qpint8));
          uint32_t new_id = XNN_INVALID_VALUE_ID;
          enum xnn_status status =
              xnn_insert_pack_lh_node(subgraph, input_id, &new_id);
          if (status != xnn_status_success) {
            return status;
          }
          subgraph->nodes[node_id].inputs[0] = new_id;
          changes++;
        }
        // To prevent issues with packing, coerce the shape of the inputs from
        // `[B, M, K]` to `[B * M, K]` for the fully-connected op.
        subgraph->values[input_id].flags |= XNN_FLAG_SQUASH_GROUPS;
      } else {
        if (input_datatype == xnn_datatype_qdint8) {
          // Short-circuit the inputs of the producer of the `qdint8` values.
          struct xnn_node* producer = &subgraph->nodes[input_value->producer];
          if (producer->type != xnn_node_type_convert) {
            xnn_log_error(
                "Expected producer node #%u of %s tensor #%u to be of type %s, "
                "but found type %s instead.",
                input_value->producer, xnn_datatype_to_string(input_datatype),
                input_id, xnn_node_type_to_string(xnn_node_type_convert),
                xnn_node_type_to_string(producer->type));
            return xnn_status_invalid_state;
          }
          struct xnn_value* new_input = &subgraph->values[producer->inputs[0]];
          node->inputs[0] = producer->inputs[0];
          if (new_input->first_consumer == input_value->producer) {
            new_input->first_consumer = node_id;
          }
          if (--input_value->num_consumers == 0) {
            xnn_node_clear(producer);
          }
          changes++;
        }
        node->params.fully_connected.assumed_input_datatype = assumed_datatype;
        node->flags |= XNN_FLAG_INLINE_LHS_PACKING;
      }
    }
  }

  // Clean up after ourselves.
  if (changes) {
    xnn_subgraph_clean_up(subgraph);
  }

  return xnn_status_success;
}

enum xnn_status xnn_subgraph_optimize(xnn_subgraph_t subgraph,
                                      uint32_t optimization_flags) {
  // Start with a clean subgraph.
  xnn_subgraph_clean_up(subgraph);

  if (!(optimization_flags & XNN_FLAG_NO_OPERATOR_FUSION)) {
    xnn_subgraph_fusion(subgraph);
  }

  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    xnn_log_error("failed to get hardware config");
    return xnn_status_unsupported_hardware;
  }

  if ((optimization_flags & XNN_FLAG_FORCE_FP16_INFERENCE) && (!xnn_is_f16_compatible_config(hardware_config))) {
    xnn_log_error("failed to force FP16 inference: hardware supports neither native nor emulated FP16 operators");
    return xnn_status_unsupported_hardware;
  }
  const bool try_native_fp16 =
    (optimization_flags & XNN_FLAG_HINT_FP16_INFERENCE) && xnn_is_f16_supported_natively(hardware_config);
  const bool force_fp16 = (optimization_flags & XNN_FLAG_FORCE_FP16_INFERENCE);
  if (try_native_fp16 || force_fp16) {
    const bool fp16_rewrite_succeeded = xnn_subgraph_rewrite_for_fp16(subgraph);
    if (force_fp16 && !fp16_rewrite_succeeded) {
      xnn_log_error("failed to force FP16 inference: subgraph is incompatible with FP16 operators");
      return xnn_status_unsupported_parameter;
    }
    if (fp16_rewrite_succeeded) {
      // Re-run xnn_subgraph_analyze_consumers_and_producers since fp16 re-write
      // inserts nodes and changes producers/consumers.
      xnn_subgraph_analyze_consumers_and_producers(subgraph);
    }
  }

#if XNN_ENABLE_SPARSE
  if ((optimization_flags & XNN_FLAG_HINT_SPARSE_INFERENCE) &&
      (xnn_is_chw_compatible_config(hardware_config))) {
    xnn_subgraph_rewrite_for_nchw(subgraph);
  }
#endif

  enum xnn_status status = xnn_subgraph_optimize_packed_lhs(subgraph);
  if (status != xnn_status_success) {
    return status;
  }

  return xnn_status_success;
}

enum xnn_status xnn_delete_subgraph(
  xnn_subgraph_t subgraph)
{
  if (subgraph != NULL) {
    if (subgraph->nodes != NULL) {
      memset(subgraph->nodes, 0, sizeof(struct xnn_node) * subgraph->num_nodes);
      xnn_release_memory(subgraph->nodes);
    }

    if (subgraph->values != NULL) {
      // Release the dynamic allocations created during FP16 rewrite, if the subgraph still has ownership of them.
      for (uint32_t i = 0; i < subgraph->num_values; i++) {
        struct xnn_value* value = &subgraph->values[i];
        if (value->fp16_compatible && value->data != NULL) {
          XNN_PRAGMA_CLANG("clang diagnostic push")
          XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wcast-qual\"")
          xnn_release_memory((void*)value->data);
          XNN_PRAGMA_CLANG("clang diagnostic pop")
        }
      }

      memset(subgraph->values, 0, sizeof(struct xnn_value) * subgraph->num_values);
      xnn_release_memory(subgraph->values);
    }

    memset(subgraph, 0, sizeof(struct xnn_subgraph));
    xnn_release_memory(subgraph);
  }
  return xnn_status_success;
}

enum xnn_node_type xnn_reduce_operator_to_node_type(enum xnn_reduce_operator type)
{
  switch (type) {
    case xnn_reduce_max:
      return xnn_node_type_static_reduce_max;
    case xnn_reduce_mean:
      return xnn_node_type_static_mean;
    case xnn_reduce_min:
      return xnn_node_type_static_reduce_min;
    case xnn_reduce_sum:
      return xnn_node_type_static_sum;
    default:
      return xnn_node_type_invalid;
  }
}

enum xnn_reduce_operator xnn_node_type_to_reduce_operator(enum xnn_node_type type)
{
  switch (type) {
    case xnn_node_type_static_mean:
      return xnn_reduce_mean;
    case xnn_node_type_static_reduce_max:
      return xnn_reduce_max;
    case xnn_node_type_static_reduce_min:
      return xnn_reduce_min;
    case xnn_node_type_static_sum:
      return xnn_reduce_sum;
    default:
      return xnn_reduce_invalid;
  }
}
