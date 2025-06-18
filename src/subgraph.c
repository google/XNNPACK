// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/subgraph.h"

#include <assert.h>
#include <inttypes.h>  // IWYU pragma: keep
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
#include "src/xnnpack/datatype.h"
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

enum xnn_status xnn_insert_clamp_node(xnn_subgraph_t subgraph, float output_min,
                                      float output_max, struct xnn_node* node) {
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

enum xnn_status xnn_insert_pack_lh_node(xnn_subgraph_t subgraph,
                                        uint32_t input_id, uint32_t* new_id) {
  const struct xnn_value* input = &subgraph->values[input_id];
  enum xnn_status status = xnn_status_uninitialized;
  switch (input->datatype) {
    case xnn_datatype_qint8: {
      // Create a copy of the input shape since it might be reallocated by the
      // subgraph when the new tensor is added.
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
      status = xnn_define_tensor_value(subgraph, input->datatype, 0, NULL, NULL,
                                       /*external_id=*/XNN_INVALID_VALUE_ID,
                                       /*flags=*/0, new_id);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }

  return xnn_define_pack_lh(subgraph, input_id, *new_id, /*flags=*/0);
}

enum xnn_status xnn_create_subgraph(uint32_t external_value_ids, uint32_t flags,
                                    xnn_subgraph_t* subgraph_out) {
  struct xnn_subgraph* subgraph = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create subgraph: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_out_of_memory;

  subgraph = xnn_allocate_zero_memory(sizeof(struct xnn_subgraph));
  if (subgraph == NULL) {
    xnn_log_error("failed to allocate %zu bytes for subgraph descriptor",
                  sizeof(struct xnn_subgraph));
    goto error;
  }

  subgraph->external_value_ids = external_value_ids;

  subgraph->values =
      xnn_allocate_zero_memory(external_value_ids * sizeof(struct xnn_value));
  if (subgraph->values == NULL) {
    xnn_log_error("failed to allocate %zu bytes for subgraph values",
                  (size_t)external_value_ids * sizeof(struct xnn_value));
    goto error;
  }
  for (size_t i = 0; i < external_value_ids; i++) {
    subgraph->values[i].id = i;
  }
  subgraph->num_values = external_value_ids;
  subgraph->num_reserved_values = external_value_ids;

  *subgraph_out = subgraph;
  return xnn_status_success;

error:
  xnn_delete_subgraph(subgraph);
  return status;
}

struct xnn_value* xnn_subgraph_new_internal_value(xnn_subgraph_t subgraph) {
  struct xnn_value* values = subgraph->values;
  const size_t size = subgraph->num_values;
  const size_t capacity = subgraph->num_reserved_values;
  if (capacity < size + 1) {
    const size_t new_capacity =
        max(min(capacity * 2, capacity + 512), capacity + 64);
    assert(new_capacity >= size + 1);
    values =
        xnn_reallocate_memory(values, new_capacity * sizeof(struct xnn_value));
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

// Copies all fields from `src_node` to `dst_node` but leaves the node ID
// unchanged.
void xnn_node_copy(struct xnn_node* dst_node, const struct xnn_node* src_node) {
  const uint32_t node_id = dst_node->id;
  *dst_node = *src_node;
  dst_node->id = node_id;
}

// Copies all fields from `src_value` to `dst_value` but leaves the value ID
// unchanged.
void xnn_value_copy(struct xnn_value* dst_value,
                    const struct xnn_value* src_value) {
  const uint32_t value_id = dst_value->id;
  *dst_value = *src_value;
  dst_value->id = value_id;
}

void xnn_runtime_value_copy(struct xnn_runtime_value* dst_value,
                            const struct xnn_value* src_value) {
  // Note: Value ID stays unchanged

  dst_value->type = src_value->type;
  dst_value->datatype = src_value->datatype;
  dst_value->quantization = src_value->quantization;
  dst_value->shape = src_value->shape;
  dst_value->size = src_value->size;
  dst_value->allocation_type = src_value->allocation_type;
  dst_value->flags = src_value->flags;
  if (src_value->num_consumers == 1) {
    dst_value->flags |= XNN_VALUE_FLAG_ONE_CONSUMER;
  }
  if (src_value->fp16_compatible) {
    dst_value->flags |= XNN_VALUE_FLAG_FP16_COMPATIBLE;
  }
  if (src_value->layout == xnn_layout_type_nchw) {
    dst_value->flags |= XNN_VALUE_FLAG_LAYOUT_NCHW;
  }
  dst_value->data = src_value->data;
  dst_value->first_consumer = src_value->first_consumer;
  dst_value->fp32_data = src_value->fp32_data;
  dst_value->gemm_config = src_value->gemm_config;
}

struct xnn_node* xnn_subgraph_new_node(xnn_subgraph_t subgraph) {
  struct xnn_node* nodes = subgraph->nodes;
  const size_t size = subgraph->num_nodes;
  const size_t capacity = subgraph->num_reserved_nodes;

  if (capacity < size + 1) {
    const size_t new_capacity =
        max(min(capacity * 2, capacity + 512), capacity + 64);
    assert(new_capacity >= size + 1);
    nodes =
        xnn_reallocate_memory(nodes, new_capacity * sizeof(struct xnn_node));
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
  xnn_node_clear(new_node);
  new_node->id = size;
  return new_node;
}

enum xnn_status xnn_subgraph_add_nodes(xnn_subgraph_t subgraph,
                                       size_t num_nodes) {
  struct xnn_node* nodes = subgraph->nodes;
  const size_t size = subgraph->num_nodes;
  const size_t capacity = subgraph->num_reserved_nodes;

  if (capacity < size + num_nodes) {
    const size_t new_capacity =
        max(min(capacity * 2, capacity + 512), capacity + max(num_nodes, 64));
    assert(new_capacity >= size + num_nodes);
    nodes =
        xnn_reallocate_memory(nodes, new_capacity * sizeof(struct xnn_node));
    if (nodes == NULL) {
      xnn_log_error("failed to allocate %zu bytes for subgraph nodes",
                    capacity * sizeof(struct xnn_node));
      return xnn_status_out_of_memory;
    }

    subgraph->num_reserved_nodes = new_capacity;
    subgraph->nodes = nodes;
  }
  subgraph->num_nodes = size + num_nodes;
  struct xnn_node* new_nodes = nodes + size;
  for (size_t i = 0; i < num_nodes; i++) {
    xnn_node_clear(&new_nodes[i]);
    new_nodes[i].id = size + i;
  }

  return xnn_status_success;
}

void xnn_subgraph_analyze_consumers_and_producers(xnn_subgraph_t subgraph) {
  // Initialize producer/consumer fields to safe defaults.
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    value->producer = XNN_INVALID_NODE_ID;
    value->first_consumer = XNN_INVALID_NODE_ID;
    value->num_consumers = 0;
  }

  // Analyse Nodes' inputs and output and update Values' producer/consumer
  // fields
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];

    if (node->type == xnn_node_type_invalid) {
      continue;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const uint32_t input_id = node->inputs[i];
      assert(input_id < subgraph->num_values);

      if (subgraph->values[input_id].num_consumers++ == 0) {
        assert(subgraph->values[input_id].first_consumer ==
               XNN_INVALID_NODE_ID);
        subgraph->values[input_id].first_consumer = n;
        subgraph->values[input_id].all_consumers_types_same = true;
      } else {
        enum xnn_node_type first_consumer_type =
            subgraph->nodes[subgraph->values[input_id].first_consumer].type;
        subgraph->values[input_id].all_consumers_types_same &=
            (first_consumer_type == node->type);
      }
    }

    for (uint32_t o = 0; o < node->num_outputs; o++) {
      const uint32_t output_id = node->outputs[o];
      assert(output_id < subgraph->num_values);
      assert(subgraph->values[output_id].producer == XNN_INVALID_NODE_ID);
      subgraph->values[output_id].producer = n;
    }
  }

  // Count extra consumer for Values which are external outputs.
  // Remove unreferenced values.
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    if (xnn_value_is_external_output(value->flags)) {
      value->num_consumers += 1;
    }
  }
}

#define XNN_LAYOUT_FLAG_COMPATIBLE_NCHW 1
#define XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW 2
#define XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC 4
#define XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER 8

static bool all_values_fp(xnn_subgraph_t subgraph,
                          const struct xnn_node* node) {
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

uint32_t xnn_check_nchw_compatibility(xnn_subgraph_t subgraph,
                                      struct xnn_node* node) {
  if (!all_values_fp(subgraph, node)) {
    if (node->type != xnn_node_type_invalid) {
      xnn_log_info("Node %s compute type is incompatible with sparse inference",
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
      // - 3x3 stride-2 convolution (no dilation, padding 1 on each side, no
      // groups, 3 input channels)
      if (node->params.convolution_2d.groups != 1) {
        xnn_log_info("Node %s groups (%" PRIu32
                     ") "
                     "is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.convolution_2d.groups);
        return 0;
      }
      if ((node->params.convolution_2d.dilation_height |
           node->params.convolution_2d.dilation_width) != 1) {
        xnn_log_info("Node %s dilation (height=%" PRIu32 ", width=%" PRIu32
                     ") "
                     "is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.convolution_2d.dilation_height,
                     node->params.convolution_2d.dilation_width);
        return 0;
      }
      if ((node->params.convolution_2d.kernel_height |
           node->params.convolution_2d.kernel_width) == 1) {
        if ((node->params.convolution_2d.input_padding_top |
             node->params.convolution_2d.input_padding_right |
             node->params.convolution_2d.input_padding_bottom |
             node->params.convolution_2d.input_padding_left) != 0) {
          xnn_log_info("Node %s (1x1 kernel) padding (top=%" PRIu32
                       ", right=%" PRIu32 ", bottom=%" PRIu32 ", left=%" PRIu32
                       ") "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.input_padding_top,
                       node->params.convolution_2d.input_padding_right,
                       node->params.convolution_2d.input_padding_bottom,
                       node->params.convolution_2d.input_padding_left);
          return 0;
        }
        if ((node->params.convolution_2d.subsampling_height |
             node->params.convolution_2d.subsampling_width) != 1) {
          xnn_log_info("Node %s (1x1 kernel) subsampling (height=%" PRIu32
                       ", width=%" PRIu32
                       ") "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.subsampling_height,
                       node->params.convolution_2d.subsampling_width);
          return 0;
        }
        return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
      } else if (node->params.convolution_2d.kernel_height == 3 &&
                 node->params.convolution_2d.kernel_width == 3) {
        if (node->params.convolution_2d.input_padding_top != 1 ||
            node->params.convolution_2d.input_padding_right != 1 ||
            node->params.convolution_2d.input_padding_bottom != 1 ||
            node->params.convolution_2d.input_padding_left != 1) {
          xnn_log_info("Node %s (3x3 kernel) padding (top=%" PRIu32
                       ", right=%" PRIu32 ", bottom=%" PRIu32 ", left=%" PRIu32
                       ") "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.input_padding_top,
                       node->params.convolution_2d.input_padding_right,
                       node->params.convolution_2d.input_padding_bottom,
                       node->params.convolution_2d.input_padding_left);
          return 0;
        }
        if ((node->params.convolution_2d.subsampling_height |
             node->params.convolution_2d.subsampling_width) != 2) {
          xnn_log_info("Node %s (3x3 kernel) subsampling (height=%" PRIu32
                       ", width=%" PRIu32
                       ") "
                       "is incompatible with sparse inference",
                       xnn_node_type_to_string(node->type),
                       node->params.convolution_2d.subsampling_height,
                       node->params.convolution_2d.subsampling_width);
          return 0;
        }
        if (node->params.convolution_2d.group_input_channels != 3) {
          xnn_log_info(
              "Node %s (3x3 kernel) input channels (%zu) "
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
      if ((node->params.depthwise_convolution_2d.dilation_height |
           node->params.depthwise_convolution_2d.dilation_width) != 1) {
        xnn_log_info("Node %s dilation (height=%" PRIu32 ", width=%" PRIu32
                     ") "
                     "is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.convolution_2d.dilation_height,
                     node->params.convolution_2d.dilation_width);
        return 0;
      }
      if (node->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
        xnn_log_info("Node %s flags (%" PRIu32
                     ") has padding incompatible with sparse inference",
                     xnn_node_type_to_string(node->type), node->flags);
        return 0;
      }
      if (node->params.depthwise_convolution_2d.depth_multiplier != 1) {
        xnn_log_info("Node %s depth_multiplier (%" PRIu32
                     ") is incompatible with sparse inference",
                     xnn_node_type_to_string(node->type),
                     node->params.depthwise_convolution_2d.depth_multiplier);
        return 0;
      }
      if (node->params.depthwise_convolution_2d.subsampling_height !=
          node->params.depthwise_convolution_2d.subsampling_width) {
        xnn_log_info("Node %s subsampling (height=%" PRIu32 ", width=%" PRIu32
                     ") "
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
          xnn_log_info(
              "Node %s subsampling_height (%" PRIu32
              ") "
              "is incompatible with sparse inference",
              xnn_node_type_to_string(node->type),
              node->params.depthwise_convolution_2d.subsampling_height);
          return 0;
      }
      if (node->params.depthwise_convolution_2d.kernel_height !=
          node->params.depthwise_convolution_2d.kernel_width) {
        xnn_log_info("Node %s kernel (height=%" PRIu32 ", width=%" PRIu32
                     ") "
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
            xnn_log_info(
                "Node %s (3x3 kernel) padding "
                "(top=%" PRIu32 ", right=%" PRIu32 ", bottom=%" PRIu32
                ", left=%" PRIu32
                ") "
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
            xnn_log_info(
                "Node %s (5x5 kernel) padding "
                "(top=%" PRIu32 ", right=%" PRIu32 ", bottom=%" PRIu32
                ", left=%" PRIu32
                ") "
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
      return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW |
             XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
    case xnn_node_type_binary_elementwise:
      if (node->binary_operator != xnn_binary_add &&
          node->binary_operator != xnn_binary_multiply) {
        // TODO(unassigned): We can probably handle any binary operator here?
        return false;
      }
      assert(node->num_inputs == 2);
      assert(node->num_outputs == 1);
      if (subgraph->values[node->inputs[0]].shape.num_dims != 4 ||
          subgraph->values[node->inputs[1]].shape.num_dims != 4) {
        xnn_log_info(
            "Node %s inputs shape is incompatible with sparse inference",
            xnn_node_type_to_string(node->type));
        return 0;
      }

      if (subgraph->values[node->inputs[0]].data != NULL) {
        // Check that the first input is representable as either a scalar, or a
        // vector
        size_t num_nonunit_dims = 0;
        for (uint32_t i = 0;
             i < subgraph->values[node->inputs[0]].shape.num_dims; i++) {
          if (subgraph->values[node->inputs[0]].shape.dim[i] != 1) {
            num_nonunit_dims += 1;
          }
        }
        if (num_nonunit_dims > 1) {
          return 0;
        }
      }

      if (subgraph->values[node->inputs[1]].data != NULL) {
        // Check that the second input is representable as either a scalar, or a
        // vector
        size_t num_nonunit_dims = 0;
        for (uint32_t i = 0;
             i < subgraph->values[node->inputs[0]].shape.num_dims; i++) {
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
        xnn_log_info(
            "Node %s inputs shape is incompatible with sparse inference",
            xnn_node_type_to_string(node->type));
        return 0;
      }
    case xnn_node_type_unary_elementwise:
      assert(node->num_inputs == 1);
      assert(node->num_outputs == 1);
      if (subgraph->values[node->inputs[0]].shape.num_dims == 4) {
        return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
      } else {
        xnn_log_info(
            "Node %s inputs shape is incompatible with sparse inference",
            xnn_node_type_to_string(node->type));
        return 0;
      }
    case xnn_node_type_static_mean:
    case xnn_node_type_static_sum:
      if (subgraph->values[node->inputs[0]].shape.num_dims == 4) {
        return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW |
               XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
      } else {
        xnn_log_info(
            "Node %s inputs shape is incompatible with sparse inference",
            xnn_node_type_to_string(node->type));
        return 0;
      }
    default:
      return false;
  }
}

void xnn_subgraph_rewrite_for_nchw(xnn_subgraph_t subgraph) {
  // Convert parts of the subgraph to NCHW for sparse inference
  // Step 1: detect NCHW-compatible Nodes
  // Step 2: detect NCHW-compatible clusters (run connected components graph
  // algorithm) Step 3: check that all NCHW-compatible Values are consumed only
  // by NCHW-compatible Nodes Step 4: switch Values' layout to NCHW
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    node->layout_flags = xnn_check_nchw_compatibility(subgraph, node);
    xnn_log_debug(
        "Node #%" PRIu32 ": %s (NCHW: %s, NHWC->NCHW: %s, NCHW->NHWC: %s)", n,
        xnn_node_type_to_string(node->type),
        node->layout_flags & XNN_LAYOUT_FLAG_COMPATIBLE_NCHW ? "yes" : "no",
        node->layout_flags & XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW ? "yes"
                                                                  : "no",
        node->layout_flags & XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC ? "yes"
                                                                  : "no");
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
          // Static data, skip this input value. Compatibility of this static
          // input with NCHW layout was validated during the initial NCHW
          // compatibility check for the Node.
          continue;
        }
        if (xnn_value_is_external(value->flags)) {
          // External value, invalid cluster
          node->layout_flags |= XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
          continue;
        }
        const uint32_t producer_id = value->producer;
        assert(producer_id != XNN_INVALID_NODE_ID);
        assert(producer_id < n);
        struct xnn_node* producer_node = &subgraph->nodes[producer_id];
        if ((producer_node->layout_flags &
             (XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW |
              XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) != 0 &&
            (producer_node->layout_flags &
             XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) == 0) {
          producer_node->layout_flags &= ~XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
          if (producer_node->cluster_leader != node->cluster_leader) {
            producer_node->cluster_leader = node->cluster_leader = math_max_u32(
                producer_node->cluster_leader, node->cluster_leader);
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

      if ((node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NCHW |
                                 XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC)) == 0) {
        continue;
      }

      for (uint32_t i = 0; i < node->num_inputs; i++) {
        const struct xnn_value* value = &subgraph->values[node->inputs[i]];
        if (value->data != NULL) {
          // Static data, skip this input value. Compatibility of this static
          // input with NCHW layout was validated during the initial NCHW
          // compatibility check for the Node.
          continue;
        }
        if (xnn_value_is_external(value->flags)) {
          // External value, invalid cluster
          node->layout_flags |= XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
          continue;
        }
        const uint32_t producer_id = value->producer;
        assert(producer_id != XNN_INVALID_NODE_ID);
        assert(producer_id < n);
        struct xnn_node* producer_node = &subgraph->nodes[producer_id];
        if ((producer_node->layout_flags &
             (XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW |
              XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) != 0 &&
            (producer_node->layout_flags &
             XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) == 0) {
          producer_node->layout_flags &= ~XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
          if (producer_node->cluster_leader != node->cluster_leader) {
            producer_node->cluster_leader = node->cluster_leader = math_max_u32(
                producer_node->cluster_leader, node->cluster_leader);
            update = true;
          }
        } else {
          node->layout_flags |= XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
        }
      }
    }
  }
  // Propagate XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER flags up to the cluster
  // leaders
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    subgraph->nodes[node->cluster_leader].layout_flags |=
        node->layout_flags & XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
  }
  // Check that all Values consumed by NCHW-compatible cluster don't have
  // NCHW-incompatible consumers
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if ((subgraph->nodes[node->cluster_leader].layout_flags &
         XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) != 0) {
      continue;
    }

    if ((node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC |
                               XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) == 0) {
      continue;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      struct xnn_value* value = &subgraph->values[node->inputs[i]];
      if (value->data != NULL) {
        // Static data, skip this input value because it doesn't have a producer
        // Node.
        continue;
      }
      assert(!xnn_value_is_external(value->flags));
      value->num_nchw_compatible_consumers += 1;
    }
  }
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if ((subgraph->nodes[node->cluster_leader].layout_flags &
         XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) != 0) {
      continue;
    }

    if ((node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC |
                               XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) == 0) {
      continue;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const struct xnn_value* value = &subgraph->values[node->inputs[i]];
      if (value->data != NULL) {
        // Static data, skip this input value because it doesn't have a producer
        // Node.
        continue;
      }
      assert(!xnn_value_is_external(value->flags));
      assert(value->num_nchw_compatible_consumers > 0);
      if (value->num_nchw_compatible_consumers != value->num_consumers) {
        subgraph->nodes[node->cluster_leader].layout_flags |=
            XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER;
      }
    }
  }
  // Evaluate if it is profitable to run the model as sparse:
  // - Compute the number of parameters and zeroes in 1x1 Convolution weights
  // - Disable sparse rewriting for clusters without 1x1 Convolutions
  // (num_params == 0)
  //   or with less than 2/3rd of zeroes in 1x1 Convolution filters
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if ((subgraph->nodes[node->cluster_leader].layout_flags &
         XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) != 0) {
      continue;
    }

    if ((node->type == xnn_node_type_convolution_2d &&
         max(node->params.convolution_2d.kernel_height,
             node->params.convolution_2d.kernel_width) == 1) ||
        node->type == xnn_node_type_fully_connected) {
      assert(node->num_inputs >= 2);

      const struct xnn_value* filter = &subgraph->values[node->inputs[1]];
      assert(filter->data != NULL);

      const size_t num_params =
          filter->shape.dim[0] * filter->shape.dim[filter->shape.num_dims - 1];
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
      xnn_log_debug("1x1 Convolution 2D Node #%" PRIu32 ": %zu / %zu sparsity",
                    n, num_zeroes, num_params);
      subgraph->nodes[node->cluster_leader].num_zeroes += num_zeroes;
    }
  }
  bool use_nchw_layout = false;
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if ((subgraph->nodes[node->cluster_leader].layout_flags &
         XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER) != 0) {
      continue;
    }

    if ((node->layout_flags & (XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC |
                               XNN_LAYOUT_FLAG_COMPATIBLE_NCHW)) == 0) {
      continue;
    }

    if (subgraph->nodes[node->cluster_leader].num_zeroes * 3 <=
        subgraph->nodes[node->cluster_leader].num_params * 2) {
      xnn_log_info("Node #%" PRIu32
                   ": sparse inference disabled: 1x1 Convolutions contain %zu "
                   "/ %zu zero weights",
                   n, subgraph->nodes[node->cluster_leader].num_zeroes,
                   subgraph->nodes[node->cluster_leader].num_params);
      continue;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      struct xnn_value* value = &subgraph->values[node->inputs[i]];
      if (value->data != NULL) {
        // Static data, skip this input value because it doesn't have a producer
        // Node.
        continue;
      }
      assert(!xnn_value_is_external(value->flags));
      assert(value->num_nchw_compatible_consumers > 0);
      assert(value->num_nchw_compatible_consumers == value->num_consumers);
      if (value->layout != xnn_layout_type_nchw) {
        value->layout = xnn_layout_type_nchw;
        xnn_log_info("set Value #%" PRIu32 " layout to NCHW", node->inputs[i]);
        use_nchw_layout = true;
      }
    }
  }
  if (use_nchw_layout) {
    xnn_log_info("XNNPACK has switched to sparse inference mode!");
  }
}

static bool any_values_fp32(xnn_subgraph_t subgraph,
                            const struct xnn_node* node) {
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

bool xnn_subgraph_rewrite_for_fp16(xnn_subgraph_t subgraph) {
  xnn_log_info("Analyzing subgraph for FP16 compatibility");

  // Count the number of consumers for each value.
  xnn_subgraph_analyze_consumers_and_producers(subgraph);

  // Convert tensors and operators in the subgraph to FP16
  // 1. Check that all operators in the subgraph are supported in FP16.
  // 2. Indicate values that must be converted to FP16.
  // 3. Replace FP32 Values with FP16 Values as Nodes' inputs/outputs.
  // 4. Insert FP32->FP16 Convert Nodes for external FP32 inputs and FP16->FP32
  // Convert Nodes for external outputs.

  const uint32_t num_original_values = subgraph->num_values;

  // Check that all operators in the subgraph are supported in FP16, bail out on
  // any unsupported one.
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if (node->type == xnn_node_type_invalid) {
      // Node was fused away, skip.
      continue;
    }

    if (!any_values_fp32(subgraph, node)) {
      xnn_log_warning("FP16 rewrite aborted: node #%" PRIu32
                      " (%s) is not FP32",
                      n, xnn_node_type_to_string(node->type));
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
      case xnn_node_type_static_reduce_min:
      case xnn_node_type_static_reduce_max:
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
        xnn_log_warning("FP16 rewrite aborted: node #%" PRIu32
                        " (%s) is not supported for FP16 inference",
                        n, xnn_node_type_to_string(node->type));
        return false;
    }
  }

  // Annotate Values to be converted to FP16 as FP16-compatible.
  // Note that static weights in [Depthwise] Convolution, Fully Connected Nodes
  // remain FP32, they will be converted to FP16 during weight repacking when
  // the operator is created.
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    switch (node->type) {
      case xnn_node_type_deconvolution_2d:
      case xnn_node_type_depthwise_convolution_2d:
        if (subgraph->values[node->inputs[0]].datatype == xnn_datatype_fp32) {
          subgraph->values[node->inputs[0]].fp16_compatible = true;
        }
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
        } else if (subgraph->values[node->inputs[0]].datatype ==
                       xnn_datatype_fp32 &&
                   (node->params.inlined_lhs_packing.packed_input_datatype ==
                        xnn_datatype_qdint8 ||
                    node->params.inlined_lhs_packing.packed_input_datatype ==
                        xnn_datatype_qduint8 ||
                    node->params.inlined_lhs_packing.packed_input_datatype ==
                        xnn_datatype_qpint8)) {
          subgraph->values[node->inputs[0]].fp16_compatible = true;
          subgraph->values[node->outputs[0]].fp16_compatible = true;
        } else if ((subgraph->values[node->inputs[0]].datatype ==
                        xnn_datatype_fp32 ||
                    subgraph->values[node->inputs[0]].datatype ==
                        xnn_datatype_pfp32) &&
                   (subgraph->values[node->inputs[1]].datatype ==
                        xnn_datatype_fp16 ||
                    subgraph->values[node->inputs[1]].datatype ==
                        xnn_datatype_fp32) &&
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
      if (xnn_value_is_static(value->allocation_type)) {
        assert(value->producer == XNN_INVALID_NODE_ID);
        const size_t fp16_size =
            xnn_tensor_get_size_by_id(subgraph, n) / 2 + XNN_EXTRA_BYTES;
        value->fp16_temp_data = xnn_allocate_zero_memory(fp16_size);
        if (value->fp16_temp_data == NULL) {
          xnn_log_error("failed to allocate %zu bytes for fp16 tensor data",
                        (size_t)fp16_size);
          goto error;
        }
      } else if (xnn_value_is_external(value->flags)) {
        struct xnn_value* fp16_value =
            xnn_subgraph_new_internal_value(subgraph);
        if (fp16_value == NULL) {
          xnn_log_error(
              "FP16 rewrite aborted: failed to allocate value for external "
              "input/output");
          goto error;
        } else {
          // Recompute value due to potential reallocation in
          // xnn_subgraph_new_internal_value
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
          fp16_value->producer = XNN_INVALID_NODE_ID;
          fp16_value->first_consumer = XNN_INVALID_NODE_ID;
          fp16_value->num_consumers = 0;
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
        // This value isn't always an external input, it could be an external
        // output of the current subgraph (due to partition), and be
        // simultaneously consumed by the current node.
        if (xnn_value_is_external_input(value->flags)) {
          num_external_inputs += 1;
        }
      }
    }
    for (uint32_t o = 0; o < node->num_outputs; o++) {
      const struct xnn_value* value = &subgraph->values[node->outputs[o]];
      if (value->fp16_id != XNN_INVALID_VALUE_ID) {
        assert(value->datatype == xnn_datatype_fp32);
        assert(subgraph->values[value->fp16_id].datatype == xnn_datatype_fp16);
        assert(xnn_value_is_external_output(value->flags));
        num_external_outputs += 1;
      }
    }
  }
  xnn_log_debug("Discovered %" PRIu32 " external inputs and %" PRIu32
                " external outputs",
                num_external_inputs, num_external_outputs);

  // Attempt to allocate memory for the Convert nodes.
  const uint32_t num_original_nodes = subgraph->num_nodes;
  if (xnn_subgraph_add_nodes(subgraph,
                             num_external_inputs + num_external_outputs) !=
      xnn_status_success) {
    xnn_log_error(
        "FP16 rewrite aborted: failed to allocate node for external "
        "input/output");
    goto error;
  }

  // From this point the subgraph and tensor data get mutated, clean failure is
  // no longer an option.

  // Replace FP32 Values in Nodes' inputs/outputs with FP16 Values.
  // - FP32 values of static tensors get converted in a new data buffer.
  // - For external inputs and outputs we create same-shaped FP16 Values and use
  // those instead.
  // - Values that are neither static nor external are converted to FP16
  // in-place
  for (uint32_t n = 0; n < num_original_values; n++) {
    struct xnn_value* value = &subgraph->values[n];
    if (value->fp16_compatible) {
      if (xnn_value_is_static(value->allocation_type)) {
        assert(value->datatype == xnn_datatype_fp32);
        const size_t num_elements = xnn_shape_multiply_all_dims(&value->shape);
        xnn_run_unary_elementwise_nc(
            xnn_unary_convert, xnn_datatype_fp32, xnn_datatype_fp16,
            /*params=*/NULL, /*input_quantization=*/NULL,
            /*output_quantization=*/NULL, 0, num_elements, 1, 1, 1, NULL,
            value->data, value->fp16_temp_data);
        // Remember pointer to the original fp32 data, nodes like convolution
        // need fp32 weights/biases.
        value->fp32_data = value->data;
        value->data = value->fp16_temp_data;
        value->fp16_temp_data = NULL;
        value->datatype = xnn_datatype_fp16;
        xnn_log_debug("FP16 rewrite: converted static FP32 tensor #%" PRIu32
                      " to FP16 in new buffer",
                      n);
      } else if (xnn_value_is_external(value->flags)) {
        assert(value->datatype == xnn_datatype_fp32);
        assert(value->fp16_id != XNN_INVALID_VALUE_ID);
        value->producer = XNN_INVALID_NODE_ID;
        value->num_consumers = 0;
        xnn_log_debug("FP16 rewrite: created FP16 tensor #%" PRIu32
                      " for external FP32 tensor #%" PRIu32,
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

  // Switch the nodes consuming/generated converted `fp32` inputs/outputs to
  // their `fp16` values.
  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if (node->type == xnn_node_type_invalid) {
      // Node was fused away, skip.
      continue;
    }

    // Fix up anything node-type specific.
    switch (node->type) {
      case xnn_node_type_static_constant_pad:
        node->params.static_pad.padding_value = fp16_ieee_from_fp32_value(
            uint32_as_float(node->params.static_pad.padding_value));
        break;
      case xnn_node_type_batch_matrix_multiply:
      case xnn_node_type_fully_connected: {
        // Patch up any LHS packing of fully-connected nodes, if needed.
        if (node->flags & XNN_FLAG_INLINE_LHS_PACKING) {
          switch (node->params.inlined_lhs_packing.packed_input_datatype) {
            case xnn_datatype_pfp32:
              // Switch from packed `fp32` to packed `fp16`.
              node->params.inlined_lhs_packing.packed_input_datatype =
                  xnn_datatype_pfp16;
              break;
            case xnn_datatype_qpint8:
              // Convert from `qpint8` back to `qdint8` since we don't have a
              // `qpint8` packing function for `f16` inputs.
              node->params.inlined_lhs_packing.packed_input_datatype =
                  xnn_datatype_qdint8;
              break;
            default:
              break;
          }
        } else if (subgraph->values[node->inputs[0]].datatype ==
                   xnn_datatype_qpint8) {
          subgraph->values[node->inputs[0]].datatype = xnn_datatype_qdint8;
        }
      } break;
      default:
        break;
    }

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const uint32_t fp16_id = subgraph->values[node->inputs[i]].fp16_id;
      if (fp16_id != XNN_INVALID_VALUE_ID) {
        struct xnn_value* fp16_value = &subgraph->values[fp16_id];
        assert(fp16_value->fp32_id == node->inputs[i]);
        if (fp16_value->first_consumer == XNN_INVALID_NODE_ID) {
          fp16_value->first_consumer = n;
        }
        fp16_value->num_consumers++;
        node->inputs[i] = fp16_id;
      }
    }
    for (uint32_t o = 0; o < node->num_outputs; o++) {
      const uint32_t fp32_id = node->outputs[o];
      const uint32_t fp16_id = subgraph->values[fp32_id].fp16_id;
      if (fp16_id != XNN_INVALID_VALUE_ID) {
        struct xnn_value* fp16_value = &subgraph->values[fp16_id];
        if (fp16_value->first_consumer == XNN_INVALID_NODE_ID &&
            fp16_value->producer == XNN_INVALID_NODE_ID) {
          assert(fp16_value->fp32_id == fp32_id);
        } else {
          // Prevent double assignments by creating a new copy of the output
          // value if it has already been written to.
          fp16_value = xnn_subgraph_new_internal_value(subgraph);
          xnn_value_copy(fp16_value, &subgraph->values[fp16_id]);
          fp16_value->first_consumer = XNN_INVALID_NODE_ID;
          fp16_value->num_consumers = 0;
          subgraph->values[fp32_id].fp16_id = fp16_value->id;
        }
        node->outputs[o] = fp16_value->id;
        fp16_value->producer = n;
      }
    }
  }

  struct xnn_node* output_node = &subgraph->nodes[subgraph->num_nodes - 1];
  for (uint32_t n = num_original_nodes; n != 0; n--) {
    const struct xnn_node* node = &subgraph->nodes[n - 1];
    // Insert Convert nodes for outputs
    for (uint32_t o = 0; o < node->num_outputs; o++) {
      const struct xnn_value* value = &subgraph->values[node->outputs[o]];
      const uint32_t fp32_id = value->fp32_id;
      if (fp32_id != XNN_INVALID_VALUE_ID &&
          subgraph->values[fp32_id].fp16_id == value->id) {
        xnn_log_debug("Inserted FP16->FP32 Convert Node from tensor #%" PRIu32
                      " to output tensor #%" PRIu32,
                      value->id, fp32_id);
        const uint32_t output_node_id = output_node->id;
        assert(output_node >= subgraph->nodes);
        xnn_node_clear(output_node);
        output_node->id = output_node_id;
        xnn_init_convert_node(output_node, value->id, fp32_id, 0 /* flags */);
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
      const uint32_t fp32_id = value->fp32_id;
      if (fp32_id != XNN_INVALID_VALUE_ID &&
          subgraph->values[fp32_id].first_consumer == n - 1) {
        // Only insert convert nodes if the value actually is an external input.
        // This value could be an external output, if that's the case, we have
        // already inserted a convert node in loop above for outputs.
        if (xnn_value_is_external_input(subgraph->values[fp32_id].flags)) {
          xnn_log_debug("Inserted FP32->FP16 Convert Node from tensor #%" PRIu32
                        " to tensor #%" PRIu32,
                        fp32_id, value->id);
          const uint32_t output_node_id = output_node->id;
          assert(output_node >= subgraph->nodes);
          xnn_node_clear(output_node);
          output_node->id = output_node_id;
          xnn_init_convert_node(output_node, fp32_id, value->id, 0 /* flags */);
          output_node -= 1;
        }
      }
    }
  }

  xnn_log_info("XNNPACK has switched to FP16 inference mode!");

  return true;

error:
  for (uint32_t n = 0; n < subgraph->num_values; n++) {
    struct xnn_value* value = &subgraph->values[n];
    // Deallocate extra memory used during static tensor rewrite.
    if (value->fp16_temp_data != NULL) {
      xnn_release_memory(value->fp16_temp_data);
    }
    // Revert marking values as FP16-compatible, as xnn_delete_subgraph() may
    // assume ownership of those that are.
    value->fp16_compatible = false;
  }

  // Clear the fp16 values created for external inputs and outputs.
  for (uint32_t n = num_original_values; n < subgraph->num_values; n++) {
    xnn_value_clear(&subgraph->values[n]);
  }

  return false;
}

static void xnn_node_replace_output(struct xnn_node* node,
                                    uint32_t old_output_id,
                                    uint32_t new_output_id) {
  for (size_t i = 0; i < node->num_outputs; i++) {
    if (node->outputs[i] == old_output_id) {
      node->outputs[i] = new_output_id;
    }
  }
}

static bool is_clamp(const struct xnn_node* node) {
  return node->type == xnn_node_type_unary_elementwise &&
         node->unary_operator == xnn_unary_clamp;
}

static bool has_clamp(const struct xnn_node* node) {
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

enum xnn_status xnn_subgraph_fusion(xnn_subgraph_t subgraph) {
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
        xnn_log_info("fuse Clamp Node #%" PRIu32
                     " into upstream Node #%" PRIu32,
                     consumer_id, producer_id);
        assert(producer->num_outputs == 1);
        assert(consumer->num_inputs == 1);
        assert(consumer->num_outputs == 1);

        const uint32_t fused_output_id = consumer->outputs[0];
        assert(fused_output_id < subgraph->num_values);
        subgraph->values[fused_output_id].producer = producer_id;
        producer->outputs[0] = fused_output_id;

        producer->activation.output_min = math_max_f32(
            producer->activation.output_min, consumer->activation.output_min);
        producer->activation.output_max = math_min_f32(
            producer->activation.output_max, consumer->activation.output_max);
        producer->params.unary.clamp.min = math_max_f32(
            producer->params.unary.clamp.min, consumer->params.unary.clamp.min);
        producer->params.unary.clamp.max = math_min_f32(
            producer->params.unary.clamp.max, consumer->params.unary.clamp.max);

        xnn_node_clear(consumer);
        xnn_value_clear(value);
      }
      // Try to fuse Constant Pad node downstream into [Depthwise] Convolution
      // 2D Node
      if (producer->type == xnn_node_type_static_constant_pad) {
        assert(producer->num_inputs == 1);
        assert(producer->num_outputs == 1);
        const bool is_spatial_2d_padding =
            value->shape.num_dims == 4 &&
            (producer->params.static_pad.pre_paddings[0] |
             producer->params.static_pad.post_paddings[0] |
             producer->params.static_pad.pre_paddings[3] |
             producer->params.static_pad.post_paddings[3]) == 0;
        const enum xnn_datatype padding_datatype =
            subgraph->values[producer->outputs[0]].datatype;
        const uint32_t padding_value =
            producer->params.static_pad.padding_value;
        const bool is_zero_padding =
            (padding_datatype == xnn_datatype_fp32 && padding_value == 0) ||
            ((padding_datatype == xnn_datatype_qint8 ||
              padding_datatype == xnn_datatype_quint8) &&
             padding_value ==
                 (uint32_t)(uint8_t)subgraph->values[producer->outputs[0]]
                     .quantization.zero_point);
        switch (consumer->type) {
          case xnn_node_type_convolution_2d:
            if (is_spatial_2d_padding && is_zero_padding &&
                !(consumer->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING)) {
              xnn_log_info("fuse Constant Pad Node #%" PRIu32
                           " into Convolution 2D Node #%" PRIu32,
                           consumer_id, producer_id);
              assert(consumer->num_inputs >= 1);
              assert(consumer->inputs[0] == producer->outputs[0]);

              consumer->params.convolution_2d.input_padding_top +=
                  producer->params.static_pad.pre_paddings[1];
              consumer->params.convolution_2d.input_padding_right +=
                  producer->params.static_pad.post_paddings[2];
              consumer->params.convolution_2d.input_padding_bottom +=
                  producer->params.static_pad.post_paddings[1];
              consumer->params.convolution_2d.input_padding_left +=
                  producer->params.static_pad.pre_paddings[2];

              consumer->inputs[0] = producer->inputs[0];

              const uint32_t fused_input_id = producer->inputs[0];
              assert(fused_input_id < subgraph->num_values);
              if (subgraph->values[fused_input_id].first_consumer ==
                  producer_id) {
                subgraph->values[fused_input_id].first_consumer = consumer_id;
              }

              xnn_node_clear(producer);
              xnn_value_clear(value);
            }
            break;
          case xnn_node_type_depthwise_convolution_2d:
            if (is_spatial_2d_padding && is_zero_padding &&
                !(consumer->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING)) {
              xnn_log_info("fuse Constant Pad Node #%" PRIu32
                           " into Depthwise Convolution 2D Node #%" PRIu32,
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
              if (subgraph->values[fused_input_id].first_consumer ==
                  producer_id) {
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

      // Try to fuse copy upstream. Copy can be fused upstream as long as this
      // value is internal. E.g. ---> (N1) --- value ---> (Copy) ---> v1 If
      // value is persistent or external, fusing copy upstream into N1 will skip
      // the write to value, N1 will write to v1 instead, which is wrong.
      if (consumer->type == xnn_node_type_copy &&
          xnn_value_is_valid(value->type) && xnn_value_is_internal(value) &&
          can_reorder_use(subgraph, consumer->outputs[0], producer_id,
                          consumer_id)) {
        xnn_log_info("value %d fuse Copy Node #%" PRIu32
                     " into upstream %s Node #%" PRIu32,
                     value->id, consumer->id,
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
      // If value is external or persistent, we cannot simply remove the copy,
      // since we need to write to value.
      if (producer->type == xnn_node_type_copy &&
          xnn_value_is_valid(value->type) && xnn_value_is_internal(value) &&
          can_reorder_use(subgraph, producer->inputs[0], producer_id,
                          consumer_id)) {
        // We need to check that value is valid here because value could have
        // been cleared by a previous optimization, this can happen if we have a
        // chain of Copy(s), e.g.:
        // ---v1--> (Copy1) ---v2--> (Copy2) ---v3--> (Copy3) ---v4-->
        // v2 could have been cleared when we fused Copy2 upstream into Copy1,
        // so v2 isn't valid anymore, but since v2's producer is also a Copy, we
        // will incorrectly try to fuse Copy1 downstream into Copy2 (again).
        xnn_log_info("value %d fuse Copy Node #%" PRIu32
                     " into downstream %s Node #%" PRIu32,
                     value->id, producer->id,
                     xnn_node_type_to_string(consumer->type), consumer->id);
        assert(producer->num_outputs == 1);
        assert(producer->num_inputs == 1);
        const uint32_t copy_input_id = producer->inputs[0];
        const uint32_t copy_output_id = producer->outputs[0];
        bool found_consumer_input = false;
        for (size_t i = 0; i < consumer->num_inputs; i++) {
          if (consumer->inputs[i] == copy_output_id) {
            consumer->inputs[i] = copy_input_id;
            ;
            found_consumer_input = true;
            // TODO(b/254734644): A consumer can only consume this value once,
            // since we asserted earlier that value has only 1 consumer, so we
            // can break here as there will be no other consumer inputs that has
            // the same id.
            break;
          }
        }
        (void)found_consumer_input;  // Silence unused variable warning in
                                     // non-debug.
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

// Returns true if `value` is a broadcast of a single constant.
static bool is_broadcasted_static(const struct xnn_value* value) {
  // It really shouldn't be possible for a value with static data to also be
  // an external input or have a producer. But some graphs do this (somehow), so
  // we need to defend against these malformed cases.
  return value->data != NULL &&
         value->allocation_type == xnn_allocation_type_static &&
         !xnn_value_is_external_input(value->flags) &&
         value->producer == XNN_INVALID_NODE_ID &&
         xnn_shape_multiply_all_dims(&value->shape) == 1;
}

static bool set_contains(const uint32_t* set, uint32_t set_size, uint32_t x) {
  for (uint32_t i = 0; i < set_size; i++) {
    if (set[i] == x) {
      return true;
    }
  }
  return false;
}

// Returns the Value ID of a unary input. Binary operators with one constant
// operand are considered unary operators by this function, and return the non-
// constant input. `unary_values` is a set of values that have already been
// determined to be part of a unary elementwise function.
static uint32_t is_pure_unary_elementwise(xnn_subgraph_t subgraph,
                                          const struct xnn_node* node,
                                          const uint32_t* unary_values,
                                          uint32_t num_unary_values) {
  switch (node->type) {
    case xnn_node_type_unary_elementwise:
      assert(node->num_inputs >= 1);
      return node->inputs[0];
    case xnn_node_type_binary_elementwise: {
      const uint32_t input_0_id = node->inputs[0];
      const uint32_t input_1_id = node->inputs[1];
      const struct xnn_value* input_0 = &subgraph->values[input_0_id];
      const struct xnn_value* input_1 = &subgraph->values[input_1_id];
      assert(node->num_inputs == 2);
      if (is_broadcasted_static(input_0) &&
          !xnn_value_is_static(input_1->allocation_type)) {
        return input_1_id;
      } else if (is_broadcasted_static(input_1) &&
                 !xnn_value_is_static(input_0->allocation_type)) {
        return input_0_id;
      } else if (set_contains(unary_values, num_unary_values, input_0_id) &&
                 set_contains(unary_values, num_unary_values, input_1_id)) {
        // This is a unary elementwise operator if we've determined that both
        // inputs are part of the same unary elementwise function. It doesn't
        // matter which operand we return as long as it is in the `unary_values`
        // set (which both are).
        return input_0_id;
      } else {
        return XNN_INVALID_VALUE_ID;
      }
    }
    default:
      return XNN_INVALID_VALUE_ID;
  }
}

// We will not fuse more than this many unary elementwise ops into a LUT in the
// function below.
#define XNN_MAX_UNARY_FUSION_NODES 10
#define XNN_MAX_UNARY_FUSION_VALUES (2 * XNN_MAX_UNARY_FUSION_NODES + 1)

// Find the index of `src_id` in `value_map`. Sets `is_new` to `true` if the
// value was inserted into the map, `false` if it was found in the map.
static uint32_t map_value_id(uint32_t* value_map, uint32_t src_id,
                             bool* is_new) {
  for (uint32_t dst_id = 0;; ++dst_id) {
    if (value_map[dst_id] == src_id) {
      if (is_new) {
        *is_new = false;
      }
      return dst_id;
    } else if (value_map[dst_id] == XNN_INVALID_VALUE_ID) {
      value_map[dst_id] = src_id;
      if (is_new) {
        *is_new = true;
      }
      return dst_id;
    }
  }
  XNN_UNREACHABLE;
}

// Copy a value to a new subgraph, if it doesn't exist already.
static uint32_t copy_value_to_static_subgraph(xnn_subgraph_t src_subgraph,
                                              const struct xnn_value* src_value,
                                              uint32_t* value_map,
                                              xnn_subgraph_t dst_subgraph) {
  bool is_new = false;
  uint32_t dst_id = map_value_id(value_map, src_value->id, &is_new);
  if (is_new) {
    assert(dst_id == dst_subgraph->num_values);
    assert(dst_id < dst_subgraph->num_reserved_values);
    struct xnn_value* dst_value = &dst_subgraph->values[dst_id];
    dst_subgraph->num_values++;
    *dst_value = *src_value;
    dst_value->id = dst_id;
    dst_value->producer = XNN_INVALID_NODE_ID;
    dst_value->first_consumer = XNN_INVALID_NODE_ID;
  }
  return dst_id;
}

// Copy a Node and its Values to a new subgraph, maintaining a map of Value IDs
// as it goes.
static struct xnn_node* copy_node_to_static_subgraph(
    xnn_subgraph_t src_subgraph, const struct xnn_node* src_node,
    uint32_t* value_map, xnn_subgraph_t dst_subgraph) {
  assert(dst_subgraph->num_nodes < dst_subgraph->num_reserved_nodes);
  struct xnn_node* dst_node = &dst_subgraph->nodes[dst_subgraph->num_nodes++];
  xnn_node_copy(dst_node, src_node);

  for (size_t i = 0; i < src_node->num_inputs; i++) {
    const struct xnn_value* value = &src_subgraph->values[src_node->inputs[i]];
    const uint32_t dst_id = copy_value_to_static_subgraph(
        src_subgraph, value, value_map, dst_subgraph);
    dst_node->inputs[i] = dst_id;
  }
  for (size_t i = 0; i < src_node->num_inputs; i++) {
    const struct xnn_value* value = &src_subgraph->values[src_node->outputs[i]];
    const uint32_t dst_id = copy_value_to_static_subgraph(
        src_subgraph, value, value_map, dst_subgraph);
    dst_node->outputs[i] = dst_id;
  }
  return dst_node;
}

// Pass 0:1:256 to the input of the subgraph representing a unary pure
// elementwise function, storing the result in `lut`.
static enum xnn_status run_subgraph_to_make_lut(xnn_subgraph_t subgraph,
                                                uint32_t input_id,
                                                uint32_t output_id,
                                                uint8_t* lut) {
  xnn_log_debug("Running unary subgraph to make LUT");

  xnn_runtime_t runtime;
  enum xnn_status status = xnn_create_runtime_v4(
      subgraph, NULL, NULL, NULL, XNN_FLAG_NO_OPERATOR_FUSION, &runtime);
  if (status != xnn_status_success) {
    return status;
  }

  const size_t ramp_size = 256;
  uint8_t ramp[256];
  for (size_t i = 0; i < 256; i++) {
    ramp[i] = i;
  }

  status = xnn_reshape_external_value(runtime, input_id, 1, &ramp_size);
  if (status != xnn_status_success) {
    goto fail;
  }

  status = xnn_reshape_runtime(runtime);
  if (status != xnn_status_success) {
    goto fail;
  }

  struct xnn_external_value externals[2];
  externals[0].id = input_id;
  externals[0].data = ramp;
  externals[1].id = output_id;
  externals[1].data = lut;

  status = xnn_setup_runtime(runtime, 2, externals);
  if (status != xnn_status_success) {
    goto fail;
  }

  status = xnn_invoke_runtime(runtime);

fail:
  xnn_delete_runtime(runtime);
  return status;
}

static bool replace_node_with_lut(xnn_subgraph_t subgraph,
                                  struct xnn_node* node, uint32_t input_id,
                                  uint32_t unary_input_id,
                                  xnn_subgraph_t unary_subgraph) {
  const uint32_t unary_output_id =
      unary_subgraph->nodes[unary_subgraph->num_nodes - 1].outputs[0];
  assert(unary_input_id != XNN_INVALID_VALUE_ID);
  assert(unary_output_id != XNN_INVALID_VALUE_ID);
  unary_subgraph->values[unary_input_id].flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
  unary_subgraph->values[unary_output_id].flags |=
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  unary_subgraph->values[unary_input_id].allocation_type =
      xnn_allocation_type_external;
  unary_subgraph->values[unary_output_id].allocation_type =
      xnn_allocation_type_external;

  uint8_t* lut = xnn_allocate_memory(256 * sizeof(uint8_t));
  if (lut == NULL) {
    xnn_log_error("failed to allocate LUT");
    return false;
  }
  enum xnn_status status = run_subgraph_to_make_lut(
      unary_subgraph, unary_input_id, unary_output_id, lut);
  if (status != xnn_status_success) {
    // Failed to generate the LUT, abandon this fusion.
    xnn_release_memory(lut);
    return false;
  }

  // We don't have any other way to store a dynamic allocation in a subgraph
  // except in a value.
  struct xnn_value* lut_value = xnn_subgraph_new_internal_value(subgraph);
  lut_value->flags |= XNN_VALUE_FLAG_NEEDS_CLEANUP;
  lut_value->data = lut;
  lut_value->datatype = xnn_datatype_quint8;
  lut_value->allocation_type = xnn_allocation_type_static;
  lut_value->shape.num_dims = 1;
  lut_value->shape.dim[0] = 256;
  lut_value->type = xnn_value_type_dense_tensor;

  // Clear the inputs that were replaced by a fused op.
  for (uint32_t i = 0; i < node->num_inputs; i++) {
    struct xnn_value* input = &subgraph->values[node->inputs[i]];
    if (input->num_consumers == 1 && input->first_consumer == node->id) {
      xnn_value_clear(input);
    }
  }

  xnn_define_unary_elementwise_lut_in_place(node, input_id, node->outputs[0],
                                            lut_value->id);
  return true;
}

void reshape_for_lut(struct xnn_value* value) {
  value->shape.num_dims = 1;
  value->shape.dim[0] = 256;
}

void xnn_subgraph_fuse_unary_quantized_into_lut(xnn_subgraph_t subgraph) {
  // Find sequences of operators that are unary, quantized, elementwise, and
  // pure functions. These can be fused into a single LUT op. Examples:
  // - softsign(x) = x/(1 + abs(x))
  // - softplus(x) = log(1 + exp(x))
  // We allow intermediate values to be datatypes other than quantized (and
  // allow convert ops), but the input and output values of the sequence must be
  // quantized.

  for (uint32_t n = 0; n < subgraph->num_nodes; n++) {
    struct xnn_node* node = &subgraph->nodes[n];
    if (node->type == xnn_node_type_invalid) {
      // Node was fused away, skip.
      continue;
    }

    const uint32_t input_id =
        is_pure_unary_elementwise(subgraph, node, NULL, 0);
    if (input_id == XNN_INVALID_VALUE_ID) {
      continue;
    }
    const struct xnn_value* input_value = &subgraph->values[input_id];
    if (input_value->datatype == xnn_datatype_invalid ||
        xnn_datatype_size_bits(input_value->datatype) != 8) {
      // This value is not a quantized 8-bit value, it can't be the input to a
      // LUT.
      continue;
    }

    // This node is a pure unary elementwise op with a quantized input. Can we
    // fuse more ops with this one? Here, we assume the ops are in order, so
    // this op must be the first in a chain we could fuse.
    // We're going to build a subgraph for all the nodes we want to fuse.
    struct xnn_subgraph unary_subgraph;
    memset(&unary_subgraph, 0, sizeof(unary_subgraph));
    struct xnn_value unary_values[XNN_MAX_UNARY_FUSION_VALUES];
    uint32_t value_map[XNN_MAX_UNARY_FUSION_VALUES];
    for (size_t i = 0; i < XNN_MAX_UNARY_FUSION_VALUES; i++) {
      value_map[i] = XNN_INVALID_VALUE_ID;
    }
    struct xnn_node unary_nodes[XNN_MAX_UNARY_FUSION_NODES];
    for (size_t i = 0; i < XNN_MAX_UNARY_FUSION_NODES; i++) {
      unary_nodes[i].id = i;
    }
    unary_subgraph.values = &unary_values[0];
    unary_subgraph.num_reserved_values = XNN_MAX_UNARY_FUSION_VALUES;
    unary_subgraph.nodes = &unary_nodes[0];
    unary_subgraph.num_reserved_nodes = XNN_MAX_UNARY_FUSION_NODES;

    // Remember the nodes we put in the unary subgraph.
    struct xnn_node* nodes_to_fuse[XNN_MAX_UNARY_FUSION_NODES];
    for (size_t i = 0; i < XNN_MAX_UNARY_FUSION_NODES; i++) {
      nodes_to_fuse[i] = NULL;
    }

    do {
      // Add the node we have to the unary subgraph.
      nodes_to_fuse[unary_subgraph.num_nodes] = node;
      const struct xnn_node* new_node = copy_node_to_static_subgraph(
          subgraph, node, value_map, &unary_subgraph);

      assert(node->num_outputs == 1);
      const struct xnn_value* output = &subgraph->values[node->outputs[0]];
      if (output->num_consumers != 1 ||
          output->first_consumer == XNN_INVALID_NODE_ID) {
        // Don't try to fuse nodes that don't have exactly one valid consumer.
        break;
      }
      assert(!xnn_value_is_external_output(output->flags));

      reshape_for_lut(&unary_subgraph.values[new_node->outputs[0]]);

      // Include the consumer in the unary subgraph.
      node = &subgraph->nodes[output->first_consumer];
    } while (is_pure_unary_elementwise(subgraph, node, value_map,
                                       XNN_MAX_UNARY_FUSION_VALUES) !=
                 XNN_INVALID_VALUE_ID &&
             unary_subgraph.num_nodes < XNN_MAX_UNARY_FUSION_NODES);

    // We need the output to be an 8-bit LUT element. Go back through the unary
    // subgraph until we find one.
    while (unary_subgraph.num_nodes > 1) {
      const struct xnn_node* unary_node =
          &unary_subgraph.nodes[unary_subgraph.num_nodes - 1];
      assert(unary_node->num_outputs == 1);
      const uint32_t unary_output_id = unary_node->outputs[0];
      struct xnn_value* unary_output = &unary_subgraph.values[unary_output_id];
      if (unary_output->datatype != xnn_datatype_invalid &&
          xnn_datatype_size_bits(unary_output->datatype) == 8) {
        break;
      }
      // Remove this node from the subgraph.
      xnn_value_clear(unary_output);
      unary_subgraph.num_nodes--;
    }

    if (unary_subgraph.num_nodes > 1) {
      // Update the last node of the fusion (the node we replace).
      node = nodes_to_fuse[unary_subgraph.num_nodes - 1];

      // Replace the fused nodes with a LUT op.
      const uint32_t unary_input_id = map_value_id(value_map, input_id, NULL);
      reshape_for_lut(&unary_subgraph.values[unary_input_id]);
      if (replace_node_with_lut(subgraph, node, input_id, unary_input_id,
                                &unary_subgraph)) {
        // We replaced this subgraph with a LUT, clear out the old values and
        // nodes.
        for (uint32_t i = 0; i + 1 < unary_subgraph.num_nodes; i++) {
          struct xnn_node* fused_node = nodes_to_fuse[i];
          assert(fused_node->num_outputs == 1);
          struct xnn_value* fused_output =
              &subgraph->values[fused_node->outputs[0]];
          xnn_log_info("Value %d fuse Node #%" PRIu32
                       " into downstream quantized LUT Node #%" PRIu32,
                       fused_output->id, fused_node->id, node->id);

          if (i > 0) {
            // Remove this consumer from the inputs.
            for (uint32_t i = 0; i < fused_node->num_inputs; i++) {
              struct xnn_value* fused_input =
                  &subgraph->values[fused_node->inputs[i]];
              assert(!xnn_value_is_external_input(fused_input->flags));
              fused_input->num_consumers--;
              if (fused_input->num_consumers == 0) {
                xnn_value_clear(fused_input);
              }
            }
          }

          // We only need to clear output values. Input values could be used by
          // other ops, and the ones that are outputs of another node in the
          // fusion will be cleared here.
          xnn_value_clear(fused_output);
          xnn_node_clear(fused_node);
        }
      }
    }
  }
}

void xnn_subgraph_clean_up(xnn_subgraph_t subgraph) {
  // Count the number of consumers for each value.
  xnn_subgraph_analyze_consumers_and_producers(subgraph);

  // Clear unreferenced values.
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    if (value->type == xnn_value_type_invalid) {
      continue;
    }

    if (!xnn_value_is_external_input(value->flags) &&
        value->num_consumers == 0) {
      if (value->producer != XNN_INVALID_NODE_ID) {
        struct xnn_node* producer = &subgraph->nodes[value->producer];
        if (producer->num_outputs == 1) {
          xnn_node_clear(&subgraph->nodes[value->producer]);
        }
      }
      xnn_value_clear(value);
    }
  }

  // Compact the nodes and sort them hierarchically (stably), if needed. The
  // temporary memory needed for `nodes_map` and `values_ready` is allocated as
  // a single block to reduce overheads.
  uint32_t* nodes_map =
      xnn_allocate_memory(sizeof(uint32_t) * subgraph->num_nodes +
                          sizeof(bool) * subgraph->num_values);
  bool* values_ready = (bool*)&nodes_map[subgraph->num_nodes];
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    values_ready[i] = value->producer == XNN_INVALID_NODE_ID ||
                      xnn_value_is_external_input(value->flags);
  }
  uint32_t left = 0;
  uint32_t num_invalid_nodes = 0;
  bool changes = false;
  while (left + num_invalid_nodes < subgraph->num_nodes) {
    for (uint32_t i = left; i < subgraph->num_nodes; i++) {
      struct xnn_node* node = &subgraph->nodes[i];

      // Skip over invalid nodes.
      if (node->type == xnn_node_type_invalid) {
        num_invalid_nodes++;
        continue;
      }

      // Check whether all inputs to this node have been produced.
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
          if (subgraph->nodes[left].type == xnn_node_type_invalid) {
            node->type = xnn_node_type_invalid;
          } else {
            memcpy(&subgraph->nodes[left + 1], &subgraph->nodes[left],
                   (i - left) * sizeof(struct xnn_node));
          }
          subgraph->nodes[left] = tmp_node;
        }
        left++;
      }
    }
  }

  // Update the node IDs in the subgraph values if they have changed.
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
    subgraph->num_nodes = left;
  }

  // Release temporarily allocated memory.
  xnn_release_memory(nodes_map);
}

static bool convert_gemm_to_qduint8(
    const enum xnn_datatype input_datatype,
    const enum xnn_node_type consumer_type,
    const enum xnn_datatype consumer_weights_type) {
  // Identify the `qdint8` and `qduint8` configs for the consumers of this op.
  const struct xnn_gemm_config* original_config = NULL;
  const struct xnn_gemm_config* unsigned_config = NULL;
  if (input_datatype == xnn_datatype_fp32) {
    if (consumer_weights_type == xnn_datatype_qcint4) {
      original_config = xnn_init_qd8_f32_qc4w_gemm_config();
      unsigned_config = xnn_init_qdu8_f32_qc4w_gemm_config();
    } else if (consumer_weights_type == xnn_datatype_qcint8) {
      original_config = xnn_init_qd8_f32_qc8w_gemm_config();
      switch (consumer_type) {
        case xnn_node_type_batch_matrix_multiply:
        case xnn_node_type_fully_connected:
          unsigned_config = xnn_init_qdu8_f32_qc8w_gemm_config();
          break;
        case xnn_node_type_convolution_2d:
        case xnn_node_type_deconvolution_2d:
          unsigned_config = xnn_init_qdu8_f32_qc8w_igemm_config();
          break;
        default:
          XNN_UNREACHABLE;
      }
    } else if (consumer_weights_type == xnn_datatype_qbint4) {
      original_config = xnn_init_qd8_f32_qb4w_gemm_config();
      unsigned_config = xnn_init_qdu8_f32_qb4w_gemm_config();
    }
  } else if (input_datatype == xnn_datatype_fp16) {
    if (consumer_weights_type == xnn_datatype_qcint4) {
      original_config = xnn_init_qd8_f16_qc4w_gemm_config();
      unsigned_config = xnn_init_qdu8_f16_qc4w_gemm_config();
    } else if (consumer_weights_type == xnn_datatype_qcint8) {
      switch (consumer_type) {
        case xnn_node_type_batch_matrix_multiply:
        case xnn_node_type_fully_connected:
          original_config = xnn_init_qd8_f16_qc8w_gemm_config();
          unsigned_config = xnn_init_qdu8_f16_qc8w_gemm_config();
          break;
        case xnn_node_type_convolution_2d:
        case xnn_node_type_deconvolution_2d:
          original_config = xnn_init_qd8_f16_qc8w_igemm_config();
          unsigned_config = xnn_init_qdu8_f16_qc8w_gemm_config();
          break;
        default:
          XNN_UNREACHABLE;
      }
    }
  }

  // If the `qduint8` config is better than the `qdint8` config, use it
  // instead.
  bool convert_to_qu8 = false;
  if (original_config && unsigned_config) {
    enum xnn_arch_flags qdu8_arch = unsigned_config->arch;
    enum xnn_arch_flags qd8_arch = original_config->arch;
    if (qdu8_arch > qd8_arch) {
      convert_to_qu8 = true;
    }
  }
  return convert_to_qu8;
}

enum xnn_status xnn_subgraph_optimize_packed_lhs(xnn_subgraph_t subgraph,
                                                 uint32_t optimization_flags) {
  // Count the number of changes made.
  size_t changes = 0;

  // Loop over the nodes in the subgraph.
  for (uint32_t node_id = 0; node_id < subgraph->num_nodes; node_id++) {
    struct xnn_node* node = &subgraph->nodes[node_id];

    // Skip anything that is not a fully-connected node.
    if (!(node->type == xnn_node_type_fully_connected ||
          node->type == xnn_node_type_batch_matrix_multiply)) {
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
      if (optimization_flags & XNN_FLAG_NO_INLINED_LHS_PACKING) {
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
        // If this is a fully-connected op, we need to coerce the shape of the
        // inputs from `[B, M, K]` to `[B * M, K]` to avoid batch-wise packing.
        if (node->type == xnn_node_type_fully_connected) {
          subgraph->values[subgraph->nodes[node_id].inputs[0]].flags |=
              XNN_FLAG_SQUASH_GROUPS;
        }
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
          // Maybe use `qduint8` instead of `qdint8`?
          xnn_log_debug(
              "Skipping %s node #%u for input #%" PRIu32 " of node #%u (%s).",
              xnn_node_type_to_string(producer->type), producer->id, input_id,
              node_id, xnn_node_type_to_string(xnn_node_type_fully_connected));
          struct xnn_value* new_input = &subgraph->values[producer->inputs[0]];
          node->inputs[0] = producer->inputs[0];
          if (new_input->first_consumer == input_value->producer) {
            new_input->first_consumer = node_id;
          }
          if (--input_value->num_consumers == 0) {
            xnn_node_clear(producer);
          }
          if (convert_gemm_to_qduint8(new_input->datatype, node->type,
                                      kernel_datatype)) {
            assumed_datatype = xnn_datatype_qduint8;
          }
          changes++;
        }
        xnn_log_debug("Setting assumed_datatype=%s for node #%u (%s).",
                      xnn_datatype_to_string(assumed_datatype), node_id,
                      xnn_node_type_to_string(node->type));
        node->params.inlined_lhs_packing.packed_input_datatype =
            assumed_datatype;
        node->flags |= XNN_FLAG_INLINE_LHS_PACKING;
      }
    }
  }

  // Second loop over the nodes to convert any `qdint8` to `qduint8` where
  // appropriate.
  for (uint32_t node_id = 0; node_id < subgraph->num_nodes; node_id++) {
    struct xnn_node* node = &subgraph->nodes[node_id];

    // Skip anything that is not a `convert` node.
    if (node->type != xnn_node_type_convert) {
      continue;
    }

    // Get a handle on the inputs/outputs.
    const uint32_t output_id = node->outputs[0];
    struct xnn_value* input_value = &subgraph->values[node->inputs[0]];
    struct xnn_value* output_value = &subgraph->values[output_id];
    const enum xnn_datatype input_datatype = input_value->datatype;
    const enum xnn_datatype output_datatype = output_value->datatype;

    // Only replace `qdint8` nodes for which all consumer are of the same type.
    if (!output_value->all_consumers_types_same ||
        output_datatype != xnn_datatype_qdint8) {
      continue;
    }

    // Get a handle on the consumer and its inputs/outputs.
    const struct xnn_node* consumer =
        &subgraph->nodes[output_value->first_consumer];
    const enum xnn_datatype consumer_weights_type =
        subgraph->values[consumer->inputs[1]].datatype;

    // If the `qduint8` config is better than the `qdint8` config, use it
    // instead.
    if (convert_gemm_to_qduint8(input_datatype, consumer->type,
                                consumer_weights_type)) {
      xnn_log_debug("Coercing type of output ID #%" PRIu32
                    " of %s operator from `%s` to `%s`.",
                    output_id, xnn_node_type_to_string(xnn_node_type_convert),
                    xnn_datatype_to_string(output_datatype),
                    xnn_datatype_to_string(xnn_datatype_qduint8));
      output_value->datatype = xnn_datatype_qduint8;
    }
  }

  // Clean up after ourselves.
  if (changes) {
    xnn_subgraph_clean_up(subgraph);
  }

  return xnn_status_success;
}

static void replace_in_set(uint32_t* set, uint32_t size, uint32_t old_value,
                           uint32_t new_value) {
  for (uint32_t i = 0; i < size; i++) {
    if (set[i] == old_value) {
      set[i] = new_value;
    }
  }
}

// Persistent values are values that can be read or written repeatedly. This
// isn't compatible with our graph model. To work around this, we implement
// persistent values with an [SSA] approach:
//  - Persistent values are passed in as inputs
//  - Writes to the persistent value creates a new value
//  - Reads read the last value written to (or the input if none).
//  - The last written value (if any) is passed out as an output.
//
// [SSA]. https://en.wikipedia.org/wiki/Static_single-assignment_form
void xnn_subgraph_rewrite_ssa(xnn_subgraph_t subgraph) {
  bool* values_written =
      (bool*)xnn_allocate_memory(sizeof(bool) * subgraph->num_values);
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    values_written[i] = false;
  }
  for (uint32_t i = 0; i < subgraph->num_nodes; i++) {
    struct xnn_node* node = &subgraph->nodes[i];
    for (uint32_t j = 0; j < node->num_outputs; j++) {
      const uint32_t output_id = node->outputs[j];
      const struct xnn_value* value = &subgraph->values[output_id];
      if (!xnn_value_is_external_output(value->flags)) {
        // We only care to rewrite external outputs. Internal values should
        // already be SSA.
        continue;
      }
      if (!values_written[output_id]) {
        // This is the first time we've seen this output.
      } else {
        // We already wrote this value. Make a new value to replace the previous
        // value with (so the external output remains the last write).
        struct xnn_value* new_value = xnn_subgraph_new_internal_value(subgraph);
        // xnn_subgraph_new_internal_value may have invalidated `value` pointer.
        value = &subgraph->values[output_id];
        xnn_value_copy(new_value, value);
        xnn_log_debug("Adding new value #%" PRIu32
                      " for already produced output #%" PRIu32,
                      new_value->id, output_id);

        // For outputs, we want only the last value to be the original output.
        // The new value is not an output, and its an internal allocation.
        new_value->flags &=
            ~(XNN_VALUE_FLAG_EXTERNAL_OUTPUT | XNN_VALUE_FLAG_EXTERNAL_INPUT);
        new_value->allocation_type = xnn_allocation_type_workspace;

        // Since we want to rewrite the previous write's value, we need to go
        // back and update previously visited nodes. We only want to do the
        // mapping after we find the first time this value is written.
        bool found = false;
        for (uint32_t k = 0; k < i; ++k) {
          struct xnn_node* node_k = &subgraph->nodes[k];
          if (found) {
            // We've written the new value, update all subsequent reads to point
            // to the new value.
            replace_in_set(node_k->inputs, node_k->num_inputs, output_id,
                           new_value->id);
            replace_in_set(node_k->outputs, node_k->num_outputs, output_id,
                           new_value->id);
          } else if (set_contains(node_k->outputs, node_k->num_outputs,
                                  output_id)) {
            // We found where this value is written. Replace only the output
            // use.
            replace_in_set(node_k->outputs, node_k->num_outputs, output_id,
                           new_value->id);
            found = true;
          } else {
            // We're still using the old value.
          }
        }
        // Replace all subsequent reads with the new value, until we find
        // another write (because subsequent reads from there need the
        // replacement for that value).
        for (uint32_t k = i; k < subgraph->num_nodes; ++k) {
          struct xnn_node* node_k = &subgraph->nodes[k];
          replace_in_set(node_k->inputs, node_k->num_inputs, output_id,
                         new_value->id);
          if (set_contains(node_k->outputs, node_k->num_outputs, output_id)) {
            break;
          }
        }
      }
      values_written[output_id] = true;
    }
  }
  xnn_release_memory(values_written);
}

enum xnn_status xnn_subgraph_optimize(xnn_subgraph_t subgraph,
                                      uint32_t optimization_flags) {
  // If the subgraph has no nodes, then there is nothing for us to do here, but
  // do print a notice to the user as this seems a bit unusual.
  if (!subgraph->num_nodes) {
    xnn_log_info("Trying to optimize subgraph with zero nodes, skipping.");
    return xnn_status_success;
  }

  // Start with a clean and ordered subgraph.
  xnn_subgraph_clean_up(subgraph);

  if (!(optimization_flags & XNN_FLAG_NO_OPERATOR_FUSION)) {
    xnn_subgraph_fusion(subgraph);
    xnn_subgraph_fuse_unary_quantized_into_lut(subgraph);
  }

  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    xnn_log_error("failed to get hardware config");
    return xnn_status_unsupported_hardware;
  }

  if ((optimization_flags & XNN_FLAG_FORCE_FP16_INFERENCE) &&
      (!xnn_is_f16_compatible_config(hardware_config))) {
    xnn_log_error(
        "failed to force FP16 inference: hardware supports neither native nor "
        "emulated FP16 operators");
    return xnn_status_unsupported_hardware;
  }
  const bool try_native_fp16 =
      (optimization_flags & XNN_FLAG_HINT_FP16_INFERENCE) &&
      xnn_is_f16_supported_natively(hardware_config);
  const bool force_fp16 = (optimization_flags & XNN_FLAG_FORCE_FP16_INFERENCE);
  if (try_native_fp16 || force_fp16) {
    const bool fp16_rewrite_succeeded = xnn_subgraph_rewrite_for_fp16(subgraph);
    if (force_fp16 && !fp16_rewrite_succeeded) {
      xnn_log_error(
          "failed to force FP16 inference: subgraph is incompatible with FP16 "
          "operators");
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

#ifdef XNN_SLINKY_ENABLED
  // If compiling with XNN_SLINKY_ENABLED defined, assume we always
  // want Slinky enabled, regardless of the runtime flag
  const bool use_slinky = true;
#else
  const bool use_slinky = optimization_flags & XNN_FLAG_SLINKY_ENABLED;
#endif
  // If we're using Slinky, don't inline packing functions as Slinky does that
  // automatically.
  if (use_slinky) {
    xnn_log_info(
        "disabling inlined LHS packing because `XNN_FLAG_SLINKY_ENABLED` is "
        "set.");
    optimization_flags |= XNN_FLAG_NO_INLINED_LHS_PACKING;
  }

  enum xnn_status status =
      xnn_subgraph_optimize_packed_lhs(subgraph, optimization_flags);
  if (status != xnn_status_success) {
    return status;
  }

  return xnn_status_success;
}

enum xnn_status xnn_delete_subgraph(xnn_subgraph_t subgraph) {
  if (subgraph != NULL) {
    if (subgraph->nodes != NULL) {
      memset(subgraph->nodes, 0, sizeof(struct xnn_node) * subgraph->num_nodes);
      xnn_release_memory(subgraph->nodes);
    }

    if (subgraph->values != NULL) {
      // Release the dynamic allocations created during FP16 rewrite, if the
      // subgraph still has ownership of them.
      for (uint32_t i = 0; i < subgraph->num_values; i++) {
        struct xnn_value* value = &subgraph->values[i];
        if (value->fp16_compatible && value->data != NULL) {
          XNN_PRAGMA_CLANG("clang diagnostic push")
          XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wcast-qual\"")
          xnn_release_memory((void*)value->data);
          XNN_PRAGMA_CLANG("clang diagnostic pop")
        }
      }

      memset(subgraph->values, 0,
             sizeof(struct xnn_value) * subgraph->num_values);
      xnn_release_memory(subgraph->values);
    }

    memset(subgraph, 0, sizeof(struct xnn_subgraph));
    xnn_release_memory(subgraph);
  }
  return xnn_status_success;
}

enum xnn_node_type xnn_reduce_operator_to_node_type(
    enum xnn_reduce_operator type) {
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

enum xnn_reduce_operator xnn_node_type_to_reduce_operator(
    enum xnn_node_type type) {
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
