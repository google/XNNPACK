// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


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
    xnn_log_error("failed to allocate %zu bytes for subgraph values", external_value_ids * sizeof(struct xnn_value));
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
  assert(node->type != xnn_node_type_invalid);
  memset(node, 0, sizeof(struct xnn_node));
}

void xnn_value_clear(struct xnn_value* value) {
  assert(value != NULL);
  assert(value->type != xnn_value_type_invalid);
  memset(value, 0, sizeof(struct xnn_value));
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

#define XNN_LAYOUT_FLAG_COMPATIBLE_NCHW      1
#define XNN_LAYOUT_FLAG_COMPATIBLE_NHWC2NCHW 2
#define XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC 4
#define XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER 8

uint32_t xnn_check_nchw_compatibility(xnn_subgraph_t subgraph, struct xnn_node* node) {
  switch (node->type) {
    case xnn_node_type_convolution_2d:
      // Supported cases:
      // - 1x1 convolution (no stride, no dilation, no padding, no groups)
      // - 3x3 stride-2 convolution (no dilation, padding 1 on each side, no groups, 3 input channels)
      if (node->params.convolution_2d.groups != 1) {
        return 0;
      }
      if ((node->params.convolution_2d.dilation_height | node->params.convolution_2d.dilation_width) != 1) {
        return 0;
      }
      if ((node->params.convolution_2d.kernel_height | node->params.convolution_2d.kernel_width) == 1) {
        if ((node->params.convolution_2d.input_padding_top | node->params.convolution_2d.input_padding_right |
             node->params.convolution_2d.input_padding_bottom | node->params.convolution_2d.input_padding_left) != 0)
        {
          return 0;
        }
        if ((node->params.convolution_2d.subsampling_height | node->params.convolution_2d.subsampling_width) != 1) {
          return 0;
        }
        return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW;
      } else if (node->params.convolution_2d.kernel_height == 3 && node->params.convolution_2d.kernel_width == 3) {
        if (node->params.convolution_2d.input_padding_top != 1 || node->params.convolution_2d.input_padding_right != 1 ||
            node->params.convolution_2d.input_padding_bottom != 1 || node->params.convolution_2d.input_padding_left != 1)
        {
          return 0;
        }
        if ((node->params.convolution_2d.subsampling_height | node->params.convolution_2d.subsampling_width) != 2) {
          return 0;
        }
        if (node->params.convolution_2d.group_input_channels != 3) {
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
        return 0;
      }
      if (node->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
        return 0;
      }
      if (node->params.depthwise_convolution_2d.depth_multiplier != 1) {
        return 0;
      }
      if (node->params.depthwise_convolution_2d.subsampling_height != node->params.depthwise_convolution_2d.subsampling_width) {
        return 0;
      }
      switch (node->params.depthwise_convolution_2d.subsampling_height) {
        case 1:
        case 2:
          break;
        default:
          return 0;
      }
      if (node->params.depthwise_convolution_2d.kernel_height != node->params.depthwise_convolution_2d.kernel_width) {
        return 0;
      }
      switch (node->params.depthwise_convolution_2d.kernel_height) {
        case 3:
          return node->params.depthwise_convolution_2d.input_padding_top == 1 &&
                 node->params.depthwise_convolution_2d.input_padding_right == 1 &&
                 node->params.depthwise_convolution_2d.input_padding_bottom == 1 &&
                 node->params.depthwise_convolution_2d.input_padding_left == 1 ? XNN_LAYOUT_FLAG_COMPATIBLE_NCHW : 0;
        case 5:
          return node->params.depthwise_convolution_2d.input_padding_top == 2 &&
                 node->params.depthwise_convolution_2d.input_padding_right == 2 &&
                 node->params.depthwise_convolution_2d.input_padding_bottom == 2 &&
                 node->params.depthwise_convolution_2d.input_padding_left == 2 ? XNN_LAYOUT_FLAG_COMPATIBLE_NCHW : 0;
        default:
          return 0;
      }
    case xnn_node_type_depth_to_space:
      return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
    case xnn_node_type_global_average_pooling_2d:
      return XNN_LAYOUT_FLAG_COMPATIBLE_NCHW | XNN_LAYOUT_FLAG_COMPATIBLE_NCHW2NHWC;
    case xnn_node_type_add2:
    case xnn_node_type_multiply2:
      assert(node->num_inputs == 2);
      assert(node->num_outputs == 1);
      if (subgraph->values[node->inputs[0]].shape.num_dims != 4 ||
          subgraph->values[node->inputs[1]].shape.num_dims != 4)
      {
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
      return subgraph->values[node->inputs[0]].shape.dim[1] > 1 &&
             subgraph->values[node->inputs[0]].shape.dim[2] > 1 ? XNN_LAYOUT_FLAG_COMPATIBLE_NCHW : 0;
    case xnn_node_type_abs:
    case xnn_node_type_bankers_rounding:
    case xnn_node_type_ceiling:
    case xnn_node_type_clamp:
    case xnn_node_type_elu:
    case xnn_node_type_floor:
    case xnn_node_type_hardswish:
    case xnn_node_type_leaky_relu:
    case xnn_node_type_negate:
    case xnn_node_type_sigmoid:
    case xnn_node_type_square:
      assert(node->num_inputs == 1);
      assert(node->num_outputs == 1);
      return subgraph->values[node->inputs[0]].shape.num_dims == 4 ? XNN_LAYOUT_FLAG_COMPATIBLE_NCHW : 0;
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
        if ((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0) {
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
  // Propagate the cluster leader to other nodes in the graph untill all the
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
        if ((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0) {
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
      assert((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0);
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
      assert((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0);
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

    if (node->type == xnn_node_type_convolution_2d &&
        max(node->params.convolution_2d.kernel_height, node->params.convolution_2d.kernel_width) == 1)
    {
      assert(node->num_inputs >= 2);

      const struct xnn_value* filter = &subgraph->values[node->inputs[1]];
      assert(filter->data != NULL);
      assert(filter->shape.num_dims == 4);

      const size_t num_params = filter->shape.dim[0] * filter->shape.dim[3];
      subgraph->nodes[node->cluster_leader].num_params += num_params;

      const float* data = (const float*) filter->data;
      size_t num_zeroes = 0;
      for (size_t i = 0; i < num_params; i++) {
        num_zeroes += (size_t) (data[i] == 0.0f);
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
      assert((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0);
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

enum xnn_status xnn_subgraph_optimize(
  xnn_subgraph_t subgraph,
  uint32_t flags)
{
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

    for (uint32_t i = 0; i < node->num_inputs; i++) {
      const uint32_t input_id = node->inputs[i];
      assert(input_id < subgraph->num_values);

      if (subgraph->values[input_id].num_consumers++ == 0) {
        assert(subgraph->values[input_id].first_consumer == XNN_INVALID_NODE_ID);
        subgraph->values[input_id].first_consumer = n;
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
    if (value->type == xnn_value_type_invalid) {
      continue;
    }

    if (value->flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) {
      value->num_consumers += 1;
    }
    if ((value->flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0 && value->num_consumers == 0) {
      xnn_value_clear(value);
    }
  }

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
      assert(consumer->type != xnn_node_type_invalid);

      // Try to fuse Clamp Node upstream into producer Node
      if (consumer->type == xnn_node_type_clamp) {
        switch (producer->type) {
          case xnn_node_type_add2:
          case xnn_node_type_average_pooling_2d:
          case xnn_node_type_clamp:
          case xnn_node_type_convolution_2d:
          case xnn_node_type_divide:
          case xnn_node_type_deconvolution_2d:
          case xnn_node_type_depthwise_convolution_2d:
          case xnn_node_type_fully_connected:
          case xnn_node_type_multiply2:
          case xnn_node_type_max_pooling_2d:
          case xnn_node_type_subtract:
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

            xnn_node_clear(consumer);
            xnn_value_clear(value);
            break;
          default:
            break;
        }
      }
      // Try to fuse Constant Pad node downstream into [Depthwise] Convolution 2D Node
      if (producer->type == xnn_node_type_static_constant_pad) {
        assert(producer->num_inputs == 1);
        assert(producer->num_outputs == 1);
        const bool is_spatial_2d_zero_padding = value->shape.num_dims == 4 &&
          (producer->params.static_pad.pre_paddings[0] | producer->params.static_pad.post_paddings[0] |
           producer->params.static_pad.pre_paddings[3] | producer->params.static_pad.post_paddings[3]) == 0 &&
           producer->params.static_pad.padding_value == 0;
        switch (consumer->type) {
          case xnn_node_type_convolution_2d:
            if (is_spatial_2d_zero_padding && !(consumer->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING)) {
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
            if (is_spatial_2d_zero_padding && !(consumer->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING)) {
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
    }
  }

  #if XNN_ENABLE_SPARSE
    if ((flags & XNN_FLAG_SPARSE_INFERENCE) && (xnn_params.init_flags & XNN_INIT_FLAG_CHW_OPT)) {
      xnn_subgraph_rewrite_for_nchw(subgraph);
    }
  #endif

  return xnn_status_success;
}

enum xnn_status xnn_delete_subgraph(
  xnn_subgraph_t subgraph)
{
  if (subgraph != NULL) {
    memset(subgraph->nodes, 0, sizeof(struct xnn_node) * subgraph->num_nodes);
    xnn_release_memory(subgraph->nodes);

    memset(subgraph->values, 0, sizeof(struct xnn_value) * subgraph->num_values);
    xnn_release_memory(subgraph->values);

    memset(subgraph, 0, sizeof(struct xnn_subgraph));
    xnn_release_memory(subgraph);
  }
  return xnn_status_success;
}
