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

  if (!xnn_params.initialized) {
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
          case xnn_node_type_depthwise_convolution_2d:
          case xnn_node_type_fully_connected:
          case xnn_node_type_multiply2:
          case xnn_node_type_max_pooling_2d:
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
      if (producer->type == xnn_node_type_constant_pad) {
        assert(producer->num_inputs == 1);
        assert(producer->num_outputs == 1);
        const bool is_spatial_2d_padding = value->shape.num_dims == 4 &&
          (producer->params.static_pad.pre_paddings[0] | producer->params.static_pad.post_paddings[0] |
           producer->params.static_pad.pre_paddings[3] | producer->params.static_pad.post_paddings[3]) == 0;
        switch (consumer->type) {
          case xnn_node_type_convolution_2d:
            if (is_spatial_2d_padding && !(consumer->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING)) {
              xnn_log_info("fuse Constant Pad Node #%"PRIu32" into Convolution 2D Node #%"PRIu32,
                consumer_id, producer_id);
              assert(consumer->num_inputs >= 1);
              assert(consumer->inputs[0] == producer->outputs[0]);

              consumer->params.convolution_2d.input_padding_top    += producer->params.static_pad.pre_paddings[1];
              consumer->params.convolution_2d.input_padding_right  += producer->params.static_pad.pre_paddings[2];
              consumer->params.convolution_2d.input_padding_bottom += producer->params.static_pad.post_paddings[1];
              consumer->params.convolution_2d.input_padding_left   += producer->params.static_pad.post_paddings[2];

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
            if (is_spatial_2d_padding && !(consumer->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING)) {
              xnn_log_info("fuse Constant Pad Node #%"PRIu32" into Depthwise Convolution 2D Node #%"PRIu32,
                consumer_id, producer_id);
              assert(consumer->num_inputs >= 1);
              assert(consumer->inputs[0] == producer->outputs[0]);

              consumer->params.depthwise_convolution_2d.input_padding_top +=
                producer->params.static_pad.pre_paddings[1];
              consumer->params.depthwise_convolution_2d.input_padding_right +=
                producer->params.static_pad.pre_paddings[2];
              consumer->params.depthwise_convolution_2d.input_padding_bottom +=
                producer->params.static_pad.post_paddings[1];
              consumer->params.depthwise_convolution_2d.input_padding_left +=
                producer->params.static_pad.post_paddings[2];

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
