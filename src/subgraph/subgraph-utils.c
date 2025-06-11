// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/subgraph/subgraph-utils.h"

#include <stdio.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph.h"

void xnn_print_flags(const int flags, const int count, const int values[],
                     const char* names) {
  const char* next_name = names;
  const char* separator = "";
  for (int i = 0; i < count - 1; ++i) {
    next_name = strchr(names, ',');
    const int len = next_name - names;
    if (flags & values[i]) {
      fprintf(stderr, "%s%.*s", separator, len, names);
      separator = "|";
    }
    names = next_name + 2;  // comma + space.
  }
  if (flags & values[count - 1]) {
    fprintf(stderr, "%s%s", separator, names);
  }
}

#define XNN_PRINT_FLAGS(flags, ...)                                \
  do {                                                             \
    const int values[] = {__VA_ARGS__};                            \
    xnn_print_flags((flags), sizeof(values) / sizeof(int), values, \
                    #__VA_ARGS__);                                 \
  } while (false)

void xnn_subgraph_log_impl(const char* filename, size_t line_number,
                           xnn_subgraph_t subgraph, FILE* out) {
  // Header.
  fprintf(out, "%s:%zu: Subgraph %p with %u nodes and %u values:\n", filename,
          line_number, subgraph, subgraph->num_nodes, subgraph->num_values);

  // Nodes.
  fprintf(out, "  Nodes:\n");
  for (int node_id = 0; node_id < subgraph->num_nodes; node_id++) {
    const struct xnn_node* node = &subgraph->nodes[node_id];
    fprintf(out, "    %03i: type=%s", node_id,
            xnn_node_type_to_string(node->type));
    switch (node->type) {
      case xnn_node_type_unary_elementwise:
        fprintf(
            out, " (%s, %s)",
            xnn_unary_operator_to_string(node->unary_operator),
            xnn_datatype_to_string(subgraph->values[node->inputs[0]].datatype));
        break;
      case xnn_node_type_binary_elementwise:
        fprintf(
            out, " (%s, %s)",
            xnn_binary_operator_to_string(node->binary_operator),
            xnn_datatype_to_string(subgraph->values[node->inputs[0]].datatype));
        break;
      case xnn_node_type_convert:
        fprintf(
            out, " (%s -> %s)",
            xnn_datatype_to_string(subgraph->values[node->inputs[0]].datatype),
            xnn_datatype_to_string(
                subgraph->values[node->outputs[0]].datatype));
        break;
      case xnn_node_type_fully_connected:
      case xnn_node_type_batch_matrix_multiply:
        fprintf(
            out, " (%s, %s, %s)",
            xnn_datatype_to_string(
                node->params.inlined_lhs_packing.packed_input_datatype !=
                        xnn_datatype_invalid
                    ? node->params.inlined_lhs_packing.packed_input_datatype
                    : subgraph->values[node->inputs[0]].datatype),
            xnn_datatype_to_string(subgraph->values[node->outputs[0]].datatype),
            xnn_datatype_to_string(subgraph->values[node->inputs[1]].datatype));
        break;
      default:
        break;
    }
    if (node->num_inputs) {
      fprintf(out, ", inputs=[%i", node->inputs[0]);
      for (int i = 1; i < node->num_inputs; i++) {
        fprintf(out, ", %i", node->inputs[i]);
      }
      fprintf(out, "]");
    }
    if (node->num_outputs) {
      fprintf(out, ", outputs=[");
      fprintf(out, "%i", node->outputs[0]);
      for (int i = 1; i < node->num_outputs; i++) {
        fprintf(out, ", %i", node->outputs[i]);
      }
      fprintf(out, "]");
    }
    if (node->flags) {
      fprintf(out, ", flags=");
      XNN_PRINT_FLAGS(
          node->flags, XNN_FLAG_INLINE_LHS_PACKING,
          XNN_FLAG_DEPTHWISE_CONVOLUTION, XNN_FLAG_TRANSPOSE_WEIGHTS,
          XNN_FLAG_TENSORFLOW_SAME_PADDING, XNN_FLAG_INPUT_NHWC,
          XNN_FLAG_TRANSPOSE_A, XNN_FLAG_TENSORFLOW_LEGACY_MODE,
          XNN_FLAG_FP32_STATIC_WEIGHTS, XNN_FLAG_FP32_STATIC_BIASES,
          XNN_FLAG_ALIGN_CORNERS, XNN_FLAG_YIELD_WORKERS,
          XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER, XNN_FLAG_KEEP_DIMS,
          XNN_FLAG_NO_BROADCAST, XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC);
    }
    fprintf(out, ".\n");
  }

  // Values.
  fprintf(out, "  Values:\n");
  for (int value_id = 0; value_id < subgraph->num_values; value_id++) {
    const struct xnn_value* value = &subgraph->values[value_id];
    fprintf(out, "    %03i: dtype=%s, shape=[", value_id,
            xnn_datatype_to_string(value->datatype));
    if (value->shape.num_dims) {
      fprintf(out, "%zu", value->shape.dim[0]);
      for (int i = 1; i < value->shape.num_dims; i++) {
        fprintf(out, ", %zu", value->shape.dim[i]);
      }
    }
    fprintf(out, "]");
    if (xnn_value_is_external(value->flags)) {
      fprintf(out, ", external=%s%s",
              xnn_value_is_external_input(value->flags) ? "input" : "",
              xnn_value_is_external_output(value->flags) ? "output" : "");
    }
    if (value->producer != XNN_INVALID_NODE_ID) {
      fprintf(out, ", producer=%u", value->producer);
    }
    if (value->num_consumers) {
      fprintf(out, ", num_consumers=%u", value->num_consumers);
    }
    if (value->first_consumer != XNN_INVALID_NODE_ID) {
      fprintf(out, ", first_consumer=%u", value->first_consumer);
    }
    if (value->fp16_compatible && value->fp16_id != XNN_INVALID_VALUE_ID) {
      fprintf(out, ", fp16_id=%u", value->fp16_id);
    }
    if (value->fp16_compatible && value->fp32_id != XNN_INVALID_VALUE_ID) {
      fprintf(out, ", fp32_id=%u", value->fp32_id);
    }
    if (value->flags &
        ~(XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) {
      fprintf(out, ", flags=");
      XNN_PRINT_FLAGS(
          value->flags, XNN_FLAG_SQUASH_GROUPS, XNN_VALUE_FLAG_ONE_CONSUMER,
          XNN_VALUE_FLAG_FP16_COMPATIBLE, XNN_VALUE_FLAG_LAYOUT_NCHW);
    }
    fprintf(out, ".\n");
  }
}
