// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/subgraph/subgraph-utils.h"

#include <stdio.h>

#include "include/xnnpack.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/subgraph.h"

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
    if (xnn_value_is_external(value)) {
      fprintf(out, ", external=%s%s",
              xnn_value_is_external_input(value) ? "input" : "",
              xnn_value_is_external_output(value) ? "output" : "");
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
    fprintf(out, ".\n");
  }
}
