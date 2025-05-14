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

void xnn_subgraph_dump(xnn_subgraph_t subgraph, FILE* out) {
  fprintf(out, "%s:%i: Subgraph %p with %u nodes and %u values:\n",
          __FUNCTION__, __LINE__, subgraph, subgraph->num_nodes,
          subgraph->num_values);
  fprintf(out, "  Nodes:\n");
  for (int node_id = 0; node_id < subgraph->num_nodes; node_id++) {
    const struct xnn_node* node = &subgraph->nodes[node_id];
    fprintf(out, "    %03i: type=%s, inputs=[", node_id,
            xnn_node_type_to_string(node->type));
    if (node->num_inputs) {
      fprintf(out, "%i", node->inputs[0]);
      for (int i = 1; i < node->num_inputs; i++) {
        fprintf(out, ", %i", node->inputs[i]);
      }
    }
    fprintf(out, "], outputs=[");
    if (node->num_outputs) {
      fprintf(out, "%i", node->outputs[0]);
      for (int i = 1; i < node->num_outputs; i++) {
        fprintf(out, ", %i", node->outputs[i]);
      }
    }
    fprintf(out, "].\n");
  }
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
    fprintf(out, "], producer=%u, first_consumer=%u.\n", value->producer,
            value->first_consumer);
  }
}
