// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/subgraph/subgraph-utils.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/mutex.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph.h"

// Use a mutex to avoid the concurrent logging of subgraphs, which messes things
// up badly.
static struct xnn_mutex mutex;

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

static void print_node_type(FILE* out, const xnn_subgraph_t subgraph,
                            const struct xnn_node* node,
                            const char* separator) {
  fprintf(out, "%s", xnn_node_type_to_string(node->type));
  switch (node->type) {
    case xnn_node_type_unary_elementwise:
      if (node->unary_operator == xnn_unary_clamp) {
        fprintf(
            out, "%s(%s [%f, %f], %s)", separator,
            xnn_unary_operator_to_string(node->unary_operator),
            node->params.unary.clamp.min, node->params.unary.clamp.max,
            xnn_datatype_to_string(subgraph->values[node->inputs[0]].datatype));
      } else {
        fprintf(
            out, "%s(%s, %s)", separator,
            xnn_unary_operator_to_string(node->unary_operator),
            xnn_datatype_to_string(subgraph->values[node->inputs[0]].datatype));
      }
      break;
    case xnn_node_type_binary_elementwise:
      fprintf(
          out, "%s(%s, %s)", separator,
          xnn_binary_operator_to_string(node->binary_operator),
          xnn_datatype_to_string(subgraph->values[node->inputs[0]].datatype));
      break;
    case xnn_node_type_convert:
      fprintf(
          out, "%s(%s -> %s)", separator,
          xnn_datatype_to_string(subgraph->values[node->inputs[0]].datatype),
          xnn_datatype_to_string(subgraph->values[node->outputs[0]].datatype));
      break;
    case xnn_node_type_fully_connected:
    case xnn_node_type_batch_matrix_multiply:
      fprintf(
          out, "%s(%s, %s, %s%s)", separator,
          xnn_datatype_to_string(
              node->packed_input_datatype != xnn_datatype_invalid
                  ? node->packed_input_datatype
                  : subgraph->values[node->inputs[0]].datatype),
          xnn_datatype_to_string(subgraph->values[node->outputs[0]].datatype),
          xnn_datatype_to_string(subgraph->values[node->inputs[1]].datatype),
          node->flags & XNN_FLAG_TRANSPOSE_WEIGHTS ? ", transposed" : "");
      break;
    case xnn_node_type_static_transpose:
      fprintf(out, "%s(perm=[%zu", separator, node->params.transpose.perm[0]);
      for (int i = 1; i < node->params.transpose.num_dims; i++) {
        fprintf(out, ", %zu", node->params.transpose.perm[i]);
      }
      fprintf(out, "])");
      break;
    case xnn_node_type_static_expand_dims:
    case xnn_node_type_static_reshape:
      fprintf(out, "%s(%s=[", separator,
              node->type == xnn_node_type_static_reshape ? "shape" : "axes");
      if (node->params.static_reshape.new_shape.num_dims) {
        fprintf(out, "%zu", node->params.static_reshape.new_shape.dim[0]);
        for (int i = 1; i < node->params.static_reshape.new_shape.num_dims;
             i++) {
          fprintf(out, ", %zu", node->params.static_reshape.new_shape.dim[i]);
        }
      }
      fprintf(out, "])");
      break;
    case xnn_node_type_static_sum:
    case xnn_node_type_static_sum_squared:
    case xnn_node_type_static_mean:
    case xnn_node_type_static_mean_squared:
      fprintf(out, "%s(axes=[", separator);
      if (node->params.reduce.num_reduction_axes) {
        fprintf(out, "%" PRIi64, node->params.reduce.reduction_axes[0]);
        for (int i = 1; i < node->params.reduce.num_reduction_axes; i++) {
          fprintf(out, ", %" PRIi64, node->params.reduce.reduction_axes[i]);
        }
      }
      fprintf(out, "])");
      break;
    default:
      break;
  }
}

void xnn_subgraph_log_impl(const char* filename, size_t line_number,
                           xnn_subgraph_t subgraph, FILE* out) {
  xnn_mutex_lock(&mutex);

  // Header.
  fprintf(out, "%s:%zu: Subgraph %p with %u nodes and %u values:\n", filename,
          line_number, subgraph, subgraph->num_nodes, subgraph->num_values);

  // Nodes.
  fprintf(out, "  Nodes:\n");
  for (int node_id = 0; node_id < subgraph->num_nodes; node_id++) {
    const struct xnn_node* node = &subgraph->nodes[node_id];
    if (node->type == xnn_node_type_invalid) {
      continue;
    }
    fprintf(out, "    %03i: ", node_id);
    print_node_type(out, subgraph, node, " ");
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
          XNN_FLAG_ALIGN_CORNERS, XNN_FLAG_DONT_SPIN_WORKERS,
          XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER, XNN_FLAG_KEEP_DIMS,
          XNN_FLAG_NO_BROADCAST, XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC,
          XNN_NODE_FLAG_DONT_ELIDE);
    }
    fprintf(out, ".\n");
  }

  // Values.
  fprintf(out, "  Values:\n");
  for (int value_id = 0; value_id < subgraph->num_values; value_id++) {
    const struct xnn_value* value = &subgraph->values[value_id];
    if (value->datatype == xnn_datatype_invalid) {
      continue;
    }
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
    if (xnn_value_is_static(value->allocation_type)) {
      fprintf(out, ", static");
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
          XNN_VALUE_FLAG_FP16_COMPATIBLE, XNN_VALUE_FLAG_LAYOUT_NCHW,
          XNN_VALUE_FLAG_SHAPE_IS_STATIC, XNN_VALUE_FLAG_IS_ZERO,
          XNN_VALUE_FLAG_IS_ONE);
    }
    fprintf(out, ".\n");
  }

  xnn_mutex_unlock(&mutex);
}

void xnn_subgraph_log_dot_impl(xnn_subgraph_t subgraph, FILE* out) {
  xnn_mutex_lock(&mutex);

  // Header.
  fprintf(out,
          "digraph {\n"
          "  labelloc=\"t\"\n"
          "  label=\"Subgraph %p with %u nodes and %u values\"\n",
          subgraph, subgraph->num_nodes, subgraph->num_values);

  // Nodes.
  for (int node_id = 0; node_id < subgraph->num_nodes; node_id++) {
    const struct xnn_node* node = &subgraph->nodes[node_id];
    if (node->type == xnn_node_type_invalid) {
      continue;
    }
    fprintf(out, "  n%03u [shape=box, label=\"#%03u: ", node->id, node->id);
    print_node_type(out, subgraph, node, "\\n");
    fprintf(out, "\"]\n");
  }

  // Values.
  for (int value_id = 0; value_id < subgraph->num_values; value_id++) {
    const struct xnn_value* value = &subgraph->values[value_id];
    if (value->datatype == xnn_datatype_invalid) {
      continue;
    }
    if (value->datatype != xnn_datatype_invalid &&
        (xnn_value_is_external(value->flags) ||
         xnn_value_is_static(value->allocation_type))) {
      fprintf(out, "  v%03u [%slabel=\"%03u: %s", value->id,
              xnn_value_is_static(value->allocation_type)
                  ? "shape=\"hexagon\", "
                  : "",
              value->id, xnn_datatype_to_string(value->datatype));
      if (xnn_shape_multiply_all_dims(&value->shape) > 1) {
        fprintf(out, "[%zu", value->shape.dim[0]);
        for (int i = 1; i < value->shape.num_dims; i++) {
          fprintf(out, ", %zu", value->shape.dim[i]);
        }
        fprintf(out, "]");
      } else {
        fprintf(out, ": %s", value->shape.num_dims ? "[" : "");
        if (value->data == NULL) {
          fprintf(out, "???");
        } else {
          switch (value->datatype) {
            case xnn_datatype_fp32:
              fprintf(out, "%f", *(const float*)value->data);
              break;
            case xnn_datatype_fp16:
              fprintf(out, "%f",
                      xnn_float16_to_float(*(const xnn_float16*)value->data));
              break;
            case xnn_datatype_int32:
              fprintf(out, "%i", *(const int*)value->data);
              break;
            default:
              fprintf(out, "???");
              break;
          }
        }
        fprintf(out, "%s", value->shape.num_dims ? "]" : "");
      }
      fprintf(out, "%s%s\"]\n",
              value->flags & XNN_VALUE_FLAG_IS_ZERO  ? ",\\nconst 0.0"
              : value->flags & XNN_VALUE_FLAG_IS_ONE ? ",\\nconst 1.0"
                                                     : "",
              !xnn_value_is_static(value->allocation_type) &&
                      value->flags & XNN_VALUE_FLAG_SHAPE_IS_STATIC
                  ? xnn_value_is_const(value->flags) ? ", static shape"
                                                     : ",\\nstatic shape"
                  : "");
      if (value->producer != XNN_INVALID_NODE_ID &&
          xnn_value_is_external(value->flags)) {
        fprintf(out, "  n%03u -> v%03u\n", value->producer, value->id);
      }
    }
  }

  // Edges.
  for (int node_id = 0; node_id < subgraph->num_nodes; node_id++) {
    const struct xnn_node* node = &subgraph->nodes[node_id];
    if (node->type == xnn_node_type_invalid) {
      continue;
    }
    for (int k = 0; k < node->num_inputs; k++) {
      const struct xnn_value* value = &subgraph->values[node->inputs[k]];
      if (value->producer != XNN_INVALID_NODE_ID) {
        fprintf(out, "  n%03u -> n%03u [label=\"v%03u: %s[", value->producer,
                node->id, value->id, xnn_datatype_to_string(value->datatype));
        if (value->shape.num_dims) {
          fprintf(out, "%zu", value->shape.dim[0]);
          for (int i = 1; i < value->shape.num_dims; i++) {
            fprintf(out, ", %zu", value->shape.dim[i]);
          }
        }
        fprintf(out, "]%s%s\"]\n",
                value->flags & XNN_VALUE_FLAG_IS_ZERO  ? ",\\nconst 0.0"
                : value->flags & XNN_VALUE_FLAG_IS_ONE ? ",\\nconst 1.0"
                                                       : "",
                value->flags & XNN_VALUE_FLAG_SHAPE_IS_STATIC
                    ? xnn_value_is_const(value->flags) ? ", static shape"
                                                       : ",\\nstatic shape"
                    : "");
      } else {
        fprintf(out, "  v%03u -> n%03u\n", value->id, node->id);
      }
    }
  }
  fprintf(out, "}\n\n");

  xnn_mutex_unlock(&mutex);
}
