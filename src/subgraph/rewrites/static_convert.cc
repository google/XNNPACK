// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "src/subgraph/rewrites/static_convert.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cinttypes>

#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/subgraph.h"

namespace xnnpack {

enum xnn_status xnn_subgraph_constant_fold_converts(xnn_subgraph_t subgraph) {
  const uint32_t num_nodes = subgraph->num_nodes;
  for (uint32_t i = 0; i < num_nodes; ++i) {
    struct xnn_node* node = &subgraph->nodes[i];
    if (node->type != xnn_node_type_convert) {
      continue;
    }

    assert(node->num_inputs == 1);
    assert(node->num_outputs == 1);

    struct xnn_value* input_value = &subgraph->values[node->inputs[0]];
    struct xnn_value* output_value = &subgraph->values[node->outputs[0]];

    if (!xnn_value_is_static(input_value->allocation_type)) {
      continue;
    }

    xnn_log_debug("Constant-folding convert node %" PRIu32 ": value %" PRIu32
                  " -> %" PRIu32,
                  i, input_value->id, output_value->id);

    assert(input_value->data != nullptr);
    assert(output_value->data == nullptr);

    // 1. Allocate static memory for the output value (FP32 weights)
    const size_t output_size = xnn_tensor_get_size(output_value);
    output_value->data =
        xnn_allocate_zero_memory(output_size + XNN_EXTRA_BYTES);
    if (output_value->data == nullptr) {
      xnn_log_error(
          "Failed to allocate %zu bytes for constant-folded value %" PRIu32,
          output_size, output_value->id);
      return xnn_status_out_of_memory;
    }
    output_value->flags |= XNN_VALUE_FLAG_NEEDS_CLEANUP;
    output_value->size = output_size;
    // We want to keep the allocation type static to allow weights to be
    // consumed by ops that expect them to be static.
    output_value->allocation_type = xnn_allocation_type_static;
    output_value->static_convert.original_data = input_value->data;

    XNN_RETURN_IF_ERROR(
        xnn_run_unary_elementwise_nc(
            xnn_unary_convert, input_value->datatype, output_value->datatype,
            /*params=*/nullptr, /*input_quantization=*/nullptr,
            /*output_quantization=*/nullptr, /*flags=*/0,
            /*batch_size=*/xnn_shape_multiply_all_dims(&input_value->shape),
            /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1,
            /*threadpool=*/nullptr, /*input=*/input_value->data,
            /*output=*/output_value->data),
        "Failed to execute constant-fold convert for node %" PRIu32, i);

    // Invalidate the Convert node so it is skipped during runtime execution
    node->type = xnn_node_type_invalid;
  }

  return xnn_status_success;
}

enum xnn_status xnn_subgraph_alias_constant_folded_data(
    xnn_subgraph_t subgraph, xnn_weights_cache_t cache) {
  if (cache != nullptr) {
    for (uint32_t i = 0; i < subgraph->num_values; ++i) {
      const struct xnn_value* value = &subgraph->values[i];
      if (value->static_convert.original_data != nullptr) {
        assert(value->data != nullptr);
        XNN_RETURN_IF_ERROR(xnn_weights_cache_alias_data(
            cache, value->data, value->static_convert.original_data));
      }
    }
  }
  return xnn_status_success;
}

}  // namespace xnnpack
