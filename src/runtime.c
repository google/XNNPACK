// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/memory-planner.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


enum xnn_status xnn_create_runtime(
  xnn_subgraph_t subgraph,
  xnn_runtime_t* runtime_out)
{
  return xnn_create_runtime_v2(subgraph, NULL /* threadpool */, 0 /* flags */, runtime_out);
}

enum xnn_status xnn_create_runtime_v2(
  xnn_subgraph_t subgraph,
  pthreadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out)
{
  struct xnn_runtime* runtime = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create runtime: XNNPACK is not initialized");
    goto error;
  }

  const uint32_t optimization_flags = XNN_FLAG_SPARSE_INFERENCE | XNN_FLAG_HINT_FP16_INFERENCE | XNN_FLAG_FORCE_FP16_INFERENCE; 
  status = xnn_subgraph_optimize(subgraph, flags & optimization_flags);
  if (status != xnn_status_success) {
    xnn_log_error("failed to optimize subgraph");
    goto error;
  }

  status = xnn_status_out_of_memory;

  runtime = xnn_allocate_zero_memory(sizeof(struct xnn_runtime));
  if (runtime == NULL) {
    xnn_log_error("failed to allocate %zu bytes for runtime descriptor", sizeof(struct xnn_runtime));
    goto error;
  }

  runtime->opdata = xnn_allocate_zero_memory(sizeof(struct xnn_operator_data) * subgraph->num_nodes);
  if (runtime->opdata == NULL) {
    xnn_log_error("failed to allocate %zu bytes for opdata descriptors",
      sizeof(struct xnn_operator_data) * subgraph->num_nodes);
    goto error;
  }
  runtime->num_ops = subgraph->num_nodes;

  if (flags & XNN_FLAG_YIELD_WORKERS) {
    struct xnn_node* last_valid_node = NULL;
    for (size_t i = 0; i < subgraph->num_nodes; i++) {
      struct xnn_node* node = subgraph->nodes + i;
      if (node->type != xnn_node_type_invalid) {
        last_valid_node = node;
      }
    }
    if (last_valid_node != NULL) {
      last_valid_node->flags |= XNN_FLAG_YIELD_WORKERS;
    }
  }

  struct xnn_value* values = subgraph->values;
  for (size_t i = 0; i < subgraph->num_nodes; i++) {
    const struct xnn_node* node = subgraph->nodes + i;

    // Ignore fused nodes
    if (node->type != xnn_node_type_invalid) {
      assert(node->create != NULL);
      status = node->create(node, values, subgraph->num_values, runtime->opdata + i);
      if (status != xnn_status_success) {
        goto error;
      }
      runtime->opdata[i].setup = node->setup;
    }
  }

  runtime->blobs = xnn_allocate_zero_memory(sizeof(struct xnn_blob) * subgraph->num_values);
  if (runtime->blobs == NULL) {
    xnn_log_error("failed to allocate %zu bytes for blob descriptors",
      sizeof(struct xnn_blob) * subgraph->num_values);
    goto error;
  }
  runtime->num_blobs = subgraph->num_values;

  struct xnn_value_allocation_tracker mem_alloc_tracker;
  xnn_init_value_allocation_tracker(&mem_alloc_tracker, subgraph);

  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    struct xnn_value* value = &subgraph->values[i];
    struct xnn_blob* blob = &runtime->blobs[i];
    if (value->datatype != xnn_datatype_invalid && value->type == xnn_value_type_dense_tensor) {
      blob->size = xnn_tensor_get_size(subgraph, i);
      blob->data = (void*) (uintptr_t) value->data;
      if (blob->data == NULL) {
        if ((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
          // Value is purely internal to the runtime, and must be allocated in its workspace.
          xnn_add_value_allocation_tracker(&mem_alloc_tracker, i, round_up_po2(blob->size, XNN_EXTRA_BYTES));
        } else {
          // Value is non-static and external to the runtime: must be specified via a call to xnn_setup_runtime.
          blob->external = true;
        }
      }
    }
  }
  xnn_plan_value_allocation_tracker(&mem_alloc_tracker);

  if (mem_alloc_tracker.mem_arena_size != 0) {
    // XNN_EXTRA_BYTES ensures that out-of-bound reads of intermediate values don't segfault.
    const size_t mem_arena_size = mem_alloc_tracker.mem_arena_size + XNN_EXTRA_BYTES;
    runtime->workspace = xnn_allocate_simd_memory(mem_arena_size);
    if (runtime->workspace == NULL) {
      xnn_log_error("failed to allocate %zu bytes for runtime workspace", mem_arena_size);
      xnn_release_value_allocation_tracker(&mem_alloc_tracker);
      goto error;
    }
    for (size_t i = 0; i < subgraph->num_values; i++) {
      const struct xnn_value* value = &subgraph->values[i];
      struct xnn_blob* blob = &runtime->blobs[i];
      if (value->datatype != xnn_datatype_invalid && value->type == xnn_value_type_dense_tensor) {
        if (value->data == NULL && !blob->external) {
          // Value is purely internal to the runtime, allocate it in the workspace.
          blob->data = (void*) ((uintptr_t) runtime->workspace + mem_alloc_tracker.usage[i].alloc_offset);
        }
      }
    }
  }
  xnn_release_value_allocation_tracker(&mem_alloc_tracker);

  runtime->threadpool = threadpool;

  *runtime_out = runtime;
  return xnn_status_success;

error:
  xnn_delete_runtime(runtime);
  return status;
}

enum xnn_status xnn_setup_runtime(
  xnn_runtime_t runtime,
  size_t num_external_values,
  const struct xnn_external_value* external_values)
{
  // Validate inputs without changing internal state.
  // This ensures that runtime stays in consistent state in case validation fails midway.
  for (size_t i = 0; i < num_external_values; i++) {
    const struct xnn_external_value* external_value = &external_values[i];
    const uint32_t value_id = external_value->id;
    if (value_id >= runtime->num_blobs) {
      xnn_log_error("failed to setup runtime: out-of-bounds ID %" PRIu32 " in external value #%zu",
        value_id, i);
      return xnn_status_invalid_parameter;
    }

    const struct xnn_blob* blob = &runtime->blobs[value_id];
    if (!blob->external) {
      xnn_log_error("failed to setup runtime: Value %" PRIu32 " is not external", value_id);
      return xnn_status_invalid_parameter;
    }
  }

  // Apply runtime state changes.
  for (size_t i = 0; i < num_external_values; i++) {
    const struct xnn_external_value* external_value = &external_values[i];
    const uint32_t value_id = external_value->id;
    struct xnn_blob* blob = &runtime->blobs[value_id];
    blob->data = external_value->data;
  }

  for (size_t i = 0; i < runtime->num_ops; i++) {
    const struct xnn_operator_data* opdata = &runtime->opdata[i];
    if (opdata->operator_object == NULL) {
      // Operator was removed during optimization
      continue;
    }

    assert(opdata->setup != NULL);
    const enum xnn_status status = opdata->setup(opdata, runtime->blobs, runtime->num_blobs, runtime->threadpool);
    if (status != xnn_status_success) {
      xnn_log_error("failed to setup runtime: error in operator #%zu", i);
      return status;
    }
  }

  return xnn_status_success;
}

enum xnn_status xnn_invoke_runtime(
  xnn_runtime_t runtime)
{
  for (size_t i = 0; i < runtime->num_ops; i++) {
    if (runtime->opdata[i].operator_object == NULL) {
      // Operator was removed after fusion
      continue;
    }

    const enum xnn_status status = xnn_run_operator(runtime->opdata[i].operator_object, runtime->threadpool);
    if (status != xnn_status_success) {
      return status;
    }
  }
  return xnn_status_success;
}

enum xnn_status xnn_delete_runtime(
  xnn_runtime_t runtime)
{
  if (runtime != NULL) {
    if (runtime->opdata != NULL) {
      for (size_t i = 0; i < runtime->num_ops; i++) {
        xnn_delete_operator(runtime->opdata[i].operator_object);
      }
      xnn_release_memory(runtime->opdata);

      xnn_release_memory(runtime->blobs);
      xnn_release_simd_memory(runtime->workspace);
    }
    xnn_release_memory(runtime);
  }
  return xnn_status_success;
}
