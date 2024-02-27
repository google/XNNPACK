// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __MACH__
#define _POSIX_C_SOURCE 199309L
#endif

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h> // For snprintf.
#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/cache.h>
#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/memory-planner.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#elif XNN_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <errno.h>
#include <time.h>
#endif

#ifndef XNN_ENABLE_JIT
  #error "XNN_ENABLE_JIT is not defined"
#endif

enum xnn_status xnn_reshape_external_value(
    xnn_runtime_t runtime,
    uint32_t external_id,
    size_t num_dims,
    const size_t* dims) {
  if (external_id >= runtime->num_values) {
    xnn_log_error("failed to reshape runtime: out-of-bounds ID %" PRIu32 " in external value",
                  external_id);
    return xnn_status_invalid_parameter;
  }
  struct xnn_value* value = &runtime->values[external_id];
  if (value->flags & XNN_VALUE_FLAG_EXTERNAL_INPUT && value->allocation_type != xnn_allocation_type_external && value->allocation_type != xnn_allocation_type_static) {
    xnn_log_error("failed to reshape runtime: Value %" PRIu32 " is neither external nor static (%d)",
                  external_id, value->allocation_type);
    return xnn_status_invalid_parameter;
  }
  struct xnn_shape* shape = &value->shape;
  shape->num_dims = num_dims;
  for (size_t i = 0; i < num_dims; ++i) {
    shape->dim[i] = dims[i];
  }
  value->size = xnn_tensor_get_size(value);
  return xnn_status_success;
}

enum xnn_status
xnn_get_external_value_shape(xnn_runtime_t runtime, uint32_t external_id, size_t* num_dims, size_t* dims)
{
  if (external_id >= runtime->num_values) {
    xnn_log_error("failed to get external value shape: out-of-bounds ID %" PRIu32 " in external value", external_id);
    return xnn_status_invalid_parameter;
  }
  struct xnn_value* value = &runtime->values[external_id];
  if (value->allocation_type != xnn_allocation_type_external) {
    xnn_log_error(
      "failed to get external value shape: Value %" PRIu32 " is not external (%d)", external_id,
      value->allocation_type);
    return xnn_status_invalid_parameter;
  }
  if (num_dims == NULL || dims == NULL) {
    xnn_log_error("failed to get external value shape: null pointer");
    return xnn_status_invalid_parameter;
  }
  *num_dims = value->shape.num_dims;
  memcpy(dims, value->shape.dim, value->shape.num_dims * sizeof(size_t));
  return xnn_status_success;
}

enum xnn_status xnn_create_workspace(xnn_workspace_t* workspace_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create workspace: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  struct xnn_workspace* workspace = NULL;
  workspace = xnn_allocate_zero_memory(sizeof(struct xnn_workspace));
  if (workspace == NULL) {
    xnn_log_error("failed to allocate %zu bytes for workspace descriptor", sizeof(struct xnn_workspace));
    return xnn_status_out_of_memory;
  }
  workspace->ref_count = 1;
  *workspace_out = workspace;
  return xnn_status_success;
}

static inline void xnn_retain_workspace(xnn_workspace_t workspace)
{
  workspace->ref_count++;
}

enum xnn_status xnn_release_workspace(xnn_workspace_t workspace)
{
  assert(workspace->ref_count != 0);
  if (--workspace->ref_count == 0) {
    xnn_release_simd_memory(workspace->data);
    xnn_release_memory(workspace);
  }
  return xnn_status_success;
}

enum xnn_status xnn_create_weights_cache_with_size(size_t size, xnn_weights_cache_t* weights_cache_out)
{
  struct xnn_weights_cache_provider* cache_provider = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create weights cache: XNNPACK is not initialized");
    goto error;
  }

  cache_provider = xnn_allocate_zero_memory(sizeof(struct xnn_weights_cache_provider));
  if (cache_provider == NULL) {
    xnn_log_error("failed to allocate %zu bytes for weights cache provider descriptor", sizeof(struct xnn_weights_cache_provider));
    goto error;
  }

  cache_provider->context = xnn_allocate_zero_memory(sizeof(struct xnn_internal_weights_cache));
  if (cache_provider->context == NULL) {
    xnn_log_error("failed to allocate %zu bytes for weights cache descriptor", sizeof(struct xnn_internal_weights_cache));
    goto error;
  }

  status = xnn_internal_init_weights_cache_with_size(cache_provider->context, size);
  if (status != xnn_status_success) {
    goto error;
  }
  cache_provider->look_up = (size_t(*)(void*, const struct xnn_weights_cache_look_up_key*))xnn_internal_weights_cache_look_up;
  cache_provider->reserve_space = (void*(*)(void*, size_t))xnn_internal_reserve_space_in_weights_cache;
  cache_provider->look_up_or_insert = (size_t (*)(void*, const struct xnn_weights_cache_look_up_key*, void*, size_t))xnn_internal_get_or_insert_weights_cache;
  cache_provider->is_finalized = (bool (*)(void*))xnn_internal_weights_cache_is_finalized;
  cache_provider->offset_to_addr = (void*(*)(void*, size_t))xnn_internal_weights_cache_offset_to_addr;
  cache_provider->delete_cache = (enum xnn_status (*)(void*))xnn_internal_delete_weights_cache;
  *weights_cache_out = cache_provider;
  return xnn_status_success;

error:
  xnn_internal_release_weights_cache(cache_provider->context);
  return status;
}

enum xnn_status xnn_create_weights_cache(xnn_weights_cache_t* weights_cache_out)
{
  return xnn_create_weights_cache_with_size(XNN_DEFAULT_WEIGHTS_BUFFER_SIZE, weights_cache_out);
}

enum xnn_status xnn_delete_weights_cache(xnn_weights_cache_t weights_cache)
{
  if XNN_LIKELY(weights_cache != NULL) {
    enum xnn_status status = xnn_internal_release_weights_cache(weights_cache->context);
    if (status != xnn_status_success) {
      return status;
    }
    xnn_release_memory(weights_cache->context);
    xnn_release_memory(weights_cache);
  }
  return xnn_status_success;
}

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
  return xnn_create_runtime_v3(subgraph, /* weights_cache */ NULL, threadpool, flags, runtime_out);
}

enum xnn_status xnn_create_runtime_v3(
  xnn_subgraph_t subgraph,
  xnn_weights_cache_t weights_cache,
  pthreadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out)
{
  xnn_workspace_t workspace;
  enum xnn_status status = xnn_create_workspace(&workspace);
  if (status != xnn_status_success) {
    return status;
  }
  status = xnn_create_runtime_v4(subgraph, weights_cache, workspace, threadpool, flags, runtime_out);
  // Release workspace regardless of return status of creating runtime.
  xnn_release_workspace(workspace);
  return status;
}

static enum xnn_status initialize_workspace_values(
    xnn_runtime_t runtime,
    struct xnn_value_allocation_tracker* mem_alloc_tracker,
    size_t old_persistent_size)
{
  assert(runtime->workspace != NULL);
  const size_t persistent_size = runtime->workspace->persistent_size;
  size_t mem_arena_size = mem_alloc_tracker->mem_arena_size + persistent_size;
  if (mem_arena_size == 0) {
    return xnn_status_success;
  }
  // Sparse microkernels can read up to 2 * XNN_EXTRA_BYTES beyond array bounds.
  mem_arena_size += 2 * XNN_EXTRA_BYTES;

  // Records how much the workspace has moved by due to allocating a larger workspace.
  ptrdiff_t workspace_data_delta = 0;
  // Allocates larger workspace here if needed.
  if (runtime->workspace->size < mem_arena_size) {
    void* old_workspace_data = runtime->workspace->data;
    void* new_workspace_data = xnn_allocate_zero_simd_memory(mem_arena_size);
    if (new_workspace_data == NULL) {
      xnn_log_error("failed to allocate %zu bytes for runtime workspace", mem_arena_size);
      return xnn_status_out_of_memory;
    }
    runtime->workspace->data = new_workspace_data;
    runtime->workspace->size = mem_arena_size;
    // Keep track of how much the workspace data moved.
    if (old_workspace_data != NULL) {
      workspace_data_delta = (uintptr_t) new_workspace_data - (uintptr_t) old_workspace_data;
      // Persistent data needs to be copied if workspace grew.
      memcpy(new_workspace_data, old_workspace_data, old_persistent_size);
      xnn_release_simd_memory(old_workspace_data);
    }
    xnn_log_debug("created workspace of size %zu, old workspace %p, new workspace %p, delta %td",
                  mem_arena_size, old_workspace_data, new_workspace_data, workspace_data_delta);
  }

  assert(runtime->workspace->size >= mem_arena_size);

  // Initialize current runtime's value pointers.
  size_t persistent_offset = 0;
  for (size_t i = 0; i < runtime->num_values; i++) {
    struct xnn_value* value = &runtime->values[i];
    if (!xnn_value_is_valid(value)) {
      continue;
    }

    if (value->allocation_type == xnn_allocation_type_workspace) {
      // Value is purely internal to the runtime, allocate it in the workspace.
      value->data =
        (void*) ((uintptr_t) runtime->workspace->data + persistent_size + mem_alloc_tracker->usage[i].alloc_offset);
      if (value->datatype == xnn_datatype_qdint8) {
        value->quantization.dynamic_params =
          (void*) ((uintptr_t) runtime->workspace->data + persistent_size + mem_alloc_tracker->usage[i].alloc_offset
                   + xnn_tensor_get_rounded_size(value));

      }
    } else if (value->allocation_type == xnn_allocation_type_persistent) {
      value->data = (void*) ((uintptr_t) runtime->workspace->data + persistent_offset);
      persistent_offset += xnn_tensor_get_rounded_size(value);
    }
  }
  assert(persistent_offset == persistent_size);

  // Initialize operator workspace values.
  for (size_t i = 0; i < runtime->num_ops; i++) {
    const struct xnn_usage_record* usage = &mem_alloc_tracker->usage[runtime->num_values + i];
    if (usage->opdata_id == XNN_INVALID_NODE_ID) {
      continue;
    }
    struct xnn_operator_data* opdata = &runtime->opdata[usage->opdata_id];
    opdata->workspace = (void*) ((uintptr_t) runtime->workspace->data + persistent_size + usage->alloc_offset);
  }

  // Adjust the value pointers of all runtimes that share this workspace.
  if (workspace_data_delta != 0) {
    for (struct xnn_runtime* rt = runtime->workspace->first_user; rt != NULL; rt = rt->next_workspace_user) {
      // The current runtime already has the correct offset.
      if (rt == runtime) {
        continue;
      }
      // This memory for this runtime has not yet been planned, so it doesn't have any pointers into workspace, so does not need to
      // be updated.
      if (!rt->memory_planned) {
        continue;
      }

      // Adjust offsets of values in workspace.
      for (size_t i = 0; i < rt->num_values; i++) {
        struct xnn_value* value = &rt->values[i];
        if (value->allocation_type == xnn_allocation_type_workspace ||
            value->allocation_type == xnn_allocation_type_persistent) {
          if (value->data != NULL) {
            // Data can be null as the runtime using this workspace might not have been set up.
            value->data = (void*) ((uintptr_t) value->data + workspace_data_delta);
            if (value->datatype == xnn_datatype_qdint8) {
              value->quantization.dynamic_params = (void*) ((uintptr_t) value->quantization.dynamic_params
                                                            + workspace_data_delta);
            }
          }
        }
      }

      // Adjust offsets of op workspaces.
      for (size_t i = 0; i < rt->num_ops; i++) {
        struct xnn_operator_data* opdata = &rt->opdata[i];
        if (opdata->operator_objects[0] == NULL) {
          // Operator was removed during optimization
          continue;
        }

        if (opdata->workspace != NULL) {
          opdata->workspace = (void*) ((uintptr_t) opdata->workspace + workspace_data_delta);
        }
      }
      // This runtime has not ever been setup yet, so it doesn't have any pointers into workspace, so does not need to
      // be updated.
      if (!rt->has_been_setup) {
        continue;
      }
      // Re-setup all the nodes to adjust input/output pointers.
      for (size_t i = 0; i < rt->num_ops; i++) {
        struct xnn_operator_data* opdata = &rt->opdata[i];
        for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
          if (opdata->operator_objects[j] == NULL) {
            // Operator was removed during optimization
            continue;
          }
          assert(opdata->setup != NULL);
          const enum xnn_status status = opdata->setup(opdata, rt->values, rt->num_values, rt->threadpool);
          if (status != xnn_status_success) {
            xnn_log_error("failed to setup runtime: error in operator #%zu", i);
            return status;
          }
        }
      }
    }
  }

  return xnn_status_success;
}

// Output can reuse input memory if both are allocated in the workspace.
// If input has more than 1 consumer, we can't track all the consumers and update the first_consumer, so bail out.
// Output memory fits in input memory. One of the inputs to a binary node could be implicitly broadcasted.
static bool input_memory_can_be_reused(const xnn_runtime_t runtime, size_t input_id, size_t output_id)
{
  if (input_id == XNN_INVALID_VALUE_ID || output_id == XNN_INVALID_VALUE_ID) {
    return false;
  }
  const struct xnn_value* input = &runtime->values[input_id];
  const struct xnn_value* output = &runtime->values[output_id];
  const bool output_memory_fits = xnn_tensor_get_size(input) == xnn_tensor_get_size(output);
  assert(input->num_consumers != 0);
  return input->allocation_type == xnn_allocation_type_workspace &&
      output->allocation_type == xnn_allocation_type_workspace &&
      input->num_consumers == 1 && output_memory_fits;
}

// An in-place operation reuses the input tensor's memory for its output. Examples are element-wise unary operations
// like activation functions. Usually, an output tensor is allocated space. For an in-place operation, we want the
// output tensor to share the input tensor's memory. We do this by calling xnn_mark_tensor_as_reuse, which:
// - sets the tensor_size of output tensor's usage record to 0
// - mark this usage record as reusing another tensor's memory
// - remember the id of the tensor which we will reuse the alloc_offset to set onto the output tensor
static void optimize_tensor_allocation_for_in_place_operations(
  struct xnn_value_allocation_tracker* tracker,
  const xnn_runtime_t runtime)
{
  for (uint32_t n = 0; n < runtime->num_ops; n++) {
    const struct xnn_operator_data* node = &runtime->opdata[n];
    switch (node->type) {
      case xnn_node_type_abs:
      case xnn_node_type_add2:
      case xnn_node_type_bankers_rounding:
      case xnn_node_type_ceiling:
      case xnn_node_type_clamp:
      case xnn_node_type_copy:
      case xnn_node_type_divide:
      case xnn_node_type_elu:
      case xnn_node_type_floor:
      case xnn_node_type_hardswish:
      case xnn_node_type_leaky_relu:
      case xnn_node_type_maximum2:
      case xnn_node_type_minimum2:
      case xnn_node_type_multiply2:
      case xnn_node_type_negate:
      case xnn_node_type_prelu:
      case xnn_node_type_sigmoid:
      case xnn_node_type_softmax:
      case xnn_node_type_square:
      case xnn_node_type_square_root:
      case xnn_node_type_squared_difference:
      case xnn_node_type_static_reshape:
      case xnn_node_type_subtract:
        // Valid operation types that can be optimized.
        break;
      default:
        continue;
    }

    // Check all of the node's input to see which we can reuse.
    uint32_t input_id = XNN_INVALID_VALUE_ID;
    for (size_t i = 0; i < node->num_inputs; i++) {
      if (input_memory_can_be_reused(runtime, node->inputs[i], node->outputs[0])) {
        input_id = node->inputs[i];
        break;  // Found an input we can reuse, early exit.
      }
    }
    // Check input_id and return if invalid.
    if (input_id == XNN_INVALID_VALUE_ID) {
      continue;
    }

    // TODO(zhin): consider aliasing input to output rather than output to input.
    struct xnn_value* output = &runtime->values[node->outputs[0]];
    if (output->num_consumers == 1) {
      uint32_t reuse_id = input_id;
      // If the tensor we are reusing is itself reused, find the "root tensor" to be reused.
      while (tracker->usage[reuse_id].reuse_value_id != XNN_INVALID_VALUE_ID) {
        reuse_id = tracker->usage[reuse_id].reuse_value_id;
      }
      // We only support when output has a single consumer because we cannot easily find all consumer nodes
      // without traversing the entire graph. This will require tracking output->last_consumer in the future.
      assert(tracker->usage[reuse_id].last_node < output->first_consumer);
      xnn_log_debug("reusing tensor id #%" PRIu32 " memory for tensor id #%" PRIu32 " Node #%" PRIu32 " %s",
                    reuse_id, output->id, node->id, xnn_node_type_to_string(node->type));
      xnn_mark_tensor_as_reuse(tracker, output->id, reuse_id, output->first_consumer);
    }
  }
}

enum xnn_status xnn_create_runtime_v4(
  xnn_subgraph_t subgraph,
  xnn_weights_cache_t weights_cache,
  xnn_workspace_t workspace,
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

  if (workspace == NULL) {
    xnn_log_error("failed to create runtime: workspace is NULL");
    status = xnn_status_invalid_parameter;
    goto error;
  }

  const uint32_t optimization_flags = XNN_FLAG_HINT_SPARSE_INFERENCE | XNN_FLAG_HINT_FP16_INFERENCE |
    XNN_FLAG_FORCE_FP16_INFERENCE | XNN_FLAG_NO_OPERATOR_FUSION;
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
      sizeof(struct xnn_operator_data) * (size_t) subgraph->num_nodes);
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

  if (flags & XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER) {
    for (size_t i = 0; i < subgraph->num_nodes; i++) {
      struct xnn_node* node = subgraph->nodes + i;
      switch (node->type) {
        case xnn_node_type_convolution_2d:
        case xnn_node_type_depthwise_convolution_2d:
        case xnn_node_type_static_resize_bilinear_2d:
          node->flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
          break;
        default:
          break;
      }
    }
  }

  struct xnn_code_cache* code_cache = NULL;
  #if XNN_PLATFORM_JIT
    if (flags & XNN_FLAG_JIT) {
      #if !XNN_ENABLE_JIT
        // Warn and continue without JIT enabled.
        xnn_log_warning("unable to enable JIT: not compiled with JIT enabled");
      #else
        code_cache = &runtime->code_cache;
        status = xnn_init_code_cache(code_cache);
        if (status != xnn_status_success) {
          goto error;
        }
      #endif
    }
  #endif

  runtime->values = xnn_allocate_zero_memory(sizeof(struct xnn_value) * subgraph->num_values);
  if (runtime->values == NULL) {
    xnn_log_error("failed to allocate %zu bytes for runtime's value descriptors",
      sizeof(struct xnn_value) * (size_t) subgraph->num_values);
    goto error;
  }

  // Run a final analysis phase, no more modifications after this point.
  xnn_subgraph_analyze_consumers_and_producers(subgraph);
  // Make a copy of subgraph values since we can change them and runtime can outlive subgraph.
  for (size_t i = 0; i < subgraph->num_values; i++) {
    xnn_value_copy(runtime->values + i, subgraph->values + i);
    // Value copy doesn't copy the id, but we want the same ID.
    runtime->values[i].id = subgraph->values[i].id;
  }
  runtime->num_values = subgraph->num_values;
  // No more optimizations should be performed on subgraph at this point, since modifications on the subgraph will not
  // be copied to the runtime's values.

  for (size_t i = 0; i < subgraph->num_nodes; i++) {
    const struct xnn_node* node = subgraph->nodes + i;

    // Initialize common fields we need for analysis.
    runtime->opdata[i].type = node->type;
    runtime->opdata[i].flags = node->flags;
    runtime->opdata[i].id = node->id;
    runtime->opdata[i].num_inputs = node->num_inputs;
    runtime->opdata[i].num_outputs = node->num_outputs;
    // Copy all inputs (not just num_inputs) to get all invalid ID (e.g. no bias).
    for (size_t input_i = 0; input_i < node->num_inputs; input_i++) {
      runtime->opdata[i].inputs[input_i] = node->inputs[input_i];
    }
    for (size_t output_i = 0; output_i < node->num_outputs; output_i++) {
      runtime->opdata[i].outputs[output_i] = node->outputs[output_i];
    }

    // Ignore fused nodes
    if (node->type != xnn_node_type_invalid) {
      assert(node->create != NULL);
      status = node->create(node, runtime->values, runtime->num_values, runtime->opdata + i, code_cache, weights_cache);
      if (status != xnn_status_success) {
        goto error;
      }
      runtime->opdata[i].setup = node->setup;
      runtime->opdata[i].reshape = node->reshape;
    }
  }

  #if XNN_PLATFORM_JIT
    if (code_cache != NULL) {
      xnn_finalize_code_memory(&code_cache->cache.code);
    }
  #endif

  for (uint32_t i = 0; i < runtime->num_values; i++) {
    struct xnn_value* value = &runtime->values[i];
    if (!xnn_value_is_valid(value)) {
      continue;
    }

    if (value->fp16_compatible && xnn_value_is_static(value)) {
      // Value is static and has been converted to FP16 in a new buffer.
      value->allocation_type = xnn_allocation_type_dynamic;
      // Runtime takes ownership of the data from subgraph.
      value->data = subgraph->values[i].data;
      subgraph->values[i].data = NULL;
    }
  }

  xnn_retain_workspace(workspace);
  runtime->workspace = workspace;
  runtime->next_workspace_user = runtime->workspace->first_user;
  runtime->workspace->first_user = runtime;

  if (flags & XNN_FLAG_BASIC_PROFILING) {
    runtime->profiling = true;
  }

  runtime->threadpool = threadpool;

  *runtime_out = runtime;
  return xnn_status_success;

error:
  xnn_delete_runtime(runtime);
  return status;
}

enum xnn_status xnn_plan_memory(
    xnn_runtime_t runtime) {
  enum xnn_status status = xnn_status_invalid_state;
  struct xnn_value_allocation_tracker mem_alloc_tracker;
  xnn_init_value_allocation_tracker(&mem_alloc_tracker, runtime);

  size_t persistent_size = 0;

  for (uint32_t i = 0; i < runtime->num_values; i++) {
    const struct xnn_value* value = &runtime->values[i];
    if (!xnn_value_is_valid(value)) {
      continue;
    }

    if (value->allocation_type == xnn_allocation_type_workspace) {
      // Value is purely internal to the runtime, and must be allocated in its workspace.
      size_t tensor_size = xnn_tensor_get_rounded_size(value);
      if (value->datatype == xnn_datatype_qdint8) {
        const size_t batch_dims_size = xnn_shape_multiply_batch_dims(&value->shape, value->quantization.num_nonbatch_dims);
        tensor_size += xnn_get_rounded_size((batch_dims_size + XNN_EXTRA_QUANTIZATION_PARAMS)
                                    * sizeof(struct xnn_dynamic_quantization_params));
      }
      xnn_add_value_allocation_tracker(&mem_alloc_tracker, i, tensor_size);
    } else if (value->allocation_type == xnn_allocation_type_persistent) {
      persistent_size += xnn_tensor_get_rounded_size(value);
    }
  }
  size_t old_persistent_size = runtime->workspace->persistent_size;
  runtime->workspace->persistent_size = persistent_size;

  for (uint32_t opdata_id = 0; opdata_id < runtime->num_ops; opdata_id++) {
    struct xnn_operator_data* opdata = &runtime->opdata[opdata_id];
    xnn_add_operator_workspace_allocation_tracker(
        &mem_alloc_tracker, runtime->num_values + opdata_id, xnn_get_rounded_size(opdata->workspace_size),
        opdata_id);
  }

  optimize_tensor_allocation_for_in_place_operations(&mem_alloc_tracker, runtime);
  xnn_plan_value_allocation_tracker(&mem_alloc_tracker);

  status = initialize_workspace_values(runtime, &mem_alloc_tracker, old_persistent_size);
  if (status != xnn_status_success) {
    goto error;
  }

  xnn_release_value_allocation_tracker(&mem_alloc_tracker);

  return xnn_status_success;

error:
  xnn_release_value_allocation_tracker(&mem_alloc_tracker);
  return status;
}

enum xnn_status xnn_reshape_runtime(
  xnn_runtime_t runtime)
{
  bool reallocation_required = false;

  for (uint32_t opdata_id = 0; opdata_id < runtime->num_ops; opdata_id++) {
    struct xnn_operator_data* opdata = &runtime->opdata[opdata_id];
    if (opdata->operator_objects[0] == NULL) {
      // Operator was removed during optimization
      continue;
    }
    assert(opdata->reshape != NULL);
    xnn_log_debug("reshaping operator %u (%s)", opdata_id,
                  xnn_operator_type_to_string(opdata->operator_objects[0]->type));
    enum xnn_status status = opdata->reshape(opdata, runtime->values, runtime->num_values, /*threadpool=*/NULL);
    if (status == xnn_status_reallocation_required) {
      reallocation_required = true;
    } else if (status != xnn_status_success) {
      xnn_log_error("Operator #%u: %s failed reshape", opdata_id, xnn_operator_type_to_string(opdata->operator_objects[0]->type));
      return status;
    }
  }
  if (reallocation_required || !runtime->memory_planned) {
    runtime->memory_planned = true;
    return xnn_plan_memory(runtime);
  } else {
  }
  return xnn_status_success;
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
    if (value_id >= runtime->num_values) {
      xnn_log_error("failed to setup runtime: out-of-bounds ID %" PRIu32 " in external value #%zu",
                    value_id, i);
      return xnn_status_invalid_parameter;
    }

    const struct xnn_value* value = &runtime->values[value_id];
    if (value->allocation_type != xnn_allocation_type_external) {
      xnn_log_error("failed to setup runtime: Value %" PRIu32 " is not external (%d)", value_id, value->allocation_type);
      return xnn_status_invalid_parameter;
    }
  }

  // Apply runtime state changes.
  for (size_t i = 0; i < num_external_values; i++) {
    const struct xnn_external_value* external_value = &external_values[i];
    const uint32_t value_id = external_value->id;
    struct xnn_value* value = &runtime->values[value_id];
    value->data = external_value->data;
  }

  for (uint32_t opdata_id = 0; opdata_id < runtime->num_ops; opdata_id++) {
    struct xnn_operator_data* opdata = &runtime->opdata[opdata_id];
    for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
      if (opdata->operator_objects[j] == NULL) {
        // Operator was removed during optimization
        continue;
      }

      assert(opdata->reshape != NULL);
      enum xnn_status status = opdata->reshape(opdata, runtime->values, runtime->num_values, runtime->threadpool);
      if (status != xnn_status_success && status != xnn_status_reallocation_required) {
        xnn_log_error("failed to setup runtime: error in reshaping operator #%u", opdata_id);
        return status;
      }
    }
  }

  enum xnn_status status = status = xnn_plan_memory(runtime);
  runtime->memory_planned = true;

  for (uint32_t opdata_id = 0; opdata_id < runtime->num_ops; opdata_id++) {
    struct xnn_operator_data* opdata = &runtime->opdata[opdata_id];
    for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
      if (opdata->operator_objects[j] == NULL) {
        // Operator was removed during optimization
        continue;
      }

      assert(opdata->setup != NULL);
      enum xnn_status status = opdata->setup(opdata, runtime->values, runtime->num_values, runtime->threadpool);
      if (status != xnn_status_success) {
        xnn_log_error("failed to setup runtime: error in setting pointers of operator #%u", opdata_id);
        return status;
      }
    }
  }

  runtime->has_been_setup = true;

  return xnn_status_success;
}

enum xnn_status xnn_setup_runtime_v2(
  xnn_runtime_t runtime,
  size_t num_external_values,
  const struct xnn_external_value* external_values)
{
  // Validate inputs without changing internal state.
  // This ensures that runtime stays in consistent state in case validation fails midway.
  for (size_t i = 0; i < num_external_values; i++) {
    const struct xnn_external_value* external_value = &external_values[i];
    const uint32_t value_id = external_value->id;
    if (value_id >= runtime->num_values) {
      xnn_log_error("failed to setup runtime: out-of-bounds ID %" PRIu32 " in external value #%zu",
                    value_id, i);
      return xnn_status_invalid_parameter;
    }

    const struct xnn_value* value = &runtime->values[value_id];
    if (value->allocation_type != xnn_allocation_type_external) {
      xnn_log_error("failed to setup runtime: Value %" PRIu32 " is not external (%d)", value_id, value->allocation_type);
      return xnn_status_invalid_parameter;
    }
  }

  // Apply runtime state changes.
  for (size_t i = 0; i < num_external_values; i++) {
    const struct xnn_external_value* external_value = &external_values[i];
    const uint32_t value_id = external_value->id;
    struct xnn_value* value = &runtime->values[value_id];
    value->data = external_value->data;
  }

  for (uint32_t opdata_id = 0; opdata_id < runtime->num_ops; opdata_id++) {
    struct xnn_operator_data* opdata = &runtime->opdata[opdata_id];

    if (opdata->operator_objects[0] == NULL) {
      // Operator was removed during optimization
      continue;
    }
    assert(opdata->setup != NULL);
    enum xnn_status status = opdata->setup(opdata, runtime->values, runtime->num_values, runtime->threadpool);
    if (status != xnn_status_success) {
      xnn_log_error("failed to setup runtime: error in setting pointers of operator #%u", opdata_id);
      return status;
    }
  }

  runtime->has_been_setup = true;

  return xnn_status_success;
}

static xnn_timestamp xnn_read_timer() {
  xnn_timestamp timestamp;
#ifdef __MACH__
  timestamp = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  if (timestamp == 0) {
    xnn_log_warning("clock_gettime failed: error code %d", errno);
  }
#elif __EMSCRIPTEN__
  timestamp = emscripten_get_now();
#elif XNN_PLATFORM_WINDOWS
  BOOL res = QueryPerformanceCounter(&timestamp);
  if (!res) {
    xnn_log_error("QueryPerformanceCounter failed: error code %u", GetLastError());
    memset(&timestamp, 0, sizeof(timestamp));
  }
#else
  int res = clock_gettime(CLOCK_MONOTONIC, &timestamp);
  if (res != 0) {
    xnn_log_error("clock_gettime failed: error code %d", errno);
    memset(&timestamp, 0, sizeof(timestamp));
  }
#endif
  return timestamp;
}

static inline uint64_t xnn_get_elapsed_time(const xnn_timestamp* start, const xnn_timestamp* end) {
#ifdef __MACH__
  const uint64_t kMicrosInNanos = 1000;
  return (*end - *start) / kMicrosInNanos;
#elif __EMSCRIPTEN__
  const double kMillisInMicros = 1.0e3;
  return (uint64_t) ((*end - *start) * kMillisInMicros);
#elif XNN_PLATFORM_WINDOWS
  const uint64_t kMicrosInSec = 1000 * 1000;
  LARGE_INTEGER frequency;
  BOOL res = QueryPerformanceFrequency(&frequency);
  if (!res) {
    xnn_log_error("QueryPerformanceFrequency failed: error code %u", GetLastError());
    return 0;
  }
  return ((end->QuadPart - start->QuadPart) * kMicrosInSec) / frequency.QuadPart;
#else
  const uint64_t kNanosInMicro = UINT64_C(1000);
  const uint64_t kNanosInSec = UINT64_C(1000000000);
  const uint64_t secs = (end->tv_sec - start->tv_sec) * kNanosInSec;
  const uint64_t ns_secs = (end->tv_nsec - start->tv_nsec);
  return (secs + ns_secs) / kNanosInMicro;
#endif
}

enum xnn_status xnn_get_runtime_profiling_info(xnn_runtime_t runtime,
                                               enum xnn_profile_info param_name,
                                               size_t param_value_size,
                                               void* param_value,
                                               size_t* param_value_size_ret)
{
  if (!runtime->profiling) {
    return xnn_status_invalid_state;
  }
  enum xnn_status status = xnn_status_success;
  size_t required_size = 0;
  const struct xnn_operator_data* opdata = runtime->opdata;
  switch (param_name) {
    case xnn_profile_info_num_operators:
      required_size = sizeof(size_t);
      if (param_value_size < required_size){
        *param_value_size_ret = required_size;
        status = xnn_status_out_of_memory;
      } else {
        size_t num_valid_ops = 0;
        for (size_t i = 0; i < runtime->num_ops; ++i) {
          if (opdata[i].operator_objects[0] != NULL) {
            num_valid_ops += 1;
          }
        }
        memcpy(param_value, &num_valid_ops, required_size);
      }
      break;
    case xnn_profile_info_operator_name:
      for (size_t i = 0; i < runtime->num_ops; ++i) {
        if (opdata[i].operator_objects[0] != NULL) {
          const char* op_name = xnn_operator_type_to_string(opdata[i].operator_objects[0]->type);
          size_t op_name_len = strlen(op_name) + 1;
          if (opdata[i].operator_objects[0]->ukernel.type != xnn_microkernel_type_default ) {
            op_name_len += strlen(xnn_microkernel_type_to_string(opdata[i].operator_objects[0]->ukernel.type)) + 1;
          }
          required_size += op_name_len;
        }
      }
      if (param_value_size < required_size) {
        *param_value_size_ret = required_size;
        status = xnn_status_out_of_memory;
      } else {
        char* name_out = (char*) param_value;
        for (size_t i = 0; i < runtime->num_ops; ++i) {
          if (opdata[i].operator_objects[0] != NULL) {
            const char* op_name = xnn_operator_type_to_string(opdata[i].operator_objects[0]->type);
            size_t op_name_len = strlen(op_name) + 1;
            if (opdata[i].operator_objects[0]->ukernel.type != xnn_microkernel_type_default ) {
              const char* ukernel_type = xnn_microkernel_type_to_string(opdata[i].operator_objects[0]->ukernel.type);
              op_name_len += strlen(ukernel_type) + 1;
              snprintf(name_out, op_name_len, "%s %s", op_name, ukernel_type);
            } else {
              snprintf(name_out, op_name_len, "%s", op_name);
            }
            name_out += op_name_len;
          }
        }
      }
      break;
    case xnn_profile_info_operator_timing:
    {
      size_t num_valid_ops = 0;
      for (size_t i = 0; i < runtime->num_ops; ++i) {
        if (opdata[i].operator_objects[0] != NULL) {
          num_valid_ops += 1;
        }
      }
      required_size = num_valid_ops * sizeof(uint64_t);
      if (param_value_size < required_size) {
        *param_value_size_ret = required_size;
        status = xnn_status_out_of_memory;
      } else {
        xnn_timestamp previous_ts = runtime->start_ts;
        uint64_t* data = (uint64_t*) param_value;
        for (size_t i = 0; i < runtime->num_ops; ++i) {
          if (opdata[i].operator_objects[0] != NULL) {
            uint64_t op_time = 0;
            for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
              if (opdata[i].operator_objects[j] != NULL) {
                op_time += xnn_get_elapsed_time(&previous_ts, &opdata[i].end_ts[j]);
                previous_ts = opdata[i].end_ts[j];
              }
            }
            *data++ = op_time;
          }
        }
      }
      break;
    }
    default:
      status = xnn_status_invalid_parameter;
  }
  return status;
}

enum xnn_status xnn_invoke_runtime(
  xnn_runtime_t runtime)
{
  if (runtime->profiling) {
    runtime->start_ts = xnn_read_timer();
  }
  for (size_t i = 0; i < runtime->num_ops; i++) {
    for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
      if (runtime->opdata[i].operator_objects[j] == NULL) {
        // Operator was removed after fusion
        continue;
      }

      const enum xnn_status status = xnn_run_operator_with_index(runtime->opdata[i].operator_objects[j], i, j, runtime->threadpool);
      if (status != xnn_status_success) {
        return status;
      }
      if (runtime->profiling) {
        runtime->opdata[i].end_ts[j] = xnn_read_timer();
      }
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
        for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
          xnn_delete_operator(runtime->opdata[i].operator_objects[j]);
        }
      }
      xnn_release_memory(runtime->opdata);

      if (runtime->values != NULL) {
        // Release the buffers created during FP16 rewrite.
        for (size_t i = 0; i < runtime->num_values; i++) {
          struct xnn_value* value = &runtime->values[i];
          if (value->allocation_type == xnn_allocation_type_dynamic) {
            xnn_release_memory(value->data);
          }
        }
        xnn_release_memory(runtime->values);
      }

      if (runtime->workspace != NULL) {
        // Remove this runtime from the list of users of the workspace.
        assert(runtime->workspace->first_user != NULL);
        if (runtime->workspace->first_user == runtime) {
          runtime->workspace->first_user = runtime->next_workspace_user;
        } else {
          xnn_runtime_t prev = runtime->workspace->first_user;
          xnn_runtime_t curr = prev->next_workspace_user;
          while (curr != runtime) {
            prev = curr;
            curr = curr->next_workspace_user;
          }
          assert(curr == runtime);
          prev->next_workspace_user = curr->next_workspace_user;
        }
        xnn_release_workspace(runtime->workspace);
      }
    }
#if XNN_PLATFORM_JIT
    if (xnn_code_cache_valid(&runtime->code_cache)) {
      xnn_release_code_cache(&runtime->code_cache);
    }
#endif
    xnn_release_memory(runtime);
  }
  return xnn_status_success;
}
