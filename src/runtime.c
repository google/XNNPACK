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
#include <xnnpack/codecache.h>
#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/memory-planner.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>

#if defined(__MACH__)
#include <errno.h>
#include <time.h>
#elif defined(EMSCRIPTEN)
#include <emscripten/emscripten.h>
#elif XNN_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <errno.h>
#include <time.h>
#endif

enum xnn_status xnn_create_weights_cache(xnn_weights_cache_t* weights_cache_out)
{
  struct xnn_weights_cache* weights_cache = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create runtime: XNNPACK is not initialized");
    goto error;
  }

  weights_cache = xnn_allocate_zero_memory(sizeof(struct xnn_weights_cache));
  if (weights_cache == NULL) {
    xnn_log_error("failed to allocate %zu bytes for weights cache descriptor", sizeof(struct xnn_weights_cache));
    goto error;
  }

  status = xnn_init_weights_cache(weights_cache);
  if (status != xnn_status_success) {
    goto error;
  }
  *weights_cache_out = weights_cache;
  return xnn_status_success;

error:
  xnn_release_weights_cache(weights_cache);
  return status;
}

enum xnn_status xnn_delete_weights_cache(xnn_weights_cache_t weights_cache)
{
  enum xnn_status status = xnn_release_weights_cache(weights_cache);
  if (status != xnn_status_success) {
    return status;
  }
  xnn_release_memory(weights_cache);
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

  struct xnn_code_cache* code_cache = NULL;
#if XNN_PLATFORM_JIT
  code_cache = &runtime->code_cache;
  status = xnn_init_code_cache(code_cache);
  if (status != xnn_status_success) {
    goto error;
  }
#endif
  const struct xnn_caches caches = {
    .code_cache = code_cache,
    .weights_cache = weights_cache,
  };

  struct xnn_value* values = subgraph->values;
  for (size_t i = 0; i < subgraph->num_nodes; i++) {
    const struct xnn_node* node = subgraph->nodes + i;

    // Ignore fused nodes
    if (node->type != xnn_node_type_invalid) {
      assert(node->create != NULL);
      status = node->create(node, values, subgraph->num_values, runtime->opdata + i, &caches);
      if (status != xnn_status_success) {
        goto error;
      }
      runtime->opdata[i].setup = node->setup;
    }
  }

#if XNN_PLATFORM_JIT
  xnn_finalize_code_memory(&code_cache->cache.code);
#endif

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

  if (flags & XNN_FLAG_BASIC_PROFILING) {
    runtime->profiling = true;
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
    if (opdata->operator_objects[0] == NULL) {
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

static xnn_timestamp xnn_read_timer() {
  xnn_timestamp timestamp;
#if defined(__MACH__)
  timestamp = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  if (timestamp == 0) {
    xnn_log_warning("clock_gettime failed: error code %d", errno);
  }
#elif defined(__EMSCRIPTEN__)
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

static uint64_t xnn_get_elapsed_time(const xnn_timestamp* start, const xnn_timestamp* end) {
#if defined(__MACH__)
  const uint64_t kMicrosInNanos = 1000;
  return (end - start) / kMicrosInNanos;
#elif defined(__EMSCRIPTEN__)
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
  const uint64_t kMicrosInNanos = 1000;
  const uint64_t kNanosInSec = 1000 * 1000 * 1000;
  uint64_t start_ns = ((uint64_t) start->tv_sec * kNanosInSec) + ((uint64_t) start->tv_nsec);
  uint64_t end_ns = ((uint64_t) end->tv_sec * kNanosInSec) + ((uint64_t) end->tv_nsec);
  return (uint64_t) ((end_ns - start_ns) / kMicrosInNanos);
#endif
}

enum xnn_status xnn_get_profile_info_runtime(xnn_runtime_t runtime,
                                             enum xnn_profile_info param_name,
                                             size_t param_value_size,
                                             void* param_value,
                                             size_t* param_value_size_ret)
{
  if (!runtime->profiling) {
    return xnn_status_invalid_state;
  }
  enum xnn_status status = xnn_status_success;
  size_t num_ops = runtime->num_ops;
  size_t required_size = 0;
  size_t write_pos = 0;
  const struct xnn_operator_data* opdata = runtime->opdata;
  switch (param_name) {
    case xnn_profile_info_default:
      *param_value_size_ret = required_size;
      break;
    case xnn_profile_info_num_operators:
      required_size = num_ops * sizeof(uint64_t);
      if (param_value_size < required_size){
        *param_value_size_ret = required_size;
        status = xnn_status_out_of_memory;
      } else {
        uint64_t* data = (uint64_t*) param_value;
        for (size_t i = 0; i < runtime->num_ops; ++i) {
          size_t op_count = 0;
          for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
            if (opdata[i].operator_objects[j] != NULL) {
              ++op_count;
            }
          }
          data[i] = op_count;
        }
      }
      break;
    case xnn_profile_info_operator_name:
      write_pos = 0;
      for (size_t i = 0; i < runtime->num_ops; ++i) {
        const char* op_name = xnn_operator_type_to_string(opdata[i].operator_objects[0]->type);
        size_t name_len = strlen(op_name) + 1;
        required_size += name_len;
        if (param_value_size >= required_size) {
          memcpy(&param_value[write_pos], op_name, name_len);
          write_pos += name_len;
        }
      }
      if (param_value_size < required_size) {
        *param_value_size_ret = required_size;
        status = xnn_status_out_of_memory;
      }
      break;
    case xnn_profile_info_operator_timing:
      write_pos = 0;
      uint64_t* data = (uint64_t*) param_value;
      xnn_timestamp previous_ts = runtime->start_ts;
      for (size_t i = 0; i < runtime->num_ops; ++i) {
        for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
          if (opdata[i].operator_objects[j] != NULL) {
            required_size += sizeof(uint64_t);
            if (param_value_size >= required_size) {
              data[write_pos++] = xnn_get_elapsed_time(&previous_ts, &opdata[i].end_ts[j]);
              previous_ts = opdata[i].end_ts[j];
            }
          }
        }
      }
      if (param_value_size < required_size) {
        *param_value_size_ret = required_size;
        status = xnn_status_out_of_memory;
      }
      break;
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
  xnn_read_timer();
  for (size_t i = 0; i < runtime->num_ops; i++) {
    for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; j++) {
      if (runtime->opdata[i].operator_objects[j] == NULL) {
        // Operator was removed after fusion
        continue;
      }

      const enum xnn_status status = xnn_run_operator(runtime->opdata[i].operator_objects[j], runtime->threadpool);
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

      xnn_release_memory(runtime->blobs);
      xnn_release_simd_memory(runtime->workspace);
    }
#if XNN_PLATFORM_JIT
    xnn_release_code_cache(&runtime->code_cache);
#endif
    xnn_release_memory(runtime);
  }
  return xnn_status_success;
}
