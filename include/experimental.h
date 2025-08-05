// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The contents of this header is experimental and subject to change or removal
// without notice.

// clang-format off

#ifndef XNNPACK_INCLUDE_EXPERIMENTAL_H_
#define XNNPACK_INCLUDE_EXPERIMENTAL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Enable Slinky (if available).
#define XNN_FLAG_SLINKY_ENABLED 0x00008000

/// If Slinky is enabled, disable any scheduling.
#define XNN_FLAG_SLINKY_NO_SCHEDULE 0x00010000

/// If Slinky is enabled, assume shapes are concrete (and rebuild pipeline in
/// reshape). This makes reshaping more expensive, but may reduce overhead in
/// some cases.
#define XNN_FLAG_SLINKY_STATIC_BOUNDS 0x00020000

/// If Slinky is enabled, disable asserts in Slinky pipelines.
#define XNN_FLAG_SLINKY_NO_CHECKS 0x00040000

/// An abstract interface of a parallel task scheduler.
struct xnn_scheduler {
  /// Get the number of tasks that can be executed concurrently.
  int (*num_threads)(struct xnn_scheduler*);

  /// Schedule `task` to be called, with `context` as its argument.
  void (*schedule)(struct xnn_scheduler*, void* context, void (*task)(void* context));
};

typedef struct xnn_scheduler* xnn_scheduler_t;

/// Create a Runtime object from a subgraph with Slinky enabled.
///
/// @param subgraph - a Subgraph object with all Values and Nodes that would be handled by the runtime. No Values or
///                   Nodes can be added to the runtime once it is constructed.
/// @param weights_cache - a cache for packed weights. The runtime will look up and reuse packed weights in this cache,
///                        this will reduce memory allocated for packed weights.
/// @param workspace - a workspace to hold internal tensors. The runtime will allocate space used for internal tensors
///                    and track them using workspace. Workspace can be shared and reused across different runtimes. If
///                    workspace is NULL, there will be no sharing: each runtime has its own workspace.
/// @param scheduler - scheduler to implement parallel task execution.
/// @param flags - binary features of the runtime. The only currently supported values are
///                XNN_FLAG_HINT_SPARSE_INFERENCE, XNN_FLAG_HINT_FP16_INFERENCE, XNN_FLAG_FORCE_FP16_INFERENCE,
///                XNN_FLAG_SLINKY_STATIC_BOUNDS, XNN_FLAG_SLINKY_NO_CHECKS, and XNN_FLAG_SLINKY_NO_SCHEDULE.
/// @param runtime_out - pointer to the variable that will be initialized with a handle to the Runtime object upon
///                      successful return. Once constructed, the Runtime object is independent of the Subgraph object
///                      used to create it.
enum xnn_status xnn_create_runtime_with_scheduler(
  xnn_subgraph_t subgraph,
  xnn_weights_cache_t weights_cache,
  xnn_scheduler_t scheduler,
  uint32_t flags,
  xnn_runtime_t* runtime_out);

typedef struct xnn_threadpool* xnn_threadpool_t;

/// Create a Threadpool object from a Scheduler.
///
/// @param scheduler - Scheduler to implement parallel task execution.
/// @param threadpool_out - The created Threadpool object.
enum xnn_status xnn_create_threadpool(
  xnn_scheduler_t scheduler,
  xnn_threadpool_t* threadpool_out);

/// Destroy a Threadpool object
///
/// @param subgraph - the Threadpool object to destroy.
enum xnn_status xnn_delete_threadpool(xnn_threadpool_t threadpool);

/// Create a Runtime object from a subgraph with Slinky enabled.
///
/// @param subgraph - a Subgraph object with all Values and Nodes that would be handled by the runtime. No Values or
///                   Nodes can be added to the runtime once it is constructed.
/// @param weights_cache - a cache for packed weights. The runtime will look up and reuse packed weights in this cache,
///                        this will reduce memory allocated for packed weights.
/// @param workspace - a workspace to hold internal tensors. The runtime will allocate space used for internal tensors
///                    and track them using workspace. Workspace can be shared and reused across different runtimes. If
///                    workspace is NULL, there will be no sharing: each runtime has its own workspace.
/// @param threadpool - Threadpool object to to implement parallel operations.
/// @param flags - binary features of the runtime. The only currently supported values are
///                XNN_FLAG_HINT_SPARSE_INFERENCE, XNN_FLAG_HINT_FP16_INFERENCE, XNN_FLAG_FORCE_FP16_INFERENCE,
///                XNN_FLAG_SLINKY_STATIC_BOUNDS, XNN_FLAG_SLINKY_NO_CHECKS, and XNN_FLAG_SLINKY_NO_SCHEDULE.
/// @param runtime_out - pointer to the variable that will be initialized with a handle to the Runtime object upon
///                      successful return. Once constructed, the Runtime object is independent of the Subgraph object
///                      used to create it.
enum xnn_status xnn_create_runtime_with_threadpool(
  xnn_subgraph_t subgraph,
  xnn_weights_cache_t weights_cache,
  xnn_threadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_INCLUDE_EXPERIMENTAL_H_
