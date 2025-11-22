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
#include "src/operators/fingerprint_id.h"

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

/// Deprecated.
#define XNN_FLAG_SLINKY_NO_CHECKS 0x00040000

#define XNN_FLAG_RUNTIME_OWNS_THREADPOOL 0x00080000

typedef struct xnn_threadpool* xnn_threadpool_t;

/// An abstract interface of a parallel task scheduler.
struct xnn_scheduler_v2 {
  /// Get the number of tasks that can be executed concurrently.
  int (*num_threads)(void* scheduler_context);

  /// Schedule `task` to be called, with `context` as its argument.
  void (*schedule)(void* scheduler_context, void* context, void (*task)(void* context));
};

/// Create a Threadpool object from a Scheduler v2.
///
/// @param scheduler - Scheduler to implement parallel task execution.
/// @param scheduler_context - Context to pass to scheduler methods.
/// @param flags - Binary feature flags of the threadpool. No flags are currently supported.
/// @param threadpool_out - The created Threadpool object.
enum xnn_status xnn_create_threadpool_v2(
  struct xnn_scheduler_v2 scheduler,
  void* scheduler_context,
  uint32_t flags,
  xnn_threadpool_t* threadpool_out);

/// Destroy a Threadpool object
///
/// @param threadpool - the Threadpool object to destroy.
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
///                XNN_FLAG_SLINKY_STATIC_BOUNDS, and XNN_FLAG_SLINKY_NO_SCHEDULE.
/// @param runtime_out - pointer to the variable that will be initialized with a handle to the Runtime object upon
///                      successful return. Once constructed, the Runtime object is independent of the Subgraph object
///                      used to create it.
enum xnn_status xnn_create_runtime_with_threadpool(
  xnn_subgraph_t subgraph,
  xnn_weights_cache_t weights_cache,
  xnn_threadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out);

/// Replace the threadpool used by a Runtime with a new Threadpool object.
///
/// The new thread pool should have the same number of threads as the current threadpool.
///
/// @param runtime - A Runtime object to update the threadpool of.
/// @param threadpool - Threadpool object to to implement parallel operations.
enum xnn_status xnn_update_runtime_with_threadpool(
  xnn_runtime_t runtime,
  xnn_threadpool_t threadpool);


typedef struct xnn_fingerprint* xnn_fingerprint_t;

struct xnn_fingerprint {
  uint32_t id;
  uint32_t value;
};

/// Check whether the given configuration matches one that is currently in use.
///
/// @returns `xnn_status_success` if the configuration matches.
enum xnn_status xnn_check_fingerprint(struct xnn_fingerprint fingerprint);

/// Return the fingerprint corresponding to the given id or NULL if it wasn't
/// set.
const struct xnn_fingerprint* xnn_get_fingerprint(uint32_t id);

/// Set the given fingerprint.
void xnn_set_fingerprint(struct xnn_fingerprint fingerprint);

/// Clear all fingerprints that were computed until now.
void xnn_clear_fingerprints();

enum xnn_status xnn_fingerprint_fully_connected_nc_f16();
enum xnn_status xnn_fingerprint_fully_connected_nc_f32_f16();
enum xnn_status xnn_fingerprint_fully_connected_nc_bf16_f32();
enum xnn_status xnn_fingerprint_fully_connected_nc_f32();
enum xnn_status xnn_fingerprint_fully_connected_nc_f32_f32_f32_nr2();
enum xnn_status xnn_fingerprint_fully_connected_nc_f32_qc4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_f32_qc8w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f16_qc4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f16_qb4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f16_qb4w_f16_scales();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f32_qc2w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f32_qc4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f32_qb4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f16_qc8w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f32_qc8w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qs8();
enum xnn_status xnn_fingerprint_fully_connected_nc_qs8_qc4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qs8_qc8w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qu8();
enum xnn_status xnn_fingerprint_fully_connected_nc_qp8_f32_qc4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qp8_f32_qc8w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qp8_f32_qb4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qp8_f32_qb4w_f16_scales();
enum xnn_status xnn_fingerprint_fully_connected_nc_pf32();
enum xnn_status xnn_fingerprint_fully_connected_nc_qdu8_f16_qc8w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qdu8_f32_qc8w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qdu8_f32_qc4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qdu8_f32_qb4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qdu8_f16_qc4w();
enum xnn_status xnn_fingerprint_fully_connected_nc_pf16();
enum xnn_status xnn_fingerprint_fully_connected_nc_pqs8_qc8w();
enum xnn_status xnn_fingerprint_fully_connected_nc_qd8_f32_qb4w_f16_scales();
enum xnn_status xnn_fingerprint_fully_connected_nc_qdu8_f32_qb4w_f16_scales();
enum xnn_status xnn_fingerprint_convolution2d_nchw(enum xnn_fingerprint_id fingerprint_id);
enum xnn_status xnn_fingerprint_convolution2d_nhwc(enum xnn_fingerprint_id fingerprint_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_INCLUDE_EXPERIMENTAL_H_
