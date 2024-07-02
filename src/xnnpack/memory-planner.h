// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_usage_record {
  // The index (to xnn_runtime_t->opdata) of the first xnn_node that uses this xnn_value.
  uint32_t first_node;
  // The index of the last xnn_node that uses this xnn_value.
  uint32_t last_node;
  // Note that 'tensor_size' includes the padding of XNN_EXTRA_BYTES.
  size_t tensor_size;
  // The memory offset of this xnn_value from the beginning of a memory buffer.
  size_t alloc_offset;
  // If an operation is performed in place, the alloc_offset of the output tensor is the same as the alloc_offset of the
  // input tensor. The id of the input tensor is recorded in this field. This is XNN_INVALID_VALUE_ID if it does not
  // reuse any tensor.
  uint32_t reuse_value_id;
  // This usage record is not tied to an actual value, but a temporary associated with an opdata, like a dynamic fully
  // connected operation. We need the opdata's id to lookup and intialize opdata's pointers.
  uint32_t opdata_id;
};

// Track the memory allocation in a memory arena for a runtime.
struct xnn_value_allocation_tracker {
  size_t mem_arena_size;
  // Representing the lifecycle of xnn_values in the 'runtime', and the array size is 'runtime->num_values +
  // runtime->num_ops'.
  struct xnn_usage_record* usage;
  // The range of value ids (i.e. the index to runtime->values) whose memory might need to be allocated.
  size_t min_value_id;
  size_t max_value_id;
};

// Initialize the memory allocation tracker for xnn_values.
XNN_INTERNAL void xnn_init_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker,
                                                    const struct xnn_runtime* runtime);

inline static void xnn_release_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker) {
  xnn_release_memory(tracker->usage);
}

// Add a to-be-allocated xnn_value (referred by 'value_id') of size 'tensor_size' to the allocation tracker.
// Note: this function assumes 'value_id's added in increasing order for simplicity as it's called inside a loop
// iterating over 'runtime->values'.
XNN_INTERNAL void xnn_add_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker,
                                                   uint32_t value_id, size_t tensor_size);

// Add a value to represent operator workspace. This is a temporary buffer that is only used during the invocation of
// operator.
XNN_INTERNAL void xnn_add_operator_workspace_allocation_tracker(
  struct xnn_value_allocation_tracker* tracker,
  uint32_t operator_workspace_value_id,
  size_t tensor_size,
  uint32_t opdata_id);

// Mark value_id as reusing the memory that is allocated to another reuse_value_id. No memory is then
// allocated to value_id. The usage record of reuse_value_id needs to be expanded to include al consumers of value_id,
// indicated by new_last_node.
XNN_INTERNAL void xnn_mark_tensor_as_reuse(
  struct xnn_value_allocation_tracker* tracker,
  uint32_t value_id,
  uint32_t reuse_value_id,
  uint32_t new_last_node);

// Plan the exact the memory allocation for intermediate tensors according to the xnn_value allocation tracker.
XNN_INTERNAL void xnn_plan_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker);

#ifdef __cplusplus
}  // extern "C"
#endif
