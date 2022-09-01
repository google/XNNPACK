// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/allocator.h>

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_value_usage {
  // The index (to xnn_subgraph_t->nodes) of the first xnn_node that uses this xnn_value.
  uint32_t first_node;
  // The index of the last xnn_node that uses this xnn_value.
  uint32_t last_node;
  // Note that 'tensor_size' includes the padding of XNN_EXTRA_BYTES.
  size_t tensor_size;
  // The memory offset of this xnn_value from the beginning of a memory buffer.
  size_t alloc_offset;
  // When an operation is performed in place, its output value overlaps completely with the input value, i.e. their
  // alloc_offset values are the same. This field is the input value who's alloc_offset this value usage can reuse.
  // This field is an index into tracker->usage denoting the input value.
  uint32_t reuse_value_id;
};

// Track the memory allocation in a memory arena for a subgraph.
struct xnn_value_allocation_tracker {
  xnn_subgraph_t subgraph;
  size_t mem_arena_size;
  // Representing the lifecycle of xnn_values in the 'subgraph', and the array size is 'subgraph->num_values'.
  struct xnn_value_usage* usage;
  // The range of value ids (i.e. the index to subgraph->values) whose memory might need to be allocated.
  size_t min_value_id;
  size_t max_value_id;
};

// Initialize the memory allocation tracker for xnn_values.
XNN_INTERNAL void xnn_init_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker,
                                                    const xnn_subgraph_t subgraph);

inline static void xnn_release_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker) {
  xnn_release_memory(tracker->usage);
}

// Add a to-be-allocated xnn_value (referred by 'value_id') of size 'tensor_size' to the allocation tracker.
// Note: this function assumes 'value_id's added in increasing order for simplicity as it's called inside a loop
// iterating over 'subgraph->values'.
XNN_INTERNAL void xnn_add_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker,
                                                   uint32_t value_id, size_t tensor_size);

// Mark a node as being able to be performed in place, the node's output (value_id) can share the same memory as
// reuse_value_id. The usage record of reuse_value_id needs to be expanded to include all consumers of
// value_id, this is indicated by new_last_node.
XNN_INTERNAL void xnn_mark_in_place_node(
  struct xnn_value_allocation_tracker* tracker,
  uint32_t value_id,
  uint32_t reuse_value_id,
  uint32_t new_last_node);

// Plan the exact the memory allocation for intermediate tensors according to the xnn_value allocation tracker.
XNN_INTERNAL void xnn_plan_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker);

#ifdef __cplusplus
}  // extern "C"
#endif
