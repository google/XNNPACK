// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/memory-planner.h>
#include <xnnpack/subgraph.h>

// Check if two xnn_value's lifecycles overlap.
inline static bool value_lifecycle_overlap(const struct xnn_usage_record* a, const struct xnn_usage_record* b) {
  assert(a->last_node >= a->first_node);
  assert(b->last_node >= b->first_node);
  if (a->first_node < b->first_node) {
    return a->last_node >= b->first_node;
  } else {
    return b->last_node >= a->first_node;
  }
}

// Use this comparison function to sort xnn_usage_record according to the
// tensor_size in decreasing order.
static inline int cmp_value_usage_tensor_size(const void* a, const void* b) {
  const size_t tensor_size_a = (*(struct xnn_usage_record *const*)a)->tensor_size;
  const size_t tensor_size_b = (*(struct xnn_usage_record *const*)b)->tensor_size;
  return (tensor_size_b > tensor_size_a) - (tensor_size_b < tensor_size_a);
}

static void populate_value_lifecycle(const struct xnn_runtime* runtime, struct xnn_usage_record* usage) {
  assert(runtime != NULL);
  if (runtime->num_ops == 0) {
    return;
  }
  // As we initialized first/last_node in each xnn_usage_record to 0 as in 'xnn_init_value_mem_allocation_tracker',
  // we start with the second node to tell whether first/last_node have been set or not, and check the first node last.
  for (uint32_t nid = 1; nid < runtime->num_ops; ++nid) {
    const struct xnn_operator_data* opdata = runtime->opdata + nid;
    for (uint32_t i = 0; i < opdata->num_inputs; ++i) {
      if (opdata->inputs[i] == XNN_INVALID_VALUE_ID) {
        continue;  // Optimized away.
      }
      if (usage[opdata->inputs[i]].first_node == 0) {
        usage[opdata->inputs[i]].first_node = nid;
      }
      usage[opdata->inputs[i]].last_node = nid;
    }
    for (uint32_t i = 0; i < opdata->num_outputs; ++i) {
      if (opdata->outputs[i] == XNN_INVALID_VALUE_ID) {
        continue;  // Optimized away.
      }
      if (usage[opdata->outputs[i]].first_node == 0) {
        usage[opdata->outputs[i]].first_node = nid;
      }
      usage[opdata->outputs[i]].last_node = nid;
    }
  }
  const struct xnn_operator_data* first_node = runtime->opdata;
  for (uint32_t i = 0; i < first_node->num_inputs; ++i) {
    if (first_node->inputs[i] == XNN_INVALID_VALUE_ID) {
      continue;  // Optimized away.
    }
    usage[first_node->inputs[i]].first_node = 0;
  }
  for (uint32_t i = 0; i < first_node->num_outputs; ++i) {
    if (first_node->outputs[i] == XNN_INVALID_VALUE_ID) {
      continue;  // Optimized away.
    }
    usage[first_node->outputs[i]].first_node = 0;
  }
  // Separate loop over all values to make sure we have usage records properly initialized with invalid reuse_value_id.
  // Some usage records are not associated with any nodes, and they will not be visited by the loops over nodes above.
  for (uint32_t i = 0; i < runtime->num_values + runtime->num_ops; i++) {
    usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
    usage[i].alloc_offset = SIZE_MAX;
    usage[i].opdata_id = XNN_INVALID_NODE_ID;
  }
}

// Represent a memory block [start, end)
struct memory_block {
  size_t start;
  size_t end;
};

// Use this comparison function to sort memory_block according to the 'start'
// in increasing order.
static inline int cmp_memory_block(const void* a, const void* b) {
  const size_t start_a = ((const struct memory_block*)a)->start;
  const size_t start_b = ((const struct memory_block*)b)->start;
  return (start_a > start_b) - (start_a < start_b);
}

// Given the current live memory blocks, return the offset in a memory arena for a to-be-allocated value of size
// 'to_alloc_size'.
static size_t find_value_alloc_offset(struct memory_block* live_mem_blocks,
                                      size_t num_mem_blocks,
                                      size_t to_alloc_size) {
  if (num_mem_blocks == 0) {
    return 0;
  }

  if (num_mem_blocks == 1) {
    return live_mem_blocks[0].end;
  }

  // Sort memory blocks according to 'start' in increasing order.
  qsort(live_mem_blocks, num_mem_blocks, sizeof(struct memory_block), cmp_memory_block);

  // Coalesce overlapping or immediate adjacent memory blocks to form a list of non-overlapping memory blocks in order
  // to find the smallest gap.
  size_t num_coalesced_mem_blocks = 1;
  for (size_t i = 1; i < num_mem_blocks; ++i) {
    const size_t current_coalesced_end =
        live_mem_blocks[num_coalesced_mem_blocks - 1].end;
    if (live_mem_blocks[i].start > current_coalesced_end) {
      assert(num_coalesced_mem_blocks <= i);
      live_mem_blocks[num_coalesced_mem_blocks] = live_mem_blocks[i];
      num_coalesced_mem_blocks++;
      continue;
    }
    if (live_mem_blocks[i].end > current_coalesced_end) {
      live_mem_blocks[num_coalesced_mem_blocks - 1].end = live_mem_blocks[i].end;
    }
  }

  size_t smallest_gap_size = SIZE_MAX;
  // The first index to live_mem_blocks that the 'to_alloc_size' should be allocated after.
  size_t smallest_gap_index = num_coalesced_mem_blocks - 1;
  for (size_t i = 0; i < num_coalesced_mem_blocks - 1; ++i) {
    assert(live_mem_blocks[i + 1].start > live_mem_blocks[i].end);
    const size_t gap = live_mem_blocks[i + 1].start - live_mem_blocks[i].end;
    if (gap >= to_alloc_size && gap < smallest_gap_size) {
      smallest_gap_index = i;
      smallest_gap_size = gap;
    }
  }
  return live_mem_blocks[smallest_gap_index].end;
}

void xnn_init_value_allocation_tracker(
  struct xnn_value_allocation_tracker* tracker,
  const struct xnn_runtime* runtime)
{
  tracker->mem_arena_size = 0;
  tracker->usage = xnn_allocate_zero_memory(sizeof(struct xnn_usage_record) * (runtime->num_values + runtime->num_ops));
#if XNN_ENABLE_MEMOPT
  populate_value_lifecycle(runtime, tracker->usage);
#endif
  tracker->min_value_id = XNN_INVALID_VALUE_ID;
  tracker->max_value_id = XNN_INVALID_VALUE_ID;
}

void xnn_mark_tensor_as_reuse(struct xnn_value_allocation_tracker* tracker,
                              uint32_t value_id,
                              uint32_t reuse_value_id,
                              uint32_t new_last_node) {
  // Set tensor_size to 0 so memory planner will not try to find memory for these tensors.
  tracker->usage[value_id].tensor_size = 0;
  tracker->usage[value_id].reuse_value_id = reuse_value_id;
  // The reused tensor has an expanded live-range.
  tracker->usage[reuse_value_id].last_node = new_last_node;
}

void xnn_add_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker,
                                      uint32_t value_id,
                                      size_t tensor_size) {
  tracker->usage[value_id].tensor_size = tensor_size;
  if (tracker->min_value_id == XNN_INVALID_VALUE_ID) {
    tracker->min_value_id = value_id;
  } else {
    // Note that values are expected to be added in increasing order.
    assert(value_id > tracker->min_value_id);
    assert(value_id > tracker->max_value_id);
  }

  tracker->max_value_id = value_id;
}

void xnn_add_operator_workspace_allocation_tracker(
  struct xnn_value_allocation_tracker* tracker,
  uint32_t operator_workspace_value_id,
  size_t tensor_size,
  uint32_t opdata_id)
{
  tracker->usage[operator_workspace_value_id].tensor_size = tensor_size;
  if (tracker->min_value_id == XNN_INVALID_VALUE_ID) {
    tracker->min_value_id = operator_workspace_value_id;
  } else {
    // Note that values are expected to be added in increasing order.
    assert(operator_workspace_value_id > tracker->min_value_id);
    assert(operator_workspace_value_id > tracker->max_value_id);
  }
  tracker->max_value_id = operator_workspace_value_id;
  tracker->usage[operator_workspace_value_id].first_node = opdata_id;
  tracker->usage[operator_workspace_value_id].last_node = opdata_id;
  tracker->usage[operator_workspace_value_id].opdata_id = opdata_id;
}

void xnn_plan_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker) {
#if XNN_ENABLE_MEMOPT
  if (tracker->min_value_id == XNN_INVALID_VALUE_ID) {
    assert(tracker->max_value_id == XNN_INVALID_VALUE_ID);
    return;
  }

  const uint32_t num_values = tracker->max_value_id - tracker->min_value_id + 1;
  struct xnn_usage_record** sorted_usage = xnn_allocate_zero_memory(sizeof(struct xnn_usage_record*) * num_values);
  size_t num_values_to_alloc = 0;
  for (size_t i = tracker->min_value_id; i <= tracker->max_value_id; ++i) {
    struct xnn_usage_record* info = tracker->usage + i;
    if (info->tensor_size != 0) {
      sorted_usage[num_values_to_alloc++] = info;
    }
  }
  qsort(sorted_usage, num_values_to_alloc, sizeof(struct xnn_usage_record*), cmp_value_usage_tensor_size);

  // Start the allocation planning process.
  struct memory_block* current_live_mem_blocks = xnn_allocate_zero_memory(
      sizeof(struct memory_block) * num_values_to_alloc);
  size_t mem_arena_size = 0;
  for (size_t i = 0; i < num_values_to_alloc; ++i) {
    size_t num_live_mem_blocks = 0;
    struct xnn_usage_record* current = sorted_usage[i];
    for (size_t j = 0; j < i; ++j) {
      const struct xnn_usage_record* allocated = sorted_usage[j];
      if (value_lifecycle_overlap(current, allocated)) {
        current_live_mem_blocks[num_live_mem_blocks++] = (struct memory_block){
            .start = allocated->alloc_offset,
            .end = allocated->alloc_offset + allocated->tensor_size,
        };
      }
    }
    current->alloc_offset = find_value_alloc_offset(current_live_mem_blocks, num_live_mem_blocks, current->tensor_size);
    if (mem_arena_size < current->alloc_offset + current->tensor_size) {
      mem_arena_size = current->alloc_offset + current->tensor_size;
    }
  }

  // Walk through all tensors that are reusing memory, and update their usage records.
  for (size_t i = tracker->min_value_id; i <= tracker->max_value_id; ++i) {
    struct xnn_usage_record* usage = &tracker->usage[i];
    uint32_t reuse_id = usage->reuse_value_id;
    if (reuse_id == XNN_INVALID_VALUE_ID) {
      continue;
    }
    assert(tracker->usage[reuse_id].alloc_offset != SIZE_MAX);
    usage->alloc_offset = tracker->usage[reuse_id].alloc_offset;
  }

  tracker->mem_arena_size = mem_arena_size;
  xnn_release_memory(sorted_usage);
  xnn_release_memory(current_live_mem_blocks);
#else
  tracker->mem_arena_size = 0;
  for (uint32_t i = tracker->min_value_id; i <= tracker->max_value_id; ++i) {
    if (tracker->usage[i].tensor_size > 0) {
      tracker->usage[i].alloc_offset = tracker->mem_arena_size;
      tracker->mem_arena_size += tracker->usage[i].tensor_size;
    }
  }
#endif
}
