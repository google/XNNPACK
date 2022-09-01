// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <xnnpack/memory-planner.h>
#include <xnnpack/subgraph.h>

// Check if two xnn_value's lifecycles overlap.
inline static bool value_lifecycle_overlap(const struct xnn_value_usage* a, const struct xnn_value_usage* b) {
  assert(a->last_node >= a->first_node);
  assert(b->last_node >= b->first_node);
  if (a->first_node < b->first_node) {
    return a->last_node >= b->first_node;
  } else {
    return b->last_node >= a->first_node;
  }
}

// Use this comparison function to sort xnn_value_usage according to the
// tensor_size in decreasing order.
static inline int cmp_value_usage_tensor_size(const void* a, const void* b) {
  struct xnn_value_usage * usage_a = (*(struct xnn_value_usage *const*)a);
  struct xnn_value_usage * usage_b = (*(struct xnn_value_usage *const*)b);
  const size_t tensor_size_a = usage_a->tensor_size;
  const size_t tensor_size_b = usage_b->tensor_size;
  // We need this ordering to handle in-place operations where the tensors overlap. Given 2 tensors of the same size, if
  // they compare equal, the ordering is unspecified by qsort, but we need them to have a defined ordering based on
  // topological sort, i.e. the earlier value should be processed first, in order to have a valid alloc_offset.
  if (tensor_size_a == tensor_size_b) {
    if (usage_a->first_node < usage_b->first_node) {
      return -1;
    } else if (usage_a->first_node > usage_b->first_node) {
      return 1;
    } else {
      // If the size, first_node, and last_node are equal, this really is the same value.
      return (usage_a->last_node - usage_b->last_node);
    }
  }
  return (tensor_size_b > tensor_size_a) - (tensor_size_b < tensor_size_a);
}

static void populate_value_lifecycle(const xnn_subgraph_t subgraph, struct xnn_value_usage* usage) {
  assert(subgraph != NULL);
  if (subgraph->num_nodes == 0) {
    return;
  }
  // As we initialized first/last_node in each xnn_value_usage to 0 as in 'xnn_init_value_mem_allocation_tracker',
  // we start with the second node to tell whether first/last_node have been set or not, and check the first node last.
  for (uint32_t nid = 1; nid < subgraph->num_nodes; ++nid) {
    const struct xnn_node* node = subgraph->nodes + nid;
    for (uint32_t i = 0; i < node->num_inputs; ++i) {
      if (usage[node->inputs[i]].first_node == 0) {
        usage[node->inputs[i]].first_node = nid;
      }
      usage[node->inputs[i]].last_node = nid;
    }
    for (uint32_t i = 0; i < node->num_outputs; ++i) {
      if (usage[node->outputs[i]].first_node == 0) {
        usage[node->outputs[i]].first_node = nid;
      }
      usage[node->outputs[i]].last_node = nid;
    }
  }
  const struct xnn_node* first_node = subgraph->nodes;
  for (uint32_t i = 0; i < first_node->num_inputs; ++i) {
    usage[first_node->inputs[i]].first_node = 0;
  }
  for (uint32_t i = 0; i < first_node->num_outputs; ++i) {
    usage[first_node->outputs[i]].first_node = 0;
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

void xnn_init_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker, const xnn_subgraph_t subgraph) {
  tracker->subgraph = subgraph;
  tracker->mem_arena_size = 0;
  tracker->usage = xnn_allocate_zero_memory(sizeof(struct xnn_value_usage) * subgraph->num_values);
#if XNN_ENABLE_MEMOPT
  populate_value_lifecycle(tracker->subgraph, tracker->usage);
#endif
  tracker->min_value_id = XNN_INVALID_VALUE_ID;
  tracker->max_value_id = XNN_INVALID_VALUE_ID;
}

void xnn_mark_in_place_node(struct xnn_value_allocation_tracker* tracker,
                              uint32_t value_id,
                              uint32_t reuse_value_id,
                              uint32_t new_last_node) {
  tracker->usage[value_id].reuse_value_id = reuse_value_id;
  tracker->usage[value_id].last_node = new_last_node;
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
  tracker->usage[value_id].alloc_offset = SIZE_MAX;
  tracker->usage[value_id].reuse_value_id = XNN_INVALID_VALUE_ID;
}

void xnn_plan_value_allocation_tracker(struct xnn_value_allocation_tracker* tracker) {
#if XNN_ENABLE_MEMOPT
  if (tracker->min_value_id == XNN_INVALID_VALUE_ID) {
    assert(tracker->max_value_id == XNN_INVALID_VALUE_ID);
    return;
  }

  const uint32_t num_values = tracker->max_value_id - tracker->min_value_id + 1;
  struct xnn_value_usage** sorted_usage = xnn_allocate_zero_memory(sizeof(struct xnn_value_usage*) * num_values);
  size_t num_values_to_alloc = 0;
  for (size_t i = tracker->min_value_id; i <= tracker->max_value_id; ++i) {
    struct xnn_value_usage* info = tracker->usage + i;
    if (info->tensor_size != 0) {
      sorted_usage[num_values_to_alloc++] = info;
    }
  }
  qsort(sorted_usage, num_values_to_alloc, sizeof(struct xnn_value_usage*), cmp_value_usage_tensor_size);

  // Start the allocation planning process.
  struct memory_block* current_live_mem_blocks = xnn_allocate_zero_memory(
      sizeof(struct memory_block) * num_values_to_alloc);
  size_t mem_arena_size = 0;
  for (size_t i = 0; i < num_values_to_alloc; ++i) {
    size_t num_live_mem_blocks = 0;
    struct xnn_value_usage* current = sorted_usage[i];
    if (current->reuse_value_id != XNN_INVALID_VALUE_ID) {
      assert(tracker->usage[current->reuse_value_id].alloc_offset != SIZE_MAX);
      current->alloc_offset = tracker->usage[current->reuse_value_id].alloc_offset;
      continue;
    }
    for (size_t j = 0; j < i; ++j) {
      const struct xnn_value_usage* allocated = sorted_usage[j];
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
