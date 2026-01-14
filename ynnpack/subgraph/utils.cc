// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

std::unique_ptr<ynn_subgraph> clone_subgraph_subset(
    const ynn_subgraph& subgraph, uint32_t input_id, uint32_t output_id,
    uint32_t& cloned_input_id, uint32_t& cloned_output_id) {
  assert(subgraph.is_valid_value(input_id));
  assert(subgraph.is_valid_value(output_id));
  assert(input_id != output_id);

  // Identify relevant nodes and values via backward traversal from output_id.
  std::unordered_set<uint32_t> relevant_values;
  std::unordered_set<const ynn_node*> relevant_nodes;
  std::vector<uint32_t> values_to_traverse;

  auto add_value_for_traversal = [&](uint32_t id) {
    if (relevant_values.find(id) == relevant_values.end()) {
      relevant_values.insert(id);
      values_to_traverse.push_back(id);
    }
  };
  add_value_for_traversal(output_id);

  size_t head = 0;
  while (head < values_to_traverse.size()) {
    uint32_t curr_id = values_to_traverse[head++];

    const ynn_value& val = subgraph.value(curr_id);
    if (val.scale_id != YNN_INVALID_VALUE_ID) {
      add_value_for_traversal(val.scale_id);
    }
    if (val.zero_point_id != YNN_INVALID_VALUE_ID) {
      add_value_for_traversal(val.zero_point_id);
    }

    if (curr_id == input_id) {
      continue;
    }

    const ynn_node* producer = subgraph.get_producer(curr_id);
    if (producer) {
      relevant_nodes.insert(producer);
      for (uint32_t input_id : producer->inputs) {
        if (input_id != YNN_INVALID_VALUE_ID) {
          add_value_for_traversal(input_id);
        }
      }
    }
  }

  if (relevant_values.find(input_id) == relevant_values.end()) {
    // The input is not connected to the output, so we can't clone it.
    return nullptr;
  }

  // Create new subgraph.
  auto new_subgraph = std::make_unique<ynn_subgraph>(
      subgraph.external_value_ids, subgraph.flags);
  std::unordered_map<uint32_t, uint32_t> old_to_new_id;

  uint32_t next_free_id = 0;
  uint32_t reserved_count = subgraph.external_value_ids;

  // Helper to clone values. Returns the ID of the new value.
  auto clone_or_get_new_value = [&](uint32_t old_id) -> uint32_t {
    if (old_id == YNN_INVALID_VALUE_ID) return YNN_INVALID_VALUE_ID;
    if (old_to_new_id.count(old_id)) return old_to_new_id[old_id];

    // Create new value.
    // Values up to `reserved_count` are pre-allocated for external inputs.
    // `new_subgraph->value(id)` provides a reference to these pre-allocated
    // slots, which we can then populate. For values beyond `reserved_count`,
    // we use `new_internal_value()` to dynamically allocate a new internal
    // value.
    ynn_value* new_val_ptr = nullptr;
    if (next_free_id < reserved_count) {
      new_val_ptr = &new_subgraph->value(next_free_id++);
    } else {
      new_val_ptr = &new_subgraph->new_internal_value();
    }
    ynn_value& new_val = *new_val_ptr;
    const ynn_value& old_val = subgraph.value(old_id);

    // Copy all fields except quantization parameters. We'll handle those
    // separately.
    new_val.type = old_val.type;
    new_val.flags = old_val.flags;
    new_val.data = old_val.data;
    new_val.extents = old_val.extents;

    // Adjust flags.
    if (old_id == input_id) {
      new_val.flags |= YNN_VALUE_FLAG_EXTERNAL_INPUT;
      new_val.flags &= ~YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    }
    if (old_id == output_id) {
      new_val.flags |= YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    } else if ((old_val.flags & YNN_VALUE_FLAG_EXTERNAL_OUTPUT) != 0) {
      // If old_id was an external output in the original subgraph but is not
      // the designated output_id for this cloned subset, we clear the
      // EXTERNAL_OUTPUT flag. This ensures the cloned subgraph is treated as
      // having only `output_id` as its external output.
      new_val.flags &= ~YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    }

    const ynn_node* producer = subgraph.get_producer(old_id);
    bool produced_in_subset =
        producer && relevant_nodes.find(producer) != relevant_nodes.end();

    if (!produced_in_subset && !new_val.is_static()) {
      new_val.flags |= YNN_VALUE_FLAG_EXTERNAL_INPUT;
    }

    old_to_new_id[old_id] = new_val.id;
    return new_val.id;
  };

  // Clone values in topological order.
  std::vector<uint32_t> sorted_values(relevant_values.begin(),
                                      relevant_values.end());
  std::sort(sorted_values.begin(), sorted_values.end());

  for (uint32_t old_id : sorted_values) {
    clone_or_get_new_value(old_id);
  }

  // Now fix up quantization links.
  for (uint32_t old_id : sorted_values) {
    uint32_t new_id = old_to_new_id[old_id];
    const ynn_value& old_val = subgraph.value(old_id);
    ynn_value& new_val = new_subgraph->value(new_id);
    if (old_val.scale_id != YNN_INVALID_VALUE_ID) {
      new_val.scale_id = clone_or_get_new_value(old_val.scale_id);
    }
    if (old_val.zero_point_id != YNN_INVALID_VALUE_ID) {
      new_val.zero_point_id = clone_or_get_new_value(old_val.zero_point_id);
    }
  }

  // Clone nodes in topological order.
  for (const auto& node : subgraph.nodes) {
    if (relevant_nodes.find(&node) != relevant_nodes.end()) {
      ynn_node new_node = node;
      new_node.inputs.clear();
      new_node.outputs.clear();

      for (uint32_t old_in : node.inputs) {
        new_node.inputs.push_back(clone_or_get_new_value(old_in));
      }
      for (uint32_t old_out : node.outputs) {
        new_node.outputs.push_back(clone_or_get_new_value(old_out));
      }

      new_subgraph->add_node(std::move(new_node));
    }
  }

  cloned_input_id = clone_or_get_new_value(input_id);
  cloned_output_id = clone_or_get_new_value(output_id);
  return new_subgraph;
}

template <typename T>
bool is_broadcast_op(const ynn_node& node) {
  const T* op = std::get_if<T>(&node.op);
  return op && op->axes.any();
}

bool allow_in_place(uint32_t input_id, uint32_t output_id,
                    const ynn_subgraph& subgraph) {
  if (input_id == YNN_INVALID_VALUE_ID) return false;

  const ynn_value& a = subgraph.value(input_id);
  const ynn_value& x = subgraph.value(output_id);

  if (x.rank() != a.rank()) {
    return false;
  }

  if (x.is_external_output()) {
    // If the output is an external output, we can't compute it in-place because
    // it might alias an input to the subgraph.
    // TODO(dsharlet): I think we could relax this constraint somewhat.
    return false;
  }

  if (type_size_bytes(a.type) != type_size_bytes(x.type) ||
      type_element_count(a.type) != type_element_count(x.type)) {
    // The types are not the same size, we can't compute in place.
    return false;
  }

  for (size_t d = 0; d < x.rank(); ++d) {
    if (!a.extents[d].defined() && x.extents[d].defined()) {
      // The input is broadcasted (and the output is not), don't allow computing
      // in place.
      return false;
    }
  }

  const ynn_node* producer = subgraph.get_producer(input_id);
  if (!producer) {
    // This input is not produced in the pipeline, we can't overwrite the
    // input (and slinky wouldn't let us anyways).
    return false;
  }

  if (is_broadcast_op<ynn_node::broadcast>(*producer) ||
      is_broadcast_op<ynn_node::broadcast_like>(*producer)) {
    // We can't compute in place with a broadcast input.
    return false;
  }

  return true;
}

}  // namespace ynn
