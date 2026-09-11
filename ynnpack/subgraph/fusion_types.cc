// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/fusion_types.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

subgraph_analysis::subgraph_analysis(ynn_subgraph& subgraph) : is_valid(true) {
  for (ynn_node& node : subgraph.nodes) {
    add_node(node);
  }
}

void subgraph_analysis::remove_consumer(uint32_t value_id, ynn_node* node) {
  auto it = consumers.find(value_id);
  if (it != consumers.end()) {
    auto& vec = it->second;
    auto node_it = std::find(vec.begin(), vec.end(), node);
    if (node_it != vec.end()) {
      vec.erase(node_it);
    }
    if (vec.empty()) {
      consumers.erase(it);
    }
  }
}

void subgraph_analysis::add_node(ynn_node& node) {
  if (!node.is_valid()) return;
  for (uint32_t input : node.inputs) {
    if (input != YNN_INVALID_VALUE_ID) {
      consumers[input].push_back(&node);
    }
  }
  for (uint32_t output : node.outputs) {
    if (output != YNN_INVALID_VALUE_ID) {
      assert(producers.find(output) == producers.end());
      producers[output] = &node;
    }
  }
}

void subgraph_analysis::remove_node(ynn_node& node) {
  if (!node.is_valid()) return;
  for (uint32_t input : node.inputs) {
    if (input != YNN_INVALID_VALUE_ID) {
      remove_consumer(input, &node);
    }
  }
  for (uint32_t output : node.outputs) {
    if (output != YNN_INVALID_VALUE_ID) {
      auto it = producers.find(output);
      if (it != producers.end() && it->second == &node) {
        producers.erase(it);
      }
    }
  }
}

void subgraph_analysis::invalidate_node(ynn_node& node) {
  remove_node(node);
  node.invalidate();
}

void subgraph_analysis::replace_input_id(ynn_node& node, uint32_t& input_id,
                                         uint32_t new_id) {
  uint32_t old_id = input_id;
  if (old_id == new_id) return;
  if (old_id != YNN_INVALID_VALUE_ID) {
    remove_consumer(old_id, &node);
  }
  input_id = new_id;
  if (new_id != YNN_INVALID_VALUE_ID) {
    consumers[new_id].push_back(&node);
  }
}

void subgraph_analysis::replace_input(ynn_node& node, size_t index,
                                      uint32_t new_id) {
  assert(index < node.inputs.size());
  replace_input_id(node, node.inputs[index], new_id);
}

void subgraph_analysis::replace_all_uses(uint32_t from_id, uint32_t to_id) {
  if (from_id == to_id) return;
  auto it = consumers.find(from_id);
  if (it == consumers.end()) return;
  std::vector<ynn_node*> from_consumers = std::move(it->second);
  consumers.erase(it);
  std::sort(from_consumers.begin(), from_consumers.end());
  from_consumers.erase(
      std::unique(from_consumers.begin(), from_consumers.end()),
      from_consumers.end());
  for (ynn_node* consumer : from_consumers) {
    for (uint32_t& input : consumer->inputs) {
      if (input == from_id) {
        input = to_id;
        if (to_id != YNN_INVALID_VALUE_ID) {
          consumers[to_id].push_back(consumer);
        }
      }
    }
  }
}
