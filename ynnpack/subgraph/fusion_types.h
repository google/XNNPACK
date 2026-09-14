// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_FUSION_TYPES_H_
#define XNNPACK_YNNPACK_SUBGRAPH_FUSION_TYPES_H_

#include <cassert>
#include <cstdint>
#include <map>
#include <type_traits>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

struct subgraph_analysis {
  bool is_valid;
  std::map<uint32_t, ynn_node*> producers;
  std::map<uint32_t, std::vector<ynn_node*>> consumers;

  ynn_node* producer_of(uint32_t id) {
    assert(is_valid);
    auto i = producers.find(id);
    return i != producers.end() ? i->second : nullptr;
  }
  const ynn_node* producer_of(uint32_t id) const {
    assert(is_valid);
    auto i = producers.find(id);
    return i != producers.end() ? i->second : nullptr;
  }

  ynn_node* single_consumer_of(uint32_t id) {
    assert(is_valid);
    auto i = consumers.find(id);
    return i != consumers.end() && i->second.size() == 1 ? i->second[0]
                                                         : nullptr;
  }
  const ynn_node* single_consumer_of(uint32_t id) const {
    assert(is_valid);
    auto i = consumers.find(id);
    return i != consumers.end() && i->second.size() == 1 ? i->second[0]
                                                         : nullptr;
  }

  void invalidate() { is_valid = false; }

  // Add all inputs/outputs of node to consumers/producers.
  void add_node(ynn_node& node);

  // Remove all inputs/outputs of node from consumers/producers.
  void remove_node(ynn_node& node);

  // Remove node from consumers/producers and invalidate it.
  void invalidate_node(ynn_node& node);

  // Update a node's inputs/outputs in the analysis while executing update_fn.
  template <typename F>
  auto update_node(ynn_node& node, F&& update_fn) -> decltype(update_fn()) {
    remove_node(node);
    if constexpr (std::is_void_v<std::invoke_result_t<F>>) {
      update_fn();
      add_node(node);
    } else {
      auto result = update_fn();
      add_node(node);
      return result;
    }
  }

  // Replace a specific input of node with new_id, updating consumers.
  void replace_input_id(ynn_node& node, uint32_t& input_id, uint32_t new_id);
  void replace_input(ynn_node& node, size_t index, uint32_t new_id);

  // Replace all uses of from_id with to_id across all consumers.
  void replace_all_uses(uint32_t from_id, uint32_t to_id);

  explicit subgraph_analysis(ynn_subgraph& subgraph);

 private:
  void remove_consumer(uint32_t value_id, ynn_node* node);
};

#endif  // XNNPACK_YNNPACK_SUBGRAPH_FUSION_TYPES_H_
