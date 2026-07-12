#include "ynnpack/subgraph/fusion_types.h"

#include <cassert>
#include <cstdint>

#include "ynnpack/subgraph/subgraph.h"

subgraph_analysis::subgraph_analysis(ynn_subgraph& subgraph) : is_valid(true) {
  for (ynn_node& node : subgraph.nodes) {
    if (!node.is_valid()) continue;
    for (uint32_t input : node.inputs) {
      consumers[input].push_back(&node);
    }
    for (uint32_t output : node.outputs) {
      assert(producers.find(output) == producers.end());
      producers[output] = &node;
    }
  }
}
