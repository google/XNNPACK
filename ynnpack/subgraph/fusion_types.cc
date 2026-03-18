#include "ynnpack/subgraph/fusion_types.h"

#include <cassert>

#include "ynnpack/subgraph/subgraph.h"

subgraph_analysis::subgraph_analysis(ynn_subgraph& subgraph) {
  for (ynn_node& node : subgraph.nodes) {
    if (!node.is_valid()) continue;
    for (uint32_t input : node.inputs) {
      consumers[input].push_back(&node);
    }
    assert(producers.find(node.outputs[0]) == producers.end());
    producers[node.outputs[0]] = &node;
  }
}
