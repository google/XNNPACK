#include "test/subgraph/rewrites/subgraph_matcher.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/subgraph/subgraph-utils.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/subgraph.h"

namespace xnnpack {

struct ValueOrNode {
  enum class Type { kUnknown, kNode, kValue };

  Type type;
  union {
    const xnn_value* value;
    const xnn_node* node;
  };

  uint32_t GetOriginalIndex() const {
    switch (type) {
      case Type::kNode:
        return node->id;
      case Type::kValue:
        return value->id;
      case Type::kUnknown:
        return 0;
    }
    // Some older compilers raise warnings about control reaching end of
    // non-void function.
    return 0;
  }

  Type GetParentType() const {
    switch (type) {
      case Type::kNode:
        return Type::kValue;
      case Type::kValue:
        return Type::kNode;
      case Type::kUnknown:
        return Type::kUnknown;
    }
    // Some older compilers raise warnings about control reaching end of
    // non-void function.
    return Type::kUnknown;
  }

  bool IsValue() const noexcept { return type == Type::kValue; }
  void Set(const xnn_value* v) noexcept {
    type = Type::kValue;
    value = v;
  }

  bool IsNode() const noexcept { return type == Type::kNode; }
  void Set(const xnn_node* n) noexcept {
    type = Type::kNode;
    node = n;
  }

  friend bool operator==(const ValueOrNode& a, const ValueOrNode& b) {
    if (a.type != b.type) {
      return false;
    }
    if (a.IsNode()) {
      auto SameType = [](const xnn_node& a, const xnn_node& b) {
        if (a.type == b.type) {
          switch (a.type) {
            case xnn_node_type_unary_elementwise:
              return a.unary_operator == b.unary_operator;
            case xnn_node_type_binary_elementwise:
              return a.binary_operator == b.binary_operator;
            case xnn_node_type_fully_connected:
            case xnn_node_type_batch_matrix_multiply:
              return a.packed_input_datatype == b.packed_input_datatype;
            default:
              return true;
          }
          return true;
        }
        return false;
      };
      return SameType(*a.node, *b.node) && a.node->flags == b.node->flags &&
             a.node->num_inputs == b.node->num_inputs &&
             a.node->num_outputs == b.node->num_outputs &&
             // When a node is created, it is memset to 0. This means that
             // there are no uninitialised padding bytes in a union and calling
             // memcmp is safe.
             std::memcmp(&a.node->params, &b.node->params,
                         sizeof(a.node->params)) == 0;
    } else if (a.IsValue()) {
      if (a.value->datatype == b.value->datatype &&
          a.value->shape.num_dims == b.value->shape.num_dims) {
        return std::memcmp(a.value->shape.dim, b.value->shape.dim,
                           a.value->shape.num_dims *
                               sizeof(a.value->shape.dim[0])) == 0;
      }
    }
    return false;
  }

  friend std::ostream& operator<<(std::ostream& os, const ValueOrNode& v) {
    switch (v.type) {
      case Type::kNode:
        os << "node[" << v.node->id << ", ";
        if (v.node->type == xnn_node_type_unary_elementwise) {
          os << xnn_unary_operator_to_string(v.node->unary_operator);
        } else if (v.node->type == xnn_node_type_binary_elementwise) {
          os << xnn_binary_operator_to_string(v.node->binary_operator);
        } else {
          os << xnn_node_type_to_string(v.node->type);
        }
        os << ']';
        break;
      case Type::kValue: {
        os << "value[" << v.value->id << ", "
           << xnn_datatype_to_string(v.value->datatype) << ", ";
        const char* sep = "";
        for (int i = 0; i < v.value->shape.num_dims; ++i) {
          os << sep << v.value->shape.dim[i];
          sep = ", ";
        }
        os << ']';
        break;
      }
      case Type::kUnknown:
        os << "unknown";
        break;
    }
    return os;
  }
};

struct SortNode : ValueOrNode {
  enum class Step { kNone, kUpwards };
  int64_t level;
  uint32_t structural_hash;
  // To skip nodes that have already been processed at a given step.
  Step last_process_step = Step::kNone;
  // Values only store their first consumer. When going up the graph we'll store
  // the consumers of a value here.
  std::vector<uint32_t> value_children;
  uint32_t temp_idx;

  friend std::ostream& operator<<(std::ostream& os, const SortNode& v) {
    os << static_cast<const ValueOrNode&>(v) << " lvl:" << v.level
       << " hash:" << v.structural_hash;
    return os;
  }

  void ComputeHash(std::vector<uint32_t>& child_hashes) {
    std::sort(child_hashes.begin(), child_hashes.end());
    uint32_t hash = 0;
    for (uint32_t h : child_hashes) {
      hash = murmur_hash3(&h, sizeof(h), hash);
    }
    switch (type) {
      case Type::kNode:
        hash = murmur_hash3(&node->type, sizeof(node->type), hash);
        // When a node is created, it is memset to 0. This means that there are
        // no uninitialised padding bytes in a union and hashing the full union
        // is safe.
        hash = murmur_hash3(&node->params, sizeof(node->params), hash);
        hash = murmur_hash3(&node->flags, sizeof(node->flags), hash);
        break;
      case Type::kValue:
        hash = murmur_hash3(&value->datatype, sizeof(value->datatype), hash);
        hash = murmur_hash3(value->shape.dim,
                            sizeof(value->shape.dim[0]) * value->shape.num_dims,
                            hash);
        break;
      case Type::kUnknown:
        break;
    }
    structural_hash = hash;
  }

  void AddChild(uint32_t consumer_index) {
    if (type == Type::kValue) {
      value_children.push_back(consumer_index);
    }
  }

  std::vector<uint32_t> GetOriginalParentIndices(uint32_t value_offset) const {
    std::vector<uint32_t> indices;
    switch (type) {
      case Type::kNode:
        indices.assign(node->inputs, node->inputs + node->num_inputs);
        for (uint32_t& i : indices) {
          i += value_offset;
        }
        break;
      case Type::kValue:
        if (value->producer != XNN_INVALID_NODE_ID) {
          indices.push_back(value->producer);
        }
        break;
      case Type::kUnknown:
        xnn_log_error("Cannot get parents of unknown node type.");
        break;
    }
    return indices;
  }

  std::vector<uint32_t> GetOriginalChildrenIndices(
      uint32_t value_offset) const {
    std::vector<uint32_t> indices;
    switch (type) {
      case Type::kNode:
        indices.assign(node->outputs, node->outputs + node->num_outputs);
        for (uint32_t& i : indices) {
          i += value_offset;
        }
        break;
      case Type::kValue:
        indices = value_children;
        break;
      case Type::kUnknown:
        xnn_log_error("Cannot get children of unknown node type.");
        break;
    }
    return indices;
  }
};

struct SortedGraphView {
  std::vector<SortNode> nodes;
  std::vector<std::pair<size_t, size_t>> edges;
};

struct SortedIndexMap {
  explicit SortedIndexMap(xnn_subgraph_t subgraph)
      : value_map(subgraph->num_values), node_map(subgraph->num_nodes) {}

  void Set(const SortNode& node, size_t idx) {
    switch (node.type) {
      case ValueOrNode::Type::kUnknown:
        break;
      case ValueOrNode::Type::kValue:
        value_map[node.value->id] = idx;
        break;
      case ValueOrNode::Type::kNode:
        node_map[node.node->id] = idx;
        break;
    }
  }

  size_t Get(ValueOrNode::Type type, uint32_t original_id) const {
    switch (type) {
      case ValueOrNode::Type::kUnknown:
        return 0;
      case ValueOrNode::Type::kValue:
        return value_map[original_id];
      case ValueOrNode::Type::kNode:
        return node_map[original_id];
    }
    // Some older compilers raise warnings about control reaching end of
    // non-void function.
    return 0;
  }

  size_t GetFirstParentIndex(const SortNode& node) {
    const std::vector<uint32_t>& parent_ids =
        node.GetOriginalParentIndices(/*value_offset=*/0);
    auto first_it = std::min_element(parent_ids.begin(), parent_ids.end());
    if (first_it == parent_ids.end()) {
      return SIZE_MAX;
    }
    return Get(node.GetParentType(), *first_it);
  }

  std::vector<size_t> value_map;
  std::vector<size_t> node_map;
};

SortedGraphView ComputeSortedView(xnn_subgraph_t subgraph) {
  xnn_subgraph_analyze_consumers_and_producers(subgraph);

  // Holds the sorted nodes.
  std::vector<SortNode> nodes;

  // We copy the graph verbatim into our structure.
  // - A node's index is node.id.
  // - A value's index is value.id + num_nodes.
  const size_t first_value = subgraph->num_nodes;
  nodes.reserve(subgraph->num_nodes + subgraph->num_values);
  for (uint32_t i = 0; i < subgraph->num_nodes; ++i) {
    nodes.emplace_back();
    nodes.back().Set(subgraph->nodes + i);
  }
  for (uint32_t i = 0; i < subgraph->num_values; ++i) {
    nodes.emplace_back();
    nodes.back().Set(subgraph->values + i);
  }
  // The next node indices to process, in order
  std::deque<uint32_t> next_nodes;
  // The root nodes that will start the downwards pass.
  std::deque<uint32_t> root_nodes;
  // Used to aggregate the structural hashes of child nodes.
  std::vector<uint32_t> child_hashes;

  // Go through the graph and find the leaf values.
  for (size_t i = first_value; i < nodes.size(); ++i) {
    if (nodes[i].value->first_consumer == XNN_INVALID_NODE_ID) {
      // Initialize the leaf's level to 0. Other nodes will be set while going
      // up the graph.
      nodes[i].level = 0;
      next_nodes.push_back(i);
    }
    if (nodes[i].value->producer == XNN_INVALID_NODE_ID) {
      root_nodes.push_back(i);
    }
  }
  // Go up the graph and compute the structure signature to break ties when we
  // sort.
  int64_t max_level = 0;
  while (!next_nodes.empty()) {
    const uint32_t node_id = next_nodes.front();
    next_nodes.pop_front();
    SortNode& node = nodes[node_id];
    if (node.last_process_step >= SortNode::Step::kUpwards) {
      continue;
    }
    node.last_process_step = SortNode::Step::kUpwards;

    if (node.IsValue() && node.value->num_consumers == 0) {
      node.level = -1;
      continue;
    }

    child_hashes.clear();
    for (uint32_t child_id : node.GetOriginalChildrenIndices(first_value)) {
      child_hashes.push_back(nodes[child_id].structural_hash);
      node.level = std::max(nodes[child_id].level + 1, node.level);
    }
    node.ComputeHash(child_hashes);
    max_level = std::max(node.level, max_level);

    const std::vector<uint32_t>& parent_ids =
        node.GetOriginalParentIndices(first_value);
    for (uint32_t parent_id : parent_ids) {
      // Keep track of value consumers, this is a no-op for nodes.
      nodes[parent_id].AddChild(node_id);
    }

    next_nodes.insert(next_nodes.end(), parent_ids.begin(), parent_ids.end());
  }

  // Reverse node levels to have the root at 0 and the leafs at max_level.
  for (size_t i = 0; i < nodes.size(); ++i) {
    nodes[i].level = max_level - nodes[i].level;
    nodes[i].temp_idx = i;
  }

  // Sort the nodes by level then re-sort within levels by structural hash,
  // break ties using parent nodes order at the previous level.
  std::sort(
      nodes.begin(), nodes.end(),
      [](const SortNode& a, const SortNode& b) { return a.level < b.level; });

  // Mapping from the subgraph values to their new positions.
  SortedIndexMap sorted_index_map(subgraph);
  size_t node_idx = 0;
  // Skip level 0 to ensure we always have a parent.
  auto level_it = nodes.begin();
  for (; level_it != nodes.end() && level_it->level == 0; ++level_it) {
    sorted_index_map.Set(*level_it, node_idx++);
  }
  auto level_it_end = level_it;
  while (level_it_end != nodes.end() && level_it_end->level <= max_level) {
    while (level_it_end != nodes.end() &&
           level_it_end->level == level_it->level) {
      ++level_it_end;
    }
    std::sort(level_it, level_it_end,
              [&](const SortNode& a, const SortNode& b) {
                if (a.structural_hash < b.structural_hash) {
                  return true;
                } else if (a.structural_hash == b.structural_hash) {
                  return sorted_index_map.GetFirstParentIndex(a) <
                         sorted_index_map.GetFirstParentIndex(b);
                }
                return false;
              });
    for (; level_it != level_it_end; ++level_it) {
      sorted_index_map.Set(*level_it, node_idx++);
    }
  }
  nodes.erase(level_it_end, nodes.end());

  // Create the edge list.
  std::vector<std::pair<size_t, size_t>> edges;
  for (size_t i = 0; i < nodes.size(); ++i) {
    SortNode& node = nodes[i];
    size_t first = 0;
    // Add edges to the children.
    if (node.IsValue()) {
      first = sorted_index_map.value_map[node.value->id];
      for (uint32_t consumer_id :
           node.GetOriginalChildrenIndices(/*value_offset=*/0)) {
        edges.emplace_back(first, sorted_index_map.node_map[consumer_id]);
      }
    } else {
      first = sorted_index_map.node_map[node.node->id];
      for (uint32_t output_id :
           node.GetOriginalChildrenIndices(/*value_offset=*/0)) {
        edges.emplace_back(first, sorted_index_map.value_map[output_id]);
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  return {.nodes = std::move(nodes), .edges = std::move(edges)};
}

// Apply the sorted order to a subgraph.
void Apply(const SortedGraphView& sorted, xnn_subgraph_t subgraph) {
  size_t sorted_idx = 0;
  std::unique_ptr<xnn_node[]> sorted_nodes =
      std::make_unique<xnn_node[]>(subgraph->num_nodes);
  for (size_t i = 0; i < sorted.nodes.size(); ++i) {
    const ValueOrNode& node = sorted.nodes[i];
    if (node.IsNode()) {
      sorted_nodes[sorted_idx] = *node.node;
      sorted_nodes[sorted_idx].id = sorted_idx;
      ++sorted_idx;
    }
  }
  std::memcpy(subgraph->nodes, sorted_nodes.get(),
              sizeof(xnn_node) * subgraph->num_nodes);
}

class LogGraphStreamAdapter {
 public:
  LogGraphStreamAdapter(const LogGraphStreamAdapter&) = delete;
  LogGraphStreamAdapter& operator=(const LogGraphStreamAdapter&) = delete;
  LogGraphStreamAdapter(LogGraphStreamAdapter&&) = delete;
  LogGraphStreamAdapter& operator=(LogGraphStreamAdapter&&) = delete;

  explicit LogGraphStreamAdapter(xnn_subgraph_t subgraph) {
    size_t len = 0;
    int c_ret = 0;
#if defined(_MSC_VER)
    FILE* out;
    c_ret = tmpfile_s(&out);
#else
    FILE* out = open_memstream(&str, &len);
#endif
    if (c_ret != 0 || !out) {
      const char error_msg[] = "<error creating tmp file to log graph>";
      str = reinterpret_cast<char*>(malloc(sizeof(error_msg)));
      std::memcpy(str, error_msg, sizeof(error_msg));
      return;
    }
    if (subgraph) {
      fputs("\n", out);
      xnn_subgraph_log_dot_impl(subgraph, out);
    } else {
      fputs("<invalid subgraph handle>", out);
    }
#if defined(_MSC_VER)
    // Get position in file for length.
    int64_t pos = _ftelli64(out);
    // Allocate buffer.
    str = malloc(pos + 1, sizeof(char));
    // Read temp file.
    fseek(out, 0, SEEK_SET);
    len = fread(str, pos, sizeof(char), pos, out);
    if (len == 0 && errno == EINVAL) {
      const char error_msg[] = "<error reading tmp file to get graph log>";
      str = malloc(sizeof(error_msg));
      std::memcpy(str, error_msg, sizeof(error_msg));
    } else {
      str[bytes_read] = '\0';
    }
#endif
    fclose(out);
  }

  ~LogGraphStreamAdapter() { free(str); }

  friend std::ostream& operator<<(std::ostream& os,
                                  const LogGraphStreamAdapter a) {
    return os << a.str;
  }

 private:
  char* str = nullptr;
};

void IsomorphicGraphMatcher::DescribeTo(std::ostream* os) const {
  if (os) {
    *os << "is isomorphic to " << LogGraphStreamAdapter(subgraph_);
  }
}

void IsomorphicGraphMatcher::DescribeNegationTo(std::ostream* os) const {
  if (os) {
    *os << "is not isomorphic to" << LogGraphStreamAdapter(subgraph_);
  }
}

bool CompareSortedGraphs(const SortedGraphView& a, const SortedGraphView& b,
                         std::ostream* os) {
  auto [a_node_it, b_node_it] = std::mismatch(a.nodes.begin(), a.nodes.end(),
                                              b.nodes.begin(), b.nodes.end());
  auto PrintNodesAndEdges = [&](const SortedGraphView& a) {
    size_t i = 0;
    *os << "\n";
    for (auto& n : a.nodes) {
      *os << i++ << ": " << n << "\n";
    }
    for (auto& e : a.edges) {
      *os << e.first << "," << e.second << " ";
    }
    i = 0;
    *os << "\n";
  };
  if (a_node_it != a.nodes.end() || b_node_it != b.nodes.end()) {
    if (os) {
      *os << "Nodes at index " << std::distance(a.nodes.begin(), a_node_it)
          << " don't match ";
      if (a_node_it == a.nodes.end()) {
        *os << "<no node>";
      } else {
        *os << *a_node_it;
      }
      *os << " != ";
      if (b_node_it == b.nodes.end()) {
        *os << "<no node>";
      } else {
        *os << *b_node_it;
      }
      PrintNodesAndEdges(a);
      PrintNodesAndEdges(b);
    }
    return false;
  }
  auto [a_edge_it, b_edge_it] = std::mismatch(a.edges.begin(), a.edges.end(),
                                              b.edges.begin(), b.edges.end());
  if (a_edge_it != a.edges.end() || b_edge_it != b.edges.end()) {
    if (os) {
      *os << "Nodes " << std::distance(a.edges.begin(), a_edge_it)
          << " don't match ";
      if (a_edge_it == a.edges.end()) {
        *os << "<no edge> ";
      } else {
        *os << '(' << a_edge_it->first << ", " << a_edge_it->second << ')';
      }
      if (b_edge_it == b.edges.end()) {
        *os << "<no edge> ";
      } else {
        *os << '(' << b_edge_it->first << ", " << b_edge_it->second << ')';
      }
      PrintNodesAndEdges(a);
      PrintNodesAndEdges(b);
    }
    return false;
  }

  return true;
}

bool IsomorphicGraphMatcher::MatchAndExplain(xnn_subgraph_t subgraph,
                                             std::ostream* listener) const {
  if (!subgraph) {
    if (listener) {
      *listener << "cannot match an invalid graph handle: "
                << testing::PrintToString(subgraph) << ".";
    }
    return false;
  }
  if (!subgraph_) {
    if (listener) {
      *listener << "cannot match against an invalid graph handle: "
                << testing::PrintToString(subgraph_)
                << ". Check the graph the matcher was built with.";
    }
    return false;
  }

  SortedGraphView sorted_graph = ComputeSortedView(subgraph);
  SortedGraphView sorted_reference = ComputeSortedView(subgraph_);
  std::stringstream error_explanation;
  if (!CompareSortedGraphs(sorted_graph, sorted_reference,
                           listener ? &error_explanation : nullptr)) {
    if (listener) {
      *listener << "Graphs are not isomorphic. Got\n"
                << LogGraphStreamAdapter(subgraph) << "\n"
                << error_explanation.str();
    }
    return false;
  }
  return true;
}

IsomorphicGraphMatcher IsIsomorphicTo(xnn_subgraph_t subgraph) {
  return IsomorphicGraphMatcher(subgraph);
}

}  // namespace xnnpack

void PrintTo(xnn_subgraph_t subgraph, std::ostream* os) {
  if (os) {
    *os << "xnn_subgraph_t(" << subgraph << ")";
  }
}
