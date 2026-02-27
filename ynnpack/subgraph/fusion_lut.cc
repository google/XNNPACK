#include "ynnpack/subgraph/fusion_lut.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <variant>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/elementwise.h"
#include "ynnpack/subgraph/fusion_types.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"
#include "slinky/runtime/buffer.h"

namespace ynn {

namespace {

constexpr size_t kLog2MaxLutInputSize = 8;

const std::array<int8_t, 256>& get_int8_lut_data() {
  static const auto data = [] {
    std::array<int8_t, 256> result;
    for (int i = -128; i < 128; ++i) {
      result[static_cast<uint8_t>(i)] = i;
    }
    return result;
  }();
  return data;
}

const std::array<uint8_t, 256>& get_uint8_lut_data() {
  static const auto data = [] {
    std::array<uint8_t, 256> result;
    for (int i = 0; i < 256; ++i) {
      result[i] = static_cast<uint8_t>(i);
    }
    return result;
  }();
  return data;
}

const void* get_lut_input_data(ynn_type type) {
  if (type == ynn_type_int8) {
    return get_int8_lut_data().data();
  } else if (type == ynn_type_uint8) {
    return get_uint8_lut_data().data();
  } else {
    assert(false);
    return nullptr;
  }
}

bool is_elementwise_node(const ynn_node& node) {
  return std::holds_alternative<ynn_node::unary_elementwise>(node.op) ||
         std::holds_alternative<ynn_node::binary_elementwise>(node.op) ||
         std::holds_alternative<ynn_node::ternary_elementwise>(node.op);
}

// Returns the set of inputs to `n` that are not static.
std::set<uint32_t> get_variable_inputs(const ynn_subgraph& subgraph,
                                       const ynn_node& n) {
  std::set<uint32_t> inputs;
  if (is_elementwise_node(n)) {
    for (size_t i = 0; i < n.inputs.size(); ++i) {
      const uint32_t id = n.inputs[i];
      if (id != YNN_INVALID_VALUE_ID) {
        if (subgraph.value(id).is_static()) {
          if (!subgraph.value(id).is_static_scalar()) {
            // A constant that is not a scalar can not be fused.
            inputs.insert(id);
          }
        } else {
          // For unary elementwise operations, the second input is the
          // parameters tensor, which we don't want to fuse.
          if (std::get_if<ynn_node::unary_elementwise>(&n.op) && i == 1) {
            continue;
          }
          inputs.insert(id);
        }
      }
    }
  }
  return inputs;
}

bool should_lut_single_node(const ynn_node& node) {
  if (auto unary = std::get_if<ynn_node::unary_elementwise>(&node.op)) {
    switch (unary->op) {
      case ynn_unary_cosine:
      case ynn_unary_cube_root:
      case ynn_unary_erf:
      case ynn_unary_exp:
      case ynn_unary_expm1:
      case ynn_unary_hardswish:
      case ynn_unary_log:
      case ynn_unary_log1p:
      case ynn_unary_reciprocal_square_root:
      case ynn_unary_sigmoid:
      case ynn_unary_sine:
      case ynn_unary_square:
      case ynn_unary_square_root:
      case ynn_unary_tanh:
        return true;
      default:
        break;
    }
  } else if (auto binary =
                 std::get_if<ynn_node::binary_elementwise>(&node.op)) {
    switch (binary->op) {
      case ynn_binary_leaky_relu:
      case ynn_binary_squared_difference:
        return true;
      default:
        break;
    }
  }
  return false;
}

// A subgraph that can be optimized with a unary LUT.
// The subgraph is a chain of elementwise nodes ending at the node with output
// `output_id`.
// The input to the start of the chain, denoted by `input_id`, must be int8 or
// uint8.
// Intermediate nodes may have any type and may have multiple consumers as long
// as their outputs eventually feed into `output_id`.
class subgraph_candidate {
 public:
  bool is_valid() const {
    return input_id_ != YNN_INVALID_VALUE_ID &&
           output_id_ != YNN_INVALID_VALUE_ID && !nodes_.empty();
  }
  size_t size() const { return nodes_.size(); }
  bool has_value(uint32_t id) const {
    return values_.find(id) != values_.end();
  }
  bool has_node(const ynn_node& node) const {
    return nodes_.find(&node) != nodes_.end();
  }
  const std::set<const ynn_node*>& get_nodes() const { return nodes_; }
  const std::set<uint32_t>& get_values() const { return values_; }
  uint32_t get_input_id() const { return input_id_; }
  uint32_t get_output_id() const { return output_id_; }
  uint32_t& output_id() { return output_id_; }

  void add_node(const ynn_node& node) { nodes_.insert(&node); }
  void add_value(uint32_t id) { values_.insert(id); }
  void set_input_id(uint32_t id) { input_id_ = id; }
  void set_output_id(uint32_t id) { output_id_ = id; }

 private:
  uint32_t input_id_ = YNN_INVALID_VALUE_ID;
  uint32_t output_id_ = YNN_INVALID_VALUE_ID;
  std::set<uint32_t> values_;
  std::set<const ynn_node*> nodes_;
};

// Finds a chain of unary, binary-with-constant, or ternary-with-constant nodes
// ending at `node`.
subgraph_candidate find_subgraph_for_unary_lut(
    const ynn_subgraph& subgraph, const ynn_node& node,
    const subgraph_analysis& analysis) {
  subgraph_candidate candidate;
  if (node.outputs.empty()) {
    return candidate;
  }
  const ynn_value& output = subgraph.value(node.outputs[0]);
  if (type_size_bytes(output.type) > 1) {
    return candidate;
  }

  std::set<uint32_t> inputs = get_variable_inputs(subgraph, node);
  if (inputs.empty()) {
    return candidate;
  }

  candidate.add_node(node);
  candidate.set_output_id(node.outputs[0]);

  subgraph_candidate best_candidate;
  std::set<uint32_t> inputs_to_traverse(inputs.begin(), inputs.end());

  auto maybe_update_best_candidate = [&](const subgraph_candidate& candidate) {
    if (inputs_to_traverse.size() == 1) {
      uint32_t in_id = *inputs_to_traverse.begin();
      if (type_size_bits(subgraph.value(in_id).type) <= kLog2MaxLutInputSize) {
        best_candidate = candidate;
        best_candidate.set_input_id(in_id);
      }
    }
  };

  auto has_path_to_output = [&](uint32_t& value_id, ynn_node& producer) {
    // Find a value in the frontier that is ready to be expanded.
    for (uint32_t id : inputs_to_traverse) {
      auto producer_it = analysis.producers.find(id);
      if (producer_it != analysis.producers.end()) {
        const ynn_node& p = *producer_it->second;
        std::set<uint32_t> next_inputs = get_variable_inputs(subgraph, p);
        if (!next_inputs.empty()) {
          value_id = id;
          producer = p;
          return true;
        }
      }
    }
    return value_id != YNN_INVALID_VALUE_ID;
  };

  maybe_update_best_candidate(candidate);
  while (!inputs_to_traverse.empty()) {
    uint32_t value_id = YNN_INVALID_VALUE_ID;
    ynn_node producer;

    if (has_path_to_output(value_id, producer)) {
      inputs_to_traverse.erase(value_id);
      candidate.add_value(value_id);
      candidate.add_node(producer);

      std::set<uint32_t> producer_inputs =
          get_variable_inputs(subgraph, producer);
      for (uint32_t id : producer_inputs) {
        if (!candidate.has_value(id)) {
          inputs_to_traverse.insert(id);
        }
      }
      maybe_update_best_candidate(candidate);
    } else {
      // No more nodes to traverse.
      break;
    }
  }

  return best_candidate;
}

}  // namespace

bool rewrite_subgraph_for_unary_lut(ynn_subgraph& subgraph,
                                    subgraph_analysis& analysis) {
  // Find the longest subgraph that can be optimized with a unary LUT.
  // We iterate through the nodes in reverse order since
  // `find_subgraph_for_unary_lut` traverses the nodes in reverse order.
  subgraph_candidate best_candidate;
  for (auto it = subgraph.nodes.rbegin(); it != subgraph.nodes.rend(); ++it) {
    ynn_node& node = *it;
    if (!node.is_valid()) continue;
    subgraph_candidate candidate =
        find_subgraph_for_unary_lut(subgraph, node, analysis);
    if (candidate.is_valid() && candidate.size() > best_candidate.size()) {
      best_candidate = candidate;
    }
  }

  if (!best_candidate.is_valid()) {
    return false;
  }

  // If candidate is a single node, check if it is worth replacing with a lut.
  if (best_candidate.size() == 1) {
    const ynn_node& node = **best_candidate.get_nodes().begin();
    if (!should_lut_single_node(node)) {
      return false;
    }
  }

  YNN_LOG_DEBUG() << "Found candidate of size " << best_candidate.size();

  // Clone the candidate subgraph.
  uint32_t lut_input_id = YNN_INVALID_VALUE_ID;
  uint32_t lut_output_id = YNN_INVALID_VALUE_ID;
  std::optional<ynn_subgraph> lut_subgraph = clone_subgraph_subset(
      subgraph, best_candidate.get_input_id(), best_candidate.get_output_id(),
      lut_input_id, lut_output_id);
  if (!lut_subgraph) {
    return false;
  }

  // Resize subgraph to be the same rank and size as the LUT table.
  const size_t range = 256;
  for (ynn_value& value : lut_subgraph->values) {
    if (value.is_valid() && !value.is_static()) {
      value.extents = {static_cast<slinky::index_t>(range)};
      value.data = nullptr;
    }
  }
  for (ynn_node& node : lut_subgraph->nodes) {
    node.checks.clear();
  }

  const ynn_value& input_value = lut_subgraph->value(lut_input_id);
  const void* input_data = get_lut_input_data(input_value.type);

  // Run the cloned subgraph.
  ynn_runtime runtime(*lut_subgraph, nullptr, 0);
  if (runtime.build() != ynn_status_success) {
    return false;
  }

  size_t input_dims = range;
  if (ynn_set_external_value_shape(&runtime, lut_input_id, 1, &input_dims) !=
      ynn_status_success) {
    return false;
  }
  if (ynn_set_external_value_data(&runtime, lut_input_id, (void*)input_data) !=
      ynn_status_success) {
    return false;
  }

  const ynn_value& output_value =
      subgraph.value(best_candidate.get_output_id());
  std::vector<uint8_t> output_data(range * type_size_bytes(output_value.type));
  if (ynn_set_external_value_data(&runtime, lut_output_id,
                                  output_data.data()) != ynn_status_success) {
    return false;
  }

  if (runtime.reshape() != ynn_status_success ||
      runtime.setup() != ynn_status_success ||
      runtime.invoke() != ynn_status_success) {
    return false;
  }

  // Create a new LUT table with the output values generated by the runtime.
  uint32_t lut_id = YNN_INVALID_VALUE_ID;
  size_t lut_dims[] = {range};
  if (ynn_define_tensor_value(&subgraph, output_value.type, 1, lut_dims,
                              output_data.data(), YNN_INVALID_VALUE_ID,
                              YNN_INVALID_VALUE_ID, YNN_VALUE_FLAG_COPY_DATA,
                              &lut_id) != ynn_status_success) {
    return false;
  }

  // Replace and invalidate all nodes in `candidate` with the new LUT node.
  ynn_node* output_node = analysis.producers[best_candidate.get_output_id()];

  define_lut(subgraph, *output_node, best_candidate.get_input_id(), lut_id,
             best_candidate.output_id());
  return true;
}

}  // namespace ynn
