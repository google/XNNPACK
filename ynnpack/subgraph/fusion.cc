// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <variant>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/subgraph/elementwise.h"
#include "ynnpack/subgraph/subgraph.h"

namespace {

struct subgraph_analysis {
  std::map<uint32_t, ynn_node*> producers;
  std::map<uint32_t, std::vector<ynn_node*>> consumers;

  explicit subgraph_analysis(ynn_subgraph& subgraph) {
    for (ynn_node& node : subgraph.nodes) {
      for (uint32_t input : node.inputs) {
        consumers[input].push_back(&node);
      }
      assert(producers.find(node.outputs[0]) == producers.end());
      producers[node.outputs[0]] = &node;
    }
  }
};

bool is_unary_node(const ynn_node& node, ynn_unary_operator op) {
  const ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&node.op);
  return unary && unary->op == op;
}
bool is_binary_node(const ynn_node& node, ynn_binary_operator op) {
  const ynn_node::binary_elementwise* binary =
      std::get_if<ynn_node::binary_elementwise>(&node.op);
  return binary && binary->op == op;
}

// Rewrite add(multiply(a, b), c) to multiply_add(a, b, c)
bool rewrite_multiply_add(ynn_subgraph& subgraph, ynn_node& node,
                          subgraph_analysis& analysis) {
  if (!is_binary_node(node, ynn_binary_add)) {
    return false;
  }

  for (int i : {0, 1}) {
    auto producer = analysis.producers.find(node.inputs[i]);
    if (producer != analysis.producers.end() &&
        analysis.consumers[producer->second->outputs[0]].size() == 1) {
      if (is_binary_node(*producer->second, ynn_binary_multiply)) {
        // This is a multiply-add. Do we have a kernel for this case?
        const ynn_value& a = subgraph.value(producer->second->inputs[0]);
        const ynn_value& b = subgraph.value(producer->second->inputs[1]);
        const ynn_value& c = subgraph.value(node.inputs[1 - i]);
        const ynn_value& x = subgraph.value(node.outputs[0]);
        const ynn::ternary_kernel_fn kernel = ynn::get_ternary_kernel(
            ynn::ternary_op::multiply_add, a.type, b.type, c.type, x.type);
        if (kernel != nullptr) {
          // Yes we do. Rewrite this to a multiply-add.
          YNN_LOG_DEBUG() << "Rewriting convert to ternary multiply";
          ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id, kernel);
          return true;
        }
      }
    }
  }
  return false;
}

// Rewrite subtract(a, multiply(b, c)) to subtract_multiply(a, b, c)
bool rewrite_subtract_multiply(ynn_subgraph& subgraph, ynn_node& node,
                               subgraph_analysis& analysis) {
  uint32_t a_id = YNN_INVALID_VALUE_ID;
  uint32_t mul_id;
  if (is_unary_node(node, ynn_unary_negate)) {
    mul_id = node.inputs[0];
  } else if (is_binary_node(node, ynn_binary_subtract)) {
    a_id = node.inputs[0];
    mul_id = node.inputs[1];
  } else {
    return false;
  }

  auto producer = analysis.producers.find(mul_id);
  if (producer != analysis.producers.end() &&
      analysis.consumers[producer->second->outputs[0]].size() == 1) {
    if (is_binary_node(*producer->second, ynn_binary_multiply)) {
      // This is a subtract_multiply. Do we have a kernel for this case?
      const ynn_value& b = subgraph.value(producer->second->inputs[0]);
      const ynn_value& c = subgraph.value(producer->second->inputs[1]);
      const ynn_value& x = subgraph.value(node.outputs[0]);
      if (a_id == YNN_INVALID_VALUE_ID) {
        a_id = subgraph.get_scalar_value_id(x.type, YNN_INVALID_VALUE_ID,
                                            YNN_INVALID_VALUE_ID, 0.0f);
      }
      const ynn_value& a = subgraph.value(a_id);

      const ynn::ternary_kernel_fn kernel = ynn::get_ternary_kernel(
          ynn::ternary_op::subtract_multiply, a.type, b.type, c.type, x.type);
      if (kernel != nullptr) {
        YNN_LOG_DEBUG() << "Rewriting convert to ternary subtract_multiply";
        ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id, kernel);
        return true;
      }
    }
  }
  return false;
}

// Rewrite x = convert(a) where a is an int32 with a scale and no zero point to
// a binary (or ternary, if the scale is itself a multiply) multiply.
bool rewrite_convert_to_multiply(ynn_subgraph& subgraph, ynn_node& node,
                                 subgraph_analysis& analysis) {
  const ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&node.op);
  if (unary == nullptr || unary->op != ynn_unary_convert) {
    return false;
  }

  // The input should have a scale and no zero point.
  const ynn_value& input = subgraph.value(node.inputs[0]);
  if (input.scale_id == YNN_INVALID_VALUE_ID ||
      input.zero_point_id != YNN_INVALID_VALUE_ID) {
    return false;
  }

  // The output should not have a scale or zero point.
  const ynn_value& output = subgraph.value(node.outputs[0]);
  if (output.scale_id != YNN_INVALID_VALUE_ID ||
      output.zero_point_id != YNN_INVALID_VALUE_ID) {
    return false;
  }

  auto producer = analysis.producers.find(input.scale_id);
  if (producer != analysis.producers.end()) {
    if (is_binary_node(*producer->second, ynn_binary_multiply)) {
      uint32_t scale_a_id = producer->second->inputs[0];
      uint32_t scale_b_id = producer->second->inputs[1];

      const ynn::ternary_kernel_fn kernel =
          ynn::get_ternary_kernel(ynn::ternary_op::multiply, input.type,
                                  subgraph.value(scale_a_id).type,
                                  subgraph.value(scale_b_id).type, output.type);
      if (kernel != nullptr) {
        // This is a ternary integer*float*float multiply, and we have a kernel
        // that matches the types we have.
        YNN_LOG_DEBUG() << "Rewriting convert to ternary multiply";
        ynn::define_ternary(subgraph, node, node.inputs[0], scale_a_id,
                            scale_b_id, node.outputs[0], kernel);
        return true;
      }
    }
  }
  ynn::binary_kernel_fn kernel = ynn::get_binary_multiply_kernel(
      input.type, subgraph.value(input.scale_id).type, output.type);
  if (kernel != nullptr) {
    // This is a binary integer*float multiply, and we have a kernel that
    // matches the types we have.
    YNN_LOG_DEBUG() << "Rewriting convert to binary multiply";
    ynn::define_binary(subgraph, node, node.inputs[0], input.scale_id,
                       node.outputs[0], ynn_binary_multiply, kernel);
  }
  return false;
}

// Rewrite x = convert(a) where a is an int32 with a scale and no zero point to
// a binary (or ternary, if the scale is itself a multiply) multiply.
bool rewrite_convert_to_quantize(ynn_subgraph& subgraph, ynn_node& node,
                                 subgraph_analysis& analysis) {
  const ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&node.op);
  if (unary == nullptr || unary->op != ynn_unary_convert) {
    return false;
  }

  // The input should be a float.
  const ynn_value& input = subgraph.value(node.inputs[0]);
  if (ynn::type_is_integral(input.type)) {
    return false;
  }

  const ynn_value& output = subgraph.value(node.outputs[0]);
  if (output.type != ynn_type_int8 && output.type != ynn_type_uint8) {
    return false;
  }
  if (output.scale_id == YNN_INVALID_VALUE_ID ||
      output.zero_point_id == YNN_INVALID_VALUE_ID) {
    return false;
  }

  ynn::ternary_kernel_fn kernel = ynn::get_ternary_kernel(
      output.type == ynn_type_int8 ? ynn::ternary_op::quantize_int8
                                   : ynn::ternary_op::quantize_uint8,
      input.type, subgraph.value(output.scale_id).type,
      subgraph.value(output.zero_point_id).type, output.type);
  if (kernel != nullptr) {
    YNN_LOG_DEBUG() << "Rewriting convert to quantize";
    ynn::define_ternary(subgraph, node, node.inputs[0], output.scale_id,
                        output.zero_point_id, node.outputs[0], kernel);
  }
  return false;
}

// Rewrite min(max(a, b), c) to clamp(a, b, c)
bool rewrite_clamp(ynn_subgraph& subgraph, ynn_node& node,
                   subgraph_analysis& analysis) {
  if (!is_binary_node(node, ynn_binary_min)) {
    return false;
  }

  for (int i : {0, 1}) {
    auto producer = analysis.producers.find(node.inputs[i]);
    if (producer != analysis.producers.end() &&
        analysis.consumers[producer->second->outputs[0]].size() == 1) {
      if (is_binary_node(*producer->second, ynn_binary_max)) {
        // This is a clamp. Do we have a kernel for this case?
        const ynn_value& a = subgraph.value(producer->second->inputs[0]);
        const ynn_value& b = subgraph.value(producer->second->inputs[1]);
        const ynn_value& c = subgraph.value(node.inputs[1 - i]);
        const ynn_value& x = subgraph.value(node.outputs[0]);

        const ynn::ternary_kernel_fn kernel = ynn::get_ternary_kernel(
            ynn::ternary_op::clamp, a.type, b.type, c.type, x.type);
        if (kernel != nullptr) {
          // Yes we do. Rewrite this to a clamp.
          YNN_LOG_DEBUG() << "Rewriting min(max(a, b), c) to ternary clamp";
          ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id, kernel);
          return true;
        }
      }
    }
  }
  return false;
}

bool can_implicitly_broadcast(const ynn_node& node, uint32_t input_id) {
  if (std::get_if<ynn_node::unary_elementwise>(&node.op) ||
      std::get_if<ynn_node::binary_elementwise>(&node.op)) {
    return true;
  }
  return false;
}

template <typename BroadcastOp>
bool remove_broadcast(ynn_subgraph& subgraph, ynn_node& node,
                      subgraph_analysis& analysis) {
  BroadcastOp* broadcast = std::get_if<BroadcastOp>(&node.op);
  if (broadcast == nullptr) {
    return false;
  }

  const ynn_value& input = subgraph.value(node.inputs[0]);
  ynn_value& output = subgraph.value(node.outputs[0]);

  std::vector<ynn_node*>& consumers = analysis.consumers[node.outputs[0]];
  if (!std::all_of(consumers.begin(), consumers.end(), [&](const ynn_node* i) {
        return can_implicitly_broadcast(*i, output.id);
      })) {
    // One of the consumers can't handle implicit broadcasts.
    return false;
  }

  if (input.is_static()) {
    // If the input is static, all broadcastable dimensions are already implicit
    // broadcasts.
    YNN_LOG_DEBUG() << "Replacing explicit broadcast of static value with "
                       "implicit broadcast";
    broadcast->axes = ynn::axes_set{};
  }

  bool simplified = false;
  for (size_t d = 0; d < broadcast->axes.size(); ++d) {
    if (!broadcast->axes[d]) continue;
    if (d >= input.extents.size() || !input.extents[d].defined() ||
        !output.extents[d].defined()) {
      YNN_LOG_DEBUG() << "Replacing explicit broadcast with implicit broadcast "
                         "in dimension "
                      << d;
      output.extents[d] = {};
      broadcast->axes[d] = false;
      simplified = true;
    }
  }

  if (!output.is_external_output() && !broadcast->axes.any()) {
    // This broadcast is a no-op, replace consumers with our input, and remove
    // the op.
    YNN_LOG_DEBUG() << "Removing broadcast";
    for (ynn_node* i : consumers) {
      for (uint32_t& id : i->inputs) {
        if (id == node.outputs[0]) {
          id = node.inputs[0];
        }
      }
    }
    node.invalidate();
    return true;
  }
  return simplified;
}

bool remove_broadcast(ynn_subgraph& subgraph, ynn_node& node,
                      subgraph_analysis& analysis) {
  return remove_broadcast<ynn_node::broadcast>(subgraph, node, analysis) ||
         remove_broadcast<ynn_node::broadcast_like>(subgraph, node, analysis);
}

}  // namespace

ynn_status ynn_subgraph::fusion() {
  subgraph_analysis analysis(*this);

  for (ynn_node& node : nodes) {
    if (!node.is_valid()) continue;

    false || rewrite_multiply_add(*this, node, analysis) ||
        rewrite_subtract_multiply(*this, node, analysis) ||
        rewrite_convert_to_multiply(*this, node, analysis) ||
        rewrite_clamp(*this, node, analysis) ||
        rewrite_convert_to_quantize(*this, node, analysis) ||
        remove_broadcast(*this, node, analysis);
  }

  return ynn_status_success;
}
