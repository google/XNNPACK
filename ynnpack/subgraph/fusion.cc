// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/fusion.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/subgraph/dot.h"
#include "ynnpack/subgraph/elementwise.h"
#include "ynnpack/subgraph/reduce.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/stencil_copy.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"

subgraph_analysis::subgraph_analysis(ynn_subgraph& subgraph) {
  for (ynn_node& node : subgraph.nodes) {
    for (uint32_t input : node.inputs) {
      consumers[input].push_back(&node);
    }
    assert(producers.find(node.outputs[0]) == producers.end());
    producers[node.outputs[0]] = &node;
  }
}

namespace {

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
          YNN_LOG_DEBUG() << "Rewriting convert to ternary multiply_add";
          ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id,
                              ynn::ternary_op::multiply_add, kernel);
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
        ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id,
                            ynn::ternary_op::subtract_multiply, kernel);
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
                            scale_b_id, node.outputs[0],
                            ynn::ternary_op::multiply, kernel);
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

  const ynn::ternary_op op = output.type == ynn_type_int8
                                 ? ynn::ternary_op::quantize_int8
                                 : ynn::ternary_op::quantize_uint8;
  ynn::ternary_kernel_fn kernel = ynn::get_ternary_kernel(
      op, input.type, subgraph.value(output.scale_id).type,
      subgraph.value(output.zero_point_id).type, output.type);
  if (kernel != nullptr) {
    YNN_LOG_DEBUG() << "Rewriting convert to quantize";
    ynn::define_ternary(subgraph, node, node.inputs[0], output.scale_id,
                        output.zero_point_id, node.outputs[0], op, kernel);
    return true;
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
          ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id,
                              ynn::ternary_op::clamp, kernel);
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

// Rewrite transpose_a(stencil_copy(x)) to stencil_copy(transpose_a(x)).
bool rewrite_transpose_stencil_copy(ynn_subgraph& subgraph, ynn_node& node,
                                    subgraph_analysis& analysis) {
  const ynn_node::transpose_a* transpose_a =
      std::get_if<ynn_node::transpose_a>(&node.op);
  if (transpose_a == nullptr) {
    return false;
  }
  const int tile_k = transpose_a->tile_k;
  const int m_dim = transpose_a->m_dim;

  auto producer_it = analysis.producers.find(node.inputs[0]);
  if (producer_it == analysis.producers.end()) {
    return false;
  }

  ynn_node* stencil_node = producer_it->second;
  const ynn_node::stencil_copy* stencil_copy_ptr =
      std::get_if<ynn_node::stencil_copy>(&stencil_node->op);
  if (stencil_copy_ptr == nullptr) {
    return false;
  }
  ynn_node::stencil_copy stencil_copy = *stencil_copy_ptr;

  const uint32_t y_id = stencil_node->outputs[0];
  if (analysis.consumers[y_id].size() != 1 ||
      subgraph.value(y_id).is_external_output()) {
    return false;
  }

  if (stencil_node->inputs.size() > 1 &&
      stencil_node->inputs[1] != YNN_INVALID_VALUE_ID) {
    // Stencil node has both a buffer and a padding input. If padding is SAME,
    // we cannot rewrite this.
    YNN_LOG_DEBUG() << "Stencil node has both a buffer and a padding input.";
    return false;
  }

  // `stencil_copy` inserts a dimension at each `new_axis` position, so
  // `stencil_copy` caused `m_dim` to increase by 1 for each new dimension
  // before `m_dim`. We restore `m_dim` to its original value.
  int new_m_dim = m_dim;
  for (const ynn_node::stencil_copy::stencil& stencil : stencil_copy.stencils) {
    // We do not support stencils with stride > 1 yet.
    if (stencil.stride != 1) {
      YNN_LOG_DEBUG() << "Stencil node has stride > 1.";
      return false;
    }

    if (stencil.new_axis < m_dim) {
      new_m_dim--;
    } else if (stencil.new_axis == m_dim) {
      // The transposed dimension is one of the new dimensions.
      YNN_LOG_DEBUG() << "Transposed dimension is a new dimension.";

      // We could handle this in some cases, if the extent and dilation of
      // all dimensions is divided evenly by tile_k.
      return false;
    }
  }

  YNN_LOG_DEBUG() << "Rewriting transpose_a(stencil_copy(x)) to "
                     "stencil_copy(transpose_a(x))";

  uint32_t stencil_input_id = stencil_node->inputs[0];
  uint32_t stencil_padding_id = stencil_node->inputs.size() > 1
                                    ? stencil_node->inputs[1]
                                    : YNN_INVALID_VALUE_ID;
  uint32_t stencil_output_id = stencil_node->outputs[0];

  // Replace stencil_copy(x) with transpose_a'(x), reusing the stencil_node's x
  // input and y output.
  ynn::define_transpose_a(subgraph, *stencil_node, tile_k, new_m_dim,
                          stencil_input_id, stencil_output_id);

  uint32_t output_id = node.outputs[0];
  // Replace transpose_a(x) with stencil_copy'(transpose_a'(x)), reusing
  // transpose_a's output.
  ynn::define_stencil_copy(subgraph, node, std::move(stencil_copy),
                           stencil_output_id, stencil_padding_id, &output_id,
                           /*flags=*/0);

  return true;
}

// Rewrites ynn_reduce_sum(x*x) to ynn_reduce_sum_squared(x).
// Specifically:
//   ynn_reduce_sum(fp32 * fp32) -> ynn_reduce_sum_squared(fp32)
//   ynn_reduce_sum(int32 * int32) -> ynn_reduce_sum_squared(int32)
bool rewrite_reduce_sum_of_squared(ynn_subgraph& subgraph, ynn_node& node,
                                   subgraph_analysis& analysis) {
  const ynn_node::reduce* reduce_op = std::get_if<ynn_node::reduce>(&node.op);
  if (reduce_op == nullptr || reduce_op->op != ynn_reduce_sum) {
    return false;
  }

  auto producer = analysis.producers.find(node.inputs[0]);
  if (producer == analysis.producers.end()) {
    return false;
  }

  ynn_node* mul_node = producer->second;
  if (!is_binary_node(*mul_node, ynn_binary_multiply)) {
    return false;
  }

  if (mul_node->inputs[0] != mul_node->inputs[1]) {
    return false;
  }

  uint32_t x_id = mul_node->inputs[0];
  const ynn_value* x = &subgraph.value(x_id);

  if (x->type != ynn_type_fp32 && x->type != ynn_type_int32) {
    return false;
  }

  YNN_LOG_DEBUG() << "Rewriting reduce_sum(x*x) to reduce_sum_squared(x)";
  ynn::define_reduce(subgraph, node, ynn_reduce_sum_squared, reduce_op->k_dims,
                     x_id, node.inputs[1], node.outputs[0],
                     reduce_op->keep_dims);

  return true;
}

// Rewrites ynn_reduce_sum of convert to ynn_reduce_sum of convert's input.
// Specifically:
//   ynn_reduce_sum(f32(x_fp16)) -> ynn_reduce_sum(x_fp16)
//   ynn_reduce_sum(f32(x_bf16)) -> ynn_reduce_sum(x_bf16)
//   ynn_reduce_sum(int32(x_int8)) -> ynn_reduce_sum(x_int8)
bool rewrite_reduce_sum_convert(ynn_subgraph& subgraph, ynn_node& node,
                                subgraph_analysis& analysis) {
  const ynn_node::reduce* reduce_op = std::get_if<ynn_node::reduce>(&node.op);
  if (reduce_op == nullptr || reduce_op->op != ynn_reduce_sum) {
    return false;
  }

  auto producer = analysis.producers.find(node.inputs[0]);
  if (producer == analysis.producers.end()) {
    return false;
  }

  ynn_node* convert_node = producer->second;
  if (!is_unary_node(*convert_node, ynn_unary_convert)) {
    return false;
  }

  const ynn_value& x = subgraph.value(convert_node->inputs[0]);
  const ynn_value& converted_x = subgraph.value(convert_node->outputs[0]);

  if (converted_x.type == ynn_type_fp32) {
    if (!ynn::type_is_floating_point(x.type)) {
      return false;
    }
  } else if (converted_x.type == ynn_type_int32) {
    if (!ynn::type_is_integral(x.type)) {
      return false;
    }
    if (x.scale_id != YNN_INVALID_VALUE_ID ||
        x.zero_point_id != YNN_INVALID_VALUE_ID) {
      return false;
    }
  } else {
    return false;
  }

  YNN_LOG_DEBUG() << "Rewriting reduce_sum(convert(x)) to reduce_sum(x)";
  ynn::define_reduce(subgraph, node, ynn_reduce_sum, reduce_op->k_dims, x.id,
                     node.inputs[1], node.outputs[0], reduce_op->keep_dims);
  return true;
}

// Rewrites ynn_reduce_sum_squared of convert to ynn_reduce_sum_squared of
// convert's input. Specifically:
//   ynn_reduce_sum_squared(f32(x_fp16)) -> ynn_reduce_sum_squared(x_fp16)
//   ynn_reduce_sum_squared(f32(x_bf16)) -> ynn_reduce_sum_squared(x_bf16)
//   ynn_reduce_sum_squared(int32(x_int8)) -> ynn_reduce_sum_squared(x_int8)
bool rewrite_reduce_sum_squared_convert(ynn_subgraph& subgraph, ynn_node& node,
                                        subgraph_analysis& analysis) {
  const ynn_node::reduce* reduce_op = std::get_if<ynn_node::reduce>(&node.op);
  if (reduce_op == nullptr || reduce_op->op != ynn_reduce_sum_squared) {
    return false;
  }

  auto producer = analysis.producers.find(node.inputs[0]);
  if (producer == analysis.producers.end()) {
    return false;
  }

  ynn_node* convert_node = producer->second;
  if (!is_unary_node(*convert_node, ynn_unary_convert)) {
    return false;
  }

  const ynn_value& x = subgraph.value(convert_node->inputs[0]);
  const ynn_value& converted_x = subgraph.value(convert_node->outputs[0]);

  if (converted_x.type == ynn_type_fp32) {
    if (!ynn::type_is_floating_point(x.type)) {
      return false;
    }
  } else if (converted_x.type == ynn_type_int32) {
    if (!ynn::type_is_integral(x.type)) {
      return false;
    }
    if (x.scale_id != YNN_INVALID_VALUE_ID ||
        x.zero_point_id != YNN_INVALID_VALUE_ID) {
      return false;
    }
  } else {
    return false;
  }

  YNN_LOG_DEBUG() << "Rewriting reduce_sum_squared(convert(x)) to "
                     "reduce_sum_squared(x)";
  ynn::define_reduce(subgraph, node, ynn_reduce_sum_squared, reduce_op->k_dims,
                     x.id, node.inputs[1], node.outputs[0],
                     reduce_op->keep_dims);
  return true;
}

// Returns the set of inputs to `n` that are not static.
std::unordered_set<uint32_t> get_variable_inputs(const ynn_subgraph& subgraph,
                                                 const ynn_node& n) {
  std::unordered_set<uint32_t> inputs;
  if (std::get_if<ynn_node::unary_elementwise>(&n.op)) {
    inputs.insert(n.inputs[0]);
  } else if (std::get_if<ynn_node::binary_elementwise>(&n.op)) {
    for (uint32_t id : n.inputs) {
      if (!subgraph.value(id).is_static()) {
        inputs.insert(id);
      }
    }
  } else if (std::get_if<ynn_node::ternary_elementwise>(&n.op)) {
    for (uint32_t id : n.inputs) {
      if (!subgraph.value(id).is_static()) {
        inputs.insert(id);
      }
    }
  }
  return inputs;
}

bool should_lut_single_node(const ynn_node& node) {
  bool supported = false;
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
        supported = true;
        break;
      default:
        break;
    }
  } else if (auto binary =
                 std::get_if<ynn_node::binary_elementwise>(&node.op)) {
    switch (binary->op) {
      case ynn_binary_leaky_relu:
      case ynn_binary_squared_difference:
        supported = true;
        break;
      default:
        break;
    }
  }
  return supported;
}

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
    if (candidate.size > best_candidate.size) {
      best_candidate = candidate;
    }
  }

  if (best_candidate.size == 0) {
    return false;
  }

  // If candidate is a single node, check if it is worth replacing with a lut.
  if (best_candidate.size == 1) {
    const ynn_node& node = **best_candidate.nodes.begin();
    if (!should_lut_single_node(node)) {
      return false;
    }
  }

  YNN_LOG_DEBUG() << "Found candidate of size " << best_candidate.size;

  // 1. Clone the subgraph in `candidate` by using clone_subgraph_subset() in
  // utils.h.
  uint32_t lut_input_id = YNN_INVALID_VALUE_ID;
  uint32_t lut_output_id = YNN_INVALID_VALUE_ID;
  std::unique_ptr<ynn_subgraph> lut_subgraph = ynn::clone_subgraph_subset(
      subgraph, best_candidate.input_id, best_candidate.output_id, lut_input_id,
      lut_output_id);
  if (!lut_subgraph) {
    return false;
  }

  // 2. Create a lut table with the same type as the output type.
  // 3. Modify the cloned subgraph to have an input size that covers the entire
  // range of the input type.
  const ynn_value& input_value = subgraph.value(best_candidate.input_id);
  const size_t range = 256;

  for (ynn_value& value : lut_subgraph->values) {
    if (value.is_valid() && !value.is_static()) {
      value.extents.resize(1);
      value.extents[0] = static_cast<slinky::index_t>(range);
      value.data = nullptr;
    }
  }

  for (ynn_node& node : lut_subgraph->nodes) {
    node.checks.clear();
  }

  std::vector<int8_t> input_data(range);
  if (input_value.type == ynn_type_int8) {
    for (int i = 0; i < range; ++i) {
      input_data[i] = static_cast<int8_t>(i - 128);
    }
  } else if (input_value.type == ynn_type_uint8) {
    for (int i = 0; i < range; ++i) {
      input_data[i] = static_cast<int8_t>(i);
    }
  } else {
    return false;
  }

  // 4. Run the cloned subgraph.
  ynn_runtime runtime(*lut_subgraph, nullptr, 0);
  if (runtime.build() != ynn_status_success) {
    return false;
  }

  size_t input_dims = range;
  if (ynn_set_external_value_shape(&runtime, lut_input_id, 1, &input_dims) !=
      ynn_status_success) {
    return false;
  }
  if (ynn_set_external_value_data(&runtime, lut_input_id, input_data.data()) !=
      ynn_status_success) {
    return false;
  }

  const ynn_value& output_value = subgraph.value(best_candidate.output_id);
  std::vector<uint8_t> output_data(range *
                                   ynn::type_size_bytes(output_value.type));
  if (ynn_set_external_value_data(&runtime, lut_output_id,
                                  output_data.data()) != ynn_status_success) {
    return false;
  }

  if (runtime.reshape() != ynn_status_success ||
      runtime.setup() != ynn_status_success ||
      runtime.invoke() != ynn_status_success) {
    return false;
  }

  // 5. Create a new LUT node using ynn::define_lut().
  uint32_t lut_id = YNN_INVALID_VALUE_ID;
  size_t lut_dims[] = {range};
  if (ynn_define_tensor_value(&subgraph, output_value.type, 1, lut_dims,
                              output_data.data(), YNN_INVALID_VALUE_ID,
                              YNN_INVALID_VALUE_ID, YNN_VALUE_FLAG_COPY_DATA,
                              &lut_id) != ynn_status_success) {
    return false;
  }

  // 6. Replace and invalidate all nodes in `candidate` with the new LUT node.
  ynn_node* output_node = analysis.producers[best_candidate.output_id];
  for (const ynn_node* node : best_candidate.nodes) {
    if (node != output_node) {
      const_cast<ynn_node*>(node)->invalidate();
    }
  }

  ynn::define_lut(subgraph, *output_node, best_candidate.input_id, lut_id,
                  best_candidate.output_id);
  return true;
}

}  // namespace

subgraph_candidate find_subgraph_for_unary_lut(ynn_subgraph& subgraph,
                                               ynn_node& node,
                                               subgraph_analysis& analysis) {
  subgraph_candidate candidate;
  if (node.outputs.empty()) {
    return candidate;
  }
  const ynn_value& output = subgraph.value(node.outputs[0]);
  if (output.type != ynn_type_int8 && output.type != ynn_type_uint8) {
    return candidate;
  }
  std::unordered_set<uint32_t> inputs = get_variable_inputs(subgraph, node);
  if (inputs.empty()) {
    return candidate;
  }

  std::unordered_set<uint32_t> frontier(inputs.begin(), inputs.end());
  candidate.size = 1;
  candidate.output_id = node.outputs[0];
  candidate.nodes.insert(&node);

  subgraph_candidate best_candidate;
  if (frontier.size() == 1) {
    uint32_t in_id = *frontier.begin();
    if (subgraph.value(in_id).type == ynn_type_int8 ||
        subgraph.value(in_id).type == ynn_type_uint8) {
      best_candidate = candidate;
      best_candidate.input_id = in_id;
    }
  }

  while (!frontier.empty()) {
    uint32_t expand_id = YNN_INVALID_VALUE_ID;
    ynn_node* producer = nullptr;

    // Find a value in the frontier that is ready to be expanded (all consumers
    // are in the subgraph).
    for (uint32_t id : frontier) {
      bool all_consumers_in_subgraph = true;
      for (const ynn_node* consumer : analysis.consumers[id]) {
        if (candidate.nodes.find(const_cast<ynn_node*>(consumer)) ==
            candidate.nodes.end()) {
          all_consumers_in_subgraph = false;
          break;
        }
      }

      if (all_consumers_in_subgraph) {
        auto producer_it = analysis.producers.find(id);
        if (producer_it != analysis.producers.end()) {
          std::unordered_set<uint32_t> next_inputs =
              get_variable_inputs(subgraph, *producer_it->second);
          if (!next_inputs.empty()) {
            expand_id = id;
            producer = producer_it->second;
            break;
          }
        }
      }
    }

    if (expand_id == YNN_INVALID_VALUE_ID) {
      break;
    }

    frontier.erase(expand_id);
    candidate.values.insert(expand_id);
    candidate.nodes.insert(producer);
    candidate.size++;

    std::unordered_set<uint32_t> producer_inputs =
        get_variable_inputs(subgraph, *producer);
    for (uint32_t id : producer_inputs) {
      if (candidate.values.find(id) == candidate.values.end()) {
        frontier.insert(id);
      }
    }

    if (frontier.size() == 1) {
      uint32_t in_id = *frontier.begin();
      if (subgraph.value(in_id).type == ynn_type_int8 ||
          subgraph.value(in_id).type == ynn_type_uint8) {
        best_candidate = candidate;
        best_candidate.input_id = in_id;
      }
    }
  }

  return best_candidate;
}

ynn_status ynn_subgraph::fusion() {
  // Fuse graph as much as possible before unary LUT optimization.
  bool changed;
  do {
    subgraph_analysis analysis(*this);
    changed = false;
    for (ynn_node& node : nodes) {
      if (!node.is_valid()) continue;

      changed = changed || rewrite_multiply_add(*this, node, analysis) ||
                rewrite_subtract_multiply(*this, node, analysis) ||
                rewrite_convert_to_multiply(*this, node, analysis) ||
                rewrite_clamp(*this, node, analysis) ||
                rewrite_convert_to_quantize(*this, node, analysis) ||
                remove_broadcast(*this, node, analysis) ||
                rewrite_transpose_stencil_copy(*this, node, analysis) ||
                rewrite_reduce_sum_of_squared(*this, node, analysis) ||
                rewrite_reduce_sum_convert(*this, node, analysis) ||
                rewrite_reduce_sum_squared_convert(*this, node, analysis);
    }
  } while (changed);

  do {
    subgraph_analysis analysis(*this);
    changed = rewrite_subgraph_for_unary_lut(*this, analysis);
  } while (changed);

  return ynn_status_success;
}
