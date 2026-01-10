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
#include <numeric>
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

namespace {

struct subgraph_analysis {
  std::map<uint32_t, ynn_node*> producers;
  std::map<uint32_t, std::vector<ynn_node*>> consumers;

  explicit subgraph_analysis(ynn_subgraph& subgraph) {
    for (ynn_node& node : subgraph.nodes) {
      if (!node.is_valid()) continue;
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

  auto producer_it = analysis.producers.find(node.inputs[0]);
  if (producer_it == analysis.producers.end()) {
    return false;
  }

  ynn_node* stencil_node = producer_it->second;
  const ynn_node::stencil_copy* stencil_copy =
      std::get_if<ynn_node::stencil_copy>(&stencil_node->op);
  if (stencil_copy == nullptr) {
    return false;
  }

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

  for (const ynn_node::stencil_copy::stencil& stencil :
       stencil_copy->stencils) {
    // We do not support stencils with stride > 1 yet.
    if (stencil.stride != 1) {
      YNN_LOG_DEBUG() << "Stencil node has stride > 1.";
      return false;
    }
  }

  YNN_LOG_DEBUG() << "Rewriting transpose_a(stencil_copy(x)) to "
                     "stencil_copy(transpose_a(x))";

  ynn_node::stencil_copy stencil_op_data = *stencil_copy;
  uint32_t stencil_input_id = stencil_node->inputs[0];
  uint32_t stencil_padding_id = stencil_node->inputs.size() > 1
                                    ? stencil_node->inputs[1]
                                    : YNN_INVALID_VALUE_ID;
  uint32_t stencil_output_id = stencil_node->outputs[0];

  // `stencil_copy` inserts a dimension at each `new_axis` position, so
  // `stencil_copy` caused `m_dim` to increase by 1 for each new dimension
  // before `m_dim`. We restore `m_dim` to its original value.
  int new_m_dim = transpose_a->m_dim -
                  std::count_if(stencil_copy->stencils.begin(),
                                stencil_copy->stencils.end(),
                                [m_dim = transpose_a->m_dim](
                                    const ynn_node::stencil_copy::stencil& i) {
                                  return i.new_axis < m_dim;
                                });

  // Replace stencil_copy(x) with transpose_a'(x), reusing the stencil_node's x
  // input and y output.
  ynn::define_transpose_a(subgraph, *stencil_node, transpose_a->tile_k,
                          new_m_dim, stencil_input_id, stencil_output_id);

  uint32_t output_id = node.outputs[0];
  // Replace transpose_a(x) with stencil_copy'(transpose_a'(x)), reusing
  // transpose_a's output.
  ynn::define_stencil_copy(subgraph, node, std::move(stencil_op_data),
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

bool is_quantized_8bit(const ynn_value& v) {
  return v.type == ynn_type_int8 || v.type == ynn_type_uint8;
}

bool is_pure_unary_elementwise(const ynn_node& node,
                               const ynn_subgraph& subgraph) {
  if (std::get_if<ynn_node::unary_elementwise>(&node.op)) return true;

  if (std::get_if<ynn_node::binary_elementwise>(&node.op)) {
    const ynn_value& a = subgraph.value(node.inputs[0]);
    const ynn_value& b = subgraph.value(node.inputs[1]);
    if (a.is_static() || b.is_static()) return true;
  }

  if (const auto* ternary =
          std::get_if<ynn_node::ternary_elementwise>(&node.op)) {
    if (ternary->op == ynn::ternary_op::clamp ||
        ternary->op == ynn::ternary_op::quantize_int8 ||
        ternary->op == ynn::ternary_op::quantize_uint8) {
      const ynn_value& input1 = subgraph.value(node.inputs[1]);
      const ynn_value& input2 = subgraph.value(node.inputs[2]);
      if (input1.is_static() && input2.is_static()) return true;
    }
  }

  return false;
}

bool generate_lut(const ynn_subgraph& subgraph,
                  const std::vector<ynn_node*>& chain, uint8_t* lut_data) {
  uint32_t chain_input_id = chain.front()->inputs[0];
  uint32_t chain_output_id = chain.back()->outputs[0];

  ynn_subgraph temp_subgraph(1, 0);
  ynn_value& input_val = temp_subgraph.values[0];
  input_val.flags |= YNN_VALUE_FLAG_EXTERNAL_INPUT;
  input_val.type = subgraph.value(chain_input_id).type;
  input_val.extents.resize(1);

  std::map<uint32_t, uint32_t> id_map;

  std::function<uint32_t(uint32_t)> copy_value =
      [&](uint32_t old_id) -> uint32_t {
    if (old_id == YNN_INVALID_VALUE_ID) return YNN_INVALID_VALUE_ID;
    if (id_map.count(old_id)) return id_map[old_id];

    const ynn_value& old_val = subgraph.value(old_id);
    if (old_id == chain_input_id) {
      id_map[old_id] = 0;
      temp_subgraph.values[0].type = old_val.type;
      temp_subgraph.values[0].zero_point_id = copy_value(old_val.zero_point_id);
      temp_subgraph.values[0].scale_id = copy_value(old_val.scale_id);
      return 0;
    }

    ynn_value& new_val = temp_subgraph.new_internal_value(old_val.type);
    id_map[old_id] = new_val.id;

    if (old_val.is_static()) {
      new_val.data = old_val.data;
      new_val.extents = old_val.extents;
    } else {
      const ynn_node* producer = subgraph.get_producer(old_id);
      if (producer) {
        bool in_chain =
            std::find(chain.begin(), chain.end(), producer) != chain.end();
        if (!in_chain) {
          ynn_node new_node = *producer;
          for (uint32_t& id : new_node.inputs) id = copy_value(id);
          for (uint32_t& id : new_node.outputs) id = copy_value(id);
          new_node.checks.clear();

          if (const auto* op = std::get_if<ynn_node::opaque>(&new_node.op);
              op && op->name && std::string(op->name) == "make_unary_params") {
            for (int i = 0; i < 4; ++i) {
              if (new_node.inputs.size() > i &&
                  new_node.inputs[i] != YNN_INVALID_VALUE_ID) {
                uint32_t input_id = new_node.inputs[i];
                if (temp_subgraph.is_valid_value(input_id)) {
                  ynn_type type = temp_subgraph.value(input_id).type;
                  bool expected_int = (i % 2 == 1);
                  if (expected_int && type != ynn_type_int32) {
                    new_node.inputs[i] = YNN_INVALID_VALUE_ID;
                  } else if (!expected_int && type != ynn_type_fp32) {
                    new_node.inputs[i] = YNN_INVALID_VALUE_ID;
                  }
                }
              }
            }
          }

          temp_subgraph.add_node(std::move(new_node));
        }
      }
    }

    new_val.zero_point_id = copy_value(old_val.zero_point_id);
    new_val.scale_id = copy_value(old_val.scale_id);

    if (!ynn::type_is_integral(new_val.type)) {
      new_val.zero_point_id = YNN_INVALID_VALUE_ID;
      new_val.scale_id = YNN_INVALID_VALUE_ID;
    }

    return new_val.id;
  };

  copy_value(chain_input_id);

  for (ynn_node* node : chain) {
    ynn_node new_node = *node;
    for (uint32_t& id : new_node.inputs) id = copy_value(id);
    for (uint32_t& id : new_node.outputs) id = copy_value(id);
    new_node.checks.clear();
    temp_subgraph.add_node(std::move(new_node));
  }

  uint32_t temp_output_id = id_map[chain_output_id];
  temp_subgraph.values[temp_output_id].flags |= YNN_VALUE_FLAG_EXTERNAL_OUTPUT;

  ynn_runtime runtime(temp_subgraph, nullptr, 0);
  if (runtime.build() != ynn_status_success) return false;

  size_t input_dims = 256;
  if (ynn_set_external_value_shape(&runtime, 0, 1, &input_dims) !=
      ynn_status_success)
    return false;
  if (runtime.reshape() != ynn_status_success) return false;

  std::vector<uint8_t> input_data(256);
  std::iota(input_data.begin(), input_data.end(), 0);
  if (ynn_set_external_value_data(&runtime, 0, input_data.data()) !=
      ynn_status_success)
    return false;

  if (ynn_set_external_value_data(&runtime, temp_output_id, lut_data) !=
      ynn_status_success)
    return false;

  if (runtime.setup() != ynn_status_success) return false;
  if (runtime.invoke() != ynn_status_success) return false;

  return true;
}

bool rewrite_unary_quantized_to_lut(ynn_subgraph& subgraph, ynn_node& node,
                                    subgraph_analysis& analysis) {
  if (!is_pure_unary_elementwise(node, subgraph)) return false;

  uint32_t input_id = node.inputs[0];
  const ynn_value& input = subgraph.value(input_id);
  if (!is_quantized_8bit(input)) return false;

  std::vector<ynn_node*> chain;
  chain.push_back(&node);

  ynn_node* current_node = &node;
  while (true) {
    uint32_t output_id = current_node->outputs[0];
    const auto& consumers = analysis.consumers[output_id];
    if (consumers.size() != 1 || subgraph.value(output_id).is_external_output())
      break;
    ynn_node* consumer = consumers[0];

    if (!is_pure_unary_elementwise(*consumer, subgraph)) break;

    chain.push_back(consumer);
    current_node = consumer;
  }

  uint32_t final_output_id = chain.back()->outputs[0];
  const ynn_value& final_output = subgraph.value(final_output_id);
  if (!is_quantized_8bit(final_output)) return false;

  if (chain.size() == 1 && is_unary_node(node, ynn_unary_convert)) {
    return false;
  }

  // Add heuristic to only rewrite if the chain is sufficiently large.
  if (chain.size() < 3) {
    return false;
  }

  std::vector<uint8_t> lut_data(256);
  if (!generate_lut(subgraph, chain, lut_data.data())) return false;

  size_t dim = 256;
  float float_lut[256];
  for (int i = 0; i < 256; ++i) float_lut[i] = (float)lut_data[i];

  // We need to define the tensor directly because it's uint8/int8 not float.
  // get_static_value_id expects float and quantizes it. But we already have the
  // bytes.

  uint32_t lut_table_id = YNN_INVALID_VALUE_ID;
  ynn_define_tensor_value(&subgraph, final_output.type, 1, &dim,
                          lut_data.data(), YNN_INVALID_VALUE_ID,
                          YNN_INVALID_VALUE_ID, YNN_VALUE_FLAG_COPY_DATA,
                          &lut_table_id);

  YNN_LOG_DEBUG() << "Rewriting chain of " << chain.size()
                  << " unary ops to LUT";

  // Replace chain with LUT node.
  // We need to fix the input of the first node if we reuse the output ID?
  // ynn_define_lut takes output_id pointer. If it points to existing id, it
  // reuses it. We want to reuse final_output_id so consumers don't need update.

  ynn_define_lut(&subgraph, input_id, lut_table_id, &final_output_id, 0);

  // Invalidate old nodes.
  for (auto* n : chain) n->invalidate();

  return true;
}

}  // namespace

ynn_status ynn_subgraph::fusion() {
  bool changed;
  do {
    subgraph_analysis analysis(*this);
    changed = false;
    for (ynn_node& node : nodes) {
      if (!node.is_valid()) continue;

      bool node_changed =
          // This pass must always run before rewrite_convert_to_quantize.
          rewrite_unary_quantized_to_lut(*this, node, analysis) ||
          rewrite_multiply_add(*this, node, analysis) ||
          rewrite_subtract_multiply(*this, node, analysis) ||
          rewrite_convert_to_multiply(*this, node, analysis) ||
          rewrite_clamp(*this, node, analysis) ||
          rewrite_convert_to_quantize(*this, node, analysis) ||
          remove_broadcast(*this, node, analysis) ||
          rewrite_transpose_stencil_copy(*this, node, analysis) ||
          rewrite_reduce_sum_of_squared(*this, node, analysis) ||
          rewrite_reduce_sum_convert(*this, node, analysis) ||
          rewrite_reduce_sum_squared_convert(*this, node, analysis);

      if (node_changed) {
        changed = true;
        break;
      }
    }
  } while (changed);

  return ynn_status_success;
}
