// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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
#include "ynnpack/subgraph/fusion_lut.h"
#include "ynnpack/subgraph/fusion_types.h"
#include "ynnpack/subgraph/reduce.h"
#include "ynnpack/subgraph/stencil_copy.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

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
    ynn_node* producer = analysis.producer_of(node.inputs[i]);
    if (producer && is_binary_node(*producer, ynn_binary_multiply) &&
        analysis.consumers[producer->outputs[0]].size() == 1) {
      // This is a multiply-add. Do we have a kernel for this case?
      const ynn_value& a = subgraph.value(producer->inputs[0]);
      const ynn_value& b = subgraph.value(producer->inputs[1]);
      const ynn_value& c = subgraph.value(node.inputs[1 - i]);
      const ynn_value& x = subgraph.value(node.outputs[0]);
      const ynn::ternary_kernel_fn kernel = ynn::get_ternary_kernel(
          ynn::ternary_op::multiply_add, a.type, b.type, c.type, x.type);
      if (kernel != nullptr) {
        // Yes we do. Rewrite this to a multiply-add.
        YNN_LOG_DEBUG() << "Rewriting multiply and add to ternary multiply_add";
        ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id,
                            ynn::ternary_op::multiply_add, kernel);
        return true;
      }
    }
  }
  return false;
}

// Rewrite multiply(multiply(a, b), c) to ternary multiply.
bool rewrite_multiply_multiply(ynn_subgraph& subgraph, ynn_node& node,
                               subgraph_analysis& analysis) {
  if (!is_binary_node(node, ynn_binary_multiply)) {
    return false;
  }

  for (int i : {0, 1}) {
    ynn_node* producer = analysis.producer_of(node.inputs[i]);
    if (producer && is_binary_node(*producer, ynn_binary_multiply) &&
        analysis.consumers[producer->outputs[0]].size() == 1) {
      // This is a multiply of a multiply. Do we have a kernel for this case?
      const ynn_value& a = subgraph.value(node.inputs[1 - i]);
      const ynn_value& b = subgraph.value(producer->inputs[0]);
      const ynn_value& c = subgraph.value(producer->inputs[1]);
      const ynn_value& x = subgraph.value(node.outputs[0]);
      const ynn::ternary_kernel_fn kernel = ynn::get_ternary_kernel(
          ynn::ternary_op::multiply, a.type, b.type, c.type, x.type);
      if (kernel != nullptr) {
        // Yes we do. Rewrite this to a multiply-add.
        YNN_LOG_DEBUG() <<
            "Rewriting multiply and multiply to ternary multiply";
        ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id,
                            ynn::ternary_op::multiply, kernel);
        return true;
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

  ynn_node* producer = analysis.producer_of(mul_id);
  if (producer && is_binary_node(*producer, ynn_binary_multiply) &&
      analysis.consumers[producer->outputs[0]].size() == 1) {
    // This is a subtract_multiply. Do we have a kernel for this case?
    const ynn_value& b = subgraph.value(producer->inputs[0]);
    const ynn_value& c = subgraph.value(producer->inputs[1]);
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
  return false;
}

// Rewrite min(max(a, b), c) to clamp(a, b, c)
bool rewrite_clamp(ynn_subgraph& subgraph, ynn_node& node,
                   subgraph_analysis& analysis) {
  if (!is_binary_node(node, ynn_binary_min)) {
    return false;
  }

  for (int i : {0, 1}) {
    ynn_node* producer = analysis.producer_of(node.inputs[i]);
    if (producer && is_binary_node(*producer, ynn_binary_max) &&
        analysis.consumers[producer->outputs[0]].size() == 1) {
      // This is a clamp. Do we have a kernel for this case?
      const ynn_value& a = subgraph.value(producer->inputs[0]);
      const ynn_value& b = subgraph.value(producer->inputs[1]);
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

bool is_static_zero(const ynn_value& value) {
  if (!value.is_static()) {
    return false;
  }
  const char* bytes = static_cast<const char*>(value.data->base);
  size_t size = value.data->size_bytes();
  return std::all_of(bytes, bytes + size, [](char c) { return c == 0; });
}

bool is_copy_node(const ynn_node& node, const ynn_subgraph& subgraph) {
  if (std::holds_alternative<ynn_node::static_pad>(node.op) ||
      std::holds_alternative<ynn_node::stencil_copy>(node.op)) {
    uint32_t padding_id =
        node.inputs.size() > 1 ? node.inputs[1] : YNN_INVALID_VALUE_ID;
    return padding_id == YNN_INVALID_VALUE_ID ||
           is_static_zero(subgraph.value(padding_id));
  }
  return false;
}

// Walks up the producer chain from `start_id`, skipping copy nodes
// whose outputs have a single non-external consumer. Returns a pointer to the
// first non-copy producer node, or nullptr if the chain cannot be
// followed.
ynn_node* find_non_copy_producer(const ynn_subgraph& subgraph,
                                 subgraph_analysis& analysis,
                                 uint32_t start_id) {
  uint32_t search_id = start_id;
  while (true) {
    ynn_node* producer = analysis.producer_of(search_id);
    if (!producer || !is_copy_node(*producer, subgraph)) {
      return producer;
    }

    // Check that the intermediate output has exactly one consumer
    // and is not an external output, so it's safe to modify the chain.
    uint32_t output_id = producer->outputs[0];
    if (analysis.consumers[output_id].size() != 1 ||
        subgraph.value(output_id).is_external_output()) {
      return nullptr;
    }
    search_id = producer->inputs[0];
  }
  return nullptr;
}

// Rewrites ynn_reduce_sum(x*x) to ynn_reduce_sum_squared(x).
// Also handles the windowed case where copy shape ops (stencil_copy,
// static_pad) appear between the multiply and the reduce:
//   reduce_sum(stencil_copy(static_pad(x*x))) ->
//     reduce_sum_squared(stencil_copy(static_pad(x)))
// Specifically:
//   ynn_reduce_sum(fp32 * fp32) -> ynn_reduce_sum_squared(fp32)
//   ynn_reduce_sum(int32 * int32) -> ynn_reduce_sum_squared(int32)
bool rewrite_reduce_sum_of_squared(ynn_subgraph& subgraph, ynn_node& node,
                                   subgraph_analysis& analysis) {
  const ynn_node::reduce* reduce_op = std::get_if<ynn_node::reduce>(&node.op);
  if (reduce_op == nullptr || reduce_op->op != ynn_reduce_sum) {
    return false;
  }

  ynn_node* mul_node =
      find_non_copy_producer(subgraph, analysis, node.inputs[0]);
  if (mul_node == nullptr || !is_binary_node(*mul_node, ynn_binary_multiply)) {
    return false;
  }

  if (mul_node->inputs[0] != mul_node->inputs[1]) {
    return false;
  }

  uint32_t mul_output_id = mul_node->outputs[0];
  if (analysis.consumers[mul_output_id].size() != 1 ||
      subgraph.value(mul_output_id).is_external_output()) {
    return false;
  }

  uint32_t x_id = mul_node->inputs[0];
  const ynn_value* x = &subgraph.value(x_id);

  if (x->type != ynn_type_fp32 && x->type != ynn_type_int32) {
    return false;
  }

  // Rewire multiply's consumers to use x directly. In the simple case
  // (reduce directly consumes multiply) this updates node.inputs[0] to x_id.
  // In the windowed case (copy ops in between) this splices out the
  // multiply from the chain.
  for (ynn_node* consumer : analysis.consumers[mul_output_id]) {
    for (uint32_t& inp : consumer->inputs) {
      if (inp == mul_output_id) {
        inp = x_id;
      }
    }
  }

  YNN_LOG_DEBUG() << "Rewriting reduce_sum(x*x) to reduce_sum_squared(x)";
  ynn::define_reduce(subgraph, node, ynn_reduce_sum_squared, reduce_op->k_dims,
                     node.inputs[0], node.inputs[1], node.outputs[0],
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

  ynn_node* convert = analysis.producer_of(node.inputs[0]);
  if (!convert || !is_unary_node(*convert, ynn_unary_convert)) {
    return false;
  }

  const ynn_value& x = subgraph.value(convert->inputs[0]);
  const ynn_value& converted_x = subgraph.value(convert->outputs[0]);

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

  ynn_node* convert = analysis.producer_of(node.inputs[0]);
  if (!convert || !is_unary_node(*convert, ynn_unary_convert)) {
    return false;
  }

  const ynn_value& x = subgraph.value(convert->inputs[0]);
  const ynn_value& converted_x = subgraph.value(convert->outputs[0]);

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

}  // namespace

}  // namespace ynn

ynn_status ynn_subgraph::fusion() {
  // Fuse graph as much as possible before unary LUT optimization.
  bool changed;
  do {
    subgraph_analysis analysis(*this);
    changed = false;
    for (ynn_node& node : nodes) {
      if (!node.is_valid()) continue;

      changed =
          changed || ynn::rewrite_multiply_add(*this, node, analysis) ||
          ynn::rewrite_multiply_multiply(*this, node, analysis) ||
          ynn::rewrite_subtract_multiply(*this, node, analysis) ||
          ynn::rewrite_clamp(*this, node, analysis) ||
          ynn::remove_broadcast(*this, node, analysis) ||
          ynn::rewrite_transpose_stencil_copy(*this, node, analysis) ||
          ynn::rewrite_reduce_sum_of_squared(*this, node, analysis) ||
          ynn::rewrite_reduce_sum_convert(*this, node, analysis) ||
          ynn::rewrite_reduce_sum_squared_convert(*this, node, analysis);
    }
  } while (changed);

  do {
    subgraph_analysis analysis(*this);
    changed = ynn::rewrite_subgraph_for_unary_lut(*this, analysis);
  } while (changed);

  return ynn_status_success;
}
