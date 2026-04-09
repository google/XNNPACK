// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/dequantize_dot/dequantize_dot.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/kernels/unary/unary.h"
#include "ynnpack/subgraph/copy.h"
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
bool is_ternary_node(const ynn_node& node, ternary_op op) {
  const ynn_node::ternary_elementwise* ternary =
      std::get_if<ynn_node::ternary_elementwise>(&node.op);
  return ternary && ternary->op == op;
}

bool is_square_node(const ynn_node& node) {
  if (is_unary_node(node, ynn_unary_square)) {
    return true;
  } else if (is_binary_node(node, ynn_binary_multiply) &&
             node.inputs[0] == node.inputs[1]) {
    return true;
  }
  return false;
}

struct scalar_arithmetic {
  // This represents an operation a*x + b
  uint32_t x_id = YNN_INVALID_VALUE_ID;
  float a = 1.0f;
  float b = 0.0f;
};

// If `node` is a linear expression of scalar constants, returns
// `scalar_arithmetic` describing the operation.
std::optional<scalar_arithmetic> is_scalar_arithmetic(
    const ynn_subgraph& subgraph, const ynn_node& node) {
  if (is_unary_node(node, ynn_unary_negate)) {
    return scalar_arithmetic{node.inputs[0], -1.0f, 0.0f};
  }
  const ynn_node::binary_elementwise* binary =
      std::get_if<ynn_node::binary_elementwise>(&node.op);
  if (binary == nullptr) return std::nullopt;

  if (const auto b = subgraph.value(node.inputs[1]).as_scalar_float()) {
    switch (binary->op) {
      case ynn_binary_add:
        return scalar_arithmetic{node.inputs[0], 1.0f, *b};
      case ynn_binary_subtract:
        return scalar_arithmetic{node.inputs[0], 1.0f, -*b};
      case ynn_binary_multiply:
        return scalar_arithmetic{node.inputs[0], *b, 0.0f};
      case ynn_binary_divide:
        return scalar_arithmetic{node.inputs[0], 1.0f / *b, 0.0f};
      default:
        return std::nullopt;
    }
  }

  if (const auto a = subgraph.value(node.inputs[0]).as_scalar_float()) {
    switch (binary->op) {
      case ynn_binary_add:
        return scalar_arithmetic{node.inputs[1], 1.0f, *a};
      case ynn_binary_subtract:
        return scalar_arithmetic{node.inputs[1], -1.0f, *a};
      case ynn_binary_multiply:
        return scalar_arithmetic{node.inputs[1], *a, 0.0f};
      default:
        return std::nullopt;
    }
  }
  return std::nullopt;
}

// Replace `from_id` with `to_id` in the subgraph, assuming `node` is the
// producer of `from_id`. If `from_id` is an external output, replaces `node`
// with a copy node from `to_id` to `from_id`, otherwise invalidates the node.
// Returns true if `node` was made obsolete.
bool replace_uses(subgraph_analysis& analysis, ynn_subgraph& subgraph,
                  ynn_node& node, uint32_t from_id, uint32_t to_id) {
  if (subgraph.value(from_id).is_external_output()) {
    // The output is external. Can we rewrite the producer of the new value to
    // produce this value instead?
    if (!subgraph.value(to_id).is_external()) {
      if (ynn_node* producer = analysis.producer_of(to_id)) {
        // Redefine the producer of `to_id` to produce `from_id` instead.
        for (uint32_t& o : producer->outputs) {
          if (o == to_id) o = from_id;
        }
        // Update consumers of the producer.
        replace_uses(analysis, subgraph, *producer, to_id, from_id);
        return true;
      }
    }
    YNN_LOG_DEBUG() << "Replacing node " << node.to_string() << " with copy";
    ynn::define_copy(subgraph, node, to_id, from_id, /*flags=*/0);
    return false;
  } else {
    YNN_LOG_DEBUG() << "Replacing uses of " << from_id << " with " << to_id;
    for (ynn_node* consumer : analysis.consumers[from_id]) {
      for (uint32_t& i : consumer->inputs) {
        if (i == from_id) i = to_id;
      }
    }
    return true;
  }
}

// Rewrite divide(x, sqrt(y)) to multiply(x, reciprocal_square_root(y))
bool rewrite_divide_sqrt(ynn_subgraph& subgraph, ynn_node& node,
                         subgraph_analysis& analysis) {
  if (!is_binary_node(node, ynn_binary_divide)) {
    return false;
  }

  ynn_node* producer = analysis.producer_of(node.inputs[1]);
  if (!producer || !is_unary_node(*producer, ynn_unary_square_root)) {
    // The denominator of the divide is not a square root.
    return false;
  }

  if (analysis.consumers[producer->outputs[0]].size() != 1 ||
      subgraph.value(producer->outputs[0]).is_external_output()) {
    // The square root is used by something else.
    return false;
  }

  // This is x/sqrt(y).
  const ynn_value& x = subgraph.value(node.inputs[0]);
  const ynn_value& sqrt_y = subgraph.value(producer->outputs[0]);
  const ynn_value& y = subgraph.value(producer->inputs[0]);
  const ynn_value& output = subgraph.value(node.outputs[0]);

  const ynn::unary_kernel_fn rsqrt_kernel = ynn::get_unary_kernel(
      ynn_unary_reciprocal_square_root, y.type, sqrt_y.type);
  const ynn::binary_kernel_fn multiply_kernel = ynn::get_binary_kernel(
      ynn_binary_multiply, x.type, sqrt_y.type, output.type);

  if (rsqrt_kernel != nullptr && multiply_kernel != nullptr) {
    YNN_LOG_DEBUG() << "Rewriting x/sqrt(y) to x*rsqrt(y)";
    ynn::define_unary(subgraph, *producer, y.id, sqrt_y.id,
                      ynn_unary_reciprocal_square_root, rsqrt_kernel);
    ynn::define_binary(subgraph, node, x.id, sqrt_y.id, output.id,
                        ynn_binary_multiply, multiply_kernel);
    return true;
  }
  return false;
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

// Rewrite get_tensor_shape(unary(x)) to get_tensor_shape(x). This is useful
// because unary(x) might otherwise be unused after another rewrite (e.g.
// sum(square(x)) => sum_squared(x)).
bool rewrite_get_tensor_shape_of_unary(ynn_subgraph& subgraph, ynn_node& node,
                                       subgraph_analysis& analysis) {
  if (!std::get_if<ynn_node::get_tensor_shape>(&node.op)) {
    return false;
  }
  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (producer && std::get_if<ynn_node::unary_elementwise>(&producer->op)) {
    // This is a get_tensor_shape of a unary elementwise op.
    YNN_LOG_DEBUG() << "Rewriting get_tensor_shape(unary_elementwise(x)) to "
                       "get_tensor_shape(x)";
    node.inputs[0] = producer->inputs[0];
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
  if (std::get_if<ynn_node::binary_elementwise>(&node.op)) {
    return true;
  } else if (std::get_if<ynn_node::ternary_elementwise>(&node.op)) {
    // TODO: b/491453504 - Not all ternary ops can implicitly broadcast all
    // operands.
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

  // transpose_a inserts a new dimension 0, update our stencil dimensions to
  // account for this.
  for (ynn_node::stencil_copy::stencil& stencil : stencil_copy.stencils) {
    stencil.axis++;
    stencil.new_axis++;
  }

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
  if (mul_node == nullptr || !is_square_node(*mul_node)) {
    return false;
  }

  uint32_t x_id = mul_node->inputs[0];
  const ynn_value* x = &subgraph.value(x_id);

  if (x->type != ynn_type_fp32 && x->type != ynn_type_int32) {
    return false;
  }

  uint32_t mul_output_id = mul_node->outputs[0];
  if (mul_output_id != node.inputs[0]) {
    // If there are intervening copy nodes, we need to rewrite the copy op to
    // consume the multiply's input. We can only do this if the reduce is the
    // only consumer of the copy, and the copy is the only consumer of the
    // multiply. `find_non_copy_producer` checks that there are no other
    // consumers of any output in this sequence of ops.
    if (analysis.consumers[mul_output_id].size() != 1 ||
        subgraph.value(mul_output_id).is_external_output()) {
      return false;
    }

    ynn_node* copy = analysis.consumers[mul_output_id].front();
    assert(copy);
    assert(analysis.consumers[copy->outputs[0]].size() == 1 &&
        !subgraph.value(copy->outputs[0]).is_external_output());
    copy->inputs[0] = mul_node->inputs[0];
  } else {
    // The reduce is consuming the multiply, just consume x instead.
    node.inputs[0] = x_id;
  }

  YNN_LOG_DEBUG() << "Rewriting reduce_sum(x*x) to reduce_sum_squared(x)";
  ynn::define_reduce(subgraph, node, ynn_reduce_sum_squared, reduce_op->k_dims,
                     node.inputs[0], node.inputs[1], &node.outputs[0],
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
                     node.inputs[1], &node.outputs[0], reduce_op->keep_dims);
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
                     x.id, node.inputs[1], &node.outputs[0],
                     reduce_op->keep_dims);
  return true;
}

bool fuse_converts(ynn_subgraph& subgraph, ynn_node& node,
                   subgraph_analysis& analysis) {
  const ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&node.op);
  if (unary == nullptr || unary->op != ynn_unary_convert) {
    return false;
  }

  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (producer == nullptr || !is_unary_node(*producer, ynn_unary_convert)) {
    return false;
  }

  const ynn_value& input = subgraph.value(producer->inputs[0]);
  const ynn_value& output = subgraph.value(node.outputs[0]);
  if (input.type != output.type) {
    return false;
  }

  assert(producer->outputs[0] == node.inputs[0]);
  const ynn_value& intermediate = subgraph.value(node.inputs[0]);
  if (!is_convert_lossless(input.type, intermediate.type) &&
      (subgraph.flags & YNN_FLAG_CONSISTENT_ARITHMETIC) != 0) {
    // This conversion loses information, and the converts might have been
    // inserted because we don't have a kernel for this type, which could vary
    // depending on the machine we're running on.
    // TODO(dsharlet): We could add a `round_to` operator for this case to fuse
    // two operators into one.
    YNN_LOG_DEBUG()
        << "Not fusing converts because YNN_FLAG_CONSISTENT_ARITHMETIC is set.";
    return false;
  }

  if (analysis.consumers[intermediate.id].size() != 1) {
    // TODO: b/488394862 - We probably should rewrite even in this case, but it
    // breaks dot bf16 rewrites until we can be explicit that we don't want this
    // sequence of converts to be treated as a round_to_bf16 op.
    return false;
  }

  if (replace_uses(analysis, subgraph, node, output.id, input.id)) {
    node.invalidate();
  }
  return true;
}

bool fuse_quantize(ynn_subgraph& subgraph, ynn_node& node,
                   subgraph_analysis& analysis) {
  if (!is_ternary_node(node, ternary_op::dequantize)) {
    return false;
  }

  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (producer == nullptr ||
      (!is_ternary_node(*producer, ternary_op::quantize_int8) &&
       !is_ternary_node(*producer, ternary_op::quantize_uint8))) {
    return false;
  }

  if (node.inputs[1] != producer->inputs[2] ||
      node.inputs[2] != producer->inputs[1]) {
    return false;
  }

  const ynn_value& input = subgraph.value(producer->inputs[0]);
  const ynn_value& output = subgraph.value(node.outputs[0]);
  if (input.type != output.type) {
    return false;
  }

  assert(producer->outputs[0] == node.inputs[0]);
  if ((subgraph.flags & YNN_FLAG_CONSISTENT_ARITHMETIC) != 0) {
    // This conversion loses information, and the converts might have been
    // inserted because we don't have a kernel for this type, which could vary
    // depending on the machine we're running on.
    // TODO(dsharlet): We could add a `round_to` operator for this case to fuse
    // two operators into one.
    YNN_LOG_DEBUG() << "Not fusing quantization because "
                       "YNN_FLAG_CONSISTENT_ARITHMETIC is set.";
    return false;
  }

  if (replace_uses(analysis, subgraph, node, output.id, input.id)) {
    node.invalidate();
  }
  return true;
}

// Rewrite multiply(dot(..., subtract_multiply(0, a, b)), c, d) to
// dequantize_dot(dot(..., YNN_INVALID_VALUE_ID), a, b, c, d, 0)
bool rewrite_dequantize_dot(ynn_subgraph& subgraph, ynn_node& node,
                            subgraph_analysis& analysis) {
  if (!is_ternary_node(node, ternary_op::multiply)) {
    return false;
  }

  ynn_node* dot_node = analysis.producer_of(node.inputs[0]);
  if (!dot_node) return false;
  const ynn_node::dot* dot_op = std::get_if<ynn_node::dot>(&dot_node->op);
  if (!dot_op || dot_node->inputs.size() < 2) return false;

  uint32_t input_c_id = dot_node->inputs[1];
  ynn_node* input_c_producer = analysis.producer_of(input_c_id);
  if (!input_c_producer ||
      !is_ternary_node(*input_c_producer, ternary_op::subtract_multiply)) {
    return false;
  }

  // Only do this rewrite if we won't break other consumers.
  if (analysis.consumers[dot_node->outputs[0]].size() != 1 ||
      subgraph.value(dot_node->outputs[0]).is_external_output()) {
    return false;
  }
  if (analysis.consumers[input_c_id].size() != 1 ||
      subgraph.value(input_c_id).is_external_output()) {
    return false;
  }

  // Check if subtract_multiply(0, a, b)
  const ynn_value& sm_a = subgraph.value(input_c_producer->inputs[0]);
  if (!sm_a.is_static_scalar() || sm_a.as_scalar_float() != 0.0f) {
    return false;
  }

  uint32_t a_offset_id = input_c_producer->inputs[1];
  uint32_t b_offset_id = input_c_producer->inputs[2];
  uint32_t a_scale_id = node.inputs[1];
  uint32_t b_scale_id = node.inputs[2];

  YNN_LOG_DEBUG() << "Rewriting multiply(dot(..., subtract_multiply(0, a, b)), "
                     "c, d) to dequantize_dot";

  const ynn_value& output = subgraph.value(node.outputs[0]);
  uint32_t offset_id = subgraph.get_scalar_value_id(
      output.type, YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID, 0.0f);

  uint32_t input1_id = dot_node->inputs[1];
  dot_node->inputs[1] = YNN_INVALID_VALUE_ID;
  bool result = ynn::define_dequantize_dot(
      subgraph, node, output.type, dot_node->outputs[0], a_offset_id,
      b_offset_id, a_scale_id, b_scale_id, offset_id, node.outputs[0],
      dequantize_dot_params{});
  if (!result) {
    // There is no kernel for dequantize_dot, so don't do rewrite and restore
    // the old value.
    dot_node->inputs[1] = input1_id;
  }
  return result;
}

// Rewrite add(dequantize_dot(..., 0), x) to dequantize_dot(..., x)
bool rewrite_dequantize_dot_add(ynn_subgraph& subgraph, ynn_node& node,
                                subgraph_analysis& analysis) {
  if (!is_binary_node(node, ynn_binary_add)) {
    return false;
  }

  for (int i : {0, 1}) {
    uint32_t dequantize_dot_output_id = node.inputs[i];
    ynn_node* dequantize_dot_node =
        analysis.producer_of(dequantize_dot_output_id);
    if (!dequantize_dot_node) continue;
    const ynn_node::dequantize_dot* rescale_op =
        std::get_if<ynn_node::dequantize_dot>(&dequantize_dot_node->op);
    if (!rescale_op) continue;

    if (analysis.consumers[dequantize_dot_output_id].size() != 1 ||
        subgraph.value(dequantize_dot_output_id).is_external_output()) {
      continue;
    }

    uint32_t offset_id = dequantize_dot_node->inputs[5];
    const ynn_value& offset = subgraph.value(offset_id);
    if (!offset.is_static_scalar() || offset.as_scalar_float() != 0.0f) {
      continue;
    }

    uint32_t new_offset_id = node.inputs[1 - i];

    YNN_LOG_DEBUG()
        << "Rewriting add(dequantize_dot(..., 0), x) to dequantize_dot(..., x)";

    const ynn_value& output = subgraph.value(node.outputs[0]);
    bool result = ynn::define_dequantize_dot(
        subgraph, node, output.type, dequantize_dot_node->inputs[0],
        dequantize_dot_node->inputs[1], dequantize_dot_node->inputs[2],
        dequantize_dot_node->inputs[3], dequantize_dot_node->inputs[4],
        new_offset_id, node.outputs[0], rescale_op->params);

    if (!result) {
      continue;
    }
    dequantize_dot_node->invalidate();
    return true;
  }
  return false;
}

// Rewrite f(x * C) to fold the arithmetic into the params.
bool fold_unary_input(ynn_subgraph& subgraph, ynn_node& node,
                              subgraph_analysis& analysis) {
  ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&node.op);
  if (unary == nullptr) {
    return false;
  }
  switch (unary->op) {
    case ynn_unary_exp:
    case ynn_unary_erf:
      break;
    default:
      return false;
  }

  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (!producer) {
    return false;
  }
  if (auto mul = is_scalar_arithmetic(subgraph, *producer)) {
    if (mul->b != 0.0f) {
      // We can't handle addition here.
      return false;
    }
    YNN_LOG_DEBUG() << "Folding multiply by " << mul->a << " into "
                    << to_string(unary->op) << ".";
    if (unary->op == ynn_unary_exp) {
      unary->params.exp.input_multiplier *= mul->a;
    } else {
      unary->params.erf.input_multiplier *= mul->a;
    }
    node.inputs[0] = mul->x_id;
    return true;
  }
  return false;
}

// Rewrite f(x) * A + B to fold the arithmetic into the params
bool fold_unary_output(ynn_subgraph& subgraph, ynn_node& node,
                        subgraph_analysis& analysis) {
  auto scalar_arithmetic = is_scalar_arithmetic(subgraph, node);
  if (!scalar_arithmetic) return false;

  ynn_node* producer = analysis.producer_of(scalar_arithmetic->x_id);
  if (producer == nullptr) return false;

  ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&producer->op);
  if (unary == nullptr) return false;

  switch (unary->op) {
    case ynn_unary_exp:
      if (scalar_arithmetic->b != 0.0f) {
        // exp does not support output offset
        return false;
      }
      break;
    case ynn_unary_erf:
    case ynn_unary_tanh:
    case ynn_unary_sine:
    case ynn_unary_cosine:
    case ynn_unary_poly3:
      break;
    default:
      return false;
  }

  // Check if the output is only used by this binary node.
  if (analysis.consumers[producer->outputs[0]].size() != 1) return false;

  YNN_LOG_DEBUG() << "Folding scalar arithmetic onto " << to_string(unary->op);

  unary->params.poly3.c0 *= scalar_arithmetic->a;
  unary->params.poly3.c1 *= scalar_arithmetic->a;
  if (unary->op == ynn_unary_poly3) {
    unary->params.poly3.c2 *= scalar_arithmetic->a;
    unary->params.poly3.c3 *= scalar_arithmetic->a;
  }
  unary->params.poly3.c0 += scalar_arithmetic->b;

  producer->outputs[0] = node.outputs[0];
  node.invalidate();
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

      changed = changed || ynn::fold_unary_input(*this, node, analysis) ||
                ynn::fold_unary_output(*this, node, analysis) ||
                ynn::rewrite_divide_sqrt(*this, node, analysis) ||
                ynn::rewrite_multiply_add(*this, node, analysis) ||
                ynn::rewrite_multiply_multiply(*this, node, analysis) ||
                ynn::rewrite_subtract_multiply(*this, node, analysis) ||
                ynn::rewrite_dequantize_dot(*this, node, analysis) ||
                ynn::rewrite_dequantize_dot_add(*this, node, analysis) ||
                ynn::rewrite_get_tensor_shape_of_unary(*this, node, analysis) ||
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

  do {
    subgraph_analysis analysis(*this);
    changed = false;
    for (ynn_node& node : nodes) {
      if (!node.is_valid()) continue;

      changed = changed || ynn::fuse_converts(*this, node, analysis) ||
                ynn::fuse_quantize(*this, node, analysis);
    }
  } while (changed);

  return ynn_status_success;
}
