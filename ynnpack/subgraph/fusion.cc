// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/algorithm.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/span.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/dequantize_dot/dequantize_dot.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/kernels/unary/unary.h"
#include "ynnpack/subgraph/copy.h"
#include "ynnpack/subgraph/dot.h"
#include "ynnpack/subgraph/elementwise.h"
#include "ynnpack/subgraph/fusion_lut.h"
#include "ynnpack/subgraph/fusion_types.h"
#include "ynnpack/subgraph/reduce.h"
#include "ynnpack/subgraph/static_slice.h"
#include "ynnpack/subgraph/static_transpose.h"
#include "ynnpack/subgraph/stencil_copy.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/expr.h"

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
  real a = 1.0;
  real b = 0.0;
};

// If `node` is a linear expression of scalar constants, returns
// `scalar_arithmetic` describing the operation.
std::optional<scalar_arithmetic> is_scalar_arithmetic(
    const ynn_subgraph& subgraph, const ynn_node& node) {
  if (is_unary_node(node, ynn_unary_negate)) {
    return scalar_arithmetic{node.inputs[0], -1.0, 0.0};
  }
  const ynn_node::binary_elementwise* binary =
      std::get_if<ynn_node::binary_elementwise>(&node.op);
  if (binary == nullptr) return std::nullopt;

  if (const auto b = subgraph.value(node.inputs[1]).as_scalar()) {
    switch (binary->op) {
      case ynn_binary_add:
        return scalar_arithmetic{node.inputs[0], 1.0, *b};
      case ynn_binary_subtract:
        return scalar_arithmetic{node.inputs[0], 1.0, -*b};
      case ynn_binary_multiply:
        return scalar_arithmetic{node.inputs[0], *b, 0.0};
      case ynn_binary_divide:
        return scalar_arithmetic{node.inputs[0], 1.0 / *b, 0.0};
      default:
        return std::nullopt;
    }
  }

  if (const auto a = subgraph.value(node.inputs[0]).as_scalar()) {
    switch (binary->op) {
      case ynn_binary_add:
        return scalar_arithmetic{node.inputs[1], 1.0, *a};
      case ynn_binary_subtract:
        return scalar_arithmetic{node.inputs[1], -1.0, *a};
      case ynn_binary_multiply:
        return scalar_arithmetic{node.inputs[1], *a, 0.0};
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

// We rewrite outer(inner(op[0], op[1]), op[2]) or
// outer(op[2], inner(op[0], op[1])) to
// ternary(op[a], op[b], op[c])
struct ternary_rewrite {
  ternary_op op = ternary_op::invalid;
  int a = 0;
  int b = 1;
  int c = 2;
  // If this rewrite requires the inner binary op to appear as operand 0 or 1 of
  // the outer binary op, this optional should be set to that index.
  std::optional<int> inner_operand = std::nullopt;
};

ternary_rewrite get_ternary_rewrite(ynn_binary_operator outer,
                                    ynn_binary_operator inner) {
  if (outer == ynn_binary_add && inner == ynn_binary_multiply) {
    return {ternary_op::multiply_add, /*a=*/0, /*b=*/1, /*c=*/2};
  } else if (outer == ynn_binary_multiply && inner == ynn_binary_multiply) {
    // TODO: b/504634617 - Our ternary multiply with mixed types assumes this
    // order. We should generalize this to allow searching for kernels with
    // operands in any order (or canonicalize based on type).
    return {ternary_op::multiply, /*a=*/2, /*b=*/0, /*c=*/1};
  } else if (outer == ynn_binary_min && inner == ynn_binary_max) {
    return {ternary_op::clamp, /*a=*/0, /*b=*/1, /*c=*/2};
  } else if (outer == ynn_binary_subtract && inner == ynn_binary_multiply) {
    return {ternary_op::subtract_multiply, /*a=*/2, /*b=*/0, /*c=*/1,
            /*inner_operand=*/1};
  } else {
    return {};
  }
}

// Rewrite outer(inner(a, b), c) to ternary(a, b, c)
bool rewrite_ternary(ynn_subgraph& subgraph, ynn_node& node,
                     subgraph_analysis& analysis) {
  const auto* outer = std::get_if<ynn_node::binary_elementwise>(&node.op);
  if (!outer) return false;

  for (int i : {0, 1}) {
    ynn_node* producer = analysis.producer_of(node.inputs[i]);
    if (!producer) continue;
    ynn_binary_operator inner;
    if (const auto* unary =
            std::get_if<ynn_node::unary_elementwise>(&producer->op)) {
      if (unary->op == ynn_unary_square) {
        inner = ynn_binary_multiply;
      } else {
        continue;
      }
    } else if (const auto* binary =
                   std::get_if<ynn_node::binary_elementwise>(&producer->op)) {
      inner = binary->op;
    } else {
      continue;
    }

    if (analysis.consumers[producer->outputs[0]].size() != 1) continue;

    ternary_rewrite r = get_ternary_rewrite(outer->op, inner);
    if (r.op == ternary_op::invalid) continue;
    if (r.inner_operand && r.inner_operand != i) continue;

    // If the inner operator was a unary op, we duplicate the operand.
    const uint32_t ops[] = {
        producer->inputs[0],
        producer->inputs[producer->inputs.size() == 1 ? 0 : 1],
        node.inputs[1 - i],
    };

    // This sequence of binary ops matches. Do we have a kernel for this case?
    const ynn_value& a = subgraph.value(ops[r.a]);
    const ynn_value& b = subgraph.value(ops[r.b]);
    const ynn_value& c = subgraph.value(ops[r.c]);
    const ynn_value& x = subgraph.value(node.outputs[0]);
    const ynn::ternary_kernel_fn kernel =
        ynn::get_ternary_kernel(r.op, a.type, b.type, c.type, x.type);
    if (kernel != nullptr) {
      // Yes we do. Rewrite this to a ternary op.
      YNN_LOG_DEBUG() << "Rewriting " << to_string(outer->op) << "("
                      << to_string(inner) << "(a, b), c) to " << to_string(r.op)
                      << "(a, b, c)";
      ynn::define_ternary(subgraph, node, a.id, b.id, c.id, x.id, r.op, kernel);
      return true;
    }
  }
  return false;
}

// Rewrite binary(convert(x), y) to binary(x, y) if a kernel exists for the
// input types.
bool rewrite_binary_convert(ynn_subgraph& subgraph, ynn_node& node,
                            subgraph_analysis& analysis) {
  const ynn_node::binary_elementwise* binary =
      std::get_if<ynn_node::binary_elementwise>(&node.op);
  if (!binary) return false;

  ynn_node* producers[2] = {analysis.producer_of(node.inputs[0]),
                            analysis.producer_of(node.inputs[1])};
  bool is_convert[2] = {
      producers[0] && is_unary_node(*producers[0], ynn_unary_convert),
      producers[1] && is_unary_node(*producers[1], ynn_unary_convert)};

  if (!is_convert[0] && !is_convert[1]) {
    return false;
  }

  const ynn_value& a =
      subgraph.value(is_convert[0] ? producers[0]->inputs[0] : node.inputs[0]);
  const ynn_value& b =
      subgraph.value(is_convert[1] ? producers[1]->inputs[0] : node.inputs[1]);
  const ynn_value& output = subgraph.value(node.outputs[0]);

  // If it's a square, we must rewrite both or neither to keep it a square.
  if (node.inputs[0] == node.inputs[1] && is_convert[0]) {
    ynn::binary_kernel_fn kernel =
        ynn::get_binary_kernel(binary->op, a.type, b.type, output.type);
    if (kernel != nullptr) {
      YNN_LOG_DEBUG() << "Rewriting " << to_string(binary->op)
                      << "(convert(x), convert(x)) to " << to_string(binary->op)
                      << "(x, x)";
      ynn::define_binary(subgraph, node, a.id, b.id, output.id, binary->op,
                         kernel);
      return true;
    }
    return false;
  }

  // If both are converts, try rewriting both.
  if (is_convert[0] && is_convert[1]) {
    ynn::binary_kernel_fn kernel =
        ynn::get_binary_kernel(binary->op, a.type, b.type, output.type);
    if (kernel != nullptr) {
      YNN_LOG_DEBUG() << "Rewriting " << to_string(binary->op)
                      << "(convert(x), convert(y)) to " << to_string(binary->op)
                      << "(x, y)";
      ynn::define_binary(subgraph, node, a.id, b.id, output.id, binary->op,
                         kernel);
      return true;
    }
  }

  // Try rewriting just one.
  for (int i : {0, 1}) {
    if (!is_convert[i]) continue;
    ynn_type type_a = i == 0 ? a.type : subgraph.value(node.inputs[0]).type;
    ynn_type type_b = i == 1 ? b.type : subgraph.value(node.inputs[1]).type;
    ynn::binary_kernel_fn kernel =
        ynn::get_binary_kernel(binary->op, type_a, type_b, output.type);
    if (kernel != nullptr) {
      YNN_LOG_DEBUG() << "Rewriting " << to_string(binary->op)
                      << "(convert(x), y) to " << to_string(binary->op)
                      << "(x, y)";
      ynn::define_binary(subgraph, node, i == 0 ? a.id : node.inputs[0],
                         i == 1 ? b.id : node.inputs[1], output.id, binary->op,
                         kernel);
      return true;
    }
  }

  return false;
}

// Rewrite convert(elementwise(a, ...)) to elementwise(a, ...) if a kernel
// exists for the output type.
bool rewrite_convert_elementwise(ynn_subgraph& subgraph, ynn_node& node,
                                 subgraph_analysis& analysis) {
  if (!is_unary_node(node, ynn_unary_convert)) return false;

  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (!producer) return false;
  if (const auto* unary =
          std::get_if<ynn_node::unary_elementwise>(&producer->op)) {
    const ynn_value& a = subgraph.value(producer->inputs[0]);
    const ynn_value& x = subgraph.value(node.outputs[0]);

    if (unary->op == ynn_unary_convert) {
      // We fuse two converts elsewhere.
      return false;
    }

    ynn::unary_kernel_fn kernel =
        ynn::get_unary_kernel(unary->op, a.type, x.type);
    if (!kernel) return false;
    YNN_LOG_DEBUG() << "Rewriting "
                    << "convert(" << to_string(unary->op) << "(a)) to "
                    << to_string(unary->op) << "(a)";
    ynn::define_unary(subgraph, node, a.id, x.id, unary->op, kernel,
                      unary->params);
    return true;
  } else if (const auto* binary =
                 std::get_if<ynn_node::binary_elementwise>(&producer->op)) {
    const ynn_value& a = subgraph.value(producer->inputs[0]);
    const ynn_value& b = subgraph.value(producer->inputs[1]);
    const ynn_value& x = subgraph.value(node.outputs[0]);

    ynn::binary_kernel_fn kernel =
        ynn::get_binary_kernel(binary->op, a.type, b.type, x.type);
    if (!kernel) return false;
    YNN_LOG_DEBUG() << "Rewriting "
                    << "convert(" << to_string(binary->op) << "(a, b)) to "
                    << to_string(binary->op) << "(a, b)";
    ynn::define_binary(subgraph, node, a.id, b.id, x.id, binary->op, kernel);
    return true;
  }
  return false;
}

// Rewrite -multiply(b, c) to subtract_multiply(a, b, c)
bool rewrite_negate_multiply(ynn_subgraph& subgraph, ynn_node& node,
                             subgraph_analysis& analysis) {
  if (!is_unary_node(node, ynn_unary_negate)) return false;

  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (producer && is_binary_node(*producer, ynn_binary_multiply) &&
      analysis.consumers[producer->outputs[0]].size() == 1) {
    // This is a subtract_multiply. Do we have a kernel for this case?
    const ynn_value& b = subgraph.value(producer->inputs[0]);
    const ynn_value& c = subgraph.value(producer->inputs[1]);
    const ynn_value& x = subgraph.value(node.outputs[0]);
    uint32_t a_id = subgraph.get_scalar_value_id(x.type, 0.0f);
    const ynn_value& a = subgraph.value(a_id);

    const ynn::ternary_kernel_fn kernel = ynn::get_ternary_kernel(
        ynn::ternary_op::subtract_multiply, a.type, b.type, c.type, x.type);
    if (kernel != nullptr) {
      YNN_LOG_DEBUG()
          << "Rewriting -multiply(b, c) to ternary subtract_multiply";
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

bool can_implicitly_broadcast(const ynn_node& node, uint32_t input_id) {
  if (std::get_if<ynn_node::binary_elementwise>(&node.op)) {
    return true;
  } else if (std::get_if<ynn_node::ternary_elementwise>(&node.op)) {
    // TODO: b/491453504 - Not all ternary ops can implicitly broadcast all
    // operands.
    return true;
  } else if (std::get_if<ynn_node::static_broadcast>(&node.op)) {
    return true;
  }
  return false;
}

// Returns true if a broadcast of `input_id` in `axes` is a no-op for `node`.
bool is_broadcast_noop(const ynn_subgraph& subgraph, const ynn_node& node,
                       uint32_t input_id, axes_set axes) {
  if (std::holds_alternative<ynn_node::unary_elementwise>(node.op) ||
      std::holds_alternative<ynn_node::binary_elementwise>(node.op) ||
      std::holds_alternative<ynn_node::ternary_elementwise>(node.op) ||
      std::holds_alternative<ynn_node::dequantize_dot>(node.op)) {
    // A broadcast is a no-op the other inputs in the broadcasted dimension are
    // broadcasts.
    for (size_t d = 0; d < axes.size(); ++d) {
      if (!axes[d]) continue;
      for (uint32_t i : node.inputs) {
        if (i == input_id) continue;
        const ynn_value& input = subgraph.value(i);
        if (!slinky::is_one(input.extent(d))) {
          return false;
        }
      }
    }
    return true;
  } else if (const auto* g = std::get_if<ynn_node::gather>(&node.op)) {
    const ynn_value& table = subgraph.value(node.inputs[0]);
    if (table.rank() == 1 && g->axes.size() == 1 && g->axes[0] == 0 &&
        input_id == node.inputs[1]) {
      return true;
    }
  } else if (const auto* t =
                 std::get_if<ynn_node::static_transpose>(&node.op)) {
    assert(input_id == node.inputs[0]);
    for (size_t i = 0; i < axes.size(); ++i) {
      if (!axes[i]) continue;
      if (i < t->permutation.size() && t->permutation[i] != i) {
        // This transpose changes a dimension we broadcast, not a no-op.
        // We could do better here, if we understood that we should change the
        // dimensions that are broadcasted.
        return false;
      }
    }
    return true;
  }
  // We can handle more ops here, especially if we allow changing which
  // dimensions are broadcasted.
  return false;
}

// Get the axes that `node` broadcasts, if any.
ynn::axes_set get_broadcast_axes(const ynn_node& node) {
  if (const auto* broadcast =
          std::get_if<ynn_node::static_broadcast>(&node.op)) {
    ynn::axes_set axes;
    assert(broadcast->new_dims.size() <= axes.size());
    for (size_t i = 0; i < broadcast->new_dims.size(); ++i) {
      if (broadcast->new_dims[i] != 0) {
        axes[i] = true;
      }
    }
    return axes;
  } else if (const auto* broadcast_like =
                 std::get_if<ynn_node::broadcast_like>(&node.op)) {
    return broadcast_like->axes;
  }
  // We don't handle broadcast here because it doesn't actually broadcast the
  // data, so it's harmless.
  return ynn::axes_set{};
}

// Rewrite f(broadcast(x)) -> broadcast(f(x)) when possible.
bool move_broadcast_to_output(ynn_subgraph& subgraph, ynn_node& broadcast,
                              subgraph_analysis& analysis) {
  const ynn::axes_set axes = get_broadcast_axes(broadcast);
  if (!axes.any()) return false;

  const ynn_value& input = subgraph.value(broadcast.inputs[0]);
  uint32_t broadcast_id = broadcast.outputs[0];

  ynn_node* consumer = analysis.single_consumer_of(broadcast_id);
  if (!consumer) return false;

  // Currently we don't handle any ops with more than one output.
  if (consumer->outputs.size() != 1) return false;

  if (!is_broadcast_noop(subgraph, *consumer, broadcast_id, axes)) {
    // This consumer needs this broadcast.
    return false;
  }

  // Currently we have consumer(broadcast(x)), we want broadcast(consumer(x)).
  for (uint32_t& input_id : consumer->inputs) {
    if (input_id == broadcast_id) {
      // Consume the broadcast input instead.
      input_id = broadcast.inputs[0];
    }
  }

  ynn_value& broadcast_output = subgraph.value(broadcast.outputs[0]);
  const ynn_value& consumer_output = subgraph.value(consumer->outputs[0]);
  broadcast_output.type = consumer_output.type;
  broadcast_output.extents = input.extents;

  broadcast.inputs[0] = broadcast.outputs[0];
  std::swap(consumer->outputs[0], broadcast.outputs[0]);
  subgraph.topological_sort();
  analysis.invalidate();

  return true;
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

bool is_expand_dims_noop(const ynn_value& input,
                         const ynn::axes_set& new_axes) {
  size_t output_rank = input.rank() + new_axes.count();
  for (int d = static_cast<int>(new_axes.size()) - 1; d >= 0; --d) {
    if (!new_axes[d]) continue;
    if (d + 1 > output_rank) {
    } else if (d + 1 == output_rank) {
      --output_rank;
    } else {
      return false;
    }
  }
  return true;
}

bool remove_broadcast_expand_dims(ynn_subgraph& subgraph, ynn_node& node,
                                  subgraph_analysis& analysis) {
  auto* transpose = std::get_if<ynn_node::static_transpose>(&node.op);
  if (!transpose) return false;

  const ynn_value& input = subgraph.value(node.inputs[0]);
  auto expand_dims_axes = get_static_expand_dims_axes(*transpose, input.rank());
  if (!expand_dims_axes) return false;

  ynn_value& output = subgraph.value(node.outputs[0]);
  if (output.is_external_output()) return false;

  if (!is_expand_dims_noop(input, *expand_dims_axes)) return false;

  std::vector<ynn_node*>& consumers = analysis.consumers[node.outputs[0]];
  if (!std::all_of(consumers.begin(), consumers.end(), [&](const ynn_node* i) {
        return can_implicitly_broadcast(*i, output.id);
      })) {
    // One of the consumers can't handle implicit broadcasts.
    return false;
  }

  // This broadcast is a no-op, replace consumers with our input, and remove
  // the op.
  YNN_LOG_DEBUG() << "Removing expand_dims";
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

bool remove_broadcast(ynn_subgraph& subgraph, ynn_node& node,
                      subgraph_analysis& analysis) {
  // expand_dims is often used to broadcast trailing dimensions and not needed
  // in this case.
  return remove_broadcast_expand_dims(subgraph, node, analysis) ||
         remove_broadcast<ynn_node::broadcast_like>(subgraph, node, analysis);
}

// If the producer of an elementwise op is a static_broadcast, and the dimension
// being broadcast is known to be a broadcast, we don't need the broadcast.
bool remove_static_broadcast_from_elementwise(ynn_subgraph& subgraph,
                                              ynn_node& node,
                                              subgraph_analysis& analysis) {
  if (!std::holds_alternative<ynn_node::binary_elementwise>(node.op) &&
      !std::holds_alternative<ynn_node::ternary_elementwise>(node.op)) {
    // This is not an op that we can assume implicitly broadcasts statically
    // extent 1 dimensions.
    return false;
  }

  bool change = false;

  for (uint32_t& input_id : node.inputs) {
    ynn_node* producer = analysis.producer_of(input_id);
    if (!producer) continue;

    ynn_node::static_broadcast* broadcast =
        std::get_if<ynn_node::static_broadcast>(&producer->op);
    if (broadcast == nullptr) continue;

    if (!can_implicitly_broadcast(node, input_id)) continue;

    const ynn_value& broadcasted = subgraph.value(producer->inputs[0]);
    auto broadcast_needed = [&](size_t d) {
      if (!broadcast->new_dims[d]) {
        // This dimension is not broadcasted, we can drop this dimension.
        return false;
      }

      if (d < broadcasted.rank() && broadcasted.extents[d].defined()) {
        // This dimension has an explicit extent, we can't implicitly broadcast
        // it.
        return true;
      }

      // We weed the broadcast if this dimension is not implicitly broadcasted
      // by any other input.
      auto implicitly_broadcasts = [&](uint32_t id) {
        if (id == input_id) return false;

        const ynn_value& input = subgraph.value(id);
        return d < input.extents.size() &&
               slinky::prove_true(input.extent(d) == broadcast->new_dims[d]);
      };
      return !std::any_of(node.inputs.begin(), node.inputs.end(),
                          implicitly_broadcasts);
    };
    if (!any_n(broadcast->new_dims.size(), broadcast_needed)) {
      YNN_LOG_DEBUG() << "Removing static_broadcast from elementwise input.";
      input_id = producer->inputs[0];
      change = true;
    }
  }

  return change;
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

// Rewrite static_expand_dims(reduce(x)) to reduce(x, keep_dims=true)
bool rewrite_expand_dims_reduce(ynn_subgraph& subgraph, ynn_node& node,
                                subgraph_analysis& analysis) {
  auto* transpose = std::get_if<ynn_node::static_transpose>(&node.op);
  if (!transpose) return false;

  const ynn_value& input = subgraph.value(node.inputs[0]);
  auto expand_dims_axes = get_static_expand_dims_axes(*transpose, input.rank());
  if (!expand_dims_axes) return false;

  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (!producer) return false;

  ynn_node::reduce* reduce = std::get_if<ynn_node::reduce>(&producer->op);
  if (!reduce || reduce->keep_dims) return false;

  if (*expand_dims_axes != reduce->k_dims) return false;

  // Check that the intermediate output has exactly one consumer
  // and is not an external output.
  uint32_t intermediate_id = producer->outputs[0];
  if (analysis.consumers[intermediate_id].size() != 1 ||
      subgraph.value(intermediate_id).is_external_output()) {
    return false;
  }

  YNN_LOG_DEBUG() << "Fusing expand_dims and " << to_string(reduce->op)
                  << " to " << to_string(reduce->op) << "(keep_dims=true)";

  std::get<ynn_node::reduce>(producer->op).keep_dims = true;
  producer->outputs[0] = node.outputs[0];
  if (producer->inputs[1] != YNN_INVALID_VALUE_ID) {
    const ynn_value& init = subgraph.value(producer->inputs[1]);
    if (!is_expand_dims_noop(init, *expand_dims_axes)) {
      // We need to move the expand_dims to the initializer input.
      uint32_t expanded_id = node.inputs[0];
      ynn::define_static_expand_dims(subgraph, node, producer->inputs[1],
                                     &expanded_id, *expand_dims_axes);
      producer->inputs[1] = expanded_id;
      subgraph.topological_sort();
      analysis.invalidate();
      return true;
    }
  }
  node.invalidate();
  return true;
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
  ynn_node::reduce* reduce_op = std::get_if<ynn_node::reduce>(&node.op);
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

  switch (x->type) {
    case ynn_type_fp64:
    case ynn_type_fp32:
    case ynn_type_int32:
      break;
    default:
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
  reduce_op->op = ynn_reduce_sum_squared;

  return true;
}

// Rewrites ynn_reduce_sum of convert to ynn_reduce_sum of convert's input.
// Specifically:
//   ynn_reduce_sum(f32(x_fp16)) -> ynn_reduce_sum(x_fp16)
//   ynn_reduce_sum(f32(x_bf16)) -> ynn_reduce_sum(x_bf16)
//   ynn_reduce_sum(int32(x_int8)) -> ynn_reduce_sum(x_int8)
bool rewrite_reduce_convert(ynn_subgraph& subgraph, ynn_node& node,
                            subgraph_analysis& analysis) {
  const ynn_node::reduce* reduce_op = std::get_if<ynn_node::reduce>(&node.op);
  if (!reduce_op) return false;

  ynn_node* convert = analysis.producer_of(node.inputs[0]);
  if (!convert || !is_unary_node(*convert, ynn_unary_convert)) {
    return false;
  }

  const ynn_value& x = subgraph.value(convert->inputs[0]);

  const ynn_value& output = subgraph.value(node.outputs[0]);
  reduce_kernel kernel = get_reduce_kernel(reduce_op->op, x.type, output.type);
  if (!kernel.k1 || !kernel.kn) {
    // We don't have a kernel for this converted reduction.
    return false;
  }

  YNN_LOG_DEBUG() << "Rewriting reduce(" << to_string(reduce_op->op)
                  << ", convert(x)) to reduce(" << to_string(reduce_op->op)
                  << ", x)";
  node.inputs[0] = x.id;
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
      ((subgraph.flags & YNN_FLAG_NO_EXCESS_PRECISION) != 0 ||
       (intermediate.flags & YNN_VALUE_FLAG_NO_EXCESS_PRECISION) != 0)) {
    if (intermediate.type == ynn_type_bf16) {
      unary_kernel_fn kernel = ynn::get_unary_kernel(ynn_unary_round_to_bf16,
                                                     input.type, output.type);
      if (kernel) {
        YNN_LOG_DEBUG() << "Rewriting convert(" << to_string(input.type)
                        << ", convert(bf16, x)) to round_to_bf16(x)";
        define_unary(subgraph, node, producer->inputs[0], node.outputs[0],
                     ynn_unary_round_to_bf16, kernel);
        return true;
      }
    }

    // This conversion loses information, and the converts might have been
    // inserted because we don't have a kernel for this type.
    YNN_LOG_DEBUG()
        << "Not fusing no-op converts because YNN_FLAG_NO_EXCESS_PRECISION or "
           "YNN_VALUE_FLAG_NO_EXCESS_PRECISION is set.";
    return false;
  }

  YNN_LOG_DEBUG() << "Rewriting convert(" << to_string(input.type)
                  << ", convert(" << to_string(intermediate.type)
                  << ", x)) to x";

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
  if ((subgraph.flags & YNN_FLAG_CONSISTENT_ARITHMETIC) != 0 ||
      (subgraph.flags & YNN_FLAG_NO_EXCESS_PRECISION) != 0) {
    // This conversion loses information, and the converts might have been
    // inserted because we don't have a kernel for this type, which could vary
    // depending on the machine we're running on.
    // TODO(dsharlet): We could add a `round_to` operator for this case to fuse
    // two operators into one.
    YNN_LOG_DEBUG() << "Not fusing quantization because "
                       "YNN_FLAG_CONSISTENT_ARITHMETIC or "
                       "YNN_FLAG_NO_EXCESS_PRECISION is set.";
    return false;
  }

  YNN_LOG_DEBUG() << "Rewriting dequantize(quantize(x)) to x";

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

  uint32_t input_c_id = dot_node->inputs[2];
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
  if (!sm_a.is_static_scalar() || sm_a.as_scalar() != 0.0f) {
    return false;
  }

  uint32_t a_offset_id = input_c_producer->inputs[1];
  uint32_t b_offset_id = input_c_producer->inputs[2];
  uint32_t a_scale_id = node.inputs[1];
  uint32_t b_scale_id = node.inputs[2];

  YNN_LOG_DEBUG() << "Rewriting multiply(dot(..., subtract_multiply(0, a, b)), "
                     "c, d) to dequantize_dot";

  const ynn_value& output = subgraph.value(node.outputs[0]);
  uint32_t offset_id = subgraph.get_scalar_value_id(output.type, 0.0f);

  uint32_t input1_id = dot_node->inputs[1];
  dot_node->inputs[2] = YNN_INVALID_VALUE_ID;
  bool result = ynn::define_dequantize_dot(
      subgraph, node, output.type, dot_node->outputs[0], a_offset_id,
      b_offset_id, a_scale_id, b_scale_id, offset_id, node.outputs[0],
      dequantize_dot_params{});
  if (!result) {
    // There is no kernel for dequantize_dot, so don't do rewrite and restore
    // the old value.
    dot_node->inputs[2] = input1_id;
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
    const ynn_node::dequantize_dot* dequantize_dot_op =
        std::get_if<ynn_node::dequantize_dot>(&dequantize_dot_node->op);
    if (!dequantize_dot_op) continue;

    if (analysis.consumers[dequantize_dot_output_id].size() != 1 ||
        subgraph.value(dequantize_dot_output_id).is_external_output()) {
      continue;
    }

    uint32_t offset_id = dequantize_dot_node->inputs[5];
    const ynn_value& offset = subgraph.value(offset_id);
    if (!offset.is_static_scalar() || offset.as_scalar() != 0.0f) {
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
        new_offset_id, node.outputs[0], dequantize_dot_op->params);

    if (!result) {
      continue;
    }
    dequantize_dot_node->invalidate();
    return true;
  }
  return false;
}

// Rewrite convert(dequantize_dot(...)) to dequantize_dot(...) with the
// target type if a kernel exists for that type.
bool rewrite_dequantize_dot_convert(ynn_subgraph& subgraph, ynn_node& node,
                                    subgraph_analysis& analysis) {
  if (!is_unary_node(node, ynn_unary_convert)) {
    return false;
  }

  uint32_t dequantize_dot_output_id = node.inputs[0];
  ynn_node* dequantize_dot_node =
      analysis.producer_of(dequantize_dot_output_id);
  if (!dequantize_dot_node) return false;
  const ynn_node::dequantize_dot* dequantize_dot_op =
      std::get_if<ynn_node::dequantize_dot>(&dequantize_dot_node->op);
  if (!dequantize_dot_op) return false;

  if (analysis.consumers[dequantize_dot_output_id].size() != 1 ||
      subgraph.value(dequantize_dot_output_id).is_external_output()) {
    return false;
  }

  const ynn_value& output = subgraph.value(node.outputs[0]);

  // Check if we have a dequantize_dot kernel for the target type.
  dequantize_dot_kernel_fn kernel = get_dequantize_dot_kernel(output.type);
  if (!kernel) return false;

  YNN_LOG_DEBUG()
      << "Rewriting convert(dequantize_dot(...)) to dequantize_dot(...) "
         "with type "
      << to_string(output.type);

  dequantize_dot_node->outputs[0] = node.outputs[0];
  node.invalidate();
  return true;
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
    case ynn_unary_expm1:
    case ynn_unary_log:
    case ynn_unary_log1p:
    case ynn_unary_erf:
    case ynn_unary_approx_erf:
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

    ynn_type input_type = subgraph.value(mul->x_id).type;
    ynn_type folded_type = subgraph.value(node.inputs[0]).type;

    unary_kernel_fn convert_kernel = nullptr;
    if (input_type != folded_type) {
      if (analysis.consumers[node.inputs[0]].size() != 1) {
        return false;
      }
      convert_kernel =
          get_unary_kernel(ynn_unary_convert, input_type, folded_type);
      if (!convert_kernel) {
        return false;
      }
    }

    YNN_LOG_DEBUG() << "Folding multiply by " << mul->a << " into "
                    << to_string(unary->op);

    if (unary->op == ynn_unary_exp || unary->op == ynn_unary_expm1) {
      unary->params.exp.input_multiplier *= mul->a;
    } else if (unary->op == ynn_unary_approx_erf) {
      unary->params.approx_erf.input_multiplier *= mul->a;
    } else {
      unary->params.erf.input_multiplier *= mul->a;
    }

    if (input_type != folded_type) {
      define_unary(subgraph, *producer, mul->x_id, producer->outputs[0],
                   ynn_unary_convert, convert_kernel);
    } else {
      node.inputs[0] = mul->x_id;
    }

    analysis.invalidate();
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
    case ynn_unary_expm1:
    case ynn_unary_log:
    case ynn_unary_log1p:
      if (scalar_arithmetic->b != 0.0f) {
        // exp/log does not support output offset
        return false;
      }
      break;
    case ynn_unary_erf:
    case ynn_unary_approx_erf:
    case ynn_unary_tanh:
    case ynn_unary_approx_tanh:
    case ynn_unary_sine:
    case ynn_unary_cosine:
    case ynn_unary_poly3:
      break;
    default:
      return false;
  }

  // Check if the output is only used by this binary node.
  if (analysis.consumers[producer->outputs[0]].size() != 1) return false;

  ynn_type folded_type = subgraph.value(producer->outputs[0]).type;
  ynn_type output_type = subgraph.value(node.outputs[0]).type;

  unary_kernel_fn convert_kernel = nullptr;
  if (folded_type != output_type) {
    convert_kernel =
        get_unary_kernel(ynn_unary_convert, folded_type, output_type);
    if (!convert_kernel) return false;
  }

  YNN_LOG_DEBUG() << "Folding scalar arithmetic onto " << to_string(unary->op);

  unary->params.poly3.c0 *= scalar_arithmetic->a;
  unary->params.poly3.c1 *= scalar_arithmetic->a;
  if (unary->op == ynn_unary_poly3) {
    unary->params.poly3.c2 *= scalar_arithmetic->a;
    unary->params.poly3.c3 *= scalar_arithmetic->a;
  }
  unary->params.poly3.c0 += scalar_arithmetic->b;

  if (folded_type != output_type) {
    define_unary(subgraph, node, producer->outputs[0], node.outputs[0],
                 ynn_unary_convert, convert_kernel);
  } else {
    producer->outputs[0] = node.outputs[0];
    node.invalidate();
  }

  analysis.invalidate();
  return true;
}

// Rewrite iota(x) * A + B to fold the arithmetic into the params
bool fold_iota_output(ynn_subgraph& subgraph, ynn_node& node,
                      subgraph_analysis& analysis) {
  // iota supports dynamic begin/stride values too, so we could also fold non-
  // constant arithmetic into it, but it's messier to do that...
  auto scalar_arithmetic = is_scalar_arithmetic(subgraph, node);
  if (!scalar_arithmetic) return false;

  ynn_node* producer = analysis.producer_of(scalar_arithmetic->x_id);
  if (!producer || !std::holds_alternative<ynn_node::iota>(producer->op)) {
    return false;
  }

  YNN_LOG_DEBUG() << "Folding iota into scalar arithmetic";

  uint32_t output_id = node.outputs[0];
  const ynn_value& output = subgraph.value(output_id);
  if (ynn::type_is_integral(output.type)) {
    if (scalar_arithmetic->a != static_cast<int>(scalar_arithmetic->a) ||
        scalar_arithmetic->b != static_cast<int>(scalar_arithmetic->b)) {
      // We can't fuse arithmetic we can't exactly represent the scalars as
      // integers.
      return false;
    }
  }

  node = *producer;
  node.outputs[0] = output_id;
  ynn_node::iota& node_iota = std::get<ynn_node::iota>(node.op);
  node_iota.params.scale *= scalar_arithmetic->a;
  node_iota.params.offset *= scalar_arithmetic->a;
  node_iota.params.offset += scalar_arithmetic->b;
  return true;
}

bool rewrite_binary(ynn_subgraph& subgraph, ynn_node& node,
                    subgraph_analysis& analysis) {
  if (is_binary_node(node, ynn_binary_multiply) &&
      node.inputs[0] == node.inputs[1]) {
    const ynn_value& input = subgraph.value(node.inputs[0]);
    const ynn_value& output = subgraph.value(node.outputs[0]);
    unary_kernel_fn kernel =
        get_unary_kernel(ynn_unary_square, input.type, output.type);
    if (!kernel) return false;

    YNN_LOG_DEBUG() << "Rewriting multiply(x, x) to square(x)";
    define_unary(subgraph, node, node.inputs[0], node.outputs[0],
                 ynn_unary_square, kernel);
    return true;
  }
  return false;
}

// Rewrite reshape to static_transpose if possible.
bool rewrite_reshape(ynn_subgraph& subgraph, ynn_node& node,
                     subgraph_analysis& analysis) {
  const ynn_node::static_reshape* reshape =
      std::get_if<ynn_node::static_reshape>(&node.op);
  if (!reshape) return false;

  const ynn_value& input = subgraph.value(node.inputs[0]);
  ynn_value& output = subgraph.value(node.outputs[0]);

  std::vector<int32_t> permutation;

  size_t i = 0;
  size_t o = 0;
  while (o < output.rank()) {
    if (i < input.rank() &&
        slinky::prove_true(output.extent(o) == input.extent(i))) {
      permutation.push_back(i);
      ++i;
      ++o;
    } else if (slinky::prove_true(output.extent(o) == 1)) {
      permutation.push_back(input.rank());
      ++o;
    } else if (i < input.rank() && slinky::prove_true(input.extent(i) == 1)) {
      ++i;
    } else {
      return false;
    }
  }

  while (i < input.rank()) {
    if (slinky::prove_true(input.extent(i) == 1)) {
      ++i;
    } else {
      return false;
    }
  }

  YNN_LOG_DEBUG() << "Rewriting reshape to static_transpose";
  ynn::define_static_transpose(subgraph, node, std::move(permutation), input.id,
                               &output.id);
  return true;
}

// Rewrite transpose(transpose(x)) to transpose(x)
bool rewrite_transpose_transpose(ynn_subgraph& subgraph, ynn_node& node,
                                 subgraph_analysis& analysis) {
  auto* transpose2 = std::get_if<ynn_node::static_transpose>(&node.op);
  if (!transpose2) return false;

  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (!producer) return false;

  auto* transpose1 = std::get_if<ynn_node::static_transpose>(&producer->op);
  if (!transpose1) return false;

  // Only rewrite if the intermediate value is not an external output
  // and has no other consumers.
  uint32_t intermediate_id = producer->outputs[0];
  if (analysis.consumers[intermediate_id].size() != 1 ||
      subgraph.value(intermediate_id).is_external_output()) {
    return false;
  }

  const ynn_value& input = subgraph.value(producer->inputs[0]);
  const ynn_value& intermediate = subgraph.value(intermediate_id);

  int32_t R_in = input.rank();
  int32_t R_int = intermediate.rank();

  std::vector<int32_t> combined_perm(transpose2->permutation.size());
  for (size_t i = 0; i < combined_perm.size(); ++i) {
    int32_t idx = transpose2->permutation[i];
    if (idx < R_int) {
      combined_perm[i] = transpose1->permutation[idx];
      if (combined_perm[i] >= R_in) {
        combined_perm[i] = R_in;
      }
    } else {
      combined_perm[i] = R_in;
    }
  }

  YNN_LOG_DEBUG() << "Rewriting transpose(transpose(x)) to transpose(x)";

  uint32_t output_id = node.outputs[0];
  ynn::define_static_transpose(subgraph, node, std::move(combined_perm),
                               producer->inputs[0], &output_id,
                               transpose1->alias && transpose2->alias);

  producer->invalidate();
  analysis.invalidate();
  return true;
}

// Rewrite op(reduce_op(x, identity), y) to reduce_op(x, y)
bool rewrite_reduce_binary_identity(ynn_subgraph& subgraph, ynn_node& node,
                                    subgraph_analysis& analysis) {
  const ynn_node::binary_elementwise* binary =
      std::get_if<ynn_node::binary_elementwise>(&node.op);
  if (!binary) return false;

  for (int i : {0, 1}) {
    ynn_node* producer = analysis.producer_of(node.inputs[i]);
    if (!producer) continue;
    const ynn_node::reduce* reduce =
        std::get_if<ynn_node::reduce>(&producer->op);
    if (!reduce) continue;

    // Check if the reduction operator is compatible with the binary operator.
    if (binary->op == ynn_binary_add &&
        (reduce->op == ynn_reduce_sum ||
         reduce->op == ynn_reduce_sum_squared)) {
    } else if (binary->op == ynn_binary_min && reduce->op == ynn_reduce_min) {
    } else if (binary->op == ynn_binary_max && reduce->op == ynn_reduce_max) {
    } else {
      continue;
    }

    if (producer->inputs[1] != YNN_INVALID_VALUE_ID) {
      // The first reduce op has a non-identity initial value.
      continue;
    }

    if (analysis.consumers[producer->outputs[0]].size() != 1 ||
        subgraph.value(producer->outputs[0]).is_external_output()) {
      // The first reduce op is used by some other op.
      continue;
    }

    uint32_t y_id = node.inputs[1 - i];
    uint32_t reduce_output_id = producer->outputs[0];

    const ynn_value& reduce_output = subgraph.value(reduce_output_id);
    const ynn_value& y = subgraph.value(y_id);

    for (size_t d = 0; d < y.rank(); ++d) {
      if (slinky::is_constant(reduce_output.extent(d), 1) &&
          !slinky::is_constant(y.extent(d), 1)) {
        // y causes x to be broadcasted, which the reduce op does not do.
        return false;
      }
    }

    // Rewrite: binary_op(reduce_op(x, identity), y) -> reduce_op(x, y)
    YNN_LOG_DEBUG() << "Rewriting " << to_string(binary->op) << "("
                    << to_string(reduce->op) << "(x, identity), y) to "
                    << to_string(reduce->op) << "(x, y)";

    uint32_t x_id = producer->inputs[0];
    uint32_t output_id = node.outputs[0];

    node = std::move(*producer);
    node.inputs = {x_id, y_id};
    node.outputs[0] = output_id;
    producer->invalidate();
    return true;
  }
  return false;
}

bool rewrite_reduce_static_transpose(ynn_subgraph& subgraph, ynn_node& node,
                                     subgraph_analysis& analysis) {
  const ynn_node::reduce* reduce_op = std::get_if<ynn_node::reduce>(&node.op);
  if (!reduce_op) return false;

  ynn_node* producer = analysis.producer_of(node.inputs[0]);
  if (!producer) return false;

  ynn_node* transpose = producer;
  const ynn_node::static_transpose* transpose_op =
      std::get_if<ynn_node::static_transpose>(&transpose->op);
  if (!transpose_op) return false;

  if (analysis.consumers[transpose->outputs[0]].size() != 1 ||
      subgraph.value(transpose->outputs[0]).is_external_output()) {
    return false;
  }

  // We have reduce(static_transpose(x, P), R). We can rewrite this to
  // static_transpose(reduce(x, R'), P').
  uint32_t x_id = transpose->inputs[0];
  uint32_t init_id = node.inputs[1];
  uint32_t y_id = node.outputs[0];

  const ynn_value& x = subgraph.value(x_id);

  ynn::axes_set new_axes;
  for (size_t d = 0; d < transpose_op->permutation.size(); ++d) {
    if (reduce_op->k_dims[d]) {
      if (transpose_op->permutation[d] >= x.rank()) {
        // This seems like it should work, but probably is of marginal value to
        // implement.
        YNN_LOG_DEBUG()
            << "Not rewriting reduce(static_transpose(x)) because the "
               "reduced dimension is a new dimension.";
        return false;
      }
      new_axes[transpose_op->permutation[d]] = true;
    }
  }

  std::vector<int32_t> new_perm;
  if (reduce_op->keep_dims) {
    new_perm = transpose_op->permutation;
  } else {
    std::vector<int> num_reduced_before(new_axes.size() + 1, 0);
    for (size_t i = 0; i < new_axes.size(); ++i) {
      num_reduced_before[i + 1] =
          num_reduced_before[i] + (new_axes.test(i) ? 1 : 0);
    }
    for (size_t d = 0; d < transpose_op->permutation.size(); ++d) {
      if (!reduce_op->k_dims[d]) {
        new_perm.push_back(transpose_op->permutation[d] -
                           num_reduced_before[transpose_op->permutation[d]]);
      }
    }
  }

  // Save properties before clearing
  ynn_reduce_operator op = reduce_op->op;
  bool keep_dims = reduce_op->keep_dims;
  bool alias = transpose_op->alias;

  bool is_identity = true;
  for (size_t i = 0; i < new_perm.size(); ++i) {
    if (new_perm[i] != i) {
      is_identity = false;
      break;
    }
  }

  if (is_identity) {
    transpose->invalidate();
    node.checks.clear();
    ynn::define_reduce(subgraph, node, op, new_axes, x_id, init_id, &y_id,
                       keep_dims);
  } else {
    // Clear checks on the old nodes to avoid runtime failure
    transpose->checks.clear();
    node.checks.clear();

    // Redefine transpose node as reduce node
    uint32_t r_id = YNN_INVALID_VALUE_ID;
    ynn::define_reduce(subgraph, *transpose, op, new_axes, x_id, init_id, &r_id,
                       keep_dims);

    // Redefine the old reduce node as a transpose node
    ynn::define_static_transpose(subgraph, node, std::move(new_perm), r_id,
                                 &y_id, alias);
  }

  return true;
}

// Rewrite static_transpose(static_broadcast(x)) to
// static_broadcast(static_transpose(x))
bool rewrite_transpose_broadcast(ynn_subgraph& subgraph, ynn_node& node,
                                 subgraph_analysis& analysis) {
  ynn_node::static_transpose* transpose =
      std::get_if<ynn_node::static_transpose>(&node.op);
  if (!transpose) return false;

  ynn_node* broadcast_node = analysis.producer_of(node.inputs[0]);
  if (!broadcast_node) return false;

  ynn_node::static_broadcast* broadcast =
      std::get_if<ynn_node::static_broadcast>(&broadcast_node->op);
  if (broadcast == nullptr) return false;

  // Only do this rewrite if we won't break other consumers of the broadcast.
  if (analysis.consumers[broadcast_node->outputs[0]].size() != 1 ||
      subgraph.value(broadcast_node->outputs[0]).is_external_output()) {
    return false;
  }

  uint32_t x_id = broadcast_node->inputs[0];
  uint32_t z_id = node.outputs[0];

  const size_t rank_y = broadcast->new_dims.size();
  std::vector<int32_t> perm = transpose->permutation;

  const ynn_value& z = subgraph.value(z_id);
  std::vector<size_t> new_dims(z.rank());
  for (size_t i = 0; i < new_dims.size(); ++i) {
    int32_t src_dim_in_y = perm[i];
    if (src_dim_in_y < rank_y) {
      new_dims[i] = broadcast->new_dims[src_dim_in_y];
    } else {
      new_dims[i] = 0;
    }
  }

  YNN_LOG_DEBUG() << "Rewriting transpose(static_broadcast(x)) to "
                     "static_broadcast(transpose(x))";

  broadcast_node->checks.clear();
  node.checks.clear();

  // Redefine B (broadcast_node) to be T' (new_transpose)
  uint32_t y_prime_id = YNN_INVALID_VALUE_ID;
  ynn::define_static_transpose(subgraph, *broadcast_node, std::move(perm), x_id,
                               &y_prime_id, transpose->alias);

  // Redefine T (node) to be B' (new_broadcast)
  ynn::define_static_broadcast(subgraph, node, std::move(new_dims), y_prime_id,
                               &z_id);

  analysis.invalidate();

  return true;
}

// Rewrites sum(a * b) to dot(a, b).
// dot(a, b) is sum(a(., k1, k2, k3, i, ...) * b(j, k1, k2, k3, ., ...)) where .
// indicates a new dimension. This rewrite looks for sums that can be transposed
// and have the new dimensions inserted to match the dot layout (and the result
// then transposed back to the sum's output layout).
bool rewrite_sum_to_dot(ynn_subgraph& subgraph, ynn_node& node,
                        subgraph_analysis& analysis) {
  const ynn_node::reduce* reduce_op = std::get_if<ynn_node::reduce>(&node.op);
  if (!reduce_op || reduce_op->op != ynn_reduce_sum) {
    return false;
  }

  ynn_node* mul_node = analysis.producer_of(node.inputs[0]);
  if (!mul_node || !is_binary_node(*mul_node, ynn_binary_multiply)) {
    return false;
  }

  if (analysis.consumers[mul_node->outputs[0]].size() != 1 ||
      subgraph.value(mul_node->outputs[0]).is_external_output()) {
    return false;
  }

  if (reduce_op->k_dims.count() > 1) {
    YNN_LOG_DEBUG()
        << "not rewriting sum(a*b) to dot(a, b) because the sum has "
           "more than 1 reduction dimension.";
    return false;
  }

  uint32_t a_id = mul_node->inputs[0];
  uint32_t b_id = mul_node->inputs[1];

  if (slinky::prove_true(subgraph.value(b_id).extent(0) == 1)) {
    // B already has a broadcast where we want it in A, swap them to make
    // the transposes less likely to be needed.
    // TODO: dsharlet - We could probably be smarter about this optimization.
    std::swap(a_id, b_id);
  }

  const ynn_value& a = subgraph.value(a_id);
  const ynn_value& b = subgraph.value(b_id);

  if (a.type != ynn_type_fp32 || b.type != ynn_type_fp32) {
    // TODO: dsharlet - Support more types.
    YNN_LOG_DEBUG() << "not rewriting sum(a*b) to dot(a, b) because the inputs "
                       "are not fp32.";
    return false;
  }

  const int max_rank = std::max(a.rank(), b.rank());
  std::vector<int32_t> perm_a(max_rank);
  std::iota(perm_a.begin(), perm_a.end(), 0);

  // 1. permute the operation such that the reduction dimensions are in the
  // first num_k_dims dimensions:
  // sum(a(k1, k2, ..., d1, d2, ...) * b(k1, k2, ..., d1, d2, ...))
  int num_k_dims = 0;
  for (size_t i = 0; i < max_rank; ++i) {
    if (reduce_op->k_dims[i]) {
      if (!slinky::prove_true(a.extent(i) == b.extent(i))) {
        // Don't handle broadcasted reduction dimensions.
        return false;
      }
      std::swap(perm_a[num_k_dims++], perm_a[i]);
    }
  }

  // Helper to find a broadcast dimension in a, and not a broadcast in b.
  // TODO: dsharlet - This should find the broadcast with the biggest extent in
  // the other buffer to maximize the value of the dot.
  auto find_broadcast = [&](const ynn_value& a, const ynn_value& b,
                            ynn::span<const int32_t> perm,
                            int exclude = -1) -> int {
    for (size_t i = num_k_dims; i < perm.size(); ++i) {
      if (static_cast<int>(i) != exclude &&
          slinky::prove_true(a.extent(perm[i]) == 1) &&
          !slinky::prove_true(b.extent(perm[i]) == 1)) {
        return i;
      }
    }
    return -1;
  };
  // 2. Find broadcasts from b to move into the i dimension for a, and from a to
  // move into the j dimension for b.
  // sum(a(., k1, k2, i, ..., d1, d2, ...) * b(j, k1, k2, ., ..., d1, d2, ...))
  int i_dim = find_broadcast(b, a, perm_a);
  int j_dim = find_broadcast(a, b, perm_a, i_dim);
  if (i_dim == -1 && j_dim == -1) {
    YNN_LOG_DEBUG()
        << "Not rewriting sum(a*b) to dot(a, b) because there are no "
           "broadcast dimensions.";
    return false;
  }

  std::vector<int32_t> perm_b = perm_a;

  if (i_dim != -1) {
    std::rotate(perm_a.begin() + num_k_dims, perm_a.begin() + i_dim,
                perm_a.begin() + i_dim + 1);
  } else {
    perm_a.insert(perm_a.begin() + num_k_dims, YNN_MAX_TENSOR_RANK);
  }
  if (j_dim != -1) {
    perm_a.erase(perm_a.begin() +
                 (j_dim < i_dim || i_dim == -1 ? j_dim + 1 : j_dim));
  }

  if (j_dim != -1) {
    std::rotate(perm_b.begin(), perm_b.begin() + j_dim,
                perm_b.begin() + j_dim + 1);
  } else {
    perm_b.insert(perm_b.begin(), YNN_MAX_TENSOR_RANK);
  }
  if (i_dim != -1) {
    perm_b.erase(perm_b.begin() +
                 (i_dim < j_dim || j_dim == -1 ? i_dim + 1 : i_dim));
  }

  YNN_LOG_DEBUG() << "Rewriting sum(a * b) to dot(a, b)";

  // Do the transposes of the inputs.
  uint32_t a_t_id = YNN_INVALID_VALUE_ID;
  uint32_t b_t_id = YNN_INVALID_VALUE_ID;
  ynn_node transpose_a, transpose_b;
  define_static_transpose(subgraph, transpose_a, perm_a, a_id, &a_t_id);
  define_static_transpose(subgraph, transpose_b, perm_b, b_id, &b_t_id);
  if (a_t_id != a_id) subgraph.add_node(std::move(transpose_a));
  if (b_t_id != b_id) subgraph.add_node(std::move(transpose_b));

  // 3. Construct the output permutation.
  uint32_t output_id = node.outputs[0];
  std::vector<int32_t> perm_output;
  const int dot_output_rank =
      (perm_a.size() == static_cast<size_t>(num_k_dims))
          ? static_cast<int>(perm_b.size()) - num_k_dims
          : static_cast<int>(std::max(perm_a.size(), perm_b.size())) + 1 -
                num_k_dims;
  bool is_perm_output_identity =
      subgraph.value(output_id).rank() == dot_output_rank;

  auto add_to_perm_output = [&](int32_t dim) {
    is_perm_output_identity =
        is_perm_output_identity && (dim == perm_output.size());
    perm_output.push_back(dim);
  };
  for (int d = 0; d < max_rank; ++d) {
    if (reduce_op->k_dims[d]) {
      if (reduce_op->keep_dims) {
        add_to_perm_output(YNN_MAX_TENSOR_RANK);
      }
    } else {
      // Find where original dimension is in the dot output.
      if (perm_b[0] == static_cast<int32_t>(d)) {
        add_to_perm_output(0);
      } else if (perm_a[num_k_dims] == static_cast<int32_t>(d)) {
        add_to_perm_output(1);
      } else {
        int dot_idx = -1;
        for (int k = num_k_dims + 1; k < perm_a.size(); ++k) {
          if (perm_a[k] == static_cast<int32_t>(d)) {
            dot_idx = k - num_k_dims + 1;
            break;
          }
        }
        if (dot_idx != -1) {
          add_to_perm_output(dot_idx);
        } else {
          // This must be a dimension that was 1 in both inputs, and was not
          // selected as i or j, and thus is not in the dot output at all.
          add_to_perm_output(YNN_MAX_TENSOR_RANK);
        }
      }
    }
  }

  // Prepare the initializer
  uint32_t init_id = node.inputs[1];
  uint32_t init_t_id = YNN_INVALID_VALUE_ID;
  if (init_id != YNN_INVALID_VALUE_ID) {
    std::vector<int32_t> perm_init(dot_output_rank, YNN_MAX_TENSOR_RANK);
    for (int d = 0; d < static_cast<int>(perm_output.size()); ++d) {
      if (perm_output[d] != YNN_MAX_TENSOR_RANK) {
        perm_init[perm_output[d]] = d;
      }
    }
    ynn_node transpose_init;
    define_static_transpose(subgraph, transpose_init, perm_init, init_id,
                            &init_t_id);
    if (init_t_id != init_id) {
      subgraph.add_node(std::move(transpose_init));
    }
  }

  // Do the dot.
  uint32_t dot_id = is_perm_output_identity ? output_id : YNN_INVALID_VALUE_ID;
  ynn_define_dot(&subgraph, num_k_dims, a_t_id, b_t_id, init_t_id, &dot_id,
                 /*flags=*/0);

  node.invalidate();
  if (!is_perm_output_identity) {
    // Transpose the output back to the original layout.
    ynn_node transpose;
    define_static_transpose(subgraph, transpose, perm_output, dot_id,
                            &output_id);
    subgraph.add_node(std::move(transpose));
  }

  subgraph.topological_sort();
  analysis.invalidate();
  return true;
}

bool rewrite_fast_math(ynn_subgraph& subgraph, ynn_node& node,
                       subgraph_analysis& analysis) {
  if ((subgraph.flags & YNN_FLAG_FAST_MATH) == 0) return false;

  ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&node.op);
  if (!unary) return false;

  struct FastMathOpRewrite {
    ynn_unary_operator op;
    ynn_unary_operator fast_op;
  };
  constexpr FastMathOpRewrite kFastMathRewrites[] = {
      {ynn_unary_erf, ynn_unary_approx_erf},
      {ynn_unary_tanh, ynn_unary_approx_tanh},
  };

  for (const auto& rewrite : kFastMathRewrites) {
    if (unary->op == rewrite.op) {
      const ynn_value& input = subgraph.value(node.inputs[0]);
      const ynn_value& output = subgraph.value(node.outputs[0]);
      const ynn::unary_kernel_fn kernel =
          ynn::get_unary_kernel(rewrite.fast_op, input.type, output.type);
      if (kernel) {
        YNN_LOG_DEBUG() << "Rewriting " << to_string(rewrite.op) << " to "
                        << to_string(rewrite.fast_op) << " (fast math)";
        unary_params new_params = unary->params;

        ynn::define_unary(subgraph, node, node.inputs[0], node.outputs[0],
                          rewrite.fast_op, kernel, new_params);
        return true;
      }
    }
  }
  return false;
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
                ynn::fold_iota_output(*this, node, analysis) ||
                ynn::rewrite_binary(*this, node, analysis) ||
                ynn::rewrite_reduce_binary_identity(*this, node, analysis) ||
                ynn::rewrite_expand_dims_reduce(*this, node, analysis) ||
                ynn::rewrite_reshape(*this, node, analysis) ||
                ynn::rewrite_transpose_transpose(*this, node, analysis) ||
                ynn::rewrite_divide_sqrt(*this, node, analysis) ||
                ynn::rewrite_sum_to_dot(*this, node, analysis) ||
                ynn::rewrite_ternary(*this, node, analysis) ||
                ynn::rewrite_binary_convert(*this, node, analysis) ||
                ynn::rewrite_convert_elementwise(*this, node, analysis) ||
                ynn::rewrite_negate_multiply(*this, node, analysis) ||
                ynn::rewrite_dequantize_dot(*this, node, analysis) ||
                ynn::rewrite_dequantize_dot_add(*this, node, analysis) ||
                ynn::rewrite_dequantize_dot_convert(*this, node, analysis) ||
                ynn::rewrite_get_tensor_shape_of_unary(*this, node, analysis) ||
                ynn::move_broadcast_to_output(*this, node, analysis) ||
                ynn::remove_broadcast(*this, node, analysis) ||
                ynn::rewrite_transpose_broadcast(*this, node, analysis) ||
                ynn::remove_static_broadcast_from_elementwise(*this, node,
                                                              analysis) ||
                ynn::rewrite_transpose_stencil_copy(*this, node, analysis) ||
                ynn::rewrite_reduce_sum_of_squared(*this, node, analysis) ||
                ynn::rewrite_reduce_convert(*this, node, analysis) ||
                ynn::rewrite_reduce_static_transpose(*this, node, analysis) ||
                ynn::rewrite_fast_math(*this, node, analysis) || false;

      if (!analysis.is_valid) {
        break;
      }
    }
    if (changed) invalidate_dead_values();
  } while (changed);

  do {
    subgraph_analysis analysis(*this);
    changed = ynn::rewrite_subgraph_for_unary_lut(*this, analysis);
    if (changed) invalidate_dead_values();
  } while (changed);

  do {
    subgraph_analysis analysis(*this);
    changed = false;
    for (ynn_node& node : nodes) {
      if (!node.is_valid()) continue;

      changed = changed || ynn::fuse_converts(*this, node, analysis) ||
                ynn::fuse_quantize(*this, node, analysis);

      if (!analysis.is_valid) {
        break;
      }
    }
    if (changed) invalidate_dead_values();
  } while (changed);

  return ynn_status_success;
}
