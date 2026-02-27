// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_TEST_MATCHERS_H_
#define XNNPACK_YNNPACK_SUBGRAPH_TEST_MATCHERS_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/base/span.h"

// This causes gmock to print the subgraph instead of just a hex encoded dump of
// the memory. It needs to be in the global namespace for argument dependent
// lookup to work.
inline std::ostream& operator<<(std::ostream& os, const ynn_subgraph& g) {
  os << "subgraph:" << std::endl;
  g.dump(os);
  return os;
}

namespace ynn {

namespace internal {

template <typename T>
const ynn_subgraph* GetSubgraph(const T& arg,
                                testing::MatchResultListener* listener) {
  const ynn_subgraph* subgraph = nullptr;
  if constexpr (std::is_pointer_v<std::decay_t<T>>) {
    subgraph = arg;
  } else {
    subgraph = &arg;
  }

  if (subgraph == nullptr) {
    *listener << "subgraph is null";
    return nullptr;
  }
  return subgraph;
}

}  // namespace internal

// Checks that the given subgraph has the given number of valid nodes.
//
// Example:
//   EXPECT_THAT(subgraph, HasValidNodeCount(3));
MATCHER_P(HasValidNodeCount, count, "") {
  const ynn_subgraph* subgraph = internal::GetSubgraph(arg, result_listener);
  if (subgraph == nullptr) {
    return false;
  }

  int actual_count =
      std::count_if(subgraph->nodes.begin(), subgraph->nodes.end(),
                    [](const ynn_node& node) { return node.is_valid(); });
  if (actual_count != count) {
    *result_listener << "has " << actual_count << " valid nodes";
    return false;
  }
  return true;
}

// Checks that the given subgraph has the given number of valid values.
//
// Example:
//   EXPECT_THAT(subgraph, HasValidValueCount(3));
MATCHER_P(HasValidValueCount, count, "") {
  const ynn_subgraph* subgraph = internal::GetSubgraph(arg, result_listener);
  if (subgraph == nullptr) {
    return false;
  }

  int actual_count =
      std::count_if(subgraph->values.begin(), subgraph->values.end(),
                    [](const ynn_value& value) { return value.is_valid(); });
  if (actual_count != count) {
    *result_listener << "has " << actual_count << " valid values";
    return false;
  }
  return true;
}

// Checks that the given node has the given number of inputs.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph), HasInputCount(2));
MATCHER_P(HasInputCount, count, "") {
  if (arg.inputs.size() != count) {
    *result_listener << "has " << arg.inputs.size() << " inputs";
    return false;
  }
  return true;
}

// Checks that the given node is a LUT.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph), IsLut());
MATCHER(IsLut, "") { return std::holds_alternative<ynn_node::lut>(arg.op); }

// Checks that the given node is a binary elementwise with the given operator.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph), IsBinary(ynn_binary_add));
MATCHER_P(IsBinary, op_type, "") {
  const ynn_node::binary_elementwise* binary =
      std::get_if<ynn_node::binary_elementwise>(&arg.op);
  return binary && binary->op == op_type;
}

// Checks that the given node is a ternary elementwise with the given operator.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph), IsTernary(ynn_ternary_add));
MATCHER_P(IsTernary, op_type, "") {
  const ynn_node::ternary_elementwise* ternary =
      std::get_if<ynn_node::ternary_elementwise>(&arg.op);
  return ternary && ternary->op == op_type;
}

// Checks that the given node is a reduce with the given operator.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph), IsReduce(ynn_reduce_sum));
MATCHER_P(IsReduce, op_type, "") {
  const ynn_node::reduce* reduce = std::get_if<ynn_node::reduce>(&arg.op);
  return reduce && reduce->op == op_type;
}

// Checks that the given node is a unary elementwise with the given operator.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph), IsUnary(ynn_unary_negate));
MATCHER_P(IsUnary, op_type, "") {
  const ynn_node::unary_elementwise* unary =
      std::get_if<ynn_node::unary_elementwise>(&arg.op);
  return unary && unary->op == op_type;
}

// Checks that the given node is a transpose_a with the given tile_k and m_dim.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph), IsTransposeA(16, 2));
MATCHER_P2(IsTransposeA, tile_k, m_dim, "") {
  const ynn_node::transpose_a* transpose =
      std::get_if<ynn_node::transpose_a>(&arg.op);
  return transpose && transpose->tile_k == tile_k && transpose->m_dim == m_dim;
}

// Checks that the given node is a stencil copy with the given stencils.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph), IsStencilCopy({{0, 1, 2, 3, 4},
//                                                         {1, 2, 3, 4, 5}}));
MATCHER_P(IsStencilCopy, stencils, "") {
  const ynn_node::stencil_copy* stencil_copy =
      std::get_if<ynn_node::stencil_copy>(&arg.op);
  if (!stencil_copy) return false;
  if (stencil_copy->stencils.size() != stencils.size()) return false;
  for (size_t i = 0; i < stencils.size(); ++i) {
    const auto& a = stencil_copy->stencils[i];
    const auto& b = stencils[i];
    if (a.axis != b.axis || a.new_axis != b.new_axis || a.extent != b.extent ||
        a.stride != b.stride || a.dilation != b.dilation) {
      return false;
    }
  }
  return true;
}

// Checks that the given value ID is valid in the given subgraph.
//
// Example:
//   EXPECT_THAT(subgraph, HasValidValueId(x_id));
MATCHER_P(HasValidValueId, value_id, "") {
  const ynn_subgraph* subgraph = internal::GetSubgraph(arg, result_listener);
  if (subgraph == nullptr) {
    return false;
  }
  if (!subgraph->value(value_id).is_valid()) {
    *result_listener << "value " << value_id << " is invalid";
    return false;
  }
  return true;
}

// Checks that the given value ID is valid in the given subgraph.
//
// Example:
//   EXPECT_THAT(x_id, IsValidValueIn(subgraph));
MATCHER_P(IsValidValueIn, subgraph, "") {
  const ynn_subgraph* s = internal::GetSubgraph(subgraph, result_listener);
  if (s == nullptr) {
    return false;
  }
  if (!s->value(arg).is_valid()) {
    *result_listener << "value " << arg << " is invalid";
    return false;
  }
  return true;
}

MATCHER_P(HasValidValueIdsImpl, ids, "") {
  const ynn_subgraph* subgraph = internal::GetSubgraph(arg, result_listener);
  if (subgraph == nullptr) {
    return false;
  }
  bool all_valid = true;
  for (uint32_t id : ids) {
    if (!subgraph->value(id).is_valid()) {
      if (!all_valid) *result_listener << "; ";
      *result_listener << "value " << id << " is invalid";
      all_valid = false;
    }
  }
  return all_valid;
}

// Checks that the subgraph has valid values with the given IDs.
//
// Example:
//   EXPECT_THAT(subgraph, HasValidValueIds(x_id, y_id));
template <typename... Args>
auto HasValidValueIds(Args... ids) {
  return HasValidValueIdsImpl(
      std::vector<uint32_t>{static_cast<uint32_t>(ids)...});
}

// Checks that the node's inputs are exactly the given value IDs, in order.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph),
//               InputsAre(x_id, YNN_INVALID_VALUE_ID));
template <typename... Args>
auto InputsAre(Args&&... args) {
  return ::testing::Field(&ynn_node::inputs,
                          ::testing::ElementsAre(std::forward<Args>(args)...));
}

// Checks that the node's inputs include the given value IDs, in any order.
//
// Example:
//   EXPECT_THAT(ProducerOf(y_id, subgraph),
//               InputsInclude(x_id, YNN_INVALID_VALUE_ID));
template <typename... Args>
auto InputsInclude(Args&&... args) {
  return ::testing::Field(
      &ynn_node::inputs,
      ::testing::IsSupersetOf({std::forward<Args>(args)...}));
}

// Returns the node that produces the given value ID.
//
// Example:
//   const ynn_node& node = ProducerOf(y_id, subgraph);
inline const ynn_node& ProducerOf(uint32_t value_id,
                                  const ynn_subgraph& subgraph) {
  const ynn_node* node = subgraph.get_producer(value_id);
  assert(node);
  return *node;
}

// Returns a span of values for static ynn_value.
//
// Example:
//   EXPECT_THAT(ValuesIn<float>(subgraph.value(x_id)),
//               testing::ElementsAre(1.0f, 2.0f, 3.0f));
template <typename T>
slinky::span<const T> ValuesIn(const ynn_value& value) {
  if (!value.is_static()) {
    return {};
  }
  size_t count = value.data->size_bytes() / sizeof(T);
  if (count == 0) {
    return {};
  }
  return slinky::span<const T>(static_cast<const T*>(value.data->base), count);
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_TEST_MATCHERS_H_
