// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/utils.h"

#include <cstddef>
#include <cstdint>
#include <variant>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

template <typename T>
bool is_broadcast_op(const ynn_node& node) {
  const T* op = std::get_if<T>(&node.op);
  return op && op->axes.any();
}

bool allow_in_place(uint32_t input_id, uint32_t output_id,
                    const ynn_subgraph& subgraph) {
  if (input_id == YNN_INVALID_VALUE_ID) return false;

  const ynn_value& a = subgraph.value(input_id);
  const ynn_value& x = subgraph.value(output_id);

  if (x.rank() != a.rank()) {
    return false;
  }

  if (type_size_bytes(a.type) != type_size_bytes(x.type) ||
      type_element_count(a.type) != type_element_count(x.type)) {
    // The types are not the same size, we can't compute in place.
    return false;
  }

  for (size_t d = 0; d < x.rank(); ++d) {
    if (!a.extents[d].defined() && x.extents[d].defined()) {
      // The input is broadcasted (and the output is not), don't allow computing
      // in place.
      return false;
    }
  }

  const ynn_node* producer = subgraph.get_producer(input_id);
  if (!producer) {
    // This input is not produced in the pipeline, we can't overwrite the
    // input (and slinky wouldn't let us anyways).
    return false;
  }

  if (is_broadcast_op<ynn_node::broadcast>(*producer) ||
      is_broadcast_op<ynn_node::broadcast_like>(*producer)) {
    // We can't compute in place with a broadcast input.
    return false;
  }

  return true;
}

}  // namespace ynn
