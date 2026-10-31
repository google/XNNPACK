#ifndef XNNPACK_YNNPACK_SUBGRAPH_COPY_H_
#define XNNPACK_YNNPACK_SUBGRAPH_COPY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/expr.h"

namespace ynn {

void define_copy(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_id,
                 uint32_t output_id, uint32_t flags);

void define_static_broadcast(ynn_subgraph& subgraph, ynn_node& node,
                             std::vector<size_t> new_dims, uint32_t input_id,
                             uint32_t* output_id);

// Defines a reshape of `input_id` to a value with extents `output_extents`.
// `groups` describes the structure of the reshape (see `ynn_node::reshape`).
// This does not check that the reshape is valid. If the reshape only adds or
// removes dimensions of extent 1, this defines a `static_transpose` instead,
// which may not define a node at all (see `define_static_transpose`).
void define_reshape(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_id,
                    uint32_t* output_id,
                    std::vector<slinky::expr> output_extents,
                    std::vector<ynn_node::reshape::group> groups);

// Given reshapes `a` and `b`, returns the groups of the reshape `b(a(x))`.
std::vector<ynn_node::reshape::group> compose_reshape_groups(
    const std::vector<ynn_node::reshape::group>& a,
    const std::vector<ynn_node::reshape::group>& b);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_COPY_H_
