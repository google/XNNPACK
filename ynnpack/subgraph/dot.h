#ifndef XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
#define XNNPACK_YNNPACK_SUBGRAPH_DOT_H_

#include <cstdint>

#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/buffer.h"

namespace ynn {

void define_transpose_a(ynn_subgraph& subgraph, ynn_node& node,
                        slinky::index_t tile_k, uint32_t input_a_id,
                        uint32_t output_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
