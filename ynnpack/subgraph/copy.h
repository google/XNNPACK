#ifndef XNNPACK_YNNPACK_SUBGRAPH_COPY_H_
#define XNNPACK_YNNPACK_SUBGRAPH_COPY_H_

#include <cstdint>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

void define_copy(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_id,
                 uint32_t output_id, uint32_t flags);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_COPY_H_
