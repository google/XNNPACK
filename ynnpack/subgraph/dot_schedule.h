// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_DOT_SCHEDULE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_DOT_SCHEDULE_H_

#include <array>
#include <cstddef>
#include <vector>

#include "ynnpack/subgraph/slinky.h"
#include "slinky/runtime/expr.h"

struct ynn_runtime;
struct ynn_runtime_value;

namespace ynn {

// Per-core cache budgets {L1, L2, L3}, in bytes, that `schedule_dot` tiles the
// loops of each task to fit: 1/2 of L1 for the micro-kernel's reused panel,
// leaving room for the streamed operand and the accumulators, and 1/4 of L2 and
// L3, leaving room for the other operand, C, hardware prefetcher streams and
// set-associativity conflicts. A per-core L3 share that is no larger than L2
// (e.g. on many-core x86 and ARM servers) can't hold more reused data than L2,
// so it is not used.
const std::array<size_t, 3>& get_dot_cache_budgets();

// L1 and L2 act as simultaneous upper bounds on the tile working set: L2/4
// leaves L2 headroom for the other operand, accumulator, and set-associativity
// conflicts on smaller-L2 CPUs, while L1*4 prevents L1 thrashing and preserves
// enough parallel tiles on large-L2 (1-2 MiB) CPUs.
size_t get_dot_l1_l2_budget();

scheduling_info make_dot_schedule(
    ynn_runtime& runtime, const ynn_runtime_value& input_a,
    const ynn_runtime_value& input_b, const ynn_runtime_value& output,
    const std::vector<slinky::var>& all_dims,
    const std::vector<slinky::expr>& all_extents,
    const std::vector<slinky::var>& output_dims, size_t num_k_dims,
    bool packed_b, bool transposed_a, int block_n_unpacked,
    float seconds_per_mac);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_DOT_SCHEDULE_H_
