// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_SLINKY_H_
#define XNNPACK_YNNPACK_SUBGRAPH_SLINKY_H_

#include <cstddef>
#include <string>
#include <vector>

#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

#define XNN_ALLOCA(T, N) reinterpret_cast<T*>(alloca((N) * sizeof(T)))

namespace ynn {

inline int axis_to_slinky_dim(int rank, int axis) {
  return axis < 0 ? -(axis + 1) : rank - (axis + 1);
}

// This helper replaces extent 1 dimensions with unbounded dimensions of stride
// 0, which allows those dimensions to broadcast.
void allow_broadcasting(slinky::raw_buffer& buf);

slinky::buffer_expr_ptr make_buffer_expr(slinky::var sym, int rank,
                                         slinky::expr elem_size);
slinky::buffer_expr_ptr make_buffer_expr(slinky::node_context& ctx,
                                         const std::string& name, int rank,
                                         slinky::expr elem_size);

// Constrain the strides of the first `dims` dimensions of a buffer such that
// the strides are dense (no padding between dimensions) and in the order that
// XNNPACK/TFlite expect.
void require_contiguous(slinky::buffer_expr& buf,
                        size_t dims = std::numeric_limits<size_t>::max());

// Make an array of dimensions that is begin, 1, ... end - 1.
std::vector<slinky::var> make_dims(int begin, int end,
                                   slinky::node_context& ctx);
std::vector<slinky::var> make_dims(int rank, slinky::node_context& ctx);

// Get bounds for an elementwise dimension, allowing for the possibility of a
// broadcast.
slinky::interval_expr elementwise_bounds(slinky::var dim,
                                         const slinky::expr& extent);

// Get bounds for a dimension we require the entire extent of, allowing for the
// possibility of a broadcast.
slinky::interval_expr all_bounds(const slinky::expr& extent);

// Make an array of bounds that is point(i) for i in dims in [begin, end),
// accounting for the possibility of broadcasting.
slinky::box_expr make_elementwise_bounds(
    const std::vector<slinky::var>& dims,
    const std::vector<slinky::expr>& extents, size_t begin, size_t end);

slinky::box_expr make_elementwise_bounds(
    const std::vector<slinky::var>& dims,
    const std::vector<slinky::expr>& extents);

slinky::interval_expr make_broadcast_bounds(const slinky::var& dim,
                                            const slinky::expr& src_extent,
                                            const slinky::expr& dst_extent,
                                            bool no_broadcast = false);

// Like make_elementwise_bounds, but inserts conditionals to handle
// broadcasting if needed.
slinky::box_expr make_broadcast_bounds(
    std::vector<slinky::var> dims, const std::vector<slinky::expr>& src_extents,
    const std::vector<slinky::expr>& dst_extents, bool no_broadcast = false);

// A loop split for a given function.
struct scheduling_split {
  slinky::var var;
  slinky::expr step;
  slinky::expr workers = slinky::loop::parallel;
  // The axis of the extent which was used to compute this split.
  slinky::index_t axis;
  // If this is true the corresponding loop is required to have this specific
  // step, i.e. it can not get scheduled in the loop of the other function
  // unless the step matches or the other loop doesn't have required step yet.
  // In the latter case this step will override the existing step of that loop.
  bool step_is_required = false;
};

// A scheduling information for a buffer -- it's expected to be attached to the
// scheduling_info of the function.
struct scheduled_buffer {
  // This potentially could be numeric_limit::max or something, but it's
  // convenient to do some math with it, so pick something smaller to avoid
  // overflows.
  static constexpr slinky::index_t root = 1000;
  slinky::buffer_expr_ptr buffer;
  // The location to store buffer at with respect to its producer compute_at
  // location:
  // * if it's 0 then it will be stored at the same loop level it's computed at.
  // * if it's root it's an outermost location.
  slinky::index_t store_at_min_depth = root;
};

struct scheduling_info {
  // This value is large enough to always be outside of any reasonable number
  // of loops.
  static constexpr slinky::index_t root = 1000;

  // A set of loop splits for a given function.
  std::vector<scheduling_split> loop_splits;
  std::vector<scheduled_buffer> scheduled_buffers;

  // This is an ID of the buffer whose extents were used to compute this
  // scheduling info.
  uint32_t base_buffer_id = 0;

  bool force_root = false;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_SLINKY_H_
