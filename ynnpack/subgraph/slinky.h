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

#include <cassert>
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

namespace internal {

// Helper to apply a function to pairs of elements in a parameter pack.
template <typename F, typename A, typename B>
void apply_to_pairs(F&& f, A&& a, B&& b) {
  f(std::forward<A>(a), std::forward<B>(b));
}

template <typename F, typename A, typename B, typename... Pairs>
void apply_to_pairs(F&& f, A&& a, B&& b, Pairs&&... pairs) {
  f(std::forward<A>(a), std::forward<B>(b));
  apply_to_pairs(std::forward<F>(f), std::forward<Pairs>(pairs)...);
}

// Helper to apply a predicate to pairs of elements in a parameter pack and
// return true if the predicate is true for all pairs.
template <typename F, typename A, typename B>
bool all_of_pairs(F&& f, A&& a, B&& b) {
  return f(std::forward<A>(a), std::forward<B>(b));
}

template <typename F, typename A, typename B, typename... Pairs>
bool all_of_pairs(F&& f, A&& a, B&& b, Pairs&&... pairs) {
  return f(std::forward<A>(a), std::forward<B>(b)) &&
         all_of_pairs(std::forward<F>(f), std::forward<Pairs>(pairs)...);
}

}  // namespace internal

inline const slinky::dim& dim0_or_broadcast(const slinky::raw_buffer& buf) {
  return buf.rank > 0 ? buf.dim(0) : slinky::dim::broadcast();
}

inline void slice0_if_not_scalar(slinky::raw_buffer& buf, slinky::index_t min) {
  if (buf.rank > 0) buf.slice(0, min);
}

inline void slice0_if_not_scalar(slinky::raw_buffer& buf) {
  if (buf.rank > 0) buf.slice(0);
}

inline bool same_bounds(const slinky::dim& a, const slinky::dim& b) {
  // Return true if the dimensions have the same min and max or if they are
  // both broadcasts.
  return (a.min() == b.min() && a.max() == b.max()) ||
         (a.stride() == 0 && b.stride() == 0);
}

template <typename... Dims>
inline bool same_bounds(const slinky::dim& a, const slinky::dim& b,
                        const Dims&... dims) {
  return same_bounds(a, b) && same_bounds(b, dims...);
}

// A dimension is contiguous if it satisfies one of the following:
//   1. Its extent is 1. In this case, we disregard stride.
//   2. Its stride is equal to its element size.
inline bool is_continguous(const slinky::dim& dim, const int element_size) {
  return dim.extent() == 1 || dim.stride() == element_size;
}

// Peels off the innermost `NumInnerDims` dimensions of `x` and `inputs`,
// and where possible, fuses dimensions of buffers from the innermost to the
// outermost.
//
// `x` is the output buffer.
// `x_dims` is an array of size `NumInnerDims` that is used to store the
// dimensions of `x` after the peeling.
//
// `inputs` is a parameter pack of pairs of pointers to the dimensions of the
// inputs after the peeling, and the buffers themselves e.g. { &a_dims[0], a,
// &b_dims[0], b, ... }. The dimensions must be pointers to arrays of size
// `NumInnerDims`.
template <int NumInnerDims, typename... DimBufferPairs>
void fuse_and_slice_leading_dims(slinky::dim* x_dims, slinky::raw_buffer& x,
                                 DimBufferPairs&&... inputs) {
  for (int i = 0; i < NumInnerDims; ++i) {
    // If the output innermost (n) dimension has extent 1, we need to make the n
    // dimension of all inputs a broadcast. This case is not expected to happen.
    // For now, we add an assert to catch this case if it does.
    assert(i != 0 || is_continguous(dim0_or_broadcast(x), x.elem_size));

    x_dims[i] = dim0_or_broadcast(x);

    // Initialize `in_dims[i]` for each input.
    internal::apply_to_pairs(
        [i](slinky::dim* in_dims, const slinky::raw_buffer& in_buf) {
          in_dims[i] = dim0_or_broadcast(in_buf);
        },
        inputs...);

    // `x` is already a view to the correct tile in the larger output buffer.
    // Inputs are not. We must explicitly set their offsets according to `x`
    // before slicing.
    internal::apply_to_pairs(
        [x_min_i = x_dims[i].min()](const slinky::dim* /*in_dims*/,
                                    slinky::raw_buffer& in_buf) {
          slice0_if_not_scalar(in_buf, x_min_i);
        },
        inputs...);
    slice0_if_not_scalar(x);

    while (x.rank > 0) {
      // First check whether fusing dimensions is possible.
      bool can_fuse_all =
          slinky::can_fuse(x_dims[i], x.dim(0)) &&
          internal::all_of_pairs(
              [x_dims, i](const slinky::dim* in_dims,
                          const slinky::raw_buffer& in_buf) {
                return same_bounds(x_dims[i], in_dims[i]) &&
                       slinky::can_fuse(in_dims[i], dim0_or_broadcast(in_buf));
              },
              inputs...);
      if (!can_fuse_all) {
        break;
      }

      // Fuse the dimensions and slice.
      x_dims[i] = slinky::fuse(x_dims[i], x.dim(0));
      internal::apply_to_pairs(
          [i, x_min_i = x.dim(0).min()](slinky::dim* in_dims,
                                        slinky::raw_buffer& in_buf) {
            in_dims[i] = slinky::fuse(in_dims[i], dim0_or_broadcast(in_buf));
            slice0_if_not_scalar(in_buf, x_min_i);
          },
          inputs...);
      x.slice(0);
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_SLINKY_H_
