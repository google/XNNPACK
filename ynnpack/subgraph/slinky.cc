// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/slinky.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace ynn {

void allow_broadcasting(slinky::raw_buffer& buf) {
  for (size_t d = 0; d < buf.rank; ++d) {
    if (buf.dim(d).min() != 0 || buf.dim(d).max() != 0) continue;
    buf.dim(d).set_stride(0);
    buf.dim(d).set_unbounded();
  }
}

slinky::buffer_expr_ptr make_buffer_expr(slinky::var sym, int rank,
                                         slinky::expr elem_size) {
  slinky::buffer_expr_ptr buf = slinky::buffer_expr::make(sym, rank, elem_size);
  if (rank > 0) {
    // YNNPACK kernels assume the innermost dimension is dense. There are
    // optimizations in slinky that may violate this assumption if we don't
    // disallow it from doing so by requiring the stride is elem_size.
    // TODO(dsharlet): This breaks a lot of slinky optimizations that check
    // stride assumptions aren't being violated. We don't generate transposes
    // that affect the innermost dimension via copy (we call a kernel instead),
    // so this may not be necessary.
    // buf->dim(0).stride = elem_size;
  }

  // Disable storage folding. Many of our callbacks do not support this.
  // TODO(b/447509228): We might want to either re-enable this, or just decide
  // that we shouldn't use it.
  for (int d = 0; d < rank; ++d) {
    buf->dim(d).fold_factor = slinky::dim::unfolded;
  }

  return buf;
}

slinky::buffer_expr_ptr make_buffer_expr(slinky::node_context& ctx,
                                         const std::string& name, int rank,
                                         slinky::expr elem_size) {
  return make_buffer_expr(ctx.insert_unique(name), rank, elem_size);
}

// Constrain the strides of the first `dims` dimensions of a buffer such that
// the strides are dense (no padding between dimensions) and in the order that
// XNNPACK/TFlite expect.
void require_contiguous(slinky::buffer_expr& buf, size_t dims) {
  slinky::expr stride = buf.elem_size();
  for (size_t d = 0; d < std::min(dims, buf.rank()); ++d) {
    buf.dim(d).stride = stride;
    buf.dim(d).fold_factor = slinky::dim::unfolded;
    stride *= buf.dim(d).extent();
  }
}

// Make an array of dimensions that is begin, 1, ... end - 1.
std::vector<slinky::var> make_dims(int begin, int end,
                                   slinky::node_context& ctx) {
  std::vector<slinky::var> result(end - begin);
  for (int i = 0; i < result.size(); ++i) {
    result[i] = ctx.insert("d" + std::to_string(begin + i));
  }
  return result;
}
std::vector<slinky::var> make_dims(int rank, slinky::node_context& ctx) {
  return make_dims(0, rank, ctx);
}

slinky::interval_expr elementwise_bounds(slinky::var dim,
                                         const slinky::expr& extent) {
  return extent.defined() ? slinky::point(dim) : slinky::point(0);
}

slinky::interval_expr all_bounds(const slinky::expr& extent) {
  return extent.defined() ? slinky::min_extent(0, extent) : slinky::point(0);
}

// Make an array of bounds that is point(i) for i in dims in [begin, end).
slinky::box_expr make_elementwise_bounds(
    const std::vector<slinky::var>& dims,
    const std::vector<slinky::expr>& extents, size_t begin, size_t end) {
  assert(end <= dims.size());
  slinky::box_expr bounds(end - begin);
  for (size_t i = 0; i < bounds.size() && begin + i < extents.size(); ++i) {
    bounds[i] = elementwise_bounds(dims[begin + i], extents[begin + i]);
  }
  return bounds;
}

slinky::box_expr make_elementwise_bounds(
    const std::vector<slinky::var>& dims,
    const std::vector<slinky::expr>& extents) {
  return make_elementwise_bounds(dims, extents, 0, dims.size());
}

slinky::interval_expr make_broadcast_bounds(const slinky::var& dim,
                                            const slinky::expr& src_extent,
                                            const slinky::expr& dst_extent,
                                            bool no_broadcast) {
  if (!src_extent.defined()) {
    return slinky::point(0);
  }
  if (no_broadcast) {
    return slinky::point(dim);
  }
  std::optional<bool> extents_equal =
      slinky::attempt_to_prove(src_extent == dst_extent);
  if (extents_equal && *extents_equal) {
    return slinky::point(dim);
  } else if (extents_equal && !*extents_equal) {
    return slinky::point(0);
  } else {
    return slinky::point(select(src_extent > 1, dim, 0));
  }
}

// Like make_elementwise_bounds, but inserts conditionals to handle
// broadcasting if needed.
slinky::box_expr make_broadcast_bounds(
    std::vector<slinky::var> dims, const std::vector<slinky::expr>& src_extents,
    const std::vector<slinky::expr>& dst_extents, bool no_broadcast) {
  slinky::box_expr bounds(src_extents.size());
  // Outputs with higher rank than inputs implicitly broadcast the extra
  // dimensions, but we don't want to put them in the bounds.
  assert(dst_extents.size() >= src_extents.size());

  for (size_t d = 0; d < bounds.size(); ++d) {
    bounds[d] = make_broadcast_bounds(dims[d], src_extents[d], dst_extents[d],
                                      no_broadcast);
  }
  return bounds;
}

}  // namespace ynn
