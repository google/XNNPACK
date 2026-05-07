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
#include <utility>
#include <vector>

#include "ynnpack/base/span.h"
#include "slinky/base/arithmetic.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/builder/substitute.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace ynn {

slinky::expr slinky_globals::get(slinky::expr value, const char* prefix) {
  value = slinky::simplify(value);
  assert(value.defined());

  if (as_constant(value) || as_variable(value)) {
    return value;
  }

  auto i = std::find_if(lets.begin(), lets.end(),
                        [&](const auto& j) { return match(j.second, value); });
  if (i == lets.end()) {
    slinky::var r = symbols.insert_unique(prefix);
    lets.push_back(std::make_pair(r, value));
    return r;
  } else {
    return i->first;
  }
}

slinky::buffer_expr_ptr slinky_globals::make_buffer_expr(
    const std::string& name, int rank, slinky::expr elem_size) {
  return ynn::make_buffer_expr(symbols.insert_unique(name), rank, elem_size);
}

slinky::var slinky_globals::make_dim(int d, const char* prefix) {
  return symbols.insert(prefix + std::to_string(d));
}

slinky::var slinky_globals::make_reduction_dim(int d) {
  return make_dim(d, reduction_dim_prefix);
}

bool slinky_globals::is_reduction_dim(slinky::var dim) {
  std::string name = symbols.name(dim);
  return !name.empty() && name[0] == reduction_dim_prefix[0];
}

bool slinky_globals::is_pure_dim(slinky::var dim) {
  std::string name = symbols.name(dim);
  return !name.empty() && name[0] == pure_dim_prefix[0];
}

std::vector<slinky::var> slinky_globals::make_dims(int begin, int end,
                                                   const char* prefix) {
  std::vector<slinky::var> result(end - begin);
  for (int i = 0; i < result.size(); ++i) {
    result[i] = symbols.insert(prefix + std::to_string(begin + i));
  }
  return result;
}

std::vector<slinky::var> slinky_globals::make_dims(int rank,
                                                   const char* prefix) {
  return make_dims(0, rank, prefix);
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

// Constrain the strides of the first `dims` dimensions of a buffer such that
// the strides are dense (no padding between dimensions) and in the order that
// XNNPACK/TFlite expect.
void require_contiguous(slinky::buffer_expr& buf, size_t dims) {
  slinky::expr stride = buf.elem_size();
  for (size_t d = 0; d < std::min(dims, buf.rank()); ++d) {
    if (prove_true(buf.dim(d).extent() == 1)) {
      buf.dim(d) = slinky::dim::broadcast();
    } else {
      buf.dim(d).stride = stride;
      buf.dim(d).fold_factor = slinky::dim::unfolded;
      stride *= buf.dim(d).extent();
    }
  }
}

slinky::interval_expr elementwise_bounds(slinky::var dim,
                                         const slinky::expr& extent) {
  return extent.defined() ? slinky::point(dim) : slinky::point(0);
}

slinky::interval_expr all_bounds(const slinky::expr& extent) {
  return extent.defined() ? slinky::range(0, extent) : slinky::point(0);
}

// Make an array of bounds that is point(i) for i in dims in [begin, end).
slinky::box_expr make_elementwise_bounds(
    const std::vector<slinky::var>& dims,
    const std::vector<slinky::expr>& extents, size_t begin, size_t end) {
  assert(end <= dims.size());
  slinky::box_expr bounds(end - begin);
  for (size_t i = 0; i < bounds.size(); ++i) {
    if (begin + i < extents.size()) {
      bounds[i] = elementwise_bounds(dims[begin + i], extents[begin + i]);
    } else {
      bounds[i] = slinky::point(0);
    }
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

std::vector<slinky::expr> make_split_factors(
    ynn::slinky_globals& globals, ynn::span<const slinky::expr> extents,
    const slinky::expr& element_cost,
    ynn::span<const slinky::expr> given_splits,
    ynn::span<const int> loop_order) {
  const int rank = extents.size();

  // Area is selected such that tiles fit better into cache, this is a
  // constant for now, but we could add a more advanced logic based on
  // hardware info.
  slinky::expr tile_area =
      slinky::ceil_div(slinky::expr(32768 * 4), element_cost);
  std::vector<slinky::expr> splits(rank);
  slinky::expr tile_area_so_far = 1;

  auto get_loop_dim = [&](int index_d) {
    return index_d < loop_order.size() ? loop_order[index_d] : index_d;
  };

  for (int index_d = 0; index_d < rank; ++index_d) {
    int d = get_loop_dim(index_d);
    assert(d < extents.size());
    if (!extents[d].defined()) continue;
    if (d < given_splits.size()) {
      splits[d] = given_splits[d];
    } else {
      slinky::expr s = slinky::simplify(slinky::max(
          1, slinky::min(tile_area / tile_area_so_far, extents[d])));
      s = globals.get(s, "s");
      splits[d] = s;
    }
    if (splits[d].defined() && slinky::prove_true(splits[d] >= extents[d])) {
      // TODO(b/458542243): We should not need to do this optimization
      // ourselves.
      splits[d] = {};
    }
    if (splits[d].defined()) {
      tile_area_so_far = slinky::simplify(tile_area_so_far * splits[d]);
    } else {
      tile_area_so_far = slinky::simplify(tile_area_so_far * extents[d]);
    }
  }
  return splits;
}

}  // namespace ynn
