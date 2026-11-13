// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/dot_pack.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/base.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/kernels/dot/pack.h"
#include "ynnpack/kernels/dot/schedule.h"
#include "ynnpack/subgraph/dot_schedule.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

using slinky::index_t;

namespace ynn {

namespace {

// Make a kernel wrapper for packing the input of a dot kernel, i.e.
// interleaving `tile_k` rows at a time.
auto make_pack_impl(int elem_count, bool transpose) {
  return [elem_count, transpose](slinky::raw_buffer input,
                                 slinky::raw_buffer output) -> index_t {
    const slinky::dim& input_0 = input.dim(0);
    const slinky::dim& input_1 = input.dim(1);

    const slinky::dim& output_ki = output.dim(0);
    const slinky::dim& output_ni = output.dim(1);
    const slinky::dim& output_ko = output.dim(2);
    const slinky::dim& output_no = output.dim(3);

    const index_t elem_size = output.elem_size;
    const index_t tile_k = output_ki.extent() * elem_count;
    const index_t block_n = output_ni.extent();
    assert(output_ki.min() == 0);
    assert(output_ni.min() == 0);
    assert(output_ki.extent() == 1 || output_ki.stride() == elem_size);
    assert(output_ni.extent() == 1 || output_ki.extent() == 1 ||
           output_ni.stride() == output_ki.stride() * output_ki.extent());
    (void)output_ki;

    // We need the extent of the intersection of the input and output bounds.
    // For sub-byte datatypes (e.g. int4), Slinky's buffer extents along dim 0
    // represent physical bytes, not logical elements.
    const index_t input_0_end = input_0.end();
    const index_t input_1_end = input_1.end();
    const index_t k_end = transpose ? input_0_end * elem_count : input_1_end;
    const index_t n_end = transpose ? input_1_end : input_0_end * elem_count;
    const index_t k =
        std::max<index_t>(0, std::min(output_ko.end() * tile_k, k_end) -
                                 output_ko.min() * tile_k);
    const index_t n =
        std::max<index_t>(0, std::min(output_no.end() * block_n, n_end) -
                                 output_no.min() * block_n);

    const slinky::dim& input_k = transpose ? input_0 : input_1;
    const slinky::dim& input_n = transpose ? input_1 : input_0;
    const index_t stride_k =
        input_k.extent() == 1 ? elem_size : input_k.stride();
    const index_t stride_n =
        input_n.extent() == 1 ? elem_size : input_n.stride();
    const index_t output_ko_stride = output_ko.stride();
    const index_t output_no_stride = output_no.stride();

    if (transpose) {
      assert(input_0.min() * elem_count <= output_ko.min() * tile_k);
      assert(input_1.min() <= output_no.min() * block_n);
      input.slice(0, output_ko.min() * tile_k / elem_count);
      input.slice(0, output_no.min() * block_n);
    } else {
      assert(input_0.min() * elem_count <= output_no.min() * block_n);
      assert(input_1.min() <= output_ko.min() * tile_k);
      input.slice(0, output_no.min() * block_n / elem_count);
      input.slice(0, output_ko.min() * tile_k);
    }
    slice_dims(output, 4);

    packer p(stride_k, stride_n, elem_size * 8 / elem_count, tile_k, block_n,
             output_ko_stride, transpose);

    slinky::for_each_element(
        [=, &p](void* output, const void* input) {
          p.pack(k, n, input, output_no_stride, output);
        },
        output, input);
    return 0;
  };
}

ynn_status create_pack_b(const ynn_node& node, ynn_runtime& runtime) {
  const ynn_node::pack_b& op = std::get<ynn_node::pack_b>(node.op);
  const ynn_runtime_value& input = runtime.value(node.inputs[0]);
  ynn_runtime_value& output = runtime.value(node.outputs[0]);

  output.make_buffer(runtime, input.buffer->elem_size());

  const int element_count = type_element_count(input.type);

  // Split + Transpose
  std::vector<slinky::var> dims =
      runtime.globals.make_dims(output.buffer->rank());

  slinky::func::input func_input = {input.buffer};
  slinky::expr logical_tile_k = output.extent(0);
  slinky::expr logical_block_n = output.extent(1);
  slinky::var ko = dims[2];
  slinky::var no = dims[3];
  // The callback expects to produce whole ki x ni tiles at once.
  if (op.transpose) {
    func_input.bounds = {
        slinky::min_extent(ko * logical_tile_k, logical_tile_k) / element_count,
        slinky::min_extent(no * logical_block_n, logical_block_n),
    };
  } else {
    func_input.bounds = {
        slinky::min_extent(no * logical_block_n, logical_block_n) /
            element_count,
        slinky::min_extent(ko * logical_tile_k, logical_tile_k),
    };
  }
  for (size_t i = 4; i < dims.size(); ++i) {
    func_input.bounds.push_back(slinky::point(dims[i]));
  }
  // This packing handles padding the input up to tile_k x tile_n.
  func_input.input_crop = {
      all_bounds(input.physical_extent(0)),
      all_bounds(input.physical_extent(1)),
  };

  slinky::call_stmt::attributes attrs;
  attrs.name = "pack_b";
  auto func = slinky::func::make(make_pack_impl(element_count, op.transpose),
                                 {std::move(func_input)},
                                 {{output.buffer, dims}}, std::move(attrs));

  // Pin ki, ni and ko to their full extents (the packing kernel produces
  // whole ki x ni tiles and all of ko at once), and let make_schedule pick
  // splits and workers for the blocks_n and batch dimensions, so the
  // packing is parallelized even when it is not fused into a dot's loop
  // nest (e.g. when it is constant folded). When it is fused, the splits
  // don't require their steps, so they adopt the loops of the dot as
  // before.
  std::vector<slinky::expr> given_splits = {output.physical_extent(0),
                                            output.physical_extent(1),
                                            output.physical_extent(2)};
  auto sched = runtime.make_schedule(dims, output.physical_extents(),
                                     output.buffer->elem_size(), given_splits);
  sched.loop_splits[0].step_is_required = true;
  sched.loop_splits[1].step_is_required = true;

  // The real bounds of the input's n dimension are blocks of size block_n
  // indexed by `no`, which breaks the scheduler's source region inference.
  // Declare a virtual 1-to-1 mapping with `no` instead, so producers of the
  // input can be fused with loops derived from it.
  if (op.transpose) {
    sched.input_scheduler_bounds = {
        {slinky::interval_expr{}, slinky::point(no)}};
  } else {
    sched.input_scheduler_bounds = {{slinky::point(no)}};
  }

  runtime.funcs.push_back({std::move(func), std::move(sched)});
  return ynn_status_success;
}

// Make a kernel wrapper for packing the input of a dot kernel, i.e.
// interleaving `tile_k` rows at a time.
// TODO(b/454146513): We should try to combine both pack_b and transpose_a into
// a `split_transpose` op that can handle padding, split, and transpose.
auto make_transpose_a_impl(int m_dim) {
  return [m_dim](slinky::buffer<const void, max_tensor_rank> input,
                 slinky::buffer<void, max_tensor_rank> output) -> index_t {
    const slinky::dim& input_k = input.dim(0);
    const slinky::dim& input_m = input.dim(m_dim);
    const slinky::dim& output_ki = output.dim(0);
    const slinky::dim& output_mi = output.dim(1);
    const slinky::dim& output_ko = output.dim(2);
    const slinky::dim& output_mo = output.dim(m_dim + 2);

    const index_t tile_k = output_ki.extent();
    const index_t tile_m = output_mi.extent();
    const index_t elem_size = input.elem_size;
    assert(output_ki.min() == 0);
    assert(output_mi.min() == 0);
    assert(output_ki.extent() == 1 || output_ki.stride() == elem_size);
    assert(output_mi.extent() == 1 || output_mi.stride() == elem_size * tile_k);
    assert(output_mo.extent() == 1 ||
           output_mo.stride() == elem_size * tile_k * tile_m);

    // We need the intersection of the input and output bounds.
    const index_t m_begin = output_mo.begin() * tile_m;
    const index_t m_end = output_mo.end() * tile_m;
    const index_t m =
        std::max<index_t>(0, std::min(m_end, input_m.end()) - m_begin);
    assert(input_k.min() <= output_ko.min() * tile_k);
    const index_t k_begin = output_ko.begin() * tile_k;
    const index_t k_end = output_ko.end() * tile_k;
    const index_t k =
        std::max<index_t>(0, std::min(k_end, input_k.end()) - k_begin);

    // We're transposing columns of the input to rows of the output, but
    // doing tile_k of them at a time.
    const index_t stride_k =
        input_k.extent() == 1 ? elem_size : input_k.stride();
    const index_t stride_m =
        input_m.extent() == 1 ? elem_size : input_m.stride();

    const index_t output_ko_stride = output_ko.stride();

    input.slice(0, slinky::in_bounds{k_begin});
    input.slice(m_dim - 1, slinky::in_bounds{m_begin});
    output.slice({0, 1, 2, static_cast<size_t>(m_dim + 2)});

    packer p(stride_k, stride_m, elem_size * 8, tile_k, /*tile_n=*/m,
             output_ko_stride);
    slinky::for_each_element(
        [=, &p](void* output, const void* input) {
          p.pack(k, m, input, /*output_block_stride=*/0, output);
        },
        output, input);
    return 0;
  };
}

ynn_status create_transpose_a(const ynn_node& node, ynn_runtime& runtime) {
  const ynn_node::transpose_a& op = std::get<ynn_node::transpose_a>(node.op);
  // pack_b gets tile_k, tile_n from the extents of the output buffer. Should
  // we do the same here?
  const index_t tile_m = op.tile_m;
  const index_t tile_k = op.tile_k;
  const int m_dim = op.m_dim;
  const ynn_runtime_value& input = runtime.value(node.inputs[0]);
  ynn_runtime_value& output = runtime.value(node.outputs[0]);

  slinky::expr elem_size = input.buffer->elem_size();
  output.make_buffer(runtime, elem_size);
  output.buffer->dim(0).stride = elem_size;
  output.buffer->dim(1).stride = elem_size * tile_k;
  output.buffer->dim(m_dim + 2).stride = elem_size * tile_k * tile_m;
  output.buffer->dim(2).stride = output.buffer->dim(m_dim + 2).stride *
                                 output.buffer->dim(m_dim + 2).extent();

  // Don't allow folding of dimensions we transpose.
  output.buffer->dim(0).fold_factor = slinky::dim::unfolded;
  output.buffer->dim(1).fold_factor = slinky::dim::unfolded;
  output.buffer->dim(m_dim + 2).fold_factor = slinky::dim::unfolded;
  output.buffer->dim(2).fold_factor = slinky::dim::unfolded;

  // Split + Transpose
  std::vector<slinky::var> dims =
      runtime.globals.make_dims(output.buffer->rank());

  slinky::expr ko = dims[2];
  slinky::expr mo = dims[m_dim + 2];

  slinky::func::input func_input = {input.buffer};
  func_input.bounds.resize(input.buffer->rank());
  if (input.buffer->rank() > 0) {
    func_input.bounds[0] = slinky::min_extent(ko * tile_k, tile_k);
  }
  for (int i = 1; i < std::min<int>(m_dim, input.buffer->rank()); ++i) {
    func_input.bounds[i] = slinky::point(dims[i + 2]);
  }
  if (static_cast<size_t>(m_dim) < input.buffer->rank()) {
    func_input.bounds[m_dim] = slinky::min_extent(mo * tile_m, tile_m);
  }
  for (size_t i = m_dim + 1; i < input.buffer->rank(); ++i) {
    func_input.bounds[i] = slinky::point(dims[i + 2]);
  }

  // This transpose handles padding the input up to tile_k and tile_m.
  func_input.input_crop.resize(input.buffer->rank());
  if (input.buffer->rank() > 0) {
    func_input.input_crop[0] = all_bounds(input.extent(0));
  }
  if (static_cast<size_t>(m_dim) < input.buffer->rank()) {
    func_input.input_crop[m_dim] = all_bounds(input.extent(m_dim));
  }

  slinky::call_stmt::attributes attrs;
  attrs.name = "transpose_a";
  auto func =
      slinky::func::make(make_transpose_a_impl(m_dim), {std::move(func_input)},
                         {{output.buffer, dims}}, std::move(attrs));

  // Pin ki, mi and the k dimensions to their full extents (the transpose
  // kernel produces whole ki x mi tiles and all of ko at once), and let
  // make_schedule pick splits and workers for the mo and batch dimensions.
  // When it is fused, the splits don't require their steps, so they adopt
  // the loops of the dot.
  std::vector<slinky::expr> given_splits(m_dim + 2);
  for (int i = 0; i < m_dim + 2; ++i) {
    given_splits[i] = output.physical_extent(i);
  }
  auto sched = runtime.make_schedule(dims, output.physical_extents(),
                                     output.buffer->elem_size(), given_splits);

  // The real bounds of the input's m dimension are tiles of size tile_m
  // indexed by `mo`, which breaks the scheduler's source region inference.
  // Declare a virtual 1-to-1 mapping with `mo` instead, so producers of the
  // input can be fused with loops derived from it.
  sched.input_scheduler_bounds.resize(1);
  sched.input_scheduler_bounds[0].resize(
      std::min<int>(m_dim + 1, input.buffer->rank()));
  if (m_dim < input.buffer->rank()) {
    sched.input_scheduler_bounds[0][m_dim] = slinky::point(mo);
  }

  runtime.funcs.push_back({std::move(func), std::move(sched)});
  return ynn_status_success;
}

}  // namespace

// Packing means transposing
// b(n, k, ...) => b(k%tile_k, n%nr, k/tile_k, n/tile_n, ...)
// where tile_n is a multiple of the kernel's tile_n, but not greater than the
// kernel's block_n.
uint32_t define_pack_b(ynn_subgraph& subgraph, const dot_type& type,
                       const dot_kernel& kernel, size_t num_k_dims,
                       uint32_t input_b_id, bool transpose) {
  const ynn_value& b = subgraph.value(input_b_id);

  ynn_value& packed_b = subgraph.new_internal_value();
  packed_b.type = b.type;
  uint32_t packed_b_id = packed_b.id;

  slinky::expr n = transpose ? b.extent(1) : b.extent(0);
  slinky::expr k1 = transpose ? b.extent(0) : b.extent(1);
  slinky::expr k2 = num_k_dims >= 2 ? b.extent(2) : 1;
  slinky::expr k3 = num_k_dims >= 3 ? b.extent(3) : 1;

  // When choosing block_n, we want to make the block at least as large as the
  // kernel's `max_block_n` (which is a power-of-2 multiple of all compatible
  // `tile_n`) so any compatible kernel can run without splitting a packed block
  // and `block_n` evenly divides standard power-of-2 column dimensions.
  slinky::expr block_n;
  if (type_size_bytes(b.type) <= 1) {
    // We don't want the stride of the loads from B to be too big.
    // TODO: b/543245536 - We should probably do this for all types, not just
    // int8 or smaller. However, this uncovers an issue of uneven split factors
    // in some cases, so as a workaround, we only do this for 8-bit or smaller
    // types.
    block_n = kernel.max_block_n;
  } else {
    // - We want to maximize block_n if the block will fit in cache.
    // Set a maximum block size to prevent `block_n` from being too large that
    // it limits concurrency in the n dimension.
    constexpr size_t max_dot_block_bytes = 64 * dot_k_step_bytes;
    const size_t cache_size =
        std::min(get_dot_l1_l2_budget(), max_dot_block_bytes);
    const int cache_elements = cache_size * 8 / type_size_bits(b.type);
    slinky::expr cache_blocks_n = slinky::floor_div<slinky::expr>(
        cache_elements, kernel.max_block_n * k1 * k2 * k3);
    block_n = kernel.max_block_n * max(1, cache_blocks_n);
  }

  // - We don't want the block to be bigger than n (the number of columns of B).
  // - We want it to be aligned to a multiple of the kernel's `tile_n`.
  block_n = min(slinky::align_up(n, kernel.tile_n), block_n);

  // Make a global variable for the alignment, which is a messy expression,
  // but keep the max outside it, so slinky can learn bounds from it
  // (hacky...).
  block_n = max(kernel.tile_n, subgraph.globals.get(block_n, "block_n"));
  slinky::expr tiles_k = slinky::ceil_div<slinky::expr>(k1, kernel.tile_k);
  slinky::expr blocks_n = slinky::ceil_div(n, block_n);

  assert(kernel.tile_k % type_element_count(b.type) == 0);
  packed_b.extents = {kernel.tile_k, block_n, tiles_k, blocks_n};
  for (slinky::expr& i : packed_b.extents) {
    i = slinky::simplify(i);
  }
  packed_b.extents.insert(packed_b.extents.end(), b.extents.begin() + 2,
                          b.extents.end());

  ynn_node node;
  node.inputs = {input_b_id};
  node.outputs = {packed_b_id};
  node.op = ynn_node::pack_b{transpose};
  node.create = create_pack_b;
  subgraph.add_node(std::move(node));
  return packed_b_id;
}

// Packing means transposing
// a(k, m, ...) => a([0, tile_k), [0, tile_m), k/tile_k, ..., m/tile_m, ...)
void define_transpose_a(ynn_subgraph& subgraph, ynn_node& node, index_t tile_m,
                        index_t tile_k, int m_dim, uint32_t input_a_id,
                        uint32_t output_id) {
  const ynn_value& a = subgraph.value(input_a_id);
  ynn_value& output = subgraph.get_output_value(&output_id, a.type);
  output.type = a.type;

  slinky::expr k = a.extent(0);
  slinky::expr m = a.extent(m_dim);
  output.extents = a.extents;
  while (output.extents.size() <= static_cast<size_t>(m_dim)) {
    output.extents.push_back(slinky::expr{});
  }
  output.extents[m_dim] =
      slinky::simplify(slinky::ceil_div<slinky::expr>(m, tile_m));
  output.extents[0] =
      slinky::simplify(slinky::ceil_div<slinky::expr>(k, tile_k));
  output.extents.insert(output.extents.begin(), {tile_k, tile_m});

  node.inputs = {input_a_id};
  node.outputs = {output.id};
  node.op = ynn_node::transpose_a{static_cast<size_t>(tile_m),
                                  static_cast<size_t>(tile_k), m_dim};
  node.create = create_transpose_a;
}

uint32_t define_transpose_a(ynn_subgraph& subgraph, index_t tile_m,
                            index_t tile_k, int32_t m_dim,
                            uint32_t input_a_id) {
  ynn_node node;
  ynn_value& output = subgraph.new_internal_value();
  ynn::define_transpose_a(subgraph, node, tile_m, tile_k, m_dim, input_a_id,
                          output.id);
  subgraph.add_node(std::move(node));
  return output.id;
}

}  // namespace ynn
