// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/dot.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/pack.h"
#include "ynnpack/kernels/dot/schedule.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"
#include "slinky/base/arithmetic.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

using slinky::index_t;

namespace ynn {

namespace {

// TODO(dsharlet): This should probably be a parameter we learn based on cpuinfo
// or other source of CPU metadata. This was determined experimentally.
constexpr index_t cache_size_l2 = 128 * 1024;

// When we want arithmetic to be consistent, we need to make all tiling
// decisions independently of any hardware dependent parameters (cache sizes,
// kernel tile sizes, etc.).
constexpr index_t consistent_block_n = 64;

// The wrapper for the kernel we use when we actually want to run a dot kernel
// on some buffers.
auto make_dot_impl(dot_type type, bool consistent_arithmetic, bool transposed_a,
                   bool pack_b, size_t num_k_dims) {
  return [type, consistent_arithmetic, transposed_a, pack_b, num_k_dims](
             slinky::raw_buffer a, slinky::raw_buffer b,
             slinky::raw_buffer init_c, slinky::raw_buffer c) -> index_t {
    // If the dot has fewer than 3 reduction dimensions, we use this dummy
    // dimension instead.
    slinky::dim dummy_dim = slinky::dim(0, 0, 0, 0);

    // Learn what we need to know about m, n, k1, k2, k3 before slicing them.
    const slinky::dim& init_c_m = init_c.rank > 1 ? init_c.dim(1) : dummy_dim;
    const slinky::dim& init_c_n = init_c.rank > 0 ? init_c.dim(0) : dummy_dim;
    const slinky::dim& c_m = c.rank > 1 ? c.dim(1) : dummy_dim;
    const slinky::dim& c_n = c.dim(0);
    const slinky::dim& a_k1 = a.dim(0);
    const slinky::dim& a_k2 = num_k_dims >= 2 ? a.dim(1) : dummy_dim;
    const slinky::dim& a_k3 = num_k_dims >= 3 ? a.dim(2) : dummy_dim;
    const slinky::dim& a_m =
        a.rank > num_k_dims ? a.dim(num_k_dims) : dummy_dim;
    const slinky::dim& b_k1i = b.dim(0);
    const slinky::dim& b_ni = b.dim(1);
    const slinky::dim& b_k1o = b.dim(2);
    const slinky::dim& b_no = b.dim(3);
    const slinky::dim& b_k2 = num_k_dims >= 2 ? b.dim(4) : dummy_dim;
    const slinky::dim& b_k3 = num_k_dims >= 3 ? b.dim(5) : dummy_dim;

    const int b_type_element_count = type_element_count(type.b);
    const index_t tile_k = b_k1i.extent() * b_type_element_count;
    const index_t a_stride_m = a_m.stride();
    const index_t a_stride_k3 = a_k3.stride();
    const index_t a_stride_k2 = a_k2.stride();
    const index_t a_stride_k1 = a_k1.stride();
    // If a is transposed, then the k dimension has been reshaped to have tile_k
    // values in each element.
    const index_t a_tile_k = transposed_a ? tile_k : 1;
    const index_t k1 = (a_k1.extent() * a_tile_k) & ~(tile_k - 1);
    const index_t k1_tail = (a_k1.extent() * a_tile_k) & (tile_k - 1);
    const index_t k2 = a_k2.extent();
    const index_t k3 = a_k3.extent();
    const index_t block_n = pack_b ? b_ni.extent() : c_n.extent();
    const index_t b_stride_k3 = b_k3.stride();
    const index_t b_stride_k2 = b_k2.stride();
    // TODO: The kernels should probably expect this stride to be multiplied
    // already.
    assert(b_k1o.extent() == 1 || b_k1o.stride() % tile_k == 0);
    const index_t b_stride_k1 = b_k1o.stride() / tile_k;
    // This next line covers up a very tricky issue: we want to be able to
    // address b by n, but when b is packed, it has been transposed such that
    // b at n = 0 and b at n = block_n are separated by an entire packed column,
    // but if b is not packed, they are separated by n elements. If b is packed,
    // this stride is a lie: it is only valid to increment b pointers by
    // `block_n` values at a time.
    assert(pack_b ? b_no.extent() == 1 || b_no.stride() % b_ni.extent() == 0
                  : b_ni.extent() == 1 || b_ni.stride() == b.elem_size);
    const index_t b_stride_n =
        pack_b ? b_no.stride() / b_ni.extent() : b_ni.stride();
    const index_t c_stride_m = c_m.stride();
    const index_t c_stride_n = c_n.stride();
    index_t init_c_stride_m = init_c_m.stride();

    // Find a kernel that is compatible with the packed data we have, and
    // matches whether A is transposed or not.
    dot_shape shape;
    shape.m = c_m.extent();
    shape.n = c_n.extent();
    shape.k1 = k1;
    shape.k2 = k2;
    shape.k3 = k3;
    dot_packed_shape packed_shape;
    packed_shape.block_n = block_n;
    packed_shape.tile_k = tile_k;
    dot_kernel kernel = get_dot_kernel(
        type, shape, &packed_shape, consistent_arithmetic,
        a_stride_m == a_stride_k1 ? std::nullopt
                                  : std::make_optional(transposed_a));
    assert(kernel.kernel);
    assert(tile_k == kernel.tile_k);
    const index_t block_m = kernel.block_m;
    const index_t block_k = kernel.block_k;

    assert(a_k1.min() == 0);
    assert(a_k2.min() == 0);
    assert(a_k3.min() == 0);
    assert(b_k1i.min() == 0);
    assert(b_k1i.extent() == 1 || b_k1i.stride() == b.elem_size);
    assert(b_ni.min() == 0);
    assert(b_ni.extent() == 1 || b_ni.stride() == b.elem_size * b_k1i.extent());
    assert(b_k1o.min() == 0);
    assert(b_k1o.extent() == 1 || b_ni.extent() == 1 ||
           b_k1o.stride() == b_ni.stride() * b_ni.extent());
    assert(b_k2.min() == 0);
    assert(b_k3.min() == 0);
    assert(!init_c_m.is_folded());
    assert(!init_c_n.is_folded());
    assert(!c_m.is_folded());
    assert(!c_n.is_folded());
    assert(!a_m.is_folded(c_m.min(), c_m.max()));
    assert(!a_k1.is_folded());
    assert(!a_k2.is_folded());
    assert(!a_k3.is_folded());
    assert(!b_k1o.is_folded());
    assert(!b_no.is_folded());
    assert(!b_k2.is_folded());
    assert(!b_k3.is_folded());

    if (init_c.base && init_c.base != c.base && c_n.extent() > 1) {
      if (init_c_n.stride() == 0) {
        // The initializer is broadcasted in the n dimension, which the kernel
        // cannot handle. We need to copy it to the output, and update the
        // initializer to point to the output.
        slinky::copy(init_c, c);
        init_c_stride_m = c_stride_m;
        init_c = c;
      } else {
        assert(init_c_n.stride() == c_stride_n);
      }
    }

    // `for_each_element` below handles the batch dimensions, we handle the loop
    // over m, and the kernel handles the rest (n, k1, k2, k3). We need to slice
    // off these dimensions so we can handle them.
    for (size_t i = 0; i < num_k_dims; ++i) {
      a.slice(0);
    }
    if (a.rank > 0) {
      a.slice(0, c_m.min());
    }
    if (pack_b) {
      // If b is packed, we must slice b at blocks of n.
      b.slice({0, 1, 2});
      b.slice(0, c_n.min() / block_n);
    } else {
      // If b is not packed, we need to just slice it at n.
      b.slice(0);
      b.slice(0, c_n.min());
      b.slice({0, 1});
    }
    for (size_t i = 1; i < num_k_dims; ++i) {
      b.slice(0);
    }
    if (init_c.rank >= 2) {
      init_c.slice(0, c_n.min());
      init_c.slice(0, c_m.min());
    } else if (init_c.rank > 0) {
      init_c.slice(0, c_n.min());
    }
    if (c.rank >= 2) {
      c.slice({0, 1});
    } else {
      c.slice(0);
    }
    // TODO: At this point, we can probably fuse dimensions of c, a, b in the
    // hopes of making i bigger, which should improve performance in cases where
    // block_m does not divide c_m.extent()

    const index_t a_stride = transposed_a ? a_stride_k1 : a_stride_m;
    // The kernels assume that the column dimension of a is stride 1 element.
    assert(transposed_a ? (a_m.extent() == 1 || a_stride_m == a.elem_size)
                        : (a_k1.extent() == 1 || a_stride_k1 == a.elem_size));

    auto call_kernel = [=, kernel = kernel.kernel](
                           index_t m, index_t n, index_t k1, const void* a,
                           const void* b, index_t init_c_stride_m,
                           const void* init_c, void* c) {
      assert(n <= block_n);
      assert(m <= block_m);
      kernel(m, n, k3, k2, k1, a_stride, a_stride_k3, a_stride_k2, a,
             b_stride_k3, b_stride_k2, b_stride_k1, b, init_c_stride_m, init_c,
             c_stride_m, c);
    };

    const size_t cache_sizes[] = {cache_size_l2};

    // We need up to 3 loops per cache level.
    dot_loop loops_storage[std::size(cache_sizes) * 3];

    if (k1) {
      auto loops = schedule_dot(cache_sizes, c_m.extent(), c_n.extent(), k1, k2,
                                k3, block_m, block_n, block_k, a.elem_size,
                                b.elem_size, loops_storage);

      slinky::for_each_element(
          [&](void* c, const void* a, const void* b, const void* init_c) {
            run_dot(loops, c_m.extent(), c_n.extent(), k1, block_m, block_n,
                    block_k, a_stride_m, a_stride_k1 / a_tile_k, a, b_stride_k1,
                    b_stride_n, b, init_c_stride_m, init_c, c_stride_m,
                    c_stride_n, c, call_kernel);
          },
          c, a, b, init_c);
    }
    if (k1_tail) {
      auto loops = schedule_dot(cache_sizes, c_m.extent(), c_n.extent(),
                                k1_tail, k2, k3, block_m, block_n, block_k,
                                a.elem_size, b.elem_size, loops_storage);
      // Dot kernels can't handle k1 not aligned to tile_k. We handle that here
      // by making a padded copy of the unaligned elements and calling the
      // kernel again.
      //
      // We do this padding+kernel call once for each value of k3, k2, which
      // is pretty inefficient, but gives us an upper bound (tile_k * block_m)
      // on the amount of memory we need to allocate for the padded area. If
      // the performance of the tail case is an issue, we can improve this at
      // the cost of a bit of complexity.
      const index_t a_elem_size = a.elem_size;
      const index_t a_padded_stride_m = a.elem_size * tile_k;
      void* a_padded = YNN_ALLOCA(uint8_t, block_m* a_padded_stride_m);
      memset(a_padded, 0, a_padded_stride_m * block_m);
      auto call_kernel_tail = [&](index_t m, index_t n, index_t k1,
                                  const void* a, const void* b,
                                  index_t init_c_stride_m, const void* init_c,
                                  void* c) {
        assert(m <= block_m);
        assert(n <= block_n);
        assert(k1 < tile_k);
        for (index_t K3 = 0; K3 < k3; ++K3) {
          for (index_t K2 = 0; K2 < k2; ++K2) {
            for (index_t i = 0; i < m; ++i) {
              memcpy(offset_bytes(a_padded, i * a_padded_stride_m),
                     offset_bytes(a, i * a_stride_m + K3 * a_stride_k3 +
                                         K2 * a_stride_k2),
                     k1 * a_elem_size);
            }
            kernel.kernel(m, n, /*k3=*/1, /*k2=*/1, tile_k, a_padded_stride_m,
                          /*a_stride_k3=*/0, /*a_stride_k2=*/0, a_padded,
                          /*b_stride_k3=*/0,
                          /*b_stride_k2=*/0, b_stride_k1,
                          offset_bytes(b, K3 * b_stride_k3 + K2 * b_stride_k2),
                          init_c_stride_m, init_c, c_stride_m, c);
            init_c_stride_m = c_stride_m;
            init_c = c;
          }
        }
      };
      slinky::for_each_element(
          [&](void* c, const void* a, const void* b, const void* init_c) {
            index_t tail_init_c_stride_m = init_c_stride_m;
            if (k1 != 0) {
              init_c = c;
              tail_init_c_stride_m = c_stride_m;
            }
            a = offset_bytes(a, a_stride_k1 * k1);
            b = offset_bytes(b, b_stride_k1 * k1);
            run_dot(loops, c_m.extent(), c_n.extent(), k1_tail, block_m,
                    block_n, block_k, a_stride_m, a_stride_k1, a, b_stride_k1,
                    b_stride_n, b, tail_init_c_stride_m, init_c, c_stride_m,
                    c_stride_n, c, call_kernel_tail);
          },
          c, a, b, init_c);
    }

    return 0;
  };
}

// Make a kernel wrapper for packing the input of a dot kernel, i.e.
// interleaving `tile_k` rows at a time.
auto make_pack_impl(int elem_count) {
  return [elem_count](slinky::raw_buffer input,
                      slinky::raw_buffer output) -> index_t {
    const slinky::dim& input_n = input.dim(0);
    const slinky::dim& input_k = input.dim(1);
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

    input.slice(0, output_no.min() * block_n / elem_count);
    input.slice(0);
    output.slice({0, 1, 2, 3});

    // Depending on the strides of the input, we might use either an interleave
    // or a transpose kernel to implement this packing.
    const bool transpose =
        input_n.extent() > 1 && input_n.stride() != elem_size;
    const index_t input_stride =
        transpose ? input_n.stride() : input_k.stride();

    // We need the extent of the intersection of the input and output bounds.
    assert(output_ko.min() == 0);
    const index_t k = std::min(output_ko.end() * tile_k, input_k.end());
    assert(input_n.min() * elem_count <= output_no.min() * block_n);
    const index_t n =
        (std::min(output_no.end() * block_n, input_n.end() * elem_count) -
         output_no.begin() * block_n);

    packer p(transpose, elem_size * 8 / elem_count, tile_k, block_n);

    slinky::for_each_element(
        [&](void* output, const void* input) {
          p.pack(k, n, input_stride, input, output_ko.stride(),
                 output_no.stride(), output);
        },
        output, input);
    return 0;
  };
}

// Packing means transposing
// b(n, k, ...) => b(k%tile_k, n%nr, k/tile_k, n/tile_n, ...)
// where tile_n is a multiple of the kernel's tile_n, but not greater than the
// kernel's block_n.
uint32_t define_pack_b(ynn_subgraph_t subgraph, const dot_type& type,
                       const dot_kernel& kernel, size_t num_k_dims,
                       bool consistent_arithmetic, uint32_t input_b_id) {
  const ynn_value& b = subgraph->value(input_b_id);

  ynn_value& packed_b = subgraph->new_internal_value();
  packed_b.type = b.type;
  uint32_t packed_b_id = packed_b.id;

  const int element_count = type_element_count(b.type);

  slinky::expr n = b.extent(0) * element_count;
  slinky::expr k1 = b.extent(1);
  slinky::expr k2 = num_k_dims >= 2 ? b.extent(2) : 1;
  slinky::expr k3 = num_k_dims >= 3 ? b.extent(3) : 1;

  const index_t elem_size_bits = type_size_bytes(b.type) * 8 / element_count;
  const index_t cache_elements = cache_size_l2 * 8 / elem_size_bits;

  // When choosing block_n, we have the following concerns:
  // - We want to make the block bigger than the kernel's `block_n`
  // - If we want consistent arithmetic: it should be independent of the kernel.
  const index_t align_block_n =
      consistent_arithmetic ? consistent_block_n : kernel.block_n;

  // - We want to maximize block_n if the block will fit in cache
  slinky::expr cache_blocks_n = slinky::floor_div<slinky::expr>(
      cache_elements, align_block_n * k1 * k2 * k3);
  slinky::expr block_n = align_block_n * max(1, cache_blocks_n);

  // - We don't want the block to be bigger than n (the number of columns of B).
  // - We want it to be aligned to a multiple of the kernel's `tile_n`.
  block_n = min(slinky::align_up(n, kernel.tile_n), block_n);

  // Make a global variable for the alignment, which is a messy expression,
  // but keep the max outside it, so slinky can learn bounds from it
  // (hacky...).
  block_n = max(kernel.tile_n, subgraph->globals.get(block_n, "block_n"));
  slinky::expr tiles_k = slinky::ceil_div<slinky::expr>(k1, kernel.tile_k);
  slinky::expr blocks_n = slinky::ceil_div(n, block_n);

  assert(kernel.tile_k % element_count == 0);
  packed_b.extents = {kernel.tile_k / element_count, block_n, tiles_k,
                      blocks_n};
  for (slinky::expr& i : packed_b.extents) {
    i = slinky::simplify(i);
  }
  packed_b.extents.insert(packed_b.extents.end(), b.extents.begin() + 2,
                          b.extents.end());

  ynn_node node;
  node.inputs = {input_b_id};
  node.outputs = {packed_b_id};
  node.op = ynn_node::pack_b{};
  node.create = [num_k_dims](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());

    const int element_count = type_element_count(input.type);

    // Split + Transpose
    std::vector<slinky::var> dims =
        runtime.globals.make_dims(output.buffer->rank());

    slinky::func::input func_input = {input.buffer};
    slinky::expr tile_k = output.extent(0);
    slinky::expr block_n = output.extent(1);
    slinky::var ki = dims[0];
    slinky::var ni = dims[1];
    slinky::var ko = dims[2];
    slinky::var no = dims[3];
    func_input.bounds = {
        slinky::point((no * block_n + ni) / element_count),
        slinky::point((ko * tile_k + ki) * element_count),
    };
    for (size_t i = 4; i < dims.size(); ++i) {
      func_input.bounds.push_back(slinky::point(dims[i]));
    }
    // This packing handles padding the input up to tile_k x tile_n.
    func_input.input_crop = {
        all_bounds(input.extent(0)),
        all_bounds(input.extent(1)),
    };

    slinky::call_stmt::attributes attrs;
    attrs.name = "pack_b";
    auto func = slinky::func::make(make_pack_impl(element_count),
                                   {std::move(func_input)},
                                   {{output.buffer, dims}}, std::move(attrs));

    auto sched = std::make_unique<scheduling_info>();

    // Here we assume that if the packing is static, this scheduling doesn't
    // matter, and if it is dynamic, that the loop over n is one loop out from
    // the innermost loop.
    ynn::scheduled_buffer sched_output_buffer = {output.buffer, 1};
    sched->scheduled_buffers.push_back(std::move(sched_output_buffer));

    // TODO(vksnk): This is a temporary workaround to avoid recomputing packed
    // buffer. The proper fix would probably involve adding a loop splits for
    // the packing function and making scheduler match it.
    if (num_k_dims > 1) {
      sched->force_root = true;
    }
    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return packed_b_id;
}

// Make a kernel wrapper for packing the input of a dot kernel, i.e.
// interleaving `tile_k` rows at a time.
// TODO(b/454146513): We should try to combine both pack_b and transpose_a into
// a `split_transpose` op that can handle padding, split, and transpose.
auto make_transpose_a_impl(index_t tile_k, int m_dim) {
  constexpr size_t max_rank = YNN_MAX_TENSOR_RANK + ynn_internal_extra_dims;
  return [tile_k, m_dim](slinky::buffer<const void, max_rank> input,
                         slinky::buffer<void, max_rank> output) -> index_t {
    const slinky::dim& input_k = input.dim(0);
    const slinky::dim& input_m = input.dim(m_dim);
    const slinky::dim& output_ko = output.dim(0);
    const slinky::dim& output_m = output.dim(m_dim);

    const index_t elem_size = input.elem_size;
    assert(output_m.extent() == 1 || output_m.stride() == elem_size * tile_k);
    (void)output_m;

    // We need the intersection of the input and output bounds.
    const index_t m =
        std::min(output_m.end(), input_m.end()) - output_m.begin();
    assert(input_k.min() <= output_ko.min() * tile_k);
    assert(output_ko.min() == 0);
    const index_t k = std::min(output_ko.end() * tile_k, input_k.end());

    // We're transposing columns of the input to rows of the output, but
    // doing tile_k of them at a time.
    // TODO(b/454131137): Support already transposed inputs here.
    packer p(/*transpose=*/true, elem_size * 8, tile_k, /*tile_n=*/m);

    const index_t input_m_stride = input_m.stride();
    const index_t output_ko_stride = output_ko.stride();

    input.slice(0);
    input.slice(m_dim - 1, output_m.min());
    output.slice({0, static_cast<size_t>(m_dim)});

    slinky::for_each_element(
        [&](void* output, const void* input) {
          p.pack(k, m, input_m_stride, input, output_ko_stride,
                 /*output_block_stride=*/0, output);
        },
        output, input);
    return 0;
  };
}

}  // namespace

// Packing means transposing
// a(k, m, ...) => a([0, tile_k), m, k/tile_k, ...)
void define_transpose_a(ynn_subgraph& subgraph, ynn_node& node, index_t tile_k,
                        int m_dim, uint32_t input_a_id, uint32_t output_id) {
  const ynn_value& a = subgraph.value(input_a_id);
  ynn_value& output = subgraph.get_output_value(&output_id, a.type);
  output.type = a.type;

  slinky::expr k = a.extent(0);
  output.extents = a.extents;
  output.extents[0] =
      slinky::simplify(slinky::ceil_div<slinky::expr>(k, tile_k));

  node.inputs = {input_a_id};
  node.outputs = {output.id};
  node.op = ynn_node::transpose_a{static_cast<size_t>(tile_k), m_dim};
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::transpose_a& op = std::get<ynn_node::transpose_a>(node.op);
    const index_t tile_k = op.tile_k;
    const int m_dim = op.m_dim;
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    slinky::expr elem_size = input.buffer->elem_size() * tile_k;
    output.make_buffer(runtime, elem_size);
    output.buffer->dim(m_dim).stride = elem_size;
    output.buffer->dim(0).stride =
        elem_size * output.buffer->dim(m_dim).extent();
    // Don't allow folding of dimensions we transpose.
    output.buffer->dim(m_dim).fold_factor = slinky::dim::unfolded;
    output.buffer->dim(0).fold_factor = slinky::dim::unfolded;

    // Split + Transpose
    std::vector<slinky::var> dims =
        runtime.globals.make_dims(output.buffer->rank());

    slinky::expr ko = dims[0];

    slinky::func::input func_input = {input.buffer};
    func_input.bounds = {
        slinky::min_extent(ko * tile_k, tile_k),
    };
    for (size_t i = 1; i < dims.size(); ++i) {
      func_input.bounds.push_back(slinky::point(dims[i]));
    }
    // This transpose handles padding the input up to tile_k.
    func_input.input_crop = {
        all_bounds(input.extent(0)),
    };

    slinky::call_stmt::attributes attrs;
    attrs.name = "transpose_a";
    auto func = slinky::func::make(make_transpose_a_impl(tile_k, m_dim),
                                   {std::move(func_input)},
                                   {{output.buffer, dims}}, std::move(attrs));

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
}

namespace {

uint32_t define_transpose_a(ynn_subgraph& subgraph, index_t tile_k,
                            int32_t m_dim, uint32_t input_a_id) {
  ynn_node node;
  ynn_value& output = subgraph.new_internal_value();
  ynn::define_transpose_a(subgraph, node, tile_k, m_dim, input_a_id, output.id);
  subgraph.add_node(std::move(node));
  return output.id;
}

std::tuple<slinky::expr, slinky::expr> choose_split_factors(
    ynn_runtime& runtime, slinky::expr m, slinky::expr n, slinky::expr k,
    slinky::expr block_n) {
  // We can only return a scalar from a slinky expression, so we pack the two
  // splits into one integer.
  auto impl = [](const slinky::call* op, slinky::eval_context& ctx) {
    index_t m = evaluate(op->args[0], ctx);
    index_t n = evaluate(op->args[1], ctx);
    index_t k = evaluate(op->args[2], ctx);
    index_t block_n = evaluate(op->args[3], ctx);

    // If k gets big, we're going to tile k anyways. It could be faster to
    // parallelize more finely, but it will waste CPU cycles due to more memory
    // traffic out of the cache.
    k = std::min<index_t>(k, 1024);

    // Considerations for task size:
    // - We want tasks to be square-ish, to maximize the number of times we can
    // use data we load from either side.
    // - Tasks shouldn't be too small, to avoid parallelism overhead.
    // - Tasks shouldn't be too large, so we get enough parallelism.
    const index_t min_area =
        std::min<index_t>(m, 64) * std::min<index_t>(n, 64);
    const index_t max_area = 256 * 256;
    // The maximum cost of a tile, according to the cost function (m + n) * k.
    const index_t max_cost = 1024 * 64;

    // A parameter indicating the target split_m/split_n ratio.
    // TODO(b/438841352): Figure out why we want tall skinny tiles, at least on
    // AMD Rome.
    const index_t aspect_ratio = 4;

    index_t split_n = std::min<index_t>(n, block_n);
    index_t split_m = std::min<index_t>(m, 16);
    while (true) {
      if (split_n * split_m >= min_area) {
        // We've reached the minimum tile size, should we stop?
        if ((split_m + split_n) * k >= max_cost ||
            split_m * split_n >= max_area) {
          // We've reached the maximum task size, we should stop.
          break;
        }
      }
      // We want to make the tile bigger, figure out which dimension to grow.
      if ((aspect_ratio * split_n < split_m || split_m >= m) && split_n < n) {
        split_n *= 2;
      } else if ((split_m < aspect_ratio * split_n || split_n >= n) &&
                 split_m < m) {
        split_m *= 2;
      } else {
        break;
      }
    }

    assert(split_n < 65536);
    assert(split_m < 32768);
    return split_m * 65536 + split_n;
  };
  slinky::expr splits = slinky::call::make(impl, {m, n, k, block_n});

  // Extract the two splits from the single index_t result.
  splits = runtime.globals.get(splits, "dot_splits");
  slinky::expr split_m = splits / 65536;
  slinky::expr split_n = splits % 65536;
  split_m = runtime.globals.get(split_m, "split_m");
  split_n = runtime.globals.get(split_n, "split_n");
  return {split_n, split_m};
}

void learn_shape_from_b(dot_shape& shape, size_t num_k_dims,
                        const ynn_value& b) {
  shape.n = as_constant(b.extent(0));
  shape.k1 = as_constant(b.extent(1));
  shape.k2 = num_k_dims >= 2 ? as_constant(b.extent(2)) : 1;
  shape.k3 = num_k_dims >= 3 ? as_constant(b.extent(3)) : 1;
}

ynn_status always_alias_transpose(ynn_subgraph& subgraph, uint32_t& id) {
  const ynn_node* b_producer = subgraph.get_producer(id);
  if (b_producer && std::get_if<ynn_node::static_transpose>(&b_producer->op)) {
    // The producer of this pack is a transpose. If it is transposing the rows
    // and columns of B, we can handle it with packing.
    const ynn_node::static_transpose& op =
        std::get<ynn_node::static_transpose>(b_producer->op);
    if (!op.alias && (op.permutation[0] == 0 || op.permutation[1] == 0)) {
      // We can handle B with a transpose of n and k1 by fusing the transpose
      // with packing, which avoids realizing the transpose into memory, which
      // is a significant optimization. To implement this, we need to make a
      // clone of the transpose op that allows aliasing, and use that instead.
      // We don't rewrite the existing transpose op in the (unlikely) event that
      // it is used elsewhere. The existing transpose op will likely be
      // invalidated as a dead operation.
      id = YNN_INVALID_VALUE_ID;
      return define_static_transpose(&subgraph, op.permutation,
                                     b_producer->inputs[0], &id,
                                     /*alias=*/true);
    }
  }
  return ynn_status_unsupported_parameter;
}

bool is_constant(const ynn_subgraph& subgraph, uint32_t id, int depth = 5) {
  if (id == YNN_INVALID_VALUE_ID) {
    return false;
  }
  if (depth-- <= 0) {
    // We hit our limit for how far we look for constants.
    return false;
  }
  const ynn_value& value = subgraph.value(id);
  if (value.is_static()) {
    return true;
  } else if (value.is_external_input()) {
    return false;
  }
  const ynn_node* producer = subgraph.get_producer(id);
  assert(producer);
  if (std::all_of(
          producer->inputs.begin(), producer->inputs.end(),
          [&](uint32_t i) { return is_constant(subgraph, i, depth); })) {
    // If all of the inputs to the producer are constant, this will be constant
    // too.
    return true;
  }
  return false;
}

bool should_pack_b(const ynn_subgraph& subgraph, size_t num_k_dims,
                   const ynn_value& a, const ynn_value& b,
                   const dot_kernel& kernel,
                   const dot_kernel& unpacked_kernel) {
  if (!unpacked_kernel.kernel ||
      (unpacked_kernel.flags & dot_flag::unaligned_b) == 0) {
    // We need to pack B.
    return true;
  }
  if (unpacked_kernel.cost > kernel.cost * 2.0f) {
    // We think the unpacked kernel is a lot slower than the packed kernel, we
    // should pack.
    return true;
  }
  if (is_constant(subgraph, b.id)) {
    // TODO(dsharlet): If B is huge and static, it might cost a lot of memory to
    // pre-pack B, and it might not be so bad to just not pack it (or pack it on
    // the fly as if B were dynamic).
    return true;
  }
  const int block_m = unpacked_kernel.block_m;
  slinky::expr a_batch = 1;
  slinky::expr b_batch = 1;
  for (size_t i = num_k_dims + 1; i < a.extents.size(); ++i) {
    if (a.extent(i).defined()) a_batch *= a.extent(i);
  }
  for (size_t i = num_k_dims + 1; i < b.extents.size(); ++i) {
    if (b.extent(i).defined()) b_batch *= b.extent(i);
  }

  // How many blocks are we going to split A into?
  slinky::expr blocks_m =
      slinky::ceil_div<slinky::expr>(a.extent(num_k_dims), block_m);

  if (slinky::prove_true(blocks_m * a_batch <= b_batch * 10)) {
    // A is small relative to B, we should not bother packing B because we're
    // not going to read it very much, and packing itself costs a read of B.
    return false;
  }
  return true;
}

}  // namespace

extern "C" {

ynn_status ynn_define_dot(ynn_subgraph_t subgraph, size_t num_k_dims,
                          uint32_t input_a_id, uint32_t input_b_id,
                          uint32_t input_c_id, uint32_t* output_id,
                          uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_a_id));
  assert(subgraph->is_valid_value(input_b_id));
  // TODO: We can handle more than this, with loops outside of dot kernels.
  assert(num_k_dims <= 3);
  assert(num_k_dims > 0);

  const bool b_transposed =
      always_alias_transpose(*subgraph, input_b_id) == ynn_status_success;

  const ynn_value& a = subgraph->value(input_a_id);
  const ynn_value& b = subgraph->value(input_b_id);
  // If any input is a float, the output should be a float.
  const ynn_type c_type = !type_is_integral(a.type) || !type_is_integral(b.type)
                              ? ynn_type_fp32
                              : ynn_type_int32;
  ynn_value& c = subgraph->get_output_value(output_id, c_type);
  if (input_c_id != YNN_INVALID_VALUE_ID) {
    const ynn_value& init_c = subgraph->value(input_c_id);
    assert(init_c.type == c_type);
    (void)init_c;
  }
  assert(c.type == c_type);

  // Kernel selection is an interesting problem to solve. Here are the issues
  // affecting it:
  // - The optimal kernel may depend significantly on the shape of A and/or B.
  // - We may not know the shape of A, or B
  // - We may need to choose a kernel to pack B, before we know the shape of A.
  // - A kernel has parameters that once chosen, limit the choice of kernel:
  //   - `tile_k`, `block_n` determine the layout of packed B values.
  //   - `transpose_a` requires a transpose of A be inserted into the graph.
  // - It may be be unprofitable to pack B, in which case, we should choose a
  //   kernel that does not require packing B.
  //
  // Because of all of these issues, our procedure is as follows:
  // 1. When constructing the graph (while shapes are symbolic), use any known
  //    shape parameters to estimate what the optimal kernel is. If we don't
  //    know a shape parameter, just guess the shape is big.
  // 2. Use this estimated kernel to determine the packing layout of B, and to
  //    insert a transpose of A if needed.
  // 3. When running the dot, we can attempt to find a better kernel for the
  //    shape we have (now fully known), as long as the better kernel is
  //    compatible with the packed B layout and the transposed-ness of A.

  dot_type type = {a.type, b.type, c.type};
  dot_shape shape;
  learn_shape_from_b(shape, num_k_dims, b);
  static constexpr dot_packed_shape no_tile_k = {0, 1};
  const dot_packed_shape* packed_shape = nullptr;
  const bool consistent_arithmetic =
      (!type_is_integral(a.type) || !type_is_integral(b.type)) &&
      (subgraph->flags & YNN_FLAG_CONSISTENT_ARITHMETIC) != 0;
  dot_kernel kernel =
      get_dot_kernel(type, shape, packed_shape, consistent_arithmetic);
  dot_kernel unpacked_kernel;
  if (b_transposed) {
    // If b is transposed, we might as well use the packing to do it.
    // TODO(dsharlet): If the input is transposed, and used elsewhere, it might
    // be better to let the input be transposed, and attempt to use an unpacked
    // kernel instead. This is a tricky global optimization to make. Two
    // transposes is not ideal, but packing should make the dot faster. It would
    // be nice if we could simply describe all the ways in which something could
    // be implemented, and let a global cost optimization decide what to do...
  } else {
    unpacked_kernel = kernel;
    if (kernel.tile_k != 1) {
      unpacked_kernel =
          get_dot_kernel(type, shape, &no_tile_k, consistent_arithmetic);
    }
  }

  // Insert a packing node (if necessary).
  const bool pack_b =
      should_pack_b(*subgraph, num_k_dims, a, b, kernel, unpacked_kernel);
  uint32_t packed_b_id = YNN_INVALID_VALUE_ID;
  if (!pack_b) {
    // We don't want or need to pack B, but we still need to reshape it as if it
    // were packed.
    static constexpr int32_t tile_k_blocks_n[2] = {-1, -4};
    ynn_status status = ynn_define_static_expand_dims(
        subgraph, 2, tile_k_blocks_n, input_b_id, &packed_b_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  } else {
    packed_b_id = define_pack_b(subgraph, type, kernel, num_k_dims,
                                consistent_arithmetic, input_b_id);
  }

  ynn_node node;
  // We need both the original input b (for shape inference only) and packed b.
  node.inputs = {input_a_id, input_b_id, input_c_id, packed_b_id};
  node.outputs = {*output_id};
  node.op = ynn_node::dot{num_k_dims};

  // Propagate shape.
  int c_rank;
  if (a.rank() == 1) {
    // TODO These special cases for a rank 1 are pretty anoying, there must be a
    // more elegant generalization of a rank 1 cases.
    c_rank = b.rank() - num_k_dims;
  } else {
    c_rank = std::max(a.rank(), b.rank()) + 1 - num_k_dims;
  }
  c.extents.clear();
  c.extents.resize(c_rank);

  // The operation is
  //
  //   output(j, ...) = c(j, ...)
  //   output(j, ...) += a(k1, k2, k3, ...) * b(j, k1, k2, k3, ...)
  //
  // So we grab the two dimensions we know, and then propagate the elementwise
  // batch dimensions.

  // inputs `b` and `c` have an elementwise dimension 0.
  subgraph->infer_elementwise_shape(node, 1, 0, 0, 0,
                                    type_element_count(b.type));
  subgraph->infer_elementwise_shape(node, 2, 0, 0, 0);

  if (c_rank >= 2) {
    subgraph->infer_elementwise_shape(node, 0, 0, num_k_dims, 1);
    subgraph->infer_elementwise_shape(node, 2, 0, 1, 1);
  }

  // The rest of the dimensions are elementwise.
  for (size_t d = 2; d < c_rank; ++d) {
    subgraph->infer_elementwise_shape(node, 0, 0, d + num_k_dims - 1, d);
    subgraph->infer_elementwise_shape(node, 1, 0, d + num_k_dims - 1, d);
    subgraph->infer_elementwise_shape(node, 2, 0, d, d);
  }

  // The k-dims must match.
  for (int d = 0; d < num_k_dims; ++d) {
    slinky::expr a_k_dim = a.extent(d);
    slinky::expr b_k_dim = b.extent(d + 1);
    node.checks.push_back(
        {a_k_dim == b_k_dim,
         {"reduction dimension ", d, " (", a_k_dim, ") of ",
          ynn_node::input_idx{0}, ") does not match reduction dimension ",
          d + 1, " (", b_k_dim, ") of ", ynn_node::input_idx{1}}});
  }

  // After shape inference, we don't need input_b any more.
  // TODO(dsharlet): With a better API for `infer_elementwise_shape`, we
  // wouldn't need to put input_b into the inputs in the first place.
  node.inputs.erase(node.inputs.begin() + 1);

  const bool transpose_a = kernel.flags & dot_flag::transpose_a;
  if (transpose_a) {
    // The kernel we want to use has a transposed a.
    node.inputs[0] =
        define_transpose_a(*subgraph, kernel.tile_k, num_k_dims, input_a_id);
  }

  // If we're using an unpacked kernel, we'll be reading columns of B, make sure
  // that we read at least a cache line at a time.
  const int b_elem_size = type_size_bytes(b.type);
  const int block_n_unpacked = consistent_arithmetic
                                   ? consistent_block_n
                                   : std::max(YNN_CACHE_LINE_SIZE / b_elem_size,
                                              unpacked_kernel.block_n);
  node.create = [consistent_arithmetic, pack_b, transpose_a, block_n_unpacked](
                    const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::dot& op = std::get<ynn_node::dot>(node.op);
    const size_t num_k_dims = op.num_k_dims;
    const ynn_runtime_value& input_a = runtime.value(node.inputs[0]);
    ynn_runtime_value input_c;
    if (node.inputs[1] != YNN_INVALID_VALUE_ID) {
      input_c = runtime.value(node.inputs[1]);
    } else {
      input_c.buffer = runtime.null_buffer();
    }
    ynn_runtime_value& packed_b = runtime.value(node.inputs[2]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    if (pack_b) {
      require_contiguous(*packed_b.buffer, 3);
    }
    output.make_buffer(runtime);

    std::vector<slinky::var> dims = runtime.globals.make_dims(output.rank());
    slinky::var j = dims[0];

    // A: We need all of the k dims, i is elementwise.
    slinky::box_expr a_bounds(num_k_dims);
    for (size_t i = 0; i < num_k_dims; ++i) {
      a_bounds[i] = all_bounds(input_a.extent(i));
    }

    // B: We need all of the k dims, j is elementwise. j has been split into
    // two dimensions.
    slinky::box_expr b_bounds(num_k_dims + 3);
    b_bounds[0] = all_bounds(packed_b.extent(0));  // ki
    b_bounds[1] = all_bounds(packed_b.extent(1));  // ni
    b_bounds[2] = all_bounds(packed_b.extent(2));  // ko
    // When we split a packed dimension, the inner part of the split remains
    // packed, but the outer part is not.
    b_bounds[3] = slinky::point(j) / packed_b.extent(1);
    for (size_t i = 1; i < num_k_dims; ++i) {
      b_bounds[i + 3] = all_bounds(packed_b.extent(i + 3));
    }

    // C: Elementwise
    slinky::box_expr c_bounds;
    if (input_c.rank() >= 1) {
      c_bounds.push_back(elementwise_bounds(dims[0], input_c.extent(0)));
    }

    // Batch dims are elementwise too.
    for (size_t i = 1; i < dims.size(); ++i) {
      if (i + num_k_dims - 1 < input_a.rank()) {
        a_bounds.push_back(
            elementwise_bounds(dims[i], input_a.extent(i + num_k_dims - 1)));
      }
      if (i >= 2 && i + 2 + num_k_dims - 1 < packed_b.rank()) {
        b_bounds.push_back(elementwise_bounds(
            dims[i], packed_b.extent(i + 2 + num_k_dims - 1)));
      }
      if (i < input_c.rank()) {
        c_bounds.push_back(elementwise_bounds(dims[i], input_c.extent(i)));
      }
    }

    assert(a_bounds.size() == input_a.rank());
    assert(b_bounds.size() == packed_b.rank());
    assert(c_bounds.size() == input_c.rank());

    slinky::call_stmt::attributes attrs;
    attrs.name = node.to_string();
    // Allow the input_c and output to be computed in-place, which means we
    // don't need to initialize the accumulator.
    if (allow_in_place(input_c.id, output.id, runtime.subgraph)) {
      attrs.allow_in_place = (1 << 2);
    }
    dot_type dot_type = {input_a.type, packed_b.type, output.type};
    auto func =
        slinky::func::make(make_dot_impl(dot_type, consistent_arithmetic,
                                         transpose_a, pack_b, num_k_dims),
                           {{input_a.buffer, std::move(a_bounds)},
                            {packed_b.buffer, std::move(b_bounds)},
                            {input_c.buffer, std::move(c_bounds)}},
                           {{output.buffer, dims}}, std::move(attrs));

    slinky::expr block_n = pack_b ? packed_b.extent(1) : block_n_unpacked;
    slinky::expr n = output.extent(0);
    slinky::expr m = output.extent(1);

    // Compute k from b because it is more likely to be constant.
    slinky::expr k = packed_b.extent(0) * packed_b.extent(2);
    for (size_t d = 1; d < num_k_dims; ++d) {
      k *= packed_b.extent(3 + d);
    }

    slinky::expr split_n, split_m;
    std::tie(split_n, split_m) =
        choose_split_factors(runtime, m, n, k, block_n);

    if (slinky::prove_true(n <= block_n)) {
      // We know n is smaller than the side of the area we want to compute,
      // don't split it.
      split_n = {};
    }

    std::vector<slinky::index_t> loop_order;
    if (output.rank() >= 2) {
      loop_order = {0, 1};
      if (pack_b && !packed_b.is_static()) {
        // Loop over n first so we don't redundantly compute the packing for
        // each split of m.
        std::swap(loop_order[0], loop_order[1]);
      }
    }

    slinky::expr splits[] = {split_n, split_m};
    auto sched = runtime.make_schedule(dims, output.buffer, node.outputs[0],
                                       splits, 1, loop_order);

    // We want to use exactly these loop splits for two innermost dot loops.
    for (size_t i = 0; i < std::min<std::size_t>(2, sched->loop_splits.size());
         i++) {
      sched->loop_splits[i].step_is_required = true;
    }

    // Schedule the output buffer to be stored at the same level as it's
    // computed at.
    ynn::scheduled_buffer sched_output_buffer = {output.buffer, 0};
    sched->scheduled_buffers.push_back(std::move(sched_output_buffer));

    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };

  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
