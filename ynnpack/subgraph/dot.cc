// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/dot.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/span.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/cost_model/cost_model.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/kernels/dot/schedule.h"
#include "ynnpack/subgraph/copy.h"
#include "ynnpack/subgraph/dot_pack.h"
#include "ynnpack/subgraph/dot_schedule.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"
#include "slinky/base/arithmetic.h"
#include "slinky/base/thread_pool.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

using slinky::index_t;

namespace ynn {

bool prefer_uint8_dot(ynn_type b_type) {
  if (!type_is_integral(b_type)) {
    return false;
  }

  const dot_cost_models& cost_models = get_dot_cost_models();

  // Get the kernels we would use for both int8 and uint8.
  dot_kernel int8 =
      get_dot_kernel({ynn_type_int8, b_type, ynn_type_int32}, cost_models);
  dot_kernel uint8 =
      get_dot_kernel({ynn_type_uint8, b_type, ynn_type_int32}, cost_models);
  if (!uint8.kernel || !uint8.cost_model) {
    return false;
  }
  if (!int8.kernel || !int8.cost_model) {
    return true;
  }

  // Find a size that is a common multiple of both kernels.
  const size_t m = int8.block_m * uint8.block_m;
  const size_t n = int8.block_n * uint8.block_n;
  const size_t k = int8.block_k * uint8.block_k;
  // Estimate the cost of both kernels.
  const float uint8_cost = uint8.estimate_cost(m, n, k);
  const float int8_cost = int8.estimate_cost(m, n, k);
  return uint8_cost < int8_cost;
}

namespace {

const dot_cost_models& get_dot_cost_models(const slinky::eval_context& ctx) {
  assert(ctx.user_data);
  return *reinterpret_cast<const dot_cost_models*>(ctx.user_data);
}

struct dot_b_layout {
  index_t tile_k;
  index_t block_n;
  index_t stride_n;
  std::array<size_t, 3> k_strides;
};

dot_b_layout extract_b_layout(slinky::raw_buffer& b, ynn_type b_type,
                              bool packed_b, size_t num_k_dims, index_t n,
                              index_t c_min_n, const slinky::dim& r_k1,
                              const slinky::dim& r_k2,
                              const slinky::dim& r_k3) {
  const slinky::dim& dummy_dim = slinky::dim::broadcast();
  dot_b_layout layout;

  if (packed_b) {
    const slinky::dim& b_k1i = b.dim(0);
    const slinky::dim& b_ni = b.dim(1);
    const slinky::dim& b_k1o = b.dim(2);
    const slinky::dim& b_no = b.dim(3);
    const slinky::dim& b_k2 = num_k_dims >= 2 ? b.dim(4) : dummy_dim;
    const slinky::dim& b_k3 = num_k_dims >= 3 ? b.dim(5) : dummy_dim;

    layout.tile_k = b_k1i.extent() * type_element_count(b_type);
    layout.block_n = b_ni.extent();
    assert(is_power_of_two(layout.tile_k));
    assert(b_k1i.min() == 0);
    assert(b_k1i.extent() == 1 || b_k1i.stride() == b.elem_size);
    assert(b_ni.min() == 0);
    assert(b_ni.extent() == 1 || b_ni.stride() == b.elem_size * b_k1i.extent());
    assert(b_k1o.extent() == 1 || b_k1o.stride() % layout.tile_k == 0);
    assert(b_k1o.stride() % layout.tile_k == 0);
    assert(b_no.extent() == 1 || b_no.stride() % layout.block_n == 0);
    assert(!b_k1o.is_folded());
    assert(!b_no.is_folded());
    assert(!b_k2.is_folded());
    assert(!b_k3.is_folded());

    layout.stride_n = b_no.stride() / layout.block_n;
    layout.k_strides = {
        static_cast<size_t>(b_k1o.stride() / layout.tile_k),
        static_cast<size_t>(b_k2.stride()),
        static_cast<size_t>(b_k3.stride()),
    };

    b.base = offset_bytes(
        b.base,
        (r_k3.min() - b_k3.min()) * layout.k_strides[2] +
            (r_k2.min() - b_k2.min()) * layout.k_strides[1] +
            (r_k1.min() - b_k1o.min() * layout.tile_k) * layout.k_strides[0]);

    assert(c_min_n % layout.block_n == 0);
    slice_dims(b, 3);
    b.slice(0, slinky::in_bounds{c_min_n / layout.block_n});
    slice_dims(b, num_k_dims - 1);
  } else {
    const slinky::dim& b_no = b.dim(0);
    const slinky::dim& b_k1o = b.dim(1);
    const slinky::dim& b_k2 = num_k_dims >= 2 ? b.dim(2) : dummy_dim;
    const slinky::dim& b_k3 = num_k_dims >= 3 ? b.dim(3) : dummy_dim;

    layout.tile_k = 1;
    layout.block_n = n;
    assert(b_no.extent() == 1 || b_no.stride() == b.elem_size);
    assert(!b_k1o.is_folded());
    assert(!b_no.is_folded());
    assert(!b_k2.is_folded());
    assert(!b_k3.is_folded());

    layout.stride_n = b_no.stride();
    layout.k_strides = {
        static_cast<size_t>(b_k1o.stride()),
        static_cast<size_t>(b_k2.stride()),
        static_cast<size_t>(b_k3.stride()),
    };

    b.base = offset_bytes(b.base,
                          (r_k3.min() - b_k3.min()) * layout.k_strides[2] +
                              (r_k2.min() - b_k2.min()) * layout.k_strides[1] +
                              (r_k1.min() - b_k1o.min()) * layout.k_strides[0]);

    const index_t slice_n = b.dim(0).extent() == 1 ? b.dim(0).min() : c_min_n;
    b.slice(0, slinky::in_bounds{slice_n});
    slice_dims(b, num_k_dims);
  }

  return layout;
}

struct dot_a_layout {
  index_t tile_k;
  index_t stride_m;
  std::array<size_t, 3> k_strides;
};

dot_a_layout extract_a_layout(slinky::raw_buffer& a, bool transposed_a,
                              size_t num_k_dims, index_t m, index_t c_min_m,
                              const slinky::dim& r_k1, const slinky::dim& r_k2,
                              const slinky::dim& r_k3) {
  const slinky::dim& dummy_dim = slinky::dim::broadcast();
  dot_a_layout layout;

  if (transposed_a) {
    const slinky::dim& a_k1i = a.dim(0);
    const slinky::dim& a_mi = a.dim(1);
    const slinky::dim& a_k1o = a.dim(2);
    const slinky::dim& a_k2 = num_k_dims >= 2 ? a.dim(3) : dummy_dim;
    const slinky::dim& a_k3 = num_k_dims >= 3 ? a.dim(4) : dummy_dim;
    const slinky::dim& a_m = a.dim(2 + num_k_dims);

    layout.tile_k = a_k1i.extent();
    layout.stride_m = a_mi.stride();
    assert(a_k1i.min() == 0);
    assert(layout.tile_k == 1 || a_k1i.stride() == a.elem_size);
    assert(!a_m.is_folded(c_min_m, c_min_m + m - 1));
    assert(!a_k1i.is_folded());
    assert(!a_mi.is_folded());
    assert(!a_k1o.is_folded());
    assert(!a_k2.is_folded());
    assert(!a_k3.is_folded());

    layout.k_strides = {
        static_cast<size_t>(a_k1o.stride() / layout.tile_k),
        static_cast<size_t>(a_k2.stride()),
        static_cast<size_t>(a_k3.stride()),
    };

    a.base = offset_bytes(
        a.base,
        (r_k3.min() - a_k3.min()) * layout.k_strides[2] +
            (r_k2.min() - a_k2.min()) * layout.k_strides[1] +
            (r_k1.min() - a_k1o.min() * layout.tile_k) * layout.k_strides[0]);

    assert(a_m.extent() == 1 || layout.stride_m == a.elem_size * layout.tile_k);
    (void)a_m;

    const index_t tile_m = a_mi.extent();
    slice_dims(a, 2 + num_k_dims);
    assert(c_min_m % tile_m == 0);
    a.slice(0, slinky::in_bounds{c_min_m / tile_m});
  } else {
    const slinky::dim& a_k1o = a.dim(0);
    const slinky::dim& a_k2 = num_k_dims >= 2 ? a.dim(1) : dummy_dim;
    const slinky::dim& a_k3 = num_k_dims >= 3 ? a.dim(2) : dummy_dim;
    const slinky::dim& a_m = a.dim(num_k_dims);

    layout.tile_k = 1;
    layout.stride_m = a_m.stride();
    assert(!a_m.is_folded(c_min_m, c_min_m + m - 1));
    assert(!a_k1o.is_folded());
    assert(!a_k2.is_folded());
    assert(!a_k3.is_folded());

    layout.k_strides = {
        static_cast<size_t>(a_k1o.stride()),
        static_cast<size_t>(a_k2.stride()),
        static_cast<size_t>(a_k3.stride()),
    };

    a.base = offset_bytes(a.base,
                          (r_k3.min() - a_k3.min()) * layout.k_strides[2] +
                              (r_k2.min() - a_k2.min()) * layout.k_strides[1] +
                              (r_k1.min() - a_k1o.min()) * layout.k_strides[0]);

    assert(a_k1o.extent() == 1 ||
           layout.k_strides[0] == static_cast<size_t>(a.elem_size));

    slice_dims(a, num_k_dims);
    a.slice(0, slinky::in_bounds{c_min_m});
  }

  return layout;
}

void run_dot_k1_tail(const std::array<size_t, 3>& cache_budgets,
                     dot_loop* loops_storage, const dot_kernel& kernel,
                     index_t m, index_t n, index_t k1, index_t k1_tail,
                     index_t k2, index_t k3, index_t tile_k, index_t block_m,
                     index_t block_n, const dot_a_layout& a_layout,
                     const dot_b_layout& b_layout, index_t init_c_stride_m,
                     index_t c_stride_m, index_t c_stride_n,
                     slinky::raw_buffer& c, const slinky::raw_buffer& a,
                     const slinky::raw_buffer& b,
                     const slinky::raw_buffer& init_c,
                     dot_kernel_state* kernel_state) {
  std::array<size_t, 3> k_tail = {
      static_cast<size_t>(k1_tail),
      static_cast<size_t>(k2),
      static_cast<size_t>(k3),
  };
  auto loops = schedule_dot(cache_budgets, m, n, k_tail, block_m, block_n,
                            a.elem_size, b.elem_size, loops_storage);
  // Dot kernels can't handle k1 not aligned to tile_k. We handle that
  // here by making a padded copy of the unaligned elements and calling the
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
  auto call_kernel_tail =
      [&](index_t m, index_t n, span<const size_t> k, const void* a,
          size_t a_stride_m, span<const size_t> a_k_strides, const void* b,
          span<const size_t> b_k_strides, index_t init_c_stride_m,
          const void* init_c, void* c, dot_kernel_state* state = nullptr) {
        assert(m <= block_m);
        assert(n <= block_n);
        assert(k[0] < tile_k);
        for (index_t K3 = 0; K3 < k3; ++K3) {
          for (index_t K2 = 0; K2 < k2; ++K2) {
            for (index_t i = 0; i < m; ++i) {
              memcpy(offset_bytes(a_padded, i * a_padded_stride_m),
                     offset_bytes(a, i * a_stride_m + K3 * a_k_strides[2] +
                                         K2 * a_k_strides[1]),
                     k[0] * a_elem_size);
            }
            kernel.kernel(
                m, n, /*k3=*/1, /*k2=*/1, tile_k, a_padded_stride_m,
                /*a_stride_k3=*/0, /*a_stride_k2=*/0, a_padded,
                /*b_stride_k3=*/0, /*b_stride_k2=*/0, b_k_strides[0],
                offset_bytes(b, K3 * b_k_strides[2] + K2 * b_k_strides[1]),
                init_c_stride_m, init_c, c_stride_m, c, state);
            init_c_stride_m = c_stride_m;
            init_c = c;
          }
        }
      };
  slinky::for_each_element(
      [=](void* c, const void* a, const void* b, const void* init_c) {
        index_t tail_init_c_stride_m = init_c_stride_m;
        if (k1 != 0) {
          init_c = c;
          tail_init_c_stride_m = c_stride_m;
        }
        a = offset_bytes(a, a_layout.k_strides[0] * k1);
        b = offset_bytes(b, b_layout.k_strides[0] * k1);
        run_dot(loops, m, n, k_tail, a_layout.stride_m, a_layout.k_strides, a,
                b_layout.k_strides, b_layout.stride_n, b, tail_init_c_stride_m,
                init_c, c_stride_m, c_stride_n, c, call_kernel_tail,
                kernel_state);
      },
      c, a, b, init_c);
}

// The wrapper for the kernel we use when we actually want to run a dot kernel
// on some buffers.
auto make_dot_impl(dot_type type, bool consistent_arithmetic, bool symmetric_b,
                   bool transposed_a, bool pack_b, size_t num_k_dims,
                   int thread_count) {
  uint32_t kernel_flags = 0;
  if (consistent_arithmetic) {
    kernel_flags |= dot_flag::consistent_arithmetic;
  }
  if (symmetric_b) {
    kernel_flags |= dot_flag::symmetric_b;
  }
  if (!pack_b) {
    kernel_flags |= dot_flag::unaligned_b;
  }
  uint64_t arch_flags = get_supported_arch_flags();

  return [type, kernel_flags, transposed_a, pack_b, num_k_dims, arch_flags,
          thread_count](const slinky::call_stmt* call,
                        const slinky::eval_context& ctx) -> index_t {
    slinky::raw_buffer a = *ctx.lookup_buffer(call->inputs[0]);
    slinky::raw_buffer b = *ctx.lookup_buffer(call->inputs[1]);
    slinky::raw_buffer init_c = *ctx.lookup_buffer(call->inputs[2]);
    slinky::raw_buffer c = *ctx.lookup_buffer(call->outputs[0]);
    const slinky::raw_buffer& reduction_bounds =
        *ctx.lookup_buffer(call->outputs[1]);

    const slinky::dim& c_n = c.dim(0);
    if (c_n.empty()) {
      // Most things below transparently handle empty dimensions, but n has some
      // alignment requirements.
      return 0;
    }

    const slinky::dim& r_k1 = reduction_bounds.dim(0);
    const slinky::dim& r_k2 = reduction_bounds.dim(1);
    const slinky::dim& r_k3 = reduction_bounds.dim(2);

    const slinky::dim& c_m = c.dim(1);
    const index_t m = c_m.extent();
    const index_t n = c_n.extent();
    const index_t c_min_m = c_m.min();
    const index_t c_min_n = c_n.min();
    const index_t c_stride_m = c_m.stride();
    const index_t c_stride_n = c_n.stride();
    assert(!c_m.is_folded());
    assert(!c_n.is_folded());

    const slinky::dim& init_c_m = init_c.dim(1);
    const slinky::dim& init_c_n = init_c.dim(0);
    assert(!init_c_m.is_folded());
    assert(!init_c_n.is_folded());
    index_t init_c_stride_m = init_c_m.stride();

    if (r_k1.min() == 0 && r_k2.min() == 0 && r_k3.min() == 0) {
      if (init_c.base && init_c.base != c.base && n > 1) {
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
    } else {
      init_c_stride_m = c_stride_m;
      init_c = c;
    }

    init_c.slice(0, slinky::in_bounds{c_min_n});
    init_c.slice(0, slinky::in_bounds{c_min_m});
    slice_dims(c, 2);

    const dot_b_layout b_layout = extract_b_layout(
        b, type.b, pack_b, num_k_dims, n, c_min_n, r_k1, r_k2, r_k3);
    const dot_a_layout a_layout = extract_a_layout(
        a, transposed_a, num_k_dims, m, c_min_m, r_k1, r_k2, r_k3);

    const index_t k1_extent = r_k1.extent();
    const index_t k1 = k1_extent & ~(b_layout.tile_k - 1);
    const index_t k1_tail = k1_extent & (b_layout.tile_k - 1);
    const index_t k2 = r_k2.extent();
    const index_t k3 = r_k3.extent();

    std::array<size_t, 3> k = {
        static_cast<size_t>(k1),
        static_cast<size_t>(k2),
        static_cast<size_t>(k3),
    };

    // Find a kernel that is compatible with the packed data we have, and
    // matches whether A is transposed or not.
    std::optional<bool> require_transpose_a = std::make_optional(transposed_a);
    if (a_layout.stride_m == a_layout.k_strides[0] * a_layout.tile_k) {
      // If the stride of m and k1 are the same (i.e. A is a vector of tile_k
      // values), then we don't care if the kernel is transposed or not.
      require_transpose_a = std::nullopt;
    }
    dot_shape shape;
    shape.m = m;
    shape.n = n;
    shape.k1 = k1;
    shape.k2 = k2;
    shape.k3 = k3;
    dot_packed_shape packed_shape;
    packed_shape.block_n = b_layout.block_n;
    packed_shape.tile_k = b_layout.tile_k;
    dot_kernel kernel = get_dot_kernel(
        type, get_dot_cost_models(ctx), shape, packed_shape, kernel_flags,
        require_transpose_a, arch_flags, thread_count);
    assert(kernel.kernel);
    assert(b_layout.tile_k == kernel.tile_k);
    const index_t block_m = kernel.block_m;

    auto call_kernel = [transposed_a, c_stride_m, kernel = kernel.kernel](
                           index_t m, index_t n, span<const size_t> k,
                           const void* a, size_t a_stride_m,
                           span<const size_t> a_k_strides, const void* b,
                           span<const size_t> b_k_strides,
                           index_t init_c_stride_m, const void* init_c, void* c,
                           dot_kernel_state* state = nullptr) {
      kernel(m, n, k[2], k[1], k[0], transposed_a ? a_k_strides[0] : a_stride_m,
             a_k_strides[2], a_k_strides[1], a, b_k_strides[2], b_k_strides[1],
             b_k_strides[0], b, init_c_stride_m, init_c, c_stride_m, c, state);
    };

    dot_kernel_state kernel_state = {};
    const std::array<size_t, 3> cache_budgets = get_dot_cache_budgets();

    // `schedule_dot` emits at most 1 k-loop, 1 loop per cache level, and 2
    // inner microkernel loops.
    dot_loop loops_storage[std::size(cache_budgets) + 3];

    if (k1) {
      auto loops =
          schedule_dot(cache_budgets, m, n, k, block_m, b_layout.block_n,
                       a.elem_size, b.elem_size, loops_storage);

      slinky::for_each_element(
          [=, &kernel_state](void* c, const void* a, const void* b,
                             const void* init_c) {
            run_dot(loops, m, n, k, a_layout.stride_m, a_layout.k_strides, a,
                    b_layout.k_strides, b_layout.stride_n, b, init_c_stride_m,
                    init_c, c_stride_m, c_stride_n, c, call_kernel,
                    &kernel_state);
          },
          c, a, b, init_c);
    }
    if (k1_tail) {
      run_dot_k1_tail(cache_budgets, loops_storage, kernel, m, n, k1, k1_tail,
                      k2, k3, b_layout.tile_k, block_m, b_layout.block_n,
                      a_layout, b_layout, init_c_stride_m, c_stride_m,
                      c_stride_n, c, a, b, init_c, &kernel_state);
    }

    return 0;
  };
}

ynn_status create_dot(const ynn_node& node, ynn_runtime& runtime) {
  const ynn_node::dot& op = std::get<ynn_node::dot>(node.op);
  if (!op.optimized) {
    YNN_LOG_ERROR() << "Cannot create runtime for unoptimized dot node; call "
                       "ynn_optimize_subgraph first.";
    return ynn_status_invalid_parameter;
  }
  const size_t num_k_dims = op.num_k_dims;
  const bool pack_b = op.packed_b;
  const bool transpose_a = op.transposed_a;
  const size_t tile_k = op.tile_k;
  const size_t tile_m = op.tile_m;
  const bool symmetric_b = (op.flags & YNN_NODE_FLAG_SYMMETRIC_B) != 0;

  ynn_runtime_value& input_a = runtime.value(node.inputs[0]);
  ynn_runtime_value& input_b = runtime.value(node.inputs[1]);
  ynn_runtime_value input_c;
  if (node.inputs[2] != YNN_INVALID_VALUE_ID) {
    input_c = runtime.value(node.inputs[2]);
  } else {
    input_c.buffer = runtime.null_buffer();
  }
  ynn_runtime_value& output = runtime.value(node.outputs[0]);

  const bool consistent_arithmetic =
      (!type_is_integral(input_a.type) || !type_is_integral(input_b.type)) &&
      (runtime.subgraph->flags & YNN_FLAG_CONSISTENT_ARITHMETIC) != 0;

  if (!transpose_a) {
    require_contiguous(*input_a.buffer, 1);
  }
  if (pack_b) {
    require_contiguous(*input_b.buffer, 3);
  } else {
    require_contiguous(*input_b.buffer, 1);
  }
  output.make_buffer(runtime);

  std::vector<slinky::var> output_dims =
      runtime.globals.make_dims(output.rank());
  slinky::var j = output_dims[0];

  slinky::buffer_expr_ptr reduction_buffer = slinky::buffer_expr::make(
      runtime.globals.symbols, "reduction", num_k_dims, 0);

  std::vector<slinky::var> all_dims;
  std::vector<slinky::var> reduction_dims;
  std::vector<slinky::expr> all_extents;

  int reduction_dim = 0;
  for (size_t d = 0; d < num_k_dims; ++d) {
    slinky::var r_dim = runtime.globals.make_reduction_dim(reduction_dim);
    all_dims.push_back(r_dim);
    reduction_dims.push_back(r_dim);

    const int a_k_dim = transpose_a ? 2 : 0;
    slinky::expr k_extent = input_a.extent(a_k_dim + d);
    if (transpose_a && d == 0) {
      // When A is transposed, its K1 dimension is split into blocks of size
      // tile_k. The logical extent of the reduction dimension should be the
      // total number of elements, so we multiply the number of blocks by the
      // block size.
      k_extent *= tile_k;
    }
    all_extents.push_back(k_extent);

    reduction_buffer->dim(reduction_dim).bounds =
        slinky::min_extent(0, k_extent);
    reduction_buffer->dim(reduction_dim).stride = 0;
    reduction_buffer->dim(reduction_dim).fold_factor = slinky::dim::unfolded;
    ++reduction_dim;
  }

  for (int i = 0; i < output.rank(); ++i) {
    all_dims.push_back(output_dims[i]);
    all_extents.push_back(output.extent(i));
  }

  // A: We need all of the k dims, i is elementwise.
  const int num_a_k_dims = num_k_dims + (transpose_a ? 2 : 0);
  slinky::box_expr a_bounds(std::min<int>(input_a.rank(), num_a_k_dims));
  if (transpose_a) {
    a_bounds[0] = all_bounds(input_a.physical_extent(0));
    a_bounds[1] = all_bounds(input_a.physical_extent(1));
    a_bounds[2] = slinky::point(slinky::simplify(reduction_dims[0] / tile_k));
    for (size_t d = 1; d < num_k_dims; ++d) {
      a_bounds[2 + d] = slinky::point(reduction_dims[d]);
    }
    if (output_dims.size() >= 2) {
      slinky::var i = output_dims[1];
      a_bounds.push_back(slinky::point(i) / tile_m);
    } else {
      a_bounds.push_back(slinky::point(0));
    }
  } else {
    for (size_t d = 0; d < a_bounds.size(); ++d) {
      a_bounds[d] = slinky::point(reduction_dims[d]);
    }
  }

  const int b_k1_dim = pack_b ? 2 : 1;
  const int b_n_dim = pack_b ? 3 : 0;
  const int b_k2_dim = pack_b ? 3 : 1;

  // B: We need all of the k dims, j is elementwise.
  slinky::box_expr b_bounds(num_k_dims + b_k2_dim);
  if (pack_b) {
    b_bounds[0] = all_bounds(input_b.physical_extent(0));  // ki
    b_bounds[1] = all_bounds(input_b.physical_extent(1));  // ni
  }
  // When we split a packed dimension, the inner part of the split remains
  // packed, but the outer part is not.
  const index_t b_tile_k = pack_b ? tile_k : 1;
  b_bounds[b_k1_dim] =
      slinky::point(slinky::simplify(reduction_dims[0] / b_tile_k));
  b_bounds[b_n_dim] =
      slinky::point(j) / (pack_b ? input_b.physical_extent(1) : 1);
  for (size_t d = 1; d < num_k_dims; ++d) {
    b_bounds[b_k2_dim + d] = slinky::point(reduction_dims[d]);
  }

  // C: Elementwise
  slinky::box_expr c_bounds;
  if (input_c.rank() >= 1) {
    c_bounds.push_back(elementwise_bounds(j, input_c.extents[0]));
  }

  // Batch dims are elementwise too.
  for (size_t i = 1; i < output_dims.size(); ++i) {
    if (transpose_a) {
      if (i >= 2 && i + num_a_k_dims - 1 < input_a.rank()) {
        a_bounds.push_back(elementwise_bounds(
            output_dims[i], input_a.extents[i + num_a_k_dims - 1]));
      }
    } else {
      if (i + num_a_k_dims - 1 < input_a.rank()) {
        a_bounds.push_back(elementwise_bounds(
            output_dims[i], input_a.extents[i + num_a_k_dims - 1]));
      }
    }
    if (i >= 2) {
      const size_t b_batch_dim = b_k2_dim + num_k_dims + (i - 2);
      if (b_batch_dim < input_b.rank()) {
        b_bounds.push_back(
            elementwise_bounds(output_dims[i], input_b.extents[b_batch_dim]));
      }
    }
    if (i < input_c.rank()) {
      c_bounds.push_back(
          elementwise_bounds(output_dims[i], input_c.extents[i]));
    }
  }

  while (a_bounds.size() < input_a.rank()) {
    a_bounds.push_back(slinky::point(0));
  }
  while (b_bounds.size() < input_b.rank()) {
    b_bounds.push_back(slinky::point(0));
  }

  assert(a_bounds.size() == input_a.rank());
  assert(b_bounds.size() == input_b.rank());
  assert(c_bounds.size() == input_c.rank());

  slinky::call_stmt::attributes attrs;
  attrs.name = node.to_string();
  // Allow the input_c and output to be computed in-place, which means we
  // don't need to initialize the accumulator.
  if (allow_in_place(input_c.id, output.id, *runtime.subgraph)) {
    attrs.allow_in_place = (1 << 2);
  }
  dot_type dot_type = {input_a.type, input_b.type, output.type};
  const int thread_count =
      runtime.threadpool() ? runtime.threadpool()->thread_count() + 1 : 1;
  auto func =
      slinky::func(make_dot_impl(dot_type, consistent_arithmetic, symmetric_b,
                                 transpose_a, pack_b, num_k_dims, thread_count),
                   {{input_a.buffer, std::move(a_bounds)},
                    {input_b.buffer, std::move(b_bounds)},
                    {input_c.buffer, std::move(c_bounds)}},
                   {{output.buffer, output_dims},
                    {reduction_buffer, std::move(reduction_dims)}},
                   {}, std::move(attrs));

  auto sched = make_dot_schedule(
      runtime, input_a, input_b, output, all_dims, all_extents, output_dims,
      num_k_dims, pack_b, transpose_a, op.block_n_unpacked, op.seconds_per_mac);
  runtime.funcs.push_back({std::move(func), std::move(sched)});
  return ynn_status_success;
}

void learn_shape_from_b(dot_shape& shape, size_t num_k_dims,
                        const ynn_value& b) {
  shape.n = as_constant(b.extent(0)).value_or(unknown_dot_extent);
  shape.k1 = as_constant(b.extent(1)).value_or(unknown_dot_extent);
  shape.k2 = num_k_dims >= 2
                 ? as_constant(b.extent(2)).value_or(unknown_dot_extent)
                 : 1;
  shape.k3 = num_k_dims >= 3
                 ? as_constant(b.extent(3)).value_or(unknown_dot_extent)
                 : 1;
}

bool can_fuse_transpose_into_pack_b(const ynn_subgraph& subgraph,
                                    uint32_t b_id) {
  const ynn_node* producer = subgraph.get_producer(b_id);
  if (!producer) return false;
  const auto* transpose_op =
      std::get_if<ynn_node::static_transpose>(&producer->op);
  if (!transpose_op || transpose_op->alias) return false;
  // `rewrite_pack_b_transpose` absorbs a non-aliased `static_transpose` into
  // `pack_b(transpose=true)` when dim 0 of the pre-transposed input becomes K
  // (dim 1 of B).
  return transpose_op->permutation.size() >= 2 &&
         transpose_op->permutation[1] == 0;
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

bool is_constant_or_gather_of_constant(const ynn_subgraph& subgraph,
                                       uint32_t id, int depth = 3) {
  if (is_constant(subgraph, id, depth)) {
    return true;
  }
  const ynn_node* producer = subgraph.get_producer(id);
  if (!producer) return false;
  if (const auto* gather = std::get_if<ynn_node::gather>(&producer->op)) {
    if (std::any_of(gather->axes.begin(), gather->axes.end(),
                    [](int32_t axis) { return axis < 2; })) {
      // The gathered dimensions are transposed by the pack, we can't
      // reassociate the ops.
      return false;
    }
    return is_constant(subgraph, producer->inputs[0], depth);
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
  if (is_constant_or_gather_of_constant(subgraph, b.id)) {
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

  // Assuming that the cost of packing is comparable to the cost of the packed
  // dot, how many blocks should we allow to be unpacked?
  int max_unpacked_blocks = 1;
  if (kernel.cost > 0.0f && unpacked_kernel.cost > kernel.cost) {
    const float relative_penalty =
        (unpacked_kernel.cost - kernel.cost) / kernel.cost;
    max_unpacked_blocks =
        std::clamp<int>(std::round(1.0f / relative_penalty), 1, 10);
  } else {
    max_unpacked_blocks = 10;
  }

  // The packing cost is not the only thing we need to consider. The logic so
  // far assumes that memory bandwidth is similar in both cases. However, the
  // loads of unpacked data are strided.
  const int b_elem_size = type_size_bytes(b.type);
  const int block_n = unpacked_kernel.block_n;
  if ((block_n * b_elem_size) % YNN_CACHE_LINE_SIZE != 0) {
    slinky::expr b_bytes = b_batch * b_elem_size;
    for (size_t i = 0; i < num_k_dims + 1; ++i) {
      if (b.extent(i).defined()) b_bytes *= b.extent(i);
    }

    const slinky::index_t cache_budget = get_dot_cache_budgets()[1];

    // The kernel doesn't load whole cache lines. This can require much more
    // memory bandwidth than the packed case.
    if (!slinky::prove_true(b_bytes < cache_budget || b.extent(0) <= block_n)) {
      // B doesn't fit in cache, we should pack if we load it more than once.
      max_unpacked_blocks = 1;
    }
  }

  // TODO: b/579134224 - Some CPUs perform very poorly with the large strided
  // loads of B in the unpacked case. It generally seems like newer CPUs do well
  // while older CPUs perform poorly. For now, we optimize for the newer CPU
  // case by not reducing `max_unpacked_blocks` for large B.

  if (slinky::prove_true(blocks_m * a_batch <= b_batch * max_unpacked_blocks)) {
    // A is small relative to B, we should not bother packing B because we're
    // not going to read it very much, and packing itself costs a read of B.
    return false;
  }
  return true;
}

ynn_type deduce_output_type(ynn_type a_type, ynn_type b_type) {
  // The rules here are:
  // - If either input is float, the result should be float, otherwise integer.
  // - The output should be max(max(a_bits, b_bits), 32) bits wide
#ifdef YNN_ENABLE_FP64
  if (a_type == ynn_type_fp64 || b_type == ynn_type_fp64) {
    return ynn_type_fp64;
  } else  // NOLINT
#endif
      if (type_is_floating_point(a_type) || type_is_floating_point(b_type)) {
    return ynn_type_fp32;
  } else {
    assert(type_is_integral(a_type) && type_is_integral(b_type));
    return ynn_type_int32;
  }
}

ynn_status define_dot(ynn_subgraph& subgraph, size_t num_k_dims,
                      uint32_t input_a_id, uint32_t input_b_id,
                      uint32_t input_c_id, uint32_t* output_id,
                      uint32_t flags) {
  assert(subgraph.is_valid_value(input_a_id));
  assert(subgraph.is_valid_value(input_b_id));
  assert(output_id);
  if (subgraph.value(input_a_id).rank() == 0) {
    // We could support broadcasting `a` more generally. For now, broadcast a
    // scalar `a` to [1, k...], where k... are the k dimensions of `b`.
    std::vector<slinky::expr> output_extents(num_k_dims + 1);
    for (size_t d = 0; d < num_k_dims; ++d) {
      output_extents[d] = subgraph.value(input_b_id).extent(d + 1);
    }
    uint32_t broadcast_a_id = YNN_INVALID_VALUE_ID;
    ynn_node broadcast;
    define_static_broadcast(subgraph, broadcast, std::move(output_extents),
                            input_a_id, &broadcast_a_id);
    if (broadcast.is_valid()) {
      subgraph.add_node(std::move(broadcast));
    }
    input_a_id = broadcast_a_id;
  }

  const ynn_value& a = subgraph.value(input_a_id);
  const ynn_value& b = subgraph.value(input_b_id);
  const ynn_type c_type = deduce_output_type(a.type, b.type);
  ynn_value& c = subgraph.get_output_value(output_id, c_type);
  if (input_c_id != YNN_INVALID_VALUE_ID) {
    const ynn_value& init_c = subgraph.value(input_c_id);
    assert(init_c.type == c.type);
    (void)init_c;
  }

  ynn_node node;
  node.inputs = {input_a_id, input_b_id, input_c_id};
  node.outputs = {*output_id};
  ynn_node::dot dot_op;
  dot_op.num_k_dims = num_k_dims;
  dot_op.flags = flags;
  node.op = dot_op;

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
  subgraph.infer_elementwise_shape(node, 1, 0, 0, 0);
  subgraph.infer_elementwise_shape(node, 2, 0, 0, 0);

  if (c_rank >= 2) {
    subgraph.infer_elementwise_shape(node, 0, 0, num_k_dims, 1);
    subgraph.infer_elementwise_shape(node, 2, 0, 1, 1);
  }

  // The rest of the dimensions are elementwise.
  for (size_t d = 2; d < c_rank; ++d) {
    subgraph.infer_elementwise_shape(node, 0, 0, d + num_k_dims - 1, d);
    subgraph.infer_elementwise_shape(node, 1, 0, d + num_k_dims - 1, d);
    subgraph.infer_elementwise_shape(node, 2, 0, d, d);
  }

  // The k-dims must match.
  for (int d = 0; d < num_k_dims; ++d) {
    slinky::expr a_k_dim = a.extent(d);
    slinky::expr b_k_dim = b.extent(d + 1);
    node.add_check(
        a_k_dim == b_k_dim,
        {"reduction dimension ", d, " (", a_k_dim, ") of ",
         ynn_node::input_idx{0}, ") does not match reduction dimension ", d + 1,
         " (", b_k_dim, ") of ", ynn_node::input_idx{1}});
  }

  node.create = create_dot;
  subgraph.add_node(std::move(node));
  return ynn_status_success;
}

}  // namespace

}  // namespace ynn

ynn_status ynn_subgraph::optimize_dots() {
  bool added_nodes = false;
  const size_t num_nodes = nodes.size();
  for (size_t i = 0; i < num_nodes; ++i) {
    if (!nodes[i].is_valid()) continue;
    const auto* dot_op = std::get_if<ynn_node::dot>(&nodes[i].op);
    if (!dot_op || dot_op->optimized) continue;

    const size_t num_k_dims = dot_op->num_k_dims;
    const uint32_t dot_flags = dot_op->flags;
    const uint32_t input_a_id = nodes[i].inputs[0];
    const uint32_t input_b_id = nodes[i].inputs[1];
    const uint32_t output_id = nodes[i].outputs[0];

    const ynn_value& a = value(input_a_id);
    const ynn_value& b = value(input_b_id);
    const ynn_value& c = value(output_id);

    // Kernel selection is an interesting problem to solve. Here are the issues
    // affecting it:
    // - The optimal kernel may depend significantly on the shape of A and/or B.
    // - We may not know the shape of A, or B
    // - We may need to choose a kernel to pack B, before we know the shape of
    // A.
    // - A kernel has parameters that once chosen, limit the choice of kernel:
    //   - `tile_k`, `block_n` determine the layout of packed B values.
    //   - `transpose_a` requires a transpose of A be inserted into the graph.
    // - It may be unprofitable to pack B, in which case, we should choose a
    //   kernel that does not require packing B.
    //
    // Because of all of these issues, our procedure is as follows:
    // 1. When optimizing the graph (while shapes are symbolic), use any known
    //    shape parameters to estimate what the optimal kernel is. If we don't
    //    know a shape parameter, just guess the shape is big.
    // 2. Use this estimated kernel to determine the packing layout of B, and to
    //    insert a transpose of A if needed.
    // 3. When running the dot, we can attempt to find a better kernel for the
    //    shape we have (now fully known), as long as the better kernel is
    //    compatible with the packed B layout and the transposed-ness of A.

    ynn::dot_type type = {a.type, b.type, c.type};
    ynn::dot_shape shape;
    ynn::learn_shape_from_b(shape, num_k_dims, b);
    static constexpr ynn::dot_packed_shape no_tile_k = {0, 1};
    static constexpr ynn::dot_packed_shape packed_shape = {};
    const bool symmetric_b = (dot_flags & YNN_NODE_FLAG_SYMMETRIC_B) != 0;
    const bool consistent_arithmetic =
        (!ynn::type_is_integral(a.type) || !ynn::type_is_integral(b.type)) &&
        (flags & YNN_FLAG_CONSISTENT_ARITHMETIC) != 0;
    uint32_t kernel_flags =
        consistent_arithmetic ? ynn::dot_flag::consistent_arithmetic : 0;
    if (symmetric_b) {
      kernel_flags |= ynn::dot_flag::symmetric_b;
    }
    const ynn::dot_cost_models& cost_models = ynn::get_dot_cost_models();

    // Dot kernel cost models are fit on small tiles that fit in L1 cache. We
    // also assume that the dot will be tiled such that the operands fit in L1
    // cache. To make these assumptions more valid, we can clamp the shape used
    // for kernel selection to a small size more likely to be well represented
    // by the cost model.
    shape.m = std::min<size_t>(shape.m, 480);
    shape.n = std::min<size_t>(shape.n, 384);
    shape.k1 = std::min<size_t>(shape.k1, 256);

    ynn::dot_kernel kernel = ynn::get_dot_kernel(type, cost_models, shape,
                                                 packed_shape, kernel_flags);
    ynn::dot_kernel unpacked_kernel;
    const bool b_transposed =
        ynn::can_fuse_transpose_into_pack_b(*this, input_b_id);
    if (b_transposed) {
      // If b is transposed, we might as well use the packing to do it.
      // TODO(dsharlet): If the input is transposed, and used elsewhere, it
      // might be better to let the input be transposed, and attempt to use an
      // unpacked kernel instead. This is a tricky global optimization to make.
      // Two transposes is not ideal, but packing should make the dot faster. It
      // would be nice if we could simply describe all the ways in which
      // something could be implemented, and let a global cost optimization
      // decide what to do...
    } else {
      unpacked_kernel = kernel;
      if (kernel.tile_k != 1) {
        unpacked_kernel =
            ynn::get_dot_kernel(type, cost_models, shape, no_tile_k,
                                kernel_flags | ynn::dot_flag::unaligned_b);
      }
    }

    const int b_elem_size = ynn::type_size_bytes(b.type);
    const bool pack_b =
        ynn::should_pack_b(*this, num_k_dims, a, b, kernel, unpacked_kernel);

    // Insert a packing node (if necessary). Note that adding values/nodes may
    // reallocate `values` and `nodes`, so do not hold references across calls.
    uint32_t packed_b_id = input_b_id;
    if (pack_b) {
      packed_b_id =
          ynn::define_pack_b(*this, type, kernel, num_k_dims, input_b_id);
      added_nodes = true;
    } else {
      kernel = unpacked_kernel;
    }

    const bool transpose_a = (kernel.flags & ynn::dot_flag::transpose_a) != 0;
    uint32_t transposed_a_id = input_a_id;
    if (transpose_a) {
      // The kernel we want to use has a transposed a.
      // By definition, `m_dim` is the first dimension after the k dims.
      const int m_dim = static_cast<int>(num_k_dims);
      transposed_a_id = ynn::define_transpose_a(
          *this, kernel.tile_m, kernel.tile_k, m_dim, input_a_id);
      added_nodes = true;
    }

    // If we're using an unpacked kernel, we'll be reading columns of B, make
    // sure that we read at least a cache line at a time.
    const int block_n_unpacked = std::max<int>(
        YNN_CACHE_LINE_SIZE / b_elem_size, unpacked_kernel.block_n);
    // Estimate the throughput of the kernel we expect to run from its cost
    // model (for the shape it was selected for), to decide how small tasks can
    // be.
    const float seconds_per_mac =
        kernel.cost /
        (static_cast<float>(std::max<size_t>(1, shape.m)) *
         std::max<size_t>(1, shape.n) * std::max<size_t>(1, shape.k1));

    ynn_node& node = nodes[i];
    auto& op = std::get<ynn_node::dot>(node.op);
    node.inputs[0] = transposed_a_id;
    node.inputs[1] = packed_b_id;
    op.packed_b = pack_b;
    op.transposed_a = transpose_a;
    op.tile_m = kernel.tile_m;
    op.tile_k = kernel.tile_k;
    op.block_n_unpacked = block_n_unpacked;
    op.seconds_per_mac = seconds_per_mac;
    op.optimized = true;
  }

  if (added_nodes) {
    topological_sort();
  }
  return ynn_status_success;
}

namespace ynn {

extern "C" {

ynn_status ynn_define_dot(ynn_subgraph_t subgraph, size_t num_k_dims,
                          uint32_t input_a_id, uint32_t input_b_id,
                          uint32_t input_c_id, uint32_t* output_id,
                          uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("dot", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("dot", subgraph, "input_a_id", input_a_id));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("dot", subgraph, "input_b_id", input_b_id));
  YNN_RETURN_IF_ERROR(validate_input_tensor("dot", subgraph, "input_c_id",
                                            input_c_id, /*optional=*/true));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("dot", subgraph, "output_id", output_id));

  if (num_k_dims == 0 || num_k_dims > 3) {
    YNN_LOG_ERROR() << "For node `dot`, `num_k_dims` must be in [1, 3], got "
                    << num_k_dims;
    return ynn_status_invalid_parameter;
  }

  uint32_t add_c_id = YNN_INVALID_VALUE_ID;
  if (input_c_id != YNN_INVALID_VALUE_ID) {
    const ynn_value& a = subgraph->value(input_a_id);
    const ynn_value& input_c = subgraph->value(input_c_id);
    if (a.rank() != 1 && slinky::prove_true(a.extent(num_k_dims) == 1) &&
        !slinky::prove_true(a.extent(num_k_dims) == input_c.extent(1))) {
      // If `a` broadcasts in `m` and `c` has `m > 1`, compute `dot(a, b)` first
      // and let `add` broadcast `m` instead of redundantly computing the
      // broadcasted dot output.
      add_c_id = input_c_id;
      input_c_id = YNN_INVALID_VALUE_ID;
    }
  }

  const ynn_value& a = subgraph->value(input_a_id);
  const ynn_value& b = subgraph->value(input_b_id);
  const ynn_type c_type = deduce_output_type(a.type, b.type);

  auto maybe_convert_c = [&](uint32_t& c_id) -> ynn_status {
    if (c_id != YNN_INVALID_VALUE_ID) {
      const ynn_value& input_c = subgraph->value(c_id);
      if (input_c.type != c_type) {
        uint32_t input_c_converted_id = YNN_INVALID_VALUE_ID;
        YNN_RETURN_IF_ERROR(ynn_define_convert(subgraph, c_id, c_type,
                                               &input_c_converted_id, flags));
        c_id = input_c_converted_id;
      }
    }
    return ynn_status_success;
  };
  YNN_RETURN_IF_ERROR(maybe_convert_c(input_c_id));
  YNN_RETURN_IF_ERROR(maybe_convert_c(add_c_id));

  uint32_t convert_to_id = YNN_INVALID_VALUE_ID;
  if (*output_id != YNN_INVALID_VALUE_ID) {
    const ynn_value& c = subgraph->value(*output_id);
    if (c.type != c_type) {
      // The type we want to compute is different from the output type. We're
      // going to compute the result into an intermediate tensor, and insert
      // a convert to the actual output_id after.
      convert_to_id = *output_id;

      // Just let define_dot make the output.
      *output_id = YNN_INVALID_VALUE_ID;
    }
  }

  if (add_c_id != YNN_INVALID_VALUE_ID) {
    uint32_t dot_output_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(define_dot(*subgraph, num_k_dims, input_a_id,
                                   input_b_id, input_c_id, &dot_output_id,
                                   flags));
    YNN_RETURN_IF_ERROR(ynn_define_binary(
        subgraph, ynn_binary_add, dot_output_id, add_c_id, output_id, 0));
  } else {
    YNN_RETURN_IF_ERROR(define_dot(*subgraph, num_k_dims, input_a_id,
                                   input_b_id, input_c_id, output_id, flags));
  }

  if (convert_to_id != YNN_INVALID_VALUE_ID) {
    // We decided above to compute the output into an intermediate tensor, and
    // convert it to the output here.
    YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_convert,
                                         *output_id, &convert_to_id, flags));
    *output_id = convert_to_id;
  }

  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
