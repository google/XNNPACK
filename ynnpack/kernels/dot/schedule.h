// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DOT_SCHEDULE_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_SCHEDULE_H_

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/span.h"
#include "ynnpack/kernels/dot/pack.h"

namespace ynn {

struct dot_kernel_state;

struct dot_loop {
  // Which dimension this loop should iterate over, or operand panel to pack.
  // `pack_b` is not yet implemented.
  enum { m = 0, n = 1, k1 = 2, k2 = 3, k3 = 4, pack_a = 5, pack_b = 6, k = k1 };

  int dim = -1;

  // The step size of this loop, in elements (unused for `pack_a`).
  size_t step = 0;

  bool operator==(const dot_loop& other) const {
    return dim == other.dim && step == other.step;
  }
  bool operator!=(const dot_loop& other) const { return !(*this == other); }
};

// The maximum number of bytes per row of A or column of B covered by one step
// of the k loops in `schedule_dot`. 2 KiB keeps the panel of B streamed by one
// kernel call (e.g. 64 columns x 2 KiB = 128 KiB) resident in L2 across the m
// loop, and keeps a 32-row packed panel of A within 64 KiB for stack
// allocation.
inline constexpr size_t dot_k_step_bytes = 2048;
inline constexpr size_t dot_max_panel_bytes = 64 * 1024;

bool should_pack_a(span<const size_t> cache_budgets, size_t m, size_t n,
                   span<const size_t> ks, size_t block_m, size_t block_n,
                   size_t kernel_block_n, size_t tile_m, size_t tile_k,
                   size_t a_elem_size, size_t b_elem_size, size_t a_stride_m,
                   span<const size_t> a_k_strides);

// Generate a set of loops we should use when running a dot, attempting to
// optimize the order and size of loop steps such that memory locality is
// maximized for each cache level. `cache_budgets` are the number of bytes of
// each (per-core) cache level that the schedule may fill with data it reuses,
// innermost first (0 means the level should not be considered). When `pack_a`
// is true, emits a `dot_loop::pack_a` step so each `block_m` panel of A is
// packed once and reused across `n`. `storage` must have room for one loop per
// cache level, plus 6 (up to 3 k loops, `pack_a`, and the two innermost block
// loops).
span<dot_loop> schedule_dot(span<const size_t> cache_budgets, size_t m,
                            size_t n, span<const size_t> ks, size_t block_m,
                            size_t block_n, size_t a_elem_size,
                            size_t b_elem_size, dot_loop* storage,
                            bool pack_a = false);

// Return the buffer size in bytes needed to hold one packed panel at `pack_dim`
// (e.g. `dot_loop::pack_a`) when executing `loops`, or 0 if `pack_dim` is not
// present in `loops`.
inline size_t packed_dot_panel_bytes(span<dot_loop> loops, int pack_dim,
                                     size_t m, size_t n, span<const size_t> ks,
                                     size_t elem_size_bits, size_t tile_k,
                                     size_t tile_p) {
  size_t panel_k[3] = {
      !ks.empty() ? ks[0] : 1,
      ks.size() >= 2 ? ks[1] : 1,
      ks.size() >= 3 ? ks[2] : 1,
  };
  for (const dot_loop& loop : loops) {
    if (loop.dim == pack_dim) {
      // Only `pack_a` is supported.
      assert(pack_dim == dot_loop::pack_a);
      const size_t p_extent = (pack_dim == dot_loop::pack_a) ? m : n;
      return dot_packer::packed_bytes(p_extent, {panel_k, ks.size()},
                                      elem_size_bits, tile_k, tile_p);
    }
    switch (loop.dim) {
      case dot_loop::m:
        m = std::min(m, loop.step);
        break;
      case dot_loop::n:
        n = std::min(n, loop.step);
        break;
      case dot_loop::k1:
      case dot_loop::k2:
      case dot_loop::k3:
        panel_k[loop.dim - dot_loop::k1] =
            std::min(panel_k[loop.dim - dot_loop::k1], loop.step);
        break;
    }
  }
  return 0;
}

// Block a dot's m dimension, calling f at each block.
template <typename DotFn>
void block_dot_m(ptrdiff_t m, size_t n, span<const size_t> ks,
                 ptrdiff_t block_m, size_t a_stride_m,
                 span<const size_t> a_k_strides, const void* a,
                 span<const size_t> b_k_strides, const void* b,
                 size_t init_c_stride_m, const void* init_c, size_t c_stride_m,
                 size_t c_stride_n, void* c, const DotFn& f,
                 dot_kernel_state* state = nullptr) {
  // m is often small (one block), we can optimize this case a bit.
  while (m > block_m) {
    f(block_m, n, ks, a, a_stride_m, a_k_strides, b, b_k_strides,
      init_c_stride_m, init_c, c, state);
    if (init_c) init_c = offset_bytes(init_c, init_c_stride_m * block_m);
    c = offset_bytes(c, c_stride_m * block_m);
    a = offset_bytes(a, a_stride_m * block_m);
    m -= block_m;
  }
  f(m, n, ks, a, a_stride_m, a_k_strides, b, b_k_strides,
    init_c_stride_m, init_c, c, state);
}

// Block a dot's n dimension, calling f at each block.
template <typename DotFn>
void block_dot_n(size_t m, ptrdiff_t n, span<const size_t> ks,
                 ptrdiff_t block_n, size_t a_stride_m,
                 span<const size_t> a_k_strides, const void* a,
                 size_t b_stride_n, span<const size_t> b_k_strides,
                 const void* b, size_t init_c_stride_m, const void* init_c,
                 size_t c_stride_m, size_t c_stride_n, void* c, const DotFn& f,
                 dot_kernel_state* state = nullptr) {
  do {
    f(m, std::min(n, block_n), ks, a, a_stride_m, a_k_strides, b, b_k_strides,
      init_c_stride_m, init_c, c, state);

    n -= block_n;
    if (init_c) init_c = offset_bytes(init_c, c_stride_n * block_n);
    c = offset_bytes(c, c_stride_n * block_n);
    b = offset_bytes(b, b_stride_n * block_n);
  } while (n > 0);
}

// Block a dot's `k_dim` reduction dimension, calling f at each block.
template <typename DotFn>
void block_dot_k(size_t k_dim, size_t m, size_t n, span<const size_t> ks,
                 ptrdiff_t block_k, size_t a_stride_m,
                 span<const size_t> a_k_strides, const void* a,
                 span<const size_t> b_k_strides, const void* b,
                 size_t init_c_stride_m, const void* init_c, size_t c_stride_m,
                 size_t c_stride_n, void* c, const DotFn& f,
                 dot_kernel_state* state = nullptr) {
  assert(k_dim < ks.size());
  assert(ks.size() <= 3);
  ptrdiff_t k = ks[k_dim];
  size_t k_block[3];
  std::copy(ks.begin(), ks.end(), k_block);
  do {
    k_block[k_dim] = std::min(k, block_k);
    f(m, n, {k_block, ks.size()}, a, a_stride_m, a_k_strides, b, b_k_strides,
      init_c_stride_m, init_c, c, state);

    // Splitting k requires care for the initializer. The dot kernels read and
    // write from a separate buffer, so for each tile that we process in k, the
    // logic must look something like:
    //
    // prev_c = init_c
    // for k:
    //   call_kernel(..., prev_c, out_c)
    //   prev_c = out_c
    //
    // This way, the first call to the kernel initializes the accumulators, and
    // subsequent calls accumulate more dot operations into the accumulator.
    init_c = c;
    init_c_stride_m = c_stride_m;

    k -= block_k;
    b = offset_bytes(b, b_k_strides[k_dim] * block_k);
    a = offset_bytes(a, a_k_strides[k_dim] * block_k);
  } while (k > 0);
}

namespace internal {

template <typename NextFn>
void run_dot_step(dot_loop loop, size_t m, size_t n, span<const size_t> ks,
                  size_t a_stride_m, span<const size_t> a_k_strides,
                  const void* a, span<const size_t> b_k_strides,
                  size_t b_stride_n, const void* b, size_t init_c_stride_m,
                  const void* init_c, size_t c_stride_m, size_t c_stride_n,
                  void* c, const NextFn& next_f, dot_kernel_state* state,
                  dot_packer* pack_a) {
  switch (loop.dim) {
    case dot_loop::m:
      return block_dot_m(m, n, ks, loop.step, a_stride_m, a_k_strides, a,
                         b_k_strides, b, init_c_stride_m, init_c, c_stride_m,
                         c_stride_n, c, next_f, state);
    case dot_loop::n:
      return block_dot_n(m, n, ks, loop.step, a_stride_m, a_k_strides, a,
                         b_stride_n, b_k_strides, b, init_c_stride_m, init_c,
                         c_stride_m, c_stride_n, c, next_f, state);
    case dot_loop::k1:
    case dot_loop::k2:
    case dot_loop::k3:
      return block_dot_k(loop.dim - dot_loop::k1, m, n, ks, loop.step,
                         a_stride_m, a_k_strides, a, b_k_strides, b,
                         init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
                         next_f, state);
    case dot_loop::pack_a: {
      assert(pack_a);
      assert(!ks.empty());
      assert(ks.size() <= 3);
      size_t packed_a_k_strides[3];
      a = pack_a->pack(m, ks, a, a_k_strides, a_stride_m, packed_a_k_strides);
      size_t packed_ks[3];
      std::copy(ks.begin(), ks.end(), packed_ks);
      packed_ks[0] = align_up(ks[0], pack_a->tile_k);
      return next_f(m, n, span<const size_t>{packed_ks, ks.size()}, a,
                    a_stride_m,
                    span<const size_t>{packed_a_k_strides, ks.size()}, b,
                    b_k_strides, init_c_stride_m, init_c, c, state);
    }
    case dot_loop::pack_b: {
      assert(false);
      break;
    }
  }
}

}  // namespace internal

template <typename DotFn>
void run_dot(span<dot_loop> loops, size_t m, size_t n, span<const size_t> ks,
             size_t a_stride_m, span<const size_t> a_k_strides, const void* a,
             span<const size_t> b_k_strides, size_t b_stride_n, const void* b,
             size_t init_c_stride_m, const void* init_c, size_t c_stride_m,
             size_t c_stride_n, void* c, const DotFn& f,
             dot_kernel_state* state = nullptr, dot_packer* pack_a = nullptr) {
  assert(!loops.empty());
  const dot_loop loop = loops.front();
  loops = loops.subspan(1);

  if (loops.empty()) {
    // There are no more loops after this one.
    internal::run_dot_step(loop, m, n, ks, a_stride_m, a_k_strides, a,
                           b_k_strides, b_stride_n, b, init_c_stride_m, init_c,
                           c_stride_m, c_stride_n, c, f, state, pack_a);
  } else {
    auto recursive_f =
        [&](size_t m, size_t n, span<const size_t> ks, const void* a,
            size_t a_stride_m, span<const size_t> a_k_strides, const void* b,
            span<const size_t> b_k_strides, size_t init_c_stride_m,
            const void* init_c, void* c, dot_kernel_state* state = nullptr) {
          run_dot(loops, m, n, ks, a_stride_m, a_k_strides, a, b_k_strides,
                  b_stride_n, b, init_c_stride_m, init_c, c_stride_m,
                  c_stride_n, c, f, state, pack_a);
        };
    internal::run_dot_step(loop, m, n, ks, a_stride_m, a_k_strides, a,
                           b_k_strides, b_stride_n, b, init_c_stride_m, init_c,
                           c_stride_m, c_stride_n, c, recursive_f, state,
                           pack_a);
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_SCHEDULE_H_
