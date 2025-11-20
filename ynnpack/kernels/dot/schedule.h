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
#include "slinky/base/span.h"

namespace ynn {

struct dot_loop {
  // Which dimension this loop should iterate over.
  enum { m = 0, n = 1, k = 2 } dim;

  // The step size of this loop, in blocks.
  size_t blocks;
};

// Generate a set of loops we should use when running a dot, attempting to
// optimize the order and size of loop steps such that memory locality is
// maximized for each cache in `cache_sizes`. `storage` must have room for at
// most 3 loops per cache size.
slinky::span<dot_loop> schedule_dot(slinky::span<const size_t> cache_sizes,
                                    size_t m, size_t n, size_t k1, size_t k2,
                                    size_t k3, size_t block_m, size_t block_n,
                                    size_t block_k, size_t a_elem_size,
                                    size_t b_elem_size, dot_loop* storage);

// Block a dot's m dimension, calling f at each block.
template <typename DotFn>
void block_dot_m(ptrdiff_t m, size_t n, size_t k, ptrdiff_t block_m,
                 size_t a_stride_m, const void* a, const void* b,
                 size_t init_c_stride_m, const void* init_c, size_t c_stride_m,
                 size_t c_stride_n, void* c, DotFn f) {
  do {
    f(std::min(m, block_m), n, k, a, b, init_c_stride_m, init_c, c);

    m -= block_m;
    if (init_c) init_c = offset_bytes(init_c, init_c_stride_m * block_m);
    c = offset_bytes(c, c_stride_m * block_m);
    a = offset_bytes(a, a_stride_m * block_m);
  } while (m > 0);
}

// Block a dot's n dimension, calling f at each block.
template <typename DotFn>
void block_dot_n(size_t m, ptrdiff_t n, size_t k, ptrdiff_t block_n,
                 const void* a, size_t b_stride_n, const void* b,
                 size_t init_c_stride_m, const void* init_c, size_t c_stride_m,
                 size_t c_stride_n, void* c, DotFn f) {
  do {
    f(m, std::min(n, block_n), k, a, b, init_c_stride_m, init_c, c);

    n -= block_n;
    if (init_c) init_c = offset_bytes(init_c, c_stride_n * block_n);
    c = offset_bytes(c, c_stride_n * block_n);
    b = offset_bytes(b, b_stride_n * block_n);
  } while (n > 0);
}

// Block a dot's k dimension, calling f at each block.
template <typename DotFn>
void block_dot_k(size_t m, size_t n, ptrdiff_t k, ptrdiff_t block_k,
                 size_t a_stride_k, const void* a, size_t b_stride_k,
                 const void* b, size_t init_c_stride_m, const void* init_c,
                 size_t c_stride_m, size_t c_stride_n, void* c, DotFn f) {
  do {
    f(m, n, std::min(k, block_k), a, b, init_c_stride_m, init_c, c);

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
    b = offset_bytes(b, b_stride_k * block_k);
    a = offset_bytes(a, a_stride_k * block_k);
  } while (k > 0);
}

template <typename DotFn>
void run_dot(slinky::span<dot_loop> loops, size_t m, size_t n, size_t k,
             size_t block_m, size_t block_n, size_t block_k, size_t a_stride_m,
             size_t a_stride_k, const void* a, size_t b_stride_k,
             size_t b_stride_n, const void* b, size_t init_c_stride_m,
             const void* init_c, size_t c_stride_m, size_t c_stride_n, void* c,
             DotFn f) {
  assert(!loops.empty());
  const dot_loop loop = loops.front();
  loops = loops.subspan(1);

  if (loops.empty()) {
    // There are no more loops after this one.
    switch (loop.dim) {
      case dot_loop::m:
        return block_dot_m(m, n, k, block_m * loop.blocks, a_stride_m, a, b,
                           init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
                           f);
      case dot_loop::n:
        return block_dot_n(m, n, k, block_n * loop.blocks, a, b_stride_n, b,
                           init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
                           f);
      case dot_loop::k:
        return block_dot_k(m, n, k, block_k * loop.blocks, a_stride_k, a,
                           b_stride_k, b, init_c_stride_m, init_c, c_stride_m,
                           c_stride_n, c, f);
    }
  } else {
    // Recursively call `run_dot` with the subsequent loops.
    auto recursive_f = [=](size_t m, size_t n, size_t k, const void* a,
                           const void* b, size_t init_c_stride_m,
                           const void* init_c, void* c) {
      run_dot(loops, m, n, k, block_m, block_n, block_k, a_stride_m, a_stride_k,
              a, b_stride_k, b_stride_n, b, init_c_stride_m, init_c, c_stride_m,
              c_stride_n, c, f);
    };
    switch (loop.dim) {
      case dot_loop::m:
        return block_dot_m(m, n, k, block_m * loop.blocks, a_stride_m, a, b,
                           init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
                           recursive_f);
      case dot_loop::n:
        return block_dot_n(m, n, k, block_n * loop.blocks, a, b_stride_n, b,
                           init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
                           recursive_f);
      case dot_loop::k:
        return block_dot_k(m, n, k, block_k * loop.blocks, a_stride_k, a,
                           b_stride_k, b, init_c_stride_m, init_c, c_stride_m,
                           c_stride_n, c, recursive_f);
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_SCHEDULE_H_
