// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_INTERLEAVE_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_INTERLEAVE_H_

#include <cstddef>
#include <cstdint>  // IWYU pragma: keep

namespace ynn {

// `m`, `n` are the logical shape (number of elements) of the input, but strides
// are a physical shape (number of bytes). `factor` is how many rows are to be
// interleaved into one row of the output. Rows in [0, factor) are zero padded.
typedef void (*interleave_kernel_fn)(size_t factor, size_t m, size_t n,
                                     size_t stride_a, const void* a, void* x);

#define YNN_INTERLEAVE_KERNEL(arch, name, M, type)                             \
  void name(size_t factor, size_t m, size_t n, size_t stride_a, const void* a, \
            void* x);
#include "ynnpack/kernels/transpose/interleave.inc"
#undef YNN_INTERLEAVE_KERNEL

interleave_kernel_fn get_interleave_kernel(size_t element_size_bits, size_t m);

// A fused alternative to `interleave_kernel_fn`. Where `interleave_kernel_fn`
// produces a single output row from `factor` input rows, this processes an
// entire `m` x `n` region in one call, producing `ceil(m / factor)` output rows
// `stride_x` bytes apart.
//
// Each output row holds `tile_n` groups of `factor` elements. Columns in
// `[n, tile_n)` and rows past `m` are zero filled, so the caller does not need
// to memset the padding.
//
// The point of this contract is to keep the whole block inside one inlined loop
// nest: the per-row contract costs an indirect call (and a memset) for every
// `factor` rows, which dominates when `n` is small.
typedef void (*interleave_block_kernel_fn)(size_t factor, size_t m, size_t n,
                                           size_t tile_n, size_t stride_a,
                                           const void* a, size_t stride_x,
                                           void* x);

#ifdef YNN_ARCH_X86_AVX512
void interleave2_x16_block_avx512(size_t factor, size_t m, size_t n,
                                  size_t tile_n, size_t stride_a, const void* a,
                                  size_t stride_x, void* x);
#endif  // YNN_ARCH_X86_AVX512

// Returns a fused block interleave kernel, or nullptr if there is none for this
// combination. Callers must fall back to `get_interleave_kernel` on nullptr.
interleave_block_kernel_fn get_interleave_block_kernel(size_t element_size_bits,
                                                       size_t m);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_INTERLEAVE_H_
