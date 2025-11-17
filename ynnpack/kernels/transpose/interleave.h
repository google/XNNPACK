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

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_INTERLEAVE_H_
