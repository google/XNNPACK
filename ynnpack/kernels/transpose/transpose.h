// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_H_

#include <array>
#include <cstddef>
#include <cstdint>  // IWYU pragma: keep
#include <functional>

#include "ynnpack/base/type.h"  // IWYU pragma: keep

namespace ynn {

// Pointer to a function that implements transposing data. The size of the
// elements to transpose is fixed by the function.
// - `a`, `x`: pointers to the input and output, respectively.
// - `m`, `n`: Logical shape of the output of the transpose.
// - `n_bytes_a`: Indicates the width of the input in bytes. Data required
// beyond this width will be zero padded in the result.
// - `stride_a`, `stride_x`: Bytes between rows of the input and output.
typedef void (*transpose_kernel_fn)(size_t m, size_t n, size_t n_bytes_a,
                                    size_t stride_a, const void* a,
                                    size_t stride_x, void* x);

#define YNN_TRANSPOSE_KERNEL(arch, name, elem_size_bits)           \
  void name(size_t m, size_t n, size_t n_bytes_a, size_t stride_a, \
            const void* a, size_t stride_x, void* x);
#include "ynnpack/kernels/transpose/transpose.inc"
#undef YNN_TRANSPOSE_KERNEL

transpose_kernel_fn get_transpose_kernel(size_t element_size_bits);

using transpose_fn =
    std::function<void(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x)>;

// Make a wrapper for a transpose kernel that runs the kernel in tiles.
transpose_fn make_tiled_transpose(size_t elem_size_bits,
                                  transpose_kernel_fn transpose_fn);
transpose_fn get_tiled_transpose(size_t elem_size_bits);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_H_
