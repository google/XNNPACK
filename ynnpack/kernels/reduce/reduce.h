// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_REDUCE_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_REDUCE_H_

#include <cstddef>
#include <cstdint>  // IWYU pragma: keep

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Reduce kernels compute the following:
//
//    C(i, j) = f(A(j, k3, k2, k1), B(k3, k2, k1), i)
//
// for all i, j, k3, k2, k1
//
// For most reductions, l will be 0, and B is ignored.
typedef void (*unary_reduce_kernel_fn)(size_t n, size_t k3, size_t k2,
                                       size_t k1, size_t a_stride_n,
                                       size_t a_stride_k3, size_t a_stride_k2,
                                       const void* a, size_t c_stride_m,
                                       void* c);

typedef void (*binary_reduce_kernel_fn)(size_t n, size_t k3, size_t k2,
                                        size_t k1, size_t a_stride_n,
                                        size_t a_stride_k3, size_t a_stride_k2,
                                        const void* a, size_t b_stride_n,
                                        size_t b_stride_k3, size_t b_stride_k2,
                                        size_t b_stride_k1, const void* b,
                                        size_t c_stride_m, void* c);

#define YNN_UNARY_REDUCE_KERNEL(arch, name, type_a, type_c)               \
  void name(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n, \
            size_t a_stride_k3, size_t a_stride_k2, const void* a,        \
            size_t c_stride_m, void* c);
#include "ynnpack/kernels/reduce/max.inc"
#include "ynnpack/kernels/reduce/min.inc"
#include "ynnpack/kernels/reduce/min_max.inc"
#include "ynnpack/kernels/reduce/sum.inc"
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_UNARY_REDUCE_KERNEL

constexpr size_t unknown_reduce_extent = static_cast<size_t>(-1);

unary_reduce_kernel_fn get_sum_kernel(ynn_type a_type, ynn_type c_type,
                                      size_t n = unknown_reduce_extent,
                                      size_t k3 = unknown_reduce_extent,
                                      size_t k2 = unknown_reduce_extent,
                                      size_t k1 = unknown_reduce_extent);

unary_reduce_kernel_fn get_sum_squared_kernel(
    ynn_type a_type, ynn_type c_type, size_t n = unknown_reduce_extent,
    size_t k3 = unknown_reduce_extent, size_t k2 = unknown_reduce_extent,
    size_t k1 = unknown_reduce_extent);

unary_reduce_kernel_fn get_min_kernel(ynn_type type,
                                      size_t n = unknown_reduce_extent,
                                      size_t k3 = unknown_reduce_extent,
                                      size_t k2 = unknown_reduce_extent,
                                      size_t k1 = unknown_reduce_extent);

unary_reduce_kernel_fn get_max_kernel(ynn_type type,
                                      size_t n = unknown_reduce_extent,
                                      size_t k3 = unknown_reduce_extent,
                                      size_t k2 = unknown_reduce_extent,
                                      size_t k1 = unknown_reduce_extent);

unary_reduce_kernel_fn get_min_max_kernel(ynn_type type,
                                          size_t n = unknown_reduce_extent,
                                          size_t k3 = unknown_reduce_extent,
                                          size_t k2 = unknown_reduce_extent,
                                          size_t k1 = unknown_reduce_extent);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_REDUCE_H_
