// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_REDUCE_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_REDUCE_H_

#include <cstddef>
#include <cstdint>  // IWYU pragma: keep
#include <type_traits>

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Reduction kernels process rank-2 buffers and produce rank-1 buffers.
// The k dimension is reduced, the n dimension is a "batch" dimension, which is
// also the dimension of the rank-1 result.
typedef void (*reduce_kernel_fn)(size_t n, size_t k, size_t a_stride_n,
                                 const void* a, void* x0, void* x1);

// These constants are basically an enum, but use std::integral_constant to
// allow dispatching to overloaded functions if necessary.
namespace reduce_dim {

inline constexpr std::integral_constant<size_t, 0> k1 = {};
inline constexpr std::integral_constant<size_t, 1> kn = {};

}  // namespace reduce_dim

struct reduce_kernel {
  reduce_kernel_fn k1 = nullptr;
  reduce_kernel_fn kn = nullptr;
};

#define YNN_REDUCE_KERNEL(arch, name, k_dim, type_a, type_c)                \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a, void* x0, \
            void* x1);
#include "ynnpack/kernels/reduce/max.inc"
#include "ynnpack/kernels/reduce/min.inc"
#include "ynnpack/kernels/reduce/min_max.inc"
#include "ynnpack/kernels/reduce/sum.inc"
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_REDUCE_KERNEL

reduce_kernel get_sum_kernel(ynn_type a_type, ynn_type c_type);
reduce_kernel get_sum_squared_kernel(ynn_type a_type, ynn_type c_type);

reduce_kernel get_min_kernel(ynn_type type);
reduce_kernel get_max_kernel(ynn_type type);
reduce_kernel get_min_max_kernel(ynn_type type);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_REDUCE_H_
