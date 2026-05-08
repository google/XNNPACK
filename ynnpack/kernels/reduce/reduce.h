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

typedef void (*reduce_kernel_fn)(size_t n, size_t k, size_t a_stride_n,
                                 const void* a, size_t c_stride_m, void* c);

struct reduce_kernel {
  reduce_kernel_fn k1;
  reduce_kernel_fn kn;
};

#define YNN_REDUCE_K1_KERNEL(arch, name, type_a, type_c)          \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a, \
            size_t c_stride_m, void* c);
#define YNN_REDUCE_KN_KERNEL(arch, name, type_a, type_c)          \
  void name(size_t n, size_t k, size_t a_stride_k, const void* a, \
            size_t c_stride_m, void* c);
#include "ynnpack/kernels/reduce/max.inc"
#include "ynnpack/kernels/reduce/min.inc"
#include "ynnpack/kernels/reduce/min_max.inc"
#include "ynnpack/kernels/reduce/sum.inc"
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_REDUCE_K1_KERNEL
#undef YNN_REDUCE_KN_KERNEL

reduce_kernel get_sum_kernel(ynn_type a_type, ynn_type c_type);
reduce_kernel get_sum_squared_kernel(ynn_type a_type, ynn_type c_type);

reduce_kernel get_min_kernel(ynn_type type);
reduce_kernel get_max_kernel(ynn_type type);
reduce_kernel get_min_max_kernel(ynn_type type);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_REDUCE_H_
