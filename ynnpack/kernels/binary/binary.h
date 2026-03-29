// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_BINARY_H_
#define XNNPACK_YNNPACK_KERNELS_BINARY_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// The stride of dimension `n` for any operand must be 0 or the size of one
// element.
typedef void (*binary_kernel_fn)(size_t m, size_t n, size_t stride_a_m,
                                 size_t stride_a_n, const void* a,
                                 size_t stride_b_m, size_t stride_b_n,
                                 const void* b, size_t stride_x_m, void* x);

#define YNN_ELEMENTWISE_KERNEL(arch, name, op, type_a, type_b, type_c) \
  void name(size_t m, size_t n, size_t stride_a_m, size_t stride_a_n,  \
            const void* a, size_t stride_b_m, size_t stride_b_n,       \
            const void* b, size_t stride_x_m, void* x);
#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

binary_kernel_fn get_binary_reference_kernel(ynn_binary_operator op,
                                             ynn_type type);

binary_kernel_fn get_binary_kernel(
    ynn_binary_operator op, ynn_type type_a, ynn_type type_b, ynn_type type_x,
    uint64_t supported_arch_flags = get_supported_arch_flags());

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_BINARY_H_
