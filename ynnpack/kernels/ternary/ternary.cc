// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/ternary/ternary.h"

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/type.h"  // IWYU pragma: keep
#include "ynnpack/include/ynnpack.h"

namespace ynn {

ternary_kernel_fn get_ternary_kernel(ternary_op op, ynn_type type_a,
                                     ynn_type type_b, ynn_type type_c,
                                     ynn_type type_x) {
#define YNN_ELEMENTWISE_KERNEL(arch, name, kernel_op, init_params_fn, A, B, C, \
                               X)                                              \
  if (ternary_op::kernel_op == op && is_arch_supported(arch)) {                \
    if (type_of<A>() == type_a && type_of<B>() == type_b &&                    \
        type_of<C>() == type_c && type_of<X>() == type_x) {                    \
      return name;                                                             \
    }                                                                          \
  }
#include "ynnpack/kernels/ternary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL
  return nullptr;
}

}  // namespace ynn
