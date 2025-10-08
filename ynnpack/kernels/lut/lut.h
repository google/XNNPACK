// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_LUT_LUT_H_
#define XNNPACK_YNNPACK_KERNELS_LUT_LUT_H_

#include <cstddef>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

typedef void (*lut_kernel_fn)(size_t n, const void* a, const void* lut,
                              void* x);

#define YNN_LUT_KERNEL(arch, name, type_a, type_x) \
  void name(size_t n, const void* a, const void* lut, void* x);
#include "ynnpack/kernels/lut/kernels.inc"
#undef YNN_LUT_KERNEL

lut_kernel_fn get_lut_kernel(ynn_type type_a, ynn_type type_x);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_LUT_LUT_H_
