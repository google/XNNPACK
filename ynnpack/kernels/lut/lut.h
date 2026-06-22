// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_LUT_LUT_H_
#define XNNPACK_YNNPACK_KERNELS_LUT_LUT_H_

#include <cstddef>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Assigns x[i] = lut[a[i]]. Returns false if any of a[i] are out of bounds.
typedef bool (*lut_kernel_fn)(size_t n, const void* idx, size_t lut_size,
                              const void* lut, void* out);

#define YNN_LUT_KERNEL(arch, name, idx_type, elem_size_bits)             \
  bool name(size_t n, const void* idx, size_t lut_size, const void* lut, \
            void* out);
#include "ynnpack/kernels/lut/kernels.inc"
#undef YNN_LUT_KERNEL

lut_kernel_fn get_lut_kernel(ynn_type idx_type, size_t elem_size_bits);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_LUT_LUT_H_
