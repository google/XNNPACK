// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_IOTA_H_
#define XNNPACK_YNNPACK_KERNELS_IOTA_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Writes output[i] = *begin + *stride * i
typedef void (*iota_kernel_fn)(size_t n, const void* begin, const void* stride,
                               void* output);

#define YNN_IOTA_KERNEL(arch, name, type) \
  void name(size_t n, const void* begin, const void* stride, void* output);
#include "ynnpack/kernels/iota/kernels.inc"
#undef YNN_IOTA_KERNEL

iota_kernel_fn get_iota_kernel(
    ynn_type type, uint64_t supported_arch_flags = get_supported_arch_flags());

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_IOTA_H_
