// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/iota/iota.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/log.h"
#include "ynnpack/kernels/iota/generic.h"

namespace ynn {

namespace {

template <typename T>
void iota_impl(size_t n, T begin, T stride, T* output) {
  if (begin == 0 && stride == 0) {
    memset(output, 0, n * sizeof(T));
  } else if (stride == 0) {
    std::fill_n(output, n, begin);
  } else {
    // Factor the loop into tiles of `consistent_tile_n`, to be consistent with
    // other floating point implementations of this kernel.
    T tile[consistent_tile_n];
    for (size_t i = 0; i < consistent_tile_n; ++i) {
      tile[i] = begin + stride * i;
    }
    while (n >= consistent_tile_n) {
      memcpy(output, tile, consistent_tile_n * sizeof(T));
      for (size_t i = 0; i < consistent_tile_n; ++i) {
        tile[i] += stride * consistent_tile_n;
      }
      output += consistent_tile_n;
      n -= consistent_tile_n;
    }
    for (size_t i = 0; i < n; ++i) {
      output[i] = tile[i];
    }
  }
}

}  // namespace

void ynn_iota_int32(size_t n, const void* begin, const void* stride,
                    void* output) {
  iota_impl<int32_t>(n, *static_cast<const int32_t*>(begin),
                     *static_cast<const int32_t*>(stride),
                     static_cast<int32_t*>(output));
}

void ynn_iota_fp32(size_t n, const void* begin, const void* stride,
                      void* output) {
  iota_impl<float>(n, *static_cast<const float*>(begin),
                   *static_cast<const float*>(stride),
                   static_cast<float*>(output));
}

iota_kernel_fn get_iota_kernel(ynn_type type, uint64_t supported_arch_flags) {
#define YNN_IOTA_KERNEL(arch, name, type_id)           \
  if (type == type_of<type_id>() &&                    \
      is_arch_supported(arch, supported_arch_flags)) { \
    YNN_LOG_DEBUG() << "Using iota kernel " << #name;  \
    return &name;                                      \
  }

#include "ynnpack/kernels/iota/kernels.inc"
#undef YNN_IOTA_KERNEL

  return nullptr;
}

}  // namespace ynn
