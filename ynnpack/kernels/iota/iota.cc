// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/iota/iota.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

template <typename T>
void iota_impl(size_t n, T offset, T stride, T* output) {
  if (stride == 0) {
    if (offset == 0) {
      memset(output, 0, n * sizeof(T));
    } else {
      std::fill_n(output, n, offset);
    }
  } else {
    assert(static_cast<size_t>(static_cast<int>(n)) == n);
    for (int i = 0; i < static_cast<int>(n); ++i) {
      output[i] = i * stride + offset;
    }
  }
}

}  // namespace

void iota_int32(size_t n, const void* begin, const void* stride,
                    void* output) {
  iota_impl<int32_t>(n, *static_cast<const int32_t*>(begin),
                     *static_cast<const int32_t*>(stride),
                     static_cast<int32_t*>(output));
}

void iota_fp32(size_t n, const void* begin, const void* stride,
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
