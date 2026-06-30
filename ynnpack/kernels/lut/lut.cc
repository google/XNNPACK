// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/lut/lut.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

template <typename Idx, typename Elem>
bool lut_impl(size_t n, const Idx* idx, size_t lut_size, const Elem* lut,
              Elem* out) {
  using IdxInfo = type_info<Idx>;
  if (IdxInfo::min() == 0 && lut_size > IdxInfo::max()) {
    // The lut is at least as big as the maximum value of the indices, we don't
    // need to check bounds.
    for (size_t j = 0; j < n; ++j) {
      out[j] = lut[IdxInfo::get(idx, j)];
    }
  } else {
    for (size_t j = 0; j < n; ++j) {
      auto idx_j = IdxInfo::get(idx, j);
      if (idx_j < 0 || idx_j >= lut_size) return false;
      out[j] = lut[idx_j];
    }
  }
  return true;
}

}  // namespace

bool lut_u2_u8(size_t n, const void* idx, size_t lut_size, const void* lut,
               void* out) {
  return lut_impl(n, reinterpret_cast<const uint2x4*>(idx), lut_size,
                  reinterpret_cast<const uint8_t*>(lut),
                  reinterpret_cast<uint8_t*>(out));
}

bool lut_u2_u16(size_t n, const void* idx, size_t lut_size, const void* lut,
                void* out) {
  return lut_impl(n, reinterpret_cast<const uint2x4*>(idx), lut_size,
                  reinterpret_cast<const uint16_t*>(lut),
                  reinterpret_cast<uint16_t*>(out));
}

bool lut_u4_u8(size_t n, const void* idx, size_t lut_size, const void* lut,
               void* out) {
  return lut_impl(n, reinterpret_cast<const uint4x2*>(idx), lut_size,
                  reinterpret_cast<const uint8_t*>(lut),
                  reinterpret_cast<uint8_t*>(out));
}

bool lut_u4_u16(size_t n, const void* idx, size_t lut_size, const void* lut,
                void* out) {
  return lut_impl(n, reinterpret_cast<const uint4x2*>(idx), lut_size,
                  reinterpret_cast<const uint16_t*>(lut),
                  reinterpret_cast<uint16_t*>(out));
}

bool lut_u8_u8(size_t n, const void* idx, size_t lut_size, const void* lut,
               void* out) {
  return lut_impl(n, reinterpret_cast<const uint8_t*>(idx), lut_size,
                  reinterpret_cast<const uint8_t*>(lut),
                  reinterpret_cast<uint8_t*>(out));
}

bool lut_u8_u16(size_t n, const void* idx, size_t lut_size, const void* lut,
                void* out) {
  return lut_impl(n, reinterpret_cast<const uint8_t*>(idx), lut_size,
                  reinterpret_cast<const uint16_t*>(lut),
                  reinterpret_cast<uint16_t*>(out));
}

bool lut_u8_u32(size_t n, const void* idx, size_t lut_size, const void* lut,
                void* out) {
  return lut_impl(n, reinterpret_cast<const uint8_t*>(idx), lut_size,
                  reinterpret_cast<const uint32_t*>(lut),
                  reinterpret_cast<uint32_t*>(out));
}

bool lut_s32_u16(size_t n, const void* idx, size_t lut_size, const void* lut,
                 void* out) {
  return lut_impl(n, reinterpret_cast<const int32_t*>(idx), lut_size,
                  reinterpret_cast<const uint16_t*>(lut),
                  reinterpret_cast<uint16_t*>(out));
}

bool lut_s32_u32(size_t n, const void* idx, size_t lut_size, const void* lut,
                 void* out) {
  return lut_impl(n, reinterpret_cast<const int32_t*>(idx), lut_size,
                  reinterpret_cast<const uint32_t*>(lut),
                  reinterpret_cast<uint32_t*>(out));
}

lut_kernel_fn get_lut_kernel(ynn_type idx_type, size_t elem_size_bits) {
#define YNN_LUT_KERNEL(arch, name, kernel_idx_type, kernel_elem_size_bits) \
  if (is_arch_supported(arch) && idx_type == type_of<kernel_idx_type>() && \
      elem_size_bits == kernel_elem_size_bits) {                           \
    return name;                                                           \
  }
#include "ynnpack/kernels/lut/kernels.inc"
#undef YNN_LUT_KERNEL
  return nullptr;
}

}  // namespace ynn
