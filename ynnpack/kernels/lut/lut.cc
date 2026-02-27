// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/lut/lut.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

template <typename A, typename X>
void lut_impl(size_t n, const A* a, const X* lut, X* x) {
  for (size_t j = 0; j < n; ++j) {
    size_t index = static_cast<int>(a[j]) - type_info<A>::min();
    x[j] = lut[index];
  }
}

}  // namespace

void lut_u8(size_t n, const void* a, const void* lut, void* x) {
  lut_impl(n, reinterpret_cast<const uint8_t*>(a),
           reinterpret_cast<const uint8_t*>(lut),
           reinterpret_cast<uint8_t*>(x));
}

lut_kernel_fn get_lut_kernel(ynn_type type_a, ynn_type type_x) {
  if (type_size_bits(type_a) == 8 && type_size_bits(type_x) == 8) {
    return lut_u8;
  }
  return nullptr;
}

}  // namespace ynn
