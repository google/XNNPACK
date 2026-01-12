// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/lut/lut.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

void lut_impl(size_t n, const uint8_t* a, const uint8_t* lut, uint8_t* x) {
  for (size_t j = 0; j < n; ++j) {
    x[j] = lut[a[j]];
  }
}

void lut_impl(size_t n, const int8_t* a, const int8_t* lut, int8_t* x) {
  for (size_t j = 0; j < n; ++j) {
    x[j] = lut[static_cast<uint8_t>(static_cast<int>(a[j]) + 128)];
  }
}

}  // namespace

void lut_u8(size_t n, const void* a, const void* lut, void* x) {
  lut_impl(n, reinterpret_cast<const uint8_t*>(a),
           reinterpret_cast<const uint8_t*>(lut),
           reinterpret_cast<uint8_t*>(x));
}

void lut_s8(size_t n, const void* a, const void* lut, void* x) {
  lut_impl(n, reinterpret_cast<const int8_t*>(a),
           reinterpret_cast<const int8_t*>(lut), reinterpret_cast<int8_t*>(x));
}

lut_kernel_fn get_lut_kernel(ynn_type type_a, ynn_type type_x) {
  if (type_a == ynn_type_uint8 && type_x == ynn_type_uint8) {
    return lut_u8;
  }
  if (type_a == ynn_type_int8 && type_x == ynn_type_int8) {
    return lut_s8;
  }
  return nullptr;
}

}  // namespace ynn
