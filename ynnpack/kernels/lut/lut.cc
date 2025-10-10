// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/lut/lut.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "ynnpack/base/base.h"

namespace ynn {

namespace {

template <typename A, typename X>
void lut_impl(size_t n, const A* a, const X* lut, X* x) {
  for (size_t j = 0; j < n; ++j) {
    x[j] = lut[a[j]];
  }
}

}  // namespace

void lut_x8(size_t n, const void* a, const void* lut, void* x) {
  lut_impl(n, reinterpret_cast<const uint8_t*>(a),
           reinterpret_cast<const uint8_t*>(lut),
           reinterpret_cast<uint8_t*>(x));
}

lut_kernel_fn get_lut_kernel(size_t elem_size_a, size_t elem_size_x) {
  if (elem_size_a == 1 && elem_size_x == 1) {
    return lut_x8;
  }
  YNN_UNREACHABLE;
  return nullptr;
}

}  // namespace ynn
