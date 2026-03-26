// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_WASM_SIMD128_H_
#define XNNPACK_YNNPACK_BASE_SIMD_WASM_SIMD128_H_

#include <wasm_simd128.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

namespace simd {

// See vec.h for architecture independent comments.

template <>
struct vec<float, 4> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(float x) : v(wasm_f32x4_splat(x)) {}  // NOLINT

  v128_t v;
};

using f32x4 = vec<float, 4>;

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, decltype(f32x4::N), f32x4 = {}) {
  return f32x4{wasm_v128_load(ptr)};
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, decltype(f32x4::N) = {}) {
  wasm_v128_store(ptr, b.v);
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_WASM_SIMD128_H_
