// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <assert.h>

inline static size_t min(size_t a, size_t b) {
  return a < b ? a : b;
}

inline static size_t max(size_t a, size_t b) {
  return a > b ? a : b;
}

inline static size_t doz(size_t a, size_t b) {
  return a >= b ? a - b : 0;
}

inline static size_t divide_round_up(size_t n, size_t q) {
  return n % q == 0 ? n / q : n / q + 1;
}

inline static size_t round_up(size_t n, size_t q) {
  return divide_round_up(n, q) * q;
}

inline static size_t round_down_po2(size_t n, size_t q) {
  assert(q != 0);
  assert((q & (q - 1)) == 0);
  return n & -q;
}

inline static size_t round_up_po2(size_t n, size_t q) {
  return round_down_po2(n + q - 1, q);
}

inline static size_t subtract_modulo(size_t a, size_t b, size_t m) {
  assert(a < m);
  assert(b < m);
  return a >= b ? a - b : a - b + m;
}

inline static float math_min_f32(float a, float b) {
  #if defined(__wasm__)
    return __builtin_wasm_min_f32(a, b);
  #else
    return a < b ? a : b;
  #endif
}

inline static float math_max_f32(float a, float b) {
  #if defined(__wasm__)
    return __builtin_wasm_max_f32(a, b);
  #else
    return a > b ? a : b;
  #endif
}
