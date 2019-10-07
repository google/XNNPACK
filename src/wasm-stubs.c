// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>

#include <fp16.h>

#include <xnnpack/common.h>

uint32_t xnn_stub_wasm_f32_sub(uint32_t a, uint32_t b) {
  return fp32_to_bits(fp32_from_bits(a) - fp32_from_bits(b));
}

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
uint32_t xnn_stub_wasm_f32_min(uint32_t a, uint32_t b) {
  return fp32_to_bits(__builtin_wasm_min_f32(fp32_from_bits(a), fp32_from_bits(b)));
}
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
