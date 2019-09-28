// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>

#include <cpuinfo.h>
#include <fp16.h>

uint32_t xnn_stub_wasm_f32_sub(uint32_t a, uint32_t b) {
  return fp32_to_bits(fp32_from_bits(a) - fp32_from_bits(b));
}

#if CPUINFO_ARCH_WASM || CPUINFO_ARCH_WASMSIMD
uint32_t xnn_stub_wasm_f32_min(uint32_t a, uint32_t b) {
  return fp32_to_bits(__builtin_wasm_min_f32(fp32_from_bits(a), fp32_from_bits(b)));
}
#endif /* CPUINFO_ARCH_WASM || CPUINFO_ARCH_WASMSIMD */
