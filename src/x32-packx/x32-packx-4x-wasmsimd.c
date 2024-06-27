// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/packx.h"


void xnn_x32_packx_ukernel_4x__wasmsimd(
    size_t m,
    size_t k,
    const uint32_t* restrict x_ptr,
    size_t x_stride,
    uint32_t* restrict y_ptr)
{
  assert(m != 0);
  assert(k != 0);

  const float* x0 = (const float*) x_ptr;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (m < 2) {
    x1 = x0;
  }
  const float* x2 = (const float*) ((uintptr_t) x1 + x_stride);
  if (m <= 2) {
    x2 = x1;
  }
  const float* x3 = (const float*) ((uintptr_t) x2 + x_stride);
  if (m != 4) {
    x3 = x2;
  }
  float* y = (float*) y_ptr;

  for (; k >= 4; k -= 4) {
    const v128_t vx0 = wasm_v128_load(x0);
    x0 += 4;
    const v128_t vx1 = wasm_v128_load(x1);
    x1 += 4;
    const v128_t vx2 = wasm_v128_load(x2);
    x2 += 4;
    const v128_t vx3 = wasm_v128_load(x3);
    x3 += 4;

    const v128_t vt0 = wasm_v32x4_shuffle(vx0, vx1, 0, 4, 1, 5);
    const v128_t vt1 = wasm_v32x4_shuffle(vx0, vx1, 2, 6, 3, 7);
    const v128_t vt2 = wasm_v32x4_shuffle(vx2, vx3, 0, 4, 1, 5);
    const v128_t vt3 = wasm_v32x4_shuffle(vx2, vx3, 2, 6, 3, 7);

    const v128_t vy0 = wasm_v32x4_shuffle(vt0, vt2, 0, 1, 4, 5);
    wasm_v128_store(y, vy0);

    const v128_t vy1 = wasm_v32x4_shuffle(vt0, vt2, 2, 3, 6, 7);
    wasm_v128_store(y + 4, vy1);

    const v128_t vy2 = wasm_v32x4_shuffle(vt1, vt3, 0, 1, 4, 5);
    wasm_v128_store(y + 8, vy2);

    const v128_t vy3 = wasm_v32x4_shuffle(vt1, vt3, 2, 3, 6, 7);
    wasm_v128_store(y + 12, vy3);

    y += 16;
  }
  if XNN_UNLIKELY(k != 0) {
    do {
      const float vx0 = *x0++;
      const float vx1 = *x1++;
      const float vx2 = *x2++;
      const float vx3 = *x3++;
      y[0] = vx0;
      y[1] = vx1;
      y[2] = vx2;
      y[3] = vx3;
      y += 4;
    } while (--k != 0);
  }
}
