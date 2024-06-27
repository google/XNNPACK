// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_x3_ukernel__wasmsimd(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % sizeof(uint32_t) == 0);

  const float* x = (const float*) input;
  const float* y = (const float*) ((uintptr_t) x + n);
  const float* z = (const float*) ((uintptr_t) y + n);
  float* o = (float*) output;

  while (n >= 4 * sizeof(uint32_t)) {
    // vx = ( x3, x2, x1, x0 )
    const v128_t vx = wasm_v128_load(x);
    x += 4;
    // vy = ( y3, y2, y1, y0 )
    const v128_t vy = wasm_v128_load(y);
    y += 4;
    // vz = ( z3, z2, z1, z0 )
    const v128_t vz = wasm_v128_load(z);
    z += 4;

    // vxy = ( y2, y0, x2, x0 )
    const v128_t vxy = wasm_v32x4_shuffle(vx, vy, 0, 2, 4, 6);
    // vyz = ( z3, z1, y3, y1 )
    const v128_t vyz = wasm_v32x4_shuffle(vy, vz, 1, 3, 5, 7);
    // vzx = ( x3, x1, z2, z0 )
    const v128_t vzx = wasm_v32x4_shuffle(vz, vx, 0, 2, 5, 7);

    // vxyz0 = ( x1, z0, y0, x0 )
    const v128_t vxyz0 = wasm_v32x4_shuffle(vxy, vzx, 0, 2, 4, 6);
    // vxyz1 = ( y2, x2, z1, y1 )
    const v128_t vxyz1 = wasm_v32x4_shuffle(vyz, vxy, 0, 2, 5, 7);
    // vxyz2 = ( z3, y3, x3, z2 )
    const v128_t vxyz2 = wasm_v32x4_shuffle(vzx, vyz, 1, 3, 5, 7);

    wasm_v128_store(o, vxyz0);
    wasm_v128_store(o + 4, vxyz1);
    wasm_v128_store(o + 8, vxyz2);
    o += 12;
    n -= 4 * sizeof(uint32_t);
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vx = *x++;
      const float vy = *y++;
      const float vz = *z++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o += 3;
      n -= sizeof(uint32_t);
    } while (n != 0);
  }
}
