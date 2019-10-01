// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/zip.h>


void xnn_x32_zip_x3_ukernel__psimd(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = (const uint32_t*) input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  const uint32_t* z = (const uint32_t*) ((uintptr_t) y + n);
  uint32_t* o = (uint32_t*) output;

  while (n >= 16) {
    // vx = ( x3, x2, x1, x0 )
    const psimd_u32 vx = psimd_load_u32(x);
    x += 4;
    // vy = ( y3, y2, y1, y0 )
    const psimd_u32 vy = psimd_load_u32(y);
    y += 4;
    // vz = ( z3, z2, z1, z0 )
    const psimd_u32 vz = psimd_load_u32(z);
    z += 4;

    // vxy = ( y2, y0, x2, x0 )
    const psimd_u32 vxy = psimd_concat_even_u32(vx, vy);
    // vyz = ( z3, z1, y3, y1 )
    const psimd_u32 vyz = psimd_concat_odd_u32(vy, vz);
    // vzx = ( x3, x1, z2, z0 )
    #ifdef __clang__
    const psimd_u32 vzx = __builtin_shufflevector(vz, vx, 0, 2, 4+1, 4+3);
    #else
    const psimd_u32 vzx = __builtin_shuffle(vz, vx, (psimd_s32) { 0, 2, 4+1, 4+3 });
    #endif

    // vxyz0 = ( x1, z0, y0, x0 )
    const psimd_u32 vxyz0 = psimd_concat_even_u32(vxy, vzx);
    // vxyz1 = ( y2, x2, z1, y1 )
    #ifdef __clang__
    const psimd_u32 vxyz1 = __builtin_shufflevector(vyz, vxy, 0, 2, 4+1, 4+3);
    #else
    const psimd_u32 vxyz1 = __builtin_shuffle(vyz, vxy, (psimd_s32) { 0, 2, 4+1, 4+3 });
    #endif
    // vxyz2 = ( z3, y3, x3, z2 )
    const psimd_u32 vxyz2 = psimd_concat_odd_u32(vzx, vyz);

    psimd_store_u32(o, vxyz0);
    psimd_store_u32(o + 4, vxyz1);
    psimd_store_u32(o + 8, vxyz2);
    o += 12;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const uint32_t vx = *x++;
      const uint32_t vy = *y++;
      const uint32_t vz = *z++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o += 3;
      n -= 4;
    } while (n != 0);
  }
}
