// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/math.h>
#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__psimd(
    size_t n,
    const float* x,
    float* y)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  psimd_f32 vmax0 = psimd_load_splat_f32(x);
  psimd_f32 vmax1 = vmax0;
  psimd_f32 vmax2 = vmax0;
  psimd_f32 vmax3 = vmax0;
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const psimd_f32 vx0 = psimd_load_f32(x);
    const psimd_f32 vx1 = psimd_load_f32(x + 4);
    const psimd_f32 vx2 = psimd_load_f32(x + 8);
    const psimd_f32 vx3 = psimd_load_f32(x + 12);
    x += 16;

    vmax0 = psimd_max_f32(vmax0, vx0);
    vmax1 = psimd_max_f32(vmax1, vx1);
    vmax2 = psimd_max_f32(vmax2, vx2);
    vmax3 = psimd_max_f32(vmax3, vx3);
  }
  psimd_f32 vmax0123 = psimd_max_f32(psimd_max_f32(vmax0, vmax1), psimd_max_f32(vmax2, vmax3));
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const psimd_f32 vx = psimd_load_f32(x);
    vmax0123 = psimd_max_f32(vmax0123, vx);
    x += 4;
  }
  float vmax = psimd_reduce_max_f32(vmax0123);
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vx = *x++;
      vmax = math_max_f32(vx, vmax);
      n -= 4;
    } while (n != 0);
  }
  *y = vmax;
}
