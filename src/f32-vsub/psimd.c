// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/common.h>
#include <xnnpack/vsub.h>


void xnn_f32_vsub_ukernel__psimd(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const psimd_f32 vy_min = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vy_max = psimd_load_splat_f32(&params->scalar.max);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const psimd_f32 va0 = psimd_load_f32(a);
    const psimd_f32 va1 = psimd_load_f32(a + 4);
    a += 8;

    const psimd_f32 vb0 = psimd_load_f32(b);
    const psimd_f32 vb1 = psimd_load_f32(b + 4);
    b += 8;

    const psimd_f32 vacc0 = psimd_sub_f32(va0, vb0);
    const psimd_f32 vacc1 = psimd_sub_f32(va1, vb1);
    const psimd_f32 vy0 = psimd_min_f32(psimd_max_f32(vacc0, vy_min), vy_max);
    const psimd_f32 vy1 = psimd_min_f32(psimd_max_f32(vacc1, vy_min), vy_max);

    psimd_store_f32(y, vy0);
    psimd_store_f32(y + 4, vy1);
    y += 8;
  }
  if (n >= 4 * sizeof(float)) {
    const psimd_f32 va = psimd_load_f32(a);
    a += 4;
    const psimd_f32 vb = psimd_load_f32(b);
    b += 4;
    const psimd_f32 vacc = psimd_sub_f32(va, vb);
    const psimd_f32 vy = psimd_min_f32(psimd_max_f32(vacc, vy_min), vy_max);
    psimd_store_f32(y, vy);
    y += 4;
    n -= 4 * sizeof(float);
  }
  if (n != 0) {
    const psimd_f32 va = psimd_load_f32(a);
    const psimd_f32 vb = psimd_load_f32(b);
    const psimd_f32 vacc = psimd_sub_f32(va, vb);
    psimd_f32 vy = psimd_min_f32(psimd_max_f32(vacc, vy_min), vy_max);
    if (n & 2 * sizeof(float)) {
      psimd_store2_f32(y, vy);
      vy = psimd_concat_hi_f32(vy, vy);
      y += 2;
    }
    if (n & 1 * sizeof(float)) {
      psimd_store1_f32(y, vy);
    }
  }
}
