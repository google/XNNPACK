// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/hswish.h>


void xnn_f32_hswish_ukernel__psimd(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const psimd_f32 vsixth = psimd_load_splat_f32(&params->scalar.sixth);
  const psimd_f32 vhalf = psimd_load_splat_f32(&params->scalar.half);
  const psimd_f32 vone = psimd_load_splat_f32(&params->scalar.one);
  const psimd_f32 vzero = psimd_splat_f32(0.0f);

  for (; n >= 16; n -= 16) {
    const psimd_f32 vx = psimd_load_f32(x);
    x += 4;

    const psimd_f32 vt = psimd_min_f32(psimd_max_f32(psimd_add_f32(psimd_mul_f32(vx, vsixth), vhalf), vzero), vone);
    const psimd_f32 vy = psimd_mul_f32(vt, vx);

    psimd_store_f32(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const psimd_f32 vx = psimd_load_f32(x);
    x += 4;

    const psimd_f32 vt = psimd_min_f32(psimd_max_f32(psimd_add_f32(psimd_mul_f32(vx, vsixth), vhalf), vzero), vone);
    psimd_f32 vy = psimd_mul_f32(vt, vx);

    if (n & 8) {
      psimd_store2_f32(y, vy);
      vy = psimd_concat_hi_f32(vy, vy);
      y += 2;
    }
    if (n & 4) {
      psimd_store1_f32(y, vy);
    }
  }
}
