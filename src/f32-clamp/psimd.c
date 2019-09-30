// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/clamp.h>


void xnn_f32_clamp_ukernel__psimd(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const psimd_f32 voutput_max = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 voutput_min = psimd_load_splat_f32(&params->scalar.min);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const psimd_f32 vx = psimd_load_f32(x);
    x += 4;

    const psimd_f32 vy = psimd_min_f32(psimd_max_f32(vx, voutput_min), voutput_max);

    psimd_store_f32(y, vy);
    y += 4;
  }
  if (n != 0) {
    const psimd_f32 vx = psimd_load_f32(x);

    psimd_f32 vy = psimd_min_f32(psimd_max_f32(vx, voutput_min), voutput_max);

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
