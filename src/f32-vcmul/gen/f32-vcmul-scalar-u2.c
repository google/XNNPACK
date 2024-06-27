// Auto-generated file. Do not edit!
//   Template: src/f32-vcmul/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vcmul_ukernel__scalar_u2(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float* ar = input_a;
  const float* ai = (const float*) ((uintptr_t) input_a + batch);
  const float* br = input_b;
  const float* bi = (const float*) ((uintptr_t) input_b + batch);
  float* or = output;
  float* oi = (float*) ((uintptr_t) output + batch);
  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float va0r = ar[0];
    const float va1r = ar[1];
    ar += 2;

    const float va0i = ai[0];
    const float va1i = ai[1];
    ai += 2;

    const float vb0r = br[0];
    const float vb1r = br[1];
    br += 2;

    const float vb0i = bi[0];
    const float vb1i = bi[1];
    bi += 2;

    const float vacc0r = va0r * vb0r - va0i * vb0i;
    const float vacc1r = va1r * vb1r - va1i * vb1i;

    const float vacc0i = va0r * vb0i + va0i * vb0r;
    const float vacc1i = va1r * vb1i + va1i * vb1r;

    or[0] = vacc0r;
    or[1] = vacc1r;
    or += 2;

    oi[0] = vacc0i;
    oi[1] = vacc1i;
    oi += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch == sizeof(float));
    const float var = *ar;
    const float vai = *ai;
    const float vbr = *br;
    const float vbi = *bi;
    const float vaccr = var * vbr - vai * vbi;
    const float vacci = var * vbi + vai * vbr;
    *or = vaccr;
    *oi = vacci;
  }
}
