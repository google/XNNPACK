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


void xnn_f32_vcmul_ukernel__scalar_u8(
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
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0r = ar[0];
    const float va1r = ar[1];
    const float va2r = ar[2];
    const float va3r = ar[3];
    const float va4r = ar[4];
    const float va5r = ar[5];
    const float va6r = ar[6];
    const float va7r = ar[7];
    ar += 8;

    const float va0i = ai[0];
    const float va1i = ai[1];
    const float va2i = ai[2];
    const float va3i = ai[3];
    const float va4i = ai[4];
    const float va5i = ai[5];
    const float va6i = ai[6];
    const float va7i = ai[7];
    ai += 8;

    const float vb0r = br[0];
    const float vb1r = br[1];
    const float vb2r = br[2];
    const float vb3r = br[3];
    const float vb4r = br[4];
    const float vb5r = br[5];
    const float vb6r = br[6];
    const float vb7r = br[7];
    br += 8;

    const float vb0i = bi[0];
    const float vb1i = bi[1];
    const float vb2i = bi[2];
    const float vb3i = bi[3];
    const float vb4i = bi[4];
    const float vb5i = bi[5];
    const float vb6i = bi[6];
    const float vb7i = bi[7];
    bi += 8;

    const float vacc0r = va0r * vb0r - va0i * vb0i;
    const float vacc1r = va1r * vb1r - va1i * vb1i;
    const float vacc2r = va2r * vb2r - va2i * vb2i;
    const float vacc3r = va3r * vb3r - va3i * vb3i;
    const float vacc4r = va4r * vb4r - va4i * vb4i;
    const float vacc5r = va5r * vb5r - va5i * vb5i;
    const float vacc6r = va6r * vb6r - va6i * vb6i;
    const float vacc7r = va7r * vb7r - va7i * vb7i;

    const float vacc0i = va0r * vb0i + va0i * vb0r;
    const float vacc1i = va1r * vb1i + va1i * vb1r;
    const float vacc2i = va2r * vb2i + va2i * vb2r;
    const float vacc3i = va3r * vb3i + va3i * vb3r;
    const float vacc4i = va4r * vb4i + va4i * vb4r;
    const float vacc5i = va5r * vb5i + va5i * vb5r;
    const float vacc6i = va6r * vb6i + va6i * vb6r;
    const float vacc7i = va7r * vb7i + va7i * vb7r;

    or[0] = vacc0r;
    or[1] = vacc1r;
    or[2] = vacc2r;
    or[3] = vacc3r;
    or[4] = vacc4r;
    or[5] = vacc5r;
    or[6] = vacc6r;
    or[7] = vacc7r;
    or += 8;

    oi[0] = vacc0i;
    oi[1] = vacc1i;
    oi[2] = vacc2i;
    oi[3] = vacc3i;
    oi[4] = vacc4i;
    oi[5] = vacc5i;
    oi[6] = vacc6i;
    oi[7] = vacc7i;
    oi += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float var = *ar++;
      const float vai = *ai++;
      const float vbr = *br++;
      const float vbi = *bi++;
      const float vaccr = var * vbr - vai * vbi;
      const float vacci = var * vbi + vai * vbr;
      *or++ = vaccr;
      *oi++ = vacci;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
