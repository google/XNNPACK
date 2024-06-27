// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vopc-fp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <string.h>

#include <arm_fp16.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"


void xnn_f16_vrdivc_minmax_ukernel__fp16arith_u4(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float16_t* a = (const float16_t*) input_a;
  const float16_t* b = (const float16_t*) input_b;
  float16_t* o = (float16_t*) output;

  float16_t vy_min, vy_max;
  memcpy(&vy_min, &params->fp16arith.min, sizeof(vy_min));
  memcpy(&vy_max, &params->fp16arith.max, sizeof(vy_max));

  const float16_t vb = *b;
  for (; batch >= 4 * sizeof(float16_t); batch -= 4 * sizeof(float16_t)) {
    float16_t vacc0 = a[0];
    float16_t vacc1 = a[1];
    float16_t vacc2 = a[2];
    float16_t vacc3 = a[3];
    a += 4;

    vacc0 = vdivh_f16(vb, vacc0);
    vacc1 = vdivh_f16(vb, vacc1);
    vacc2 = vdivh_f16(vb, vacc2);
    vacc3 = vdivh_f16(vb, vacc3);


    vacc0 = vmaxnmh_f16(vacc0, vy_min);
    vacc1 = vmaxnmh_f16(vacc1, vy_min);
    vacc2 = vmaxnmh_f16(vacc2, vy_min);
    vacc3 = vmaxnmh_f16(vacc3, vy_min);

    vacc0 = vminnmh_f16(vacc0, vy_max);
    vacc1 = vminnmh_f16(vacc1, vy_max);
    vacc2 = vminnmh_f16(vacc2, vy_max);
    vacc3 = vminnmh_f16(vacc3, vy_max);

    o[0] = vacc0;
    o[1] = vacc1;
    o[2] = vacc2;
    o[3] = vacc3;
    o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float16_t vacc = *a++;
      vacc = vdivh_f16(vb, vacc);
      vacc = vmaxnmh_f16(vacc, vy_min);
      vacc = vminnmh_f16(vacc, vy_max);
      *o++ = vacc;
      batch -= sizeof(float16_t);
    } while (batch != 0);
  }
}
