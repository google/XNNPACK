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

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vbinary.h>


void xnn_f16_vrsubc_minmax_ukernel__fp16arith_x4(
    size_t n,
    const void* restrict a_ptr,
    const void* restrict b_ptr,
    void* restrict y_ptr,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float16_t) == 0);
  assert(a_ptr != NULL);
  assert(b_ptr != NULL);
  assert(y_ptr != NULL);

  const float16_t* a = (const float16_t*) a_ptr;
  const float16_t* b = (const float16_t*) b_ptr;
  float16_t* y = (float16_t*) y_ptr;

  float16_t vy_min, vy_max;
  memcpy(&vy_min, &params->fp16arith.min, sizeof(vy_min));
  memcpy(&vy_max, &params->fp16arith.max, sizeof(vy_max));

  const float16_t vb = *b;
  for (; n >= 4 * sizeof(float16_t); n -= 4 * sizeof(float16_t)) {
    float16_t vacc0 = a[0];
    float16_t vacc1 = a[1];
    float16_t vacc2 = a[2];
    float16_t vacc3 = a[3];
    a += 4;

    vacc0 = vsubh_f16(vb, vacc0);
    vacc1 = vsubh_f16(vb, vacc1);
    vacc2 = vsubh_f16(vb, vacc2);
    vacc3 = vsubh_f16(vb, vacc3);


    vacc0 = vmaxh_f16(vacc0, vy_min);
    vacc1 = vmaxh_f16(vacc1, vy_min);
    vacc2 = vmaxh_f16(vacc2, vy_min);
    vacc3 = vmaxh_f16(vacc3, vy_min);

    vacc0 = vminh_f16(vacc0, vy_max);
    vacc1 = vminh_f16(vacc1, vy_max);
    vacc2 = vminh_f16(vacc2, vy_max);
    vacc3 = vminh_f16(vacc3, vy_max);

    y[0] = vacc0;
    y[1] = vacc1;
    y[2] = vacc2;
    y[3] = vacc3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      float16_t vacc = *a++;
      vacc = vsubh_f16(vb, vacc);
      vacc = vmaxh_f16(vacc, vy_min);
      vacc = vminh_f16(vacc, vy_max);
      *y++ = vacc;
      n -= sizeof(float16_t);
    } while (n != 0);
  }
}
