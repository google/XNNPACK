// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vop-fp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_fp16.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f16_vsqrdiff_ukernel__fp16arith_x4(
    size_t n,
    const void* restrict a_ptr,
    const void* restrict b_ptr,
    void* restrict y_ptr,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float16_t) == 0);
  assert(a_ptr != NULL);
  assert(b_ptr != NULL);
  assert(y_ptr != NULL);

  const float16_t* a = (const float16_t*) a_ptr;
  const float16_t* b = (const float16_t*) b_ptr;
  float16_t* y = (float16_t*) y_ptr;


  for (; n >= 4 * sizeof(float16_t); n -= 4 * sizeof(float16_t)) {
    const float16_t va0 = *a++;
    const float16_t va1 = *a++;
    const float16_t va2 = *a++;
    const float16_t va3 = *a++;

    const float16_t vb0 = *b++;
    const float16_t vb1 = *b++;
    const float16_t vb2 = *b++;
    const float16_t vb3 = *b++;

    float16_t vacc0 = vsubh_f16(va0, vb0);
    float16_t vacc1 = vsubh_f16(va1, vb1);
    float16_t vacc2 = vsubh_f16(va2, vb2);
    float16_t vacc3 = vsubh_f16(va3, vb3);

    vacc0 = vmulh_f16(vacc0, vacc0);
    vacc1 = vmulh_f16(vacc1, vacc1);
    vacc2 = vmulh_f16(vacc2, vacc2);
    vacc3 = vmulh_f16(vacc3, vacc3);


    *y++ = vacc0;
    *y++ = vacc1;
    *y++ = vacc2;
    *y++ = vacc3;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const float16_t va = *a++;
      const float16_t vb = *b++;
      float16_t vacc = vsubh_f16(va, vb);
      vacc = vmulh_f16(vacc, vacc);
      *y++ = vacc;
      n -= sizeof(float16_t);
    } while (n != 0);
  }
}
