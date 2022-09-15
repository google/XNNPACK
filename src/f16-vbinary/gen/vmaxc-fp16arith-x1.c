// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vopc-fp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_fp16.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vbinary.h>


void xnn_f16_vmaxc_ukernel__fp16arith_x1(
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


  const float16_t vb = *b;
  do {
    float16_t vacc = *a++;
    vacc = vmaxh_f16(vacc, vb);
    *y++ = vacc;
    n -= sizeof(float16_t);
  } while (n != 0);
}
