// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/vunary.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>

void xnn_f32_vrelu_ukernel__scalar_x1(
    size_t n,
    const float* x_ptr,
    float* y_ptr,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x_ptr != NULL);
  assert(y_ptr != NULL);

  const uint32_t* x = (const uint32_t*)x_ptr;
  uint32_t* y = (uint32_t*)y_ptr;

  for (; n >= sizeof(uint32_t); n -= sizeof(uint32_t)) {
    uint32_t vacc = *x++;
    vacc =  ((vacc >> 31) - 1) & vacc;
    *y++ = vacc;
  }
}
