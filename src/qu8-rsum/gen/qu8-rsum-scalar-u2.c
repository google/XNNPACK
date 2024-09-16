// Auto-generated file. Do not edit!
//   Template: src/qu8-rsum/scalar.c.in
//   Generator: tools/xngen
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"


void xnn_qu8_rsum_ukernel__scalar_u2(
    size_t batch,
    const uint8_t* restrict input,
    uint32_t* restrict output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  uint32_t vacc0 = 0;
  for (; batch >= 2; batch -= 2) {
    const uint32_t vt0 = (uint32_t) input[0];
    const uint32_t vt1 = (uint32_t) input[1];
    input += 2;

    vacc0 += vt0;
    vacc0 += vt1;
  }

  if XNN_UNLIKELY(batch != 0) {
    const uint32_t vt = (uint32_t) *input;
    vacc0 += vt;
  }

  *output += vacc0;
}
