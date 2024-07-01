// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/scalar.c.in
//   Generator: tools/xngen
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"


void xnn_qs8_rsum_ukernel__scalar_u4(
    size_t batch,
    const int8_t* restrict input,
    int32_t* restrict output,
    const union xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  int32_t vacc0 = 0;
  for (; batch >= 4; batch -= 4) {
    const int32_t vt0 = (int32_t) input[0];
    const int32_t vt1 = (int32_t) input[1];
    const int32_t vt2 = (int32_t) input[2];
    const int32_t vt3 = (int32_t) input[3];
    input += 4;

    vacc0 += vt0;
    vacc0 += vt1;
    vacc0 += vt2;
    vacc0 += vt3;
  }

  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t vt = (int32_t) *input++;
      vacc0 += vt;
      batch -= sizeof(int8_t);
    } while (batch != 0);
  }

  *output += vacc0;
}
