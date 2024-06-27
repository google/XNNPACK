// Auto-generated file. Do not edit!
//   Template: src/cs16-vsquareabs/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/math.h"
#include "xnnpack/vsquareabs.h"


void xnn_cs16_vsquareabs_ukernel__scalar_x3(
    size_t batch,
    const int16_t* input,
    uint32_t* output)
{
  assert(batch != 0);
  assert(batch % (sizeof(int16_t) * 2) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 6 * sizeof(int16_t); batch -= 6 * sizeof(int16_t)) {
    const int32_t vr0 = (int32_t) input[0];
    const int32_t vi0 = (int32_t) input[1];
    const int32_t vr1 = (int32_t) input[2];
    const int32_t vi1 = (int32_t) input[3];
    const int32_t vr2 = (int32_t) input[4];
    const int32_t vi2 = (int32_t) input[5];
    input += 3 * 2;

    uint32_t vacc0 = (uint32_t) (vr0 * vr0);
    uint32_t vacc1 = (uint32_t) (vr1 * vr1);
    uint32_t vacc2 = (uint32_t) (vr2 * vr2);

    vacc0 += (uint32_t) (vi0 * vi0);
    vacc1 += (uint32_t) (vi1 * vi1);
    vacc2 += (uint32_t) (vi2 * vi2);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output += 3;
  }
  if XNN_LIKELY(batch != 0) {
    do {
      const int32_t vr = (int32_t) input[0];
      const int32_t vi = (int32_t) input[1];
      input += 2;

      uint32_t vacc = (uint32_t) (vr * vr);
      vacc += (uint32_t) (vi * vi);

      *output++ = vacc;
      batch -= sizeof(int16_t) * 2;
    } while (batch != 0);
  }
}
