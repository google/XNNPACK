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

#include <xnnpack/math.h>
#include <xnnpack/vsquareabs.h>


void xnn_cs16_vsquareabs_ukernel__scalar_x4(
    size_t batch,
    const int16_t* input,
    uint32_t* output) {

  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4; batch -= 4) {
    const int32_t vr0 = (int32_t) input[0];
    const int32_t vi0 = (int32_t) input[1];
    const int32_t vr1 = (int32_t) input[2];
    const int32_t vi1 = (int32_t) input[3];
    const int32_t vr2 = (int32_t) input[4];
    const int32_t vi2 = (int32_t) input[5];
    const int32_t vr3 = (int32_t) input[6];
    const int32_t vi3 = (int32_t) input[7];
    input += 4 * 2;

    const uint32_t vrsquare0 = (uint32_t) (vr0 * vr0);
    const uint32_t visquare0 = (uint32_t) (vi0 * vi0);
    const uint32_t vrsquare1 = (uint32_t) (vr1 * vr1);
    const uint32_t visquare1 = (uint32_t) (vi1 * vi1);
    const uint32_t vrsquare2 = (uint32_t) (vr2 * vr2);
    const uint32_t visquare2 = (uint32_t) (vi2 * vi2);
    const uint32_t vrsquare3 = (uint32_t) (vr3 * vr3);
    const uint32_t visquare3 = (uint32_t) (vi3 * vi3);

    const uint32_t vout0 = vrsquare0 + visquare0;
    const uint32_t vout1 = vrsquare1 + visquare1;
    const uint32_t vout2 = vrsquare2 + visquare2;
    const uint32_t vout3 = vrsquare3 + visquare3;

    output[0] = vout0;
    output[1] = vout1;
    output[2] = vout2;
    output[3] = vout3;
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t vr = (int32_t) input[0];
      const int32_t vi = (int32_t) input[1];
      input += 2;

      const uint32_t vrsquare = (uint32_t) (vr * vr);
      const uint32_t visquare = (uint32_t) (vi * vi);

      const uint32_t vout = vrsquare + visquare;

      *output++ = vout;
    } while (--batch != 0);
  }
}
