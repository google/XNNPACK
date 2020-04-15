// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/clamp.h>


void xnn_u8_clamp_ukernel__scalar_x4(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);

  const uint8_t voutput_max = params->scalar.max;
  const uint8_t voutput_min = params->scalar.min;

  for (; n >= 4 * sizeof(uint8_t); n -= 4 * sizeof(uint8_t)) {
    uint8_t vt0 = x[0];
    uint8_t vt1 = x[1];
    uint8_t vt2 = x[2];
    uint8_t vt3 = x[3];
    x += 4;

    vt0 = XNN_UNPREDICTABLE(vt0 < voutput_min) ? voutput_min : vt0;
    vt1 = XNN_UNPREDICTABLE(vt1 < voutput_min) ? voutput_min : vt1;
    vt2 = XNN_UNPREDICTABLE(vt2 < voutput_min) ? voutput_min : vt2;
    vt3 = XNN_UNPREDICTABLE(vt3 < voutput_min) ? voutput_min : vt3;

    vt0 = XNN_UNPREDICTABLE(vt0 > voutput_max) ? voutput_max : vt0;
    vt1 = XNN_UNPREDICTABLE(vt1 > voutput_max) ? voutput_max : vt1;
    vt2 = XNN_UNPREDICTABLE(vt2 > voutput_max) ? voutput_max : vt2;
    vt3 = XNN_UNPREDICTABLE(vt3 > voutput_max) ? voutput_max : vt3;

    y[0] = vt0;
    y[1] = vt1;
    y[2] = vt2;
    y[3] = vt3;
    y += 4;
  }

  if XNN_UNLIKELY(n != 0) {
    do {
      uint8_t vt = *x++;
      vt = XNN_UNPREDICTABLE(vt < voutput_min) ? voutput_min : vt;
      vt = XNN_UNPREDICTABLE(vt > voutput_max) ? voutput_max : vt;
      *y++ = vt;

      n -= sizeof(uint8_t);
    } while (n != 0);
  }
}
