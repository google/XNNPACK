// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_s8_vclamp_ukernel__scalar_x4(
    size_t n,
    const int8_t* x,
    int8_t* y,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);

  const int32_t voutput_max = params->scalar.max;
  const int32_t voutput_min = params->scalar.min;

  for (; n >= 4 * sizeof(int8_t); n -= 4 * sizeof(int8_t)) {
    int32_t vt0 = (int32_t) x[0];
    int32_t vt1 = (int32_t) x[1];
    int32_t vt2 = (int32_t) x[2];
    int32_t vt3 = (int32_t) x[3];
    x += 4;

    vt0 = math_max_s32(vt0, voutput_min);
    vt1 = math_max_s32(vt1, voutput_min);
    vt2 = math_max_s32(vt2, voutput_min);
    vt3 = math_max_s32(vt3, voutput_min);

    vt0 = math_min_s32(vt0, voutput_max);
    vt1 = math_min_s32(vt1, voutput_max);
    vt2 = math_min_s32(vt2, voutput_max);
    vt3 = math_min_s32(vt3, voutput_max);

    y[0] = (int8_t) vt0;
    y[1] = (int8_t) vt1;
    y[2] = (int8_t) vt2;
    y[3] = (int8_t) vt3;
    y += 4;
  }

  if XNN_UNLIKELY(n != 0) {
    do {
      int32_t vt = (int32_t) *x++;
      vt = math_max_s32(vt, voutput_min);
      vt = math_min_s32(vt, voutput_max);
      *y++ = (int8_t) vt;

      n -= sizeof(int8_t);
    } while (n != 0);
  }
}
