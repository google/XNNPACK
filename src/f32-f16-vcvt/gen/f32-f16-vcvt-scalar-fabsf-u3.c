// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/scalar-fabsf.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vcvt.h"


void xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3(
    size_t batch,
    const float* input,
    void* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vscale_to_inf = 0x1.0p+112f;
  const uint32_t vexp_bias = UINT32_C(0x07800000);
  const float vscale_to_zero = 0x1.0p-110f;
  const uint32_t vexpw_max = UINT32_C(0x7F800000);
  const uint32_t vbias_min = UINT32_C(0x40000000);
  const uint16_t vexph_mask = UINT16_C(0x7C00);
  const uint16_t vmanth_mask = UINT16_C(0x0FFF);
  const uint16_t vnanh = UINT16_C(0x7E00);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 3 * sizeof(float); batch -= 3 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    input += 3;

    const float vabsx0 = fabsf(vx0);
    const float vabsx1 = fabsf(vx1);
    const float vabsx2 = fabsf(vx2);
    uint32_t vsignw0 = float_as_uint32(vx0);
    uint32_t vsignw1 = float_as_uint32(vx1);
    uint32_t vsignw2 = float_as_uint32(vx2);

    const uint32_t vnonsignw0 = float_as_uint32(vabsx0);
    const uint32_t vnonsignw1 = float_as_uint32(vabsx1);
    const uint32_t vnonsignw2 = float_as_uint32(vabsx2);
    float vf0 = vabsx0 * vscale_to_inf;
    float vf1 = vabsx1 * vscale_to_inf;
    float vf2 = vabsx2 * vscale_to_inf;

    uint32_t vbias0 = vnonsignw0 + vexp_bias;
    uint32_t vbias1 = vnonsignw1 + vexp_bias;
    uint32_t vbias2 = vnonsignw2 + vexp_bias;
    vsignw0 ^= vnonsignw0;
    vsignw1 ^= vnonsignw1;
    vsignw2 ^= vnonsignw2;

    vf0 *= vscale_to_zero;
    vf1 *= vscale_to_zero;
    vf2 *= vscale_to_zero;
    vbias0 &= vexpw_max;
    vbias1 &= vexpw_max;
    vbias2 &= vexpw_max;

    vbias0 = math_max_u32(vbias0, vbias_min);
    vbias1 = math_max_u32(vbias1, vbias_min);
    vbias2 = math_max_u32(vbias2, vbias_min);

    vf0 += uint32_as_float(vbias0);
    vf1 += uint32_as_float(vbias1);
    vf2 += uint32_as_float(vbias2);

    const uint32_t vbits0 = float_as_uint32(vf0);
    const uint32_t vbits1 = float_as_uint32(vf1);
    const uint32_t vbits2 = float_as_uint32(vf2);

    const uint16_t vexph0 = (uint16_t) (vbits0 >> 13) & vexph_mask;
    const uint16_t vexph1 = (uint16_t) (vbits1 >> 13) & vexph_mask;
    const uint16_t vexph2 = (uint16_t) (vbits2 >> 13) & vexph_mask;
    const uint16_t vmanth0 = (uint16_t) vbits0 & vmanth_mask;
    const uint16_t vmanth1 = (uint16_t) vbits1 & vmanth_mask;
    const uint16_t vmanth2 = (uint16_t) vbits2 & vmanth_mask;
    const uint16_t vsignh0 = (uint16_t) (vsignw0 >> 16);
    const uint16_t vsignh1 = (uint16_t) (vsignw1 >> 16);
    const uint16_t vsignh2 = (uint16_t) (vsignw2 >> 16);

    uint16_t vh0 = vexph0 + vmanth0;
    uint16_t vh1 = vexph1 + vmanth1;
    uint16_t vh2 = vexph2 + vmanth2;
    if XNN_UNPREDICTABLE(vnonsignw0 > vexpw_max) {
      vh0 = vnanh;
    }
    if XNN_UNPREDICTABLE(vnonsignw1 > vexpw_max) {
      vh1 = vnanh;
    }
    if XNN_UNPREDICTABLE(vnonsignw2 > vexpw_max) {
      vh2 = vnanh;
    }
    vh0 |= vsignh0;
    vh1 |= vsignh1;
    vh2 |= vsignh2;

    o[0] = vh0;
    o[1] = vh1;
    o[2] = vh2;
    o += 3;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;

      const float vabsx = fabsf(vx);
      uint32_t vsignw = float_as_uint32(vx);

      const uint32_t vnonsignw = float_as_uint32(vabsx);
      float vf = vabsx * vscale_to_inf;

      uint32_t vbias = vnonsignw + vexp_bias;
      vsignw ^= vnonsignw;

      vf *= vscale_to_zero;
      vbias &= vexpw_max;

      vbias = math_max_u32(vbias, vbias_min);

      vf += uint32_as_float(vbias);

      const uint32_t vbits = float_as_uint32(vf);

      const uint16_t vexph = (uint16_t) (vbits >> 13) & vexph_mask;
      const uint16_t vmanth = (uint16_t) vbits & vmanth_mask;
      const uint16_t vsignh = (uint16_t) (vsignw >> 16);

      uint16_t vh = vexph + vmanth;
      if XNN_UNPREDICTABLE(vnonsignw > vexpw_max) {
        vh = vnanh;
      }
      vh |= vsignh;

      *o++ = vh;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}
