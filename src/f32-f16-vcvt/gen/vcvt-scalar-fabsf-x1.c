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

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vcvt.h>

#include <fp16.h>


void xnn_f32_f16_vcvt_ukernel__scalar_fabsf_x1(
    size_t n,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vscale_to_inf = params->scalar_fabsf.scale_to_inf;
  const uint32_t vexp_bias = params->scalar_fabsf.exp_bias;
  const float vscale_to_zero = params->scalar_fabsf.scale_to_zero;
  const uint32_t vexpw_max = params->scalar_fabsf.expw_max;
  const uint32_t vbias_min = params->scalar_fabsf.bias_min;
  const uint16_t vexph_mask = params->scalar_fabsf.exph_mask;
  const uint16_t vmanth_mask = params->scalar_fabsf.manth_mask;
  const uint16_t vnanh = params->scalar_fabsf.nanh;

  uint16_t* o = (uint16_t*) output;
  do {
    const float vx = *input++;

    const float vabsx = fabsf(vx);
    uint32_t vsignw = fp32_to_bits(vx);

    const uint32_t vnonsignw = fp32_to_bits(vabsx);
    float vf = vabsx * vscale_to_inf;

    uint32_t vbias = vnonsignw + vexp_bias;
    vsignw ^= vnonsignw;

    vf *= vscale_to_zero;
    vbias &= vexpw_max;

    vbias = math_max_u32(vbias, vbias_min);

    vf += fp32_from_bits(vbias);

    const uint32_t vbits = fp32_to_bits(vf);

    const uint16_t vexph = (uint16_t) (vbits >> 13) & vexph_mask;
    const uint16_t vmanth = (uint16_t) vbits & vmanth_mask;
    const uint16_t vsignh = (uint16_t) (vsignw >> 16);

    uint16_t vh = vexph + vmanth;
    if XNN_UNPREDICTABLE(vnonsignw > vexpw_max) {
      vh = vnanh;
    }
    vh |= vsignh;

    *o++ = vh;

    n -= sizeof(float);
  } while (n != 0);
}
