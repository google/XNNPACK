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
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
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

    n -= sizeof(uint16_t);
  } while (n != 0);
}
