// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/scalar-bitcast.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vcvt.h>

#include <fp16.h>


void xnn_f32_f16_vcvt_ukernel__scalar_bitcast_x2(
    size_t n,
    const float* input,
    void* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t vnonsign_mask = UINT32_C(0x7FFFFFFF);
  const uint32_t vexp_bias = UINT32_C(0x07800000);
  const float vscale_to_inf = 0x1.0p+112f;
  const uint32_t vexpw_max = UINT32_C(0x7F800000);
  const float vscale_to_zero = 0x1.0p-110f;
  const uint32_t vbias_min = UINT32_C(0x40000000);
  const uint16_t vexph_mask = UINT16_C(0x7C00);
  const uint16_t vmanth_mask = UINT16_C(0x0FFF);
  const uint16_t vnanh = UINT16_C(0x7E00);

  uint16_t* o = (uint16_t*) output;
  for (; n >= 2 * sizeof(uint16_t); n -= 2 * sizeof(uint16_t)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    const uint32_t vw0 = fp32_to_bits(vx0);
    const uint32_t vw1 = fp32_to_bits(vx1);
    const uint32_t vnonsignw0 = vw0 & vnonsign_mask;
    const uint32_t vnonsignw1 = vw1 & vnonsign_mask;

    float vf0 = fp32_from_bits(vnonsignw0);
    float vf1 = fp32_from_bits(vnonsignw1);
    const uint32_t vsignw0 = vw0 ^ vnonsignw0;
    const uint32_t vsignw1 = vw1 ^ vnonsignw1;
    uint32_t vbias0 = vnonsignw0 + vexp_bias;
    uint32_t vbias1 = vnonsignw1 + vexp_bias;

    vf0 *= vscale_to_inf;
    vf1 *= vscale_to_inf;
    vbias0 &= vexpw_max;
    vbias1 &= vexpw_max;

    vf0 *= vscale_to_zero;
    vf1 *= vscale_to_zero;
    vbias0 = math_max_u32(vbias0, vbias_min);
    vbias1 = math_max_u32(vbias1, vbias_min);

    vf0 += fp32_from_bits(vbias0);
    vf1 += fp32_from_bits(vbias1);

    const uint32_t vbits0 = fp32_to_bits(vf0);
    const uint32_t vbits1 = fp32_to_bits(vf1);

    const uint16_t vexph0 = (uint16_t) (vbits0 >> 13) & vexph_mask;
    const uint16_t vexph1 = (uint16_t) (vbits1 >> 13) & vexph_mask;
    const uint16_t vmanth0 = (uint16_t) vbits0 & vmanth_mask;
    const uint16_t vmanth1 = (uint16_t) vbits1 & vmanth_mask;
    const uint16_t vsignh0 = (uint16_t) (vsignw0 >> 16);
    const uint16_t vsignh1 = (uint16_t) (vsignw1 >> 16);

    uint16_t vh0 = vexph0 + vmanth0;
    uint16_t vh1 = vexph1 + vmanth1;
    if XNN_UNPREDICTABLE(vnonsignw0 > vexpw_max) {
      vh0 = vnanh;
    }
    if XNN_UNPREDICTABLE(vnonsignw1 > vexpw_max) {
      vh1 = vnanh;
    }
    vh0 |= vsignh0;
    vh1 |= vsignh1;

    o[0] = vh0;
    o[1] = vh1;
    o += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    const float vx = *input;

    const uint32_t vw = fp32_to_bits(vx);
    const uint32_t vnonsignw = vw & vnonsign_mask;

    float vf = fp32_from_bits(vnonsignw);
    const uint32_t vsignw = vw ^ vnonsignw;
    uint32_t vbias = vnonsignw + vexp_bias;

    vf *= vscale_to_inf;
    vbias &= vexpw_max;

    vf *= vscale_to_zero;
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

    *o = vh;
  }
}
