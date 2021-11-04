// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/scalar-float.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>

#include <fp16.h>


void xnn_f32_f16_vcvt_ukernel__scalar_float_x2(
    size_t n,
    const float* input,
    void* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = (const float*) input;
  const uint32_t vsign_mask = UINT32_C(0x80000000);
  const uint32_t vbias_mask = UINT32_C(0xFF000000);
  const uint32_t vmin_bias = UINT32_C(0x71000000);
  const uint32_t vbase_offset = UINT32_C(0x07800000);
  const uint32_t vexp_mask = UINT32_C(0x00007C00);
  const uint32_t vmantissa_mask = UINT32_C(0x00000FFF);
  const uint16_t vmagic_mask = UINT16_C(0x7E00);
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
  for (; n >= 2 * sizeof(uint16_t); n -= 2 * sizeof(uint16_t)) {
    const float vh0 = i[0];
    const float vh1 = i[1];
    i += 2;

    float vbase0 = (fabsf(vh0) * scale_to_inf) * scale_to_zero;
    float vbase1 = (fabsf(vh1) * scale_to_inf) * scale_to_zero;

    const uint32_t vw0 = fp32_to_bits(vh0);
    const uint32_t vw1 = fp32_to_bits(vh1);

    const uint32_t v2w0 = vw0 + vw0;
    const uint32_t v2w1 = vw1 + vw1;

    const uint32_t vsign0 = vw0 & vsign_mask;
    const uint32_t vsign1 = vw1 & vsign_mask;

    uint32_t vbias0 = v2w0 & vbias_mask;
    uint32_t vbias1 = v2w1 & vbias_mask;

    vbias0 = XNN_UNPREDICTABLE(vbias0 < vmin_bias) ? vmin_bias : vbias0;
    vbias1 = XNN_UNPREDICTABLE(vbias1 < vmin_bias) ? vmin_bias : vbias1;

    vbase0 = fp32_from_bits((vbias0 >> 1) + vbase_offset) + vbase0;
    vbase1 = fp32_from_bits((vbias1 >> 1) + vbase_offset) + vbase1;

    const uint32_t vbits0 = fp32_to_bits(vbase0);
    const uint32_t vbits1 = fp32_to_bits(vbase1);

    const uint32_t vexp_bits0 = (vbits0 >> 13) & vexp_mask;
    const uint32_t vexp_bits1 = (vbits1 >> 13) & vexp_mask;

    const uint32_t vmantissa_bits0 = vbits0 & vmantissa_mask;
    const uint32_t vmantissa_bits1 = vbits1 & vmantissa_mask;

    const uint32_t vnonsign0 = vexp_bits0 + vmantissa_bits0;
    const uint32_t vnonsign1 = vexp_bits1 + vmantissa_bits1;

    const uint16_t vr0 = (vsign0 >> 16) | (XNN_UNPREDICTABLE(v2w0 > vbias_mask) ? vmagic_mask : vnonsign0);
    const uint16_t vr1 = (vsign1 >> 16) | (XNN_UNPREDICTABLE(v2w1 > vbias_mask) ? vmagic_mask : vnonsign1);

    ((uint16_t*) output)[0] = vr0;
    ((uint16_t*) output)[1] = vr1;

    output = (uint16_t*) output + 2;
  }
  if XNN_UNLIKELY(n != 0) {
    const float vh = *i;

    const uint16_t vf = fp16_ieee_from_fp32_value(vh);

    ((uint16_t*) output)[0] = vf;
  }
}
