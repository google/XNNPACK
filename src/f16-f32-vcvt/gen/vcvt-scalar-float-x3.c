// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/scalar-float.c.in
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


void xnn_f16_f32_vcvt_ukernel__scalar_float_x3(
    size_t n,
    const void* input,
    float* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  const uint32_t vsign_mask = UINT32_C(0x80000000);
  const uint32_t vexp_offset = UINT32_C(0x70000000);
  const float vexp_scale = 0x1.0p-112f;
  const uint32_t vmagic_mask = UINT32_C(0x3F000000);
  const float vmagic_bias = 0.5f;
  const uint32_t vdenormalized_cutoff = UINT32_C(0x08000000);
  for (; n >= 3 * sizeof(float); n -= 3 * sizeof(float)) {
    const uint16_t vh0 = i[0];
    const uint16_t vh1 = i[1];
    const uint16_t vh2 = i[2];
    i += 3;

    const uint32_t vw0 = (uint32_t) vh0 << 16;
    const uint32_t vw1 = (uint32_t) vh1 << 16;
    const uint32_t vw2 = (uint32_t) vh2 << 16;

    const uint32_t vsign0 = vw0 & vsign_mask;
    const uint32_t vsign1 = vw1 & vsign_mask;
    const uint32_t vsign2 = vw2 & vsign_mask;

    const uint32_t v2w0 = vw0 + vw0;
    const uint32_t v2w1 = vw1 + vw1;
    const uint32_t v2w2 = vw2 + vw2;

    const uint32_t vnorm0 = fp32_to_bits(fp32_from_bits((v2w0 >> 4) + vexp_offset) * vexp_scale);
    const uint32_t vnorm1 = fp32_to_bits(fp32_from_bits((v2w1 >> 4) + vexp_offset) * vexp_scale);
    const uint32_t vnorm2 = fp32_to_bits(fp32_from_bits((v2w2 >> 4) + vexp_offset) * vexp_scale);

    const uint32_t vdenorm0 = fp32_to_bits(fp32_from_bits((v2w0 >> 17) | vmagic_mask) - vmagic_bias);
    const uint32_t vdenorm1 = fp32_to_bits(fp32_from_bits((v2w1 >> 17) | vmagic_mask) - vmagic_bias);
    const uint32_t vdenorm2 = fp32_to_bits(fp32_from_bits((v2w2 >> 17) | vmagic_mask) - vmagic_bias);

    const uint32_t vf0 = vsign0 | (XNN_UNPREDICTABLE(v2w0 < vdenormalized_cutoff) ? vdenorm0 : vnorm0);
    const uint32_t vf1 = vsign1 | (XNN_UNPREDICTABLE(v2w1 < vdenormalized_cutoff) ? vdenorm1 : vnorm1);
    const uint32_t vf2 = vsign2 | (XNN_UNPREDICTABLE(v2w2 < vdenormalized_cutoff) ? vdenorm2 : vnorm2);

    output[0] = fp32_from_bits(vf0);
    output[1] = fp32_from_bits(vf1);
    output[2] = fp32_from_bits(vf2);
    output += 3;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const uint16_t vh = *i++;

      const float vf = fp16_ieee_to_fp32_value(vh);

      *output++ = vf;

      n -= sizeof(float);
    } while (n != 0);
  }
}
