// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/scalar.c.in
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


void xnn_f16_f32_vcvt_ukernel__scalar_x1(
    size_t n,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t vsign_mask = params->scalar.sign_mask;
  const uint32_t vexp_offset = params->scalar.exp_offset;
  const float vexp_scale = params->scalar.exp_scale;
  const uint32_t vmagic_mask = params->scalar.magic_mask;
  const float vmagic_bias = params->scalar.magic_bias;
  const uint32_t vdenorm_cutoff = params->scalar.denorm_cutoff;

  const uint16_t* i = (const uint16_t*) input;
  uint32_t* o = (uint32_t*) output;
  do {
    const uint16_t vh = *i++;

    const uint32_t vw = (uint32_t) vh << 16;
    const uint32_t vsign = vw & vsign_mask;
    const uint32_t v2w = vw + vw;
    const uint32_t vnorm = fp32_to_bits(fp32_from_bits((v2w >> 4) + vexp_offset) * vexp_scale);
    const uint32_t vdenorm = fp32_to_bits(fp32_from_bits((v2w >> 17) | vmagic_mask) - vmagic_bias);
    const uint32_t vf = vsign | (XNN_UNPREDICTABLE(v2w < vdenorm_cutoff) ? vdenorm : vnorm);

    *o++ = vf;

    n -= sizeof(uint16_t);
  } while (n != 0);
}
