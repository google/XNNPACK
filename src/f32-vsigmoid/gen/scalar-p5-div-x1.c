// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/scalar-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>

#include <fp16/bitcasts.h>


void xnn_f32_vsigmoid_ukernel__scalar_p5_div_x1(
    size_t n,
    const float* x,
    float* y,
    const void* params)
{
  assert(n % sizeof(float) == 0);

  const float vmagic_bias = 0x1.8000FEp23f;
  const float vminus_log2e = -0x1.715476p+0f;
  const float vln2_hi = 0x1.62E400p-1f;
  const float vln2_lo = 0x1.7F7D1Cp-20f;
  const float vc5 = -0x1.0F9F9Cp-7f;
  const float vc4 =  0x1.573A1Ap-5f;
  const float vc3 = -0x1.555A80p-3f;
  const float vc2 =  0x1.FFFDC6p-2f;
  const float vc1 = -0x1.FFFFF6p-1f;
  const float vone = 1.0f;
  const float vdenorm_cutoff = 0x1.5D589Ep+6f;

  do {
    const float vx = *x++;

    const float vz = fabsf(vx);

    float vn = vz * vminus_log2e + vmagic_bias;
    const float vs = fp32_from_bits(fp32_to_bits(vn) << 23);
    vn -= vmagic_bias;

    float vt = vn * vln2_hi + vz;
    vt = vn * vln2_lo + vt;

    float vp = vt * vc5 + vc4;
    vp = vt * vp + vc3;
    vp = vt * vp + vc2;
    vp = vt * vp + vc1;

    vt *= vs;
    const float ve = vt * vp + vs;
    const float vd = ve + vone;

    float vf = ve / vd;
    if XNN_UNPREDICTABLE(vz > vdenorm_cutoff) {
      vf = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx > 0.0f) {
      vf = vone - vf;
    }

    *y++ = vf;

    n -= sizeof(float);
  } while (n != 0);
}
