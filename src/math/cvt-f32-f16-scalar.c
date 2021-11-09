// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>

#include <fp16/bitcasts.h>


void xnn_math_f32_f16_cvt__scalar(
    size_t n,
    const float* input,
    void* output)
{
  assert(n % (sizeof(uint16_t)) == 0);

  const float vscale_to_inf = 0x1.0p+112f;
  const float vscale_to_zero = 0x1.0p-110f;
  const uint32_t vnonsign_mask = UINT32_C(0x7FFFFFFF);
  const uint32_t vexp_bias = UINT32_C(0x07800000);
  const uint32_t vexpw_max = UINT32_C(0x7F800000);
  const uint32_t vbias_min = UINT32_C(0x40000000);
  const uint16_t vexph_mask = UINT16_C(0x7C00);
  const uint16_t vmanth_mask = UINT16_C(0x0FFF);
  const uint16_t vnanh = UINT16_C(0x7E00);

  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= sizeof(uint16_t)) {
    const float vi = *input++;

    const uint32_t vw = fp32_to_bits(vi);
    const uint32_t vnonsignw = vw & vnonsign_mask;
    const uint32_t vsignw = vw ^ vnonsignw;

    const uint32_t vbias = math_max_u32((vnonsignw + vexp_bias) & vexpw_max, vbias_min);
    const float vf = (fp32_from_bits(vnonsignw) * vscale_to_inf) * vscale_to_zero + fp32_from_bits(vbias);

    const uint32_t vbits = fp32_to_bits(vf);
    const uint16_t vexph = (uint16_t) (vbits >> 13) & vexph_mask;
    const uint16_t vmanth = (uint16_t) vbits & vmanth_mask;
    const uint16_t vnonsignh = vexph + vmanth;
    const uint16_t vsignh = (uint16_t) (vsignw >> 16);

    const uint16_t vh = vsignh | (XNN_UNPREDICTABLE(vnonsignw > vexpw_max) ? vnanh : vnonsignh);

    *o++ = vh;
  }
}
