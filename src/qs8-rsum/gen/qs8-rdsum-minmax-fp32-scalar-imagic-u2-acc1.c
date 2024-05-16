// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/scalar.c.in
//   Generator: tools/xngen
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>


void xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2(
    size_t batch,
    const int8_t* restrict input,
    int8_t* restrict output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  int32_t vacc0 = vinit_bias;
  for (; batch >= 2; batch -= 2) {
    const int32_t vt0 = (int32_t) input[0];
    const int32_t vt1 = (int32_t) input[1];
    input += 2;

    vacc0 += vt0;
    vacc0 += vt1;
  }

  if XNN_UNLIKELY(batch != 0) {
    const int32_t vt = (int32_t) *input;
    vacc0 += vt;
  }

  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;

  float vfpacc = (float) vacc0 * vscale;
  vfpacc += vmagic_bias;
  int32_t vout = (int32_t) float_as_uint32(vfpacc);
  vout = math_max_s32(vout, vmagic_min);
  vout = math_min_s32(vout, vmagic_max);
  vout -= vmagic_bias_less_zero_point;

  *output += (int8_t) vout;
}
