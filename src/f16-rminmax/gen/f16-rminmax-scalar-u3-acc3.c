// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>
#include <fp16/fp16.h>

void xnn_f16_rminmax_ukernel__scalar_u3_acc3(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  float vmin0 = fp16_ieee_to_fp32_value(*i);
  float vmax0 = fp16_ieee_to_fp32_value(*i);
  float vmin1 = vmin0;
  float vmax1 = vmax0;
  float vmin2 = vmin0;
  float vmax2 = vmax0;
  for (; batch >= 3 * sizeof(uint16_t); batch -= 3 * sizeof(uint16_t)) {
    const float vt0 = fp16_ieee_to_fp32_value(i[0]);
    const float vt1 = fp16_ieee_to_fp32_value(i[1]);
    const float vt2 = fp16_ieee_to_fp32_value(i[2]);
    i += 3;

    vmin0 = math_min_f32(vmin0, vt0);
    vmax0 = math_max_f32(vmax0, vt0);
    vmin1 = math_min_f32(vmin1, vt1);
    vmax1 = math_max_f32(vmax1, vt1);
    vmin2 = math_min_f32(vmin2, vt2);
    vmax2 = math_max_f32(vmax2, vt2);
  }
  vmin0 = math_min_f32(vmin0, vmin1);
  vmax0 = math_max_f32(vmax0, vmax1);
  vmin0 = math_min_f32(vmin0, vmin2);
  vmax0 = math_max_f32(vmax0, vmax2);

  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vt = fp16_ieee_to_fp32_value(*i++);
      vmin0 = math_min_f32(vmin0, vt);
      vmax0 = math_max_f32(vmax0, vt);
      batch -= sizeof(uint16_t);
    } while (batch != 0);
  }
  o[0] = fp16_ieee_from_fp32_value(vmin0);
  o[1] = fp16_ieee_from_fp32_value(vmax0);
}
