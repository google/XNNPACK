// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"


void xnn_f16_rminmax_ukernel__scalar_u4_acc2(
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

  int16_t vt = math_signcomplement_f16(*i);
  int16_t vmin0 = vt;
  int16_t vmax0 = vt;
  int16_t vmin1 = vt;
  int16_t vmax1 = vt;
  for (; batch >= 4 * sizeof(uint16_t); batch -= 4 * sizeof(uint16_t)) {
    const int16_t vt0 = math_signcomplement_f16(i[0]);
    const int16_t vt1 = math_signcomplement_f16(i[1]);
    const int16_t vt2 = math_signcomplement_f16(i[2]);
    const int16_t vt3 = math_signcomplement_f16(i[3]);
    i += 4;

    vmin0 = math_min_s16(vmin0, vt0);
    vmax0 = math_max_s16(vmax0, vt0);
    vmin1 = math_min_s16(vmin1, vt1);
    vmax1 = math_max_s16(vmax1, vt1);
    vmin0 = math_min_s16(vmin0, vt2);
    vmax0 = math_max_s16(vmax0, vt2);
    vmin1 = math_min_s16(vmin1, vt3);
    vmax1 = math_max_s16(vmax1, vt3);
  }
  vmin0 = math_min_s16(vmin0, vmin1);
  vmax0 = math_max_s16(vmax0, vmax1);

  if XNN_UNLIKELY(batch != 0) {
    do {
      vt = math_signcomplement_f16(*i++);
      vmin0 = math_min_s16(vmin0, vt);
      vmax0 = math_max_s16(vmax0, vt);
      batch -= sizeof(uint16_t);
    } while (batch != 0);
  }
  o[0] = (uint16_t) math_signcomplement_f16((uint16_t) vmin0);
  o[1] = (uint16_t) math_signcomplement_f16((uint16_t) vmax0);
}
