// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/scalar.c.in
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

static int16_t math_signcompliment_f16(const uint16_t a) {
  return (a & 0x7FFF) ^ -((int16_t) a < 0);
}

static uint16_t math_max_f16(const uint16_t a, const uint16_t b) {
  return math_signcompliment_f16(a) > math_signcompliment_f16(b) ? a : b;
}


void xnn_f16_rmax_ukernel__scalar_u2_acc2(
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

  uint16_t vmax0 = *i;
  uint16_t vmax1 = vmax0;
  for (; batch >= 2 * sizeof(uint16_t); batch -= 2 * sizeof(uint16_t)) {
    const uint16_t vt0 = i[0];
    const uint16_t vt1 = i[1];
    i += 2;

    vmax0 = math_max_f16(vmax0, vt0);
    vmax1 = math_max_f16(vmax1, vt1);
  }
  vmax0 = math_max_f16(vmax0, vmax1);

  if XNN_UNLIKELY(batch != 0) {
    const uint16_t vt = *i;
    vmax0 = math_max_f16(vmax0, vt);
  }
  o[0] = vmax0;
}
