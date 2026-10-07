// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vunary/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vabs_ukernel__scalar_u4(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const uint16_t vnonsign_mask = UINT16_C(0x7FFF);
  for (; batch >= 4 * sizeof(uint16_t); batch -= 4 * sizeof(uint16_t)) {
    uint16_t vacc0 = i[0];
    uint16_t vacc1 = i[1];
    uint16_t vacc2 = i[2];
    uint16_t vacc3 = i[3];
    i += 4;

    vacc0 = vacc0 & vnonsign_mask;
    vacc1 = vacc1 & vnonsign_mask;
    vacc2 = vacc2 & vnonsign_mask;
    vacc3 = vacc3 & vnonsign_mask;

    o[0] = vacc0;
    o[1] = vacc1;
    o[2] = vacc2;
    o[3] = vacc3;
    o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      uint16_t vacc = *i++;
      vacc = vacc & vnonsign_mask;
      *o++ = vacc;
      batch -= sizeof(uint16_t);
    } while (batch != 0);
  }
}
