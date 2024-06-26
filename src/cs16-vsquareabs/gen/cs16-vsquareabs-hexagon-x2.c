// Auto-generated file. Do not edit!
//   Template: src/cs16-vsquareabs/hexagon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "xnnpack/vsquareabs.h"


void xnn_cs16_vsquareabs_ukernel__hexagon_x2(
    size_t batch,
    const int16_t* input,
    uint32_t* output) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % (sizeof(int16_t) * 2) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HEXAGON_Vect64* i = (const HEXAGON_Vect64*) input;
  HEXAGON_Vect64* o = (HEXAGON_Vect64*) output;
  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    HEXAGON_Vect64 vacc = *i++;
    vacc = Q6_P_vdmpy_PP_sat(vacc, vacc);
    *o++ = vacc;
  }
  if XNN_LIKELY(batch != 0) {
    assert(batch == 2 * sizeof(int16_t));

    const HEXAGON_Vect32 vi = *((const HEXAGON_Vect32*) i);
    HEXAGON_Vect32 vacc = Q6_R_mpy_RlRl(vi, vi);
    vacc = Q6_R_mpyacc_RhRh_sat(vacc, vi, vi);
    *((HEXAGON_Vect32*) o) = vacc;
  }
}
