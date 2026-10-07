// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vunary/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vabs_ukernel__hvx_u128(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const HVX_Vector vnonsign_mask = Q6_Vh_vsplat_R(0x7FFF);
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    HVX_Vector vacc0 = *((const HVX_UVector*) i);
    HVX_Vector vacc1 = *((const HVX_UVector*) (i + 64));
    i += 128;

    vacc0 = Q6_V_vand_VV(vacc0, vnonsign_mask);
    vacc1 = Q6_V_vand_VV(vacc1, vnonsign_mask);

    *((HVX_UVector*) o) = vacc0;
    *((HVX_UVector*) (o + 64)) = vacc1;
    o += 128;
  }
  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    HVX_Vector vacc = *((const HVX_UVector*) i);
    i += 64;
    vacc = Q6_V_vand_VV(vacc, vnonsign_mask);
    *((HVX_UVector*) o) = vacc;
    o += 64;
  }
  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector vacc = *((const HVX_UVector*) i);
    vacc = Q6_V_vand_VV(vacc, vnonsign_mask);
    Q6_V_vstu_variable(o, batch, vacc);
  }
}
