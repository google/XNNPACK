// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-rminmax/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"


void xnn_bf16_rmin_ukernel__neon_u24_acc3(
    size_t batch,
    const xnn_bfloat16* input,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const uint16x8_t vsign_mask = vdupq_n_u16(UINT16_C(0x7FFF));

  int16x8_t vmin0 = vreinterpretq_s16_u16(vld1q_dup_u16(o));
  vmin0 = veorq_s16(vmin0, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vmin0, 15)), vsign_mask)));
  int16x8_t vmin1 = vmin0;
  int16x8_t vmin2 = vmin0;
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    const int16x8_t vi0 = vreinterpretq_s16_u16(vld1q_u16(i)); i += 8;
    const int16x8_t vi1 = vreinterpretq_s16_u16(vld1q_u16(i)); i += 8;
    const int16x8_t vi2 = vreinterpretq_s16_u16(vld1q_u16(i)); i += 8;

    const int16x8_t vt0 = veorq_s16(vi0, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vi0, 15)), vsign_mask)));
    const int16x8_t vt1 = veorq_s16(vi1, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vi1, 15)), vsign_mask)));
    const int16x8_t vt2 = veorq_s16(vi2, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vi2, 15)), vsign_mask)));

    vmin0 = vminq_s16(vmin0, vt0);
    vmin1 = vminq_s16(vmin1, vt1);
    vmin2 = vminq_s16(vmin2, vt2);
  }
  vmin0 = vminq_s16(vmin0, vmin1);
  vmin0 = vminq_s16(vmin0, vmin2);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const int16x8_t vi = vreinterpretq_s16_u16(vld1q_u16(i)); i += 8;
    const int16x8_t vt = veorq_s16(vi, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vi, 15)), vsign_mask)));
    vmin0 = vminq_s16(vmin0, vt);
  }
  int16x4_t vmin_lo = vmin_s16(vget_low_s16(vmin0), vget_high_s16(vmin0));

  if (XNN_UNLIKELY(batch != 0)) {
    const int16x8_t vi = vreinterpretq_s16_u16(vld1q_u16(i));
    const int16x8_t vt = veorq_s16(vi, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vi, 15)), vsign_mask)));
    int16x4_t vt_lo = vget_low_s16(vt);
    if (batch & (4 * sizeof(uint16_t))) {
      vmin_lo = vmin_s16(vmin_lo, vt_lo);
      vt_lo = vget_high_s16(vt);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmin_lo = vmin_s16(vmin_lo, vext_s16(vmin_lo, vt_lo, 2));
      vt_lo = vext_s16(vt_lo, vt_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmin_lo = vmin_s16(vmin_lo, vext_s16(vmin_lo, vt_lo, 1));
    }
  }
  #if XNN_ARCH_ARM64
    int16_t vmin_s = vminv_s16(vmin_lo);
  #else
    vmin_lo = vpmin_s16(vmin_lo, vmin_lo);
    vmin_lo = vpmin_s16(vmin_lo, vmin_lo);
    int16_t vmin_s = vget_lane_s16(vmin_lo, 0);
  #endif
  o[0] = (uint16_t)(vmin_s ^ ((vmin_s >> 15) & 0x7FFF));
}
