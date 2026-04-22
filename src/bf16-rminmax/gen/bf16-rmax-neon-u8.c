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


void xnn_bf16_rmax_ukernel__neon_u8(
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

  int16x8_t vmax0 = vreinterpretq_s16_u16(vld1q_dup_u16((uint16_t*)((uintptr_t) o + 0 * sizeof(uint16_t))));
  vmax0 = veorq_s16(vmax0, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vmax0, 15)), vsign_mask)));
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const int16x8_t vi = vreinterpretq_s16_u16(vld1q_u16(i)); i += 8;
    const int16x8_t vt = veorq_s16(vi, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vi, 15)), vsign_mask)));
    vmax0 = vmaxq_s16(vmax0, vt);
  }
  int16x4_t vmax_lo = vmax_s16(vget_low_s16(vmax0), vget_high_s16(vmax0));

  if (XNN_UNLIKELY(batch != 0)) {
    const int16x8_t vi = vreinterpretq_s16_u16(vld1q_u16(i));
    const int16x8_t vt = veorq_s16(vi, vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(vshrq_n_s16(vi, 15)), vsign_mask)));
    int16x4_t vt_lo = vget_low_s16(vt);
    if (batch & (4 * sizeof(uint16_t))) {
      vmax_lo = vmax_s16(vmax_lo, vt_lo);
      vt_lo = vget_high_s16(vt);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmax_lo = vmax_s16(vmax_lo, vext_s16(vmax_lo, vt_lo, 2));
      vt_lo = vext_s16(vt_lo, vt_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmax_lo = vmax_s16(vmax_lo, vext_s16(vmax_lo, vt_lo, 1));
    }
  }
  #if XNN_ARCH_ARM64
    int16_t vmax_s = vmaxv_s16(vmax_lo);
  #else
    vmax_lo = vpmax_s16(vmax_lo, vmax_lo);
    vmax_lo = vpmax_s16(vmax_lo, vmax_lo);
    int16_t vmax_s = vget_lane_s16(vmax_lo, 0);
  #endif
  o[0] = (uint16_t)(vmax_s ^ ((vmax_s >> 15) & 0x7FFF));
}
