// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-raddstoreexpminusmax/rr2-p2.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/raddstoreexpminusmax.h"
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"


static XNN_INLINE xnn_simd_f16_t expminusmax_f16(
    xnn_simd_f16_t vx, xnn_simd_f16_t vmax)
{
  XNN_SIMD_CONST_F16_FROM_INT16(vlog2e, 0x3DC5);
  XNN_SIMD_CONST_F16_FROM_INT16(vmagic_bias, 0x660F);
  XNN_SIMD_CONST_F16_FROM_INT16(vminus_ln2_hi, 0xB98C);
  XNN_SIMD_CONST_F16_FROM_INT16(vminus_ln2_lo, 0x0AF4);
  XNN_SIMD_CONST_F16_FROM_INT16(vc2, 0x37F9);
  XNN_SIMD_CONST_F16_FROM_INT16(vc1, 0x3C0E);
  XNN_SIMD_CONST_F16_FROM_INT16(vdenorm_cutoff, 0xC8DA);

  vx = xnn_sub_f16(vx, vmax);
  xnn_simd_f16_t vn = xnn_fmadd_f16(vx, vlog2e, vmagic_bias);
  const xnn_simd_f16_t vs = xnn_sll_f16(vn, 10);
  vn = xnn_sub_f16(vn, vmagic_bias);

  xnn_simd_f16_t vt = xnn_fmadd_f16(vn, vminus_ln2_hi, vx);
  vt = xnn_fmadd_f16(vn, vminus_ln2_lo, vt);
  const xnn_simd_f16_t vp = xnn_fmadd_f16(vc2, vt, vc1);
  vt = xnn_mul_f16(vt, vs);

  const xnn_simd_f16_t vf = xnn_fmadd_f16(vp, vt, vs);
  return xnn_andnot_f16(xnn_cmplt_f16(vx, vdenorm_cutoff), vf);
}

void xnn_f16_raddstoreexpminusmax_ukernel__wasmrelaxedsimdfp16_rr2_p2_u8(
    size_t batch,
    const xnn_float16* input,
    const xnn_float16* max,
    xnn_float16* output,
    float* sum,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const xnn_simd_f16_t vmax = xnn_set1_f16(*max);
  xnn_simd_f16_accumulator_t vacc = xnn_zero_f16_accumulator();
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vf = expminusmax_f16(xnn_loadu_f16(input), vmax);
    input += xnn_simd_size_f16;
    xnn_storeu_f16(output, vf);
    output += xnn_simd_size_f16;
    vacc = xnn_accumulate_f16(vacc, vf);
  }

  float vsum = xnn_reduce_f16_accumulator(vacc);
  if XNN_UNLIKELY(batch != 0) {
    const size_t num_elements = batch / sizeof(xnn_float16);
    const xnn_simd_f16_t vf =
        expminusmax_f16(xnn_load_tail_f16(input, num_elements), vmax);
    xnn_store_tail_f16(output, vf, num_elements);

    XNN_ALIGN(64) xnn_float16 values[xnn_simd_size_f16];
    xnn_storeu_f16(values, vf);
    for (size_t i = 0; i < num_elements; i++) {
      vsum += xnn_float16_to_float(values[i]);
    }
  }
  *sum = vsum;
}
