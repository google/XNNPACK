// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-raddstoreexpminusmax/f16-f32acc-rr2-p5.c.in
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

#include "src/xnnpack/simd/f32-wasmrelaxedsimd.h"

#undef XNN_SIMD_HAS_NATIVE_FMA
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"


static XNN_INLINE xnn_simd_f32_t load_f16_f32(const xnn_float16* input) {
  return xnn_cvt_f32_f16(xnn_loadu_f16(input));
}

static XNN_INLINE xnn_simd_f32_t load_tail_f16_f32(
    const xnn_float16* input, size_t elements)
{
  return xnn_cvt_f32_f16(xnn_load_tail_f16(input, elements));
}

static XNN_INLINE xnn_simd_f32_t store_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value)
{
  const xnn_simd_f16_t rounded = xnn_cvt_f16_f32(value);
  xnn_store_tail_f16(output, rounded, xnn_simd_size_f32);
  return xnn_cvt_f32_f16(rounded);
}

static XNN_INLINE xnn_simd_f32_t store_tail_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value, size_t elements)
{
  const xnn_simd_f16_t rounded = xnn_cvt_f16_f32(value);
  xnn_store_tail_f16(output, rounded, elements);
  return xnn_cvt_f32_f16(rounded);
}

static XNN_INLINE xnn_simd_f32_t expminusmax_f32(
    xnn_simd_f32_t vx, xnn_simd_f32_t vmax)
{
  XNN_SIMD_CONST_F32(vlog2e, 0x1.715476p+0f);
  XNN_SIMD_CONST_F32(vmagic_bias, 0x1.8000FEp23f);
  XNN_SIMD_CONST_F32(vminus_ln2_hi, -0x1.62E400p-1f);
  XNN_SIMD_CONST_F32(vminus_ln2_lo, -0x1.7F7D1Cp-20f);
  XNN_SIMD_CONST_F32(vc5, 0x1.0F9F9Cp-7f);
  XNN_SIMD_CONST_F32(vc4, 0x1.573A1Ap-5f);
  XNN_SIMD_CONST_F32(vc3, 0x1.555A80p-3f);
  XNN_SIMD_CONST_F32(vc2, 0x1.FFFDC6p-2f);
  XNN_SIMD_CONST_F32(vc1, 0x1.FFFFF6p-1f);
  XNN_SIMD_CONST_F32(vdenorm_cutoff, -0x1.5D589Ep6f);

  vx = xnn_max_f32(xnn_sub_f32(vx, vmax), vdenorm_cutoff);
  xnn_simd_f32_t vn = xnn_fmadd_f32(vx, vlog2e, vmagic_bias);
  const xnn_simd_f32_t vs = xnn_sll_f32(vn, 23);
  vn = xnn_sub_f32(vn, vmagic_bias);

  xnn_simd_f32_t vt = xnn_fmadd_f32(vn, vminus_ln2_hi, vx);
  vt = xnn_fmadd_f32(vn, vminus_ln2_lo, vt);

  xnn_simd_f32_t vp = xnn_fmadd_f32(vc5, vt, vc4);
  vp = xnn_fmadd_f32(vp, vt, vc3);
  vp = xnn_fmadd_f32(vp, vt, vc2);
  vp = xnn_fmadd_f32(vp, vt, vc1);

  vt = xnn_mul_f32(vt, vs);
  return xnn_fmadd_f32(vt, vp, vs);
}

void xnn_f16_f32acc_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4(
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

  const xnn_simd_f32_t vmax = xnn_set1_f32(xnn_float16_to_float(*max));
  xnn_simd_f32_t vacc = xnn_zero_f32();
  for (; batch >= 4 * sizeof(xnn_float16);
       batch -= 4 * sizeof(xnn_float16)) {
    const xnn_simd_f32_t vf = expminusmax_f32(load_f16_f32(input), vmax);
    input += 4;
    vacc = xnn_add_f32(vacc, store_f32_f16(output, vf));
    output += 4;
  }

  float vsum = xnn_reduce_add_f32(vacc);
  if XNN_UNLIKELY(batch != 0) {
    const size_t num_elements = batch / sizeof(xnn_float16);
    const xnn_simd_f32_t vf =
        expminusmax_f32(load_tail_f16_f32(input, num_elements), vmax);
    const xnn_simd_f32_t rounded =
        store_tail_f32_f16(output, vf, num_elements);

    XNN_ALIGN(128) float values[xnn_simd_size_f32];
    xnn_storeu_f32(values, rounded);
    for (size_t i = 0; i < num_elements; i++) {
      vsum += values[i];
    }
  }
  *sum = vsum;
}
