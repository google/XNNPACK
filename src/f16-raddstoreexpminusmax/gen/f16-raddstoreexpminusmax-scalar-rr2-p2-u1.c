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
#include "src/xnnpack/simd/f16-scalar.h"


static XNN_INLINE xnn_simd_f16_t setexp_f16(xnn_simd_f16_t vx) {
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic, 1039.0f);
  return xnn_sll_f16(xnn_add_f16(vx, vmagic), 10);
}

static XNN_INLINE xnn_simd_f16_t qd_round_f16(xnn_simd_f16_t vx) {
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic, 1536.0f);
  return xnn_sub_f16(xnn_add_f16(vmagic, vx), vmagic);
}

static XNN_INLINE xnn_simd_f16_t expminusmax_f16(
    xnn_simd_f16_t vx, xnn_simd_f16_t vmax)
{
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_1, 0.6933594f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_2, 0.24255371f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(valpha_3, 0.05517578f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vlog2e, 1.4423828f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(v16, 16.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vm15, -15.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);

  vx = xnn_sub_f16(vx, vmax);
  xnn_simd_f16_t vz_prime = xnn_mul_f16(vx, vlog2e);
  vz_prime = xnn_min_f16(xnn_max_f16(vz_prime, vm15), v16);
  const xnn_simd_f16_t vz = qd_round_f16(vz_prime);
  const xnn_simd_f16_t vr = xnn_sub_f16(vz_prime, vz);
  const xnn_simd_f16_t v2z = setexp_f16(vz);

  xnn_simd_f16_t v2r = xnn_fmadd_f16(vr, valpha_3, valpha_2);
  v2r = xnn_fmadd_f16(vr, v2r, valpha_1);
  v2r = xnn_fmadd_f16(vr, v2r, vone);
  return xnn_mul_f16(v2z, v2r);
}

void xnn_f16_raddstoreexpminusmax_ukernel__scalar_rr2_p2_u1(
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
  float vsum = 0.0f;
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vf = expminusmax_f16(xnn_loadu_f16(input), vmax);
    input += xnn_simd_size_f16;
    xnn_storeu_f16(output, vf);
    output += xnn_simd_size_f16;
    vsum += xnn_reduce_add_f16(vf);
  }

  *sum = vsum;
}
