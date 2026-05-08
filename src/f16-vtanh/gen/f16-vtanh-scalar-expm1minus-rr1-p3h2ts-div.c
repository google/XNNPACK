// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/simd-expm1minus.c.in
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
#include "src/xnnpack/simd/f16-scalar.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vtanh_ukernel__scalar_expm1minus_rr1_p3h2ts_div_u1(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vsat_cutoff, 0x1.208p+2f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic_bias, 0x1.83Cp+9f); // 0x620F
  XNN_SIMD_CONST_F16_FROM_FLOAT(vminus_log2e, -0x1.714p+0f); // 0xBDC5
  XNN_SIMD_CONST_F16_FROM_FLOAT(vln2, 0x1.630p-1f); // 0x398C
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc3, -0x1.56Cp+0f); // 0xBD5B
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc2, 0x1.020p+1f); // 0x4008
  XNN_SIMD_CONST_F16_FROM_FLOAT(vtwo, 2.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vminus_one, -1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vsign_mask, -0.0f);

  const xnn_float16* i = input;
  xnn_float16* o = output;
  for (; batch >= 1 * xnn_simd_bytes_f16; batch -= 1 * xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx = xnn_loadu_f16(i); i += 1;

    xnn_simd_f16_t vz = xnn_abs_f16(vx);

    vz = xnn_min_f16(vz, vsat_cutoff);

    xnn_simd_f16_t vn = xnn_fmadd_f16(vz, vminus_log2e, vmagic_bias);

    const xnn_simd_f16_t vs = xnn_sll_f16(vn, 10);
    vn = xnn_sub_f16(vn, vmagic_bias);
    const xnn_simd_f16_t vt = xnn_fmadd_f16(vn, vln2, vz);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vt, vc3, vc2);
    vp = xnn_fnmadd_f16(vp, vt, vtwo);

    const xnn_simd_f16_t vts = xnn_mul_f16(vt, vs);
    const xnn_simd_f16_t vsmo = xnn_add_f16(vs, vminus_one);
    const xnn_simd_f16_t vemo = xnn_fnmadd_f16(vp, vts, vsmo);
    const xnn_simd_f16_t vepo = xnn_add_f16(vemo, vtwo);

    xnn_simd_f16_t vy = xnn_div_f16(vemo, vepo);

    vy = xnn_xor_f16(vy, xnn_and_f16(xnn_xor_f16(vx, vy), vsign_mask));

    xnn_storeu_f16(o, vy); o += 1;
  }

}

void xnn_f16_vtanh_ukernel__scalar_expm1minus_rr1_p3h2ts_div_u2(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vsat_cutoff, 0x1.208p+2f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic_bias, 0x1.83Cp+9f); // 0x620F
  XNN_SIMD_CONST_F16_FROM_FLOAT(vminus_log2e, -0x1.714p+0f); // 0xBDC5
  XNN_SIMD_CONST_F16_FROM_FLOAT(vln2, 0x1.630p-1f); // 0x398C
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc3, -0x1.56Cp+0f); // 0xBD5B
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc2, 0x1.020p+1f); // 0x4008
  XNN_SIMD_CONST_F16_FROM_FLOAT(vtwo, 2.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vminus_one, -1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vsign_mask, -0.0f);

  const xnn_float16* i = input;
  xnn_float16* o = output;
  for (; batch >= 2 * xnn_simd_bytes_f16; batch -= 2 * xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx0 = xnn_loadu_f16(i); i += 1;
    const xnn_simd_f16_t vx1 = xnn_loadu_f16(i); i += 1;

    xnn_simd_f16_t vz0 = xnn_abs_f16(vx0);
    xnn_simd_f16_t vz1 = xnn_abs_f16(vx1);

    vz0 = xnn_min_f16(vz0, vsat_cutoff);
    vz1 = xnn_min_f16(vz1, vsat_cutoff);

    xnn_simd_f16_t vn0 = xnn_fmadd_f16(vz0, vminus_log2e, vmagic_bias);
    xnn_simd_f16_t vn1 = xnn_fmadd_f16(vz1, vminus_log2e, vmagic_bias);

    const xnn_simd_f16_t vs0 = xnn_sll_f16(vn0, 10);
    vn0 = xnn_sub_f16(vn0, vmagic_bias);
    const xnn_simd_f16_t vs1 = xnn_sll_f16(vn1, 10);
    vn1 = xnn_sub_f16(vn1, vmagic_bias);
    const xnn_simd_f16_t vt0 = xnn_fmadd_f16(vn0, vln2, vz0);
    const xnn_simd_f16_t vt1 = xnn_fmadd_f16(vn1, vln2, vz1);
    xnn_simd_f16_t vp0 = xnn_fmadd_f16(vt0, vc3, vc2);
    xnn_simd_f16_t vp1 = xnn_fmadd_f16(vt1, vc3, vc2);
    vp0 = xnn_fnmadd_f16(vp0, vt0, vtwo);
    vp1 = xnn_fnmadd_f16(vp1, vt1, vtwo);

    const xnn_simd_f16_t vts0 = xnn_mul_f16(vt0, vs0);
    const xnn_simd_f16_t vsmo0 = xnn_add_f16(vs0, vminus_one);
    const xnn_simd_f16_t vts1 = xnn_mul_f16(vt1, vs1);
    const xnn_simd_f16_t vsmo1 = xnn_add_f16(vs1, vminus_one);
    const xnn_simd_f16_t vemo0 = xnn_fnmadd_f16(vp0, vts0, vsmo0);
    const xnn_simd_f16_t vemo1 = xnn_fnmadd_f16(vp1, vts1, vsmo1);
    const xnn_simd_f16_t vepo0 = xnn_add_f16(vemo0, vtwo);
    const xnn_simd_f16_t vepo1 = xnn_add_f16(vemo1, vtwo);

    xnn_simd_f16_t vy0 = xnn_div_f16(vemo0, vepo0);
    xnn_simd_f16_t vy1 = xnn_div_f16(vemo1, vepo1);

    vy0 = xnn_xor_f16(vy0, xnn_and_f16(xnn_xor_f16(vx0, vy0), vsign_mask));
    vy1 = xnn_xor_f16(vy1, xnn_and_f16(xnn_xor_f16(vx1, vy1), vsign_mask));

    xnn_storeu_f16(o, vy0); o += 1;
    xnn_storeu_f16(o, vy1); o += 1;
  }
  for (; batch >= 1 * xnn_simd_bytes_f16; batch -= 1 * xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx = xnn_loadu_f16(i); i += 1;

    xnn_simd_f16_t vz = xnn_abs_f16(vx);

    vz = xnn_min_f16(vz, vsat_cutoff);

    xnn_simd_f16_t vn = xnn_fmadd_f16(vz, vminus_log2e, vmagic_bias);

    const xnn_simd_f16_t vs = xnn_sll_f16(vn, 10);
    vn = xnn_sub_f16(vn, vmagic_bias);
    const xnn_simd_f16_t vt = xnn_fmadd_f16(vn, vln2, vz);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vt, vc3, vc2);
    vp = xnn_fnmadd_f16(vp, vt, vtwo);

    const xnn_simd_f16_t vts = xnn_mul_f16(vt, vs);
    const xnn_simd_f16_t vsmo = xnn_add_f16(vs, vminus_one);
    const xnn_simd_f16_t vemo = xnn_fnmadd_f16(vp, vts, vsmo);
    const xnn_simd_f16_t vepo = xnn_add_f16(vemo, vtwo);

    xnn_simd_f16_t vy = xnn_div_f16(vemo, vepo);

    vy = xnn_xor_f16(vy, xnn_and_f16(xnn_xor_f16(vx, vy), vsign_mask));

    xnn_storeu_f16(o, vy); o += 1;
  }

}

void xnn_f16_vtanh_ukernel__scalar_expm1minus_rr1_p3h2ts_div_u4(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vsat_cutoff, 0x1.208p+2f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic_bias, 0x1.83Cp+9f); // 0x620F
  XNN_SIMD_CONST_F16_FROM_FLOAT(vminus_log2e, -0x1.714p+0f); // 0xBDC5
  XNN_SIMD_CONST_F16_FROM_FLOAT(vln2, 0x1.630p-1f); // 0x398C
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc3, -0x1.56Cp+0f); // 0xBD5B
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc2, 0x1.020p+1f); // 0x4008
  XNN_SIMD_CONST_F16_FROM_FLOAT(vtwo, 2.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vminus_one, -1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vsign_mask, -0.0f);

  const xnn_float16* i = input;
  xnn_float16* o = output;
  for (; batch >= 4 * xnn_simd_bytes_f16; batch -= 4 * xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx0 = xnn_loadu_f16(i); i += 1;
    const xnn_simd_f16_t vx1 = xnn_loadu_f16(i); i += 1;
    const xnn_simd_f16_t vx2 = xnn_loadu_f16(i); i += 1;
    const xnn_simd_f16_t vx3 = xnn_loadu_f16(i); i += 1;

    xnn_simd_f16_t vz0 = xnn_abs_f16(vx0);
    xnn_simd_f16_t vz1 = xnn_abs_f16(vx1);
    xnn_simd_f16_t vz2 = xnn_abs_f16(vx2);
    xnn_simd_f16_t vz3 = xnn_abs_f16(vx3);

    vz0 = xnn_min_f16(vz0, vsat_cutoff);
    vz1 = xnn_min_f16(vz1, vsat_cutoff);
    vz2 = xnn_min_f16(vz2, vsat_cutoff);
    vz3 = xnn_min_f16(vz3, vsat_cutoff);

    xnn_simd_f16_t vn0 = xnn_fmadd_f16(vz0, vminus_log2e, vmagic_bias);
    xnn_simd_f16_t vn1 = xnn_fmadd_f16(vz1, vminus_log2e, vmagic_bias);
    xnn_simd_f16_t vn2 = xnn_fmadd_f16(vz2, vminus_log2e, vmagic_bias);
    xnn_simd_f16_t vn3 = xnn_fmadd_f16(vz3, vminus_log2e, vmagic_bias);

    const xnn_simd_f16_t vs0 = xnn_sll_f16(vn0, 10);
    vn0 = xnn_sub_f16(vn0, vmagic_bias);
    const xnn_simd_f16_t vs1 = xnn_sll_f16(vn1, 10);
    vn1 = xnn_sub_f16(vn1, vmagic_bias);
    const xnn_simd_f16_t vs2 = xnn_sll_f16(vn2, 10);
    vn2 = xnn_sub_f16(vn2, vmagic_bias);
    const xnn_simd_f16_t vs3 = xnn_sll_f16(vn3, 10);
    vn3 = xnn_sub_f16(vn3, vmagic_bias);
    const xnn_simd_f16_t vt0 = xnn_fmadd_f16(vn0, vln2, vz0);
    const xnn_simd_f16_t vt1 = xnn_fmadd_f16(vn1, vln2, vz1);
    const xnn_simd_f16_t vt2 = xnn_fmadd_f16(vn2, vln2, vz2);
    const xnn_simd_f16_t vt3 = xnn_fmadd_f16(vn3, vln2, vz3);
    xnn_simd_f16_t vp0 = xnn_fmadd_f16(vt0, vc3, vc2);
    xnn_simd_f16_t vp1 = xnn_fmadd_f16(vt1, vc3, vc2);
    xnn_simd_f16_t vp2 = xnn_fmadd_f16(vt2, vc3, vc2);
    xnn_simd_f16_t vp3 = xnn_fmadd_f16(vt3, vc3, vc2);
    vp0 = xnn_fnmadd_f16(vp0, vt0, vtwo);
    vp1 = xnn_fnmadd_f16(vp1, vt1, vtwo);
    vp2 = xnn_fnmadd_f16(vp2, vt2, vtwo);
    vp3 = xnn_fnmadd_f16(vp3, vt3, vtwo);

    const xnn_simd_f16_t vts0 = xnn_mul_f16(vt0, vs0);
    const xnn_simd_f16_t vsmo0 = xnn_add_f16(vs0, vminus_one);
    const xnn_simd_f16_t vts1 = xnn_mul_f16(vt1, vs1);
    const xnn_simd_f16_t vsmo1 = xnn_add_f16(vs1, vminus_one);
    const xnn_simd_f16_t vts2 = xnn_mul_f16(vt2, vs2);
    const xnn_simd_f16_t vsmo2 = xnn_add_f16(vs2, vminus_one);
    const xnn_simd_f16_t vts3 = xnn_mul_f16(vt3, vs3);
    const xnn_simd_f16_t vsmo3 = xnn_add_f16(vs3, vminus_one);
    const xnn_simd_f16_t vemo0 = xnn_fnmadd_f16(vp0, vts0, vsmo0);
    const xnn_simd_f16_t vemo1 = xnn_fnmadd_f16(vp1, vts1, vsmo1);
    const xnn_simd_f16_t vemo2 = xnn_fnmadd_f16(vp2, vts2, vsmo2);
    const xnn_simd_f16_t vemo3 = xnn_fnmadd_f16(vp3, vts3, vsmo3);
    const xnn_simd_f16_t vepo0 = xnn_add_f16(vemo0, vtwo);
    const xnn_simd_f16_t vepo1 = xnn_add_f16(vemo1, vtwo);
    const xnn_simd_f16_t vepo2 = xnn_add_f16(vemo2, vtwo);
    const xnn_simd_f16_t vepo3 = xnn_add_f16(vemo3, vtwo);

    xnn_simd_f16_t vy0 = xnn_div_f16(vemo0, vepo0);
    xnn_simd_f16_t vy1 = xnn_div_f16(vemo1, vepo1);
    xnn_simd_f16_t vy2 = xnn_div_f16(vemo2, vepo2);
    xnn_simd_f16_t vy3 = xnn_div_f16(vemo3, vepo3);

    vy0 = xnn_xor_f16(vy0, xnn_and_f16(xnn_xor_f16(vx0, vy0), vsign_mask));
    vy1 = xnn_xor_f16(vy1, xnn_and_f16(xnn_xor_f16(vx1, vy1), vsign_mask));
    vy2 = xnn_xor_f16(vy2, xnn_and_f16(xnn_xor_f16(vx2, vy2), vsign_mask));
    vy3 = xnn_xor_f16(vy3, xnn_and_f16(xnn_xor_f16(vx3, vy3), vsign_mask));

    xnn_storeu_f16(o, vy0); o += 1;
    xnn_storeu_f16(o, vy1); o += 1;
    xnn_storeu_f16(o, vy2); o += 1;
    xnn_storeu_f16(o, vy3); o += 1;
  }
  for (; batch >= 1 * xnn_simd_bytes_f16; batch -= 1 * xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx = xnn_loadu_f16(i); i += 1;

    xnn_simd_f16_t vz = xnn_abs_f16(vx);

    vz = xnn_min_f16(vz, vsat_cutoff);

    xnn_simd_f16_t vn = xnn_fmadd_f16(vz, vminus_log2e, vmagic_bias);

    const xnn_simd_f16_t vs = xnn_sll_f16(vn, 10);
    vn = xnn_sub_f16(vn, vmagic_bias);
    const xnn_simd_f16_t vt = xnn_fmadd_f16(vn, vln2, vz);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vt, vc3, vc2);
    vp = xnn_fnmadd_f16(vp, vt, vtwo);

    const xnn_simd_f16_t vts = xnn_mul_f16(vt, vs);
    const xnn_simd_f16_t vsmo = xnn_add_f16(vs, vminus_one);
    const xnn_simd_f16_t vemo = xnn_fnmadd_f16(vp, vts, vsmo);
    const xnn_simd_f16_t vepo = xnn_add_f16(vemo, vtwo);

    xnn_simd_f16_t vy = xnn_div_f16(vemo, vepo);

    vy = xnn_xor_f16(vy, xnn_and_f16(xnn_xor_f16(vx, vy), vsign_mask));

    xnn_storeu_f16(o, vy); o += 1;
  }

}
