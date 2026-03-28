// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-raddstoreexpminusmax/rvvfp16arith-rr2-p2.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/raddstoreexpminusmax.h"


void xnn_f16_raddstoreexpminusmax_ukernel__rvvfp16arith_rr2_p2_u2v(
    size_t batch,
    const xnn_float16* input,
    const xnn_float16* max,
    xnn_float16* output,
    float* sum,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const xnn_float16 vlog2e = 0x1.715476p0f;
  const xnn_float16 vmagic_bias = 0x1.83Cp+10f;
  const xnn_float16 vminus_ln2_hi = -0x1.630p-1f;
  const xnn_float16 vminus_ln2_lo = 0x1.BD0p-13f;
  const xnn_float16 vc2 = 0x1.FF3A32p-2f;
  const xnn_float16 vc1 = 0x1.039E10p+0f;
  const xnn_float16 vdenorm_cutoff = -0x1.368000p+3f;

  const xnn_float16* i = input;
  xnn_float16* o = output;

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;

  size_t vlmax = __riscv_vsetvl_e16m2(batch);
  vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

  do {
    size_t vl = __riscv_vsetvl_e16m2(batch); batch -= vl;

    vfloat16m2_t vi = __riscv_vle16_v_f16m2(i, vl); i += vl;

    const vfloat16m2_t vx = __riscv_vfsub(vi, *max, vl);

    vfloat16m2_t vn = __riscv_vfmv_v_f_f16m2(vmagic_bias, vl);
    vn = __riscv_vfmacc(vn, vlog2e, vx, vl);

    const vfloat16m2_t vs = __riscv_vreinterpret_f16m2(__riscv_vsll(__riscv_vreinterpret_i16m2(vn), 10, vl));

    vn = __riscv_vfsub(vn, vmagic_bias, vl);

    vfloat16m2_t vt = __riscv_vmv_v(vx, vl);
    vt = __riscv_vfmacc(vt, vminus_ln2_hi, vn, vl);
    vt = __riscv_vfmacc(vt, vminus_ln2_lo, vn, vl);

    vfloat16m2_t vp = __riscv_vfmv_v_f_f16m2(vc1, vl);
    vp = __riscv_vfmacc(vp, vc2, vt, vl);

    vt = __riscv_vfmul(vt, vs, vl);

    vfloat16m2_t vf = __riscv_vmv_v(vs, vl);
    vf = __riscv_vfmacc(vf, vp, vt, vl);

    const vbool8_t vmask = __riscv_vmflt(vx, vdenorm_cutoff, vl);
    vf = __riscv_vfmerge(vf, 0.0f, vmask, vl);

    __riscv_vse16(o, vf, vl); o += vl;

    vacc = __riscv_vfwadd_wv(vacc, vf, vl);
  } while (batch > 0);

  vfloat32m1_t v0 = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  *sum = __riscv_vfmv_f(__riscv_vfredusum(vacc, v0, vlmax));
}
