// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/rvv-rr2-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vsigmoid_ukernel__rvv_rr2_p5_div_u8v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vmagic_bias = 0x1.8000FEp23f;
  const float vminus_log2e = -0x1.715476p0f;
  const float vln2_hi = 0x1.62E400p-1f;
  const float vln2_lo = 0x1.7F7D1Cp-20f;
  const float vc5 = -0x1.0F9F9Cp-7f;
  const float vc4 = 0x1.573A1Ap-5f;
  const float vc3 = -0x1.555A80p-3f;
  const float vc2 = 0x1.FFFDC6p-2f;
  const float vc1 = -0x1.FFFFF6p-1f;
  const float vone = 1.0f;
  const float vdenorm_cutoff = 0x1.5D589Ep+6f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  do {
    size_t vl = __riscv_vsetvl_e32m8(batch); batch -= vl;
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, vl); input += vl;

    vfloat32m8_t vz = __riscv_vfabs(vx, vl);

    vfloat32m8_t vn = __riscv_vfadd(__riscv_vfmul(vz, vminus_log2e, vl), vmagic_bias, vl);

    vfloat32m8_t vs = __riscv_vreinterpret_f32m8(__riscv_vsll(__riscv_vreinterpret_u32m8(vn), 23, vl));
    vn = __riscv_vfsub(vn, vmagic_bias, vl);

    vfloat32m8_t vt = __riscv_vfadd(__riscv_vfmul(vn, vln2_hi, vl), vz, vl);
    vt = __riscv_vfmacc(vt, vln2_lo, vn, vl);

    vfloat32m8_t vp = __riscv_vfadd(__riscv_vfmul(vt, vc5, vl), vc4, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vt, vp, vl), vc3, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vt, vp, vl), vc2, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vt, vp, vl), vc1, vl);

    vt = __riscv_vfmul(vt, vs, vl);
    vfloat32m8_t ve = __riscv_vfadd(__riscv_vfmul(vt, vp, vl), vs, vl);
    vfloat32m8_t vd = __riscv_vfadd(ve, vone, vl);

    vfloat32m8_t vf = __riscv_vfdiv(ve, vd, vl);

    vbool4_t vmask = __riscv_vmfgt(vz, vdenorm_cutoff, vl);
    vf = __riscv_vfmerge(vf, 0.0f, vmask, vl);
    vmask = __riscv_vmfgt(vx, 0.0f, vl);
    vf = __riscv_vfrsub_mu(vmask, vf, vf, vone, vl);

    __riscv_vse32(output, vf, vl); output += vl;

  } while (batch != 0);
}
