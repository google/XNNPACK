// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-velu/rvvfp16arith-rr1-p3.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_velu_ukernel__rvvfp16arith_rr1_p3_u1v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_elu_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_float16 vprescale = params->scalar.prescale;
  const xnn_float16 valpha = params->scalar.alpha;
  const xnn_float16 vbeta = params->scalar.beta;

  const xnn_float16 vsat_cutoff = -0x1.0A4p+3f;
  const xnn_float16 vmagic_bias = 0x1.83Cp+10f;
  const xnn_float16 vlog2e = 0x1.714p+0f;
  const xnn_float16 vminus_ln2 = -0x1.630p-1f;
  const xnn_float16 vc3 = 0x1.56Cp-3f;
  const xnn_float16 vc2 = 0x1.020p-1f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;
  do {
    const size_t n = __riscv_vsetvl_e16m1(batch);

    vfloat16m1_t vx = __riscv_vle16_v_f16m1(input, n);
    input += n;

    // Compute reduced argument z = max(prescale * x, sat_cutoff).
    vfloat16m1_t vz = __riscv_vfmul(vx, vprescale, n);
    vz = __riscv_vfmax(vz, vsat_cutoff, n);

    // Compute n = round(z / ln(2)) using magic bias.
    vfloat16m1_t vn = __riscv_vfmv_v_f_f16m1(vmagic_bias, n);
    vn = __riscv_vfmacc(vn, vlog2e, vz, n);

    // Create 2^n by shifting n into the exponent field.
    vfloat16m1_t vs = __riscv_vreinterpret_f16m1(
        __riscv_vsll(__riscv_vreinterpret_i16m1(vn), 10, n));

    // Subtract magic bias.
    vn = __riscv_vfsub(vn, vmagic_bias, n);

    // Compute reduced argument t = z - n * ln(2).
    vfloat16m1_t vt = __riscv_vfmacc(vz, vminus_ln2, vn, n);

    // Compute degree-3 polynomial for exp(t) - 1.
    vfloat16m1_t vp = __riscv_vfmv_v_f_f16m1(vc2, n);
    vp = __riscv_vfmacc(vp, vc3, vt, n);
    vp = __riscv_vfmul(vp, vt, n);

    // Reconstruct exp(z): t*s, s-1, p*t+t, (p+s-1)*(-alpha)
    vt = __riscv_vfmul(vt, vs, n);
    vs = __riscv_vfsub(vs, 1.0f, n);
    vp = __riscv_vfmadd(vp, vt, vt, n);
    vfloat16m1_t ve = __riscv_vfmul(
        __riscv_vfadd(vp, vs, n), valpha, n);

    // Select: y = x < 0 ? e : x * beta
    vfloat16m1_t vy = __riscv_vfmul(vx, vbeta, n);
    vbool16_t mask = __riscv_vmflt(vx, 0.0f, n);
    vy = __riscv_vmerge(vy, ve, mask, n);

    __riscv_vse16(output, vy, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
