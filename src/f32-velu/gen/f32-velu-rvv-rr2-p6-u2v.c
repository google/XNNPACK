// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-velu/rvv-rr2-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_velu_ukernel__rvv_rr2_p6_u2v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_elu_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vprescale = params->scalar.prescale;
  const float valpha = params->scalar.alpha;
  const float vbeta = params->scalar.beta;

  const float vsat_cutoff = -0x1.154246p+4f;
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vlog2e = 0x1.715476p+0f;
  const float vminus_ln2_hi = -0x1.62E440p-1f;
  const float vminus_ln2_lo = 0x1.0105C6p-21f;
  const float vc6 = 0x1.6b7338p-10f;
  const float vc5 = 0x1.12278Ep-7f;
  const float vc4 = 0x1.555716p-5f;
  const float vc3 = 0x1.5554B0p-3f;
  const float vc2 = 0x1.FFFFFEp-2f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m2(batch);

    vfloat32m2_t vx = __riscv_vle32_v_f32m2(input, n);
    input += n;

    // Compute reduced argument z = max(prescale * x, sat_cutoff).
    vfloat32m2_t vz = __riscv_vfmul_vf_f32m2(vx, vprescale, n);
    vz = __riscv_vfmax_vf_f32m2(vz, vsat_cutoff, n);

    // Compute reduced argument n = round(z / ln(2)).
    // Use magic bias to get rounding for free.
    vfloat32m2_t vn = __riscv_vfmacc_vf_f32m2(
        __riscv_vfmv_v_f_f32m2(vmagic_bias, n), vlog2e, vz, n);

    // Create 2^n by shifting n (as integer) into the exponent field.
    vint32m2_t ven = __riscv_vsll_vx_i32m2(
        __riscv_vreinterpret_v_f32m2_i32m2(vn), 23, n);
    vfloat32m2_t vs = __riscv_vreinterpret_v_i32m2_f32m2(ven);

    // Subtract magic bias to get the reduced argument.
    vn = __riscv_vfsub_vf_f32m2(vn, vmagic_bias, n);

    // Compute reduced argument t = z - n * ln(2).
    // Use Cody-Waite range reduction (two constants to represent ln(2)).
    vfloat32m2_t vt = __riscv_vfmacc_vf_f32m2(vz, vminus_ln2_hi, vn, n);
    vt = __riscv_vfmacc_vf_f32m2(vt, vminus_ln2_lo, vn, n);

    // Compute degree-6 polynomial approximation for exp(t) - 1 using Horner's method.
    //   p = t * (t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    vfloat32m2_t vp = __riscv_vfmv_v_f_f32m2(vc5, n);
    vp = __riscv_vfmacc_vf_f32m2(vp, vc6, vt, n);
    vp = __riscv_vfmadd_vv_f32m2(vp, vt, __riscv_vfmv_v_f_f32m2(vc4, n), n);
    vp = __riscv_vfmadd_vv_f32m2(vp, vt, __riscv_vfmv_v_f_f32m2(vc3, n), n);
    vp = __riscv_vfmadd_vv_f32m2(vp, vt, __riscv_vfmv_v_f_f32m2(vc2, n), n);
    vp = __riscv_vfmul_vv_f32m2(vp, vt, n);

    // Reconstruct the exp(z) value:
    //   t * s
    //   s - 1
    //   p = (p * t) + t
    //   e = (p + (s - 1)) * alpha
    vt = __riscv_vfmul_vv_f32m2(vt, vs, n);
    vs = __riscv_vfsub_vf_f32m2(vs, 1.0f, n);
    vp = __riscv_vfmadd_vv_f32m2(vp, vt, vt, n);
    vfloat32m2_t ve = __riscv_vfmul_vf_f32m2(
        __riscv_vfadd_vv_f32m2(vp, vs, n), valpha, n);

    // Select between the ELU and linear parts:
    //   y = x < 0 ? e : x * beta
    vfloat32m2_t vy = __riscv_vfmul_vf_f32m2(vx, vbeta, n);
    vbool16_t mask = __riscv_vmflt_vf_f32m2_b16(vx, 0.0f, n);
    vy = __riscv_vmerge_vvm_f32m2(vy, ve, mask, n);

    __riscv_vse32_v_f32m2(output, vy, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
