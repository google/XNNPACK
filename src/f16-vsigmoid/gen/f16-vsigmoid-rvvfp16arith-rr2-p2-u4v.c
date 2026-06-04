// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsigmoid/rvvfp16arith-rr2-p2.c.in
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


void xnn_f16_vsigmoid_ukernel__rvvfp16arith_rr2_p2_u4v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_float16 vmagic_bias = 0x1.83Cp+10f;
  const xnn_float16 vminus_log2e = -0x1.714p+0f;
  const xnn_float16 vln2_hi = 0x1.630p-1f;
  const xnn_float16 vln2_lo = -0x1.BD0p-13f;
  const xnn_float16 vc2 = 0x1.FE4p-2f;
  const xnn_float16 vc1 = -0x1.038p+0f;
  const xnn_float16 vdenorm_cutoff = 0x1.368p+3f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;
  do {
    const size_t n = __riscv_vsetvl_e16m4(batch);

    vfloat16m4_t vx = __riscv_vle16_v_f16m4(input, n);
    input += n;

    vfloat16m4_t vz = __riscv_vfabs(vx, n);

    // Compute n = round(z * (-log2e)) + magic_bias.
    vfloat16m4_t vn = __riscv_vfmv_v_f_f16m4(vmagic_bias, n);
    vn = __riscv_vfmacc(vn, vminus_log2e, vz, n);

    // Create 2^n.
    vfloat16m4_t vs = __riscv_vreinterpret_f16m4(
        __riscv_vsll(__riscv_vreinterpret_i16m4(vn), 10, n));

    vn = __riscv_vfsub(vn, vmagic_bias, n);

    // t = z - n * ln2 (Cody-Waite).
    vfloat16m4_t vt = __riscv_vfmacc(vz, vln2_hi, vn, n);
    vt = __riscv_vfmacc(vt, vln2_lo, vn, n);

    // Polynomial: p = c1 + c2 * t.
    vfloat16m4_t vp = __riscv_vfmv_v_f_f16m4(vc1, n);
    vp = __riscv_vfmacc(vp, vc2, vt, n);

    // Reconstruct: f = s + (p * t) * s.
    vt = __riscv_vfmul(vt, vs, n);
    vfloat16m4_t vf = __riscv_vfmacc(vs, vp, vt, n);

    // Flush denorms.
    vbool4_t vdenorm_mask = __riscv_vmfgt(vz, vdenorm_cutoff, n);
    vf = __riscv_vfmerge(vf, 0.0f, vdenorm_mask, n);

    // sigmoid = f / (1 + f).
    vfloat16m4_t vy = __riscv_vfdiv(vf, __riscv_vfadd(vf, 1.0f, n), n);

    // Fix sign: if x >= 0, sigmoid = 1 - vy.
    vbool4_t vsign_mask = __riscv_vmfge(vx, 0.0f, n);
    vfloat16m4_t vy_pos = __riscv_vfrsub(vy, 1.0f, n);
    vy = __riscv_vmerge(vy, vy_pos, vsign_mask, n);

    __riscv_vse16(output, vy, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
