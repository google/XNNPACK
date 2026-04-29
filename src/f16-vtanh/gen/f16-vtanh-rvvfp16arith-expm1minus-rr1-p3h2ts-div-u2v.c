// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/rvvfp16arith-expm1minus-rr1-p3h2ts.c.in
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
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_vtanh_ukernel__rvvfp16arith_expm1minus_rr1_p3h2ts_div_u2v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_float16 vsat_cutoff = 0x1.208p+2f;
  const xnn_float16 vmagic_bias = 0x1.83Cp+10f;
  const xnn_float16 vminus_log2e = -0x1.714p+0f;
  const xnn_float16 vln2 = 0x1.630p-1f;
  const xnn_float16 vc3 = -0x1.56Cp-3f;
  const xnn_float16 vc2 = 0x1.020p+0f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;
  do {
    const size_t n = __riscv_vsetvl_e16m2(batch);

    vfloat16m2_t vx = __riscv_vle16_v_f16m2(input, n);
    input += n;

    // Extract sign and compute z = -|x|.
    vuint16m2_t vux = __riscv_vreinterpret_u16m2(vx);
    vuint16m2_t vsign = __riscv_vand(vux, 0x8000, n);
    vfloat16m2_t vz = __riscv_vfneg(__riscv_vfabs(vx, n), n);
    vz = __riscv_vfmax(vz, -vsat_cutoff, n);

    // Compute n = round(z * (-log2e)) + magic_bias.
    vfloat16m2_t vn = __riscv_vfmv_v_f_f16m2(vmagic_bias, n);
    vn = __riscv_vfmacc(vn, vminus_log2e, vz, n);

    // Create 2^n.
    vfloat16m2_t vs = __riscv_vreinterpret_f16m2(
        __riscv_vsll(__riscv_vreinterpret_i16m2(vn), 10, n));

    vn = __riscv_vfsub(vn, vmagic_bias, n);

    // Reduced argument: t = z + n * ln2.
    vfloat16m2_t vt = __riscv_vfmacc(vz, vln2, vn, n);

    // Degree-3 polynomial for exp(t)-1.
    vfloat16m2_t vp = __riscv_vfmv_v_f_f16m2(vc2, n);
    vp = __riscv_vfmacc(vp, vc3, vt, n);
    vp = __riscv_vfmul(vp, vt, n);

    // Reconstruct: expm1 = (p * t + t) * s + (s - 1)
    vt = __riscv_vfmul(vt, vs, n);
    vs = __riscv_vfsub(vs, 1.0f, n);
    vp = __riscv_vfmadd(vp, vt, vt, n);
    vfloat16m2_t vexpm1 = __riscv_vfadd(vp, vs, n);

    // tanh = expm1 / (expm1 + 2).
    vfloat16m2_t vy = __riscv_vfdiv(vexpm1,
        __riscv_vfadd(vexpm1, 2.0f, n), n);

    // Restore sign.
    vuint16m2_t vuy = __riscv_vreinterpret_u16m2(vy);
    vuy = __riscv_vxor(vuy, vsign, n);
    vy = __riscv_vreinterpret_f16m2(vuy);

    __riscv_vse16(output, vy, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
