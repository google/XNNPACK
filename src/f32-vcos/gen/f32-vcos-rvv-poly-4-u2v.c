// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsin/rvv-poly-4.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vcos_ukernel__rvv_poly_4_u2v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  const float vmagic = 12582912.0f;  // 1.5 * 2^23
  const float vtwo_over_pi = 0.63661977236758134308f;
  const float vpi_over_two = 1.5707963267948966f;
  const float vhalf_pi_0 = -1.57079637e+00f;
  const float vhalf_pi_1 = 4.37113883e-08f;
  const float vhalf_pi_2 = 1.71512451e-15f;

  const float vc4 = 2.6052248359e-06f;
  const float vc3 = -1.9809075457e-04f;
  const float vc2 = 8.3330506459e-03f;
  const float vc1 = -1.6666658223e-01f;
  const float vc0 = 1.0000000000e+00f;

  do {
    size_t vl = __riscv_vsetvl_e32m2(batch); batch -= vl;
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(input, vl); input += vl;

    vfloat32m2_t vy = __riscv_vfmul(vx, vtwo_over_pi, vl);
    vfloat32m2_t vk = __riscv_vfadd(vy, vmagic, vl);
    vfloat32m2_t vk_float = __riscv_vfsub(vk, vmagic, vl);

    vx = __riscv_vfmacc_vf_f32m2(vx, vhalf_pi_0, vk_float, vl);
    vx = __riscv_vfmacc_vf_f32m2(vx, vhalf_pi_1, vk_float, vl);
    vx = __riscv_vfmacc_vf_f32m2(vx, vhalf_pi_2, vk_float, vl);

    vfloat32m2_t vk_quad = __riscv_vfadd(vk, 1.0f, vl);
    vuint32m2_t vu_quad = __riscv_vreinterpret_u32m2(vk_quad);
    vbool16_t vmask_odd = __riscv_vmseq(__riscv_vand(vu_quad, 1u, vl), 1u, vl);
    vuint32m2_t vsign_flip = __riscv_vsll(__riscv_vsrl(vu_quad, 1u, vl), 31u, vl);

    vfloat32m2_t vabs_x = __riscv_vfabs(vx, vl);
    vfloat32m2_t vsub_x = __riscv_vfrsub(vabs_x, vpi_over_two, vl);
    vx = __riscv_vmerge(vx, vsub_x, vmask_odd, vl);
    vx = __riscv_vreinterpret_f32m2(__riscv_vxor(__riscv_vreinterpret_u32m2(vx), vsign_flip, vl));

    const vfloat32m2_t vx2 = __riscv_vfmul(vx, vx, vl);

    vfloat32m2_t vp = __riscv_vfadd(__riscv_vfmul(vx2, vc4, vl), vc3, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vc2, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vc1, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vc0, vl);

    vfloat32m2_t vy_res = __riscv_vfmul(vx, vp, vl);

    __riscv_vse32_v_f32m2(output, vy_res, vl); output += vl;
  } while (batch != 0);
}
