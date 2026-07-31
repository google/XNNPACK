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


void xnn_f32_vsin_ukernel__rvv_poly_4_u4v(
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
  const float vone_over_pi = 0.31830988618379067154f;
  const float vpi_0 = -3.14159274e+00f;
  const float vpi_1 = 8.74227766e-08f;
  const float vpi_2 = 3.43024902e-15f;

  const float vc4 = 2.6052248359e-06f;
  const float vc3 = -1.9809075457e-04f;
  const float vc2 = 8.3330506459e-03f;
  const float vc1 = -1.6666658223e-01f;
  const float vc0 = 1.0000000000e+00f;

  do {
    size_t vl = __riscv_vsetvl_e32m4(batch); batch -= vl;
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(input, vl); input += vl;

    vfloat32m4_t vy = __riscv_vfmul(vx, vone_over_pi, vl);
    vfloat32m4_t vk = __riscv_vfadd(vy, vmagic, vl);
    vfloat32m4_t vk_float = __riscv_vfsub(vk, vmagic, vl);

    vx = __riscv_vfmacc_vf_f32m4(vx, vpi_0, vk_float, vl);
    vx = __riscv_vfmacc_vf_f32m4(vx, vpi_1, vk_float, vl);
    vx = __riscv_vfmacc_vf_f32m4(vx, vpi_2, vk_float, vl);

    vuint32m4_t vu_k = __riscv_vreinterpret_u32m4(vk);
    vuint32m4_t vsign_flip = __riscv_vsll(vu_k, 31u, vl);
    vx = __riscv_vreinterpret_f32m4(__riscv_vxor(__riscv_vreinterpret_u32m4(vx), vsign_flip, vl));

    const vfloat32m4_t vx2 = __riscv_vfmul(vx, vx, vl);

    vfloat32m4_t vp = __riscv_vfadd(__riscv_vfmul(vx2, vc4, vl), vc3, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vc2, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vc1, vl);
    vp = __riscv_vfadd(__riscv_vfmul(vx2, vp, vl), vc0, vl);

    vfloat32m4_t vy_res = __riscv_vfmul(vx, vp, vl);

    __riscv_vse32_v_f32m4(output, vy_res, vl); output += vl;
  } while (batch != 0);
}
