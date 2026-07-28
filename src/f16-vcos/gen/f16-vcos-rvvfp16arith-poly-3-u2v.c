// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsin/rvv-poly-3.c.in
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


#if XNN_ENABLE_RISCV_FP16_VECTOR && (XNN_ARCH_RISCV)
void xnn_f16_vcos_ukernel__rvvfp16arith_poly_3_u2v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;

  const _Float16 vmagic = 1536.0f16;  // 1.5 * 2^10
  const _Float16 vtwo_over_pi = 0.63671875f16;
  const _Float16 vpi_over_two = 1.5703125f16;
  const _Float16 vhalf_pi_0 = -1.5703125f16;
  const _Float16 vhalf_pi_1 = -4.837513e-4f16;

  const _Float16 vc3 = -1.98846e-4f16;
  const _Float16 vc2 = 8.3336e-3f16;
  const _Float16 vc1 = -1.666666e-1f16;
  const _Float16 vc0 = 1.0f16;

  do {
    size_t vl = __riscv_vsetvl_e16m2(batch); batch -= vl;
    vfloat16m2_t vx = __riscv_vle16_v_f16m2((const _Float16*)input, vl); input += vl;

    vfloat16m2_t vy = __riscv_vfmul(vx, vtwo_over_pi, vl);
    vfloat16m2_t vk = __riscv_vfadd(vy, vmagic, vl);
    vfloat16m2_t vk_float = __riscv_vfsub(vk, vmagic, vl);

    vx = __riscv_vfadd(__riscv_vfmul(vk_float, vhalf_pi_0, vl), vx, vl);
    vx = __riscv_vfadd(__riscv_vfmul(vk_float, vhalf_pi_1, vl), vx, vl);

    vfloat16m2_t vk_quad = __riscv_vfadd(vk, 1.0f16, vl);
    vuint16m2_t vu_quad = __riscv_vreinterpret_u16m2(vk_quad);
    vbool8_t vmask_odd = __riscv_vmseq(__riscv_vand(vu_quad, (uint16_t)1, vl), (uint16_t)1, vl);
    vuint16m2_t vsign_flip = __riscv_vsll(__riscv_vsrl(vu_quad, 1u, vl), 15u, vl);

    vfloat16m2_t vabs_x = __riscv_vfabs(vx, vl);
    vfloat16m2_t vsub_x = __riscv_vfrsub(vabs_x, vpi_over_two, vl);
    vx = __riscv_vmerge(vx, vsub_x, vmask_odd, vl);
    vx = __riscv_vreinterpret_f16m2(__riscv_vxor(__riscv_vreinterpret_u16m2(vx), vsign_flip, vl));

    const vfloat16m2_t vx2 = __riscv_vfmul(vx, vx, vl);

    vfloat16m2_t vp = __riscv_vfmv_v_f_f16m2(vc2, vl);
    vp = __riscv_vfmacc(vp, vc3, vx2, vl);
    vfloat16m2_t vp1 = __riscv_vfmv_v_f_f16m2(vc1, vl);
    vp1 = __riscv_vfmacc(vp1, vp, vx2, vl);
    vfloat16m2_t vp0 = __riscv_vfmv_v_f_f16m2(vc0, vl);
    vp = __riscv_vfmacc(vp0, vp1, vx2, vl);

    vfloat16m2_t vy_res = __riscv_vfmul(vx, vp, vl);

    __riscv_vse16_v_f16m2((_Float16*)output, vy_res, vl); output += vl;
  } while (batch != 0);
}
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && (XNN_ARCH_RISCV)
