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
void xnn_f16_vsin_ukernel__rvvfp16arith_poly_3_u8v(
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
  const _Float16 vone_over_pi = 0.318359375f16;
  const _Float16 vpi_0 = -3.140625f16;
  const _Float16 vpi_1 = -9.675026e-4f16;

  const _Float16 vc3 = -1.98846e-4f16;
  const _Float16 vc2 = 8.3336e-3f16;
  const _Float16 vc1 = -1.666666e-1f16;
  const _Float16 vc0 = 1.0f16;

  do {
    size_t vl = __riscv_vsetvl_e16m8(batch); batch -= vl;
    vfloat16m8_t vx = __riscv_vle16_v_f16m8((const _Float16*)input, vl); input += vl;

    vfloat16m8_t vy = __riscv_vfmul(vx, vone_over_pi, vl);
    vfloat16m8_t vk = __riscv_vfadd(vy, vmagic, vl);
    vfloat16m8_t vk_float = __riscv_vfsub(vk, vmagic, vl);

    vx = __riscv_vfadd(__riscv_vfmul(vk_float, vpi_0, vl), vx, vl);
    vx = __riscv_vfadd(__riscv_vfmul(vk_float, vpi_1, vl), vx, vl);

    vuint16m8_t vu_k = __riscv_vreinterpret_u16m8(vk);
    vuint16m8_t vsign_flip = __riscv_vsll(vu_k, 15u, vl);
    vx = __riscv_vreinterpret_f16m8(__riscv_vxor(__riscv_vreinterpret_u16m8(vx), vsign_flip, vl));

    const vfloat16m8_t vx2 = __riscv_vfmul(vx, vx, vl);

    vfloat16m8_t vp = __riscv_vfmv_v_f_f16m8(vc2, vl);
    vp = __riscv_vfmacc(vp, vc3, vx2, vl);
    vfloat16m8_t vp1 = __riscv_vfmv_v_f_f16m8(vc1, vl);
    vp1 = __riscv_vfmacc(vp1, vp, vx2, vl);
    vfloat16m8_t vp0 = __riscv_vfmv_v_f_f16m8(vc0, vl);
    vp = __riscv_vfmacc(vp0, vp1, vx2, vl);

    vfloat16m8_t vy_res = __riscv_vfmul(vx, vp, vl);

    __riscv_vse16_v_f16m8((_Float16*)output, vy_res, vl); output += vl;
  } while (batch != 0);
}
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && (XNN_ARCH_RISCV)
