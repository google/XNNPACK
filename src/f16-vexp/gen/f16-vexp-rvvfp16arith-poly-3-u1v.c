// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vexp/rvv-poly-3.c.in
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
#include "src/xnnpack/microparams.h"


static XNN_INLINE vfloat16m1_t xnn_setexp_f16(vfloat16m1_t vx, size_t vl) {
  // If `x` is an xnn_float16ing point value in the range [-15, 16], then
  // `(x + magic) << 10` will generate the floating point value corresponding
  // to `2^round(x)` (2^-15 and 2^16 will flush to zero and infinity,
  // respectively).
  const xnn_float16 vmagic = 1039.0f;  // 2^10 + 15.
  return __riscv_vreinterpret_f16m1(__riscv_vsll(__riscv_vreinterpret_u16m1(__riscv_vfadd(vx, vmagic, vl)), 10, vl));
}

// Quick-and-dirty round to nearest, only works for xnn_float16s in the range
// `[2^-9, 2^9)`.
static XNN_INLINE vfloat16m1_t xnn_qd_round_f16(vfloat16m1_t vx, size_t vl) {
  // If `x` is an xnn_float16ing point value in the range `[2^-9, 2^9)`, then
  // `(x + magic) - magic`` will generate the floating point value corresponding
  // to `round(x)`.
  const xnn_float16 vmagic = 1536.0f;  // 2^10 + 2^9.
  return __riscv_vfsub(__riscv_vfadd(vx, vmagic, vl), vmagic, vl);
}


void xnn_f16_vexp_ukernel__rvvfp16arith_poly_3_u1v(
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

  // The monomial coefficients of the interpolation polynomial (`valpha_0` = 1).
  const xnn_float16 valpha_1 = 0.6933594f;
  const xnn_float16 valpha_2 = 0.24255371f;
  const xnn_float16 valpha_3 = 0.05517578f;
        
  // Some useful constants.
  const xnn_float16 vlog2e = 1.4423828f;
  const xnn_float16 v16 = 16.0f;
  const xnn_float16 vm15 = -15.0f;
  const xnn_float16 vone = 1.0f;

  do {
    size_t vl = __riscv_vsetvl_e16m1(batch); batch -= vl;
    vfloat16m1_t vx = __riscv_vle16_v_f16m1(input, vl); input += vl;
      
    // Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
    vfloat16m1_t vz_prime = __riscv_vfmul(vx, vlog2e, vl);
    vz_prime = __riscv_vfmin(__riscv_vfmax(vz_prime, vm15, vl), v16, vl);

    // Decompose x * log2e into `z` (integer part) and `r` (remainder).
    const vfloat16m1_t vz = xnn_qd_round_f16(vz_prime, vl);
    const vfloat16m1_t vr = __riscv_vfsub(vz_prime, vz, vl);

    // Compute 2^z.
    vfloat16m1_t v2z = xnn_setexp_f16(vz, vl);

    // Evaluate the interpolation polynomial for `2^r`.
    vfloat16m1_t v2r = __riscv_vfadd(__riscv_vfmul(vr, valpha_3, vl), valpha_2, vl);
    v2r = __riscv_vfadd(__riscv_vfmul(vr, v2r, vl), valpha_1, vl);
    v2r = __riscv_vfadd(__riscv_vfmul(vr, v2r, vl), vone, vl);

    // Compute 2^z * 2^r.
    const vfloat16m1_t vy = __riscv_vfmul(v2z, v2r, vl);

    __riscv_vse16(output, vy, vl); output += vl;

  } while (batch > 0); 
}
