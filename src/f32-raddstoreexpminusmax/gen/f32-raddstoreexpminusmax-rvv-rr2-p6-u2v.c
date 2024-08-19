// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/rvv-rr2-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/raddstoreexpminusmax.h"


static inline vfloat32m2_t eval_poly_horner(vfloat32m2_t x,
                                                  float c6, float c5,
                                                  float c4, float c3, float c2,
                                                  float c1, float c0, size_t vl) {
  vfloat32m2_t z;
  vfloat32m2_t y = __riscv_vfmv_v_f_f32m2(c5, vl);
  y = __riscv_vfmacc_vf_f32m2(y, c6, x, vl);

  z = __riscv_vfmv_v_f_f32m2(c4, vl);
  y = __riscv_vfmadd_vv_f32m2(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m2(c3, vl);
  y = __riscv_vfmadd_vv_f32m2(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m2(c2, vl);
  y = __riscv_vfmadd_vv_f32m2(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m2(c1, vl);
  y = __riscv_vfmadd_vv_f32m2(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m2(c0, vl);
  y = __riscv_vfmadd_vv_f32m2(y, x, z, vl);
  return y;
}

/// @brief Computes the exponential function on vector of float32 values with a
/// 1-ULP error bound in the range [-87, 0]. Smaller inputs are flushed to
/// exp(-0x1.5d589ep6f) ~= 0x1.6a0a64p-127f while the result is undefined for
/// inputs greater than zero as well as NaNs.
///
/// This function is intended for use in computing softmax, whose inputs are
/// pre-normalized by subtracting the maximum, resulting in inputs in (-inf, 0).
/// One of these inputs will contribute exp(0) = 1 to the final sum, so any
/// inputs flushed upwards to -0x1.5d589ep6f and thus contributing at most
/// 0x1.6a0a64p-127f to the total, will not result of softmax unless at least
/// ~2^100 of them are summed in ascending order.
///
/// Exploitation of these properties results in a faster exponential by avoiding
/// the need to handle edge cases that arise from very large or small exponents.
///
/// @param[in] x Input vector of float32 values
/// @param[in] vl Length of vector x
/// @return Result of applying softexp() to elements of x
static inline vfloat32m2_t softexp_f32m2(
    vfloat32m2_t x, size_t vl,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) {
  // Ensure that q = RN(x/log(2)) >= e_min, so that 2^q can be computed safely
  // with a simple shift into the exponent field.
  // xmin = round(-126.5 * log(2), single, RU) ~ -87.68311309814453125

  const float xmin = -0x1.5ebb82p6;
  const float r_ln2f = 0x1.715476p+0f;
  const float l2uf = 0x1.62E400p-1f;
  const float l2lf = 0x1.7F7D1Cp-20f;
  const float c6 = 0x1.6850e4p-10f;
  const float c5 = 0x1.123bccp-7;
  const float c4 = 0x1.555b98p-5f;
  const float c3 = 0x1.55548ep-3f;
  const float c2 = 0x1.fffff8p-2f;

  // const float xmin = -0x1.5ebb82p6;
  x = __riscv_vfmax_vf_f32m2(x, xmin, vl);

  // 0. Reduction x = s * q ln(2)
  // const float r_ln2f = 0x1.715476p0f;  // single(1/log(2));
  // const float l2uf = 0x1.62e4p-1f;     // round(log(2), 24-8, RN);
  // const float l2lf = 0x1.7f7d1cp-20f;  // round(log(2) - l2uf, single, RN);
  vfloat32m2_t v = __riscv_vfmul_vf_f32m2(x, r_ln2f, vl);

  vint16m1_t q = __riscv_vfncvt_x_f_w_i16m1(v, vl);
  vfloat32m2_t z = __riscv_vfwcvt_f_x_v_f32m2(q, vl);

  // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
  vfloat32m2_t s = __riscv_vfnmsac_vf_f32m2(x, l2uf, z, vl);
  s = __riscv_vfnmsac_vf_f32m2(s, l2lf, z, vl);

  // 1. Approximate e^s by degree-6 polynomial approximation
  vfloat32m2_t u = eval_poly_horner(s, c6, c5, c4, c3, c2, 1.0f, 1.0f, vl);

  // 2. Reconstruction: compute u = u*2^q
  const int16_t p = (24 - 1);
  const int16_t bias = (128 - 1);
  vint32m2_t qw = __riscv_vwadd_vx_i32m2(q, bias, vl);
  vint32m2_t qq = __riscv_vsll_vx_i32m2(qw, p, vl);
  vfloat32m2_t qf = __riscv_vreinterpret_v_i32m2_f32m2(qq);
  u = __riscv_vfmul_vv_f32m2(u, qf, vl);
  return u;
}

void xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  size_t n = batch >> 2;
  size_t avl = n;
  size_t vl = __riscv_vsetvl_e32m2(n);

  vfloat32m2_t vsum = __riscv_vfmv_v_f_f32m2(0.0f, vl);
  do {
    vl = __riscv_vsetvl_e32m2(avl);
    avl -= vl;
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(input, vl);
    vx = __riscv_vfsub_vf_f32m2(vx, *max, vl);
    input += vl;
    vfloat32m2_t vexp = softexp_f32m2(vx, vl, params);
    __riscv_vse32_v_f32m2(output, vexp, vl);
    output += vl;
    vsum = __riscv_vfadd_vv_f32m2_tu(vsum, vsum, vexp, vl);
  } while(avl > 0);

  vfloat32m1_t v0 = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  *sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vsum, v0, n));
}
