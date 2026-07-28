// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsin/poly-3.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f16-scalar.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vcos_ukernel__scalar_poly_3_u1(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic, 1536.0f);  // 1.5 * 2^10
  XNN_SIMD_CONST_F16_FROM_FLOAT(vtwo_over_pi, 0.63671875f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vpi_over_two, 1.5703125f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf_pi_0, -1.5703125f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf_pi_1, -4.837513e-4f);

  // Monomial coefficients for sin(x)/x on [-pi/2, pi/2]
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc3, -1.98846e-4f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc2, 8.3336e-3f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc1, -1.666666e-1f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc0, 1.0f);


  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;

    xnn_simd_f16_t vy = xnn_mul_f16(vx, vtwo_over_pi);
    xnn_simd_f16_t vk = xnn_add_f16(vy, vmagic);
    xnn_simd_f16_t vk_float = xnn_sub_f16(vk, vmagic);

    vx = xnn_fmadd_f16(vk_float, vhalf_pi_0, vx);
    vx = xnn_fmadd_f16(vk_float, vhalf_pi_1, vx);

    xnn_simd_f16_t vk_quad = xnn_add_f16(vk, vone);

    xnn_simd_f16_t vmask_odd = xnn_sra_f16(xnn_sll_f16(vk_quad, 15), 15);
    xnn_simd_f16_t vsign_flip = xnn_sll_f16(xnn_srl_f16(vk_quad, 1), 15);

    xnn_simd_f16_t vabs_x = xnn_abs_f16(vx);
    xnn_simd_f16_t vsub_x = xnn_sub_f16(vpi_over_two, vabs_x);
    vx = xnn_or_f16(xnn_and_f16(vmask_odd, vsub_x), xnn_andnot_f16(vmask_odd, vx));
    vx = xnn_xor_f16(vx, vsign_flip);

    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, vc3, vc2);
    vp = xnn_fmadd_f16(vx2, vp, vc1);
    vp = xnn_fmadd_f16(vx2, vp, vc0);

    const xnn_simd_f16_t vy_res = xnn_mul_f16(vx, vp);
    xnn_storeu_f16(output, vy_res);
    output += xnn_simd_size_f16;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    xnn_simd_f16_t vy = xnn_mul_f16(vx, vtwo_over_pi);
    xnn_simd_f16_t vk = xnn_add_f16(vy, vmagic);
    xnn_simd_f16_t vk_float = xnn_sub_f16(vk, vmagic);

    vx = xnn_fmadd_f16(vk_float, vhalf_pi_0, vx);
    vx = xnn_fmadd_f16(vk_float, vhalf_pi_1, vx);

    xnn_simd_f16_t vk_quad = xnn_add_f16(vk, vone);

    xnn_simd_f16_t vmask_odd = xnn_sra_f16(xnn_sll_f16(vk_quad, 15), 15);
    xnn_simd_f16_t vsign_flip = xnn_sll_f16(xnn_srl_f16(vk_quad, 1), 15);

    xnn_simd_f16_t vabs_x = xnn_abs_f16(vx);
    xnn_simd_f16_t vsub_x = xnn_sub_f16(vpi_over_two, vabs_x);
    vx = xnn_or_f16(xnn_and_f16(vmask_odd, vsub_x), xnn_andnot_f16(vmask_odd, vx));
    vx = xnn_xor_f16(vx, vsign_flip);

    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, vc3, vc2);
    vp = xnn_fmadd_f16(vx2, vp, vc1);
    vp = xnn_fmadd_f16(vx2, vp, vc0);

    const xnn_simd_f16_t vy_res = xnn_mul_f16(vx, vp);
    xnn_store_tail_f16(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_vcos_ukernel__scalar_poly_3_u2(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic, 1536.0f);  // 1.5 * 2^10
  XNN_SIMD_CONST_F16_FROM_FLOAT(vtwo_over_pi, 0.63671875f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vpi_over_two, 1.5703125f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf_pi_0, -1.5703125f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf_pi_1, -4.837513e-4f);

  // Monomial coefficients for sin(x)/x on [-pi/2, pi/2]
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc3, -1.98846e-4f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc2, 8.3336e-3f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc1, -1.666666e-1f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc0, 1.0f);

  for (; batch >= 2 * sizeof(xnn_float16); batch -= 2 * sizeof(xnn_float16)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    input += 2;

    xnn_simd_f16_t vy_0 = xnn_mul_f16(vx_0, vtwo_over_pi);
    xnn_simd_f16_t vy_1 = xnn_mul_f16(vx_1, vtwo_over_pi);

    xnn_simd_f16_t vk_0 = xnn_add_f16(vy_0, vmagic);
    xnn_simd_f16_t vk_1 = xnn_add_f16(vy_1, vmagic);

    xnn_simd_f16_t vk_float_0 = xnn_sub_f16(vk_0, vmagic);
    xnn_simd_f16_t vk_float_1 = xnn_sub_f16(vk_1, vmagic);

    vx_0 = xnn_fmadd_f16(vk_float_0, vhalf_pi_0, vx_0);
    vx_1 = xnn_fmadd_f16(vk_float_1, vhalf_pi_0, vx_1);
    vx_0 = xnn_fmadd_f16(vk_float_0, vhalf_pi_1, vx_0);
    vx_1 = xnn_fmadd_f16(vk_float_1, vhalf_pi_1, vx_1);

    xnn_simd_f16_t vk_quad_0 = xnn_add_f16(vk_0, vone);
    xnn_simd_f16_t vk_quad_1 = xnn_add_f16(vk_1, vone);

    xnn_simd_f16_t vmask_odd_0 = xnn_sra_f16(xnn_sll_f16(vk_quad_0, 15), 15);
    xnn_simd_f16_t vmask_odd_1 = xnn_sra_f16(xnn_sll_f16(vk_quad_1, 15), 15);
    xnn_simd_f16_t vsign_flip_0 = xnn_sll_f16(xnn_srl_f16(vk_quad_0, 1), 15);
    xnn_simd_f16_t vsign_flip_1 = xnn_sll_f16(xnn_srl_f16(vk_quad_1, 1), 15);

    xnn_simd_f16_t vabs_x_0 = xnn_abs_f16(vx_0);
    xnn_simd_f16_t vabs_x_1 = xnn_abs_f16(vx_1);
    xnn_simd_f16_t vsub_x_0 = xnn_sub_f16(vpi_over_two, vabs_x_0);
    xnn_simd_f16_t vsub_x_1 = xnn_sub_f16(vpi_over_two, vabs_x_1);
    vx_0 = xnn_or_f16(xnn_and_f16(vmask_odd_0, vsub_x_0), xnn_andnot_f16(vmask_odd_0, vx_0));
    vx_1 = xnn_or_f16(xnn_and_f16(vmask_odd_1, vsub_x_1), xnn_andnot_f16(vmask_odd_1, vx_1));
    vx_0 = xnn_xor_f16(vx_0, vsign_flip_0);
    vx_1 = xnn_xor_f16(vx_1, vsign_flip_1);

    const xnn_simd_f16_t vx2_0 = xnn_mul_f16(vx_0, vx_0);
    const xnn_simd_f16_t vx2_1 = xnn_mul_f16(vx_1, vx_1);

    xnn_simd_f16_t vp_0 = xnn_fmadd_f16(vx2_0, vc3, vc2);
    xnn_simd_f16_t vp_1 = xnn_fmadd_f16(vx2_1, vc3, vc2);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, vc1);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, vc1);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, vc0);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, vc0);

    const xnn_simd_f16_t vy_res_0 = xnn_mul_f16(vx_0, vp_0);
    const xnn_simd_f16_t vy_res_1 = xnn_mul_f16(vx_1, vp_1);

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_res_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_res_1);
    output += 2;
  }

  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;

    xnn_simd_f16_t vy = xnn_mul_f16(vx, vtwo_over_pi);
    xnn_simd_f16_t vk = xnn_add_f16(vy, vmagic);
    xnn_simd_f16_t vk_float = xnn_sub_f16(vk, vmagic);

    vx = xnn_fmadd_f16(vk_float, vhalf_pi_0, vx);
    vx = xnn_fmadd_f16(vk_float, vhalf_pi_1, vx);

    xnn_simd_f16_t vk_quad = xnn_add_f16(vk, vone);

    xnn_simd_f16_t vmask_odd = xnn_sra_f16(xnn_sll_f16(vk_quad, 15), 15);
    xnn_simd_f16_t vsign_flip = xnn_sll_f16(xnn_srl_f16(vk_quad, 1), 15);

    xnn_simd_f16_t vabs_x = xnn_abs_f16(vx);
    xnn_simd_f16_t vsub_x = xnn_sub_f16(vpi_over_two, vabs_x);
    vx = xnn_or_f16(xnn_and_f16(vmask_odd, vsub_x), xnn_andnot_f16(vmask_odd, vx));
    vx = xnn_xor_f16(vx, vsign_flip);

    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, vc3, vc2);
    vp = xnn_fmadd_f16(vx2, vp, vc1);
    vp = xnn_fmadd_f16(vx2, vp, vc0);

    const xnn_simd_f16_t vy_res = xnn_mul_f16(vx, vp);
    xnn_storeu_f16(output, vy_res);
    output += xnn_simd_size_f16;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    xnn_simd_f16_t vy = xnn_mul_f16(vx, vtwo_over_pi);
    xnn_simd_f16_t vk = xnn_add_f16(vy, vmagic);
    xnn_simd_f16_t vk_float = xnn_sub_f16(vk, vmagic);

    vx = xnn_fmadd_f16(vk_float, vhalf_pi_0, vx);
    vx = xnn_fmadd_f16(vk_float, vhalf_pi_1, vx);

    xnn_simd_f16_t vk_quad = xnn_add_f16(vk, vone);

    xnn_simd_f16_t vmask_odd = xnn_sra_f16(xnn_sll_f16(vk_quad, 15), 15);
    xnn_simd_f16_t vsign_flip = xnn_sll_f16(xnn_srl_f16(vk_quad, 1), 15);

    xnn_simd_f16_t vabs_x = xnn_abs_f16(vx);
    xnn_simd_f16_t vsub_x = xnn_sub_f16(vpi_over_two, vabs_x);
    vx = xnn_or_f16(xnn_and_f16(vmask_odd, vsub_x), xnn_andnot_f16(vmask_odd, vx));
    vx = xnn_xor_f16(vx, vsign_flip);

    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, vc3, vc2);
    vp = xnn_fmadd_f16(vx2, vp, vc1);
    vp = xnn_fmadd_f16(vx2, vp, vc0);

    const xnn_simd_f16_t vy_res = xnn_mul_f16(vx, vp);
    xnn_store_tail_f16(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_vcos_ukernel__scalar_poly_3_u4(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic, 1536.0f);  // 1.5 * 2^10
  XNN_SIMD_CONST_F16_FROM_FLOAT(vtwo_over_pi, 0.63671875f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vpi_over_two, 1.5703125f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf_pi_0, -1.5703125f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf_pi_1, -4.837513e-4f);

  // Monomial coefficients for sin(x)/x on [-pi/2, pi/2]
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc3, -1.98846e-4f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc2, 8.3336e-3f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc1, -1.666666e-1f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc0, 1.0f);

  for (; batch >= 4 * sizeof(xnn_float16); batch -= 4 * sizeof(xnn_float16)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_2 = xnn_loadu_f16(input + 2 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_3 = xnn_loadu_f16(input + 3 * xnn_simd_size_f16);
    input += 4;

    xnn_simd_f16_t vy_0 = xnn_mul_f16(vx_0, vtwo_over_pi);
    xnn_simd_f16_t vy_1 = xnn_mul_f16(vx_1, vtwo_over_pi);
    xnn_simd_f16_t vy_2 = xnn_mul_f16(vx_2, vtwo_over_pi);
    xnn_simd_f16_t vy_3 = xnn_mul_f16(vx_3, vtwo_over_pi);

    xnn_simd_f16_t vk_0 = xnn_add_f16(vy_0, vmagic);
    xnn_simd_f16_t vk_1 = xnn_add_f16(vy_1, vmagic);
    xnn_simd_f16_t vk_2 = xnn_add_f16(vy_2, vmagic);
    xnn_simd_f16_t vk_3 = xnn_add_f16(vy_3, vmagic);

    xnn_simd_f16_t vk_float_0 = xnn_sub_f16(vk_0, vmagic);
    xnn_simd_f16_t vk_float_1 = xnn_sub_f16(vk_1, vmagic);
    xnn_simd_f16_t vk_float_2 = xnn_sub_f16(vk_2, vmagic);
    xnn_simd_f16_t vk_float_3 = xnn_sub_f16(vk_3, vmagic);

    vx_0 = xnn_fmadd_f16(vk_float_0, vhalf_pi_0, vx_0);
    vx_1 = xnn_fmadd_f16(vk_float_1, vhalf_pi_0, vx_1);
    vx_2 = xnn_fmadd_f16(vk_float_2, vhalf_pi_0, vx_2);
    vx_3 = xnn_fmadd_f16(vk_float_3, vhalf_pi_0, vx_3);
    vx_0 = xnn_fmadd_f16(vk_float_0, vhalf_pi_1, vx_0);
    vx_1 = xnn_fmadd_f16(vk_float_1, vhalf_pi_1, vx_1);
    vx_2 = xnn_fmadd_f16(vk_float_2, vhalf_pi_1, vx_2);
    vx_3 = xnn_fmadd_f16(vk_float_3, vhalf_pi_1, vx_3);

    xnn_simd_f16_t vk_quad_0 = xnn_add_f16(vk_0, vone);
    xnn_simd_f16_t vk_quad_1 = xnn_add_f16(vk_1, vone);
    xnn_simd_f16_t vk_quad_2 = xnn_add_f16(vk_2, vone);
    xnn_simd_f16_t vk_quad_3 = xnn_add_f16(vk_3, vone);

    xnn_simd_f16_t vmask_odd_0 = xnn_sra_f16(xnn_sll_f16(vk_quad_0, 15), 15);
    xnn_simd_f16_t vmask_odd_1 = xnn_sra_f16(xnn_sll_f16(vk_quad_1, 15), 15);
    xnn_simd_f16_t vmask_odd_2 = xnn_sra_f16(xnn_sll_f16(vk_quad_2, 15), 15);
    xnn_simd_f16_t vmask_odd_3 = xnn_sra_f16(xnn_sll_f16(vk_quad_3, 15), 15);
    xnn_simd_f16_t vsign_flip_0 = xnn_sll_f16(xnn_srl_f16(vk_quad_0, 1), 15);
    xnn_simd_f16_t vsign_flip_1 = xnn_sll_f16(xnn_srl_f16(vk_quad_1, 1), 15);
    xnn_simd_f16_t vsign_flip_2 = xnn_sll_f16(xnn_srl_f16(vk_quad_2, 1), 15);
    xnn_simd_f16_t vsign_flip_3 = xnn_sll_f16(xnn_srl_f16(vk_quad_3, 1), 15);

    xnn_simd_f16_t vabs_x_0 = xnn_abs_f16(vx_0);
    xnn_simd_f16_t vabs_x_1 = xnn_abs_f16(vx_1);
    xnn_simd_f16_t vabs_x_2 = xnn_abs_f16(vx_2);
    xnn_simd_f16_t vabs_x_3 = xnn_abs_f16(vx_3);
    xnn_simd_f16_t vsub_x_0 = xnn_sub_f16(vpi_over_two, vabs_x_0);
    xnn_simd_f16_t vsub_x_1 = xnn_sub_f16(vpi_over_two, vabs_x_1);
    xnn_simd_f16_t vsub_x_2 = xnn_sub_f16(vpi_over_two, vabs_x_2);
    xnn_simd_f16_t vsub_x_3 = xnn_sub_f16(vpi_over_two, vabs_x_3);
    vx_0 = xnn_or_f16(xnn_and_f16(vmask_odd_0, vsub_x_0), xnn_andnot_f16(vmask_odd_0, vx_0));
    vx_1 = xnn_or_f16(xnn_and_f16(vmask_odd_1, vsub_x_1), xnn_andnot_f16(vmask_odd_1, vx_1));
    vx_2 = xnn_or_f16(xnn_and_f16(vmask_odd_2, vsub_x_2), xnn_andnot_f16(vmask_odd_2, vx_2));
    vx_3 = xnn_or_f16(xnn_and_f16(vmask_odd_3, vsub_x_3), xnn_andnot_f16(vmask_odd_3, vx_3));
    vx_0 = xnn_xor_f16(vx_0, vsign_flip_0);
    vx_1 = xnn_xor_f16(vx_1, vsign_flip_1);
    vx_2 = xnn_xor_f16(vx_2, vsign_flip_2);
    vx_3 = xnn_xor_f16(vx_3, vsign_flip_3);

    const xnn_simd_f16_t vx2_0 = xnn_mul_f16(vx_0, vx_0);
    const xnn_simd_f16_t vx2_1 = xnn_mul_f16(vx_1, vx_1);
    const xnn_simd_f16_t vx2_2 = xnn_mul_f16(vx_2, vx_2);
    const xnn_simd_f16_t vx2_3 = xnn_mul_f16(vx_3, vx_3);

    xnn_simd_f16_t vp_0 = xnn_fmadd_f16(vx2_0, vc3, vc2);
    xnn_simd_f16_t vp_1 = xnn_fmadd_f16(vx2_1, vc3, vc2);
    xnn_simd_f16_t vp_2 = xnn_fmadd_f16(vx2_2, vc3, vc2);
    xnn_simd_f16_t vp_3 = xnn_fmadd_f16(vx2_3, vc3, vc2);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, vc1);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, vc1);
    vp_2 = xnn_fmadd_f16(vx2_2, vp_2, vc1);
    vp_3 = xnn_fmadd_f16(vx2_3, vp_3, vc1);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, vc0);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, vc0);
    vp_2 = xnn_fmadd_f16(vx2_2, vp_2, vc0);
    vp_3 = xnn_fmadd_f16(vx2_3, vp_3, vc0);

    const xnn_simd_f16_t vy_res_0 = xnn_mul_f16(vx_0, vp_0);
    const xnn_simd_f16_t vy_res_1 = xnn_mul_f16(vx_1, vp_1);
    const xnn_simd_f16_t vy_res_2 = xnn_mul_f16(vx_2, vp_2);
    const xnn_simd_f16_t vy_res_3 = xnn_mul_f16(vx_3, vp_3);

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_res_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_res_1);
    xnn_storeu_f16(output + 2 * xnn_simd_size_f16, vy_res_2);
    xnn_storeu_f16(output + 3 * xnn_simd_size_f16, vy_res_3);
    output += 4;
  }

  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;

    xnn_simd_f16_t vy = xnn_mul_f16(vx, vtwo_over_pi);
    xnn_simd_f16_t vk = xnn_add_f16(vy, vmagic);
    xnn_simd_f16_t vk_float = xnn_sub_f16(vk, vmagic);

    vx = xnn_fmadd_f16(vk_float, vhalf_pi_0, vx);
    vx = xnn_fmadd_f16(vk_float, vhalf_pi_1, vx);

    xnn_simd_f16_t vk_quad = xnn_add_f16(vk, vone);

    xnn_simd_f16_t vmask_odd = xnn_sra_f16(xnn_sll_f16(vk_quad, 15), 15);
    xnn_simd_f16_t vsign_flip = xnn_sll_f16(xnn_srl_f16(vk_quad, 1), 15);

    xnn_simd_f16_t vabs_x = xnn_abs_f16(vx);
    xnn_simd_f16_t vsub_x = xnn_sub_f16(vpi_over_two, vabs_x);
    vx = xnn_or_f16(xnn_and_f16(vmask_odd, vsub_x), xnn_andnot_f16(vmask_odd, vx));
    vx = xnn_xor_f16(vx, vsign_flip);

    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, vc3, vc2);
    vp = xnn_fmadd_f16(vx2, vp, vc1);
    vp = xnn_fmadd_f16(vx2, vp, vc0);

    const xnn_simd_f16_t vy_res = xnn_mul_f16(vx, vp);
    xnn_storeu_f16(output, vy_res);
    output += xnn_simd_size_f16;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    xnn_simd_f16_t vy = xnn_mul_f16(vx, vtwo_over_pi);
    xnn_simd_f16_t vk = xnn_add_f16(vy, vmagic);
    xnn_simd_f16_t vk_float = xnn_sub_f16(vk, vmagic);

    vx = xnn_fmadd_f16(vk_float, vhalf_pi_0, vx);
    vx = xnn_fmadd_f16(vk_float, vhalf_pi_1, vx);

    xnn_simd_f16_t vk_quad = xnn_add_f16(vk, vone);

    xnn_simd_f16_t vmask_odd = xnn_sra_f16(xnn_sll_f16(vk_quad, 15), 15);
    xnn_simd_f16_t vsign_flip = xnn_sll_f16(xnn_srl_f16(vk_quad, 1), 15);

    xnn_simd_f16_t vabs_x = xnn_abs_f16(vx);
    xnn_simd_f16_t vsub_x = xnn_sub_f16(vpi_over_two, vabs_x);
    vx = xnn_or_f16(xnn_and_f16(vmask_odd, vsub_x), xnn_andnot_f16(vmask_odd, vx));
    vx = xnn_xor_f16(vx, vsign_flip);

    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, vc3, vc2);
    vp = xnn_fmadd_f16(vx2, vp, vc1);
    vp = xnn_fmadd_f16(vx2, vp, vc0);

    const xnn_simd_f16_t vy_res = xnn_mul_f16(vx, vp);
    xnn_store_tail_f16(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_vcos_ukernel__scalar_poly_3_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 1);

  XNN_SIMD_CONST_F16_FROM_FLOAT(vmagic, 1536.0f);  // 1.5 * 2^10
  XNN_SIMD_CONST_F16_FROM_FLOAT(vtwo_over_pi, 0.63671875f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vpi_over_two, 1.5703125f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.0f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf_pi_0, -1.5703125f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vhalf_pi_1, -4.837513e-4f);

  // Monomial coefficients for sin(x)/x on [-pi/2, pi/2]
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc3, -1.98846e-4f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc2, 8.3336e-3f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc1, -1.666666e-1f);
  XNN_SIMD_CONST_F16_FROM_FLOAT(vc0, 1.0f);

  for (; batch >= 8 * sizeof(xnn_float16); batch -= 8 * sizeof(xnn_float16)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_2 = xnn_loadu_f16(input + 2 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_3 = xnn_loadu_f16(input + 3 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_4 = xnn_loadu_f16(input + 4 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_5 = xnn_loadu_f16(input + 5 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_6 = xnn_loadu_f16(input + 6 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_7 = xnn_loadu_f16(input + 7 * xnn_simd_size_f16);
    input += 8;

    xnn_simd_f16_t vy_0 = xnn_mul_f16(vx_0, vtwo_over_pi);
    xnn_simd_f16_t vy_1 = xnn_mul_f16(vx_1, vtwo_over_pi);
    xnn_simd_f16_t vy_2 = xnn_mul_f16(vx_2, vtwo_over_pi);
    xnn_simd_f16_t vy_3 = xnn_mul_f16(vx_3, vtwo_over_pi);
    xnn_simd_f16_t vy_4 = xnn_mul_f16(vx_4, vtwo_over_pi);
    xnn_simd_f16_t vy_5 = xnn_mul_f16(vx_5, vtwo_over_pi);
    xnn_simd_f16_t vy_6 = xnn_mul_f16(vx_6, vtwo_over_pi);
    xnn_simd_f16_t vy_7 = xnn_mul_f16(vx_7, vtwo_over_pi);

    xnn_simd_f16_t vk_0 = xnn_add_f16(vy_0, vmagic);
    xnn_simd_f16_t vk_1 = xnn_add_f16(vy_1, vmagic);
    xnn_simd_f16_t vk_2 = xnn_add_f16(vy_2, vmagic);
    xnn_simd_f16_t vk_3 = xnn_add_f16(vy_3, vmagic);
    xnn_simd_f16_t vk_4 = xnn_add_f16(vy_4, vmagic);
    xnn_simd_f16_t vk_5 = xnn_add_f16(vy_5, vmagic);
    xnn_simd_f16_t vk_6 = xnn_add_f16(vy_6, vmagic);
    xnn_simd_f16_t vk_7 = xnn_add_f16(vy_7, vmagic);

    xnn_simd_f16_t vk_float_0 = xnn_sub_f16(vk_0, vmagic);
    xnn_simd_f16_t vk_float_1 = xnn_sub_f16(vk_1, vmagic);
    xnn_simd_f16_t vk_float_2 = xnn_sub_f16(vk_2, vmagic);
    xnn_simd_f16_t vk_float_3 = xnn_sub_f16(vk_3, vmagic);
    xnn_simd_f16_t vk_float_4 = xnn_sub_f16(vk_4, vmagic);
    xnn_simd_f16_t vk_float_5 = xnn_sub_f16(vk_5, vmagic);
    xnn_simd_f16_t vk_float_6 = xnn_sub_f16(vk_6, vmagic);
    xnn_simd_f16_t vk_float_7 = xnn_sub_f16(vk_7, vmagic);

    vx_0 = xnn_fmadd_f16(vk_float_0, vhalf_pi_0, vx_0);
    vx_1 = xnn_fmadd_f16(vk_float_1, vhalf_pi_0, vx_1);
    vx_2 = xnn_fmadd_f16(vk_float_2, vhalf_pi_0, vx_2);
    vx_3 = xnn_fmadd_f16(vk_float_3, vhalf_pi_0, vx_3);
    vx_4 = xnn_fmadd_f16(vk_float_4, vhalf_pi_0, vx_4);
    vx_5 = xnn_fmadd_f16(vk_float_5, vhalf_pi_0, vx_5);
    vx_6 = xnn_fmadd_f16(vk_float_6, vhalf_pi_0, vx_6);
    vx_7 = xnn_fmadd_f16(vk_float_7, vhalf_pi_0, vx_7);
    vx_0 = xnn_fmadd_f16(vk_float_0, vhalf_pi_1, vx_0);
    vx_1 = xnn_fmadd_f16(vk_float_1, vhalf_pi_1, vx_1);
    vx_2 = xnn_fmadd_f16(vk_float_2, vhalf_pi_1, vx_2);
    vx_3 = xnn_fmadd_f16(vk_float_3, vhalf_pi_1, vx_3);
    vx_4 = xnn_fmadd_f16(vk_float_4, vhalf_pi_1, vx_4);
    vx_5 = xnn_fmadd_f16(vk_float_5, vhalf_pi_1, vx_5);
    vx_6 = xnn_fmadd_f16(vk_float_6, vhalf_pi_1, vx_6);
    vx_7 = xnn_fmadd_f16(vk_float_7, vhalf_pi_1, vx_7);

    xnn_simd_f16_t vk_quad_0 = xnn_add_f16(vk_0, vone);
    xnn_simd_f16_t vk_quad_1 = xnn_add_f16(vk_1, vone);
    xnn_simd_f16_t vk_quad_2 = xnn_add_f16(vk_2, vone);
    xnn_simd_f16_t vk_quad_3 = xnn_add_f16(vk_3, vone);
    xnn_simd_f16_t vk_quad_4 = xnn_add_f16(vk_4, vone);
    xnn_simd_f16_t vk_quad_5 = xnn_add_f16(vk_5, vone);
    xnn_simd_f16_t vk_quad_6 = xnn_add_f16(vk_6, vone);
    xnn_simd_f16_t vk_quad_7 = xnn_add_f16(vk_7, vone);

    xnn_simd_f16_t vmask_odd_0 = xnn_sra_f16(xnn_sll_f16(vk_quad_0, 15), 15);
    xnn_simd_f16_t vmask_odd_1 = xnn_sra_f16(xnn_sll_f16(vk_quad_1, 15), 15);
    xnn_simd_f16_t vmask_odd_2 = xnn_sra_f16(xnn_sll_f16(vk_quad_2, 15), 15);
    xnn_simd_f16_t vmask_odd_3 = xnn_sra_f16(xnn_sll_f16(vk_quad_3, 15), 15);
    xnn_simd_f16_t vmask_odd_4 = xnn_sra_f16(xnn_sll_f16(vk_quad_4, 15), 15);
    xnn_simd_f16_t vmask_odd_5 = xnn_sra_f16(xnn_sll_f16(vk_quad_5, 15), 15);
    xnn_simd_f16_t vmask_odd_6 = xnn_sra_f16(xnn_sll_f16(vk_quad_6, 15), 15);
    xnn_simd_f16_t vmask_odd_7 = xnn_sra_f16(xnn_sll_f16(vk_quad_7, 15), 15);
    xnn_simd_f16_t vsign_flip_0 = xnn_sll_f16(xnn_srl_f16(vk_quad_0, 1), 15);
    xnn_simd_f16_t vsign_flip_1 = xnn_sll_f16(xnn_srl_f16(vk_quad_1, 1), 15);
    xnn_simd_f16_t vsign_flip_2 = xnn_sll_f16(xnn_srl_f16(vk_quad_2, 1), 15);
    xnn_simd_f16_t vsign_flip_3 = xnn_sll_f16(xnn_srl_f16(vk_quad_3, 1), 15);
    xnn_simd_f16_t vsign_flip_4 = xnn_sll_f16(xnn_srl_f16(vk_quad_4, 1), 15);
    xnn_simd_f16_t vsign_flip_5 = xnn_sll_f16(xnn_srl_f16(vk_quad_5, 1), 15);
    xnn_simd_f16_t vsign_flip_6 = xnn_sll_f16(xnn_srl_f16(vk_quad_6, 1), 15);
    xnn_simd_f16_t vsign_flip_7 = xnn_sll_f16(xnn_srl_f16(vk_quad_7, 1), 15);

    xnn_simd_f16_t vabs_x_0 = xnn_abs_f16(vx_0);
    xnn_simd_f16_t vabs_x_1 = xnn_abs_f16(vx_1);
    xnn_simd_f16_t vabs_x_2 = xnn_abs_f16(vx_2);
    xnn_simd_f16_t vabs_x_3 = xnn_abs_f16(vx_3);
    xnn_simd_f16_t vabs_x_4 = xnn_abs_f16(vx_4);
    xnn_simd_f16_t vabs_x_5 = xnn_abs_f16(vx_5);
    xnn_simd_f16_t vabs_x_6 = xnn_abs_f16(vx_6);
    xnn_simd_f16_t vabs_x_7 = xnn_abs_f16(vx_7);
    xnn_simd_f16_t vsub_x_0 = xnn_sub_f16(vpi_over_two, vabs_x_0);
    xnn_simd_f16_t vsub_x_1 = xnn_sub_f16(vpi_over_two, vabs_x_1);
    xnn_simd_f16_t vsub_x_2 = xnn_sub_f16(vpi_over_two, vabs_x_2);
    xnn_simd_f16_t vsub_x_3 = xnn_sub_f16(vpi_over_two, vabs_x_3);
    xnn_simd_f16_t vsub_x_4 = xnn_sub_f16(vpi_over_two, vabs_x_4);
    xnn_simd_f16_t vsub_x_5 = xnn_sub_f16(vpi_over_two, vabs_x_5);
    xnn_simd_f16_t vsub_x_6 = xnn_sub_f16(vpi_over_two, vabs_x_6);
    xnn_simd_f16_t vsub_x_7 = xnn_sub_f16(vpi_over_two, vabs_x_7);
    vx_0 = xnn_or_f16(xnn_and_f16(vmask_odd_0, vsub_x_0), xnn_andnot_f16(vmask_odd_0, vx_0));
    vx_1 = xnn_or_f16(xnn_and_f16(vmask_odd_1, vsub_x_1), xnn_andnot_f16(vmask_odd_1, vx_1));
    vx_2 = xnn_or_f16(xnn_and_f16(vmask_odd_2, vsub_x_2), xnn_andnot_f16(vmask_odd_2, vx_2));
    vx_3 = xnn_or_f16(xnn_and_f16(vmask_odd_3, vsub_x_3), xnn_andnot_f16(vmask_odd_3, vx_3));
    vx_4 = xnn_or_f16(xnn_and_f16(vmask_odd_4, vsub_x_4), xnn_andnot_f16(vmask_odd_4, vx_4));
    vx_5 = xnn_or_f16(xnn_and_f16(vmask_odd_5, vsub_x_5), xnn_andnot_f16(vmask_odd_5, vx_5));
    vx_6 = xnn_or_f16(xnn_and_f16(vmask_odd_6, vsub_x_6), xnn_andnot_f16(vmask_odd_6, vx_6));
    vx_7 = xnn_or_f16(xnn_and_f16(vmask_odd_7, vsub_x_7), xnn_andnot_f16(vmask_odd_7, vx_7));
    vx_0 = xnn_xor_f16(vx_0, vsign_flip_0);
    vx_1 = xnn_xor_f16(vx_1, vsign_flip_1);
    vx_2 = xnn_xor_f16(vx_2, vsign_flip_2);
    vx_3 = xnn_xor_f16(vx_3, vsign_flip_3);
    vx_4 = xnn_xor_f16(vx_4, vsign_flip_4);
    vx_5 = xnn_xor_f16(vx_5, vsign_flip_5);
    vx_6 = xnn_xor_f16(vx_6, vsign_flip_6);
    vx_7 = xnn_xor_f16(vx_7, vsign_flip_7);

    const xnn_simd_f16_t vx2_0 = xnn_mul_f16(vx_0, vx_0);
    const xnn_simd_f16_t vx2_1 = xnn_mul_f16(vx_1, vx_1);
    const xnn_simd_f16_t vx2_2 = xnn_mul_f16(vx_2, vx_2);
    const xnn_simd_f16_t vx2_3 = xnn_mul_f16(vx_3, vx_3);
    const xnn_simd_f16_t vx2_4 = xnn_mul_f16(vx_4, vx_4);
    const xnn_simd_f16_t vx2_5 = xnn_mul_f16(vx_5, vx_5);
    const xnn_simd_f16_t vx2_6 = xnn_mul_f16(vx_6, vx_6);
    const xnn_simd_f16_t vx2_7 = xnn_mul_f16(vx_7, vx_7);

    xnn_simd_f16_t vp_0 = xnn_fmadd_f16(vx2_0, vc3, vc2);
    xnn_simd_f16_t vp_1 = xnn_fmadd_f16(vx2_1, vc3, vc2);
    xnn_simd_f16_t vp_2 = xnn_fmadd_f16(vx2_2, vc3, vc2);
    xnn_simd_f16_t vp_3 = xnn_fmadd_f16(vx2_3, vc3, vc2);
    xnn_simd_f16_t vp_4 = xnn_fmadd_f16(vx2_4, vc3, vc2);
    xnn_simd_f16_t vp_5 = xnn_fmadd_f16(vx2_5, vc3, vc2);
    xnn_simd_f16_t vp_6 = xnn_fmadd_f16(vx2_6, vc3, vc2);
    xnn_simd_f16_t vp_7 = xnn_fmadd_f16(vx2_7, vc3, vc2);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, vc1);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, vc1);
    vp_2 = xnn_fmadd_f16(vx2_2, vp_2, vc1);
    vp_3 = xnn_fmadd_f16(vx2_3, vp_3, vc1);
    vp_4 = xnn_fmadd_f16(vx2_4, vp_4, vc1);
    vp_5 = xnn_fmadd_f16(vx2_5, vp_5, vc1);
    vp_6 = xnn_fmadd_f16(vx2_6, vp_6, vc1);
    vp_7 = xnn_fmadd_f16(vx2_7, vp_7, vc1);
    vp_0 = xnn_fmadd_f16(vx2_0, vp_0, vc0);
    vp_1 = xnn_fmadd_f16(vx2_1, vp_1, vc0);
    vp_2 = xnn_fmadd_f16(vx2_2, vp_2, vc0);
    vp_3 = xnn_fmadd_f16(vx2_3, vp_3, vc0);
    vp_4 = xnn_fmadd_f16(vx2_4, vp_4, vc0);
    vp_5 = xnn_fmadd_f16(vx2_5, vp_5, vc0);
    vp_6 = xnn_fmadd_f16(vx2_6, vp_6, vc0);
    vp_7 = xnn_fmadd_f16(vx2_7, vp_7, vc0);

    const xnn_simd_f16_t vy_res_0 = xnn_mul_f16(vx_0, vp_0);
    const xnn_simd_f16_t vy_res_1 = xnn_mul_f16(vx_1, vp_1);
    const xnn_simd_f16_t vy_res_2 = xnn_mul_f16(vx_2, vp_2);
    const xnn_simd_f16_t vy_res_3 = xnn_mul_f16(vx_3, vp_3);
    const xnn_simd_f16_t vy_res_4 = xnn_mul_f16(vx_4, vp_4);
    const xnn_simd_f16_t vy_res_5 = xnn_mul_f16(vx_5, vp_5);
    const xnn_simd_f16_t vy_res_6 = xnn_mul_f16(vx_6, vp_6);
    const xnn_simd_f16_t vy_res_7 = xnn_mul_f16(vx_7, vp_7);

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_res_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_res_1);
    xnn_storeu_f16(output + 2 * xnn_simd_size_f16, vy_res_2);
    xnn_storeu_f16(output + 3 * xnn_simd_size_f16, vy_res_3);
    xnn_storeu_f16(output + 4 * xnn_simd_size_f16, vy_res_4);
    xnn_storeu_f16(output + 5 * xnn_simd_size_f16, vy_res_5);
    xnn_storeu_f16(output + 6 * xnn_simd_size_f16, vy_res_6);
    xnn_storeu_f16(output + 7 * xnn_simd_size_f16, vy_res_7);
    output += 8;
  }

  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;

    xnn_simd_f16_t vy = xnn_mul_f16(vx, vtwo_over_pi);
    xnn_simd_f16_t vk = xnn_add_f16(vy, vmagic);
    xnn_simd_f16_t vk_float = xnn_sub_f16(vk, vmagic);

    vx = xnn_fmadd_f16(vk_float, vhalf_pi_0, vx);
    vx = xnn_fmadd_f16(vk_float, vhalf_pi_1, vx);

    xnn_simd_f16_t vk_quad = xnn_add_f16(vk, vone);

    xnn_simd_f16_t vmask_odd = xnn_sra_f16(xnn_sll_f16(vk_quad, 15), 15);
    xnn_simd_f16_t vsign_flip = xnn_sll_f16(xnn_srl_f16(vk_quad, 1), 15);

    xnn_simd_f16_t vabs_x = xnn_abs_f16(vx);
    xnn_simd_f16_t vsub_x = xnn_sub_f16(vpi_over_two, vabs_x);
    vx = xnn_or_f16(xnn_and_f16(vmask_odd, vsub_x), xnn_andnot_f16(vmask_odd, vx));
    vx = xnn_xor_f16(vx, vsign_flip);

    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, vc3, vc2);
    vp = xnn_fmadd_f16(vx2, vp, vc1);
    vp = xnn_fmadd_f16(vx2, vp, vc0);

    const xnn_simd_f16_t vy_res = xnn_mul_f16(vx, vp);
    xnn_storeu_f16(output, vy_res);
    output += xnn_simd_size_f16;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

    xnn_simd_f16_t vy = xnn_mul_f16(vx, vtwo_over_pi);
    xnn_simd_f16_t vk = xnn_add_f16(vy, vmagic);
    xnn_simd_f16_t vk_float = xnn_sub_f16(vk, vmagic);

    vx = xnn_fmadd_f16(vk_float, vhalf_pi_0, vx);
    vx = xnn_fmadd_f16(vk_float, vhalf_pi_1, vx);

    xnn_simd_f16_t vk_quad = xnn_add_f16(vk, vone);

    xnn_simd_f16_t vmask_odd = xnn_sra_f16(xnn_sll_f16(vk_quad, 15), 15);
    xnn_simd_f16_t vsign_flip = xnn_sll_f16(xnn_srl_f16(vk_quad, 1), 15);

    xnn_simd_f16_t vabs_x = xnn_abs_f16(vx);
    xnn_simd_f16_t vsub_x = xnn_sub_f16(vpi_over_two, vabs_x);
    vx = xnn_or_f16(xnn_and_f16(vmask_odd, vsub_x), xnn_andnot_f16(vmask_odd, vx));
    vx = xnn_xor_f16(vx, vsign_flip);

    const xnn_simd_f16_t vx2 = xnn_mul_f16(vx, vx);
    xnn_simd_f16_t vp = xnn_fmadd_f16(vx2, vc3, vc2);
    vp = xnn_fmadd_f16(vx2, vp, vc1);
    vp = xnn_fmadd_f16(vx2, vp, vc0);

    const xnn_simd_f16_t vy_res = xnn_mul_f16(vx, vp);
    xnn_store_tail_f16(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}
