// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsin/poly-4.c.in
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
#include "src/xnnpack/simd/f32-scalar.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vsin_ukernel__scalar_poly_4_u1(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  XNN_SIMD_CONST_F32(vmagic, 12582912.0f);  // 1.5 * 2^23
  XNN_SIMD_CONST_F32(vone_over_pi, 0.31830988618379067154f);
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vpi_0, -3.14159274e+00f);
    XNN_SIMD_CONST_F32(vpi_1, 8.74227766e-08f);
    XNN_SIMD_CONST_F32(vpi_2, 3.43024902e-15f);
  #else
    XNN_SIMD_CONST_F32(vpi_0, -3.14062500e+00f);
    XNN_SIMD_CONST_F32(vpi_1, -9.67502594e-04f);
    XNN_SIMD_CONST_F32(vpi_2, -1.50990672e-07f);
    XNN_SIMD_CONST_F32(vpi_3, -5.12656584e-12f);
    XNN_SIMD_CONST_F32(vpi_4, -1.22514845e-16f);
  #endif

  // Monomial coefficients for sin(x)/x on [-pi/2, pi/2]
  XNN_SIMD_CONST_F32(vc4, 2.6052248359e-06f);
  XNN_SIMD_CONST_F32(vc3, -1.9809075457e-04f);
  XNN_SIMD_CONST_F32(vc2, 8.3330506459e-03f);
  XNN_SIMD_CONST_F32(vc1, -1.6666658223e-01f);
  XNN_SIMD_CONST_F32(vc0, 1.0000000000e+00f);


  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_mul_f32(vx, vone_over_pi);
    xnn_simd_f32_t vk = xnn_add_f32(vy, vmagic);
    xnn_simd_f32_t vk_float = xnn_sub_f32(vk, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
    #else
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_3, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_4, vx);
    #endif

    xnn_simd_f32_t vsign_flip = xnn_sll_f32(vk, 31);
    vx = xnn_xor_f32(vx, vsign_flip);

    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc4, vc3);
    vp = xnn_fmadd_f32(vx2, vp, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_storeu_f32(output, vy_res);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_mul_f32(vx, vone_over_pi);
    xnn_simd_f32_t vk = xnn_add_f32(vy, vmagic);
    xnn_simd_f32_t vk_float = xnn_sub_f32(vk, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
    #else
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_3, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_4, vx);
    #endif

    xnn_simd_f32_t vsign_flip = xnn_sll_f32(vk, 31);
    vx = xnn_xor_f32(vx, vsign_flip);

    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc4, vc3);
    vp = xnn_fmadd_f32(vx2, vp, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_tail_f32(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsin_ukernel__scalar_poly_4_u2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  XNN_SIMD_CONST_F32(vmagic, 12582912.0f);  // 1.5 * 2^23
  XNN_SIMD_CONST_F32(vone_over_pi, 0.31830988618379067154f);
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vpi_0, -3.14159274e+00f);
    XNN_SIMD_CONST_F32(vpi_1, 8.74227766e-08f);
    XNN_SIMD_CONST_F32(vpi_2, 3.43024902e-15f);
  #else
    XNN_SIMD_CONST_F32(vpi_0, -3.14062500e+00f);
    XNN_SIMD_CONST_F32(vpi_1, -9.67502594e-04f);
    XNN_SIMD_CONST_F32(vpi_2, -1.50990672e-07f);
    XNN_SIMD_CONST_F32(vpi_3, -5.12656584e-12f);
    XNN_SIMD_CONST_F32(vpi_4, -1.22514845e-16f);
  #endif

  // Monomial coefficients for sin(x)/x on [-pi/2, pi/2]
  XNN_SIMD_CONST_F32(vc4, 2.6052248359e-06f);
  XNN_SIMD_CONST_F32(vc3, -1.9809075457e-04f);
  XNN_SIMD_CONST_F32(vc2, 8.3330506459e-03f);
  XNN_SIMD_CONST_F32(vc1, -1.6666658223e-01f);
  XNN_SIMD_CONST_F32(vc0, 1.0000000000e+00f);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2;

    xnn_simd_f32_t vy_0 = xnn_mul_f32(vx_0, vone_over_pi);
    xnn_simd_f32_t vy_1 = xnn_mul_f32(vx_1, vone_over_pi);

    xnn_simd_f32_t vk_0 = xnn_add_f32(vy_0, vmagic);
    xnn_simd_f32_t vk_1 = xnn_add_f32(vy_1, vmagic);

    xnn_simd_f32_t vk_float_0 = xnn_sub_f32(vk_0, vmagic);
    xnn_simd_f32_t vk_float_1 = xnn_sub_f32(vk_1, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_0, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_0, vx_1);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_1, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_1, vx_1);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_2, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_2, vx_1);
    #else
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_0, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_0, vx_1);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_1, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_1, vx_1);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_2, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_2, vx_1);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_3, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_3, vx_1);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_4, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_4, vx_1);
    #endif

    xnn_simd_f32_t vsign_flip_0 = xnn_sll_f32(vk_0, 31);
    xnn_simd_f32_t vsign_flip_1 = xnn_sll_f32(vk_1, 31);
    vx_0 = xnn_xor_f32(vx_0, vsign_flip_0);
    vx_1 = xnn_xor_f32(vx_1, vsign_flip_1);

    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);

    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, vc4, vc3);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, vc4, vc3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc2);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc2);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc1);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc0);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc0);

    const xnn_simd_f32_t vy_res_0 = xnn_mul_f32(vx_0, vp_0);
    const xnn_simd_f32_t vy_res_1 = xnn_mul_f32(vx_1, vp_1);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_res_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_res_1);
    output += 2;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_mul_f32(vx, vone_over_pi);
    xnn_simd_f32_t vk = xnn_add_f32(vy, vmagic);
    xnn_simd_f32_t vk_float = xnn_sub_f32(vk, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
    #else
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_3, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_4, vx);
    #endif

    xnn_simd_f32_t vsign_flip = xnn_sll_f32(vk, 31);
    vx = xnn_xor_f32(vx, vsign_flip);

    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc4, vc3);
    vp = xnn_fmadd_f32(vx2, vp, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_storeu_f32(output, vy_res);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_mul_f32(vx, vone_over_pi);
    xnn_simd_f32_t vk = xnn_add_f32(vy, vmagic);
    xnn_simd_f32_t vk_float = xnn_sub_f32(vk, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
    #else
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_3, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_4, vx);
    #endif

    xnn_simd_f32_t vsign_flip = xnn_sll_f32(vk, 31);
    vx = xnn_xor_f32(vx, vsign_flip);

    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc4, vc3);
    vp = xnn_fmadd_f32(vx2, vp, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_tail_f32(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsin_ukernel__scalar_poly_4_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  XNN_SIMD_CONST_F32(vmagic, 12582912.0f);  // 1.5 * 2^23
  XNN_SIMD_CONST_F32(vone_over_pi, 0.31830988618379067154f);
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vpi_0, -3.14159274e+00f);
    XNN_SIMD_CONST_F32(vpi_1, 8.74227766e-08f);
    XNN_SIMD_CONST_F32(vpi_2, 3.43024902e-15f);
  #else
    XNN_SIMD_CONST_F32(vpi_0, -3.14062500e+00f);
    XNN_SIMD_CONST_F32(vpi_1, -9.67502594e-04f);
    XNN_SIMD_CONST_F32(vpi_2, -1.50990672e-07f);
    XNN_SIMD_CONST_F32(vpi_3, -5.12656584e-12f);
    XNN_SIMD_CONST_F32(vpi_4, -1.22514845e-16f);
  #endif

  // Monomial coefficients for sin(x)/x on [-pi/2, pi/2]
  XNN_SIMD_CONST_F32(vc4, 2.6052248359e-06f);
  XNN_SIMD_CONST_F32(vc3, -1.9809075457e-04f);
  XNN_SIMD_CONST_F32(vc2, 8.3330506459e-03f);
  XNN_SIMD_CONST_F32(vc1, -1.6666658223e-01f);
  XNN_SIMD_CONST_F32(vc0, 1.0000000000e+00f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4;

    xnn_simd_f32_t vy_0 = xnn_mul_f32(vx_0, vone_over_pi);
    xnn_simd_f32_t vy_1 = xnn_mul_f32(vx_1, vone_over_pi);
    xnn_simd_f32_t vy_2 = xnn_mul_f32(vx_2, vone_over_pi);
    xnn_simd_f32_t vy_3 = xnn_mul_f32(vx_3, vone_over_pi);

    xnn_simd_f32_t vk_0 = xnn_add_f32(vy_0, vmagic);
    xnn_simd_f32_t vk_1 = xnn_add_f32(vy_1, vmagic);
    xnn_simd_f32_t vk_2 = xnn_add_f32(vy_2, vmagic);
    xnn_simd_f32_t vk_3 = xnn_add_f32(vy_3, vmagic);

    xnn_simd_f32_t vk_float_0 = xnn_sub_f32(vk_0, vmagic);
    xnn_simd_f32_t vk_float_1 = xnn_sub_f32(vk_1, vmagic);
    xnn_simd_f32_t vk_float_2 = xnn_sub_f32(vk_2, vmagic);
    xnn_simd_f32_t vk_float_3 = xnn_sub_f32(vk_3, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_0, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_0, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_0, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_0, vx_3);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_1, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_1, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_1, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_1, vx_3);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_2, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_2, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_2, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_2, vx_3);
    #else
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_0, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_0, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_0, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_0, vx_3);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_1, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_1, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_1, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_1, vx_3);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_2, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_2, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_2, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_2, vx_3);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_3, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_3, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_3, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_3, vx_3);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_4, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_4, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_4, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_4, vx_3);
    #endif

    xnn_simd_f32_t vsign_flip_0 = xnn_sll_f32(vk_0, 31);
    xnn_simd_f32_t vsign_flip_1 = xnn_sll_f32(vk_1, 31);
    xnn_simd_f32_t vsign_flip_2 = xnn_sll_f32(vk_2, 31);
    xnn_simd_f32_t vsign_flip_3 = xnn_sll_f32(vk_3, 31);
    vx_0 = xnn_xor_f32(vx_0, vsign_flip_0);
    vx_1 = xnn_xor_f32(vx_1, vsign_flip_1);
    vx_2 = xnn_xor_f32(vx_2, vsign_flip_2);
    vx_3 = xnn_xor_f32(vx_3, vsign_flip_3);

    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);
    const xnn_simd_f32_t vx2_3 = xnn_mul_f32(vx_3, vx_3);

    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, vc4, vc3);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, vc4, vc3);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, vc4, vc3);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, vc4, vc3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc2);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc2);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vc2);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vc2);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc1);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vc1);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vc1);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc0);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc0);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vc0);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vc0);

    const xnn_simd_f32_t vy_res_0 = xnn_mul_f32(vx_0, vp_0);
    const xnn_simd_f32_t vy_res_1 = xnn_mul_f32(vx_1, vp_1);
    const xnn_simd_f32_t vy_res_2 = xnn_mul_f32(vx_2, vp_2);
    const xnn_simd_f32_t vy_res_3 = xnn_mul_f32(vx_3, vp_3);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_res_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_res_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_res_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_res_3);
    output += 4;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_mul_f32(vx, vone_over_pi);
    xnn_simd_f32_t vk = xnn_add_f32(vy, vmagic);
    xnn_simd_f32_t vk_float = xnn_sub_f32(vk, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
    #else
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_3, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_4, vx);
    #endif

    xnn_simd_f32_t vsign_flip = xnn_sll_f32(vk, 31);
    vx = xnn_xor_f32(vx, vsign_flip);

    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc4, vc3);
    vp = xnn_fmadd_f32(vx2, vp, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_storeu_f32(output, vy_res);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_mul_f32(vx, vone_over_pi);
    xnn_simd_f32_t vk = xnn_add_f32(vy, vmagic);
    xnn_simd_f32_t vk_float = xnn_sub_f32(vk, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
    #else
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_3, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_4, vx);
    #endif

    xnn_simd_f32_t vsign_flip = xnn_sll_f32(vk, 31);
    vx = xnn_xor_f32(vx, vsign_flip);

    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc4, vc3);
    vp = xnn_fmadd_f32(vx2, vp, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_tail_f32(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsin_ukernel__scalar_poly_4_u8(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  XNN_SIMD_CONST_F32(vmagic, 12582912.0f);  // 1.5 * 2^23
  XNN_SIMD_CONST_F32(vone_over_pi, 0.31830988618379067154f);
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vpi_0, -3.14159274e+00f);
    XNN_SIMD_CONST_F32(vpi_1, 8.74227766e-08f);
    XNN_SIMD_CONST_F32(vpi_2, 3.43024902e-15f);
  #else
    XNN_SIMD_CONST_F32(vpi_0, -3.14062500e+00f);
    XNN_SIMD_CONST_F32(vpi_1, -9.67502594e-04f);
    XNN_SIMD_CONST_F32(vpi_2, -1.50990672e-07f);
    XNN_SIMD_CONST_F32(vpi_3, -5.12656584e-12f);
    XNN_SIMD_CONST_F32(vpi_4, -1.22514845e-16f);
  #endif

  // Monomial coefficients for sin(x)/x on [-pi/2, pi/2]
  XNN_SIMD_CONST_F32(vc4, 2.6052248359e-06f);
  XNN_SIMD_CONST_F32(vc3, -1.9809075457e-04f);
  XNN_SIMD_CONST_F32(vc2, 8.3330506459e-03f);
  XNN_SIMD_CONST_F32(vc1, -1.6666658223e-01f);
  XNN_SIMD_CONST_F32(vc0, 1.0000000000e+00f);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_4 = xnn_loadu_f32(input + 4 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_5 = xnn_loadu_f32(input + 5 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_6 = xnn_loadu_f32(input + 6 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_7 = xnn_loadu_f32(input + 7 * xnn_simd_size_f32);
    input += 8;

    xnn_simd_f32_t vy_0 = xnn_mul_f32(vx_0, vone_over_pi);
    xnn_simd_f32_t vy_1 = xnn_mul_f32(vx_1, vone_over_pi);
    xnn_simd_f32_t vy_2 = xnn_mul_f32(vx_2, vone_over_pi);
    xnn_simd_f32_t vy_3 = xnn_mul_f32(vx_3, vone_over_pi);
    xnn_simd_f32_t vy_4 = xnn_mul_f32(vx_4, vone_over_pi);
    xnn_simd_f32_t vy_5 = xnn_mul_f32(vx_5, vone_over_pi);
    xnn_simd_f32_t vy_6 = xnn_mul_f32(vx_6, vone_over_pi);
    xnn_simd_f32_t vy_7 = xnn_mul_f32(vx_7, vone_over_pi);

    xnn_simd_f32_t vk_0 = xnn_add_f32(vy_0, vmagic);
    xnn_simd_f32_t vk_1 = xnn_add_f32(vy_1, vmagic);
    xnn_simd_f32_t vk_2 = xnn_add_f32(vy_2, vmagic);
    xnn_simd_f32_t vk_3 = xnn_add_f32(vy_3, vmagic);
    xnn_simd_f32_t vk_4 = xnn_add_f32(vy_4, vmagic);
    xnn_simd_f32_t vk_5 = xnn_add_f32(vy_5, vmagic);
    xnn_simd_f32_t vk_6 = xnn_add_f32(vy_6, vmagic);
    xnn_simd_f32_t vk_7 = xnn_add_f32(vy_7, vmagic);

    xnn_simd_f32_t vk_float_0 = xnn_sub_f32(vk_0, vmagic);
    xnn_simd_f32_t vk_float_1 = xnn_sub_f32(vk_1, vmagic);
    xnn_simd_f32_t vk_float_2 = xnn_sub_f32(vk_2, vmagic);
    xnn_simd_f32_t vk_float_3 = xnn_sub_f32(vk_3, vmagic);
    xnn_simd_f32_t vk_float_4 = xnn_sub_f32(vk_4, vmagic);
    xnn_simd_f32_t vk_float_5 = xnn_sub_f32(vk_5, vmagic);
    xnn_simd_f32_t vk_float_6 = xnn_sub_f32(vk_6, vmagic);
    xnn_simd_f32_t vk_float_7 = xnn_sub_f32(vk_7, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_0, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_0, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_0, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_0, vx_3);
      vx_4 = xnn_fmadd_f32(vk_float_4, vpi_0, vx_4);
      vx_5 = xnn_fmadd_f32(vk_float_5, vpi_0, vx_5);
      vx_6 = xnn_fmadd_f32(vk_float_6, vpi_0, vx_6);
      vx_7 = xnn_fmadd_f32(vk_float_7, vpi_0, vx_7);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_1, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_1, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_1, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_1, vx_3);
      vx_4 = xnn_fmadd_f32(vk_float_4, vpi_1, vx_4);
      vx_5 = xnn_fmadd_f32(vk_float_5, vpi_1, vx_5);
      vx_6 = xnn_fmadd_f32(vk_float_6, vpi_1, vx_6);
      vx_7 = xnn_fmadd_f32(vk_float_7, vpi_1, vx_7);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_2, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_2, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_2, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_2, vx_3);
      vx_4 = xnn_fmadd_f32(vk_float_4, vpi_2, vx_4);
      vx_5 = xnn_fmadd_f32(vk_float_5, vpi_2, vx_5);
      vx_6 = xnn_fmadd_f32(vk_float_6, vpi_2, vx_6);
      vx_7 = xnn_fmadd_f32(vk_float_7, vpi_2, vx_7);
    #else
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_0, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_0, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_0, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_0, vx_3);
      vx_4 = xnn_fmadd_f32(vk_float_4, vpi_0, vx_4);
      vx_5 = xnn_fmadd_f32(vk_float_5, vpi_0, vx_5);
      vx_6 = xnn_fmadd_f32(vk_float_6, vpi_0, vx_6);
      vx_7 = xnn_fmadd_f32(vk_float_7, vpi_0, vx_7);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_1, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_1, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_1, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_1, vx_3);
      vx_4 = xnn_fmadd_f32(vk_float_4, vpi_1, vx_4);
      vx_5 = xnn_fmadd_f32(vk_float_5, vpi_1, vx_5);
      vx_6 = xnn_fmadd_f32(vk_float_6, vpi_1, vx_6);
      vx_7 = xnn_fmadd_f32(vk_float_7, vpi_1, vx_7);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_2, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_2, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_2, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_2, vx_3);
      vx_4 = xnn_fmadd_f32(vk_float_4, vpi_2, vx_4);
      vx_5 = xnn_fmadd_f32(vk_float_5, vpi_2, vx_5);
      vx_6 = xnn_fmadd_f32(vk_float_6, vpi_2, vx_6);
      vx_7 = xnn_fmadd_f32(vk_float_7, vpi_2, vx_7);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_3, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_3, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_3, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_3, vx_3);
      vx_4 = xnn_fmadd_f32(vk_float_4, vpi_3, vx_4);
      vx_5 = xnn_fmadd_f32(vk_float_5, vpi_3, vx_5);
      vx_6 = xnn_fmadd_f32(vk_float_6, vpi_3, vx_6);
      vx_7 = xnn_fmadd_f32(vk_float_7, vpi_3, vx_7);
      vx_0 = xnn_fmadd_f32(vk_float_0, vpi_4, vx_0);
      vx_1 = xnn_fmadd_f32(vk_float_1, vpi_4, vx_1);
      vx_2 = xnn_fmadd_f32(vk_float_2, vpi_4, vx_2);
      vx_3 = xnn_fmadd_f32(vk_float_3, vpi_4, vx_3);
      vx_4 = xnn_fmadd_f32(vk_float_4, vpi_4, vx_4);
      vx_5 = xnn_fmadd_f32(vk_float_5, vpi_4, vx_5);
      vx_6 = xnn_fmadd_f32(vk_float_6, vpi_4, vx_6);
      vx_7 = xnn_fmadd_f32(vk_float_7, vpi_4, vx_7);
    #endif

    xnn_simd_f32_t vsign_flip_0 = xnn_sll_f32(vk_0, 31);
    xnn_simd_f32_t vsign_flip_1 = xnn_sll_f32(vk_1, 31);
    xnn_simd_f32_t vsign_flip_2 = xnn_sll_f32(vk_2, 31);
    xnn_simd_f32_t vsign_flip_3 = xnn_sll_f32(vk_3, 31);
    xnn_simd_f32_t vsign_flip_4 = xnn_sll_f32(vk_4, 31);
    xnn_simd_f32_t vsign_flip_5 = xnn_sll_f32(vk_5, 31);
    xnn_simd_f32_t vsign_flip_6 = xnn_sll_f32(vk_6, 31);
    xnn_simd_f32_t vsign_flip_7 = xnn_sll_f32(vk_7, 31);
    vx_0 = xnn_xor_f32(vx_0, vsign_flip_0);
    vx_1 = xnn_xor_f32(vx_1, vsign_flip_1);
    vx_2 = xnn_xor_f32(vx_2, vsign_flip_2);
    vx_3 = xnn_xor_f32(vx_3, vsign_flip_3);
    vx_4 = xnn_xor_f32(vx_4, vsign_flip_4);
    vx_5 = xnn_xor_f32(vx_5, vsign_flip_5);
    vx_6 = xnn_xor_f32(vx_6, vsign_flip_6);
    vx_7 = xnn_xor_f32(vx_7, vsign_flip_7);

    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);
    const xnn_simd_f32_t vx2_3 = xnn_mul_f32(vx_3, vx_3);
    const xnn_simd_f32_t vx2_4 = xnn_mul_f32(vx_4, vx_4);
    const xnn_simd_f32_t vx2_5 = xnn_mul_f32(vx_5, vx_5);
    const xnn_simd_f32_t vx2_6 = xnn_mul_f32(vx_6, vx_6);
    const xnn_simd_f32_t vx2_7 = xnn_mul_f32(vx_7, vx_7);

    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, vc4, vc3);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, vc4, vc3);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, vc4, vc3);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, vc4, vc3);
    xnn_simd_f32_t vp_4 = xnn_fmadd_f32(vx2_4, vc4, vc3);
    xnn_simd_f32_t vp_5 = xnn_fmadd_f32(vx2_5, vc4, vc3);
    xnn_simd_f32_t vp_6 = xnn_fmadd_f32(vx2_6, vc4, vc3);
    xnn_simd_f32_t vp_7 = xnn_fmadd_f32(vx2_7, vc4, vc3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc2);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc2);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vc2);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vc2);
    vp_4 = xnn_fmadd_f32(vx2_4, vp_4, vc2);
    vp_5 = xnn_fmadd_f32(vx2_5, vp_5, vc2);
    vp_6 = xnn_fmadd_f32(vx2_6, vp_6, vc2);
    vp_7 = xnn_fmadd_f32(vx2_7, vp_7, vc2);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc1);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vc1);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vc1);
    vp_4 = xnn_fmadd_f32(vx2_4, vp_4, vc1);
    vp_5 = xnn_fmadd_f32(vx2_5, vp_5, vc1);
    vp_6 = xnn_fmadd_f32(vx2_6, vp_6, vc1);
    vp_7 = xnn_fmadd_f32(vx2_7, vp_7, vc1);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc0);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc0);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, vc0);
    vp_3 = xnn_fmadd_f32(vx2_3, vp_3, vc0);
    vp_4 = xnn_fmadd_f32(vx2_4, vp_4, vc0);
    vp_5 = xnn_fmadd_f32(vx2_5, vp_5, vc0);
    vp_6 = xnn_fmadd_f32(vx2_6, vp_6, vc0);
    vp_7 = xnn_fmadd_f32(vx2_7, vp_7, vc0);

    const xnn_simd_f32_t vy_res_0 = xnn_mul_f32(vx_0, vp_0);
    const xnn_simd_f32_t vy_res_1 = xnn_mul_f32(vx_1, vp_1);
    const xnn_simd_f32_t vy_res_2 = xnn_mul_f32(vx_2, vp_2);
    const xnn_simd_f32_t vy_res_3 = xnn_mul_f32(vx_3, vp_3);
    const xnn_simd_f32_t vy_res_4 = xnn_mul_f32(vx_4, vp_4);
    const xnn_simd_f32_t vy_res_5 = xnn_mul_f32(vx_5, vp_5);
    const xnn_simd_f32_t vy_res_6 = xnn_mul_f32(vx_6, vp_6);
    const xnn_simd_f32_t vy_res_7 = xnn_mul_f32(vx_7, vp_7);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_res_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_res_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_res_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_res_3);
    xnn_storeu_f32(output + 4 * xnn_simd_size_f32, vy_res_4);
    xnn_storeu_f32(output + 5 * xnn_simd_size_f32, vy_res_5);
    xnn_storeu_f32(output + 6 * xnn_simd_size_f32, vy_res_6);
    xnn_storeu_f32(output + 7 * xnn_simd_size_f32, vy_res_7);
    output += 8;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_mul_f32(vx, vone_over_pi);
    xnn_simd_f32_t vk = xnn_add_f32(vy, vmagic);
    xnn_simd_f32_t vk_float = xnn_sub_f32(vk, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
    #else
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_3, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_4, vx);
    #endif

    xnn_simd_f32_t vsign_flip = xnn_sll_f32(vk, 31);
    vx = xnn_xor_f32(vx, vsign_flip);

    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc4, vc3);
    vp = xnn_fmadd_f32(vx2, vp, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_storeu_f32(output, vy_res);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_mul_f32(vx, vone_over_pi);
    xnn_simd_f32_t vk = xnn_add_f32(vy, vmagic);
    xnn_simd_f32_t vk_float = xnn_sub_f32(vk, vmagic);

    #if XNN_SIMD_HAS_NATIVE_FMA
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
    #else
      vx = xnn_fmadd_f32(vk_float, vpi_0, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_1, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_2, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_3, vx);
      vx = xnn_fmadd_f32(vk_float, vpi_4, vx);
    #endif

    xnn_simd_f32_t vsign_flip = xnn_sll_f32(vk, 31);
    vx = xnn_xor_f32(vx, vsign_flip);

    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc4, vc3);
    vp = xnn_fmadd_f32(vx2, vp, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_tail_f32(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
