// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsin/f16-f32-poly-3.c.in
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

#include "src/xnnpack/simd/f32-wasmrelaxedsimd.h"

#undef XNN_SIMD_HAS_NATIVE_FMA
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"

#include "src/xnnpack/vunary.h"

// Helper functions for f16 <-> f32 conversion using xnn_simd_f32_t.
static XNN_INLINE xnn_simd_f32_t xnn_loadu_f16_f32(const xnn_float16* ptr) {
  return xnn_cvt_f32_f16(xnn_loadu_f16(ptr));
}

static XNN_INLINE void xnn_store_f32_f16(xnn_float16* ptr, xnn_simd_f32_t v) {
  xnn_store_tail_f16(ptr, xnn_cvt_f16_f32(v), xnn_simd_size_f32);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_f16_f32(const xnn_float16* ptr, size_t elements) {
  return xnn_cvt_f32_f16(xnn_load_tail_f16(ptr, elements));
}

static XNN_INLINE void xnn_store_f32_f16_tail(xnn_float16* ptr, xnn_simd_f32_t v, size_t elements) {
  xnn_store_tail_f16(ptr, xnn_cvt_f16_f32(v), elements);
}


void xnn_f16_f32acc_vsin_ukernel__wasmrelaxedsimd_poly_3_u4(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

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
  XNN_SIMD_CONST_F32(vc3, -1.9884618814e-04f);
  XNN_SIMD_CONST_F32(vc2, 8.3336085081e-03f);
  XNN_SIMD_CONST_F32(vc1, -1.6666665673e-01f);
  XNN_SIMD_CONST_F32(vc0, 1.0000000000e+00f);


  for (; batch >= 4 * sizeof(xnn_float16); batch -= 4 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 4;

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
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc3, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_f32_f16(output, vy_res);
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_f16_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

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
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc3, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_f32_f16_tail(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_f32acc_vsin_ukernel__wasmrelaxedsimd_poly_3_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

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
  XNN_SIMD_CONST_F32(vc3, -1.9884618814e-04f);
  XNN_SIMD_CONST_F32(vc2, 8.3336085081e-03f);
  XNN_SIMD_CONST_F32(vc1, -1.6666665673e-01f);
  XNN_SIMD_CONST_F32(vc0, 1.0000000000e+00f);

  for (; batch >= 8 * sizeof(xnn_float16); batch -= 8 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f16_f32(input + 0 * 4);
    xnn_simd_f32_t vx_1 = xnn_loadu_f16_f32(input + 1 * 4);
    input += 8;

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

    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, vc3, vc2);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, vc3, vc2);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc1);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, vc0);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, vc0);

    const xnn_simd_f32_t vy_res_0 = xnn_mul_f32(vx_0, vp_0);
    const xnn_simd_f32_t vy_res_1 = xnn_mul_f32(vx_1, vp_1);

    xnn_store_f32_f16(output + 0 * 4, vy_res_0);
    xnn_store_f32_f16(output + 1 * 4, vy_res_1);
    output += 8;
  }

  for (; batch >= 4 * sizeof(xnn_float16); batch -= 4 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 4;

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
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc3, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_f32_f16(output, vy_res);
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_f16_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

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
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc3, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_f32_f16_tail(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_f32acc_vsin_ukernel__wasmrelaxedsimd_poly_3_u16(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

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
  XNN_SIMD_CONST_F32(vc3, -1.9884618814e-04f);
  XNN_SIMD_CONST_F32(vc2, 8.3336085081e-03f);
  XNN_SIMD_CONST_F32(vc1, -1.6666665673e-01f);
  XNN_SIMD_CONST_F32(vc0, 1.0000000000e+00f);

  for (; batch >= 16 * sizeof(xnn_float16); batch -= 16 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f16_f32(input + 0 * 4);
    xnn_simd_f32_t vx_1 = xnn_loadu_f16_f32(input + 1 * 4);
    xnn_simd_f32_t vx_2 = xnn_loadu_f16_f32(input + 2 * 4);
    xnn_simd_f32_t vx_3 = xnn_loadu_f16_f32(input + 3 * 4);
    input += 16;

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

    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, vc3, vc2);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, vc3, vc2);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, vc3, vc2);
    xnn_simd_f32_t vp_3 = xnn_fmadd_f32(vx2_3, vc3, vc2);
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

    xnn_store_f32_f16(output + 0 * 4, vy_res_0);
    xnn_store_f32_f16(output + 1 * 4, vy_res_1);
    xnn_store_f32_f16(output + 2 * 4, vy_res_2);
    xnn_store_f32_f16(output + 3 * 4, vy_res_3);
    output += 16;
  }

  for (; batch >= 4 * sizeof(xnn_float16); batch -= 4 * sizeof(xnn_float16)) {
    xnn_simd_f32_t vx = xnn_loadu_f16_f32(input);
    input += 4;

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
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc3, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_f32_f16(output, vy_res);
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_f16_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);

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
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, vc3, vc2);
    vp = xnn_fmadd_f32(vx2, vp, vc1);
    vp = xnn_fmadd_f32(vx2, vp, vc0);

    const xnn_simd_f32_t vy_res = xnn_mul_f32(vx, vp);
    xnn_store_f32_f16_tail(output, vy_res, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}
