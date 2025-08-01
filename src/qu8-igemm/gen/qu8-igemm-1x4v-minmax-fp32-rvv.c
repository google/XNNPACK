// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Microchip, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/igemm.h"

void xnn_qu8_igemm_minmax_fp32_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;

  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point;
  const float vscale = params->fp32_scalar.scale;
  const int32_t output_zero_point = params->fp32_scalar.output_zero_point;

  const int32_t vb_zero_point = params->fp32_scalar.kernel_zero_point;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vint32m4_t vacc0 = __riscv_vle32_v_i32m4((const int32_t*)w, vl);
    w = (const void*) ((const int32_t*) w + nr);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) (uint32_t) *a0++;

        const vuint8m1_t vb = __riscv_vle8_v_u8m1((const uint8_t*) w, vl);
        const vint16m2_t vb0 = __riscv_vsub_vx_i16m2(__riscv_vreinterpret_i16m2(__riscv_vzext_vf2(vb, vl)), vb_zero_point, vl);

        w = (const void*) ((const uint8_t*) w + nr);

        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb0, vl);

        k -= sizeof(uint8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vfloat32m4_t vfpacc0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);

    vfpacc0 = __riscv_vfmul_vf_f32m4(vfpacc0, vscale, vl);

    vfpacc0 = __riscv_vfmax_vf_f32m4(vfpacc0, output_min_less_zero_point, vl);
    vfpacc0 = __riscv_vfmin_vf_f32m4(vfpacc0, output_max_less_zero_point, vl);

    vint16m2_t vout0 = __riscv_vfncvt_x(vfpacc0, vl);

    vout0 = __riscv_vadd_vx_i16m2(vout0, (int16_t) output_zero_point, vl);

    vuint8m1_t vout80 = __riscv_vncvt_x_x_w_u8m1(__riscv_vreinterpret_u16m2(vout0), vl);

    __riscv_vse8_v_u8m1(c0, vout80, vl);

    c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

    a = (const uint8_t**restrict) ((uintptr_t) a - ks);

  } while (nc != 0);
}
