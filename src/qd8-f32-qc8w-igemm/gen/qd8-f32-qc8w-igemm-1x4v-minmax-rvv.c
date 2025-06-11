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

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
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

  float* c0 = c;

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;


  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vint32m4_t vksum = __riscv_vle32_v_i32m4((const int32_t*)w, vl);
    const int32_t vinput_zero_point = quantization_params->zero_point;
    vint32m4_t vacc0 = __riscv_vmul_vx_i32m4(vksum, vinput_zero_point, vl);
    w = (const void*) ((const int32_t*) w + nr);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;

        const vint8m1_t vb = __riscv_vle8_v_i8m1((const int8_t*) w, vl);
        const vint16m2_t vb0 = __riscv_vsext_vf2(vb, vl);

        w = (const void*) ((const int8_t*) w + nr);

        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb0, vl);

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vfloat32m4_t vfpacc0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);

    const float vinput_scale = quantization_params->inv_scale;
    vfpacc0 = __riscv_vfmul_vf_f32m4(vfpacc0, vinput_scale, vl);

    const vfloat32m4_t vscale = __riscv_vle32_v_f32m4((const float*) w, vl);
    vfpacc0 = __riscv_vfmul_vv_f32m4(vfpacc0, vscale, vl);

    w = (const void*) ((const float*) w + nr);

    const vfloat32m4_t vbias = __riscv_vle32_v_f32m4((const float*) w, vl);
    vfpacc0 = __riscv_vfadd_vv_f32m4(vfpacc0, vbias, vl);

    w = (const void*) ((const float*) w + nr);

    const float voutput_min = params->scalar.min;
    vfpacc0 = __riscv_vfmax_vf_f32m4(vfpacc0, voutput_min, vl);

    const float voutput_max = params->scalar.max;
    vfpacc0 = __riscv_vfmin_vf_f32m4(vfpacc0, voutput_max, vl);

    __riscv_vse32_v_f32m4(c0, vfpacc0, vl);

    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const int8_t**restrict) ((uintptr_t) a - ks);

  } while (nc != 0);
}
