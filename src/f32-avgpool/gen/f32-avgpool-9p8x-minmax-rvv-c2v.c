// Auto-generated file. Do not edit!
//   Template: src/f32-avgpool/rvv_9p8x.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include "xnnpack/avgpool.h"
#include <riscv_vector.h>

void xnn_f32_avgpool_minmax_ukernel_9p8x__rvv_c2v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);
  assert((input_offset & 3) == 0);

  input_offset >>= XNN_LOG2_SIZEOF_FLOAT;

  do {
    {
      const float *i[9];
      for (size_t kk = 0; kk < 9; ++kk) {
        assert(input[kk] != NULL);
        i[kk] = (input[kk] != zero ? input[kk] + input_offset : zero) ;
      }
      input += 9;

      float* b = buffer;
      for (size_t c = channels; c != 0; ) {
        int32_t n = __riscv_vsetvl_e32m2(c);
        c -= n;

        vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i[0], n); i[0] += n;
        vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i[1], n); i[1] += n;
        vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i[2], n); i[2] += n;
        vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i[3], n); i[3] += n;
        vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i[4], n); i[4] += n;
        vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i[5], n); i[5] += n;
        vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i[6], n); i[6] += n;
        vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i[7], n); i[7] += n;
        vfloat32m2_t i8_f32v = __riscv_vle32_v_f32m2(i[8], n); i[8] += n;

        vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(i0_f32v, i1_f32v, n);
        vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(i2_f32v, i3_f32v, n);
        vfloat32m2_t sum45_f32v = __riscv_vfadd_vv_f32m2(i4_f32v, i5_f32v, n);
        vfloat32m2_t sum67_f32v = __riscv_vfadd_vv_f32m2(i6_f32v, i7_f32v, n);
        vfloat32m2_t sum018_f32v = __riscv_vfadd_vv_f32m2(sum01_f32v, i8_f32v, n);
        vfloat32m2_t sum2345_f32v = __riscv_vfadd_vv_f32m2(sum23_f32v, sum45_f32v, n);
        vfloat32m2_t sum01678_f32v = __riscv_vfadd_vv_f32m2(sum018_f32v, sum67_f32v, n);
        vfloat32m2_t sum_f32v = __riscv_vfadd_vv_f32m2(sum2345_f32v, sum01678_f32v, n);
        __riscv_vse32_v_f32m2(b, sum_f32v, n); b += n;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float *i[8];
      for (size_t kk = 0; kk < 8; ++kk) {
        assert(input[kk] != NULL);
        i[kk] = (input[kk] != zero ? input[kk] + input_offset : zero) ;
      }
      input += 8;

      float* b = buffer;
      for (size_t c = channels; c != 0; ) {
        int32_t n = __riscv_vsetvl_e32m2(c);
        c -= n;

        vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i[0], n); i[0] += n;
        vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i[1], n); i[1] += n;
        vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i[2], n); i[2] += n;
        vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i[3], n); i[3] += n;
        vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i[4], n); i[4] += n;
        vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i[5], n); i[5] += n;
        vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i[6], n); i[6] += n;
        vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i[7], n); i[7] += n;
        vfloat32m2_t vacc_f32v = __riscv_vle32_v_f32m2(b, n);

        vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(i0_f32v, i1_f32v, n);
        vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(i2_f32v, i3_f32v, n);
        vfloat32m2_t sum45_f32v = __riscv_vfadd_vv_f32m2(i4_f32v, i5_f32v, n);
        vfloat32m2_t sum67_f32v = __riscv_vfadd_vv_f32m2(i6_f32v, i7_f32v, n);
        vfloat32m2_t sum01a_f32v = __riscv_vfadd_vv_f32m2(sum01_f32v, vacc_f32v, n);
        vfloat32m2_t sum2345_f32v = __riscv_vfadd_vv_f32m2(sum23_f32v, sum45_f32v, n);
        vfloat32m2_t sum0167a_f32v = __riscv_vfadd_vv_f32m2(sum01a_f32v, sum67_f32v, n);
        vfloat32m2_t sum_f32v = __riscv_vfadd_vv_f32m2(sum2345_f32v, sum0167a_f32v, n);
        __riscv_vse32_v_f32m2(b, sum_f32v, n); b += n;
      }
    }

    {
      const float *i[8];
      for (size_t kk = 0; kk < k; ++kk) {
        assert(input[kk] != NULL);
        i[kk] = (input[kk] != zero ? input[kk] + input_offset : zero) ;
      }
      for (size_t tail = k; tail < 8; ++tail) {
        i[tail] = zero;
      }
      input = (const float**) ((uintptr_t) input + input_increment);

      const float scale = params->scalar.scale;
      const float min = params->scalar.min;
      const float max = params->scalar.max;
      float* b = buffer;

      for (size_t c = channels; c != 0; ) {
        int32_t n = __riscv_vsetvl_e32m2(c);
        c -= n;

        vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i[0], n); i[0] += n;
        vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i[1], n); i[1] += n;
        vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i[2], n); i[2] += n;
        vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i[3], n); i[3] += n;
        vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i[4], n); i[4] += n;
        vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i[5], n); i[5] += n;
        vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i[6], n); i[6] += n;
        vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i[7], n); i[7] += n;
        vfloat32m2_t vacc_f32v = __riscv_vle32_v_f32m2(b, n); b += n;

        vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(i0_f32v, i1_f32v, n);
        vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(i2_f32v, i3_f32v, n);
        vfloat32m2_t sum45_f32v = __riscv_vfadd_vv_f32m2(i4_f32v, i5_f32v, n);
        vfloat32m2_t sum67_f32v = __riscv_vfadd_vv_f32m2(i6_f32v, i7_f32v, n);
        vfloat32m2_t sum01a_f32v = __riscv_vfadd_vv_f32m2(sum01_f32v, vacc_f32v, n);
        vfloat32m2_t sum2345_f32v = __riscv_vfadd_vv_f32m2(sum23_f32v, sum45_f32v, n);
        vfloat32m2_t sum0167a_f32v = __riscv_vfadd_vv_f32m2(sum01a_f32v, sum67_f32v, n);
        vfloat32m2_t sum_f32v = __riscv_vfadd_vv_f32m2(sum2345_f32v, sum0167a_f32v, n);
        vfloat32m2_t out_f32v = __riscv_vfmul_vf_f32m2(sum_f32v, scale, n);
        out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, min, n), max, n);
        __riscv_vse32_v_f32m2(output, out_f32v, n); output += n;
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
