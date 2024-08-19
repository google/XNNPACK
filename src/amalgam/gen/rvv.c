// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/pavgpool.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "xnnpack/transpose.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/vunary.h"


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

void xnn_f32_avgpool_minmax_ukernel_9x__rvv_c2v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);
  assert((input_offset & 3) == 0);

  input_offset >>= XNN_LOG2_SIZEOF_FLOAT;

  const float scale = params->scalar.scale;
  const float min = params->scalar.min;
  const float max = params->scalar.max;

  do {
    const float *i[9];
    for (size_t kk = 0; kk < kernel_elements; ++kk) {
      assert(input[kk] != NULL);
      i[kk] = (input[kk] != zero ? input[kk] + input_offset : zero) ;
    }
    for (size_t tail = kernel_elements; tail < 9; ++tail) {
      i[tail] = zero;
    }
    input = (const float**) ((uintptr_t) input + input_increment);

    for (size_t c = channels; c != 0; ) {
      int32_t n = __riscv_vsetvl_e32m2(c);

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
      vfloat32m2_t out_f32v = __riscv_vfmul_vf_f32m2(sum_f32v, scale, n);
      out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, min, n), max, n);
      __riscv_vse32_v_f32m2(output, out_f32v, n); output += n;

      c -= n;
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_gavgpool_minmax_ukernel_7p7x__rvv_c2v(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - channels * sizeof(float);

  float* b = buffer;
  for (size_t c = channels; c != 0; ) {
    int32_t n = __riscv_vsetvl_e32m2(c);

    vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
    vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, n); i1 += n;
    vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, n); i2 += n;
    vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, n); i3 += n;
    vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, n); i4 += n;
    vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, n); i5 += n;
    vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, n); i6 += n;

    vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(i0_f32v, i1_f32v, n);
    vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(i2_f32v, i3_f32v, n);
    vfloat32m2_t sum45_f32v = __riscv_vfadd_vv_f32m2(i4_f32v, i5_f32v, n);
    vfloat32m2_t sum016_f32v = __riscv_vfadd_vv_f32m2(sum01_f32v, i6_f32v, n);
    vfloat32m2_t sum2345_f32v = __riscv_vfadd_vv_f32m2(sum23_f32v, sum45_f32v, n);
    vfloat32m2_t sum_f32v = __riscv_vfadd_vv_f32m2(sum2345_f32v, sum016_f32v, n);
    __riscv_vse32_v_f32m2(b, sum_f32v, n); b += n;

    c -= n;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    b = buffer;

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    i4 = (const float*) ((uintptr_t) i4 + input_increment);
    i5 = (const float*) ((uintptr_t) i5 + input_increment);
    i6 = (const float*) ((uintptr_t) i6 + input_increment);

    for (size_t c = channels; c != 0; ) {
      int32_t n = __riscv_vsetvl_e32m2(c);

      vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
      vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, n); i1 += n;
      vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, n); i2 += n;
      vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, n); i3 += n;
      vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, n); i4 += n;
      vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, n); i5 += n;
      vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, n); i6 += n;
      vfloat32m2_t vacc_f32v = __riscv_vle32_v_f32m2(b, n);

      vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(i0_f32v, i1_f32v, n);
      vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(i2_f32v, i3_f32v, n);
      vfloat32m2_t sum45_f32v = __riscv_vfadd_vv_f32m2(i4_f32v, i5_f32v, n);
      vfloat32m2_t sum6a_f32v = __riscv_vfadd_vv_f32m2(i6_f32v, vacc_f32v, n);
      vfloat32m2_t sum0123_f32v = __riscv_vfadd_vv_f32m2(sum01_f32v, sum23_f32v, n);
      vfloat32m2_t sum456a_f32v = __riscv_vfadd_vv_f32m2(sum45_f32v, sum6a_f32v, n);
      vfloat32m2_t sum_f32v = __riscv_vfadd_vv_f32m2(sum0123_f32v, sum456a_f32v, n);
      __riscv_vse32_v_f32m2(b, sum_f32v, n); b += n;

      c -= n;
    }
  }

  i0 = (const float*) ((uintptr_t) i0 + input_increment);
  i1 = (const float*) ((uintptr_t) i1 + input_increment);
  if (rows < 2) {
    i1 = zero;
  }
  i2 = (const float*) ((uintptr_t) i2 + input_increment);
  if (rows <= 2) {
    i2 = zero;
  }
  i3 = (const float*) ((uintptr_t) i3 + input_increment);
  if (rows < 4) {
    i3 = zero;
  }
  i4 = (const float*) ((uintptr_t) i4 + input_increment);
  if (rows <= 4) {
    i4 = zero;
  }
  i5 = (const float*) ((uintptr_t) i5 + input_increment);
  if (rows < 6) {
    i5 = zero;
  }
  i6 = (const float*) ((uintptr_t) i6 + input_increment);
  if (rows <= 6) {
    i6 = zero;
  }
  const float scale = params->scalar.scale;
  const float min = params->scalar.min;
  const float max = params->scalar.max;

  b = buffer;
  for (; channels != 0; ) {
    int32_t n = __riscv_vsetvl_e32m2(channels);

    vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
    vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, n); i1 += n;
    vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, n); i2 += n;
    vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, n); i3 += n;
    vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, n); i4 += n;
    vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, n); i5 += n;
    vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, n); i6 += n;
    vfloat32m2_t vacc_f32v = __riscv_vle32_v_f32m2(b, n); b += n;

    vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(i0_f32v, i1_f32v, n);
    vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(i2_f32v, i3_f32v, n);
    vfloat32m2_t sum45_f32v = __riscv_vfadd_vv_f32m2(i4_f32v, i5_f32v, n);
    vfloat32m2_t sum6a_f32v = __riscv_vfadd_vv_f32m2(i6_f32v, vacc_f32v, n);
    vfloat32m2_t sum0123_f32v = __riscv_vfadd_vv_f32m2(sum01_f32v, sum23_f32v, n);
    vfloat32m2_t sum456a_f32v = __riscv_vfadd_vv_f32m2(sum45_f32v, sum6a_f32v, n);
    vfloat32m2_t sum_f32v = __riscv_vfadd_vv_f32m2(sum0123_f32v, sum456a_f32v, n);
    vfloat32m2_t out_f32v = __riscv_vfmul_vf_f32m2(sum_f32v, scale, n);
    out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, min, n), max, n);
    __riscv_vse32_v_f32m2(output, out_f32v, n); output += n;

    channels -= n;
  }
}

void xnn_f32_gavgpool_minmax_ukernel_7x__rvv_c2v(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  if (rows < 2) {
    i1 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  if (rows <= 2) {
    i2 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  if (rows < 4) {
    i3 = zero;
  }
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  if (rows <= 4) {
    i4 = zero;
  }
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  if (rows < 6) {
    i5 = zero;
  }
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  if (rows <= 6) {
    i6 = zero;
  }

  const float scale = params->scalar.scale;
  const float min = params->scalar.min;
  const float max = params->scalar.max;
  for (; channels != 0; ) {
    int32_t n = __riscv_vsetvl_e32m2(channels);

    vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
    vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, n); i1 += n;
    vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, n); i2 += n;
    vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, n); i3 += n;
    vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, n); i4 += n;
    vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, n); i5 += n;
    vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, n); i6 += n;

    vfloat32m2_t sum01_f32v = __riscv_vfadd_vv_f32m2(i0_f32v, i1_f32v, n);
    vfloat32m2_t sum23_f32v = __riscv_vfadd_vv_f32m2(i2_f32v, i3_f32v, n);
    vfloat32m2_t sum45_f32v = __riscv_vfadd_vv_f32m2(i4_f32v, i5_f32v, n);
    vfloat32m2_t sum016_f32v = __riscv_vfadd_vv_f32m2(sum01_f32v, i6_f32v, n);
    vfloat32m2_t sum2345_f32v = __riscv_vfadd_vv_f32m2(sum23_f32v, sum45_f32v, n);
    vfloat32m2_t sum_f32v = __riscv_vfadd_vv_f32m2(sum2345_f32v, sum016_f32v, n);
    vfloat32m2_t out_f32v = __riscv_vfmul_vf_f32m2(sum_f32v, scale, n);
    out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, min, n), max, n);
    __riscv_vse32_v_f32m2(output, out_f32v, n); output += n;

    channels -= n;
  }
}

void xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_c2v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  do {
    float* o = output;
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      do {
        int32_t n = __riscv_vsetvl_e32m2(c);

        vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
        vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, n); i1 += n;
        vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, n); i2 += n;
        vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, n); i3 += n;
        vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, n); i4 += n;
        vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, n); i5 += n;
        vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, n); i6 += n;
        vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i7, n); i7 += n;
        vfloat32m2_t i8_f32v = __riscv_vle32_v_f32m2(i8, n); i8 += n;

        vfloat32m2_t max01_f32v = __riscv_vfmax_vv_f32m2(i0_f32v, i1_f32v, n);
        vfloat32m2_t max23_f32v = __riscv_vfmax_vv_f32m2(i2_f32v, i3_f32v, n);
        vfloat32m2_t max45_f32v = __riscv_vfmax_vv_f32m2(i4_f32v, i5_f32v, n);
        vfloat32m2_t max67_f32v = __riscv_vfmax_vv_f32m2(i6_f32v, i7_f32v, n);
        vfloat32m2_t max018_f32v = __riscv_vfmax_vv_f32m2(max01_f32v, i8_f32v, n);

        vfloat32m2_t max2345_f32v = __riscv_vfmax_vv_f32m2(max23_f32v, max45_f32v, n);
        vfloat32m2_t max01678_f32v = __riscv_vfmax_vv_f32m2(max67_f32v, max018_f32v, n);
        vfloat32m2_t out_f32v = __riscv_vfmax_vv_f32m2(max2345_f32v, max01678_f32v, n);
        out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, output_min, n), output_max, n);
        __riscv_vse32_v_f32m2(o, out_f32v, n); o += n;

        c -= n;
      } while (c != 0);
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      do {
        int32_t n = __riscv_vsetvl_e32m2(c);

        vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
        vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, n); i1 += n;
        vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, n); i2 += n;
        vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, n); i3 += n;
        vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, n); i4 += n;
        vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, n); i5 += n;
        vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, n); i6 += n;
        vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i7, n); i7 += n;
        vfloat32m2_t i8_f32v = __riscv_vle32_v_f32m2(o, n);

        vfloat32m2_t max01_f32v = __riscv_vfmax_vv_f32m2(i0_f32v, i1_f32v, n);
        vfloat32m2_t max23_f32v = __riscv_vfmax_vv_f32m2(i2_f32v, i3_f32v, n);
        vfloat32m2_t max45_f32v = __riscv_vfmax_vv_f32m2(i4_f32v, i5_f32v, n);
        vfloat32m2_t max67_f32v = __riscv_vfmax_vv_f32m2(i6_f32v, i7_f32v, n);
        vfloat32m2_t max018_f32v = __riscv_vfmax_vv_f32m2(max01_f32v, i8_f32v, n);

        vfloat32m2_t max2345_f32v = __riscv_vfmax_vv_f32m2(max23_f32v, max45_f32v, n);
        vfloat32m2_t max01678_f32v = __riscv_vfmax_vv_f32m2(max67_f32v, max018_f32v, n);
        vfloat32m2_t out_f32v = __riscv_vfmax_vv_f32m2(max2345_f32v, max01678_f32v, n);
        out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, output_min, n), output_max, n);
        __riscv_vse32_v_f32m2(o, out_f32v, n); o += n;

        c -= n;
      } while (c != 0);
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_pavgpool_minmax_ukernel_9p8x__rvv_c1v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    const float* multiplier,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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
        int32_t n = __riscv_vsetvl_e32m1(c);

        vfloat32m1_t i0_f32v = __riscv_vle32_v_f32m1(i[0], n); i[0] += n;
        vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i[1], n); i[1] += n;
        vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i[2], n); i[2] += n;
        vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i[3], n); i[3] += n;
        vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i[4], n); i[4] += n;
        vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i[5], n); i[5] += n;
        vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i[6], n); i[6] += n;
        vfloat32m1_t i7_f32v = __riscv_vle32_v_f32m1(i[7], n); i[7] += n;
        vfloat32m1_t i8_f32v = __riscv_vle32_v_f32m1(i[8], n); i[8] += n;

        vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(i0_f32v, i1_f32v, n);
        vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(i2_f32v, i3_f32v, n);
        vfloat32m1_t sum45_f32v = __riscv_vfadd_vv_f32m1(i4_f32v, i5_f32v, n);
        vfloat32m1_t sum67_f32v = __riscv_vfadd_vv_f32m1(i6_f32v, i7_f32v, n);
        vfloat32m1_t sum018_f32v = __riscv_vfadd_vv_f32m1(sum01_f32v, i8_f32v, n);
        vfloat32m1_t sum2345_f32v = __riscv_vfadd_vv_f32m1(sum23_f32v, sum45_f32v, n);
        vfloat32m1_t sum01678_f32v = __riscv_vfadd_vv_f32m1(sum018_f32v, sum67_f32v, n);
        vfloat32m1_t sum_f32v = __riscv_vfadd_vv_f32m1(sum2345_f32v, sum01678_f32v, n);
        __riscv_vse32_v_f32m1(b, sum_f32v, n); b += n;

        c -= n;
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
        int32_t n = __riscv_vsetvl_e32m1(c);

        vfloat32m1_t i0_f32v = __riscv_vle32_v_f32m1(i[0], n); i[0] += n;
        vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i[1], n); i[1] += n;
        vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i[2], n); i[2] += n;
        vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i[3], n); i[3] += n;
        vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i[4], n); i[4] += n;
        vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i[5], n); i[5] += n;
        vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i[6], n); i[6] += n;
        vfloat32m1_t i7_f32v = __riscv_vle32_v_f32m1(i[7], n); i[7] += n;
        vfloat32m1_t vacc_f32v = __riscv_vle32_v_f32m1(b, n);

        vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(i0_f32v, i1_f32v, n);
        vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(i2_f32v, i3_f32v, n);
        vfloat32m1_t sum45_f32v = __riscv_vfadd_vv_f32m1(i4_f32v, i5_f32v, n);
        vfloat32m1_t sum67_f32v = __riscv_vfadd_vv_f32m1(i6_f32v, i7_f32v, n);
        vfloat32m1_t sum01a_f32v = __riscv_vfadd_vv_f32m1(sum01_f32v, vacc_f32v, n);
        vfloat32m1_t sum2345_f32v = __riscv_vfadd_vv_f32m1(sum23_f32v, sum45_f32v, n);
        vfloat32m1_t sum0167a_f32v = __riscv_vfadd_vv_f32m1(sum01a_f32v, sum67_f32v, n);
        vfloat32m1_t sum_f32v = __riscv_vfadd_vv_f32m1(sum2345_f32v, sum0167a_f32v, n);
        __riscv_vse32_v_f32m1(b, sum_f32v, n); b += n;

        c -= n;
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

      const float mult = *multiplier++;

      const float min = params->scalar.min;
      const float max = params->scalar.max;
      float* b = buffer;
      for (size_t c = channels; c != 0; ) {
        int32_t n = __riscv_vsetvl_e32m1(c);

        vfloat32m1_t i0_f32v = __riscv_vle32_v_f32m1(i[0], n); i[0] += n;
        vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i[1], n); i[1] += n;
        vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i[2], n); i[2] += n;
        vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i[3], n); i[3] += n;
        vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i[4], n); i[4] += n;
        vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i[5], n); i[5] += n;
        vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i[6], n); i[6] += n;
        vfloat32m1_t i7_f32v = __riscv_vle32_v_f32m1(i[7], n); i[7] += n;
        vfloat32m1_t vacc_f32v = __riscv_vle32_v_f32m1(b, n); b += n;

        vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(i0_f32v, i1_f32v, n);
        vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(i2_f32v, i3_f32v, n);
        vfloat32m1_t sum45_f32v = __riscv_vfadd_vv_f32m1(i4_f32v, i5_f32v, n);
        vfloat32m1_t sum67_f32v = __riscv_vfadd_vv_f32m1(i6_f32v, i7_f32v, n);
        vfloat32m1_t sum01a_f32v = __riscv_vfadd_vv_f32m1(sum01_f32v, vacc_f32v, n);
        vfloat32m1_t sum2345_f32v = __riscv_vfadd_vv_f32m1(sum23_f32v, sum45_f32v, n);
        vfloat32m1_t sum0167a_f32v = __riscv_vfadd_vv_f32m1(sum01a_f32v, sum67_f32v, n);
        vfloat32m1_t sum_f32v = __riscv_vfadd_vv_f32m1(sum2345_f32v, sum0167a_f32v, n);
        vfloat32m1_t out_f32v = __riscv_vfmul_vf_f32m1(sum_f32v, mult, n);
        out_f32v = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(out_f32v, min, n), max, n);
        __riscv_vse32_v_f32m1(output, out_f32v, n); output += n;

        c -= n;
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_pavgpool_minmax_ukernel_9x__rvv_c1v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    const float* multiplier,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);
  assert((input_offset & 3) == 0);

  input_offset >>= XNN_LOG2_SIZEOF_FLOAT;

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;

  do {
    const float *i[9];
    for (size_t kk = 0; kk < kernel_elements; ++kk) {
      assert(input[kk] != NULL);
      i[kk] = (input[kk] != zero ? input[kk] + input_offset : zero) ;
    }
    for (size_t tail = kernel_elements; tail < 9; ++tail) {
      i[tail] = zero;
    }

    input = (const float**) ((uintptr_t) input + input_increment);

    const float mult = *multiplier++;

    for (size_t c = channels; c != 0; ) {
      int32_t n = __riscv_vsetvl_e32m1(c);

      vfloat32m1_t i0_f32v = __riscv_vle32_v_f32m1(i[0], n); i[0] += n;
      vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i[1], n); i[1] += n;
      vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i[2], n); i[2] += n;
      vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i[3], n); i[3] += n;
      vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i[4], n); i[4] += n;
      vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i[5], n); i[5] += n;
      vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i[6], n); i[6] += n;
      vfloat32m1_t i7_f32v = __riscv_vle32_v_f32m1(i[7], n); i[7] += n;
      vfloat32m1_t i8_f32v = __riscv_vle32_v_f32m1(i[8], n); i[8] += n;

      vfloat32m1_t sum01_f32v = __riscv_vfadd_vv_f32m1(i0_f32v, i1_f32v, n);
      vfloat32m1_t sum23_f32v = __riscv_vfadd_vv_f32m1(i2_f32v, i3_f32v, n);
      vfloat32m1_t sum45_f32v = __riscv_vfadd_vv_f32m1(i4_f32v, i5_f32v, n);
      vfloat32m1_t sum67_f32v = __riscv_vfadd_vv_f32m1(i6_f32v, i7_f32v, n);
      vfloat32m1_t sum018_f32v = __riscv_vfadd_vv_f32m1(sum01_f32v, i8_f32v, n);
      vfloat32m1_t sum2345_f32v = __riscv_vfadd_vv_f32m1(sum23_f32v, sum45_f32v, n);
      vfloat32m1_t sum01678_f32v = __riscv_vfadd_vv_f32m1(sum018_f32v, sum67_f32v, n);
      vfloat32m1_t sum_f32v = __riscv_vfadd_vv_f32m1(sum2345_f32v, sum01678_f32v, n);
      vfloat32m1_t out_f32v = __riscv_vfmul_vf_f32m1(sum_f32v, mult, n);
      out_f32v = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(out_f32v, output_min, n), output_max, n);
      __riscv_vse32_v_f32m1(output, out_f32v, n); output += n;

      c -= n;
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

static inline vfloat32m4_t eval_poly_horner(vfloat32m4_t x,
                                                  float c6, float c5,
                                                  float c4, float c3, float c2,
                                                  float c1, float c0, size_t vl) {
  vfloat32m4_t z;
  vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(c5, vl);
  y = __riscv_vfmacc_vf_f32m4(y, c6, x, vl);

  z = __riscv_vfmv_v_f_f32m4(c4, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m4(c3, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m4(c2, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m4(c1, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m4(c0, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);
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
static inline vfloat32m4_t softexp_f32m4(
    vfloat32m4_t x, size_t vl,
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
  x = __riscv_vfmax_vf_f32m4(x, xmin, vl);

  // 0. Reduction x = s * q ln(2)
  // const float r_ln2f = 0x1.715476p0f;  // single(1/log(2));
  // const float l2uf = 0x1.62e4p-1f;     // round(log(2), 24-8, RN);
  // const float l2lf = 0x1.7f7d1cp-20f;  // round(log(2) - l2uf, single, RN);
  vfloat32m4_t v = __riscv_vfmul_vf_f32m4(x, r_ln2f, vl);

  vint16m2_t q = __riscv_vfncvt_x_f_w_i16m2(v, vl);
  vfloat32m4_t z = __riscv_vfwcvt_f_x_v_f32m4(q, vl);

  // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
  vfloat32m4_t s = __riscv_vfnmsac_vf_f32m4(x, l2uf, z, vl);
  s = __riscv_vfnmsac_vf_f32m4(s, l2lf, z, vl);

  // 1. Approximate e^s by degree-6 polynomial approximation
  vfloat32m4_t u = eval_poly_horner(s, c6, c5, c4, c3, c2, 1.0f, 1.0f, vl);

  // 2. Reconstruction: compute u = u*2^q
  const int16_t p = (24 - 1);
  const int16_t bias = (128 - 1);
  vint32m4_t qw = __riscv_vwadd_vx_i32m4(q, bias, vl);
  vint32m4_t qq = __riscv_vsll_vx_i32m4(qw, p, vl);
  vfloat32m4_t qf = __riscv_vreinterpret_v_i32m4_f32m4(qq);
  u = __riscv_vfmul_vv_f32m4(u, qf, vl);
  return u;
}

void xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v(
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
  size_t vl = __riscv_vsetvl_e32m4(n);

  vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  do {
    vl = __riscv_vsetvl_e32m4(avl);
    avl -= vl;
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(input, vl);
    vx = __riscv_vfsub_vf_f32m4(vx, *max, vl);
    input += vl;
    vfloat32m4_t vexp = softexp_f32m4(vx, vl, params);
    __riscv_vse32_v_f32m4(output, vexp, vl);
    output += vl;
    vsum = __riscv_vfadd_vv_f32m4_tu(vsum, vsum, vexp, vl);
  } while(avl > 0);

  vfloat32m1_t v0 = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  *sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsum, v0, n));
}

void xnn_f32_rmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t N = batch >> 2;
  size_t avl;
  size_t vl = __riscv_vsetvl_e32m8(N);

  vfloat32m8_t t0 = __riscv_vle32_v_f32m8(input, vl);
  input += vl;

  for (avl = N - vl; avl; avl -= vl, input += vl) {
    vl = __riscv_vsetvl_e32m8(avl);
    vfloat32m8_t vec = __riscv_vle32_v_f32m8(input, vl);
    t0 = __riscv_vfmax_vv_f32m8_tu(t0, t0, vec, vl);
  }

  vfloat32m1_t fmax = __riscv_vfmv_s_f_f32m1(-INFINITY, 1);
  output[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(t0, fmax, N));
}

void xnn_f32_rminmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t N = batch >> 2;
  size_t avl;
  size_t vl = __riscv_vsetvl_e32m8(N);

  vfloat32m8_t t0 = __riscv_vle32_v_f32m8(input, vl);
  input += vl;
  vfloat32m8_t t1 = __riscv_vmv_v_v_f32m8(t0, vl);

  for (avl = N - vl; avl; avl -= vl, input += vl) {
    vl = __riscv_vsetvl_e32m8(avl);
    vfloat32m8_t vec = __riscv_vle32_v_f32m8(input, vl);
    t0 = __riscv_vfmin_vv_f32m8_tu(t0, t0, vec, vl);
    t1 = __riscv_vfmax_vv_f32m8_tu(t1, t1, vec, vl);
  }

  vfloat32m1_t fmin = __riscv_vfmv_s_f_f32m1(INFINITY, 1);
  vfloat32m1_t fmax = __riscv_vfmv_s_f_f32m1(-INFINITY, 1);
  output[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m8_f32m1(t0, fmin, N));
  output[1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(t1, fmax, N));
}

void xnn_f32_vadd_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = __riscv_vfadd_vv_f32m8(va, vb, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vaddc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfadd_vf_f32m8(va, b, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vdiv_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = __riscv_vfdiv_vv_f32m8(va, vb, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vdivc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfdiv_vf_f32m8(va, b, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = __riscv_vfmax_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmaxc_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfmax_vf_f32m8(va, b, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmin_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = __riscv_vfmin_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vminc_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfmin_vf_f32m8(va, b, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmul_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = __riscv_vfmul_vv_f32m8(va, vb, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmulc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfmul_vf_f32m8(va, b, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vrdivc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfrdiv_vf_f32m8(va, b, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vrsubc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfrsub_vf_f32m8(va, b, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vsqrdiff_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = __riscv_vfsub_vv_f32m8(va, vb, vl);
    vacc = __riscv_vfmul_vv_f32m8(vacc, vacc, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vsqrdiffc_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfsub_vf_f32m8(va, b, vl);
    vacc = __riscv_vfmul_vv_f32m8(vacc, vacc, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vsub_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = __riscv_vfsub_vv_f32m8(va, vb, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vsubc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfsub_vf_f32m8(va, b, vl);
    vacc = __riscv_vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = __riscv_vfmin_vf_f32m8(vacc, output_max, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vcmul_ukernel__rvv_u2v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  const float* ar = input_a;
  const float* ai = input_a + batch;
  const float* br = input_b;
  const float* bi = input_b + batch;
  float* or = output;
  float* oi = output + batch;

  for (; batch > 0; ) {
    size_t n = __riscv_vsetvl_e32m2(batch); batch -= n;
    vfloat32m2_t ar_f32v = __riscv_vle32_v_f32m2(ar, n); ar += n;
    vfloat32m2_t ai_f32v = __riscv_vle32_v_f32m2(ai, n); ai += n;
    vfloat32m2_t br_f32v = __riscv_vle32_v_f32m2(br, n); br += n;
    vfloat32m2_t bi_f32v = __riscv_vle32_v_f32m2(bi, n); bi += n;
    vfloat32m2_t ai_bi_f32v = __riscv_vfmul_vv_f32m2(ai_f32v, bi_f32v, n);
    vfloat32m2_t ai_br_f32v = __riscv_vfmul_vv_f32m2(ai_f32v, br_f32v, n);
    vfloat32m2_t ar_br_sub_ai_bi_f32v = __riscv_vfmsac_vv_f32m2(ai_bi_f32v, ar_f32v, br_f32v, n);
    vfloat32m2_t ar_bi_plus_ai_br_f32v = __riscv_vfmacc_vv_f32m2(ai_br_f32v, ar_f32v, bi_f32v, n);
    __riscv_vse32_v_f32m2(or, ar_br_sub_ai_bi_f32v, n); or += n;
    __riscv_vse32_v_f32m2(oi, ar_bi_plus_ai_br_f32v, n); oi += n;
  }
}

void xnn_f32_vlrelu_ukernel__rvv_u4v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float slope = params->scalar.slope;
  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  do {
    size_t n = __riscv_vsetvl_e32m4(batch); batch -= n;
    vfloat32m4_t in_f32v = __riscv_vle32_v_f32m4(input, n); input += n;
    vbool8_t mask_f32v = __riscv_vmflt_vf_f32m4_b8(in_f32v, 0.0f, n);
    vfloat32m4_t out_f32v = __riscv_vfmul_vf_f32m4_mu(mask_f32v, in_f32v, in_f32v, slope, n);
    __riscv_vse32_v_f32m4(output, out_f32v, n); output += n;
  } while (batch != 0);
}

void xnn_f32_vrelu_ukernel__rvv_u4v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float zero = 0.0f;
  size_t batch_ = batch >> XNN_LOG2_SIZEOF_FLOAT;

  for (; batch_ > 0; ) {
    size_t n = __riscv_vsetvl_e32m4(batch_); batch_ -= n;
    vfloat32m4_t in_f32v = __riscv_vle32_v_f32m4(input, n); input += n;
    vfloat32m4_t out_f32v = __riscv_vfmax_vf_f32m4(in_f32v, zero, n);
    __riscv_vse32_v_f32m4(output, out_f32v, n); output += n;
  }
}

void xnn_f32_vrsqrt_ukernel__rvv_rsqrt_u4v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  vfloat32m4_t onephalf_f32v = __riscv_vfmv_v_f_f32m4(1.5f, __riscv_vsetvl_e32m4(batch));
  for (; batch > 0; ) {
    int32_t n = __riscv_vsetvl_e32m4(batch); batch -= n;
    vfloat32m4_t in_f32v = __riscv_vle32_v_f32m4(input, n); input += n;

    vfloat32m4_t t0_f32v = __riscv_vfrsqrt7_v_f32m4(in_f32v, n);
    vfloat32m4_t in_half_f32v = __riscv_vfmul_vf_f32m4(in_f32v, 0.5f, n);

    // First Newton-Raphson iteration
    vfloat32m4_t t1_f32v = __riscv_vfmul_vv_f32m4(t0_f32v, t0_f32v, n);
    vfloat32m4_t t2_f32v = __riscv_vfnmsub_vv_f32m4(in_half_f32v, t1_f32v, onephalf_f32v, n);
    t0_f32v = __riscv_vfmul_vv_f32m4(t0_f32v, t2_f32v, n);

    // Second Newton-Raphson iteration
    t1_f32v = __riscv_vfmul_vv_f32m4(t0_f32v, t0_f32v, n);
    t2_f32v = __riscv_vfnmsub_vv_f32m4(in_half_f32v, t1_f32v, onephalf_f32v, n);
    vfloat32m4_t y_f32v = __riscv_vfmul_vv_f32m4(t0_f32v, t2_f32v, n);

    __riscv_vse32_v_f32m4(output, y_f32v, n); output += n;
  }
}

void xnn_qs8_vmul_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->scalar.a_zero_point;
  const int32_t b_zero_point = params->scalar.b_zero_point;
  const float scale = params->scalar.scale;
  const float output_min_less_zero_point = (int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point;
  const float output_max_less_zero_point = (int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point;
  const float magic_bias = 12582912.0f;
  const int32_t magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  do {
    const size_t n = __riscv_vsetvl_e8m2(batch);

    vint8m2_t in_a_i8v = __riscv_vle8_v_i8m2(input_a, n); input_a += n;
    vint8m2_t in_b_i8v = __riscv_vle8_v_i8m2(input_b, n); input_b += n;
    vint16m4_t a_i16v = __riscv_vwsub_vx_i16m4(in_a_i8v, a_zero_point, n);
    vint16m4_t b_i16v = __riscv_vwsub_vx_i16m4(in_b_i8v, b_zero_point, n);

    vint32m8_t acc_i32v = __riscv_vwmul_vv_i32m8(a_i16v, b_i16v, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vint32m8_t out_i32v = __riscv_vfcvt_x_f_v_i32m8(acc_f32v, n);
    out_i32v = __riscv_vsub_vx_i32m8(out_i32v, magic_bias_less_output_zero_point, n);
    vint16m4_t out_i16v = __riscv_vncvt_x_x_w_i16m4(out_i32v, n);
    vint8m2_t out_i8v = __riscv_vncvt_x_x_w_i8m2(out_i16v, n);
    __riscv_vse8_v_i8m2(output, out_i8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->scalar.a_zero_point;
  const float scale = params->scalar.scale;
  const float output_min_less_zero_point = (int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point;
  const float output_max_less_zero_point = (int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point;
  const float magic_bias = 12582912.0f;
  const int32_t magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  const int32_t vb = (int32_t) *input_b - params->scalar.b_zero_point;
  do {
    const size_t n = __riscv_vsetvl_e8m2(batch);

    vint8m2_t in_a_i8v = __riscv_vle8_v_i8m2(input_a, n); input_a += n;
    vint16m4_t a_i16v = __riscv_vwsub_vx_i16m4(in_a_i8v, a_zero_point, n);

    vint32m8_t acc_i32v = __riscv_vwmul_vx_i32m8(a_i16v, vb, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vint32m8_t out_i32v = __riscv_vfcvt_x_f_v_i32m8(acc_f32v, n);
    out_i32v = __riscv_vsub_vx_i32m8(out_i32v, magic_bias_less_output_zero_point, n);
    vint16m4_t out_i16v = __riscv_vncvt_x_x_w_i16m4(out_i32v, n);
    vint8m2_t out_i8v = __riscv_vncvt_x_x_w_i8m2(out_i16v, n);
    __riscv_vse8_v_i8m2(output, out_i8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qu8_vmul_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->scalar.a_zero_point;
  const int32_t b_zero_point = params->scalar.b_zero_point;
  const float scale = params->scalar.scale;
  const float output_min_less_zero_point = (int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point;
  const float output_max_less_zero_point = (int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point;
  const float magic_bias = 12582912.0f;
  const int32_t magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  do {
    const size_t n = __riscv_vsetvl_e8m2(batch);

    vuint8m2_t in_a_u8v = __riscv_vle8_v_u8m2(input_a, n); input_a += n;
    vuint8m2_t in_b_u8v = __riscv_vle8_v_u8m2(input_b, n); input_b += n;
    vuint16m4_t a_u16v = __riscv_vwsubu_vx_u16m4(in_a_u8v, a_zero_point, n);
    vuint16m4_t b_u16v = __riscv_vwsubu_vx_u16m4(in_b_u8v, b_zero_point, n);
    vint16m4_t a_i16v = __riscv_vreinterpret_v_u16m4_i16m4(a_u16v);
    vint16m4_t b_i16v = __riscv_vreinterpret_v_u16m4_i16m4(b_u16v);

    vint32m8_t acc_i32v = __riscv_vwmul_vv_i32m8(a_i16v, b_i16v, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vuint32m8_t out_u32v = __riscv_vfcvt_xu_f_v_u32m8(acc_f32v, n);
    out_u32v = __riscv_vsub_vx_u32m8(out_u32v, magic_bias_less_output_zero_point, n);
    vuint16m4_t out_u16v = __riscv_vncvt_x_x_w_u16m4(out_u32v, n);
    vuint8m2_t out_u8v = __riscv_vncvt_x_x_w_u8m2(out_u16v, n);
    __riscv_vse8_v_u8m2(output, out_u8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qu8_vmulc_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->scalar.a_zero_point;
  const float scale = params->scalar.scale;
  const float output_min_less_zero_point = (int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point;
  const float output_max_less_zero_point = (int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point;
  const float magic_bias = 12582912.0f;
  const int32_t magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  const int32_t vb = (int32_t) *input_b - params->scalar.b_zero_point;
  do {
    const size_t n = __riscv_vsetvl_e8m2(batch);

    vuint8m2_t in_a_u8v = __riscv_vle8_v_u8m2(input_a, n); input_a += n;
    vuint16m4_t a_u16v = __riscv_vwsubu_vx_u16m4(in_a_u8v, a_zero_point, n);
    vint16m4_t a_i16v = __riscv_vreinterpret_v_u16m4_i16m4(a_u16v);

    vint32m8_t acc_i32v = __riscv_vwmul_vx_i32m8(a_i16v, vb, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vuint32m8_t out_u32v = __riscv_vfcvt_xu_f_v_u32m8(acc_f32v, n);
    out_u32v = __riscv_vsub_vx_u32m8(out_u32v, magic_bias_less_output_zero_point, n);
    vuint16m4_t out_u16v = __riscv_vncvt_x_x_w_u16m4(out_u32v, n);
    vuint8m2_t out_u8v = __riscv_vncvt_x_x_w_u8m2(out_u16v, n);
    __riscv_vse8_v_u8m2(output, out_u8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_x32_transposec_ukernel__16x8_rvv(
  const uint32_t* input,
  uint32_t* output,
  size_t input_stride,
  size_t output_stride,
  size_t block_width,
  size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 16;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;

  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t* o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t* o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);
  uint32_t* o4 = (uint32_t*) ((uintptr_t) o3 + output_stride);
  uint32_t* o5 = (uint32_t*) ((uintptr_t) o4 + output_stride);
  uint32_t* o6 = (uint32_t*) ((uintptr_t) o5 + output_stride);
  uint32_t* o7 = (uint32_t*) ((uintptr_t) o6 + output_stride);

  do {
    size_t bh = block_height;
    size_t vl = __riscv_vsetvl_e32m1(tile_height);
    for (; bh >= 16; bh -= 16) {
      if (block_width >= tile_width) {
        vuint32m1x8_t tuple = __riscv_vlsseg8e32_v_u32m1x8(i0, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x8_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x8_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x8_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x8_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
        vuint32m1_t v_d4 = __riscv_vget_v_u32m1x8_u32m1(tuple, 4);
        __riscv_vse32_v_u32m1(o4, v_d4, vl);
        vuint32m1_t v_d5 = __riscv_vget_v_u32m1x8_u32m1(tuple, 5);
        __riscv_vse32_v_u32m1(o5, v_d5, vl);
        vuint32m1_t v_d6 = __riscv_vget_v_u32m1x8_u32m1(tuple, 6);
        __riscv_vse32_v_u32m1(o6, v_d6, vl);
        vuint32m1_t v_d7 = __riscv_vget_v_u32m1x8_u32m1(tuple, 7);
        __riscv_vse32_v_u32m1(o7, v_d7, vl);

      } else {
        switch (block_width) {
          case 7: {
            vuint32m1x7_t tuple = __riscv_vlsseg7e32_v_u32m1x7(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x7_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x7_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x7_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x7_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x7_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x7_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            vuint32m1_t v_d6 = __riscv_vget_v_u32m1x7_u32m1(tuple, 6);
            __riscv_vse32_v_u32m1(o6, v_d6, vl);
            break;
          }

          case 6: {
            vuint32m1x6_t tuple = __riscv_vlsseg6e32_v_u32m1x6(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x6_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x6_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x6_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x6_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x6_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x6_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            break;
          }

          case 5: {
            vuint32m1x5_t tuple = __riscv_vlsseg5e32_v_u32m1x5(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x5_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x5_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x5_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x5_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x5_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            break;
          }

          case 4: {
            vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            break;
          }

          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }

          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i0, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      o7 = (uint32_t*) ((uintptr_t) o7 + tile_hbytes);
      o6 = (uint32_t*) ((uintptr_t) o6 + tile_hbytes);
      o5 = (uint32_t*) ((uintptr_t) o5 + tile_hbytes);
      o4 = (uint32_t*) ((uintptr_t) o4 + tile_hbytes);
      o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint32_t* i = i0;
      vl = __riscv_vsetvl_e32m1(bh);
      if (block_width >= tile_width) {
        vuint32m1x8_t tuple = __riscv_vlsseg8e32_v_u32m1x8(i, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x8_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x8_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x8_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x8_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
        vuint32m1_t v_d4 = __riscv_vget_v_u32m1x8_u32m1(tuple, 4);
        __riscv_vse32_v_u32m1(o4, v_d4, vl);
        vuint32m1_t v_d5 = __riscv_vget_v_u32m1x8_u32m1(tuple, 5);
        __riscv_vse32_v_u32m1(o5, v_d5, vl);
        vuint32m1_t v_d6 = __riscv_vget_v_u32m1x8_u32m1(tuple, 6);
        __riscv_vse32_v_u32m1(o6, v_d6, vl);
        vuint32m1_t v_d7 = __riscv_vget_v_u32m1x8_u32m1(tuple, 7);
        __riscv_vse32_v_u32m1(o7, v_d7, vl);
      } else {
        switch(block_width) {
          case 7: {
            vuint32m1x7_t tuple = __riscv_vlsseg7e32_v_u32m1x7(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x7_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x7_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x7_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x7_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x7_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x7_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            vuint32m1_t v_d6 = __riscv_vget_v_u32m1x7_u32m1(tuple, 6);
            __riscv_vse32_v_u32m1(o6, v_d6, vl);
            break;
          }
          case 6: {
            vuint32m1x6_t tuple = __riscv_vlsseg6e32_v_u32m1x6(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x6_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x6_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x6_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x6_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x6_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x6_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            break;
          }
          case 5: {
            vuint32m1x5_t tuple = __riscv_vlsseg5e32_v_u32m1x5(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x5_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x5_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x5_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x5_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x5_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            break;
          }
          case 4: {
            vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            break;
          }
          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }
          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      if (bh & 8) {
        o7 += 8;
        o6 += 8;
        o5 += 8;
        o4 += 8;
        o3 += 8;
        o2 += 8;
        o1 += 8;
        o0 += 8;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 8);
      }
      if (bh & 4) {
        o7 += 4;
        o6 += 4;
        o5 += 4;
        o4 += 4;
        o3 += 4;
        o2 += 4;
        o1 += 4;
        o0 += 4;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 4);
      }
      if (bh & 2) {
        o7 += 2;
        o6 += 2;
        o5 += 2;
        o4 += 2;
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 2);
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);

    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint32_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint32_t*) ((uintptr_t) o3 + output_reset);
    o4 = (uint32_t*) ((uintptr_t) o4 + output_reset);
    o5 = (uint32_t*) ((uintptr_t) o5 + output_reset);
    o6 = (uint32_t*) ((uintptr_t) o6 + output_reset);
    o7 = (uint32_t*) ((uintptr_t) o7 + output_reset);

    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x32_transposec_ukernel__32x8_rvv(
  const uint32_t* input,
  uint32_t* output,
  size_t input_stride,
  size_t output_stride,
  size_t block_width,
  size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 32;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;

  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t* o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t* o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);
  uint32_t* o4 = (uint32_t*) ((uintptr_t) o3 + output_stride);
  uint32_t* o5 = (uint32_t*) ((uintptr_t) o4 + output_stride);
  uint32_t* o6 = (uint32_t*) ((uintptr_t) o5 + output_stride);
  uint32_t* o7 = (uint32_t*) ((uintptr_t) o6 + output_stride);

  do {
    size_t bh = block_height;
    size_t vl = __riscv_vsetvl_e32m1(tile_height);
    for (; bh >= 32; bh -= 32) {
      if (block_width >= tile_width) {
        vuint32m1x8_t tuple = __riscv_vlsseg8e32_v_u32m1x8(i0, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x8_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x8_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x8_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x8_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
        vuint32m1_t v_d4 = __riscv_vget_v_u32m1x8_u32m1(tuple, 4);
        __riscv_vse32_v_u32m1(o4, v_d4, vl);
        vuint32m1_t v_d5 = __riscv_vget_v_u32m1x8_u32m1(tuple, 5);
        __riscv_vse32_v_u32m1(o5, v_d5, vl);
        vuint32m1_t v_d6 = __riscv_vget_v_u32m1x8_u32m1(tuple, 6);
        __riscv_vse32_v_u32m1(o6, v_d6, vl);
        vuint32m1_t v_d7 = __riscv_vget_v_u32m1x8_u32m1(tuple, 7);
        __riscv_vse32_v_u32m1(o7, v_d7, vl);

      } else {
        switch (block_width) {
          case 7: {
            vuint32m1x7_t tuple = __riscv_vlsseg7e32_v_u32m1x7(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x7_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x7_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x7_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x7_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x7_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x7_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            vuint32m1_t v_d6 = __riscv_vget_v_u32m1x7_u32m1(tuple, 6);
            __riscv_vse32_v_u32m1(o6, v_d6, vl);
            break;
          }

          case 6: {
            vuint32m1x6_t tuple = __riscv_vlsseg6e32_v_u32m1x6(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x6_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x6_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x6_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x6_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x6_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x6_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            break;
          }

          case 5: {
            vuint32m1x5_t tuple = __riscv_vlsseg5e32_v_u32m1x5(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x5_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x5_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x5_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x5_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x5_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            break;
          }

          case 4: {
            vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            break;
          }

          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }

          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i0, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      o7 = (uint32_t*) ((uintptr_t) o7 + tile_hbytes);
      o6 = (uint32_t*) ((uintptr_t) o6 + tile_hbytes);
      o5 = (uint32_t*) ((uintptr_t) o5 + tile_hbytes);
      o4 = (uint32_t*) ((uintptr_t) o4 + tile_hbytes);
      o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint32_t* i = i0;
      vl = __riscv_vsetvl_e32m1(bh);
      if (block_width >= tile_width) {
        vuint32m1x8_t tuple = __riscv_vlsseg8e32_v_u32m1x8(i, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x8_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x8_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x8_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x8_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
        vuint32m1_t v_d4 = __riscv_vget_v_u32m1x8_u32m1(tuple, 4);
        __riscv_vse32_v_u32m1(o4, v_d4, vl);
        vuint32m1_t v_d5 = __riscv_vget_v_u32m1x8_u32m1(tuple, 5);
        __riscv_vse32_v_u32m1(o5, v_d5, vl);
        vuint32m1_t v_d6 = __riscv_vget_v_u32m1x8_u32m1(tuple, 6);
        __riscv_vse32_v_u32m1(o6, v_d6, vl);
        vuint32m1_t v_d7 = __riscv_vget_v_u32m1x8_u32m1(tuple, 7);
        __riscv_vse32_v_u32m1(o7, v_d7, vl);
      } else {
        switch(block_width) {
          case 7: {
            vuint32m1x7_t tuple = __riscv_vlsseg7e32_v_u32m1x7(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x7_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x7_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x7_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x7_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x7_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x7_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            vuint32m1_t v_d6 = __riscv_vget_v_u32m1x7_u32m1(tuple, 6);
            __riscv_vse32_v_u32m1(o6, v_d6, vl);
            break;
          }
          case 6: {
            vuint32m1x6_t tuple = __riscv_vlsseg6e32_v_u32m1x6(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x6_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x6_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x6_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x6_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x6_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x6_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            break;
          }
          case 5: {
            vuint32m1x5_t tuple = __riscv_vlsseg5e32_v_u32m1x5(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x5_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x5_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x5_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x5_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x5_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            break;
          }
          case 4: {
            vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            break;
          }
          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }
          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      if (bh & 16) {
        o7 += 16;
        o6 += 16;
        o5 += 16;
        o4 += 16;
        o3 += 16;
        o2 += 16;
        o1 += 16;
        o0 += 16;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 16);
      }
      if (bh & 8) {
        o7 += 8;
        o6 += 8;
        o5 += 8;
        o4 += 8;
        o3 += 8;
        o2 += 8;
        o1 += 8;
        o0 += 8;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 8);
      }
      if (bh & 4) {
        o7 += 4;
        o6 += 4;
        o5 += 4;
        o4 += 4;
        o3 += 4;
        o2 += 4;
        o1 += 4;
        o0 += 4;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 4);
      }
      if (bh & 2) {
        o7 += 2;
        o6 += 2;
        o5 += 2;
        o4 += 2;
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 2);
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);

    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint32_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint32_t*) ((uintptr_t) o3 + output_reset);
    o4 = (uint32_t*) ((uintptr_t) o4 + output_reset);
    o5 = (uint32_t*) ((uintptr_t) o5 + output_reset);
    o6 = (uint32_t*) ((uintptr_t) o6 + output_reset);
    o7 = (uint32_t*) ((uintptr_t) o7 + output_reset);

    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x32_transposec_ukernel__4x4_rvv(
  const uint32_t* input,
  uint32_t* output,
  size_t input_stride,
  size_t output_stride,
  size_t block_width,
  size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;

  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t* o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t* o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);

  do {
    size_t bh = block_height;
    size_t vl = __riscv_vsetvl_e32m1(tile_height);
    for (; bh >= 4; bh -= 4) {
      if (block_width >= tile_width) {
        vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i0, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);

      } else {
        switch (block_width) {
          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }

          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i0, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint32_t* i = i0;
      vl = __riscv_vsetvl_e32m1(bh);
      if (block_width >= tile_width) {
        vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
      } else {
        switch(block_width) {
          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }
          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      if (bh & 2) {
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 2);
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);

    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint32_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint32_t*) ((uintptr_t) o3 + output_reset);

    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x32_transposec_ukernel__8x8_rvv(
  const uint32_t* input,
  uint32_t* output,
  size_t input_stride,
  size_t output_stride,
  size_t block_width,
  size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 8;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;

  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t* o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t* o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);
  uint32_t* o4 = (uint32_t*) ((uintptr_t) o3 + output_stride);
  uint32_t* o5 = (uint32_t*) ((uintptr_t) o4 + output_stride);
  uint32_t* o6 = (uint32_t*) ((uintptr_t) o5 + output_stride);
  uint32_t* o7 = (uint32_t*) ((uintptr_t) o6 + output_stride);

  do {
    size_t bh = block_height;
    size_t vl = __riscv_vsetvl_e32m1(tile_height);
    for (; bh >= 8; bh -= 8) {
      if (block_width >= tile_width) {
        vuint32m1x8_t tuple = __riscv_vlsseg8e32_v_u32m1x8(i0, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x8_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x8_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x8_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x8_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
        vuint32m1_t v_d4 = __riscv_vget_v_u32m1x8_u32m1(tuple, 4);
        __riscv_vse32_v_u32m1(o4, v_d4, vl);
        vuint32m1_t v_d5 = __riscv_vget_v_u32m1x8_u32m1(tuple, 5);
        __riscv_vse32_v_u32m1(o5, v_d5, vl);
        vuint32m1_t v_d6 = __riscv_vget_v_u32m1x8_u32m1(tuple, 6);
        __riscv_vse32_v_u32m1(o6, v_d6, vl);
        vuint32m1_t v_d7 = __riscv_vget_v_u32m1x8_u32m1(tuple, 7);
        __riscv_vse32_v_u32m1(o7, v_d7, vl);

      } else {
        switch (block_width) {
          case 7: {
            vuint32m1x7_t tuple = __riscv_vlsseg7e32_v_u32m1x7(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x7_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x7_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x7_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x7_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x7_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x7_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            vuint32m1_t v_d6 = __riscv_vget_v_u32m1x7_u32m1(tuple, 6);
            __riscv_vse32_v_u32m1(o6, v_d6, vl);
            break;
          }

          case 6: {
            vuint32m1x6_t tuple = __riscv_vlsseg6e32_v_u32m1x6(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x6_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x6_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x6_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x6_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x6_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x6_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            break;
          }

          case 5: {
            vuint32m1x5_t tuple = __riscv_vlsseg5e32_v_u32m1x5(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x5_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x5_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x5_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x5_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x5_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            break;
          }

          case 4: {
            vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            break;
          }

          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }

          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i0, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      o7 = (uint32_t*) ((uintptr_t) o7 + tile_hbytes);
      o6 = (uint32_t*) ((uintptr_t) o6 + tile_hbytes);
      o5 = (uint32_t*) ((uintptr_t) o5 + tile_hbytes);
      o4 = (uint32_t*) ((uintptr_t) o4 + tile_hbytes);
      o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint32_t* i = i0;
      vl = __riscv_vsetvl_e32m1(bh);
      if (block_width >= tile_width) {
        vuint32m1x8_t tuple = __riscv_vlsseg8e32_v_u32m1x8(i, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x8_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x8_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x8_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x8_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
        vuint32m1_t v_d4 = __riscv_vget_v_u32m1x8_u32m1(tuple, 4);
        __riscv_vse32_v_u32m1(o4, v_d4, vl);
        vuint32m1_t v_d5 = __riscv_vget_v_u32m1x8_u32m1(tuple, 5);
        __riscv_vse32_v_u32m1(o5, v_d5, vl);
        vuint32m1_t v_d6 = __riscv_vget_v_u32m1x8_u32m1(tuple, 6);
        __riscv_vse32_v_u32m1(o6, v_d6, vl);
        vuint32m1_t v_d7 = __riscv_vget_v_u32m1x8_u32m1(tuple, 7);
        __riscv_vse32_v_u32m1(o7, v_d7, vl);
      } else {
        switch(block_width) {
          case 7: {
            vuint32m1x7_t tuple = __riscv_vlsseg7e32_v_u32m1x7(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x7_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x7_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x7_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x7_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x7_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x7_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            vuint32m1_t v_d6 = __riscv_vget_v_u32m1x7_u32m1(tuple, 6);
            __riscv_vse32_v_u32m1(o6, v_d6, vl);
            break;
          }
          case 6: {
            vuint32m1x6_t tuple = __riscv_vlsseg6e32_v_u32m1x6(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x6_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x6_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x6_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x6_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x6_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x6_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            break;
          }
          case 5: {
            vuint32m1x5_t tuple = __riscv_vlsseg5e32_v_u32m1x5(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x5_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x5_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x5_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x5_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x5_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            break;
          }
          case 4: {
            vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            break;
          }
          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }
          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      if (bh & 4) {
        o7 += 4;
        o6 += 4;
        o5 += 4;
        o4 += 4;
        o3 += 4;
        o2 += 4;
        o1 += 4;
        o0 += 4;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 4);
      }
      if (bh & 2) {
        o7 += 2;
        o6 += 2;
        o5 += 2;
        o4 += 2;
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 2);
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);

    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint32_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint32_t*) ((uintptr_t) o3 + output_reset);
    o4 = (uint32_t*) ((uintptr_t) o4 + output_reset);
    o5 = (uint32_t*) ((uintptr_t) o5 + output_reset);
    o6 = (uint32_t*) ((uintptr_t) o6 + output_reset);
    o7 = (uint32_t*) ((uintptr_t) o7 + output_reset);

    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
