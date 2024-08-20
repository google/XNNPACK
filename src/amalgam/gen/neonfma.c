// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gemm.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/igemm.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/prefetch.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "xnnpack/spmm.h"
#include "xnnpack/vmulcaddc.h"
#include "xnnpack/vunary.h"


void xnn_f32_dwconv_minmax_ukernel_25p8c__neonfma_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }

    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;
      float32x4_t vacc4567p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
      float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);
      float32x4_t vacc4567p1 = vmulq_f32(vi1x4567, vk1x4567);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi3x4567, vk3x4567);

      const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk4x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);

      const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk5x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi5x0123, vk5x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi5x4567, vk5x4567);

      const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi6x4567 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk6x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi6x4567, vk6x4567);

      const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi7x4567 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vk7x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk7x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi7x0123, vk7x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi7x4567, vk7x4567);

      const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vi8x4567 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vk8x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk8x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi8x4567, vk8x4567);

      const float32x4_t vi9x0123 = vld1q_f32(i9); i9 += 4;
      const float32x4_t vi9x4567 = vld1q_f32(i9); i9 += 4;
      const float32x4_t vk9x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk9x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi9x0123, vk9x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi9x4567, vk9x4567);

      const float32x4_t vi10x0123 = vld1q_f32(i10); i10 += 4;
      const float32x4_t vi10x4567 = vld1q_f32(i10); i10 += 4;
      const float32x4_t vk10x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk10x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi10x0123, vk10x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi10x4567, vk10x4567);

      const float32x4_t vi11x0123 = vld1q_f32(i11); i11 += 4;
      const float32x4_t vi11x4567 = vld1q_f32(i11); i11 += 4;
      const float32x4_t vk11x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk11x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi11x0123, vk11x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi11x4567, vk11x4567);

      const float32x4_t vi12x0123 = vld1q_f32(i12); i12 += 4;
      const float32x4_t vi12x4567 = vld1q_f32(i12); i12 += 4;
      const float32x4_t vk12x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk12x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi12x0123, vk12x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi12x4567, vk12x4567);

      const float32x4_t vi13x0123 = vld1q_f32(i13); i13 += 4;
      const float32x4_t vi13x4567 = vld1q_f32(i13); i13 += 4;
      const float32x4_t vk13x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk13x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi13x0123, vk13x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi13x4567, vk13x4567);

      const float32x4_t vi14x0123 = vld1q_f32(i14); i14 += 4;
      const float32x4_t vi14x4567 = vld1q_f32(i14); i14 += 4;
      const float32x4_t vk14x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk14x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi14x0123, vk14x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi14x4567, vk14x4567);

      const float32x4_t vi15x0123 = vld1q_f32(i15); i15 += 4;
      const float32x4_t vi15x4567 = vld1q_f32(i15); i15 += 4;
      const float32x4_t vk15x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk15x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi15x0123, vk15x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi15x4567, vk15x4567);

      const float32x4_t vi16x0123 = vld1q_f32(i16); i16 += 4;
      const float32x4_t vi16x4567 = vld1q_f32(i16); i16 += 4;
      const float32x4_t vk16x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk16x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi16x0123, vk16x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi16x4567, vk16x4567);

      const float32x4_t vi17x0123 = vld1q_f32(i17); i17 += 4;
      const float32x4_t vi17x4567 = vld1q_f32(i17); i17 += 4;
      const float32x4_t vk17x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk17x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi17x0123, vk17x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi17x4567, vk17x4567);

      const float32x4_t vi18x0123 = vld1q_f32(i18); i18 += 4;
      const float32x4_t vi18x4567 = vld1q_f32(i18); i18 += 4;
      const float32x4_t vk18x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk18x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi18x0123, vk18x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi18x4567, vk18x4567);

      const float32x4_t vi19x0123 = vld1q_f32(i19); i19 += 4;
      const float32x4_t vi19x4567 = vld1q_f32(i19); i19 += 4;
      const float32x4_t vk19x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk19x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi19x0123, vk19x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi19x4567, vk19x4567);

      const float32x4_t vi20x0123 = vld1q_f32(i20); i20 += 4;
      const float32x4_t vi20x4567 = vld1q_f32(i20); i20 += 4;
      const float32x4_t vk20x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk20x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi20x0123, vk20x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi20x4567, vk20x4567);

      const float32x4_t vi21x0123 = vld1q_f32(i21); i21 += 4;
      const float32x4_t vi21x4567 = vld1q_f32(i21); i21 += 4;
      const float32x4_t vk21x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk21x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi21x0123, vk21x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi21x4567, vk21x4567);

      const float32x4_t vi22x0123 = vld1q_f32(i22); i22 += 4;
      const float32x4_t vi22x4567 = vld1q_f32(i22); i22 += 4;
      const float32x4_t vk22x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk22x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi22x0123, vk22x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi22x4567, vk22x4567);

      const float32x4_t vi23x0123 = vld1q_f32(i23); i23 += 4;
      const float32x4_t vi23x4567 = vld1q_f32(i23); i23 += 4;
      const float32x4_t vk23x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk23x4567 = vld1q_f32(w); w += 4;
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi23x0123, vk23x0123);
      vacc4567p1 = vfmaq_f32(vacc4567p1, vi23x4567, vk23x4567);

      const float32x4_t vi24x0123 = vld1q_f32(i24); i24 += 4;
      const float32x4_t vi24x4567 = vld1q_f32(i24); i24 += 4;
      const float32x4_t vk24x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk24x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi24x0123, vk24x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi24x4567, vk24x4567);

      // Add up all accumulators to vacc01234567p0
      vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);
      vacc4567p0 = vaddq_f32(vacc4567p0, vacc4567p1);

      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      float32x4_t vacc4567 = vmaxq_f32(vacc4567p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);
      vacc4567 = vminq_f32(vacc4567, vmax);

      vst1q_f32(output, vacc0123); output += 4;
      vst1q_f32(output, vacc4567); output += 4;
    }
    for (; c >= 4; c -= 4) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w + 4);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w + 12);
      float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w + 20);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w + 28);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);

      const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vk4x0123 = vld1q_f32(w + 36);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

      const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vk5x0123 = vld1q_f32(w + 44);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi5x0123, vk5x0123);

      const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vk6x0123 = vld1q_f32(w + 52);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);

      const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vk7x0123 = vld1q_f32(w + 60);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi7x0123, vk7x0123);

      const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vk8x0123 = vld1q_f32(w + 68);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);

      const float32x4_t vi9x0123 = vld1q_f32(i9); i9 += 4;
      const float32x4_t vk9x0123 = vld1q_f32(w + 76);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi9x0123, vk9x0123);

      const float32x4_t vi10x0123 = vld1q_f32(i10); i10 += 4;
      const float32x4_t vk10x0123 = vld1q_f32(w + 84);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi10x0123, vk10x0123);

      const float32x4_t vi11x0123 = vld1q_f32(i11); i11 += 4;
      const float32x4_t vk11x0123 = vld1q_f32(w + 92);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi11x0123, vk11x0123);

      const float32x4_t vi12x0123 = vld1q_f32(i12); i12 += 4;
      const float32x4_t vk12x0123 = vld1q_f32(w + 100);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi12x0123, vk12x0123);

      const float32x4_t vi13x0123 = vld1q_f32(i13); i13 += 4;
      const float32x4_t vk13x0123 = vld1q_f32(w + 108);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi13x0123, vk13x0123);

      const float32x4_t vi14x0123 = vld1q_f32(i14); i14 += 4;
      const float32x4_t vk14x0123 = vld1q_f32(w + 116);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi14x0123, vk14x0123);

      const float32x4_t vi15x0123 = vld1q_f32(i15); i15 += 4;
      const float32x4_t vk15x0123 = vld1q_f32(w + 124);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi15x0123, vk15x0123);

      const float32x4_t vi16x0123 = vld1q_f32(i16); i16 += 4;
      const float32x4_t vk16x0123 = vld1q_f32(w + 132);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi16x0123, vk16x0123);

      const float32x4_t vi17x0123 = vld1q_f32(i17); i17 += 4;
      const float32x4_t vk17x0123 = vld1q_f32(w + 140);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi17x0123, vk17x0123);

      const float32x4_t vi18x0123 = vld1q_f32(i18); i18 += 4;
      const float32x4_t vk18x0123 = vld1q_f32(w + 148);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi18x0123, vk18x0123);

      const float32x4_t vi19x0123 = vld1q_f32(i19); i19 += 4;
      const float32x4_t vk19x0123 = vld1q_f32(w + 156);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi19x0123, vk19x0123);

      const float32x4_t vi20x0123 = vld1q_f32(i20); i20 += 4;
      const float32x4_t vk20x0123 = vld1q_f32(w + 164);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi20x0123, vk20x0123);

      const float32x4_t vi21x0123 = vld1q_f32(i21); i21 += 4;
      const float32x4_t vk21x0123 = vld1q_f32(w + 172);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi21x0123, vk21x0123);

      const float32x4_t vi22x0123 = vld1q_f32(i22); i22 += 4;
      const float32x4_t vk22x0123 = vld1q_f32(w + 180);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi22x0123, vk22x0123);

      const float32x4_t vi23x0123 = vld1q_f32(i23); i23 += 4;
      const float32x4_t vk23x0123 = vld1q_f32(w + 188);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi23x0123, vk23x0123);

      const float32x4_t vi24x0123 = vld1q_f32(i24); i24 += 4;
      const float32x4_t vk24x0123 = vld1q_f32(w + 196);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi24x0123, vk24x0123);

      // Add up all accumulators to vacc0123p0
      vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      vst1q_f32(output, vacc0123); output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      float32x4_t vacc0123p0 = vld1q_f32(w);


      const float32x4_t vi0x0123 = vld1q_f32(i0);
      const float32x4_t vk0x0123 = vld1q_f32(w + 8);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1);
      const float32x4_t vk1x0123 = vld1q_f32(w + 16);
      float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2);
      const float32x4_t vk2x0123 = vld1q_f32(w + 24);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3);
      const float32x4_t vk3x0123 = vld1q_f32(w + 32);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);

      const float32x4_t vi4x0123 = vld1q_f32(i4);
      const float32x4_t vk4x0123 = vld1q_f32(w + 40);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

      const float32x4_t vi5x0123 = vld1q_f32(i5);
      const float32x4_t vk5x0123 = vld1q_f32(w + 48);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi5x0123, vk5x0123);

      const float32x4_t vi6x0123 = vld1q_f32(i6);
      const float32x4_t vk6x0123 = vld1q_f32(w + 56);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);

      const float32x4_t vi7x0123 = vld1q_f32(i7);
      const float32x4_t vk7x0123 = vld1q_f32(w + 64);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi7x0123, vk7x0123);

      const float32x4_t vi8x0123 = vld1q_f32(i8);
      const float32x4_t vk8x0123 = vld1q_f32(w + 72);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);

      const float32x4_t vi9x0123 = vld1q_f32(i9);
      const float32x4_t vk9x0123 = vld1q_f32(w + 80);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi9x0123, vk9x0123);

      const float32x4_t vi10x0123 = vld1q_f32(i10);
      const float32x4_t vk10x0123 = vld1q_f32(w + 88);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi10x0123, vk10x0123);

      const float32x4_t vi11x0123 = vld1q_f32(i11);
      const float32x4_t vk11x0123 = vld1q_f32(w + 96);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi11x0123, vk11x0123);

      const float32x4_t vi12x0123 = vld1q_f32(i12);
      const float32x4_t vk12x0123 = vld1q_f32(w + 104);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi12x0123, vk12x0123);

      const float32x4_t vi13x0123 = vld1q_f32(i13);
      const float32x4_t vk13x0123 = vld1q_f32(w + 112);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi13x0123, vk13x0123);

      const float32x4_t vi14x0123 = vld1q_f32(i14);
      const float32x4_t vk14x0123 = vld1q_f32(w + 120);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi14x0123, vk14x0123);

      const float32x4_t vi15x0123 = vld1q_f32(i15);
      const float32x4_t vk15x0123 = vld1q_f32(w + 128);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi15x0123, vk15x0123);

      const float32x4_t vi16x0123 = vld1q_f32(i16);
      const float32x4_t vk16x0123 = vld1q_f32(w + 136);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi16x0123, vk16x0123);

      const float32x4_t vi17x0123 = vld1q_f32(i17);
      const float32x4_t vk17x0123 = vld1q_f32(w + 144);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi17x0123, vk17x0123);

      const float32x4_t vi18x0123 = vld1q_f32(i18);
      const float32x4_t vk18x0123 = vld1q_f32(w + 152);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi18x0123, vk18x0123);

      const float32x4_t vi19x0123 = vld1q_f32(i19);
      const float32x4_t vk19x0123 = vld1q_f32(w + 160);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi19x0123, vk19x0123);

      const float32x4_t vi20x0123 = vld1q_f32(i20);
      const float32x4_t vk20x0123 = vld1q_f32(w + 168);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi20x0123, vk20x0123);

      const float32x4_t vi21x0123 = vld1q_f32(i21);
      const float32x4_t vk21x0123 = vld1q_f32(w + 176);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi21x0123, vk21x0123);

      const float32x4_t vi22x0123 = vld1q_f32(i22);
      const float32x4_t vk22x0123 = vld1q_f32(w + 184);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi22x0123, vk22x0123);

      const float32x4_t vi23x0123 = vld1q_f32(i23);
      const float32x4_t vk23x0123 = vld1q_f32(w + 192);
      vacc0123p1 = vfmaq_f32(vacc0123p1, vi23x0123, vk23x0123);

      const float32x4_t vi24x0123 = vld1q_f32(i24);
      const float32x4_t vk24x0123 = vld1q_f32(w + 200);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi24x0123, vk24x0123);

      // Add up all accumulators to vacc0123p0
      vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      float32x2_t vacc01 = vget_low_f32(vacc0123);
      if (c & 2) {
        vst1_f32(output, vacc01); output += 2;
        vacc01 = vget_high_f32(vacc0123);
      }
      if (c & 1) {
        vst1_lane_f32(output, vacc01, 0); output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p8c__neonfma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }

    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;
      float32x4_t vacc4567p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi1x4567, vk1x4567);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      float32x4_t vacc4567 = vmaxq_f32(vacc4567p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);
      vacc4567 = vminq_f32(vacc4567, vmax);

      vst1q_f32(output, vacc0123); output += 4;
      vst1q_f32(output, vacc4567); output += 4;
    }
    for (; c >= 4; c -= 4) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w + 4);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w + 12);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w + 20);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      vst1q_f32(output, vacc0123); output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      float32x4_t vacc0123p0 = vld1q_f32(w);


      const float32x4_t vi0x0123 = vld1q_f32(i0);
      const float32x4_t vk0x0123 = vld1q_f32(w + 8);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1);
      const float32x4_t vk1x0123 = vld1q_f32(w + 16);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2);
      const float32x4_t vk2x0123 = vld1q_f32(w + 24);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      float32x2_t vacc01 = vget_low_f32(vacc0123);
      if (c & 2) {
        vst1_f32(output, vacc01); output += 2;
        vacc01 = vget_high_f32(vacc0123);
      }
      if (c & 1) {
        vst1_lane_f32(output, vacc01, 0); output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p8c__neonfma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }

    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;
      float32x4_t vacc4567p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi1x4567, vk1x4567);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi3x4567, vk3x4567);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      float32x4_t vacc4567 = vmaxq_f32(vacc4567p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);
      vacc4567 = vminq_f32(vacc4567, vmax);

      vst1q_f32(output, vacc0123); output += 4;
      vst1q_f32(output, vacc4567); output += 4;
    }
    for (; c >= 4; c -= 4) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w + 4);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w + 12);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w + 20);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w + 28);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      vst1q_f32(output, vacc0123); output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      float32x4_t vacc0123p0 = vld1q_f32(w);


      const float32x4_t vi0x0123 = vld1q_f32(i0);
      const float32x4_t vk0x0123 = vld1q_f32(w + 8);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1);
      const float32x4_t vk1x0123 = vld1q_f32(w + 16);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2);
      const float32x4_t vk2x0123 = vld1q_f32(w + 24);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3);
      const float32x4_t vk3x0123 = vld1q_f32(w + 32);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      float32x2_t vacc01 = vget_low_f32(vacc0123);
      if (c & 2) {
        vst1_f32(output, vacc01); output += 2;
        vacc01 = vget_high_f32(vacc0123);
      }
      if (c & 1) {
        vst1_lane_f32(output, vacc01, 0); output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  do {
    const float* w = weights;

    // First pass to process 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 8; c -= 8) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;
        float32x4_t vacc4567p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;

        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;

        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);
        float32x4_t vacc4567p1 = vmulq_f32(vi1x4567, vk1x4567);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;

        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;

        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);
        vacc4567p1 = vfmaq_f32(vacc4567p1, vi3x4567, vk3x4567);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk4x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);
        vacc4567p0 = vaddq_f32(vacc4567p0, vacc4567p1);

        vst1q_f32(b, vacc0123p0); b += 4;
        vst1q_f32(b, vacc4567p0); b += 4;
      }

      if (c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;

        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;

        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;

        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;

        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

        vst1q_f32(b, vacc0123p0); b += 4;
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 8; c -= 8) {
        float32x4_t vacc0123p0 = vld1q_f32(b);
        float32x4_t vacc4567p0 = vld1q_f32(b + 4);


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;

        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;

        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);
        float32x4_t vacc4567p1 = vmulq_f32(vi1x4567, vk1x4567);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;

        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;

        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);
        vacc4567p1 = vfmaq_f32(vacc4567p1, vi3x4567, vk3x4567);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk4x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);
        vacc4567p0 = vaddq_f32(vacc4567p0, vacc4567p1);

        vst1q_f32(b, vacc0123p0); b += 4;
        vst1q_f32(b, vacc4567p0); b += 4;
      }

      if (c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(b);


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;

        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;

        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;

        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;

        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

        vst1q_f32(b, vacc0123p0); b += 4;
      }
    }

    // Last pass to process up to 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        float32x4_t vacc0123p0 = vld1q_f32(b); b += 4;
        float32x4_t vacc4567p0 = vld1q_f32(b); b += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;

        float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk0x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;

        float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk1x4567 = vld1q_f32(w); w += 4;

        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);
        float32x4_t vacc4567p1 = vmulq_f32(vi1x4567, vk1x4567);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;

        float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk2x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;

        float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk3x4567 = vld1q_f32(w); w += 4;

        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);
        vacc4567p1 = vfmaq_f32(vacc4567p1, vi3x4567, vk3x4567);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;

        float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk4x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);
        vacc4567p0 = vaddq_f32(vacc4567p0, vacc4567p1);

        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
        float32x4_t vacc4567 = vmaxq_f32(vacc4567p0, vmin);

        vacc0123 = vminq_f32(vacc0123, vmax);
        vacc4567 = vminq_f32(vacc4567, vmax);

        vst1q_f32(output, vacc0123); output += 4;
        vst1q_f32(output, vacc4567); output += 4;
      }


      for (; c >= 4; c -= 4) {
        float32x4_t vacc0123p0 = vld1q_f32(b); b += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;

        float32x4_t vk0x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;

        float32x4_t vk1x0123 = vld1q_f32(w); w += 4;

        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;

        float32x4_t vk2x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;

        float32x4_t vk3x0123 = vld1q_f32(w); w += 4;

        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;

        float32x4_t vk4x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);


        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);

        vacc0123 = vminq_f32(vacc0123, vmax);

        vst1q_f32(output, vacc0123); output += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(b);

        const float32x4_t vi0x0123 = vld1q_f32(i0);
        float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1);
        float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2);
        float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3);
        float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4);
        float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
        vacc0123 = vminq_f32(vacc0123, vmax);

        float32x2_t vacc01 = vget_low_f32(vacc0123);
        if (c & 2) {
          vst1_f32(output, vacc01); output += 2;
          vacc01 = vget_high_f32(vacc0123);
        }
        if (c & 1) {
          vst1_lane_f32(output, vacc01, 0); output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }

    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;
      float32x4_t vacc4567p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi1x4567, vk1x4567);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi3x4567, vk3x4567);

      const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk4x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);

      const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk5x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi5x4567, vk5x4567);

      const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi6x4567 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk6x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi6x4567, vk6x4567);

      const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi7x4567 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vk7x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk7x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi7x0123, vk7x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi7x4567, vk7x4567);

      const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vi8x4567 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vk8x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk8x4567 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi8x4567, vk8x4567);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      float32x4_t vacc4567 = vmaxq_f32(vacc4567p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);
      vacc4567 = vminq_f32(vacc4567, vmax);

      vst1q_f32(output, vacc0123); output += 4;
      vst1q_f32(output, vacc4567); output += 4;
    }
    for (; c >= 4; c -= 4) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w + 4);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w + 12);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w + 20);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w + 28);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);

      const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vk4x0123 = vld1q_f32(w + 36);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

      const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vk5x0123 = vld1q_f32(w + 44);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);

      const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vk6x0123 = vld1q_f32(w + 52);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);

      const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vk7x0123 = vld1q_f32(w + 60);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi7x0123, vk7x0123);

      const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vk8x0123 = vld1q_f32(w + 68);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      vst1q_f32(output, vacc0123); output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      float32x4_t vacc0123p0 = vld1q_f32(w);


      const float32x4_t vi0x0123 = vld1q_f32(i0);
      const float32x4_t vk0x0123 = vld1q_f32(w + 8);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1);
      const float32x4_t vk1x0123 = vld1q_f32(w + 16);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2);
      const float32x4_t vk2x0123 = vld1q_f32(w + 24);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3);
      const float32x4_t vk3x0123 = vld1q_f32(w + 32);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);

      const float32x4_t vi4x0123 = vld1q_f32(i4);
      const float32x4_t vk4x0123 = vld1q_f32(w + 40);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

      const float32x4_t vi5x0123 = vld1q_f32(i5);
      const float32x4_t vk5x0123 = vld1q_f32(w + 48);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);

      const float32x4_t vi6x0123 = vld1q_f32(i6);
      const float32x4_t vk6x0123 = vld1q_f32(w + 56);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);

      const float32x4_t vi7x0123 = vld1q_f32(i7);
      const float32x4_t vk7x0123 = vld1q_f32(w + 64);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi7x0123, vk7x0123);

      const float32x4_t vi8x0123 = vld1q_f32(i8);
      const float32x4_t vk8x0123 = vld1q_f32(w + 72);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      float32x2_t vacc01 = vget_low_f32(vacc0123);
      if (c & 2) {
        vst1_f32(output, vacc01); output += 2;
        vacc01 = vget_high_f32(vacc0123);
      }
      if (c & 1) {
        vst1_lane_f32(output, vacc01, 0); output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      float32x4_t va0 = vld1q_f32(a0); a0 += 4;


      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c0);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c0);

      va0 = vextq_f32(va0, va0, 1);

      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c1);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c1);

      va0 = vextq_f32(va0, va0, 1);

      const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c2);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c2);

      va0 = vextq_f32(va0, va0, 1);

      const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c3);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c3);


      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      float32x4_t va0 = vld1q_f32(a0); a0 = (const float*) ((uintptr_t) a0 + k);


      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c0, vb0123c0);
      const float32x4_t vmska0x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c0, vb4567c0);

      va0 = vextq_f32(va0, va0, 1);

      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c1, vb0123c1);
      const float32x4_t vmska0x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c1, vb4567c1);

      va0 = vextq_f32(va0, va0, 1);

      const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c2, vb0123c2);
      const float32x4_t vmska0x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c2, vb4567c2);

      va0 = vextq_f32(va0, va0, 1);

      const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c3, vb0123c3);
      const float32x4_t vmska0x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c3, vb4567c3);

    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;

        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      float32x4_t va0 = vld1q_f32(a0); a0 += 4;
      float32x4_t va1 = vld1q_f32(a1); a1 += 4;
      float32x4_t va2 = vld1q_f32(a2); a2 += 4;
      float32x4_t va3 = vld1q_f32(a3); a3 += 4;


      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c0);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c0);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c0);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c0);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c0);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c0);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c0);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c0);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);

      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c1);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c1);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c1);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c1);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c1);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c1);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c1);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c1);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);

      const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c2);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c2);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c2);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c2);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c2);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c2);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c2);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c2);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);

      const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c3);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c3);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c3);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c3);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c3);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c3);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c3);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c3);


      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      float32x4_t va0 = vld1q_f32(a0); a0 = (const float*) ((uintptr_t) a0 + k);
      float32x4_t va1 = vld1q_f32(a1); a1 = (const float*) ((uintptr_t) a1 + k);
      float32x4_t va2 = vld1q_f32(a2); a2 = (const float*) ((uintptr_t) a2 + k);
      float32x4_t va3 = vld1q_f32(a3); a3 = (const float*) ((uintptr_t) a3 + k);


      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c0, vb0123c0);
      const float32x4_t vmska1x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c0, vb0123c0);
      const float32x4_t vmska2x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c0, vb0123c0);
      const float32x4_t vmska3x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c0, vb0123c0);
      const float32x4_t vmska0x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c0, vb4567c0);
      const float32x4_t vmska1x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c0, vb4567c0);
      const float32x4_t vmska2x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c0, vb4567c0);
      const float32x4_t vmska3x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c0, vb4567c0);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);

      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c1, vb0123c1);
      const float32x4_t vmska1x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c1, vb0123c1);
      const float32x4_t vmska2x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c1, vb0123c1);
      const float32x4_t vmska3x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c1, vb0123c1);
      const float32x4_t vmska0x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c1, vb4567c1);
      const float32x4_t vmska1x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c1, vb4567c1);
      const float32x4_t vmska2x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c1, vb4567c1);
      const float32x4_t vmska3x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c1, vb4567c1);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);

      const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c2, vb0123c2);
      const float32x4_t vmska1x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c2, vb0123c2);
      const float32x4_t vmska2x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c2, vb0123c2);
      const float32x4_t vmska3x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c2, vb0123c2);
      const float32x4_t vmska0x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c2, vb4567c2);
      const float32x4_t vmska1x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c2, vb4567c2);
      const float32x4_t vmska2x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c2, vb4567c2);
      const float32x4_t vmska3x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c2, vb4567c2);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);

      const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c3, vb0123c3);
      const float32x4_t vmska1x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c3, vb0123c3);
      const float32x4_t vmska2x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c3, vb0123c3);
      const float32x4_t vmska3x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c3, vb0123c3);
      const float32x4_t vmska0x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c3, vb4567c3);
      const float32x4_t vmska1x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c3, vb4567c3);
      const float32x4_t vmska2x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c3, vb4567c3);
      const float32x4_t vmska3x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c3, vb4567c3);

    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c3, vacc3); c3 += 2;

        vacc0 = vget_high_f32(vacc0x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc3 = vget_high_f32(vacc3x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c3, vacc3, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc5x0 = vacc0x0;
    float32x4_t vacc5x1 = vacc0x1;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      float32x4_t va0 = vld1q_f32(a0); a0 += 4;
      float32x4_t va1 = vld1q_f32(a1); a1 += 4;
      float32x4_t va2 = vld1q_f32(a2); a2 += 4;
      float32x4_t va3 = vld1q_f32(a3); a3 += 4;
      float32x4_t va4 = vld1q_f32(a4); a4 += 4;
      float32x4_t va5 = vld1q_f32(a5); a5 += 4;


      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c0);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c0);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c0);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c0);
      vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c0);
      vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c0);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c0);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c0);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c0);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c0);
      vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c0);
      vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c0);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);

      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c1);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c1);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c1);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c1);
      vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c1);
      vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c1);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c1);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c1);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c1);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c1);
      vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c1);
      vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c1);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);

      const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c2);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c2);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c2);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c2);
      vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c2);
      vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c2);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c2);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c2);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c2);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c2);
      vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c2);
      vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c2);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);

      const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c3);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c3);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c3);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c3);
      vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c3);
      vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c3);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c3);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c3);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c3);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c3);
      vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c3);
      vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c3);


      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      float32x4_t va0 = vld1q_f32(a0); a0 = (const float*) ((uintptr_t) a0 + k);
      float32x4_t va1 = vld1q_f32(a1); a1 = (const float*) ((uintptr_t) a1 + k);
      float32x4_t va2 = vld1q_f32(a2); a2 = (const float*) ((uintptr_t) a2 + k);
      float32x4_t va3 = vld1q_f32(a3); a3 = (const float*) ((uintptr_t) a3 + k);
      float32x4_t va4 = vld1q_f32(a4); a4 = (const float*) ((uintptr_t) a4 + k);
      float32x4_t va5 = vld1q_f32(a5); a5 = (const float*) ((uintptr_t) a5 + k);


      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c0, vb0123c0);
      const float32x4_t vmska1x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c0, vb0123c0);
      const float32x4_t vmska2x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c0, vb0123c0);
      const float32x4_t vmska3x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c0, vb0123c0);
      const float32x4_t vmska4x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c0, vb0123c0);
      const float32x4_t vmska5x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
      vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c0, vb0123c0);
      const float32x4_t vmska0x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c0, vb4567c0);
      const float32x4_t vmska1x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c0, vb4567c0);
      const float32x4_t vmska2x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c0, vb4567c0);
      const float32x4_t vmska3x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c0, vb4567c0);
      const float32x4_t vmska4x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c0, vb4567c0);
      const float32x4_t vmska5x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
      vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c0, vb4567c0);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);

      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c1, vb0123c1);
      const float32x4_t vmska1x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c1, vb0123c1);
      const float32x4_t vmska2x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c1, vb0123c1);
      const float32x4_t vmska3x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c1, vb0123c1);
      const float32x4_t vmska4x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c1, vb0123c1);
      const float32x4_t vmska5x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
      vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c1, vb0123c1);
      const float32x4_t vmska0x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c1, vb4567c1);
      const float32x4_t vmska1x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c1, vb4567c1);
      const float32x4_t vmska2x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c1, vb4567c1);
      const float32x4_t vmska3x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c1, vb4567c1);
      const float32x4_t vmska4x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c1, vb4567c1);
      const float32x4_t vmska5x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
      vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c1, vb4567c1);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);

      const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c2, vb0123c2);
      const float32x4_t vmska1x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c2, vb0123c2);
      const float32x4_t vmska2x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c2, vb0123c2);
      const float32x4_t vmska3x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c2, vb0123c2);
      const float32x4_t vmska4x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c2, vb0123c2);
      const float32x4_t vmska5x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
      vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c2, vb0123c2);
      const float32x4_t vmska0x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c2, vb4567c2);
      const float32x4_t vmska1x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c2, vb4567c2);
      const float32x4_t vmska2x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c2, vb4567c2);
      const float32x4_t vmska3x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c2, vb4567c2);
      const float32x4_t vmska4x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c2, vb4567c2);
      const float32x4_t vmska5x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
      vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c2, vb4567c2);

      va0 = vextq_f32(va0, va0, 1);
      va1 = vextq_f32(va1, va1, 1);
      va2 = vextq_f32(va2, va2, 1);
      va3 = vextq_f32(va3, va3, 1);
      va4 = vextq_f32(va4, va4, 1);
      va5 = vextq_f32(va5, va5, 1);

      const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

      const float32x4_t vmska0x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c3, vb0123c3);
      const float32x4_t vmska1x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c3, vb0123c3);
      const float32x4_t vmska2x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c3, vb0123c3);
      const float32x4_t vmska3x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c3, vb0123c3);
      const float32x4_t vmska4x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c3, vb0123c3);
      const float32x4_t vmska5x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
      vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c3, vb0123c3);
      const float32x4_t vmska0x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c3, vb4567c3);
      const float32x4_t vmska1x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c3, vb4567c3);
      const float32x4_t vmska2x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c3, vb4567c3);
      const float32x4_t vmska3x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c3, vb4567c3);
      const float32x4_t vmska4x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c3, vb4567c3);
      const float32x4_t vmska5x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
      vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c3, vb4567c3);

    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc4x0 = vminq_f32(vacc4x0, vmax);
    vacc5x0 = vminq_f32(vacc5x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);
    vacc4x1 = vminq_f32(vacc4x1, vmax);
    vacc5x1 = vminq_f32(vacc5x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc4x0 = vmaxq_f32(vacc4x0, vmin);
    vacc5x0 = vmaxq_f32(vacc5x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);
    vacc4x1 = vmaxq_f32(vacc4x1, vmin);
    vacc5x1 = vmaxq_f32(vacc5x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c4, vacc4x0);
      vst1q_f32(c4 + 4, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c5, vacc5x0);
      vst1q_f32(c5 + 4, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;
        vst1q_f32(c4, vacc4x0); c4 += 4;
        vst1q_f32(c5, vacc5x0); c5 += 4;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
        vacc4x0 = vacc4x1;
        vacc5x0 = vacc5x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      float32x2_t vacc4 = vget_low_f32(vacc4x0);
      float32x2_t vacc5 = vget_low_f32(vacc5x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c3, vacc3); c3 += 2;
        vst1_f32(c4, vacc4); c4 += 2;
        vst1_f32(c5, vacc5); c5 += 2;

        vacc0 = vget_high_f32(vacc0x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc3 = vget_high_f32(vacc3x0);
        vacc4 = vget_high_f32(vacc4x0);
        vacc5 = vget_high_f32(vacc5x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c3, vacc3, 0);
        vst1_lane_f32(c4, vacc4, 0);
        vst1_lane_f32(c5, vacc5, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_ibilinear_chw_ukernel__neonfma_p8(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t input_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  do {
    const float** i = input;
    const float* w = weights;
    size_t p = output_pixels;
    for (; p >= 8; p -= 8) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      const float* itl4 = (const float*) ((uintptr_t) i[8] + input_offset);
      const float* ibl4 = (const float*) ((uintptr_t) i[9] + input_offset);
      const float* itl5 = (const float*) ((uintptr_t) i[10] + input_offset);
      const float* ibl5 = (const float*) ((uintptr_t) i[11] + input_offset);
      const float* itl6 = (const float*) ((uintptr_t) i[12] + input_offset);
      const float* ibl6 = (const float*) ((uintptr_t) i[13] + input_offset);
      const float* itl7 = (const float*) ((uintptr_t) i[14] + input_offset);
      const float* ibl7 = (const float*) ((uintptr_t) i[15] + input_offset);
      i += 2 * 8;

      const float32x4x2_t vw0123 = vld2q_f32(w + 0);
      const float32x4x2_t vw4567 = vld2q_f32(w + 8);
      w += 2 * 8;

      const float32x2_t vtltr0 = vld1_f32(itl0);
      const float32x2_t vblbr0 = vld1_f32(ibl0);
      const float32x2_t vtltr1 = vld1_f32(itl1);
      const float32x2_t vblbr1 = vld1_f32(ibl1);
      const float32x2_t vtltr2 = vld1_f32(itl2);
      const float32x2_t vblbr2 = vld1_f32(ibl2);
      const float32x2_t vtltr3 = vld1_f32(itl3);
      const float32x2_t vblbr3 = vld1_f32(ibl3);
      const float32x2_t vtltr4 = vld1_f32(itl4);
      const float32x2_t vblbr4 = vld1_f32(ibl4);
      const float32x2_t vtltr5 = vld1_f32(itl5);
      const float32x2_t vblbr5 = vld1_f32(ibl5);
      const float32x2_t vtltr6 = vld1_f32(itl6);
      const float32x2_t vblbr6 = vld1_f32(ibl6);
      const float32x2_t vtltr7 = vld1_f32(itl7);
      const float32x2_t vblbr7 = vld1_f32(ibl7);

      const float32x4_t valphah0123 = vw0123.val[0];
      const float32x4_t valphav0123 = vw0123.val[1];
      const float32x4_t valphah4567 = vw4567.val[0];
      const float32x4_t valphav4567 = vw4567.val[1];

      const float32x4_t vtltr01 = vcombine_f32(vtltr0, vtltr1);
      const float32x4_t vblbr01 = vcombine_f32(vblbr0, vblbr1);
      const float32x4_t vtltr23 = vcombine_f32(vtltr2, vtltr3);
      const float32x4_t vblbr23 = vcombine_f32(vblbr2, vblbr3);
      const float32x4_t vtltr45 = vcombine_f32(vtltr4, vtltr5);
      const float32x4_t vblbr45 = vcombine_f32(vblbr4, vblbr5);
      const float32x4_t vtltr67 = vcombine_f32(vtltr6, vtltr7);
      const float32x4_t vblbr67 = vcombine_f32(vblbr6, vblbr7);

      const float32x4_t vldrd01 = vsubq_f32(vblbr01, vtltr01);
      const float32x4_t vldrd23 = vsubq_f32(vblbr23, vtltr23);
      const float32x4_t vldrd45 = vsubq_f32(vblbr45, vtltr45);
      const float32x4_t vldrd67 = vsubq_f32(vblbr67, vtltr67);

      const float32x4x2_t vld_t0123 = vuzpq_f32(vldrd01, vldrd23);
      const float32x4_t vld0123 = vld_t0123.val[0];
      const float32x4_t vrd0123 = vld_t0123.val[1];
      const float32x4x2_t vld_t4567 = vuzpq_f32(vldrd45, vldrd67);
      const float32x4_t vld4567 = vld_t4567.val[0];
      const float32x4_t vrd4567 = vld_t4567.val[1];

      const float32x4x2_t vtl_t0123 = vuzpq_f32(vtltr01, vtltr23);
      const float32x4_t vtl0123 = vtl_t0123.val[0];
      const float32x4_t vtr0123 = vtl_t0123.val[1];
      const float32x4x2_t vtl_t4567 = vuzpq_f32(vtltr45, vtltr67);
      const float32x4_t vtl4567 = vtl_t4567.val[0];
      const float32x4_t vtr4567 = vtl_t4567.val[1];

      const float32x4_t vl0123 = vfmaq_f32(vtl0123, vld0123, valphav0123);
      const float32x4_t vr0123 = vfmaq_f32(vtr0123, vrd0123, valphav0123);
      const float32x4_t vl4567 = vfmaq_f32(vtl4567, vld4567, valphav4567);
      const float32x4_t vr4567 = vfmaq_f32(vtr4567, vrd4567, valphav4567);

      const float32x4_t vd0123 = vsubq_f32(vr0123, vl0123);
      const float32x4_t vd4567 = vsubq_f32(vr4567, vl4567);

      const float32x4_t vo0123 = vfmaq_f32(vl0123, vd0123, valphah0123);
      const float32x4_t vo4567 = vfmaq_f32(vl4567, vd4567, valphah4567);

      vst1q_f32(output + 0, vo0123);
      vst1q_f32(output + 4, vo4567);
      output += 8;
    }

    for (; p >= 4; p -= 4) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      i += 8;

      const float32x4x2_t vw = vld2q_f32(w);
      w += 8;

      const float32x2_t vtltr0 = vld1_f32(itl0);
      const float32x2_t vblbr0 = vld1_f32(ibl0);
      const float32x2_t vtltr1 = vld1_f32(itl1);
      const float32x2_t vblbr1 = vld1_f32(ibl1);
      const float32x2_t vtltr2 = vld1_f32(itl2);
      const float32x2_t vblbr2 = vld1_f32(ibl2);
      const float32x2_t vtltr3 = vld1_f32(itl3);
      const float32x2_t vblbr3 = vld1_f32(ibl3);

      const float32x4_t valphah = vw.val[0];
      const float32x4_t valphav = vw.val[1];

      const float32x4_t vtltr01 = vcombine_f32(vtltr0, vtltr1);
      const float32x4_t vblbr01 = vcombine_f32(vblbr0, vblbr1);
      const float32x4_t vtltr23 = vcombine_f32(vtltr2, vtltr3);
      const float32x4_t vblbr23 = vcombine_f32(vblbr2, vblbr3);

      const float32x4_t vldrd01 = vsubq_f32(vblbr01, vtltr01);
      const float32x4_t vldrd23 = vsubq_f32(vblbr23, vtltr23);

      const float32x4x2_t vld_t = vuzpq_f32(vldrd01, vldrd23);
      const float32x4_t vld = vld_t.val[0];
      const float32x4_t vrd = vld_t.val[1];

      const float32x4x2_t vtl_t = vuzpq_f32(vtltr01, vtltr23);
      const float32x4_t vtl = vtl_t.val[0];
      const float32x4_t vtr = vtl_t.val[1];

      const float32x4_t vl = vfmaq_f32(vtl, vld, valphav);
      const float32x4_t vr = vfmaq_f32(vtr, vrd, valphav);

      const float32x4_t vd = vsubq_f32(vr, vl);
      const float32x4_t vo = vfmaq_f32(vl, vd, valphah);

      vst1q_f32(output, vo);
      output += 4;
    }

    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const float32x2x2_t vw = vld2_f32(w);
        w += 4;

        const float32x2_t valphah = vw.val[0];
        const float32x2_t valphav = vw.val[1];

        const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
        const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
        const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const float32x2_t vtltr0 = vld1_f32(itl0);
        const float32x2_t vblbr0 = vld1_f32(ibl0);
        const float32x2_t vtltr1 = vld1_f32(itl1);
        const float32x2_t vblbr1 = vld1_f32(ibl1);

        const float32x2_t vldrd0 = vsub_f32(vblbr0, vtltr0);
        const float32x2_t vldrd1 = vsub_f32(vblbr1, vtltr1);

        const float32x2x2_t vld_t = vuzp_f32(vldrd0, vldrd1);
        const float32x2_t vld = vld_t.val[0];
        const float32x2_t vrd = vld_t.val[1];

        const float32x2x2_t vtl_t = vuzp_f32(vtltr0, vtltr1);
        const float32x2_t vtl = vtl_t.val[0];
        const float32x2_t vtr = vtl_t.val[1];

        const float32x2_t vl = vfma_f32(vtl, vld, valphav);
        const float32x2_t vr = vfma_f32(vtr, vrd, valphav);

        const float32x2_t vd = vsub_f32(vr, vl);
        const float32x2_t vo = vfma_f32(vl, vd, valphah);

        vst1_f32(output, vo);
        output += 2;
      }

      if (p & 1) {
        // We are computing the following formula:
        //   result = (1 - alpha_h) * (1 - alpha_v) * top_left +
        //                 alpha_h  * (1 - alpha_v) * top_right +
        //            (1 - alpha_h) *      alpha_v  * bottom_left +
        //                 alpha_h  *      alpha_v  * bottom_right.
        //
        // Rearranging gives
        //   result =    left + alpha_h * (right        - left),
        // where
        //   left =  top_left + alpha_v * (bottom_left  - top_left),
        //  right = top_right + alpha_v * (bottom_right - top_right).

        const float alphah = *w;
        const float32x2_t valphav = vld1_dup_f32(w + 1);
        w += 2;

        const float* itl = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl = (const float*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        const float32x2_t vtltr = vld1_f32(itl);
        const float32x2_t vblbr = vld1_f32(ibl);

        // Compute at once
        //    left_diff = bottom_left  - top_left
        //   right_diff = bottom_right - top_right
        const float32x2_t vldrd = vsub_f32(vblbr, vtltr);
        const float32x2_t vlr = vfma_f32(vtltr, vldrd, valphav);

        // Extract them and compute the result.
        const float l = vget_lane_f32(vlr, 0);
        const float r = vget_lane_f32(vlr, 1);

        *output++ = l + alphah * (r - l);
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}

void xnn_f32_ibilinear_ukernel__neonfma_c8(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  do {
    const float* i0 = (const float*) ((uintptr_t) input[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) input[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) input[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const float32x2_t valphahv = vld1_f32(weights); weights += 2;
    #if XNN_ARCH_ARM
      const float32x4_t valphah = vdupq_lane_f32(valphahv, 0);
      const float32x4_t valphav = vdupq_lane_f32(valphahv, 1);
    #endif

    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const float32x4_t vtl0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vtr0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vbl0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vbr0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vtl4567 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vtr4567 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vbl4567 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vbr4567 = vld1q_f32(i3); i3 += 4;

      const float32x4_t vtd0123 = vsubq_f32(vtr0123, vtl0123);
      const float32x4_t vbd0123 = vsubq_f32(vbr0123, vbl0123);
      const float32x4_t vtd4567 = vsubq_f32(vtr4567, vtl4567);
      const float32x4_t vbd4567 = vsubq_f32(vbr4567, vbl4567);

      #if XNN_ARCH_ARM
      const float32x4_t vt0123 = vfmaq_f32(vtl0123, vtd0123, valphah);
      const float32x4_t vb0123 = vfmaq_f32(vbl0123, vbd0123, valphah);
      const float32x4_t vt4567 = vfmaq_f32(vtl4567, vtd4567, valphah);
      const float32x4_t vb4567 = vfmaq_f32(vbl4567, vbd4567, valphah);
      #else
      const float32x4_t vt0123 = vfmaq_lane_f32(vtl0123, vtd0123, valphahv, 0);
      const float32x4_t vb0123 = vfmaq_lane_f32(vbl0123, vbd0123, valphahv, 0);
      const float32x4_t vt4567 = vfmaq_lane_f32(vtl4567, vtd4567, valphahv, 0);
      const float32x4_t vb4567 = vfmaq_lane_f32(vbl4567, vbd4567, valphahv, 0);
      #endif

      const float32x4_t vd0123 = vsubq_f32(vb0123, vt0123);
      const float32x4_t vd4567 = vsubq_f32(vb4567, vt4567);

      #if XNN_ARCH_ARM
      const float32x4_t vo0123 = vfmaq_f32(vt0123, vd0123, valphav);
      const float32x4_t vo4567 = vfmaq_f32(vt4567, vd4567, valphav);
      #else
      const float32x4_t vo0123 = vfmaq_lane_f32(vt0123, vd0123, valphahv, 1);
      const float32x4_t vo4567 = vfmaq_lane_f32(vt4567, vd4567, valphahv, 1);
      #endif

      vst1q_f32(output, vo0123); output += 4;
      vst1q_f32(output, vo4567); output += 4;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float32x4_t vtl0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vtr0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vbl0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vbr0123 = vld1q_f32(i3); i3 += 4;

      const float32x4_t vtd0123 = vsubq_f32(vtr0123, vtl0123);
      const float32x4_t vbd0123 = vsubq_f32(vbr0123, vbl0123);

      #if XNN_ARCH_ARM
      const float32x4_t vt0123 = vfmaq_f32(vtl0123, vtd0123, valphah);
      const float32x4_t vb0123 = vfmaq_f32(vbl0123, vbd0123, valphah);
      #else
      const float32x4_t vt0123 = vfmaq_lane_f32(vtl0123, vtd0123, valphahv, 0);
      const float32x4_t vb0123 = vfmaq_lane_f32(vbl0123, vbd0123, valphahv, 0);
      #endif

      const float32x4_t vd0123 = vsubq_f32(vb0123, vt0123);

      #if XNN_ARCH_ARM
      const float32x4_t vo0123 = vfmaq_f32(vt0123, vd0123, valphav);
      #else
      const float32x4_t vo0123 = vfmaq_lane_f32(vt0123, vd0123, valphahv, 1);
      #endif

      vst1q_f32(output, vo0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const float32x4_t vtl0123 = vld1q_f32(i0);
      const float32x4_t vtr0123 = vld1q_f32(i1);
      const float32x4_t vbl0123 = vld1q_f32(i2);
      const float32x4_t vbr0123 = vld1q_f32(i3);

      const float32x4_t vtd0123 = vsubq_f32(vtr0123, vtl0123);
      const float32x4_t vbd0123 = vsubq_f32(vbr0123, vbl0123);

        #if XNN_ARCH_ARM
        const float32x4_t vt0123 = vfmaq_f32(vtl0123, vtd0123, valphah);
        const float32x4_t vb0123 = vfmaq_f32(vbl0123, vbd0123, valphah);
        #else
        const float32x4_t vt0123 = vfmaq_lane_f32(vtl0123, vtd0123, valphahv, 0);
        const float32x4_t vb0123 = vfmaq_lane_f32(vbl0123, vbd0123, valphahv, 0);
        #endif

      const float32x4_t vd0123 = vsubq_f32(vb0123, vt0123);

      #if XNN_ARCH_ARM
      float32x4_t vo0123 = vfmaq_f32(vt0123, vd0123, valphav);
      #else
      float32x4_t vo0123 = vfmaq_lane_f32(vt0123, vd0123, valphahv, 1);
      #endif

      float32x2_t vo01 = vget_low_f32(vo0123);
      if (c & (2 * sizeof(float))) {
        vst1_f32(output, vo01); output += 2;
        vo01 = vget_high_f32(vo0123);
      }
      if (c & (1 * sizeof(float))) {
        vst1_lane_f32(output, vo01, 0); output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        float32x4_t va0 = vld1q_f32(a0); a0 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w + 0);
        const float32x4_t vb4567c0 = vld1q_f32(w + 4);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c0);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c0);

        va0 = vextq_f32(va0, va0, 1);

        const float32x4_t vb0123c1 = vld1q_f32(w + 8);
        const float32x4_t vb4567c1 = vld1q_f32(w + 12);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c1);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c1);

        va0 = vextq_f32(va0, va0, 1);

        const float32x4_t vb0123c2 = vld1q_f32(w + 16);
        const float32x4_t vb4567c2 = vld1q_f32(w + 20);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c2);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c2);

        va0 = vextq_f32(va0, va0, 1);

        const float32x4_t vb0123c3 = vld1q_f32(w + 24);
        const float32x4_t vb4567c3 = vld1q_f32(w + 28);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c3);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c3);


        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        float32x4_t va0 = vld1q_f32(a0); a0 = (const float*) ((uintptr_t) a0 + k);


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c0, vb0123c0);
        const float32x4_t vmska0x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c0, vb4567c0);

        va0 = vextq_f32(va0, va0, 1);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c1, vb0123c1);
        const float32x4_t vmska0x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c1, vb4567c1);

        va0 = vextq_f32(va0, va0, 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c2, vb0123c2);
        const float32x4_t vmska0x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c2, vb4567c2);

        va0 = vextq_f32(va0, va0, 1);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c3, vb0123c3);
        const float32x4_t vmska0x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c3, vb4567c3);

      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;

        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        float32x4_t va3 = vld1q_f32(a3); a3 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w + 0);
        const float32x4_t vb4567c0 = vld1q_f32(w + 4);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c0);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c0);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c0);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c0);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c0);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c0);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c0);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c0);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);

        const float32x4_t vb0123c1 = vld1q_f32(w + 8);
        const float32x4_t vb4567c1 = vld1q_f32(w + 12);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c1);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c1);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c1);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c1);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c1);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c1);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c1);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c1);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);

        const float32x4_t vb0123c2 = vld1q_f32(w + 16);
        const float32x4_t vb4567c2 = vld1q_f32(w + 20);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c2);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c2);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c2);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c2);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c2);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c2);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c2);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c2);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);

        const float32x4_t vb0123c3 = vld1q_f32(w + 24);
        const float32x4_t vb4567c3 = vld1q_f32(w + 28);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c3);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c3);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c3);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c3);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c3);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c3);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c3);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c3);


        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        float32x4_t va0 = vld1q_f32(a0); a0 = (const float*) ((uintptr_t) a0 + k);
        float32x4_t va1 = vld1q_f32(a1); a1 = (const float*) ((uintptr_t) a1 + k);
        float32x4_t va2 = vld1q_f32(a2); a2 = (const float*) ((uintptr_t) a2 + k);
        float32x4_t va3 = vld1q_f32(a3); a3 = (const float*) ((uintptr_t) a3 + k);


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c0, vb0123c0);
        const float32x4_t vmska1x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c0, vb0123c0);
        const float32x4_t vmska2x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c0, vb0123c0);
        const float32x4_t vmska3x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c0, vb0123c0);
        const float32x4_t vmska0x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c0, vb4567c0);
        const float32x4_t vmska1x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c0, vb4567c0);
        const float32x4_t vmska2x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c0, vb4567c0);
        const float32x4_t vmska3x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c0, vb4567c0);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c1, vb0123c1);
        const float32x4_t vmska1x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c1, vb0123c1);
        const float32x4_t vmska2x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c1, vb0123c1);
        const float32x4_t vmska3x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c1, vb0123c1);
        const float32x4_t vmska0x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c1, vb4567c1);
        const float32x4_t vmska1x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c1, vb4567c1);
        const float32x4_t vmska2x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c1, vb4567c1);
        const float32x4_t vmska3x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c1, vb4567c1);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c2, vb0123c2);
        const float32x4_t vmska1x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c2, vb0123c2);
        const float32x4_t vmska2x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c2, vb0123c2);
        const float32x4_t vmska3x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c2, vb0123c2);
        const float32x4_t vmska0x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c2, vb4567c2);
        const float32x4_t vmska1x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c2, vb4567c2);
        const float32x4_t vmska2x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c2, vb4567c2);
        const float32x4_t vmska3x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c2, vb4567c2);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c3, vb0123c3);
        const float32x4_t vmska1x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c3, vb0123c3);
        const float32x4_t vmska2x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c3, vb0123c3);
        const float32x4_t vmska3x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c3, vb0123c3);
        const float32x4_t vmska0x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c3, vb4567c3);
        const float32x4_t vmska1x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c3, vb4567c3);
        const float32x4_t vmska2x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c3, vb4567c3);
        const float32x4_t vmska3x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c3, vb4567c3);

      }

      p -= 4 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c3, vacc3x0); c3 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc3x0 = vacc3x1;
        vacc2x0 = vacc2x1;
        vacc1x0 = vacc1x1;
        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c3, vacc3); c3 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c0, vacc0); c0 += 2;

        vacc3 = vget_high_f32(vacc3x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c3, vacc3, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc5x0 = vacc0x0;
    float32x4_t vacc5x1 = vacc0x1;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        float32x4_t va3 = vld1q_f32(a3); a3 += 4;
        float32x4_t va4 = vld1q_f32(a4); a4 += 4;
        float32x4_t va5 = vld1q_f32(a5); a5 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w + 0);
        const float32x4_t vb4567c0 = vld1q_f32(w + 4);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c0);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c0);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c0);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c0);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c0);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c0);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c0);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c0);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c0);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c0);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c0);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c0);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);

        const float32x4_t vb0123c1 = vld1q_f32(w + 8);
        const float32x4_t vb4567c1 = vld1q_f32(w + 12);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c1);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c1);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c1);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c1);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c1);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c1);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c1);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c1);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c1);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c1);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c1);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c1);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);

        const float32x4_t vb0123c2 = vld1q_f32(w + 16);
        const float32x4_t vb4567c2 = vld1q_f32(w + 20);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c2);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c2);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c2);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c2);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c2);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c2);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c2);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c2);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c2);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c2);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c2);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c2);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);

        const float32x4_t vb0123c3 = vld1q_f32(w + 24);
        const float32x4_t vb4567c3 = vld1q_f32(w + 28);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123c3);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123c3);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123c3);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123c3);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123c3);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123c3);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567c3);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567c3);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567c3);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567c3);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567c3);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567c3);


        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        float32x4_t va0 = vld1q_f32(a0); a0 = (const float*) ((uintptr_t) a0 + k);
        float32x4_t va1 = vld1q_f32(a1); a1 = (const float*) ((uintptr_t) a1 + k);
        float32x4_t va2 = vld1q_f32(a2); a2 = (const float*) ((uintptr_t) a2 + k);
        float32x4_t va3 = vld1q_f32(a3); a3 = (const float*) ((uintptr_t) a3 + k);
        float32x4_t va4 = vld1q_f32(a4); a4 = (const float*) ((uintptr_t) a4 + k);
        float32x4_t va5 = vld1q_f32(a5); a5 = (const float*) ((uintptr_t) a5 + k);


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c0, vb0123c0);
        const float32x4_t vmska1x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c0, vb0123c0);
        const float32x4_t vmska2x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c0, vb0123c0);
        const float32x4_t vmska3x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c0, vb0123c0);
        const float32x4_t vmska4x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c0, vb0123c0);
        const float32x4_t vmska5x0123c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c0, vmovq_n_f32(0.0f))));
        vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c0, vb0123c0);
        const float32x4_t vmska0x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c0, vb4567c0);
        const float32x4_t vmska1x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c0, vb4567c0);
        const float32x4_t vmska2x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c0, vb4567c0);
        const float32x4_t vmska3x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c0, vb4567c0);
        const float32x4_t vmska4x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c0, vb4567c0);
        const float32x4_t vmska5x4567c0 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c0, vmovq_n_f32(0.0f))));
        vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c0, vb4567c0);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c1, vb0123c1);
        const float32x4_t vmska1x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c1, vb0123c1);
        const float32x4_t vmska2x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c1, vb0123c1);
        const float32x4_t vmska3x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c1, vb0123c1);
        const float32x4_t vmska4x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c1, vb0123c1);
        const float32x4_t vmska5x0123c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c1, vmovq_n_f32(0.0f))));
        vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c1, vb0123c1);
        const float32x4_t vmska0x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c1, vb4567c1);
        const float32x4_t vmska1x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c1, vb4567c1);
        const float32x4_t vmska2x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c1, vb4567c1);
        const float32x4_t vmska3x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c1, vb4567c1);
        const float32x4_t vmska4x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c1, vb4567c1);
        const float32x4_t vmska5x4567c1 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c1, vmovq_n_f32(0.0f))));
        vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c1, vb4567c1);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c2, vb0123c2);
        const float32x4_t vmska1x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c2, vb0123c2);
        const float32x4_t vmska2x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c2, vb0123c2);
        const float32x4_t vmska3x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c2, vb0123c2);
        const float32x4_t vmska4x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c2, vb0123c2);
        const float32x4_t vmska5x0123c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c2, vmovq_n_f32(0.0f))));
        vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c2, vb0123c2);
        const float32x4_t vmska0x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c2, vb4567c2);
        const float32x4_t vmska1x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c2, vb4567c2);
        const float32x4_t vmska2x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c2, vb4567c2);
        const float32x4_t vmska3x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c2, vb4567c2);
        const float32x4_t vmska4x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c2, vb4567c2);
        const float32x4_t vmska5x4567c2 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c2, vmovq_n_f32(0.0f))));
        vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c2, vb4567c2);

        va0 = vextq_f32(va0, va0, 1);
        va1 = vextq_f32(va1, va1, 1);
        va2 = vextq_f32(va2, va2, 1);
        va3 = vextq_f32(va3, va3, 1);
        va4 = vextq_f32(va4, va4, 1);
        va5 = vextq_f32(va5, va5, 1);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        const float32x4_t vmska0x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc0x0 = vfmaq_f32(vacc0x0, vmska0x0123c3, vb0123c3);
        const float32x4_t vmska1x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc1x0 = vfmaq_f32(vacc1x0, vmska1x0123c3, vb0123c3);
        const float32x4_t vmska2x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc2x0 = vfmaq_f32(vacc2x0, vmska2x0123c3, vb0123c3);
        const float32x4_t vmska3x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc3x0 = vfmaq_f32(vacc3x0, vmska3x0123c3, vb0123c3);
        const float32x4_t vmska4x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc4x0 = vfmaq_f32(vacc4x0, vmska4x0123c3, vb0123c3);
        const float32x4_t vmska5x0123c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb0123c3, vmovq_n_f32(0.0f))));
        vacc5x0 = vfmaq_f32(vacc5x0, vmska5x0123c3, vb0123c3);
        const float32x4_t vmska0x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va0), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc0x1 = vfmaq_f32(vacc0x1, vmska0x4567c3, vb4567c3);
        const float32x4_t vmska1x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va1), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc1x1 = vfmaq_f32(vacc1x1, vmska1x4567c3, vb4567c3);
        const float32x4_t vmska2x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va2), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc2x1 = vfmaq_f32(vacc2x1, vmska2x4567c3, vb4567c3);
        const float32x4_t vmska3x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va3), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc3x1 = vfmaq_f32(vacc3x1, vmska3x4567c3, vb4567c3);
        const float32x4_t vmska4x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va4), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc4x1 = vfmaq_f32(vacc4x1, vmska4x4567c3, vb4567c3);
        const float32x4_t vmska5x4567c3 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(va5), vceqq_f32(vb4567c3, vmovq_n_f32(0.0f))));
        vacc5x1 = vfmaq_f32(vacc5x1, vmska5x4567c3, vb4567c3);

      }

      p -= 6 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc4x0 = vminq_f32(vacc4x0, vmax);
    vacc5x0 = vminq_f32(vacc5x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);
    vacc4x1 = vminq_f32(vacc4x1, vmax);
    vacc5x1 = vminq_f32(vacc5x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc4x0 = vmaxq_f32(vacc4x0, vmin);
    vacc5x0 = vmaxq_f32(vacc5x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);
    vacc4x1 = vmaxq_f32(vacc4x1, vmin);
    vacc5x1 = vmaxq_f32(vacc5x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c5, vacc5x0);
      vst1q_f32(c5 + 4, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      vst1q_f32(c4, vacc4x0);
      vst1q_f32(c4 + 4, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c5, vacc5x0); c5 += 4;
        vst1q_f32(c4, vacc4x0); c4 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc5x0 = vacc5x1;
        vacc4x0 = vacc4x1;
        vacc3x0 = vacc3x1;
        vacc2x0 = vacc2x1;
        vacc1x0 = vacc1x1;
        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc5 = vget_low_f32(vacc5x0);
      float32x2_t vacc4 = vget_low_f32(vacc4x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c5, vacc5); c5 += 2;
        vst1_f32(c4, vacc4); c4 += 2;
        vst1_f32(c3, vacc3); c3 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c0, vacc0); c0 += 2;

        vacc5 = vget_high_f32(vacc5x0);
        vacc4 = vget_high_f32(vacc4x0);
        vacc3 = vget_high_f32(vacc3x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c5, vacc5, 0);
        vst1_lane_f32(c4, vacc4, 0);
        vst1_lane_f32(c3, vacc3, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

extern XNN_INTERNAL const float xnn_table_exp2_k_over_64[64];

void xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p17f);
  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0x3F));
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFF0Ap-2f);
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float32x4_t vminus_ln2 = vmovq_n_f32(-0x1.62E430p-1f);
  XNN_FORCE_REALIZATION(vminus_ln2);

  const float32x4_t vi_max = vld1q_dup_f32(max);

  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vi0123 = vld1q_f32(input); input += 4;
    const float32x4_t vi4567 = vld1q_f32(input); input += 4;
    const float32x4_t vi89AB = vld1q_f32(input); input += 4;
    const float32x4_t viCDEF = vld1q_f32(input); input += 4;

    const float32x4_t vx0123 = vsubq_f32(vi0123, vi_max);
    const float32x4_t vx4567 = vsubq_f32(vi4567, vi_max);
    const float32x4_t vx89AB = vsubq_f32(vi89AB, vi_max);
    const float32x4_t vxCDEF = vsubq_f32(viCDEF, vi_max);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vx0123, vlog2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vx4567, vlog2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vx89AB, vlog2e);
    float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vxCDEF, vlog2e);

    const int32x4_t ve0123 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn0123), vmovq_n_s32(INT32_C(0x3F))), 17);
    const int32x4_t ve4567 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn4567), vmovq_n_s32(INT32_C(0x3F))), 17);
    const int32x4_t ve89AB = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn89AB), vmovq_n_s32(INT32_C(0x3F))), 17);
    const int32x4_t veCDEF = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vnCDEF), vmovq_n_s32(INT32_C(0x3F))), 17);

    const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask));
    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask));
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);
    const uint64x2_t vidx89AB = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn89AB), vindex_mask));
    const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0);
    const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1);
    const uint64x2_t vidxCDEF = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vnCDEF), vindex_mask));
    const uint64_t vidxCD = vgetq_lane_u64(vidxCDEF, 0);
    const uint64_t vidxEF = vgetq_lane_u64(vidxCDEF, 1);

    float32x2_t vl01 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx01]);
    float32x2_t vl23 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx23]);
    float32x2_t vl45 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx45]);
    float32x2_t vl67 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx67]);
    float32x2_t vl89 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx89]);
    float32x2_t vlAB = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidxAB]);
    float32x2_t vlCD = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidxCD]);
    float32x2_t vlEF = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidxEF]);

    vl01 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx01 >> 32)], vl01, 1);
    vl23 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx23 >> 32)], vl23, 1);
    const float32x4_t vl0123 = vcombine_f32(vl01, vl23);
    vl45 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx45 >> 32)], vl45, 1);
    vl67 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx67 >> 32)], vl67, 1);
    const float32x4_t vl4567 = vcombine_f32(vl45, vl67);
    vl89 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx89 >> 32)], vl89, 1);
    vlAB = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidxAB >> 32)], vlAB, 1);
    const float32x4_t vl89AB = vcombine_f32(vl89, vlAB);
    vlCD = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidxCD >> 32)], vlCD, 1);
    vlEF = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidxEF >> 32)], vlEF, 1);
    const float32x4_t vlCDEF = vcombine_f32(vlCD, vlEF);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl0123), ve0123));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl4567), ve4567));
    const float32x4_t vs89AB = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl89AB), ve89AB));
    const float32x4_t vsCDEF = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vlCDEF), veCDEF));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);

    float32x4_t vt0123 = vfmaq_f32(vx0123, vn0123, vminus_ln2);
    float32x4_t vt4567 = vfmaq_f32(vx4567, vn4567, vminus_ln2);
    float32x4_t vt89AB = vfmaq_f32(vx89AB, vn89AB, vminus_ln2);
    float32x4_t vtCDEF = vfmaq_f32(vxCDEF, vnCDEF, vminus_ln2);

    float32x4_t vp0123 = vmulq_f32(vt0123, vc2);
    float32x4_t vp4567 = vmulq_f32(vt4567, vc2);
    float32x4_t vp89AB = vmulq_f32(vt89AB, vc2);
    float32x4_t vpCDEF = vmulq_f32(vtCDEF, vc2);

    vp0123 = vfmaq_f32(vt0123, vt0123, vp0123);
    vp4567 = vfmaq_f32(vt4567, vt4567, vp4567);
    vp89AB = vfmaq_f32(vt89AB, vt89AB, vp89AB);
    vpCDEF = vfmaq_f32(vtCDEF, vtCDEF, vpCDEF);

    float32x4_t vf0123 = vfmaq_f32(vs0123, vs0123, vp0123);
    float32x4_t vf4567 = vfmaq_f32(vs4567, vs4567, vp4567);
    float32x4_t vf89AB = vfmaq_f32(vs89AB, vs89AB, vp89AB);
    float32x4_t vfCDEF = vfmaq_f32(vsCDEF, vsCDEF, vpCDEF);

    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcltq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcltq_f32(vx4567, vdenorm_cutoff)));
    vf89AB = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf89AB), vcltq_f32(vx89AB, vdenorm_cutoff)));
    vfCDEF = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vfCDEF), vcltq_f32(vxCDEF, vdenorm_cutoff)));

    vst1q_f32(output, vf0123); output += 4;
    vst1q_f32(output, vf4567); output += 4;
    vst1q_f32(output, vf89AB); output += 4;
    vst1q_f32(output, vfCDEF); output += 4;

    vacc0 = vaddq_f32(vacc0, vf0123);
    vacc0 = vaddq_f32(vacc0, vf4567);
    vacc0 = vaddq_f32(vacc0, vf89AB);
    vacc0 = vaddq_f32(vacc0, vfCDEF);
  }

  float32x4_t vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vi = vld1q_f32(input); input += 4;

    const float32x4_t vx = vsubq_f32(vi, vi_max);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17);

    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2);

    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vfmaq_f32(vt, vt, vp);

    float32x4_t vf = vfmaq_f32(vs, vs, vp);

    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));

    vst1q_f32(output, vf); output += 4;

    vacc = vaddq_f32(vacc, vf);
  }
#if XNN_ARCH_ARM64
  float vacc_lo = vaddvq_f32(vacc);
#else
  float32x2_t vacc_lo = vadd_f32(vget_high_f32(vacc), vget_low_f32(vacc));
#endif
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const float32x4_t vi = vld1q_f32(input); input += 4;

    const float32x4_t vx = vsubq_f32(vi, vi_max);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17);

    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2);

    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vfmaq_f32(vt, vt, vp);

    float32x4_t vf = vfmaq_f32(vs, vs, vp);

    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));

    float32x2_t vf_lo = vget_low_f32(vf);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vf_lo); output += 2;

      #if XNN_ARCH_ARM64
        vacc_lo += vaddv_f32(vf_lo);
      #else
        vacc_lo = vadd_f32(vacc_lo, vf_lo);
      #endif

      vf_lo = vget_high_f32(vf);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vf_lo, 0);

      #if XNN_ARCH_ARM64
        vacc_lo += vget_lane_f32(vf_lo, 0);
      #else
        vacc_lo = vadd_f32(vacc_lo, vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vf_lo), 32)));
      #endif
    }
  }
#if XNN_ARCH_ARM64
  *sum = vacc_lo;
#else
  vst1_lane_f32(sum, vpadd_f32(vacc_lo, vacc_lo), 0);
#endif
}

void xnn_f32_spmm_minmax_ukernel_32x1__neonfma_pipelined(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  #if XNN_ARCH_ARM64
    const float32x4x2_t vminmax = vld2q_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vminmax.val[0];
    const float32x4_t vmax = vminmax.val[1];
  #else
    const float32x2x2_t vminmax = vld2_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vcombine_f32(vminmax.val[0], vminmax.val[0]);
    const float32x4_t vmax = vcombine_f32(vminmax.val[1], vminmax.val[1]);
  #endif

  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    float32x4_t vw = vld1q_dup_f32(w); w += 1;
    intptr_t diff = *dmap++;
    float32x4_t vi0123 = vld1q_f32(input);
    float32x4_t vi4567 = vld1q_f32(input + 4);
    float32x4_t vi89AB = vld1q_f32(input + 8);
    float32x4_t viCDEF = vld1q_f32(input + 12);
    float32x4_t viGHIJ = vld1q_f32(input + 16);
    float32x4_t viKLMN = vld1q_f32(input + 20);
    float32x4_t viOPQR = vld1q_f32(input + 24);
    float32x4_t viSTUV = vld1q_f32(input + 28);
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123 = vw;
      float32x4_t vacc4567 = vw;
      float32x4_t vacc89AB = vw;
      float32x4_t vaccCDEF = vw;
      float32x4_t vaccGHIJ = vw;
      float32x4_t vaccKLMN = vw;
      float32x4_t vaccOPQR = vw;
      float32x4_t vaccSTUV = vw;
      vw = vld1q_dup_f32(w); w += 1;
      if XNN_LIKELY(nnz != 0) {
        do {
          vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
          vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
          vacc89AB = vfmaq_f32(vacc89AB, vi89AB, vw);
          vaccCDEF = vfmaq_f32(vaccCDEF, viCDEF, vw);
          vaccGHIJ = vfmaq_f32(vaccGHIJ, viGHIJ, vw);
          vaccKLMN = vfmaq_f32(vaccKLMN, viKLMN, vw);
          vaccOPQR = vfmaq_f32(vaccOPQR, viOPQR, vw);
          vaccSTUV = vfmaq_f32(vaccSTUV, viSTUV, vw);
          input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
          xnn_prefetch_to_l1(input + 16);
          xnn_prefetch_to_l1(input + 32);
          diff = *dmap++;
          vw = vld1q_dup_f32(w); w += 1;
          xnn_prefetch_to_l1(w + 32);
          vi0123 = vld1q_f32(input);
          vi4567 = vld1q_f32(input + 4);
          vi89AB = vld1q_f32(input + 8);
          viCDEF = vld1q_f32(input + 12);
          viGHIJ = vld1q_f32(input + 16);
          viKLMN = vld1q_f32(input + 20);
          viOPQR = vld1q_f32(input + 24);
          viSTUV = vld1q_f32(input + 28);
        } while (--nnz != 0);
      }
      float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
      float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
      float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
      float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);
      float32x4_t voutGHIJ = vminq_f32(vaccGHIJ, vmax);
      float32x4_t voutKLMN = vminq_f32(vaccKLMN, vmax);
      float32x4_t voutOPQR = vminq_f32(vaccOPQR, vmax);
      float32x4_t voutSTUV = vminq_f32(vaccSTUV, vmax);
      vout0123 = vmaxq_f32(vout0123, vmin);
      vout4567 = vmaxq_f32(vout4567, vmin);
      vout89AB = vmaxq_f32(vout89AB, vmin);
      voutCDEF = vmaxq_f32(voutCDEF, vmin);
      voutGHIJ = vmaxq_f32(voutGHIJ, vmin);
      voutKLMN = vmaxq_f32(voutKLMN, vmin);
      voutOPQR = vmaxq_f32(voutOPQR, vmin);
      voutSTUV = vmaxq_f32(voutSTUV, vmin);
      vst1q_f32(output, vout0123);
      vst1q_f32(output + 4, vout4567);
      vst1q_f32(output + 8, vout89AB);
      vst1q_f32(output + 12, voutCDEF);
      vst1q_f32(output + 16, voutGHIJ);
      vst1q_f32(output + 20, voutKLMN);
      vst1q_f32(output + 24, voutOPQR);
      vst1q_f32(output + 28, voutSTUV);
      output = (float*) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        float32x4_t vacc89AB = vacc0123;
        float32x4_t vaccCDEF = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vb = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vfmaq_f32(vacc0123, vi0123, vb);
            vacc4567 = vfmaq_f32(vacc4567, vi4567, vb);
            vacc89AB = vfmaq_f32(vacc89AB, vi89AB, vb);
            vaccCDEF = vfmaq_f32(vaccCDEF, viCDEF, vb);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
        float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
        float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);
        vout0123 = vmaxq_f32(vout0123, vmin);
        vout4567 = vmaxq_f32(vout4567, vmin);
        vout89AB = vmaxq_f32(vout89AB, vmin);
        voutCDEF = vmaxq_f32(voutCDEF, vmin);
        vst1q_f32(output, vout0123);
        vst1q_f32(output + 4, vout4567);
        vst1q_f32(output + 8, vout89AB);
        vst1q_f32(output + 12, voutCDEF);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vb = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vfmaq_f32(vacc0123, vi0123, vb);
            vacc4567 = vfmaq_f32(vacc4567, vi4567, vb);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
        vout0123 = vmaxq_f32(vout0123, vmin);
        vout4567 = vmaxq_f32(vout4567, vmin);
        vst1q_f32(output, vout0123);
        vst1q_f32(output + 4, vout4567);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vb = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vfmaq_f32(vacc0123, vi0123, vb);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        vout0123 = vmaxq_f32(vout0123, vmin);
        vst1q_f32(output, vout0123);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x2_t vb = vld1_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc01 = vfma_f32(vacc01, vi01, vb);
          } while (--nnz != 0);
        }
        float32x2_t vout01 = vmin_f32(vacc01, vget_low_f32(vmax));
        vout01 = vmax_f32(vout01, vget_low_f32(vmin));
        vst1_f32(output, vout01);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x2_t vb = vld1_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0 = vfma_f32(vacc0, vi0, vb);
          } while (--nnz != 0);
        }
        float32x2_t vout0 = vmin_f32(vacc0, vget_low_f32(vmax));
        vout0 = vmax_f32(vout0, vget_low_f32(vmin));
        vst1_lane_f32(output, vout0, 0);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}

extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vsat_cutoff = vmovq_n_f32(-0x1.154246p+4f);
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p19f);
  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0xF));
  const float32x4_t vc3 = vmovq_n_f32(0x1.55561Cp-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.0001ECp-1f);
  const float32x4_t vone = vmovq_n_f32(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);

  const float32x4_t vminus_ln2 = vmovq_n_f32(-0x1.62E430p-1f);
  XNN_FORCE_REALIZATION(vminus_ln2);

  const float32x4_t vprescale = vld1q_dup_f32(&params->scalar.prescale);
  const float32x4_t valpha = vld1q_dup_f32(&params->scalar.alpha);
  const float32x4_t vbeta = vld1q_dup_f32(&params->scalar.beta);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(input); input += 4;
    float32x4_t vx4567 = vld1q_f32(input); input += 4;
    float32x4_t vx89AB = vld1q_f32(input); input += 4;
    float32x4_t vxCDEF = vld1q_f32(input); input += 4;

    const float32x4_t vz0123 = vmaxq_f32(vmulq_f32(vx0123, vprescale), vsat_cutoff);
    const float32x4_t vz4567 = vmaxq_f32(vmulq_f32(vx4567, vprescale), vsat_cutoff);
    const float32x4_t vz89AB = vmaxq_f32(vmulq_f32(vx89AB, vprescale), vsat_cutoff);
    const float32x4_t vzCDEF = vmaxq_f32(vmulq_f32(vxCDEF, vprescale), vsat_cutoff);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vlog2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vlog2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vz89AB, vlog2e);
    float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vzCDEF, vlog2e);

    const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask), 2));
    const int32x4_t ven0123 = vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 19);
    const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask), 2));
    const int32x4_t ven4567 = vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 19);
    const uint64x2_t vidx89AB = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn89AB), vindex_mask), 2));
    const int32x4_t ven89AB = vshlq_n_s32(vreinterpretq_s32_f32(vn89AB), 19);
    const uint64x2_t vidxCDEF = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vnCDEF), vindex_mask), 2));
    const int32x4_t venCDEF = vshlq_n_s32(vreinterpretq_s32_f32(vnCDEF), 19);

    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    int32x2_t vl01 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx01));
    int32x2_t vl23 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx23));
    vl01 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx01 >> 32)), vl01, 1);
    vl23 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx23 >> 32)), vl23, 1);
    const int32x4_t vl0123 = vcombine_s32(vl01, vl23);
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);
    int32x2_t vl45 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx45));
    int32x2_t vl67 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx67));
    vl45 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx45 >> 32)), vl45, 1);
    vl67 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx67 >> 32)), vl67, 1);
    const int32x4_t vl4567 = vcombine_s32(vl45, vl67);
    const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0);
    const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1);
    int32x2_t vl89 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx89));
    int32x2_t vlAB = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxAB));
    vl89 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx89 >> 32)), vl89, 1);
    vlAB = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxAB >> 32)), vlAB, 1);
    const int32x4_t vl89AB = vcombine_s32(vl89, vlAB);
    const uint64_t vidxCD = vgetq_lane_u64(vidxCDEF, 0);
    const uint64_t vidxEF = vgetq_lane_u64(vidxCDEF, 1);
    int32x2_t vlCD = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxCD));
    int32x2_t vlEF = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxEF));
    vlCD = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxCD >> 32)), vlCD, 1);
    vlEF = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxEF >> 32)), vlEF, 1);
    const int32x4_t vlCDEF = vcombine_s32(vlCD, vlEF);

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vl0123, ven0123));
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vl4567, ven4567));
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    float32x4_t vs89AB = vreinterpretq_f32_s32(vaddq_s32(vl89AB, ven89AB));
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);
    float32x4_t vsCDEF = vreinterpretq_f32_s32(vaddq_s32(vlCDEF, venCDEF));

    float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vminus_ln2);
    float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vminus_ln2);
    float32x4_t vt89AB = vfmaq_f32(vz89AB, vn89AB, vminus_ln2);
    float32x4_t vtCDEF = vfmaq_f32(vzCDEF, vnCDEF, vminus_ln2);

    float32x4_t vp0123 = vfmaq_f32(vc2, vc3, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc2, vc3, vt4567);
    float32x4_t vp89AB = vfmaq_f32(vc2, vc3, vt89AB);
    float32x4_t vpCDEF = vfmaq_f32(vc2, vc3, vtCDEF);

    vp0123 = vmulq_f32(vp0123, vt0123);
    vp4567 = vmulq_f32(vp4567, vt4567);
    vp89AB = vmulq_f32(vp89AB, vt89AB);
    vpCDEF = vmulq_f32(vpCDEF, vtCDEF);

    vt0123 = vmulq_f32(vt0123, vs0123);
    vs0123 = vsubq_f32(vs0123, vone);
    vt4567 = vmulq_f32(vt4567, vs4567);
    vs4567 = vsubq_f32(vs4567, vone);
    vt89AB = vmulq_f32(vt89AB, vs89AB);
    vs89AB = vsubq_f32(vs89AB, vone);
    vtCDEF = vmulq_f32(vtCDEF, vsCDEF);
    vsCDEF = vsubq_f32(vsCDEF, vone);

    vp0123 = vfmaq_f32(vt0123, vp0123, vt0123);
    vp4567 = vfmaq_f32(vt4567, vp4567, vt4567);
    vp89AB = vfmaq_f32(vt89AB, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vtCDEF, vpCDEF, vtCDEF);

    const float32x4_t ve0123 = vmulq_f32(vaddq_f32(vp0123, vs0123), valpha);
    const float32x4_t ve4567 = vmulq_f32(vaddq_f32(vp4567, vs4567), valpha);
    const float32x4_t ve89AB = vmulq_f32(vaddq_f32(vp89AB, vs89AB), valpha);
    const float32x4_t veCDEF = vmulq_f32(vaddq_f32(vpCDEF, vsCDEF), valpha);

    const uint32x4_t vm0123 = vcltq_f32(vx0123, vmovq_n_f32(0.0f));
    vx0123 = vmulq_f32(vx0123, vbeta);
    const uint32x4_t vm4567 = vcltq_f32(vx4567, vmovq_n_f32(0.0f));
    vx4567 = vmulq_f32(vx4567, vbeta);
    const uint32x4_t vm89AB = vcltq_f32(vx89AB, vmovq_n_f32(0.0f));
    vx89AB = vmulq_f32(vx89AB, vbeta);
    const uint32x4_t vmCDEF = vcltq_f32(vxCDEF, vmovq_n_f32(0.0f));
    vxCDEF = vmulq_f32(vxCDEF, vbeta);

    const float32x4_t vy0123 = vbslq_f32(vm0123, ve0123, vx0123);
    const float32x4_t vy4567 = vbslq_f32(vm4567, ve4567, vx4567);
    const float32x4_t vy89AB = vbslq_f32(vm89AB, ve89AB, vx89AB);
    const float32x4_t vyCDEF = vbslq_f32(vmCDEF, veCDEF, vxCDEF);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vz = vmaxq_f32(vmulq_f32(vx, vprescale), vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vlog2e);
    const uint64x2_t vidx = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask), 2));
    const int32x4_t ven = vshlq_n_s32(vreinterpretq_s32_f32(vn), 19);

    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    int32x2_t vl_lo = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    int32x2_t vl_hi = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    vl_lo = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)), vl_lo, 1);
    vl_hi = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)), vl_hi, 1);

    vn = vsubq_f32(vn, vmagic_bias);
    const int32x4_t vl = vcombine_s32(vl_lo, vl_hi);

    float32x4_t vt = vfmaq_f32(vz, vn, vminus_ln2);
    float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vl, ven));

    float32x4_t vp = vfmaq_f32(vc2, vc3, vt);
    vp = vmulq_f32(vp, vt);

    vt = vmulq_f32(vt, vs);
    vs = vsubq_f32(vs, vone);
    vp = vfmaq_f32(vt, vp, vt);
    const float32x4_t ve = vmulq_f32(vaddq_f32(vp, vs), valpha);

    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vx = vmulq_f32(vx, vbeta);
    const float32x4_t vy = vbslq_f32(vm, ve, vx);

    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    float32x4_t vx = vld1q_f32(input);

    const float32x4_t vz = vmaxq_f32(vmulq_f32(vx, vprescale), vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vlog2e);
    const uint64x2_t vidx = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask), 2));
    const int32x4_t ven = vshlq_n_s32(vreinterpretq_s32_f32(vn), 19);

    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    int32x2_t vl_lo = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    int32x2_t vl_hi = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    vl_lo = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)), vl_lo, 1);
    vl_hi = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)), vl_hi, 1);

    vn = vsubq_f32(vn, vmagic_bias);
    const int32x4_t vl = vcombine_s32(vl_lo, vl_hi);

    float32x4_t vt = vfmaq_f32(vz, vn, vminus_ln2);
    float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vl, ven));

    float32x4_t vp = vfmaq_f32(vc2, vc3, vt);
    vp = vmulq_f32(vp, vt);

    vt = vmulq_f32(vt, vs);
    vs = vsubq_f32(vs, vone);
    vp = vfmaq_f32(vt, vp, vt);
    const float32x4_t ve = vmulq_f32(vaddq_f32(vp, vs), valpha);

    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vx = vmulq_f32(vx, vbeta);
    const float32x4_t vy = vbslq_f32(vm, ve, vx);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}

void xnn_f32_velu_ukernel__neonfma_rr1_p6_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vsat_cutoff = vmovq_n_f32(-0x1.154246p+4f);
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.8000FEp23f);
  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  const float32x4_t vc6 = vmovq_n_f32(0x1.6b7338p-10f);
  const float32x4_t vc5 = vmovq_n_f32(0x1.12278Ep-7f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.555716p-5f);
  const float32x4_t vc3 = vmovq_n_f32(0x1.5554B0p-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFFFEp-2f);
  const float32x4_t vone = vmovq_n_f32(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vc6);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);

  const float32x4_t vminus_ln2 = vmovq_n_f32(-0x1.62E430p-1f);
  XNN_FORCE_REALIZATION(vminus_ln2);

  const float32x4_t vprescale = vld1q_dup_f32(&params->scalar.prescale);
  const float32x4_t valpha = vld1q_dup_f32(&params->scalar.alpha);
  const float32x4_t vbeta = vld1q_dup_f32(&params->scalar.beta);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(input); input += 4;
    float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const float32x4_t vz0123 = vmaxq_f32(vmulq_f32(vx0123, vprescale), vsat_cutoff);
    const float32x4_t vz4567 = vmaxq_f32(vmulq_f32(vx4567, vprescale), vsat_cutoff);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vlog2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vlog2e);

    float32x4_t vs0123 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 23));
    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    float32x4_t vs4567 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 23));
    vn4567 = vsubq_f32(vn4567, vmagic_bias);

    float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vminus_ln2);
    float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vminus_ln2);

    float32x4_t vp0123 = vfmaq_f32(vc5, vc6, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc5, vc6, vt4567);

    vp0123 = vfmaq_f32(vc4, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc4, vp4567, vt4567);

    vp0123 = vfmaq_f32(vc3, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc3, vp4567, vt4567);

    vp0123 = vfmaq_f32(vc2, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc2, vp4567, vt4567);

    vp0123 = vmulq_f32(vp0123, vt0123);
    vp4567 = vmulq_f32(vp4567, vt4567);

    vt0123 = vmulq_f32(vt0123, vs0123);
    vs0123 = vsubq_f32(vs0123, vone);
    vt4567 = vmulq_f32(vt4567, vs4567);
    vs4567 = vsubq_f32(vs4567, vone);

    vp0123 = vfmaq_f32(vt0123, vp0123, vt0123);
    vp4567 = vfmaq_f32(vt4567, vp4567, vt4567);

    const float32x4_t ve0123 = vmulq_f32(vaddq_f32(vp0123, vs0123), valpha);
    const float32x4_t ve4567 = vmulq_f32(vaddq_f32(vp4567, vs4567), valpha);

    const uint32x4_t vm0123 = vcltq_f32(vx0123, vmovq_n_f32(0.0f));
    vx0123 = vmulq_f32(vx0123, vbeta);
    const uint32x4_t vm4567 = vcltq_f32(vx4567, vmovq_n_f32(0.0f));
    vx4567 = vmulq_f32(vx4567, vbeta);

    const float32x4_t vy0123 = vbslq_f32(vm0123, ve0123, vx0123);
    const float32x4_t vy4567 = vbslq_f32(vm4567, ve4567, vx4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vz = vmaxq_f32(vmulq_f32(vx, vprescale), vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vlog2e);
    float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vfmaq_f32(vz, vn, vminus_ln2);

    float32x4_t vp = vfmaq_f32(vc5, vc6, vt);
    vp = vfmaq_f32(vc4, vp, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vmulq_f32(vp, vt);

    vt = vmulq_f32(vt, vs);
    vs = vsubq_f32(vs, vone);
    vp = vfmaq_f32(vt, vp, vt);
    const float32x4_t ve = vmulq_f32(vaddq_f32(vp, vs), valpha);

    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vx = vmulq_f32(vx, vbeta);
    const float32x4_t vy = vbslq_f32(vm, ve, vx);

    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    float32x4_t vx = vld1q_f32(input);

    const float32x4_t vz = vmaxq_f32(vmulq_f32(vx, vprescale), vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vlog2e);
    float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vfmaq_f32(vz, vn, vminus_ln2);

    float32x4_t vp = vfmaq_f32(vc5, vc6, vt);
    vp = vfmaq_f32(vc4, vp, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vmulq_f32(vp, vt);

    vt = vmulq_f32(vt, vs);
    vs = vsubq_f32(vs, vone);
    vp = vfmaq_f32(vt, vp, vt);
    const float32x4_t ve = vmulq_f32(vaddq_f32(vp, vs), valpha);

    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vx = vmulq_f32(vx, vbeta);
    const float32x4_t vy = vbslq_f32(vm, ve, vx);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}

void xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float32x4_t vscale0123 = vld1q_f32(w); w += 4;

      float32x4_t vacc0x0123 = vld1q_f32(i0); i0 += 4;
      float32x4_t vacc1x0123 = vld1q_f32(i1); i1 += 4;


      const float32x4_t vbias0123 = vld1q_f32(w); w += 4;

      vacc0x0123 = vfmaq_f32(vbias0123, vscale0123, vacc0x0123);
      vacc1x0123 = vfmaq_f32(vbias0123, vscale0123, vacc1x0123);

      vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
      vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);

      vacc0x0123 = vminq_f32(vacc0x0123, vmax);
      vacc1x0123 = vminq_f32(vacc1x0123, vmax);

      vst1q_f32(o0, vacc0x0123); o0 += 4;
      vst1q_f32(o1, vacc1x0123); o1 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const float32x4_t vscale0123 = vld1q_f32(w);

      float32x4_t vacc0x0123 = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + c);
      float32x4_t vacc1x0123 = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + c);


      const float32x4_t vbias0123 = vld1q_f32(w + 4);

      vacc0x0123 = vfmaq_f32(vbias0123, vscale0123, vacc0x0123);
      vacc1x0123 = vfmaq_f32(vbias0123, vscale0123, vacc1x0123);

      vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
      vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);

      vacc0x0123 = vminq_f32(vacc0x0123, vmax);
      vacc1x0123 = vminq_f32(vacc1x0123, vmax);

      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      float32x2_t vacc1x01 = vget_low_f32(vacc1x0123);
      if (c & (2 * sizeof(float))) {
        vst1_f32(o0, vacc0x01); o0 += 2;
        vst1_f32(o1, vacc1x01); o1 += 2;

        vacc0x01 = vget_high_f32(vacc0x0123);
        vacc1x01 = vget_high_f32(vacc1x0123);
      }
      if (c & (1 * sizeof(float))) {
        vst1_lane_f32(o0, vacc0x01, 0); o0 += 1;
        vst1_lane_f32(o1, vacc1x01, 0); o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64];

void xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p17f);
  const float32x4_t vminus_log2e = vmovq_n_f32(-0x1.715476p0f);
  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0x3F));
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFF0Ap-2f);
  const float32x4_t vone = vmovq_n_f32(1.0f);
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(0x1.5D589Ep+6f);

  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_log2e);
  // XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vc2);
  // XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float32x4_t vln2 = vmovq_n_f32(0x1.62E430p-1f);
  XNN_FORCE_REALIZATION(vln2);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;
    const float32x4_t vx89AB = vld1q_f32(input); input += 4;
    const float32x4_t vxCDEF = vld1q_f32(input); input += 4;

    const float32x4_t vz0123 = vabsq_f32(vx0123);
    const float32x4_t vz4567 = vabsq_f32(vx4567);
    const float32x4_t vz89AB = vabsq_f32(vx89AB);
    const float32x4_t vzCDEF = vabsq_f32(vxCDEF);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vminus_log2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vminus_log2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vz89AB, vminus_log2e);
    float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vzCDEF, vminus_log2e);

    const int32x4_t ve0123 = vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 17);
    const int32x4_t ve4567 = vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 17);
    const int32x4_t ve89AB = vshlq_n_s32(vreinterpretq_s32_f32(vn89AB), 17);
    const int32x4_t veCDEF = vshlq_n_s32(vreinterpretq_s32_f32(vnCDEF), 17);

    // Use bits 0:6 bits of batch, as integer, as an index for table lookup of l := 2**(batch % 64).
    const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask));
    const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask));
    const uint64x2_t vidx89AB = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn89AB), vindex_mask));
    const uint64x2_t vidxCDEF = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vnCDEF), vindex_mask));

    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    float32x2_t vl01 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx01]);
    float32x2_t vl23 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx23]);
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);
    float32x2_t vl45 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx45]);
    float32x2_t vl67 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx67]);
    const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0);
    const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1);
    float32x2_t vl89 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx89]);
    float32x2_t vlAB = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxAB]);
    const uint64_t vidxCD = vgetq_lane_u64(vidxCDEF, 0);
    const uint64_t vidxEF = vgetq_lane_u64(vidxCDEF, 1);
    float32x2_t vlCD = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxCD]);
    float32x2_t vlEF = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxEF]);

    vl01 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx01 >> 32)], vl01, 1);
    vl23 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx23 >> 32)], vl23, 1);
    const float32x4_t vl0123 = vcombine_f32(vl01, vl23);
    vl45 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx45 >> 32)], vl45, 1);
    vl67 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx67 >> 32)], vl67, 1);
    const float32x4_t vl4567 = vcombine_f32(vl45, vl67);
    vl89 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx89 >> 32)], vl89, 1);
    vlAB = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxAB >> 32)], vlAB, 1);
    const float32x4_t vl89AB = vcombine_f32(vl89, vlAB);
    vlCD = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxCD >> 32)], vlCD, 1);
    vlEF = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxEF >> 32)], vlEF, 1);
    const float32x4_t vlCDEF = vcombine_f32(vlCD, vlEF);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl0123), ve0123));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl4567), ve4567));
    const float32x4_t vs89AB = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl89AB), ve89AB));
    const float32x4_t vsCDEF = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vlCDEF), veCDEF));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);

    float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vln2);
    float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vln2);
    float32x4_t vt89AB = vfmaq_f32(vz89AB, vn89AB, vln2);
    float32x4_t vtCDEF = vfmaq_f32(vzCDEF, vnCDEF, vln2);

    float32x4_t vp0123 = vmulq_f32(vt0123, vc2);
    float32x4_t vp4567 = vmulq_f32(vt4567, vc2);
    float32x4_t vp89AB = vmulq_f32(vt89AB, vc2);
    float32x4_t vpCDEF = vmulq_f32(vtCDEF, vc2);

    vp0123 = vfmsq_f32(vt0123, vp0123, vt0123);
    vp4567 = vfmsq_f32(vt4567, vp4567, vt4567);
    vp89AB = vfmsq_f32(vt89AB, vp89AB, vt89AB);
    vpCDEF = vfmsq_f32(vtCDEF, vpCDEF, vtCDEF);

    const float32x4_t vy0123 = vfmsq_f32(vs0123, vs0123, vp0123);
    const float32x4_t vy4567 = vfmsq_f32(vs4567, vs4567, vp4567);
    const float32x4_t vy89AB = vfmsq_f32(vs89AB, vs89AB, vp89AB);
    const float32x4_t vyCDEF = vfmsq_f32(vsCDEF, vsCDEF, vpCDEF);

    const float32x4_t vd0123 = vaddq_f32(vy0123, vone);
    const float32x4_t vd4567 = vaddq_f32(vy4567, vone);
    const float32x4_t vd89AB = vaddq_f32(vy89AB, vone);
    const float32x4_t vdCDEF = vaddq_f32(vyCDEF, vone);

    float32x4_t vr0123 = vrecpeq_f32(vd0123);
    float32x4_t vr4567 = vrecpeq_f32(vd4567);
    float32x4_t vr89AB = vrecpeq_f32(vd89AB);
    float32x4_t vrCDEF = vrecpeq_f32(vdCDEF);

    vr0123 = vmulq_f32(vr0123, vrecpsq_f32(vr0123, vd0123));
    vr4567 = vmulq_f32(vr4567, vrecpsq_f32(vr4567, vd4567));
    vr89AB = vmulq_f32(vr89AB, vrecpsq_f32(vr89AB, vd89AB));
    vrCDEF = vmulq_f32(vrCDEF, vrecpsq_f32(vrCDEF, vdCDEF));

    vr0123 = vmulq_f32(vr0123, vrecpsq_f32(vr0123, vd0123));
    vr4567 = vmulq_f32(vr4567, vrecpsq_f32(vr4567, vd4567));
    vr89AB = vmulq_f32(vr89AB, vrecpsq_f32(vr89AB, vd89AB));
    vrCDEF = vmulq_f32(vrCDEF, vrecpsq_f32(vrCDEF, vdCDEF));

    float32x4_t vf0123 = vmulq_f32(vy0123, vr0123);
    float32x4_t vf4567 = vmulq_f32(vy4567, vr4567);
    float32x4_t vf89AB = vmulq_f32(vy89AB, vr89AB);
    float32x4_t vfCDEF = vmulq_f32(vyCDEF, vrCDEF);

    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcagtq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcagtq_f32(vx4567, vdenorm_cutoff)));
    vf89AB = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf89AB), vcagtq_f32(vx89AB, vdenorm_cutoff)));
    vfCDEF = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vfCDEF), vcagtq_f32(vxCDEF, vdenorm_cutoff)));

    const uint32x4_t vm0123 = vcltq_f32(vx0123, vmovq_n_f32(0.0f));
    const uint32x4_t vm4567 = vcltq_f32(vx4567, vmovq_n_f32(0.0f));
    const uint32x4_t vm89AB = vcltq_f32(vx89AB, vmovq_n_f32(0.0f));
    const uint32x4_t vmCDEF = vcltq_f32(vxCDEF, vmovq_n_f32(0.0f));

    vf0123 = vbslq_f32(vm0123, vf0123, vsubq_f32(vone, vf0123));
    vf4567 = vbslq_f32(vm4567, vf4567, vsubq_f32(vone, vf4567));
    vf89AB = vbslq_f32(vm89AB, vf89AB, vsubq_f32(vone, vf89AB));
    vfCDEF = vbslq_f32(vmCDEF, vfCDEF, vsubq_f32(vone, vfCDEF));

    vst1q_f32(output, vf0123); output += 4;
    vst1q_f32(output, vf4567); output += 4;
    vst1q_f32(output, vf89AB); output += 4;
    vst1q_f32(output, vfCDEF); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vz = vabsq_f32(vx);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);
    const int32x4_t ve = vshlq_n_s32(vreinterpretq_s32_f32(vn), 17);

    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);

    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));
    vn = vsubq_f32(vn, vmagic_bias);
    float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vfmsq_f32(vt, vp, vt);

    const float32x4_t vy = vfmsq_f32(vs, vs, vp);
    const float32x4_t vd = vaddq_f32(vy, vone);

    float32x4_t vr = vrecpeq_f32(vd);
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    float32x4_t vf = vmulq_f32(vy, vr);
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    vst1q_f32(output, vf); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    const float32x4_t vz = vabsq_f32(vx);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);
    const int32x4_t ve = vshlq_n_s32(vreinterpretq_s32_f32(vn), 17);

    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);

    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));
    vn = vsubq_f32(vn, vmagic_bias);
    float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vfmsq_f32(vt, vp, vt);

    const float32x4_t vy = vfmsq_f32(vs, vs, vp);
    const float32x4_t vd = vaddq_f32(vy, vone);

    float32x4_t vr = vrecpeq_f32(vd);
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    float32x4_t vf = vmulq_f32(vy, vr);
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    float32x2_t vf_lo = vget_low_f32(vf);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vf_lo); output += 2;
      vf_lo = vget_high_f32(vf);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vf_lo, 0);
    }
  }
}

void xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const float32x4_t vsat_cutoff = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.sat_cutoff);
  const float32x4_t vminus_log2e = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.minus_log2e);

  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.magic_bias);

  const float32x4_t vln2 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.ln2);

  const float32x4_t vc6 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c6);
  const float32x4_t vc5 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c5);
  const float32x4_t vc4 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c4);
  const float32x4_t vc3 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c3);
  const float32x4_t vc2 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c2);

  const float32x4_t vone = vmovq_n_f32(1.0f);
  const float32x4_t vtwo = vmovq_n_f32(2.0f);

  const uint32x4_t vsign_mask = vmovq_n_u32(UINT32_C(0x80000000));

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;

    float32x4_t vz0123 = vabsq_f32(vx0123);
    float32x4_t vz4567 = vabsq_f32(vx4567);
    vz0123 = vminq_f32(vz0123, vsat_cutoff);
    vz4567 = vminq_f32(vz4567, vsat_cutoff);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vminus_log2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vminus_log2e);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 23));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 23));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);

    const float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vln2);
    const float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vln2);

    float32x4_t vp0123 = vfmaq_f32(vc5, vc6, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc5, vc6, vt4567);
    vp0123 = vfmaq_f32(vc4, vp0123, vt0123);
    vp0123 = vfmaq_f32(vc3, vp0123, vt0123);
    vp0123 = vfmaq_f32(vc2, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc4, vp4567, vt4567);
    vp4567 = vfmaq_f32(vc3, vp4567, vt4567);
    vp4567 = vfmaq_f32(vc2, vp4567, vt4567);
    vp0123 = vfmsq_f32(vtwo, vp0123, vt0123);
    vp4567 = vfmsq_f32(vtwo, vp4567, vt4567);

    const float32x4_t vts0123 = vmulq_f32(vt0123, vs0123);
    const float32x4_t vsmo0123 = vsubq_f32(vs0123, vone);
    const float32x4_t vts4567 = vmulq_f32(vt4567, vs4567);
    const float32x4_t vsmo4567 = vsubq_f32(vs4567, vone);
    const float32x4_t vemo0123 = vfmsq_f32(vsmo0123, vp0123, vts0123);
    const float32x4_t vemo4567 = vfmsq_f32(vsmo4567, vp4567, vts4567);

    const float32x4_t vepo0123 = vaddq_f32(vemo0123, vtwo);
    const float32x4_t vepo4567 = vaddq_f32(vemo4567, vtwo);

    float32x4_t vrepo0123 = vrecpeq_f32(vepo0123);
    float32x4_t vrepo4567 = vrecpeq_f32(vepo4567);
    float32x4_t verepo0123 = vfmsq_f32(vone, vrepo0123, vepo0123);
    float32x4_t verepo4567 = vfmsq_f32(vone, vrepo4567, vepo4567);
    vrepo0123 = vfmaq_f32(vrepo0123, vrepo0123, verepo0123);
    vrepo4567 = vfmaq_f32(vrepo4567, vrepo4567, verepo4567);
    verepo0123 = vfmsq_f32(vone, vrepo0123, vepo0123);
    verepo4567 = vfmsq_f32(vone, vrepo4567, vepo4567);
    vrepo0123 = vfmaq_f32(vrepo0123, vrepo0123, verepo0123);
    vrepo4567 = vfmaq_f32(vrepo4567, vrepo4567, verepo4567);

    float32x4_t vy0123 = vmulq_f32(vemo0123, vrepo0123);
    float32x4_t vy4567 = vmulq_f32(vemo4567, vrepo4567);

    vy0123 = vbslq_f32(vsign_mask, vx0123, vy0123);
    vy4567 = vbslq_f32(vsign_mask, vx4567, vy4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    float32x4_t vz = vabsq_f32(vx);
    vz = vminq_f32(vz, vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    vn = vsubq_f32(vn, vmagic_bias);

    const float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vfmaq_f32(vc5, vc6, vt);
    vp = vfmaq_f32(vc4, vp, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmsq_f32(vtwo, vp, vt);

    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vfmsq_f32(vsmo, vp, vts);

    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    float32x4_t vrepo = vrecpeq_f32(vepo);
    float32x4_t verepo = vfmsq_f32(vone, vrepo, vepo);
    vrepo = vfmaq_f32(vrepo, vrepo, verepo);
    verepo = vfmsq_f32(vone, vrepo, vepo);
    vrepo = vfmaq_f32(vrepo, vrepo, verepo);

    float32x4_t vy = vmulq_f32(vemo, vrepo);

    vy = vbslq_f32(vsign_mask, vx, vy);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    float32x4_t vz = vabsq_f32(vx);
    vz = vminq_f32(vz, vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    vn = vsubq_f32(vn, vmagic_bias);

    const float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vfmaq_f32(vc5, vc6, vt);
    vp = vfmaq_f32(vc4, vp, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmsq_f32(vtwo, vp, vt);

    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vfmsq_f32(vsmo, vp, vts);

    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    float32x4_t vrepo = vrecpeq_f32(vepo);
    float32x4_t verepo = vfmsq_f32(vone, vrepo, vepo);
    vrepo = vfmaq_f32(vrepo, vrepo, verepo);
    verepo = vfmsq_f32(vone, vrepo, vepo);
    vrepo = vfmaq_f32(vrepo, vrepo, verepo);

    float32x4_t vy = vmulq_f32(vemo, vrepo);

    vy = vbslq_f32(vsign_mask, vx, vy);

    float32x2_t vy_low = vget_low_f32(vy);

    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_low); output += 2;
      vy_low = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_low, 0);
    }
  }
}
