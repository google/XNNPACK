// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/packw.h>
#include <xnnpack/prefetch.h>
#include <xnnpack/prelu.h>
#include <xnnpack/reduce.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vunary.h>


void xnn_f32_dwconv_minmax_ukernel_25p16c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
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
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_loadu_ps(i3);
      i3 += 16;

      const __m512 vk3x0123456789ABCDEF = _mm512_load_ps(w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi4x0123456789ABCDEF = _mm512_loadu_ps(i4);
      i4 += 16;

      const __m512 vk4x0123456789ABCDEF = _mm512_load_ps(w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi5x0123456789ABCDEF = _mm512_loadu_ps(i5);
      i5 += 16;

      const __m512 vk5x0123456789ABCDEF = _mm512_load_ps(w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi6x0123456789ABCDEF = _mm512_loadu_ps(i6);
      i6 += 16;

      const __m512 vk6x0123456789ABCDEF = _mm512_load_ps(w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi7x0123456789ABCDEF = _mm512_loadu_ps(i7);
      i7 += 16;

      const __m512 vk7x0123456789ABCDEF = _mm512_load_ps(w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi8x0123456789ABCDEF = _mm512_loadu_ps(i8);
      i8 += 16;

      const __m512 vk8x0123456789ABCDEF = _mm512_load_ps(w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi9x0123456789ABCDEF = _mm512_loadu_ps(i9);
      i9 += 16;

      const __m512 vk9x0123456789ABCDEF = _mm512_load_ps(w + 160);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi10x0123456789ABCDEF = _mm512_loadu_ps(i10);
      i10 += 16;

      const __m512 vk10x0123456789ABCDEF = _mm512_load_ps(w + 176);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi11x0123456789ABCDEF = _mm512_loadu_ps(i11);
      i11 += 16;

      const __m512 vk11x0123456789ABCDEF = _mm512_load_ps(w + 192);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi12x0123456789ABCDEF = _mm512_loadu_ps(i12);
      i12 += 16;

      const __m512 vk12x0123456789ABCDEF = _mm512_load_ps(w + 208);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi13x0123456789ABCDEF = _mm512_loadu_ps(i13);
      i13 += 16;

      const __m512 vk13x0123456789ABCDEF = _mm512_load_ps(w + 224);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi14x0123456789ABCDEF = _mm512_loadu_ps(i14);
      i14 += 16;

      const __m512 vk14x0123456789ABCDEF = _mm512_load_ps(w + 240);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi15x0123456789ABCDEF = _mm512_loadu_ps(i15);
      i15 += 16;

      const __m512 vk15x0123456789ABCDEF = _mm512_load_ps(w + 256);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi16x0123456789ABCDEF = _mm512_loadu_ps(i16);
      i16 += 16;

      const __m512 vk16x0123456789ABCDEF = _mm512_load_ps(w + 272);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi17x0123456789ABCDEF = _mm512_loadu_ps(i17);
      i17 += 16;

      const __m512 vk17x0123456789ABCDEF = _mm512_load_ps(w + 288);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi18x0123456789ABCDEF = _mm512_loadu_ps(i18);
      i18 += 16;

      const __m512 vk18x0123456789ABCDEF = _mm512_load_ps(w + 304);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi19x0123456789ABCDEF = _mm512_loadu_ps(i19);
      i19 += 16;

      const __m512 vk19x0123456789ABCDEF = _mm512_load_ps(w + 320);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi20x0123456789ABCDEF = _mm512_loadu_ps(i20);
      i20 += 16;

      const __m512 vk20x0123456789ABCDEF = _mm512_load_ps(w + 336);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi21x0123456789ABCDEF = _mm512_loadu_ps(i21);
      i21 += 16;

      const __m512 vk21x0123456789ABCDEF = _mm512_load_ps(w + 352);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi22x0123456789ABCDEF = _mm512_loadu_ps(i22);
      i22 += 16;

      const __m512 vk22x0123456789ABCDEF = _mm512_load_ps(w + 368);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi23x0123456789ABCDEF = _mm512_loadu_ps(i23);
      i23 += 16;

      const __m512 vk23x0123456789ABCDEF = _mm512_load_ps(w + 384);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi24x0123456789ABCDEF = _mm512_loadu_ps(i24);
      i24 += 16;

      const __m512 vk24x0123456789ABCDEF = _mm512_load_ps(w + 400);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 416;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));

      __m512 vacc0123456789ABCDEFp0 = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i0);
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i3);
      const __m512 vk3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi4x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i4);
      const __m512 vk4x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi5x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i5);
      const __m512 vk5x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi6x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i6);
      const __m512 vk6x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi7x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i7);
      const __m512 vk7x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi8x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i8);
      const __m512 vk8x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi9x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i9);
      const __m512 vk9x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 160);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi10x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i10);
      const __m512 vk10x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 176);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi11x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i11);
      const __m512 vk11x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 192);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi12x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i12);
      const __m512 vk12x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 208);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi13x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i13);
      const __m512 vk13x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 224);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi14x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i14);
      const __m512 vk14x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 240);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi15x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i15);
      const __m512 vk15x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 256);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi16x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i16);
      const __m512 vk16x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 272);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi17x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i17);
      const __m512 vk17x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 288);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi18x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i18);
      const __m512 vk18x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 304);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi19x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i19);
      const __m512 vk19x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 320);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi20x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i20);
      const __m512 vk20x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 336);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi21x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i21);
      const __m512 vk21x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 352);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi22x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i22);
      const __m512 vk22x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 368);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi23x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i23);
      const __m512 vk23x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 384);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi24x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i24);
      const __m512 vk24x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 400);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p16c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
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
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 64;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));

      __m512 vacc0123456789ABCDEFp0 = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i0);
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p16c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
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
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_loadu_ps(i3);
      i3 += 16;

      const __m512 vk3x0123456789ABCDEF = _mm512_load_ps(w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 80;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));

      __m512 vacc0123456789ABCDEFp0 = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i0);
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i3);
      const __m512 vk3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
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
      size_t c = channels;
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(w);
        __m512 vacc1p0 = _mm512_load_ps(w + 16);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        const __m512 vk0x0 = _mm512_load_ps(w + 32);
        const __m512 vk0x1 = _mm512_load_ps(w + 48);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        const __m512 vk1x0 = _mm512_load_ps(w + 64);
        const __m512 vk1x1 = _mm512_load_ps(w + 80);
        vacc0p0 = _mm512_fmadd_ps(vi1x0, vk1x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi1x1, vk1x1, vacc1p0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        const __m512 vi2x1 = _mm512_loadu_ps(i2 + 16);
        i2 += 32;

        const __m512 vk2x0 = _mm512_load_ps(w + 96);
        const __m512 vk2x1 = _mm512_load_ps(w + 112);
        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi2x1, vk2x1, vacc1p0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        const __m512 vi3x1 = _mm512_loadu_ps(i3 + 16);
        i3 += 32;

        const __m512 vk3x0 = _mm512_load_ps(w + 128);
        const __m512 vk3x1 = _mm512_load_ps(w + 144);
        vacc0p0 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi3x1, vk3x1, vacc1p0);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        const __m512 vi4x1 = _mm512_loadu_ps(i4 + 16);
        i4 += 32;

        const __m512 vk4x0 = _mm512_load_ps(w + 160);
        const __m512 vk4x1 = _mm512_load_ps(w + 176);
        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi4x1, vk4x1, vacc1p0);

        w += 192;


        _mm512_store_ps(b, vacc0p0);
        _mm512_store_ps(b + 16, vacc1p0);
        b += 32;
      }

      for (; c >= 16; c -= 16) {
        __m512 vaccp0 = _mm512_load_ps(w);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        const __m512 vk0x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi1x0, vk1x0, vaccp0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        const __m512 vk2x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        const __m512 vk3x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp0);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        const __m512 vk4x0 = _mm512_load_ps(w + 80);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 96;


        _mm512_store_ps(b, vaccp0);
        b += 16;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));
        __m512 vaccp0 = _mm512_load_ps(w);


        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);

        const __m512 vk0x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi1x0, vk1x0, vaccp0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);

        const __m512 vk2x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);

        const __m512 vk3x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp0);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);

        const __m512 vk4x0 = _mm512_load_ps(w + 80);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 96;


        _mm512_store_ps(b, vaccp0);
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

      size_t c = channels;
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(b);
        __m512 vacc1p0 = _mm512_load_ps(b + 16);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        const __m512 vk0x0 = _mm512_load_ps(w);
        const __m512 vk0x1 = _mm512_load_ps(w + 16);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        const __m512 vk1x1 = _mm512_load_ps(w + 48);
        vacc0p0 = _mm512_fmadd_ps(vi1x0, vk1x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi1x1, vk1x1, vacc1p0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        const __m512 vi2x1 = _mm512_loadu_ps(i2 + 16);
        i2 += 32;

        const __m512 vk2x0 = _mm512_load_ps(w + 64);
        const __m512 vk2x1 = _mm512_load_ps(w + 80);
        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi2x1, vk2x1, vacc1p0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        const __m512 vi3x1 = _mm512_loadu_ps(i3 + 16);
        i3 += 32;

        const __m512 vk3x0 = _mm512_load_ps(w + 96);
        const __m512 vk3x1 = _mm512_load_ps(w + 112);
        vacc0p0 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi3x1, vk3x1, vacc1p0);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        const __m512 vi4x1 = _mm512_loadu_ps(i4 + 16);
        i4 += 32;

        const __m512 vk4x0 = _mm512_load_ps(w + 128);
        const __m512 vk4x1 = _mm512_load_ps(w + 144);
        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi4x1, vk4x1, vacc1p0);

        w += 160;


        _mm512_store_ps(b, vacc0p0);
        _mm512_store_ps(b + 16, vacc1p0);
        b += 32;
      }

      for (; c >= 16; c -= 16) {
        __m512 vaccp0 = _mm512_load_ps(b);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        const __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        const __m512 vk1x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi1x0, vk1x0, vaccp0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        const __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        const __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp0);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        const __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;


        _mm512_store_ps(b, vaccp0);
        b += 16;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));
        __m512 vaccp0 = _mm512_load_ps(b);


        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);

        const __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);

        const __m512 vk1x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi1x0, vk1x0, vaccp0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);

        const __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);

        const __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp0);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);

        const __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;


        _mm512_store_ps(b, vaccp0);
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
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(b);
        __m512 vacc1p0 = _mm512_load_ps(b + 16);
        b += 32;


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        __m512 vk0x0 = _mm512_load_ps(w);
        __m512 vk0x1 = _mm512_load_ps(w + 16);

        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        __m512 vk1x0 = _mm512_load_ps(w + 32);
        __m512 vk1x1 = _mm512_load_ps(w + 48);

        vacc0p0 = _mm512_fmadd_ps(vi1x0, vk1x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi1x1, vk1x1, vacc1p0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        const __m512 vi2x1 = _mm512_loadu_ps(i2 + 16);
        i2 += 32;

        __m512 vk2x0 = _mm512_load_ps(w + 64);
        __m512 vk2x1 = _mm512_load_ps(w + 80);

        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi2x1, vk2x1, vacc1p0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        const __m512 vi3x1 = _mm512_loadu_ps(i3 + 16);
        i3 += 32;

        __m512 vk3x0 = _mm512_load_ps(w + 96);
        __m512 vk3x1 = _mm512_load_ps(w + 112);

        vacc0p0 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi3x1, vk3x1, vacc1p0);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        const __m512 vi4x1 = _mm512_loadu_ps(i4 + 16);
        i4 += 32;

        __m512 vk4x0 = _mm512_load_ps(w + 128);
        __m512 vk4x1 = _mm512_load_ps(w + 144);

        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi4x1, vk4x1, vacc1p0);

        w += 160;


        __m512 vacc0 = _mm512_max_ps(vmin, vacc0p0);
        __m512 vacc1 = _mm512_max_ps(vmin, vacc1p0);

        vacc0 = _mm512_min_ps(vmax, vacc0);
        vacc1 = _mm512_min_ps(vmax, vacc1);

        _mm512_storeu_ps(output, vacc0);
        _mm512_storeu_ps(output + 16, vacc1);
        output += 32;
      }


      for (; c >= 16; c -= 16) {
        __m512 vaccp0 = _mm512_load_ps(b);
        b += 16;


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        __m512 vk0x0 = _mm512_load_ps(w);

        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        __m512 vk1x0 = _mm512_load_ps(w + 16);

        vaccp0 = _mm512_fmadd_ps(vi1x0, vk1x0, vaccp0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        __m512 vk2x0 = _mm512_load_ps(w + 32);

        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        __m512 vk3x0 = _mm512_load_ps(w + 48);

        vaccp0 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp0);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        __m512 vk4x0 = _mm512_load_ps(w + 64);

        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;



        __m512 vacc = _mm512_max_ps(vmin, vaccp0);

        vacc = _mm512_min_ps(vmax, vacc);

        _mm512_storeu_ps(output, vacc);
        output += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        __m512 vaccp0 = _mm512_load_ps(b);
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));

        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);
        __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);
        __m512 vk1x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi1x0, vk1x0, vaccp0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);
        __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);
        __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp0);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);
        __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);


        __m512 vacc = _mm512_max_ps(vmin, vaccp0);
        vacc = _mm512_min_ps(vmax, vacc);

        _mm512_mask_storeu_ps(output, vmask, vacc);
        output += c;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p16c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
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
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_loadu_ps(i3);
      i3 += 16;

      const __m512 vk3x0123456789ABCDEF = _mm512_load_ps(w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi4x0123456789ABCDEF = _mm512_loadu_ps(i4);
      i4 += 16;

      const __m512 vk4x0123456789ABCDEF = _mm512_load_ps(w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi5x0123456789ABCDEF = _mm512_loadu_ps(i5);
      i5 += 16;

      const __m512 vk5x0123456789ABCDEF = _mm512_load_ps(w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi6x0123456789ABCDEF = _mm512_loadu_ps(i6);
      i6 += 16;

      const __m512 vk6x0123456789ABCDEF = _mm512_load_ps(w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi7x0123456789ABCDEF = _mm512_loadu_ps(i7);
      i7 += 16;

      const __m512 vk7x0123456789ABCDEF = _mm512_load_ps(w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi8x0123456789ABCDEF = _mm512_loadu_ps(i8);
      i8 += 16;

      const __m512 vk8x0123456789ABCDEF = _mm512_load_ps(w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 160;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - UINT32_C(1)));

      __m512 vacc0123456789ABCDEFp0 = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i0);
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i3);
      const __m512 vk3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi4x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i4);
      const __m512 vk4x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi5x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i5);
      const __m512 vk5x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi6x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i6);
      const __m512 vk6x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi7x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i7);
      const __m512 vk7x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi8x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i8);
      const __m512 vk8x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast(
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
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    w += 16;

    size_t k = kc;
    do {
      const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
      w += 16;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);

      a0 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      assert(nc != 0);
      assert(nc < 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1));
      _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast(
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
  assert(mr <= 7);
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
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    __m512 vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc5x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc6x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    w += 16;

    size_t k = kc;
    do {
      const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
      w += 16;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
      const __m512 va1 = _mm512_set1_ps(*a1);
      vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1, vb0123456789ABCDEF, vacc1x0123456789ABCDEF);
      const __m512 va2 = _mm512_set1_ps(*a2);
      vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2, vb0123456789ABCDEF, vacc2x0123456789ABCDEF);
      const __m512 va3 = _mm512_set1_ps(*a3);
      vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3, vb0123456789ABCDEF, vacc3x0123456789ABCDEF);
      const __m512 va4 = _mm512_set1_ps(*a4);
      vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4, vb0123456789ABCDEF, vacc4x0123456789ABCDEF);
      const __m512 va5 = _mm512_set1_ps(*a5);
      vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5, vb0123456789ABCDEF, vacc5x0123456789ABCDEF);
      const __m512 va6 = _mm512_set1_ps(*a6);
      vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6, vb0123456789ABCDEF, vacc6x0123456789ABCDEF);

      a0 += 1;
      a1 += 1;
      a2 += 1;
      a3 += 1;
      a4 += 1;
      a5 += 1;
      a6 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vmin, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vmin, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vmin, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vmin, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vmin, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vmin, vacc6x0123456789ABCDEF);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vmax, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vmax, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vmax, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vmax, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vmax, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vmax, vacc6x0123456789ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c6, vacc6x0123456789ABCDEF);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);
      a6 = (const float*) ((uintptr_t) a6 - kc);

      nc -= 16;
    } else {
      assert(nc != 0);
      assert(nc < 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1));
      _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vacc1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vacc2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3, vmask, vacc3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4, vmask, vacc4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c5, vmask, vacc5x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c6, vmask, vacc6x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
        w += 16;

        const __m512 va0 = _mm512_set1_ps(*a0);
        vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);

        a0 += 1;

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1));

        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (7 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    __m512 vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc5x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc6x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    w += 16;

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
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      a += 7;

      size_t k = kc;
      do {
        const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
        w += 16;

        const __m512 va0 = _mm512_set1_ps(*a0);
        vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
        const __m512 va1 = _mm512_set1_ps(*a1);
        vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1, vb0123456789ABCDEF, vacc1x0123456789ABCDEF);
        const __m512 va2 = _mm512_set1_ps(*a2);
        vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2, vb0123456789ABCDEF, vacc2x0123456789ABCDEF);
        const __m512 va3 = _mm512_set1_ps(*a3);
        vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3, vb0123456789ABCDEF, vacc3x0123456789ABCDEF);
        const __m512 va4 = _mm512_set1_ps(*a4);
        vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4, vb0123456789ABCDEF, vacc4x0123456789ABCDEF);
        const __m512 va5 = _mm512_set1_ps(*a5);
        vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5, vb0123456789ABCDEF, vacc5x0123456789ABCDEF);
        const __m512 va6 = _mm512_set1_ps(*a6);
        vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6, vb0123456789ABCDEF, vacc6x0123456789ABCDEF);

        a0 += 1;
        a1 += 1;
        a2 += 1;
        a3 += 1;
        a4 += 1;
        a5 += 1;
        a6 += 1;

        k -= sizeof(float);
      } while (k != 0);
      p -= 7 * sizeof(void*);
    } while (p != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vmin, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vmin, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vmin, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vmin, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vmin, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vmin, vacc6x0123456789ABCDEF);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vmax, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vmax, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vmax, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vmax, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vmax, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vmax, vacc6x0123456789ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c6, vacc6x0123456789ABCDEF);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1));

        _mm512_mask_storeu_ps(c6, vmask, vacc6x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c5, vmask, vacc5x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c4, vmask, vacc4x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c3, vmask, vacc3x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c2, vmask, vacc2x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c1, vmask, vacc1x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_prelu_ukernel__avx512f_2x16(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride)
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

  const __m512 vzero = _mm512_setzero_ps();
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(float); c -= 16 * sizeof(float)) {
      const __m512 vw0123456789ABCDEF = _mm512_load_ps(w);
      w += 16;

      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;
      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __mmask16 vsign0x0123456789ABCDEF = _mm512_cmp_ps_mask(vi0x0123456789ABCDEF, vzero, _CMP_LT_OQ);
      const __m512 vacc0x0123456789ABCDEF = _mm512_mask_mul_ps(vi0x0123456789ABCDEF, vsign0x0123456789ABCDEF, vi0x0123456789ABCDEF, vw0123456789ABCDEF);
      const __mmask16 vsign1x0123456789ABCDEF = _mm512_cmp_ps_mask(vi1x0123456789ABCDEF, vzero, _CMP_LT_OQ);
      const __m512 vacc1x0123456789ABCDEF = _mm512_mask_mul_ps(vi1x0123456789ABCDEF, vsign1x0123456789ABCDEF, vi1x0123456789ABCDEF, vw0123456789ABCDEF);

      _mm512_storeu_ps(o0, vacc0x0123456789ABCDEF);
      o0 += 16;
      _mm512_storeu_ps(o1, vacc1x0123456789ABCDEF);
      o1 += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1 * sizeof(float));
      assert(c <= 15 * sizeof(float));
      // Prepare mask for valid 32-bit elements (depends on c).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) (UINT32_C(1) << (c >> XNN_LOG2_SIZEOF_FLOAT)) - UINT32_C(1));

      const __m512 vw = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0 = _mm512_maskz_loadu_ps(vmask, i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const __m512 vi1 = _mm512_maskz_loadu_ps(vmask, i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const __mmask16 vsign0 = _mm512_cmp_ps_mask(vi0, vzero, _CMP_LT_OQ);
      const __m512 vacc0 = _mm512_mask_mul_ps(vi0, vsign0, vi0, vw);
      const __mmask16 vsign1 = _mm512_cmp_ps_mask(vi1, vzero, _CMP_LT_OQ);
      const __m512 vacc1 = _mm512_mask_mul_ps(vi1, vsign1, vi1, vw);

      _mm512_mask_storeu_ps(o0, vmask, vacc0);
      o0 = (float*) ((uintptr_t) o0 + c);
      _mm512_mask_storeu_ps(o1, vmask, vacc1);
      o1 = (float*) ((uintptr_t) o1 + c);
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_rmax_ukernel__avx512f_u64_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m512 vmax0 = _mm512_set1_ps(*input);
  __m512 vmax1 = vmax0;
  __m512 vmax2 = vmax0;
  __m512 vmax3 = vmax0;
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vt0 = _mm512_loadu_ps(input);
    const __m512 vt1 = _mm512_loadu_ps(input + 16);
    const __m512 vt2 = _mm512_loadu_ps(input + 32);
    const __m512 vt3 = _mm512_loadu_ps(input + 48);
    input += 64;

    vmax0 = _mm512_max_ps(vmax0, vt0);
    vmax1 = _mm512_max_ps(vmax1, vt1);
    vmax2 = _mm512_max_ps(vmax2, vt2);
    vmax3 = _mm512_max_ps(vmax3, vt3);
  }
  vmax0 = _mm512_max_ps(vmax0, vmax1);
  vmax2 = _mm512_max_ps(vmax2, vmax3);
  vmax0 = _mm512_max_ps(vmax0, vmax2);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vt = _mm512_loadu_ps(input);
    input += 16;

    vmax0 = _mm512_max_ps(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vt = _mm512_maskz_loadu_ps(vmask, input);

    vmax0 = _mm512_mask_max_ps(vmax0, vmask, vmax0, vt);
  }
  __m256 vmax256 = _mm256_max_ps(_mm512_castps512_ps256(vmax0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vmax0), 1)));
  __m128 vmax = _mm_max_ps(_mm256_castps256_ps128(vmax256), _mm256_extractf128_ps(vmax256, 1));
  vmax = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vmax = _mm_max_ss(vmax, _mm_movehdup_ps(vmax));
  _mm_store_ss(output, vmax);
}

void xnn_f32_rminmax_ukernel__avx512f_u64_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m512 vmin0 = _mm512_set1_ps(*input);
  __m512 vmax0 = vmin0;
  __m512 vmin1 = vmin0;
  __m512 vmax1 = vmax0;
  __m512 vmin2 = vmin0;
  __m512 vmax2 = vmax0;
  __m512 vmin3 = vmin0;
  __m512 vmax3 = vmax0;
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vt0 = _mm512_loadu_ps(input);
    const __m512 vt1 = _mm512_loadu_ps(input + 16);
    const __m512 vt2 = _mm512_loadu_ps(input + 32);
    const __m512 vt3 = _mm512_loadu_ps(input + 48);
    input += 64;

    vmin0 = _mm512_min_ps(vmin0, vt0);
    vmax0 = _mm512_max_ps(vmax0, vt0);
    vmin1 = _mm512_min_ps(vmin1, vt1);
    vmax1 = _mm512_max_ps(vmax1, vt1);
    vmin2 = _mm512_min_ps(vmin2, vt2);
    vmax2 = _mm512_max_ps(vmax2, vt2);
    vmin3 = _mm512_min_ps(vmin3, vt3);
    vmax3 = _mm512_max_ps(vmax3, vt3);
  }
  vmin0 = _mm512_min_ps(vmin0, vmin1);
  vmax0 = _mm512_max_ps(vmax0, vmax1);
  vmin2 = _mm512_min_ps(vmin2, vmin3);
  vmax2 = _mm512_max_ps(vmax2, vmax3);
  vmin0 = _mm512_min_ps(vmin0, vmin2);
  vmax0 = _mm512_max_ps(vmax0, vmax2);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vt = _mm512_loadu_ps(input);
    input += 16;

    vmin0 = _mm512_min_ps(vmin0, vt);
    vmax0 = _mm512_max_ps(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vt = _mm512_maskz_loadu_ps(vmask, input);

    vmin0 = _mm512_mask_min_ps(vmin0, vmask, vmin0, vt);
    vmax0 = _mm512_mask_max_ps(vmax0, vmask, vmax0, vt);
  }
  __m256 vmin256 = _mm256_min_ps(_mm512_castps512_ps256(vmin0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vmin0), 1)));
  __m256 vmax256 = _mm256_max_ps(_mm512_castps512_ps256(vmax0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vmax0), 1)));
  __m128 vmin = _mm_min_ps(_mm256_castps256_ps128(vmin256), _mm256_extractf128_ps(vmin256, 1));
  __m128 vmax = _mm_max_ps(_mm256_castps256_ps128(vmax256), _mm256_extractf128_ps(vmax256, 1));
  vmin = _mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin));
  vmax = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vmin = _mm_min_ss(vmin, _mm_movehdup_ps(vmin));
  vmax = _mm_max_ss(vmax, _mm_movehdup_ps(vmax));
  _mm_store_ss(output, vmin);
  _mm_store_ss(output + 1, vmax);
}

void xnn_f32_rsum_ukernel__avx512f_u64_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  __m512 vacc2 = _mm512_setzero_ps();
  __m512 vacc3 = _mm512_setzero_ps();
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vt0 = _mm512_loadu_ps(input);
    const __m512 vt1 = _mm512_loadu_ps(input + 16);
    const __m512 vt2 = _mm512_loadu_ps(input + 32);
    const __m512 vt3 = _mm512_loadu_ps(input + 48);
    input += 64;

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
    vacc2 = _mm512_add_ps(vacc2, vt2);
    vacc3 = _mm512_add_ps(vacc3, vt3);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  vacc2 = _mm512_add_ps(vacc2, vacc3);
  vacc0 = _mm512_add_ps(vacc0, vacc2);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vt = _mm512_loadu_ps(input);
    input += 16;

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vt = _mm512_maskz_loadu_ps(vmask, input);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }

  __m256 vacc256 = _mm256_add_ps(_mm512_castps512_ps256(vacc0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256), _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));
  _mm_store_ss(output, vacc);
}

void xnn_f32_vadd_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_add_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_add_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_add_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_add_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vaddc_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_add_ps(vacc0, vb);
    vacc1 = _mm512_add_ps(vacc1, vb);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_add_ps(vacc, vb);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_add_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vdiv_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_div_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_div_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_div_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_div_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vdivc_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_div_ps(vacc0, vb);
    vacc1 = _mm512_div_ps(vacc1, vb);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_div_ps(vacc, vb);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_div_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmax_ukernel__avx512f_u32(
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


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_max_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_max_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;



    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_max_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;


    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_max_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmaxc_ukernel__avx512f_u32(
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

  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_max_ps(vacc0, vb);
    vacc1 = _mm512_max_ps(vacc1, vb);



    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_max_ps(vacc, vb);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_max_ps(vmask, vacc, vb);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmin_ukernel__avx512f_u32(
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


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_min_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_min_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;



    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_min_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;


    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_min_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vminc_ukernel__avx512f_u32(
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

  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_min_ps(vacc0, vb);
    vacc1 = _mm512_min_ps(vacc1, vb);



    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_min_ps(vacc, vb);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_min_ps(vmask, vacc, vb);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmul_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_mul_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_mul_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_mul_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_mul_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmulc_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_mul_ps(vacc0, vb);
    vacc1 = _mm512_mul_ps(vacc1, vb);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_mul_ps(vacc, vb);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_mul_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vrdivc_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_div_ps(vb, vacc0);
    vacc1 = _mm512_div_ps(vb, vacc1);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_div_ps(vb, vacc);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_div_ps(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vrsubc_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vb, vacc0);
    vacc1 = _mm512_sub_ps(vb, vacc1);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vb, vacc);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vsqrdiff_ukernel__avx512f_u32(
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


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_sub_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;

    vacc0 = _mm512_mul_ps(vacc0, vacc0);
    vacc1 = _mm512_mul_ps(vacc1, vacc1);


    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_mul_ps(vacc, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_mul_ps(vmask, vacc, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vsqrdiffc_ukernel__avx512f_u32(
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

  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vacc0, vb);
    vacc1 = _mm512_sub_ps(vacc1, vb);

    vacc0 = _mm512_mul_ps(vacc0, vacc0);
    vacc1 = _mm512_mul_ps(vacc1, vacc1);


    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vacc, vb);
    vacc = _mm512_mul_ps(vacc, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_mul_ps(vmask, vacc, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vsub_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_sub_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vsubc_minmax_ukernel__avx512f_u32(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vacc0, vb);
    vacc1 = _mm512_sub_ps(vacc1, vb);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vacc, vb);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vclamp_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEF);

    vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

    _mm512_storeu_ps(output, vacc0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input);
    vacc = _mm512_max_ps(vmin, vacc);
    vacc = _mm512_min_ps(vmax, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_velu_ukernel__avx512f_rr1_p6_u128(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vprescale = _mm512_set1_ps(params->avx512_rr1_p6.prescale);
  const __m512 valpha = _mm512_set1_ps(params->avx512_rr1_p6.alpha);
  const __m512 vbeta = _mm512_set1_ps(params->avx512_rr1_p6.beta);
  const __m512 vsat_cutoff = _mm512_set1_ps(params->avx512_rr1_p6.sat_cutoff);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_rr1_p6.magic_bias);
  const __m512 vlog2e = _mm512_set1_ps(params->avx512_rr1_p6.log2e);
  const __m512 vminus_ln2 = _mm512_set1_ps(params->avx512_rr1_p6.minus_ln2);
  const __m512 vc6 = _mm512_set1_ps(params->avx512_rr1_p6.c6);
  const __m512 vc5 = _mm512_set1_ps(params->avx512_rr1_p6.c5);
  const __m512 vc4 = _mm512_set1_ps(params->avx512_rr1_p6.c4);
  const __m512 vc3 = _mm512_set1_ps(params->avx512_rr1_p6.c3);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_rr1_p6.c2);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    __m512 vx0 = _mm512_loadu_ps(input);
    __m512 vx1 = _mm512_loadu_ps(input + 16);
    __m512 vx2 = _mm512_loadu_ps(input + 32);
    __m512 vx3 = _mm512_loadu_ps(input + 48);
    __m512 vx4 = _mm512_loadu_ps(input + 64);
    __m512 vx5 = _mm512_loadu_ps(input + 80);
    __m512 vx6 = _mm512_loadu_ps(input + 96);
    __m512 vx7 = _mm512_loadu_ps(input + 112);
    input += 128;

    const __m512 vz0 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx0, vprescale));
    const __m512 vz1 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx1, vprescale));
    const __m512 vz2 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx2, vprescale));
    const __m512 vz3 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx3, vprescale));
    const __m512 vz4 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx4, vprescale));
    const __m512 vz5 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx5, vprescale));
    const __m512 vz6 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx6, vprescale));
    const __m512 vz7 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx7, vprescale));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vlog2e, vmagic_bias);
    __m512 vn4 = _mm512_fmadd_ps(vz4, vlog2e, vmagic_bias);
    __m512 vn5 = _mm512_fmadd_ps(vz5, vlog2e, vmagic_bias);
    __m512 vn6 = _mm512_fmadd_ps(vz6, vlog2e, vmagic_bias);
    __m512 vn7 = _mm512_fmadd_ps(vz7, vlog2e, vmagic_bias);

    __m512 vs0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn0), 23));
    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    __m512 vs1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn1), 23));
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    __m512 vs2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn2), 23));
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    __m512 vs3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn3), 23));
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);
    __m512 vs4 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn4), 23));
    vn4 = _mm512_sub_ps(vn4, vmagic_bias);
    __m512 vs5 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn5), 23));
    vn5 = _mm512_sub_ps(vn5, vmagic_bias);
    __m512 vs6 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn6), 23));
    vn6 = _mm512_sub_ps(vn6, vmagic_bias);
    __m512 vs7 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn7), 23));
    vn7 = _mm512_sub_ps(vn7, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vz2);
    __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2, vz3);
    __m512 vt4 = _mm512_fmadd_ps(vn4, vminus_ln2, vz4);
    __m512 vt5 = _mm512_fmadd_ps(vn5, vminus_ln2, vz5);
    __m512 vt6 = _mm512_fmadd_ps(vn6, vminus_ln2, vz6);
    __m512 vt7 = _mm512_fmadd_ps(vn7, vminus_ln2, vz7);

    __m512 vp0 = _mm512_fmadd_ps(vc6, vt0, vc5);
    __m512 vp1 = _mm512_fmadd_ps(vc6, vt1, vc5);
    __m512 vp2 = _mm512_fmadd_ps(vc6, vt2, vc5);
    __m512 vp3 = _mm512_fmadd_ps(vc6, vt3, vc5);
    __m512 vp4 = _mm512_fmadd_ps(vc6, vt4, vc5);
    __m512 vp5 = _mm512_fmadd_ps(vc6, vt5, vc5);
    __m512 vp6 = _mm512_fmadd_ps(vc6, vt6, vc5);
    __m512 vp7 = _mm512_fmadd_ps(vc6, vt7, vc5);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc4);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc4);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc4);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc4);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc4);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc4);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc4);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc4);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc3);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc3);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc3);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc3);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vc2);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vc2);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vc2);

    vp0 = _mm512_mul_ps(vp0, vt0);
    vt0 = _mm512_mul_ps(vt0, vs0);
    vp1 = _mm512_mul_ps(vp1, vt1);
    vt1 = _mm512_mul_ps(vt1, vs1);
    vp2 = _mm512_mul_ps(vp2, vt2);
    vt2 = _mm512_mul_ps(vt2, vs2);
    vp3 = _mm512_mul_ps(vp3, vt3);
    vt3 = _mm512_mul_ps(vt3, vs3);
    vp4 = _mm512_mul_ps(vp4, vt4);
    vt4 = _mm512_mul_ps(vt4, vs4);
    vp5 = _mm512_mul_ps(vp5, vt5);
    vt5 = _mm512_mul_ps(vt5, vs5);
    vp6 = _mm512_mul_ps(vp6, vt6);
    vt6 = _mm512_mul_ps(vt6, vs6);
    vp7 = _mm512_mul_ps(vp7, vt7);
    vt7 = _mm512_mul_ps(vt7, vs7);

    vs0 = _mm512_fmsub_ps(vs0, valpha, valpha);
    vs1 = _mm512_fmsub_ps(vs1, valpha, valpha);
    vs2 = _mm512_fmsub_ps(vs2, valpha, valpha);
    vs3 = _mm512_fmsub_ps(vs3, valpha, valpha);
    vs4 = _mm512_fmsub_ps(vs4, valpha, valpha);
    vs5 = _mm512_fmsub_ps(vs5, valpha, valpha);
    vs6 = _mm512_fmsub_ps(vs6, valpha, valpha);
    vs7 = _mm512_fmsub_ps(vs7, valpha, valpha);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vt0);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vt1);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vt2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vt3);
    vp4 = _mm512_fmadd_ps(vp4, vt4, vt4);
    vp5 = _mm512_fmadd_ps(vp5, vt5, vt5);
    vp6 = _mm512_fmadd_ps(vp6, vt6, vt6);
    vp7 = _mm512_fmadd_ps(vp7, vt7, vt7);

    const __m512 vzero = _mm512_setzero_ps();
    __m512 vy0 = _mm512_fmadd_ps(vp0, valpha, vs0);
    const __mmask16 vsign0 = _mm512_cmp_ps_mask(vx0, vzero, _CMP_NLT_US);
    __m512 vy1 = _mm512_fmadd_ps(vp1, valpha, vs1);
    const __mmask16 vsign1 = _mm512_cmp_ps_mask(vx1, vzero, _CMP_NLT_US);
    __m512 vy2 = _mm512_fmadd_ps(vp2, valpha, vs2);
    const __mmask16 vsign2 = _mm512_cmp_ps_mask(vx2, vzero, _CMP_NLT_US);
    __m512 vy3 = _mm512_fmadd_ps(vp3, valpha, vs3);
    const __mmask16 vsign3 = _mm512_cmp_ps_mask(vx3, vzero, _CMP_NLT_US);
    __m512 vy4 = _mm512_fmadd_ps(vp4, valpha, vs4);
    const __mmask16 vsign4 = _mm512_cmp_ps_mask(vx4, vzero, _CMP_NLT_US);
    __m512 vy5 = _mm512_fmadd_ps(vp5, valpha, vs5);
    const __mmask16 vsign5 = _mm512_cmp_ps_mask(vx5, vzero, _CMP_NLT_US);
    __m512 vy6 = _mm512_fmadd_ps(vp6, valpha, vs6);
    const __mmask16 vsign6 = _mm512_cmp_ps_mask(vx6, vzero, _CMP_NLT_US);
    __m512 vy7 = _mm512_fmadd_ps(vp7, valpha, vs7);
    const __mmask16 vsign7 = _mm512_cmp_ps_mask(vx7, vzero, _CMP_NLT_US);

    vy0 = _mm512_mask_mul_ps(vy0, vsign0, vx0, vbeta);
    vy1 = _mm512_mask_mul_ps(vy1, vsign1, vx1, vbeta);
    vy2 = _mm512_mask_mul_ps(vy2, vsign2, vx2, vbeta);
    vy3 = _mm512_mask_mul_ps(vy3, vsign3, vx3, vbeta);
    vy4 = _mm512_mask_mul_ps(vy4, vsign4, vx4, vbeta);
    vy5 = _mm512_mask_mul_ps(vy5, vsign5, vx5, vbeta);
    vy6 = _mm512_mask_mul_ps(vy6, vsign6, vx6, vbeta);
    vy7 = _mm512_mask_mul_ps(vy7, vsign7, vx7, vbeta);

    _mm512_storeu_ps(output, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    _mm512_storeu_ps(output + 32, vy2);
    _mm512_storeu_ps(output + 48, vy3);
    _mm512_storeu_ps(output + 64, vy4);
    _mm512_storeu_ps(output + 80, vy5);
    _mm512_storeu_ps(output + 96, vy6);
    _mm512_storeu_ps(output + 112, vy7);
    output += 128;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc6, vt, vc5);
    vp = _mm512_fmadd_ps(vp, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx = _mm512_maskz_loadu_ps(vmask, input);

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    __m512 vs = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vn), 23));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc6, vt, vc5);
    vp = _mm512_fmadd_ps(vp, vt, vc4);
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vhswish_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vsixth = _mm512_set1_ps(params->avx512.sixth);
  const __m512 vhalf = _mm512_set1_ps(params->avx512.half);
  const __m512 vone = _mm512_set1_ps(params->avx512.one);
  const __m512 vzero = _mm512_setzero_ps();

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;
    __m512 vacc = _mm512_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm512_max_ps(vacc, vzero);
    vacc = _mm512_min_ps(vacc, vone);
    vacc = _mm512_mul_ps(vacc, vx);
    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    __m512 vacc = _mm512_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm512_max_ps(vacc, vzero);
    vacc = _mm512_min_ps(vacc, vone);
    vacc = _mm512_mul_ps(vacc, vx);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vlrelu_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vslope = _mm512_set1_ps(params->scalar.slope);
  const __m512 vzero = _mm512_setzero_ps();

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __mmask16 vsign0123456789ABCDEF = _mm512_cmp_ps_mask(vacc0123456789ABCDEF, vzero, _CMP_LT_OQ);

    vacc0123456789ABCDEF = _mm512_mask_mul_ps(vacc0123456789ABCDEF, vsign0123456789ABCDEF, vacc0123456789ABCDEF, vslope);

    _mm512_storeu_ps(output, vacc0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input);
    const __mmask16 vsign = _mm512_cmp_ps_mask(vacc, vzero, _CMP_LT_OQ);
    vacc = _mm512_mask_mul_ps(vacc, vsign, vacc, vslope);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vrndd_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_NEG_INF);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_NEG_INF);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vrndne_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_NEAREST_INT);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vrndu_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_POS_INF);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_POS_INF);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vrndz_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_ZERO);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_ZERO);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vrsqrt_ukernel__avx512f_rsqrt_u32(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m512 vthree = _mm512_set1_ps(params->avx512.three);
  const __m512 vneg_half = _mm512_set1_ps(params->avx512.neg_half);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(input);
    const __m512 vx1 = _mm512_loadu_ps(input + 16);
    input += 32;

    // Generate the initial 14-bit approximation.
    const __m512 vt0_0 = _mm512_rsqrt14_ps(vx0);
    const __m512 vt0_1 = _mm512_rsqrt14_ps(vx1);

    // Do a single Newton-Raphson step as described above.
    const __m512 vt1_0 = _mm512_mul_ps(vt0_0, vt0_0);
    const __m512 vt1_1 = _mm512_mul_ps(vt0_1, vt0_1);
    const __m512 vt3_0 = _mm512_fmsub_ps(vx0, vt1_0, vthree);
    const __m512 vt3_1 = _mm512_fmsub_ps(vx1, vt1_1, vthree);
    const __m512 vt4_0 = _mm512_mul_ps(vneg_half, vt0_0);
    const __m512 vt4_1 = _mm512_mul_ps(vneg_half, vt0_1);
    const __m512 vy0 = _mm512_mul_ps(vt3_0, vt4_0);
    const __m512 vy1 = _mm512_mul_ps(vt3_1, vt4_1);

    // Store the results.
    _mm512_storeu_ps(output, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    // Generate the initial 14-bit approximation.
    const __m512 vt0 = _mm512_rsqrt14_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m512 vt1 = _mm512_mul_ps(vt0, vt0);
    const __m512 vt3 = _mm512_fmsub_ps(vx, vt1, vthree);
    const __m512 vt4 = _mm512_mul_ps(vneg_half, vt0);
    const __m512 vy = _mm512_mul_ps(vt3, vt4);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);

    // Generate the initial 14-bit approximation.
    const __m512 vt0 = _mm512_rsqrt14_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m512 vt1 = _mm512_mul_ps(vt0, vt0);
    const __m512 vt3 = _mm512_fmsub_ps(vx, vt1, vthree);
    const __m512 vt4 = _mm512_mul_ps(vneg_half, vt0);
    __m512 vy = _mm512_mul_ps(vt3, vt4);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512_rr2_lut32_p2.sign_mask);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_rr2_lut32_p2.magic_bias);
  const __m512 vlog2e = _mm512_set1_ps(params->avx512_rr2_lut32_p2.log2e);
  const __m512 vtable_lo = _mm512_load_ps(params->avx512_rr2_lut32_p2.table_lo);
  const __m512 vtable_hi = _mm512_load_ps(params->avx512_rr2_lut32_p2.table_hi);
  const __m512 vminus_ln2_hi = _mm512_set1_ps(params->avx512_rr2_lut32_p2.minus_ln2_hi);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(params->avx512_rr2_lut32_p2.minus_ln2_lo);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_rr2_lut32_p2.c2);
  const __m512 vc1 = _mm512_set1_ps(params->avx512_rr2_lut32_p2.c1);
  const __m512 vone = _mm512_set1_ps(params->avx512_rr2_lut32_p2.one);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(input);
    const __m512 vx1 = _mm512_loadu_ps(input + 16);
    const __m512 vx2 = _mm512_loadu_ps(input + 32);
    const __m512 vx3 = _mm512_loadu_ps(input + 48);
    input += 64;

    const __m512 vz0 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx0), vsign_mask));
    const __m512 vz1 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx1), vsign_mask));
    const __m512 vz2 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx2), vsign_mask));
    const __m512 vz3 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx3), vsign_mask));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vlog2e, vmagic_bias);

    const __m512 vl0 = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn0), vtable_hi);
    const __m512 vl1 = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn1), vtable_hi);
    const __m512 vl2 = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn2), vtable_hi);
    const __m512 vl3 = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn3), vtable_hi);

    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_hi, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2_hi, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2_hi, vz2);
    __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2_hi, vz3);

    vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_lo, vt0);
    vt1 = _mm512_fmadd_ps(vn1, vminus_ln2_lo, vt1);
    vt2 = _mm512_fmadd_ps(vn2, vminus_ln2_lo, vt2);
    vt3 = _mm512_fmadd_ps(vn3, vminus_ln2_lo, vt3);

    __m512 vp0 = _mm512_fmadd_ps(vt0, vc2, vc1);
    __m512 vp1 = _mm512_fmadd_ps(vt1, vc2, vc1);
    __m512 vp2 = _mm512_fmadd_ps(vt2, vc2, vc1);
    __m512 vp3 = _mm512_fmadd_ps(vt3, vc2, vc1);

    vt0 = _mm512_mul_ps(vt0, vl0);
    vt1 = _mm512_mul_ps(vt1, vl1);
    vt2 = _mm512_mul_ps(vt2, vl2);
    vt3 = _mm512_mul_ps(vt3, vl3);

    vp0 = _mm512_fmadd_ps(vt0, vp0, vl0);
    vp1 = _mm512_fmadd_ps(vt1, vp1, vl1);
    vp2 = _mm512_fmadd_ps(vt2, vp2, vl2);
    vp3 = _mm512_fmadd_ps(vt3, vp3, vl3);

    const __m512 ve0 = _mm512_scalef_ps(vp0, vn0);
    const __m512 ve1 = _mm512_scalef_ps(vp1, vn1);
    const __m512 ve2 = _mm512_scalef_ps(vp2, vn2);
    const __m512 ve3 = _mm512_scalef_ps(vp3, vn3);

    const __m512 vd0 = _mm512_add_ps(ve0, vone);
    const __m512 vd1 = _mm512_add_ps(ve1, vone);
    const __m512 vd2 = _mm512_add_ps(ve2, vone);
    const __m512 vd3 = _mm512_add_ps(ve3, vone);

    __m512 vf0 = _mm512_div_ps(ve0, vd0);
    __m512 vf1 = _mm512_div_ps(ve1, vd1);
    __m512 vf2 = _mm512_div_ps(ve2, vd2);
    __m512 vf3 = _mm512_div_ps(ve3, vd3);

    vf0 = _mm512_mask_sub_ps(vf0, _mm512_testn_epi32_mask(_mm512_castps_si512(vx0), vsign_mask), vone, vf0);
    vf1 = _mm512_mask_sub_ps(vf1, _mm512_testn_epi32_mask(_mm512_castps_si512(vx1), vsign_mask), vone, vf1);
    vf2 = _mm512_mask_sub_ps(vf2, _mm512_testn_epi32_mask(_mm512_castps_si512(vx2), vsign_mask), vone, vf2);
    vf3 = _mm512_mask_sub_ps(vf3, _mm512_testn_epi32_mask(_mm512_castps_si512(vx3), vsign_mask), vone, vf3);

    _mm512_storeu_ps(output, vf0);
    _mm512_storeu_ps(output + 16, vf1);
    _mm512_storeu_ps(output + 32, vf2);
    _mm512_storeu_ps(output + 48, vf3);
    output += 64;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512 vl = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn), vtable_hi);
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vz);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    __m512 vp = _mm512_fmadd_ps(vt, vc2, vc1);
    vt = _mm512_mul_ps(vt, vl);
    vp = _mm512_fmadd_ps(vt, vp, vl);

    const __m512 ve = _mm512_scalef_ps(vp, vn);
    const __m512 vd = _mm512_add_ps(ve, vone);

    __m512 vf = _mm512_div_ps(ve, vd);

    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_storeu_ps(output, vf);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512 vl = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn), vtable_hi);
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vz);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    __m512 vp = _mm512_fmadd_ps(vt, vc2, vc1);
    vt = _mm512_mul_ps(vt, vl);
    vp = _mm512_fmadd_ps(vt, vp, vl);

    const __m512 ve = _mm512_scalef_ps(vp, vn);
    const __m512 vd = _mm512_add_ps(ve, vone);

    __m512 vf = _mm512_div_ps(ve, vd);

    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_mask_storeu_ps(output, vmask, vf);
  }
}

void xnn_f32_vabs_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_abs_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vnonsign_mask = _mm512_set1_epi32((int) params->avx512.nonsign_mask);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512i vx0123456789ABCDEF = _mm512_loadu_si512(input);
    input += 16;

    const __m512i vy0123456789ABCDEF = _mm512_and_epi32(vx0123456789ABCDEF, vnonsign_mask);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512i vx = _mm512_maskz_loadu_epi32(vmask, input);
    const __m512i vy = _mm512_and_epi32(vx, vnonsign_mask);
    _mm512_mask_storeu_epi32(output, vmask, vy);
  }
}

void xnn_f32_vneg_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_neg_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512.sign_mask);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512i vx0123456789ABCDEF = _mm512_loadu_si512(input);
    input += 16;

    const __m512i vy0123456789ABCDEF = _mm512_xor_epi32(vx0123456789ABCDEF, vsign_mask);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512i vx = _mm512_maskz_loadu_epi32(vmask, input);
    const __m512i vy = _mm512_xor_epi32(vx, vsign_mask);
    _mm512_mask_storeu_epi32(output, vmask, vy);
  }
}

void xnn_f32_vsqr_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_mul_ps(vx0123456789ABCDEF, vx0123456789ABCDEF);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_mul_ps(vx, vx);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 16
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m512 vb0 = _mm512_loadu_ps(b);
        _mm512_store_ps(packed_w, vb0);
        b += 16;
      } else {
        const __m512 vzero = _mm512_setzero_ps();
        _mm512_store_ps(packed_w, vzero);
      }
      packed_w += 16;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;
      const float* w4 = w3 + kc;
      const float* w5 = w4 + kc;
      const float* w6 = w5 + kc;
      const float* w7 = w6 + kc;
      const float* w8 = w7 + kc;
      const float* w9 = w8 + kc;
      const float* w10 = w9 + kc;
      const float* w11 = w10 + kc;
      const float* w12 = w11 + kc;
      const float* w13 = w12 + kc;
      const float* w14 = w13 + kc;
      const float* w15 = w14 + kc;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        // Load first 4 rows of N into low part of each register
        __m512 v0x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w0));
        w0 += 4;
        __m512 v1x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w1));
        w1 += 4;
        __m512 v2x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w2));
        w2 += 4;
        __m512 v3x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w3));
        w3 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w8), 2);
        w8 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w9), 2);
        w9 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w10), 2);
        w10 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w11), 2);
        w11 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w12), 3);
        w12 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w13), 3);
        w13 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w14), 3);
        w14 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w15), 3);
        w15 += 4;

        // Transpose 2x2
        const __m512 vtmp0x0123 = _mm512_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m512 vtmp1x0123 = _mm512_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m512 vtmp2x0123 = _mm512_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m512 vtmp3x0123 = _mm512_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        xnn_prefetch_to_l1((const int8_t*) w15 + 128);
         // Transpose 4x4
        v0x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3

        _mm512_store_ps(packed_w, v0x0123);
        _mm512_store_ps(packed_w + 16, v1x0123);
        _mm512_store_ps(packed_w + 32, v2x0123);
        _mm512_store_ps(packed_w + 48, v3x0123);
        packed_w += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        if (k & 2) {
          // Read blocks of 4x2
          // a b
          // c d
          // e f
          // g h
          __m128 v0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 2;
          __m128 v1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 2;
          __m128 v2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 2;
          __m128 v3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 2;
          __m128 v4 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 2;
          __m128 v5 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 2;
          __m128 v6 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 2;
          __m128 v7 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 2;
          __m128 v8 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 2;
          __m128 v9 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 2;
          __m128 v10 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 2;
          __m128 v11 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 2;
          __m128 v12 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 2;
          __m128 v13 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 2;
          __m128 v14 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 2;
          __m128 v15 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w15));
          w15 += 2;

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a c e g   from row 0, 1
          v1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a c e g   from row 0, 1
          v5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a c b d   from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // e g f h   from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a c e g   from row 0, 1
          v9 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a c b d   from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a c e g   from row 0, 1
          v13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // b d f h   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v1);
          _mm_store_ps(packed_w + 20, v5);
          _mm_store_ps(packed_w + 24, v9);
          _mm_store_ps(packed_w + 28, v13);
          packed_w += 32;
        }
        if (k & 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
          __m128 v0 = _mm_load_ss(w0);
          w0 += 1;
          __m128 v1 = _mm_load_ss(w1);
          w1 += 1;
          __m128 v2 = _mm_load_ss(w2);
          w2 += 1;
          __m128 v3 = _mm_load_ss(w3);
          w3 += 1;
          __m128 v4 = _mm_load_ss(w4);
          w4 += 1;
          __m128 v5 = _mm_load_ss(w5);
          w5 += 1;
          __m128 v6 = _mm_load_ss(w6);
          w6 += 1;
          __m128 v7 = _mm_load_ss(w7);
          w7 += 1;
          __m128 v8 = _mm_load_ss(w8);
          w8 += 1;
          __m128 v9 = _mm_load_ss(w9);
          w9 += 1;
          __m128 v10 = _mm_load_ss(w10);
          w10 += 1;
          __m128 v11 = _mm_load_ss(w11);
          w11 += 1;
          __m128 v12 = _mm_load_ss(w12);
          w12 += 1;
          __m128 v13 = _mm_load_ss(w13);
          w13 += 1;
          __m128 v14 = _mm_load_ss(w14);
          w14 += 1;
          __m128 v15 = _mm_load_ss(w15);
          w15 += 1;

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a b  from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // c d  from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a b  from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a b c d   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          packed_w += 16;
        }
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (16 - n);
      } else {
        const __m512 vzero = _mm512_setzero_ps();
        _mm512_store_ps(packed_w, vzero);
        packed_w += 16;
      }

      // NR remainder has less than 16 rows so last row is not loaded
      // For SR=4 the
      const float* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const float* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const float* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const float* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const float* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const float* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const float* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const float* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const float* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const float* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const float* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const float* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const float* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const float* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        // Load first 4 rows of N into low part of each register
        __m512 v0x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w0));
        w0 += 4;
        __m512 v1x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w1));
        w1 += 4;
        __m512 v2x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w2));
        w2 += 4;
        // castps leaves upper 128 bits undefined, so zero them.
        __m512 v3x0123 = _mm512_zextps128_ps512(_mm_loadu_ps(w3));
        w3 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w8), 2);
        w8 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w9), 2);
        w9 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w10), 2);
        w10 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w11), 2);
        w11 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w12), 3);
        w12 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w13), 3);
        w13 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w14), 3);
        w14 += 4;

        // Transpose 2x2
        const __m512 vtmp0x0123 = _mm512_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m512 vtmp1x0123 = _mm512_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m512 vtmp2x0123 = _mm512_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m512 vtmp3x0123 = _mm512_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        // Transpose 4x4
        v0x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3

        _mm512_store_ps(packed_w, v0x0123);
        _mm512_store_ps(packed_w + 16, v1x0123);
        _mm512_store_ps(packed_w + 32, v2x0123);
        _mm512_store_ps(packed_w + 48, v3x0123);
        packed_w += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        if (k & 2) {
          // Read blocks of 4x2
          // a b
          // c d
          // e f
          // g h
          __m128 v0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 2;
          __m128 v1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 2;
          __m128 v2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 2;
          __m128 v3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 2;
          __m128 v4 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 2;
          __m128 v5 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 2;
          __m128 v6 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 2;
          __m128 v7 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 2;
          __m128 v8 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 2;
          __m128 v9 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 2;
          __m128 v10 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 2;
          __m128 v11 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 2;
          __m128 v12 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 2;
          __m128 v13 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 2;
          __m128 v14 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 2;

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a c e g   from row 0, 1
          v1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a c e g   from row 0, 1
          v5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a c b d   from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // e g f h   from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a c e g   from row 0, 1
          v9 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a c b d   from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v14);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a c e g   from row 0, 1
          v13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // b d f h   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v1);
          _mm_store_ps(packed_w + 20, v5);
          _mm_store_ps(packed_w + 24, v9);
          _mm_store_ps(packed_w + 28, v13);
          packed_w += 32;
        }
        if (k & 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
          __m128 v0 = _mm_load_ss(w0);
          w0 += 1;
          __m128 v1 = _mm_load_ss(w1);
          w1 += 1;
          __m128 v2 = _mm_load_ss(w2);
          w2 += 1;
          __m128 v3 = _mm_load_ss(w3);
          w3 += 1;
          __m128 v4 = _mm_load_ss(w4);
          w4 += 1;
          __m128 v5 = _mm_load_ss(w5);
          w5 += 1;
          __m128 v6 = _mm_load_ss(w6);
          w6 += 1;
          __m128 v7 = _mm_load_ss(w7);
          w7 += 1;
          __m128 v8 = _mm_load_ss(w8);
          w8 += 1;
          __m128 v9 = _mm_load_ss(w9);
          w9 += 1;
          __m128 v10 = _mm_load_ss(w10);
          w10 += 1;
          __m128 v11 = _mm_load_ss(w11);
          w11 += 1;
          __m128 v12 = _mm_load_ss(w12);
          w12 += 1;
          __m128 v13 = _mm_load_ss(w13);
          w13 += 1;
          __m128 v14 = _mm_load_ss(w14);
          w14 += 1;

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a b  from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // c d  from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a b  from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v14);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a b c d   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          packed_w += 16;
        }
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
