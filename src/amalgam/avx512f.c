// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/prelu.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vunary.h>


void xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
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


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

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


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_up16x3__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
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


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

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


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
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


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

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


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
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


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

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


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

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
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
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
    vacc0x0123456789ABCDEF = _mm512_max_ps(vacc0x0123456789ABCDEF, vmin);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vacc0x0123456789ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
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
    vacc0x0123456789ABCDEF = _mm512_max_ps(vacc0x0123456789ABCDEF, vmin);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vacc1x0123456789ABCDEF, vmin);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vacc2x0123456789ABCDEF, vmin);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vacc3x0123456789ABCDEF, vmin);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vacc4x0123456789ABCDEF, vmin);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vacc5x0123456789ABCDEF, vmin);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vacc6x0123456789ABCDEF, vmin);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vacc0x0123456789ABCDEF, vmax);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vacc1x0123456789ABCDEF, vmax);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vacc2x0123456789ABCDEF, vmax);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vacc3x0123456789ABCDEF, vmax);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vacc4x0123456789ABCDEF, vmax);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vacc5x0123456789ABCDEF, vmax);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vacc6x0123456789ABCDEF, vmax);

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

      a6 = (const float*) ((uintptr_t) a6 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

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

void xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
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
    vacc0x0123456789ABCDEF = _mm512_max_ps(vacc0x0123456789ABCDEF, vmin);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vacc0x0123456789ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

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
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
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
    vacc0x0123456789ABCDEF = _mm512_max_ps(vacc0x0123456789ABCDEF, vmin);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vacc1x0123456789ABCDEF, vmin);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vacc2x0123456789ABCDEF, vmin);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vacc3x0123456789ABCDEF, vmin);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vacc4x0123456789ABCDEF, vmin);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vacc5x0123456789ABCDEF, vmin);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vacc6x0123456789ABCDEF, vmin);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vacc0x0123456789ABCDEF, vmax);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vacc1x0123456789ABCDEF, vmax);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vacc2x0123456789ABCDEF, vmax);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vacc3x0123456789ABCDEF, vmax);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vacc4x0123456789ABCDEF, vmax);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vacc5x0123456789ABCDEF, vmax);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vacc6x0123456789ABCDEF, vmax);

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
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

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
    const float*restrict input,
    size_t input_stride,
    const float*restrict weights,
    float*restrict output,
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
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << (c >> 2 /* log2(sizeof(float))*/)) - UINT32_C(1)));

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

void xnn_f32_vadd_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(b);
    const __m512 vbGHIJKLMNOPQRSTUV = _mm512_loadu_ps(b + 16);
    b += 32;

    __m512 vy0123456789ABCDEF = _mm512_add_ps(va0123456789ABCDEF, vb0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_add_ps(vaGHIJKLMNOPQRSTUV, vbGHIJKLMNOPQRSTUV);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    const __m512 vb = _mm512_loadu_ps(b);
    b += 16;

    __m512 vy = _mm512_add_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, b);

    __m512 vy = _mm512_add_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vaddc_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_add_ps(va0123456789ABCDEF, vb);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_add_ps(vaGHIJKLMNOPQRSTUV, vb);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_add_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_add_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vdiv_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(b);
    const __m512 vbGHIJKLMNOPQRSTUV = _mm512_loadu_ps(b + 16);
    b += 32;

    __m512 vy0123456789ABCDEF = _mm512_div_ps(va0123456789ABCDEF, vb0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_div_ps(vaGHIJKLMNOPQRSTUV, vbGHIJKLMNOPQRSTUV);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    const __m512 vb = _mm512_loadu_ps(b);
    b += 16;

    __m512 vy = _mm512_div_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, b);

    __m512 vy = _mm512_div_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vdivc_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_div_ps(va0123456789ABCDEF, vb);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_div_ps(vaGHIJKLMNOPQRSTUV, vb);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_div_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_div_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);


  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(b);
    const __m512 vbGHIJKLMNOPQRSTUV = _mm512_loadu_ps(b + 16);
    b += 32;

    __m512 vy0123456789ABCDEF = _mm512_max_ps(va0123456789ABCDEF, vb0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vaGHIJKLMNOPQRSTUV, vbGHIJKLMNOPQRSTUV);



    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    const __m512 vb = _mm512_loadu_ps(b);
    b += 16;

    __m512 vy = _mm512_max_ps(va, vb);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, b);

    __m512 vy = _mm512_max_ps(va, vb);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vmaxc_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);


  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_max_ps(va0123456789ABCDEF, vb);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vaGHIJKLMNOPQRSTUV, vb);



    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_max_ps(va, vb);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_max_ps(va, vb);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vmin_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);


  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(b);
    const __m512 vbGHIJKLMNOPQRSTUV = _mm512_loadu_ps(b + 16);
    b += 32;

    __m512 vy0123456789ABCDEF = _mm512_min_ps(va0123456789ABCDEF, vb0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vaGHIJKLMNOPQRSTUV, vbGHIJKLMNOPQRSTUV);



    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    const __m512 vb = _mm512_loadu_ps(b);
    b += 16;

    __m512 vy = _mm512_min_ps(va, vb);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, b);

    __m512 vy = _mm512_min_ps(va, vb);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vminc_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);


  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_min_ps(va0123456789ABCDEF, vb);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vaGHIJKLMNOPQRSTUV, vb);



    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_min_ps(va, vb);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_min_ps(va, vb);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vmul_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(b);
    const __m512 vbGHIJKLMNOPQRSTUV = _mm512_loadu_ps(b + 16);
    b += 32;

    __m512 vy0123456789ABCDEF = _mm512_mul_ps(va0123456789ABCDEF, vb0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vaGHIJKLMNOPQRSTUV, vbGHIJKLMNOPQRSTUV);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    const __m512 vb = _mm512_loadu_ps(b);
    b += 16;

    __m512 vy = _mm512_mul_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, b);

    __m512 vy = _mm512_mul_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vmulc_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_mul_ps(va0123456789ABCDEF, vb);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vaGHIJKLMNOPQRSTUV, vb);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_mul_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_mul_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vrdivc_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_div_ps(vb, va0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_div_ps(vb, vaGHIJKLMNOPQRSTUV);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_div_ps(vb, va);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_div_ps(vb, va);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vrsubc_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_sub_ps(vb, va0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_sub_ps(vb, vaGHIJKLMNOPQRSTUV);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_sub_ps(vb, va);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_sub_ps(vb, va);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vsqrdiff_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);


  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(b);
    const __m512 vbGHIJKLMNOPQRSTUV = _mm512_loadu_ps(b + 16);
    b += 32;

    __m512 vy0123456789ABCDEF = _mm512_sub_ps(va0123456789ABCDEF, vb0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_sub_ps(vaGHIJKLMNOPQRSTUV, vbGHIJKLMNOPQRSTUV);

    vy0123456789ABCDEF = _mm512_mul_ps(vy0123456789ABCDEF, vy0123456789ABCDEF);
    vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vyGHIJKLMNOPQRSTUV, vyGHIJKLMNOPQRSTUV);


    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    const __m512 vb = _mm512_loadu_ps(b);
    b += 16;

    __m512 vy = _mm512_sub_ps(va, vb);
    vy = _mm512_mul_ps(vy, vy);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, b);

    __m512 vy = _mm512_sub_ps(va, vb);
    vy = _mm512_mul_ps(vy, vy);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vsqrdiffc_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);


  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_sub_ps(va0123456789ABCDEF, vb);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_sub_ps(vaGHIJKLMNOPQRSTUV, vb);

    vy0123456789ABCDEF = _mm512_mul_ps(vy0123456789ABCDEF, vy0123456789ABCDEF);
    vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vyGHIJKLMNOPQRSTUV, vyGHIJKLMNOPQRSTUV);


    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_sub_ps(va, vb);
    vy = _mm512_mul_ps(vy, vy);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_sub_ps(va, vb);
    vy = _mm512_mul_ps(vy, vy);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vsub_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    const __m512 vb0123456789ABCDEF = _mm512_loadu_ps(b);
    const __m512 vbGHIJKLMNOPQRSTUV = _mm512_loadu_ps(b + 16);
    b += 32;

    __m512 vy0123456789ABCDEF = _mm512_sub_ps(va0123456789ABCDEF, vb0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_sub_ps(vaGHIJKLMNOPQRSTUV, vbGHIJKLMNOPQRSTUV);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    const __m512 vb = _mm512_loadu_ps(b);
    b += 16;

    __m512 vy = _mm512_sub_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);
    const __m512 vb = _mm512_maskz_loadu_ps(vmask, b);

    __m512 vy = _mm512_sub_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vsubc_minmax_ukernel__avx512f_x32(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  const __m512 vb = _mm512_set1_ps(*b);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 va0123456789ABCDEF = _mm512_loadu_ps(a);
    const __m512 vaGHIJKLMNOPQRSTUV = _mm512_loadu_ps(a + 16);
    a += 32;

    __m512 vy0123456789ABCDEF = _mm512_sub_ps(va0123456789ABCDEF, vb);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_sub_ps(vaGHIJKLMNOPQRSTUV, vb);


    vy0123456789ABCDEF = _mm512_max_ps(vy0123456789ABCDEF, vy_min);
    vyGHIJKLMNOPQRSTUV = _mm512_max_ps(vyGHIJKLMNOPQRSTUV, vy_min);

    vy0123456789ABCDEF = _mm512_min_ps(vy0123456789ABCDEF, vy_max);
    vyGHIJKLMNOPQRSTUV = _mm512_min_ps(vyGHIJKLMNOPQRSTUV, vy_max);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    _mm512_storeu_ps(y + 16, vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 va = _mm512_loadu_ps(a);
    a += 16;

    __m512 vy = _mm512_sub_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 va = _mm512_maskz_loadu_ps(vmask, a);

    __m512 vy = _mm512_sub_ps(va, vb);
    vy = _mm512_max_ps(vy, vy_min);
    vy = _mm512_min_ps(vy, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vclamp_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m512 vy_min = _mm512_set1_ps(params->scalar.min);
  const __m512 vy_max = _mm512_set1_ps(params->scalar.max);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(x);
    x += 16;

    vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEF, vy_min);

    vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vy_max);

    _mm512_storeu_ps(y, vacc0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, x);
    vacc = _mm512_max_ps(vacc, vy_min);
    vacc = _mm512_min_ps(vacc, vy_max);
    _mm512_mask_storeu_ps(y, vmask, vacc);
  }
}

void xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m512 vprescale = _mm512_set1_ps(params->avx512_rr1_lut16_p3.prescale);
  const __m512 valpha = _mm512_set1_ps(params->avx512_rr1_lut16_p3.alpha);
  const __m512 vbeta = _mm512_set1_ps(params->avx512_rr1_lut16_p3.beta);
  const __m512 vsat_cutoff = _mm512_set1_ps(params->avx512_rr1_lut16_p3.sat_cutoff);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_rr1_lut16_p3.magic_bias);
  const __m512 vlog2e = _mm512_set1_ps(params->avx512_rr1_lut16_p3.log2e);
  const __m512 vminus_ln2 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.minus_ln2);
  const __m512 vc3 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.c3);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.c2);
  const __m512i vtable = _mm512_load_si512(params->avx512_rr1_lut16_p3.table);

  for (; n >= 64 * sizeof(float); n -= 64 * sizeof(float)) {
    __m512 vx0 = _mm512_loadu_ps(x);
    __m512 vx1 = _mm512_loadu_ps(x + 16);
    __m512 vx2 = _mm512_loadu_ps(x + 32);
    __m512 vx3 = _mm512_loadu_ps(x + 48);
    x += 64;

    const __m512 vz0 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx0, vprescale));
    const __m512 vz1 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx1, vprescale));
    const __m512 vz2 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx2, vprescale));
    const __m512 vz3 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx3, vprescale));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vlog2e, vmagic_bias);

    const __m512i ven0 = _mm512_slli_epi32(_mm512_castps_si512(vn0), 19);
    const __m512i vl0 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn0), vtable);
    const __m512i ven1 = _mm512_slli_epi32(_mm512_castps_si512(vn1), 19);
    const __m512i vl1 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn1), vtable);
    const __m512i ven2 = _mm512_slli_epi32(_mm512_castps_si512(vn2), 19);
    const __m512i vl2 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn2), vtable);
    const __m512i ven3 = _mm512_slli_epi32(_mm512_castps_si512(vn3), 19);
    const __m512i vl3 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn3), vtable);

    __m512 vs0 = _mm512_castsi512_ps(_mm512_add_epi32(vl0, ven0));
    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    __m512 vs1 = _mm512_castsi512_ps(_mm512_add_epi32(vl1, ven1));
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    __m512 vs2 = _mm512_castsi512_ps(_mm512_add_epi32(vl2, ven2));
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    __m512 vs3 = _mm512_castsi512_ps(_mm512_add_epi32(vl3, ven3));
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vz2);
    __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2, vz3);

    __m512 vp0 = _mm512_fmadd_ps(vc3, vt0, vc2);
    __m512 vp1 = _mm512_fmadd_ps(vc3, vt1, vc2);
    __m512 vp2 = _mm512_fmadd_ps(vc3, vt2, vc2);
    __m512 vp3 = _mm512_fmadd_ps(vc3, vt3, vc2);

    vp0 = _mm512_mul_ps(vp0, vt0);
    vt0 = _mm512_mul_ps(vt0, vs0);
    vp1 = _mm512_mul_ps(vp1, vt1);
    vt1 = _mm512_mul_ps(vt1, vs1);
    vp2 = _mm512_mul_ps(vp2, vt2);
    vt2 = _mm512_mul_ps(vt2, vs2);
    vp3 = _mm512_mul_ps(vp3, vt3);
    vt3 = _mm512_mul_ps(vt3, vs3);

    vs0 = _mm512_fmsub_ps(vs0, valpha, valpha);
    vs1 = _mm512_fmsub_ps(vs1, valpha, valpha);
    vs2 = _mm512_fmsub_ps(vs2, valpha, valpha);
    vs3 = _mm512_fmsub_ps(vs3, valpha, valpha);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vt0);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vt1);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vt2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vt3);

    const __m512 vzero = _mm512_setzero_ps();
    __m512 vy0 = _mm512_fmadd_ps(vp0, valpha, vs0);
    const __mmask16 vsign0 = _mm512_cmp_ps_mask(vx0, vzero, _CMP_NLT_US);
    __m512 vy1 = _mm512_fmadd_ps(vp1, valpha, vs1);
    const __mmask16 vsign1 = _mm512_cmp_ps_mask(vx1, vzero, _CMP_NLT_US);
    __m512 vy2 = _mm512_fmadd_ps(vp2, valpha, vs2);
    const __mmask16 vsign2 = _mm512_cmp_ps_mask(vx2, vzero, _CMP_NLT_US);
    __m512 vy3 = _mm512_fmadd_ps(vp3, valpha, vs3);
    const __mmask16 vsign3 = _mm512_cmp_ps_mask(vx3, vzero, _CMP_NLT_US);

    vy0 = _mm512_mask_mul_ps(vy0, vsign0, vx0, vbeta);
    vy1 = _mm512_mask_mul_ps(vy1, vsign1, vx1, vbeta);
    vy2 = _mm512_mask_mul_ps(vy2, vsign2, vx2, vbeta);
    vy3 = _mm512_mask_mul_ps(vy3, vsign3, vx3, vbeta);

    _mm512_storeu_ps(y, vy0);
    _mm512_storeu_ps(y + 16, vy1);
    _mm512_storeu_ps(y + 32, vy2);
    _mm512_storeu_ps(y + 48, vy3);
    y += 64;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m512 vx = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512i ven = _mm512_slli_epi32(_mm512_castps_si512(vn), 19);
    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);
    __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ven));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc3, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_storeu_ps(y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    __m512 vx = _mm512_maskz_loadu_ps(vmask, x);

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512i ven = _mm512_slli_epi32(_mm512_castps_si512(vn), 19);
    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);
    __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ven));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc3, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vhswish_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m512 vsixth = _mm512_set1_ps(params->avx512.sixth);
  const __m512 vhalf = _mm512_set1_ps(params->avx512.half);
  const __m512 vone = _mm512_set1_ps(params->avx512.one);
  const __m512 vzero = _mm512_setzero_ps();

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(x);
    x += 16;
    __m512 vacc = _mm512_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm512_max_ps(vacc, vzero);
    vacc = _mm512_min_ps(vacc, vone);
    vacc = _mm512_mul_ps(vacc, vx);
    _mm512_storeu_ps(y, vacc);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    __m512 vacc = _mm512_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm512_max_ps(vacc, vzero);
    vacc = _mm512_min_ps(vacc, vone);
    vacc = _mm512_mul_ps(vacc, vx);
    _mm512_mask_storeu_ps(y, vmask, vacc);
  }
}

void xnn_f32_vlrelu_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m512 vslope = _mm512_set1_ps(params->scalar.slope);
  const __m512 vzero = _mm512_setzero_ps();

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(x);
    x += 16;

    const __mmask16 vsign0123456789ABCDEF = _mm512_cmp_ps_mask(vacc0123456789ABCDEF, vzero, _CMP_LT_OQ);

    vacc0123456789ABCDEF = _mm512_mask_mul_ps(vacc0123456789ABCDEF, vsign0123456789ABCDEF, vacc0123456789ABCDEF, vslope);

    _mm512_storeu_ps(y, vacc0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, x);
    const __mmask16 vsign = _mm512_mask_cmp_ps_mask(vmask, vacc, vzero, _CMP_LT_OQ);
    vacc = _mm512_mask_mul_ps(vacc, vsign, vacc, vslope);
    _mm512_mask_storeu_ps(y, vmask, vacc);
  }
}

void xnn_f32_vrndd_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_NEG_INF);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_NEG_INF);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vrndne_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_NEAREST_INT);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vrndu_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_POS_INF);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_POS_INF);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vrndz_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_ZERO);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_ZERO);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}

void xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n % sizeof(float) == 0);

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

  for (; n >= 64 * sizeof(float); n -= 64 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(x);
    const __m512 vx1 = _mm512_loadu_ps(x + 16);
    const __m512 vx2 = _mm512_loadu_ps(x + 32);
    const __m512 vx3 = _mm512_loadu_ps(x + 48);
    x += 64;

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

    _mm512_storeu_ps(y, vf0);
    _mm512_storeu_ps(y + 16, vf1);
    _mm512_storeu_ps(y + 32, vf2);
    _mm512_storeu_ps(y + 48, vf3);
    y += 64;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(x);
    x += 16;

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

    _mm512_storeu_ps(y, vf);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
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

    _mm512_mask_storeu_ps(y, vmask, vf);
  }
}

void xnn_f32_vabs_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_abs_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m512i vnonsign_mask = _mm512_set1_epi32((int) params->avx512.nonsign_mask);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512i vx0123456789ABCDEF = _mm512_loadu_si512(x);
    x += 16;

    const __m512i vy0123456789ABCDEF = _mm512_and_epi32(vx0123456789ABCDEF, vnonsign_mask);

    _mm512_storeu_si512(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512i vx = _mm512_maskz_loadu_epi32(vmask, x);
    const __m512i vy = _mm512_and_epi32(vx, vnonsign_mask);
    _mm512_mask_storeu_epi32(y, vmask, vy);
  }
}

void xnn_f32_vneg_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_neg_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512.sign_mask);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512i vx0123456789ABCDEF = _mm512_loadu_si512(x);
    x += 16;

    const __m512i vy0123456789ABCDEF = _mm512_xor_epi32(vx0123456789ABCDEF, vsign_mask);

    _mm512_storeu_si512(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512i vx = _mm512_maskz_loadu_epi32(vmask, x);
    const __m512i vy = _mm512_xor_epi32(vx, vsign_mask);
    _mm512_mask_storeu_epi32(y, vmask, vy);
  }
}

void xnn_f32_vsqr_ukernel__avx512f_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vy0123456789ABCDEF = _mm512_mul_ps(vx0123456789ABCDEF, vx0123456789ABCDEF);

    _mm512_storeu_ps(y, vy0123456789ABCDEF);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    const __m512 vy = _mm512_mul_ps(vx, vx);
    _mm512_mask_storeu_ps(y, vmask, vy);
  }
}
