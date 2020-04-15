// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/up-avx512.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmax = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.max));
  const __m512 vmin = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.min));
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    assert(i1 != NULL);
    const float* i2 = input[2];
    assert(i2 != NULL);
    const float* i3 = input[3];
    assert(i3 != NULL);
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 32; c -= 32) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);
      __m512 vaccGHIJKLMNOPQRSTUVp0 = _mm512_load_ps(w + 16);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      const __m512 vi0xGHIJKLMNOPQRSTUV = _mm512_loadu_ps(i0 + 16);
      i0 += 32;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 32);
      const __m512 vk0xGHIJKLMNOPQRSTUV = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);
      vaccGHIJKLMNOPQRSTUVp0 = _mm512_fmadd_ps(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV, vaccGHIJKLMNOPQRSTUVp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      const __m512 vi1xGHIJKLMNOPQRSTUV = _mm512_loadu_ps(i1 + 16);
      i1 += 32;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 64);
      const __m512 vk1xGHIJKLMNOPQRSTUV = _mm512_load_ps(w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);
      vaccGHIJKLMNOPQRSTUVp0 = _mm512_fmadd_ps(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV, vaccGHIJKLMNOPQRSTUVp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      const __m512 vi2xGHIJKLMNOPQRSTUV = _mm512_loadu_ps(i2 + 16);
      i2 += 32;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 96);
      const __m512 vk2xGHIJKLMNOPQRSTUV = _mm512_load_ps(w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);
      vaccGHIJKLMNOPQRSTUVp0 = _mm512_fmadd_ps(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV, vaccGHIJKLMNOPQRSTUVp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_loadu_ps(i3);
      const __m512 vi3xGHIJKLMNOPQRSTUV = _mm512_loadu_ps(i3 + 16);
      i3 += 32;

      const __m512 vk3x0123456789ABCDEF = _mm512_load_ps(w + 128);
      const __m512 vk3xGHIJKLMNOPQRSTUV = _mm512_load_ps(w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);
      vaccGHIJKLMNOPQRSTUVp0 = _mm512_fmadd_ps(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV, vaccGHIJKLMNOPQRSTUVp0);

      w += 160;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      __m512 vaccGHIJKLMNOPQRSTUV = _mm512_max_ps(vaccGHIJKLMNOPQRSTUVp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);
      vaccGHIJKLMNOPQRSTUV = _mm512_min_ps(vaccGHIJKLMNOPQRSTUV, vmax);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      _mm512_storeu_ps(output + 16, vaccGHIJKLMNOPQRSTUV);
      output += 32;
    }
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);

      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_loadu_ps(i3);
      i3 += 16;

      const __m512 vk3x0123456789ABCDEF = _mm512_load_ps(w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 16;


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
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i3);
      const __m512 vk3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vacc0123456789ABCDEFp0, vmin);
      vacc0123456789ABCDEF = _mm512_min_ps(vacc0123456789ABCDEF, vmax);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
