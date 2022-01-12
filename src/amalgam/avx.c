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
#include <xnnpack/lut.h>
#include <xnnpack/math.h>
#include <xnnpack/prelu.h>
#include <xnnpack/vaddsub.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vcvt.h>
#include <xnnpack/vmul.h>
#include <xnnpack/vunary.h>


void xnn_f16_f32_vcvt_ukernel__avx_int16_x16(
    size_t n,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vsign_mask = _mm_load_si128((const __m128i*) params->sse_int16.sign_mask);
  const __m128i vexp_offset = _mm_load_si128((const __m128i*) params->sse_int16.exp_offset);
  const __m128 vexp_scale = _mm_load_ps(params->sse_int16.exp_scale);
  const __m128i vmagic_mask = _mm_load_si128((const __m128i*) params->sse_int16.magic_mask);
  const __m128 vmagic_bias = _mm_load_ps(params->sse_int16.magic_bias);
  const __m128i vdenorm_cutoff = _mm_load_si128((const __m128i*) params->sse_int16.denorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    const __m128i vh0 = _mm_loadu_si128((const __m128i*) i);
    const __m128i vh1 = _mm_loadu_si128((const __m128i*) (i + 8));
    i += 16;

    const __m128i vsign0 = _mm_and_si128(vh0, vsign_mask);
    const __m128i vsign1 = _mm_and_si128(vh1, vsign_mask);

    const __m128i vnonsign0 = _mm_xor_si128(vh0, vsign0);
    const __m128i vnonsign1 = _mm_xor_si128(vh1, vsign1);

    const __m128i vprenorm0 = _mm_slli_epi16(vnonsign0, 13);
    const __m128i vprenorm1 = _mm_add_epi16(_mm_srli_epi16(vnonsign0, 3), vexp_offset);
    const __m128i vprenorm2 = _mm_slli_epi16(vnonsign1, 13);
    const __m128i vprenorm3 = _mm_add_epi16(_mm_srli_epi16(vnonsign1, 3), vexp_offset);

    const __m128i vnorm0 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm0, vprenorm1)), vexp_scale));
    const __m128i vnorm1 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm0, vprenorm1)), vexp_scale));
    const __m128i vnorm2 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm2, vprenorm3)), vexp_scale));
    const __m128i vnorm3 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm2, vprenorm3)), vexp_scale));

    const __m128i vdenorm0 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign0, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm1 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign0, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm2 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign1, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm3 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign1, vmagic_mask)), vmagic_bias));

    const __m128i vmask0 = _mm_cmpgt_epi16(vnonsign0, vdenorm_cutoff);
    const __m128i vmask1 = _mm_cmpgt_epi16(vnonsign1, vdenorm_cutoff);

    const __m128i vf0 = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign0),
      _mm_blendv_epi8(vdenorm0, vnorm0, _mm_cvtepi16_epi32(vmask0)));
    const __m128i vf1 = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign0),
      _mm_blendv_epi8(vdenorm1, vnorm1, _mm_unpackhi_epi16(vmask0, vmask0)));
    const __m128i vf2 = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign1),
      _mm_blendv_epi8(vdenorm2, vnorm2, _mm_cvtepi16_epi32(vmask1)));
    const __m128i vf3 = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign1),
      _mm_blendv_epi8(vdenorm3, vnorm3, _mm_unpackhi_epi16(vmask1, vmask1)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf0));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf1));
    _mm_storeu_ps(output + 8, _mm_castsi128_ps(vf2));
    _mm_storeu_ps(output + 12, _mm_castsi128_ps(vf3));
    output += 16;
  }
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    const __m128i vh = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m128i vsign = _mm_and_si128(vh, vsign_mask);

    const __m128i vnonsign = _mm_xor_si128(vh, vsign);

    const __m128i vprenorm_lo = _mm_slli_epi16(vnonsign, 13);
    const __m128i vprenorm_hi = _mm_add_epi16(_mm_srli_epi16(vnonsign, 3), vexp_offset);

    const __m128i vnorm_lo = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));
    const __m128i vnorm_hi = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));

    const __m128i vdenorm_lo = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm_hi = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign, vmagic_mask)), vmagic_bias));

    const __m128i vmask = _mm_cmpgt_epi16(vnonsign, vdenorm_cutoff);

    const __m128i vf_lo = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_blendv_epi8(vdenorm_lo, vnorm_lo, _mm_cvtepi16_epi32(vmask)));

    const __m128i vf_hi = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
      _mm_blendv_epi8(vdenorm_hi, vnorm_hi, _mm_unpackhi_epi16(vmask, vmask)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf_lo));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf_hi));
    output += 8;
  }
  if XNN_UNPREDICTABLE(n != 0) {
    const __m128i vh = _mm_loadu_si128((const __m128i*) i);

    const __m128i vsign = _mm_and_si128(vh, vsign_mask);

    const __m128i vnonsign = _mm_xor_si128(vh, vsign);

    const __m128i vprenorm_lo = _mm_slli_epi16(vnonsign, 13);
    const __m128i vprenorm_hi = _mm_add_epi16(_mm_srli_epi16(vnonsign, 3), vexp_offset);

    const __m128i vnorm_lo = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));
    const __m128i vnorm_hi = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));

    const __m128i vdenorm_lo = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm_hi = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign, vmagic_mask)), vmagic_bias));

    const __m128i vmask = _mm_cmpgt_epi16(vnonsign, vdenorm_cutoff);

    __m128i vf = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_blendv_epi8(vdenorm_lo, vnorm_lo, _mm_cvtepi16_epi32(vmask)));

    if (n & (4 * sizeof(uint16_t))) {
      _mm_storeu_ps(output, _mm_castsi128_ps(vf));
      output += 4;

      vf = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
        _mm_blendv_epi8(vdenorm_hi, vnorm_hi, _mm_unpackhi_epi16(vmask, vmask)));
    }
    if (n & (2 * sizeof(uint16_t))) {
      _mm_storel_pi((__m64*) output, _mm_castsi128_ps(vf));
      output += 2;

      vf = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(vf), _mm_castsi128_ps(vf)));
    }
    if (n & (1 * sizeof(uint16_t))) {
      _mm_store_ss(output, _mm_castsi128_ps(vf));
    }
  }
}

void xnn_f32_dwconv_minmax_ukernel_up16x3__avx(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
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
      __m256 vacc01234567p0 = _mm256_load_ps(w);
      __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi2x89ABCDEF, vk2x89ABCDEF));

      w += 64;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm256_storeu_ps(output, vacc01234567);
      _mm256_storeu_ps(output + 8, vacc89ABCDEF);
      output += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &params->avx.mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_up16x4__avx(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
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
      __m256 vacc01234567p0 = _mm256_load_ps(w);
      __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi3x89ABCDEF, vk3x89ABCDEF));

      w += 80;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm256_storeu_ps(output, vacc01234567);
      _mm256_storeu_ps(output + 8, vacc89ABCDEF);
      output += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &params->avx.mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_up16x9__avx(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
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
      __m256 vacc01234567p0 = _mm256_load_ps(w);
      __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi3x89ABCDEF, vk3x89ABCDEF));

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      const __m256 vi4x89ABCDEF = _mm256_loadu_ps(i4 + 8);
      i4 += 16;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      const __m256 vk4x89ABCDEF = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi4x89ABCDEF, vk4x89ABCDEF));

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      const __m256 vi5x89ABCDEF = _mm256_loadu_ps(i5 + 8);
      i5 += 16;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      const __m256 vk5x89ABCDEF = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi5x01234567, vk5x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi5x89ABCDEF, vk5x89ABCDEF));

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      const __m256 vi6x89ABCDEF = _mm256_loadu_ps(i6 + 8);
      i6 += 16;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      const __m256 vk6x89ABCDEF = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi6x89ABCDEF, vk6x89ABCDEF));

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      const __m256 vi7x89ABCDEF = _mm256_loadu_ps(i7 + 8);
      i7 += 16;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      const __m256 vk7x89ABCDEF = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi7x01234567, vk7x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi7x89ABCDEF, vk7x89ABCDEF));

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      const __m256 vi8x89ABCDEF = _mm256_loadu_ps(i8 + 8);
      i8 += 16;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      const __m256 vk8x89ABCDEF = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi8x01234567, vk8x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi8x89ABCDEF, vk8x89ABCDEF));

      w += 160;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm256_storeu_ps(output, vacc01234567);
      _mm256_storeu_ps(output + 8, vacc89ABCDEF);
      output += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi5x01234567, vk5x01234567));

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi7x01234567, vk7x01234567));

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi8x01234567, vk8x01234567));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &params->avx.mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));

      const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

      const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);
      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi5x01234567, vk5x01234567));

      const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);
      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

      const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);
      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi7x01234567, vk7x01234567));

      const __m256 vi8x01234567 = _mm256_maskload_ps(i8, vmask);
      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi8x01234567, vk8x01234567));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_up8x25__avx(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
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
      __m256 vacc01234567p0 = _mm256_load_ps(w);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi5x01234567, vk5x01234567));

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi7x01234567, vk7x01234567));

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi8x01234567, vk8x01234567));

      const __m256 vi9x01234567 = _mm256_loadu_ps(i9);
      i9 += 8;

      const __m256 vk9x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi9x01234567, vk9x01234567));

      const __m256 vi10x01234567 = _mm256_loadu_ps(i10);
      i10 += 8;

      const __m256 vk10x01234567 = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi10x01234567, vk10x01234567));

      const __m256 vi11x01234567 = _mm256_loadu_ps(i11);
      i11 += 8;

      const __m256 vk11x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi11x01234567, vk11x01234567));

      const __m256 vi12x01234567 = _mm256_loadu_ps(i12);
      i12 += 8;

      const __m256 vk12x01234567 = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi12x01234567, vk12x01234567));

      const __m256 vi13x01234567 = _mm256_loadu_ps(i13);
      i13 += 8;

      const __m256 vk13x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi13x01234567, vk13x01234567));

      const __m256 vi14x01234567 = _mm256_loadu_ps(i14);
      i14 += 8;

      const __m256 vk14x01234567 = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi14x01234567, vk14x01234567));

      const __m256 vi15x01234567 = _mm256_loadu_ps(i15);
      i15 += 8;

      const __m256 vk15x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi15x01234567, vk15x01234567));

      const __m256 vi16x01234567 = _mm256_loadu_ps(i16);
      i16 += 8;

      const __m256 vk16x01234567 = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi16x01234567, vk16x01234567));

      const __m256 vi17x01234567 = _mm256_loadu_ps(i17);
      i17 += 8;

      const __m256 vk17x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi17x01234567, vk17x01234567));

      const __m256 vi18x01234567 = _mm256_loadu_ps(i18);
      i18 += 8;

      const __m256 vk18x01234567 = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi18x01234567, vk18x01234567));

      const __m256 vi19x01234567 = _mm256_loadu_ps(i19);
      i19 += 8;

      const __m256 vk19x01234567 = _mm256_load_ps(w + 160);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi19x01234567, vk19x01234567));

      const __m256 vi20x01234567 = _mm256_loadu_ps(i20);
      i20 += 8;

      const __m256 vk20x01234567 = _mm256_load_ps(w + 168);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi20x01234567, vk20x01234567));

      const __m256 vi21x01234567 = _mm256_loadu_ps(i21);
      i21 += 8;

      const __m256 vk21x01234567 = _mm256_load_ps(w + 176);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi21x01234567, vk21x01234567));

      const __m256 vi22x01234567 = _mm256_loadu_ps(i22);
      i22 += 8;

      const __m256 vk22x01234567 = _mm256_load_ps(w + 184);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi22x01234567, vk22x01234567));

      const __m256 vi23x01234567 = _mm256_loadu_ps(i23);
      i23 += 8;

      const __m256 vk23x01234567 = _mm256_load_ps(w + 192);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi23x01234567, vk23x01234567));

      const __m256 vi24x01234567 = _mm256_loadu_ps(i24);
      i24 += 8;

      const __m256 vk24x01234567 = _mm256_load_ps(w + 200);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi24x01234567, vk24x01234567));

      w += 208;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &params->avx.mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));

      const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
      const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

      const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);
      const __m256 vk5x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi5x01234567, vk5x01234567));

      const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);
      const __m256 vk6x01234567 = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

      const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);
      const __m256 vk7x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi7x01234567, vk7x01234567));

      const __m256 vi8x01234567 = _mm256_maskload_ps(i8, vmask);
      const __m256 vk8x01234567 = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi8x01234567, vk8x01234567));

      const __m256 vi9x01234567 = _mm256_maskload_ps(i9, vmask);
      const __m256 vk9x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi9x01234567, vk9x01234567));

      const __m256 vi10x01234567 = _mm256_maskload_ps(i10, vmask);
      const __m256 vk10x01234567 = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi10x01234567, vk10x01234567));

      const __m256 vi11x01234567 = _mm256_maskload_ps(i11, vmask);
      const __m256 vk11x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi11x01234567, vk11x01234567));

      const __m256 vi12x01234567 = _mm256_maskload_ps(i12, vmask);
      const __m256 vk12x01234567 = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi12x01234567, vk12x01234567));

      const __m256 vi13x01234567 = _mm256_maskload_ps(i13, vmask);
      const __m256 vk13x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi13x01234567, vk13x01234567));

      const __m256 vi14x01234567 = _mm256_maskload_ps(i14, vmask);
      const __m256 vk14x01234567 = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi14x01234567, vk14x01234567));

      const __m256 vi15x01234567 = _mm256_maskload_ps(i15, vmask);
      const __m256 vk15x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi15x01234567, vk15x01234567));

      const __m256 vi16x01234567 = _mm256_maskload_ps(i16, vmask);
      const __m256 vk16x01234567 = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi16x01234567, vk16x01234567));

      const __m256 vi17x01234567 = _mm256_maskload_ps(i17, vmask);
      const __m256 vk17x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi17x01234567, vk17x01234567));

      const __m256 vi18x01234567 = _mm256_maskload_ps(i18, vmask);
      const __m256 vk18x01234567 = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi18x01234567, vk18x01234567));

      const __m256 vi19x01234567 = _mm256_maskload_ps(i19, vmask);
      const __m256 vk19x01234567 = _mm256_load_ps(w + 160);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi19x01234567, vk19x01234567));

      const __m256 vi20x01234567 = _mm256_maskload_ps(i20, vmask);
      const __m256 vk20x01234567 = _mm256_load_ps(w + 168);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi20x01234567, vk20x01234567));

      const __m256 vi21x01234567 = _mm256_maskload_ps(i21, vmask);
      const __m256 vk21x01234567 = _mm256_load_ps(w + 176);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi21x01234567, vk21x01234567));

      const __m256 vi22x01234567 = _mm256_maskload_ps(i22, vmask);
      const __m256 vk22x01234567 = _mm256_load_ps(w + 184);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi22x01234567, vk22x01234567));

      const __m256 vi23x01234567 = _mm256_maskload_ps(i23, vmask);
      const __m256 vk23x01234567 = _mm256_load_ps(w + 192);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi23x01234567, vk23x01234567));

      const __m256 vi24x01234567 = _mm256_maskload_ps(i24, vmask);
      const __m256 vk24x01234567 = _mm256_load_ps(w + 200);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi24x01234567, vk24x01234567));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_f16_vcvt_ukernel__avx_x24(
    size_t n,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vnonsign_mask = _mm_load_ps((const float*) params->sse2.nonsign_mask);
  const __m128i vexp_bias = _mm_load_si128((const __m128i*) params->sse2.exp_bias);
  const __m128 vscale_to_inf = _mm_load_ps(params->sse2.scale_to_inf);
  const __m128i vexpw_max = _mm_load_si128((const __m128i*) params->sse2.expw_max);
  const __m128 vscale_to_zero = _mm_load_ps(params->sse2.scale_to_zero);
  const __m128i vbias_min = _mm_load_si128((const __m128i*) params->sse2.bias_min);
  const __m128i vmanth_mask = _mm_load_si128((const __m128i*) params->sse2.manth_mask);
  const __m128i vexph_mask = _mm_load_si128((const __m128i*) params->sse2.exph_mask);
  const __m128i vnanh = _mm_load_si128((const __m128i*) params->sse2.nanh);

  uint16_t* o = (uint16_t*) output;
  for (; n >= 24 * sizeof(float); n -= 24 * sizeof(float)) {
    const __m128 vx0 = _mm_loadu_ps(input);
    const __m128 vx1 = _mm_loadu_ps(input + 4);
    const __m128 vx2 = _mm_loadu_ps(input + 8);
    const __m128 vx3 = _mm_loadu_ps(input + 12);
    const __m128 vx4 = _mm_loadu_ps(input + 16);
    const __m128 vx5 = _mm_loadu_ps(input + 20);
    input += 24;

    const __m128 vabsx0 = _mm_and_ps(vx0, vnonsign_mask);
    const __m128 vabsx1 = _mm_and_ps(vx1, vnonsign_mask);
    const __m128 vabsx2 = _mm_and_ps(vx2, vnonsign_mask);
    const __m128 vabsx3 = _mm_and_ps(vx3, vnonsign_mask);
    const __m128 vabsx4 = _mm_and_ps(vx4, vnonsign_mask);
    const __m128 vabsx5 = _mm_and_ps(vx5, vnonsign_mask);

    const __m128 vsignx0 = _mm_xor_ps(vx0, vabsx0);
    const __m128 vsignx1 = _mm_xor_ps(vx1, vabsx1);
    const __m128 vsignx2 = _mm_xor_ps(vx2, vabsx2);
    const __m128 vsignx3 = _mm_xor_ps(vx3, vabsx3);
    const __m128 vsignx4 = _mm_xor_ps(vx4, vabsx4);
    const __m128 vsignx5 = _mm_xor_ps(vx5, vabsx5);

    __m128i vbias0 = _mm_add_epi32(_mm_castps_si128(vabsx0), vexp_bias);
    __m128i vbias1 = _mm_add_epi32(_mm_castps_si128(vabsx1), vexp_bias);
    __m128i vbias2 = _mm_add_epi32(_mm_castps_si128(vabsx2), vexp_bias);
    __m128i vbias3 = _mm_add_epi32(_mm_castps_si128(vabsx3), vexp_bias);
    __m128i vbias4 = _mm_add_epi32(_mm_castps_si128(vabsx4), vexp_bias);
    __m128i vbias5 = _mm_add_epi32(_mm_castps_si128(vabsx5), vexp_bias);

    __m128 vf0 = _mm_mul_ps(vabsx0, vscale_to_inf);
    __m128 vf1 = _mm_mul_ps(vabsx1, vscale_to_inf);
    __m128 vf2 = _mm_mul_ps(vabsx2, vscale_to_inf);
    __m128 vf3 = _mm_mul_ps(vabsx3, vscale_to_inf);
    __m128 vf4 = _mm_mul_ps(vabsx4, vscale_to_inf);
    __m128 vf5 = _mm_mul_ps(vabsx5, vscale_to_inf);

    const __m128i vnanmaskw0 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx0), vexpw_max);
    const __m128i vnanmaskw1 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx1), vexpw_max);
    const __m128i vnanmaskw2 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx2), vexpw_max);
    const __m128i vnanmaskw3 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx3), vexpw_max);
    const __m128i vnanmaskw4 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx4), vexpw_max);
    const __m128i vnanmaskw5 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx5), vexpw_max);

    vbias0 = _mm_and_si128(vbias0, vexpw_max);
    vbias1 = _mm_and_si128(vbias1, vexpw_max);
    vbias2 = _mm_and_si128(vbias2, vexpw_max);
    vbias3 = _mm_and_si128(vbias3, vexpw_max);
    vbias4 = _mm_and_si128(vbias4, vexpw_max);
    vbias5 = _mm_and_si128(vbias5, vexpw_max);

    vf0 = _mm_mul_ps(vf0, vscale_to_zero);
    vf1 = _mm_mul_ps(vf1, vscale_to_zero);
    vf2 = _mm_mul_ps(vf2, vscale_to_zero);
    vf3 = _mm_mul_ps(vf3, vscale_to_zero);
    vf4 = _mm_mul_ps(vf4, vscale_to_zero);
    vf5 = _mm_mul_ps(vf5, vscale_to_zero);

    const __m128i vnanmaskh0 = _mm_packs_epi32(vnanmaskw0, vnanmaskw1);
    const __m128i vnanmaskh1 = _mm_packs_epi32(vnanmaskw2, vnanmaskw3);
    const __m128i vnanmaskh2 = _mm_packs_epi32(vnanmaskw4, vnanmaskw5);

    const __m128i vsignh0 = _mm_packs_epi32(_mm_castps_si128(vsignx0), _mm_castps_si128(vsignx1));
    const __m128i vsignh1 = _mm_packs_epi32(_mm_castps_si128(vsignx2), _mm_castps_si128(vsignx3));
    const __m128i vsignh2 = _mm_packs_epi32(_mm_castps_si128(vsignx4), _mm_castps_si128(vsignx5));

    vbias0 = _mm_max_epi16(vbias0, vbias_min);
    vbias1 = _mm_max_epi16(vbias1, vbias_min);
    vbias2 = _mm_max_epi16(vbias2, vbias_min);
    vbias3 = _mm_max_epi16(vbias3, vbias_min);
    vbias4 = _mm_max_epi16(vbias4, vbias_min);
    vbias5 = _mm_max_epi16(vbias5, vbias_min);


    vf0 = _mm_add_ps(vf0, _mm_castsi128_ps(vbias0));
    vf1 = _mm_add_ps(vf1, _mm_castsi128_ps(vbias1));
    vf2 = _mm_add_ps(vf2, _mm_castsi128_ps(vbias2));
    vf3 = _mm_add_ps(vf3, _mm_castsi128_ps(vbias3));
    vf4 = _mm_add_ps(vf4, _mm_castsi128_ps(vbias4));
    vf5 = _mm_add_ps(vf5, _mm_castsi128_ps(vbias5));


    __m128i vexpw0 = _mm_srli_epi32(_mm_castps_si128(vf0), 13);
    __m128i vexpw1 = _mm_srli_epi32(_mm_castps_si128(vf1), 13);
    __m128i vexpw2 = _mm_srli_epi32(_mm_castps_si128(vf2), 13);
    __m128i vexpw3 = _mm_srli_epi32(_mm_castps_si128(vf3), 13);
    __m128i vexpw4 = _mm_srli_epi32(_mm_castps_si128(vf4), 13);
    __m128i vexpw5 = _mm_srli_epi32(_mm_castps_si128(vf5), 13);

    const __m128i vmantw0 = _mm_and_si128(_mm_castps_si128(vf0), vmanth_mask);
    const __m128i vmantw1 = _mm_and_si128(_mm_castps_si128(vf1), vmanth_mask);
    const __m128i vmantw2 = _mm_and_si128(_mm_castps_si128(vf2), vmanth_mask);
    const __m128i vmantw3 = _mm_and_si128(_mm_castps_si128(vf3), vmanth_mask);
    const __m128i vmantw4 = _mm_and_si128(_mm_castps_si128(vf4), vmanth_mask);
    const __m128i vmantw5 = _mm_and_si128(_mm_castps_si128(vf5), vmanth_mask);

    vexpw0 = _mm_and_si128(vexpw0, vexph_mask);
    vexpw1 = _mm_and_si128(vexpw1, vexph_mask);
    vexpw2 = _mm_and_si128(vexpw2, vexph_mask);
    vexpw3 = _mm_and_si128(vexpw3, vexph_mask);
    vexpw4 = _mm_and_si128(vexpw4, vexph_mask);
    vexpw5 = _mm_and_si128(vexpw5, vexph_mask);

    const __m128i vnonsignw0 = _mm_add_epi32(vmantw0, vexpw0);
    const __m128i vnonsignw1 = _mm_add_epi32(vmantw1, vexpw1);
    const __m128i vnonsignw2 = _mm_add_epi32(vmantw2, vexpw2);
    const __m128i vnonsignw3 = _mm_add_epi32(vmantw3, vexpw3);
    const __m128i vnonsignw4 = _mm_add_epi32(vmantw4, vexpw4);
    const __m128i vnonsignw5 = _mm_add_epi32(vmantw5, vexpw5);

    const __m128i vnonsignh0 = _mm_packs_epi32(vnonsignw0, vnonsignw1);
    const __m128i vnonsignh1 = _mm_packs_epi32(vnonsignw2, vnonsignw3);
    const __m128i vnonsignh2 = _mm_packs_epi32(vnonsignw4, vnonsignw5);

    const __m128i vabsh0 = _mm_blendv_epi8(vnonsignh0, vnanh, vnanmaskh0);
    const __m128i vabsh1 = _mm_blendv_epi8(vnonsignh1, vnanh, vnanmaskh1);
    const __m128i vabsh2 = _mm_blendv_epi8(vnonsignh2, vnanh, vnanmaskh2);

    const __m128i vh0 = _mm_or_si128(vabsh0, vsignh0);
    const __m128i vh1 = _mm_or_si128(vabsh1, vsignh1);
    const __m128i vh2 = _mm_or_si128(vabsh2, vsignh2);

    _mm_storeu_si128((__m128i*) o, vh0);
    _mm_storeu_si128((__m128i*) (o + 8), vh1);
    _mm_storeu_si128((__m128i*) (o + 16), vh2);
    o += 24;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m128 vx_lo = _mm_loadu_ps(input);
    const __m128 vx_hi = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vabsx_lo = _mm_and_ps(vx_lo, vnonsign_mask);
    const __m128 vabsx_hi = _mm_and_ps(vx_hi, vnonsign_mask);

    const __m128 vsignx_lo = _mm_xor_ps(vx_lo, vabsx_lo);
    const __m128 vsignx_hi = _mm_xor_ps(vx_hi, vabsx_hi);
    __m128i vbias_lo = _mm_add_epi32(_mm_castps_si128(vabsx_lo), vexp_bias);
    __m128i vbias_hi = _mm_add_epi32(_mm_castps_si128(vabsx_hi), vexp_bias);
    __m128 vf_lo = _mm_mul_ps(vabsx_lo, vscale_to_inf);
    __m128 vf_hi = _mm_mul_ps(vabsx_hi, vscale_to_inf);
    const __m128i vnanmaskw_lo = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_lo), vexpw_max);
    const __m128i vnanmaskw_hi = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_hi), vexpw_max);

    vbias_lo = _mm_and_si128(vbias_lo, vexpw_max);
    vbias_hi = _mm_and_si128(vbias_hi, vexpw_max);
    vf_lo = _mm_mul_ps(vf_lo, vscale_to_zero);
    vf_hi = _mm_mul_ps(vf_hi, vscale_to_zero);
    const __m128i vnanmaskh = _mm_packs_epi32(vnanmaskw_lo, vnanmaskw_hi);
    const __m128i vsignh = _mm_packs_epi32(_mm_castps_si128(vsignx_lo), _mm_castps_si128(vsignx_hi));

    vbias_lo = _mm_max_epi16(vbias_lo, vbias_min);
    vbias_hi = _mm_max_epi16(vbias_hi, vbias_min);

    vf_lo = _mm_add_ps(vf_lo, _mm_castsi128_ps(vbias_lo));
    vf_hi = _mm_add_ps(vf_hi, _mm_castsi128_ps(vbias_hi));

    __m128i vexpw_lo = _mm_srli_epi32(_mm_castps_si128(vf_lo), 13);
    __m128i vexpw_hi = _mm_srli_epi32(_mm_castps_si128(vf_hi), 13);
    const __m128i vmantw_lo = _mm_and_si128(_mm_castps_si128(vf_lo), vmanth_mask);
    const __m128i vmantw_hi = _mm_and_si128(_mm_castps_si128(vf_hi), vmanth_mask);

    vexpw_lo = _mm_and_si128(vexpw_lo, vexph_mask);
    vexpw_hi = _mm_and_si128(vexpw_hi, vexph_mask);

    const __m128i vnonsignw_lo = _mm_add_epi32(vmantw_lo, vexpw_lo);
    const __m128i vnonsignw_hi = _mm_add_epi32(vmantw_hi, vexpw_hi);

    const __m128i vnonsignh = _mm_packs_epi32(vnonsignw_lo, vnonsignw_hi);

    const __m128i vabsh = _mm_blendv_epi8(vnonsignh, vnanh, vnanmaskh);

    const __m128i vh = _mm_or_si128(vabsh, vsignh);

    _mm_storeu_si128((__m128i*) o, vh);
    o += 8;
  }
  if XNN_UNPREDICTABLE(n != 0) {
    const __m128 vx_lo = _mm_loadu_ps(input);
    const float* input_hi = (const float*) ((uintptr_t) input + (n & (4 * sizeof(float))));
    const __m128 vx_hi = _mm_loadu_ps(input_hi);

    const __m128 vabsx_lo = _mm_and_ps(vx_lo, vnonsign_mask);
    const __m128 vabsx_hi = _mm_and_ps(vx_hi, vnonsign_mask);

    const __m128 vsignx_lo = _mm_xor_ps(vx_lo, vabsx_lo);
    const __m128 vsignx_hi = _mm_xor_ps(vx_hi, vabsx_hi);
    __m128i vbias_lo = _mm_add_epi32(_mm_castps_si128(vabsx_lo), vexp_bias);
    __m128i vbias_hi = _mm_add_epi32(_mm_castps_si128(vabsx_hi), vexp_bias);
    __m128 vf_lo = _mm_mul_ps(vabsx_lo, vscale_to_inf);
    __m128 vf_hi = _mm_mul_ps(vabsx_hi, vscale_to_inf);
    const __m128i vnanmaskw_lo = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_lo), vexpw_max);
    const __m128i vnanmaskw_hi = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_hi), vexpw_max);

    vbias_lo = _mm_and_si128(vbias_lo, vexpw_max);
    vbias_hi = _mm_and_si128(vbias_hi, vexpw_max);
    vf_lo = _mm_mul_ps(vf_lo, vscale_to_zero);
    vf_hi = _mm_mul_ps(vf_hi, vscale_to_zero);
    const __m128i vnanmaskh = _mm_packs_epi32(vnanmaskw_lo, vnanmaskw_hi);
    const __m128i vsignh = _mm_packs_epi32(_mm_castps_si128(vsignx_lo), _mm_castps_si128(vsignx_hi));

    vbias_lo = _mm_max_epi16(vbias_lo, vbias_min);
    vbias_hi = _mm_max_epi16(vbias_hi, vbias_min);

    vf_lo = _mm_add_ps(vf_lo, _mm_castsi128_ps(vbias_lo));
    vf_hi = _mm_add_ps(vf_hi, _mm_castsi128_ps(vbias_hi));

    __m128i vexpw_lo = _mm_srli_epi32(_mm_castps_si128(vf_lo), 13);
    __m128i vexpw_hi = _mm_srli_epi32(_mm_castps_si128(vf_hi), 13);
    const __m128i vmantw_lo = _mm_and_si128(_mm_castps_si128(vf_lo), vmanth_mask);
    const __m128i vmantw_hi = _mm_and_si128(_mm_castps_si128(vf_hi), vmanth_mask);

    vexpw_lo = _mm_and_si128(vexpw_lo, vexph_mask);
    vexpw_hi = _mm_and_si128(vexpw_hi, vexph_mask);

    const __m128i vnonsignw_lo = _mm_add_epi32(vmantw_lo, vexpw_lo);
    const __m128i vnonsignw_hi = _mm_add_epi32(vmantw_hi, vexpw_hi);

    const __m128i vnonsignh = _mm_packs_epi32(vnonsignw_lo, vnonsignw_hi);

    const __m128i vabsh = _mm_blendv_epi8(vnonsignh, vnanh, vnanmaskh);

    __m128i vh = _mm_or_si128(vabsh, vsignh);

    if (n & (4 * sizeof(float))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (n & (2 * sizeof(float))) {
      *((uint32_t*) o) = (uint32_t) _mm_cvtsi128_si32(vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (n & (1 * sizeof(float))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast(
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
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    w += 16;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;

      const __m256 vb01234567 = _mm256_load_ps(w);
      const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
      w += 16;

      vacc0x01234567 = _mm256_add_ps(vacc0x01234567, _mm256_mul_ps(va0, vb01234567));
      vacc0x89ABCDEF = _mm256_add_ps(vacc0x89ABCDEF, _mm256_mul_ps(va0, vb89ABCDEF));

      k -= sizeof(float);
    } while (k != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast(
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
  assert(mr <= 5);
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

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc4x01234567 = vacc0x01234567;
    __m256 vacc4x89ABCDEF = vacc0x89ABCDEF;
    w += 16;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1 = _mm256_broadcast_ss(a1);
      a1 += 1;
      const __m256 va2 = _mm256_broadcast_ss(a2);
      a2 += 1;
      const __m256 va3 = _mm256_broadcast_ss(a3);
      a3 += 1;
      const __m256 va4 = _mm256_broadcast_ss(a4);
      a4 += 1;

      const __m256 vb01234567 = _mm256_load_ps(w);
      const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
      w += 16;

      vacc0x01234567 = _mm256_add_ps(vacc0x01234567, _mm256_mul_ps(va0, vb01234567));
      vacc1x01234567 = _mm256_add_ps(vacc1x01234567, _mm256_mul_ps(va1, vb01234567));
      vacc2x01234567 = _mm256_add_ps(vacc2x01234567, _mm256_mul_ps(va2, vb01234567));
      vacc3x01234567 = _mm256_add_ps(vacc3x01234567, _mm256_mul_ps(va3, vb01234567));
      vacc4x01234567 = _mm256_add_ps(vacc4x01234567, _mm256_mul_ps(va4, vb01234567));
      vacc0x89ABCDEF = _mm256_add_ps(vacc0x89ABCDEF, _mm256_mul_ps(va0, vb89ABCDEF));
      vacc1x89ABCDEF = _mm256_add_ps(vacc1x89ABCDEF, _mm256_mul_ps(va1, vb89ABCDEF));
      vacc2x89ABCDEF = _mm256_add_ps(vacc2x89ABCDEF, _mm256_mul_ps(va2, vb89ABCDEF));
      vacc3x89ABCDEF = _mm256_add_ps(vacc3x89ABCDEF, _mm256_mul_ps(va3, vb89ABCDEF));
      vacc4x89ABCDEF = _mm256_add_ps(vacc4x89ABCDEF, _mm256_mul_ps(va4, vb89ABCDEF));

      k -= sizeof(float);
    } while (k != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc1x01234567 = _mm256_max_ps(vacc1x01234567, vmin);
    vacc2x01234567 = _mm256_max_ps(vacc2x01234567, vmin);
    vacc3x01234567 = _mm256_max_ps(vacc3x01234567, vmin);
    vacc4x01234567 = _mm256_max_ps(vacc4x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = _mm256_max_ps(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = _mm256_max_ps(vacc2x89ABCDEF, vmin);
    vacc3x89ABCDEF = _mm256_max_ps(vacc3x89ABCDEF, vmin);
    vacc4x89ABCDEF = _mm256_max_ps(vacc4x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc1x01234567 = _mm256_min_ps(vacc1x01234567, vmax);
    vacc2x01234567 = _mm256_min_ps(vacc2x01234567, vmax);
    vacc3x01234567 = _mm256_min_ps(vacc3x01234567, vmax);
    vacc4x01234567 = _mm256_min_ps(vacc4x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = _mm256_min_ps(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = _mm256_min_ps(vacc2x89ABCDEF, vmax);
    vacc3x89ABCDEF = _mm256_min_ps(vacc3x89ABCDEF, vmax);
    vacc4x89ABCDEF = _mm256_min_ps(vacc4x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c4, vacc4x01234567);
      _mm256_storeu_ps(c4 + 8, vacc4x89ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c4, vacc4x01234567);
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc4x01234567 = vacc4x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c4 += 8;
        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c4, vacc4x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast(
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
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
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
        const __m256 vb01234567 = _mm256_load_ps(w);
        const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
        w += 16;

        const __m256 va0 = _mm256_broadcast_ss(a0);
        a0 += 1;

        vacc0x01234567 = _mm256_add_ps(vacc0x01234567, _mm256_mul_ps(va0, vb01234567));
        vacc0x89ABCDEF = _mm256_add_ps(vacc0x89ABCDEF, _mm256_mul_ps(va0, vb89ABCDEF));
        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_5x16__avx_broadcast(
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
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (5 * sizeof(void*)) == 0);
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

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc4x01234567 = vacc0x01234567;
    __m256 vacc4x89ABCDEF = vacc0x89ABCDEF;
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
      a += 5;

      size_t k = kc;
      do {
        const __m256 vb01234567 = _mm256_load_ps(w);
        const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
        w += 16;

        const __m256 va0 = _mm256_broadcast_ss(a0);
        a0 += 1;
        const __m256 va1 = _mm256_broadcast_ss(a1);
        a1 += 1;
        const __m256 va2 = _mm256_broadcast_ss(a2);
        a2 += 1;
        const __m256 va3 = _mm256_broadcast_ss(a3);
        a3 += 1;
        const __m256 va4 = _mm256_broadcast_ss(a4);
        a4 += 1;

        vacc0x01234567 = _mm256_add_ps(vacc0x01234567, _mm256_mul_ps(va0, vb01234567));
        vacc0x89ABCDEF = _mm256_add_ps(vacc0x89ABCDEF, _mm256_mul_ps(va0, vb89ABCDEF));
        vacc1x01234567 = _mm256_add_ps(vacc1x01234567, _mm256_mul_ps(va1, vb01234567));
        vacc1x89ABCDEF = _mm256_add_ps(vacc1x89ABCDEF, _mm256_mul_ps(va1, vb89ABCDEF));
        vacc2x01234567 = _mm256_add_ps(vacc2x01234567, _mm256_mul_ps(va2, vb01234567));
        vacc2x89ABCDEF = _mm256_add_ps(vacc2x89ABCDEF, _mm256_mul_ps(va2, vb89ABCDEF));
        vacc3x01234567 = _mm256_add_ps(vacc3x01234567, _mm256_mul_ps(va3, vb01234567));
        vacc3x89ABCDEF = _mm256_add_ps(vacc3x89ABCDEF, _mm256_mul_ps(va3, vb89ABCDEF));
        vacc4x01234567 = _mm256_add_ps(vacc4x01234567, _mm256_mul_ps(va4, vb01234567));
        vacc4x89ABCDEF = _mm256_add_ps(vacc4x89ABCDEF, _mm256_mul_ps(va4, vb89ABCDEF));
        k -= sizeof(float);
      } while (k != 0);
      p -= 5 * sizeof(void*);
    } while (p != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc1x01234567 = _mm256_max_ps(vacc1x01234567, vmin);
    vacc2x01234567 = _mm256_max_ps(vacc2x01234567, vmin);
    vacc3x01234567 = _mm256_max_ps(vacc3x01234567, vmin);
    vacc4x01234567 = _mm256_max_ps(vacc4x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = _mm256_max_ps(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = _mm256_max_ps(vacc2x89ABCDEF, vmin);
    vacc3x89ABCDEF = _mm256_max_ps(vacc3x89ABCDEF, vmin);
    vacc4x89ABCDEF = _mm256_max_ps(vacc4x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc1x01234567 = _mm256_min_ps(vacc1x01234567, vmax);
    vacc2x01234567 = _mm256_min_ps(vacc2x01234567, vmax);
    vacc3x01234567 = _mm256_min_ps(vacc3x01234567, vmax);
    vacc4x01234567 = _mm256_min_ps(vacc4x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = _mm256_min_ps(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = _mm256_min_ps(vacc2x89ABCDEF, vmax);
    vacc3x89ABCDEF = _mm256_min_ps(vacc3x89ABCDEF, vmax);
    vacc4x89ABCDEF = _mm256_min_ps(vacc4x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c4, vacc4x01234567);
      _mm256_storeu_ps(c4 + 8, vacc4x89ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c4, vacc4x01234567);
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc4x01234567 = vacc4x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c4 += 8;
        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c4, vacc4x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_f32_prelu_ukernel__avx_2x16(
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

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(float); c -= 16 * sizeof(float)) {
      const __m256 vw01234567 = _mm256_load_ps(w);
      const __m256 vw89ABCDEF = _mm256_load_ps(w + 8);
      w += 16;

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;
      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vprod0x01234567 = _mm256_mul_ps(vi0x01234567, vw01234567);
      const __m256 vprod0x89ABCDEF = _mm256_mul_ps(vi0x89ABCDEF, vw89ABCDEF);
      const __m256 vprod1x01234567 = _mm256_mul_ps(vi1x01234567, vw01234567);
      const __m256 vprod1x89ABCDEF = _mm256_mul_ps(vi1x89ABCDEF, vw89ABCDEF);

      const __m256 vacc0x01234567 = _mm256_blendv_ps(vi0x01234567, vprod0x01234567, vi0x01234567);
      const __m256 vacc0x89ABCDEF = _mm256_blendv_ps(vi0x89ABCDEF, vprod0x89ABCDEF, vi0x89ABCDEF);
      const __m256 vacc1x01234567 = _mm256_blendv_ps(vi1x01234567, vprod1x01234567, vi1x01234567);
      const __m256 vacc1x89ABCDEF = _mm256_blendv_ps(vi1x89ABCDEF, vprod1x89ABCDEF, vi1x89ABCDEF);

      _mm256_storeu_ps(o0, vacc0x01234567);
      _mm256_storeu_ps(o0 + 8, vacc0x89ABCDEF);
      o0 += 16;
      _mm256_storeu_ps(o1, vacc1x01234567);
      _mm256_storeu_ps(o1 + 8, vacc1x89ABCDEF);
      o1 += 16;
    }
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const __m256 vw = _mm256_load_ps(w);
      w += 8;

      const __m256 vi0 = _mm256_loadu_ps(i0);
      i0 += 8;
      const __m256 vi1 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vprod0 = _mm256_mul_ps(vi0, vw);
      const __m256 vprod1 = _mm256_mul_ps(vi1, vw);

      const __m256 vacc0 = _mm256_blendv_ps(vi0, vprod0, vi0);
      const __m256 vacc1 = _mm256_blendv_ps(vi1, vprod1, vi1);

      _mm256_storeu_ps(o0, vacc0);
      o0 += 8;
      _mm256_storeu_ps(o1, vacc1);
      o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1 * sizeof(float));
      assert(c <= 7 * sizeof(float));
      __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - c));

      const __m256 vw = _mm256_maskload_ps(w, vmask);

      const __m256 vi0 = _mm256_maskload_ps(i0, vmask);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const __m256 vi1 = _mm256_maskload_ps(i1, vmask);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const __m256 vprod0 = _mm256_mul_ps(vi0, vw);
      const __m256 vprod1 = _mm256_mul_ps(vi1, vw);

      __m256 vacc0 = _mm256_blendv_ps(vi0, vprod0, vi0);
      __m256 vacc1 = _mm256_blendv_ps(vi1, vprod1, vi1);

      __m128 vacc0_lo = _mm256_castps256_ps128(vacc0);
      __m128 vacc1_lo = _mm256_castps256_ps128(vacc1);
      if (c & (4 * sizeof(float))) {
        _mm_storeu_ps(o0, vacc0_lo);
        _mm_storeu_ps(o1, vacc1_lo);

        vacc0_lo = _mm256_extractf128_ps(vacc0, 1);
        vacc1_lo = _mm256_extractf128_ps(vacc1, 1);

        o0 += 4;
        o1 += 4;
      }
      if (c & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) o0, vacc0_lo);
        _mm_storel_pi((__m64*) o1, vacc1_lo);

        vacc0_lo = _mm_movehl_ps(vacc0_lo, vacc0_lo);
        vacc1_lo = _mm_movehl_ps(vacc1_lo, vacc1_lo);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        _mm_store_ss(o0, vacc0_lo);
        _mm_store_ss(o1, vacc1_lo);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_qs8_vcvt_ukernel__avx_x32(
    size_t n,
    const float* x,
    int8_t* y,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->avx.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->avx.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx.output_min);

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    __m256 vx01234567 = _mm256_loadu_ps(x);
    __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    __m256 vxGHIJKLMN = _mm256_loadu_ps(x + 16);
    __m256 vxOPQRSTUV = _mm256_loadu_ps(x + 24);
    x += 32;

    vx01234567 = _mm256_mul_ps(vx01234567, vscale);
    vx89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vscale);
    vxGHIJKLMN = _mm256_mul_ps(vxGHIJKLMN, vscale);
    vxOPQRSTUV = _mm256_mul_ps(vxOPQRSTUV, vscale);

    vx01234567 = _mm256_min_ps(vx01234567, voutput_max_less_zero_point);
    vx89ABCDEF = _mm256_min_ps(vx89ABCDEF, voutput_max_less_zero_point);
    vxGHIJKLMN = _mm256_min_ps(vxGHIJKLMN, voutput_max_less_zero_point);
    vxOPQRSTUV = _mm256_min_ps(vxOPQRSTUV, voutput_max_less_zero_point);

    const __m256i vacc01234567 = _mm256_cvtps_epi32(vx01234567);
    const __m256i vacc89ABCDEF = _mm256_cvtps_epi32(vx89ABCDEF);
    const __m256i vaccGHIJKLMN = _mm256_cvtps_epi32(vxGHIJKLMN);
    const __m256i vaccOPQRSTUV = _mm256_cvtps_epi32(vxOPQRSTUV);

    __m128i vy01234567 = _mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extractf128_si256(vacc01234567, 1));
    __m128i vy89ABCDEF = _mm_packs_epi32(_mm256_castsi256_si128(vacc89ABCDEF), _mm256_extractf128_si256(vacc89ABCDEF, 1));
    __m128i vyGHIJKLMN = _mm_packs_epi32(_mm256_castsi256_si128(vaccGHIJKLMN), _mm256_extractf128_si256(vaccGHIJKLMN, 1));
    __m128i vyOPQRSTUV = _mm_packs_epi32(_mm256_castsi256_si128(vaccOPQRSTUV), _mm256_extractf128_si256(vaccOPQRSTUV, 1));

    vy01234567 = _mm_adds_epi16(vy01234567, voutput_zero_point);
    vy89ABCDEF = _mm_adds_epi16(vy89ABCDEF, voutput_zero_point);
    vyGHIJKLMN = _mm_adds_epi16(vyGHIJKLMN, voutput_zero_point);
    vyOPQRSTUV = _mm_adds_epi16(vyOPQRSTUV, voutput_zero_point);

    __m128i vy0123456789ABCDEF = _mm_packs_epi16(vy01234567, vy89ABCDEF);
    __m128i vyGHIJKLMNOPQRSTUV = _mm_packs_epi16(vyGHIJKLMN, vyOPQRSTUV);

    vy0123456789ABCDEF = _mm_max_epi8(vy0123456789ABCDEF, voutput_min);
    vyGHIJKLMNOPQRSTUV = _mm_max_epi8(vyGHIJKLMNOPQRSTUV, voutput_min);

    _mm_storeu_si128((__m128i*) y, vy0123456789ABCDEF);
    _mm_storeu_si128((__m128i*) (y + 16), vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(x);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);
    x += 8;

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extractf128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, voutput_min);

    _mm_storel_epi64((__m128i*) y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    __m256 vx = _mm256_maskload_ps(x, vmask);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extractf128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, voutput_min);

    if (n & (4 * sizeof(float))) {
      *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vy);
      y += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    if (n & (2 * sizeof(float))) {
      *((uint16_t*) y) = (uint16_t) _mm_extract_epi16(vy, 0);
      y += 2;
      vy = _mm_srli_epi32(vy, 16);
    }
    if (n & (1 * sizeof(float))) {
      *y = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_f32_qu8_vcvt_ukernel__avx_x32(
    size_t n,
    const float* x,
    uint8_t* y,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->avx.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->avx.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx.output_min);

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    __m256 vx01234567 = _mm256_loadu_ps(x);
    __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    __m256 vxGHIJKLMN = _mm256_loadu_ps(x + 16);
    __m256 vxOPQRSTUV = _mm256_loadu_ps(x + 24);
    x += 32;

    vx01234567 = _mm256_mul_ps(vx01234567, vscale);
    vx89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vscale);
    vxGHIJKLMN = _mm256_mul_ps(vxGHIJKLMN, vscale);
    vxOPQRSTUV = _mm256_mul_ps(vxOPQRSTUV, vscale);

    vx01234567 = _mm256_min_ps(vx01234567, voutput_max_less_zero_point);
    vx89ABCDEF = _mm256_min_ps(vx89ABCDEF, voutput_max_less_zero_point);
    vxGHIJKLMN = _mm256_min_ps(vxGHIJKLMN, voutput_max_less_zero_point);
    vxOPQRSTUV = _mm256_min_ps(vxOPQRSTUV, voutput_max_less_zero_point);

    const __m256i vacc01234567 = _mm256_cvtps_epi32(vx01234567);
    const __m256i vacc89ABCDEF = _mm256_cvtps_epi32(vx89ABCDEF);
    const __m256i vaccGHIJKLMN = _mm256_cvtps_epi32(vxGHIJKLMN);
    const __m256i vaccOPQRSTUV = _mm256_cvtps_epi32(vxOPQRSTUV);

    __m128i vy01234567 = _mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extractf128_si256(vacc01234567, 1));
    __m128i vy89ABCDEF = _mm_packs_epi32(_mm256_castsi256_si128(vacc89ABCDEF), _mm256_extractf128_si256(vacc89ABCDEF, 1));
    __m128i vyGHIJKLMN = _mm_packs_epi32(_mm256_castsi256_si128(vaccGHIJKLMN), _mm256_extractf128_si256(vaccGHIJKLMN, 1));
    __m128i vyOPQRSTUV = _mm_packs_epi32(_mm256_castsi256_si128(vaccOPQRSTUV), _mm256_extractf128_si256(vaccOPQRSTUV, 1));

    vy01234567 = _mm_adds_epi16(vy01234567, voutput_zero_point);
    vy89ABCDEF = _mm_adds_epi16(vy89ABCDEF, voutput_zero_point);
    vyGHIJKLMN = _mm_adds_epi16(vyGHIJKLMN, voutput_zero_point);
    vyOPQRSTUV = _mm_adds_epi16(vyOPQRSTUV, voutput_zero_point);

    __m128i vy0123456789ABCDEF = _mm_packus_epi16(vy01234567, vy89ABCDEF);
    __m128i vyGHIJKLMNOPQRSTUV = _mm_packus_epi16(vyGHIJKLMN, vyOPQRSTUV);

    vy0123456789ABCDEF = _mm_max_epu8(vy0123456789ABCDEF, voutput_min);
    vyGHIJKLMNOPQRSTUV = _mm_max_epu8(vyGHIJKLMNOPQRSTUV, voutput_min);

    _mm_storeu_si128((__m128i*) y, vy0123456789ABCDEF);
    _mm_storeu_si128((__m128i*) (y + 16), vyGHIJKLMNOPQRSTUV);
    y += 32;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(x);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);
    x += 8;

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extractf128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packus_epi16(vy, vy);
    vy = _mm_max_epu8(vy, voutput_min);

    _mm_storel_epi64((__m128i*) y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    __m256 vx = _mm256_maskload_ps(x, vmask);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extractf128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packus_epi16(vy, vy);
    vy = _mm_max_epu8(vy, voutput_min);

    if (n & (4 * sizeof(float))) {
      *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vy);
      y += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    if (n & (2 * sizeof(float))) {
      *((uint16_t*) y) = (uint16_t) _mm_extract_epi16(vy, 0);
      y += 2;
      vy = _mm_srli_epi32(vy, 16);
    }
    if (n & (1 * sizeof(float))) {
      *y = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_f32_vadd_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    const __m256 vb01234567 = _mm256_loadu_ps(b);
    const __m256 vb89ABCDEF = _mm256_loadu_ps(b + 8);
    b += 16;

    __m256 vy01234567 = _mm256_add_ps(va01234567, vb01234567);
    __m256 vy89ABCDEF = _mm256_add_ps(va89ABCDEF, vb89ABCDEF);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    const __m256 vb = _mm256_loadu_ps(b);
    b += 8;

    __m256 vy = _mm256_add_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);
    const __m256 vb = _mm256_maskload_ps(b, vmask);

    __m256 vy = _mm256_add_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vaddc_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_add_ps(va01234567, vb);
    __m256 vy89ABCDEF = _mm256_add_ps(va89ABCDEF, vb);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_add_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_add_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vdiv_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    const __m256 vb01234567 = _mm256_loadu_ps(b);
    const __m256 vb89ABCDEF = _mm256_loadu_ps(b + 8);
    b += 16;

    __m256 vy01234567 = _mm256_div_ps(va01234567, vb01234567);
    __m256 vy89ABCDEF = _mm256_div_ps(va89ABCDEF, vb89ABCDEF);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    const __m256 vb = _mm256_loadu_ps(b);
    b += 8;

    __m256 vy = _mm256_div_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);
    const __m256 vb = _mm256_maskload_ps(b, vmask);

    __m256 vy = _mm256_div_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vdivc_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_div_ps(va01234567, vb);
    __m256 vy89ABCDEF = _mm256_div_ps(va89ABCDEF, vb);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_div_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_div_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vmax_ukernel__avx_x16(
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


  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    const __m256 vb01234567 = _mm256_loadu_ps(b);
    const __m256 vb89ABCDEF = _mm256_loadu_ps(b + 8);
    b += 16;

    __m256 vy01234567 = _mm256_max_ps(va01234567, vb01234567);
    __m256 vy89ABCDEF = _mm256_max_ps(va89ABCDEF, vb89ABCDEF);



    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    const __m256 vb = _mm256_loadu_ps(b);
    b += 8;

    __m256 vy = _mm256_max_ps(va, vb);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);
    const __m256 vb = _mm256_maskload_ps(b, vmask);

    __m256 vy = _mm256_max_ps(va, vb);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vmaxc_ukernel__avx_x16(
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


  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_max_ps(va01234567, vb);
    __m256 vy89ABCDEF = _mm256_max_ps(va89ABCDEF, vb);



    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_max_ps(va, vb);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_max_ps(va, vb);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vmin_ukernel__avx_x16(
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


  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    const __m256 vb01234567 = _mm256_loadu_ps(b);
    const __m256 vb89ABCDEF = _mm256_loadu_ps(b + 8);
    b += 16;

    __m256 vy01234567 = _mm256_min_ps(va01234567, vb01234567);
    __m256 vy89ABCDEF = _mm256_min_ps(va89ABCDEF, vb89ABCDEF);



    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    const __m256 vb = _mm256_loadu_ps(b);
    b += 8;

    __m256 vy = _mm256_min_ps(va, vb);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);
    const __m256 vb = _mm256_maskload_ps(b, vmask);

    __m256 vy = _mm256_min_ps(va, vb);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vminc_ukernel__avx_x16(
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


  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_min_ps(va01234567, vb);
    __m256 vy89ABCDEF = _mm256_min_ps(va89ABCDEF, vb);



    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_min_ps(va, vb);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_min_ps(va, vb);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vmul_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    const __m256 vb01234567 = _mm256_loadu_ps(b);
    const __m256 vb89ABCDEF = _mm256_loadu_ps(b + 8);
    b += 16;

    __m256 vy01234567 = _mm256_mul_ps(va01234567, vb01234567);
    __m256 vy89ABCDEF = _mm256_mul_ps(va89ABCDEF, vb89ABCDEF);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    const __m256 vb = _mm256_loadu_ps(b);
    b += 8;

    __m256 vy = _mm256_mul_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);
    const __m256 vb = _mm256_maskload_ps(b, vmask);

    __m256 vy = _mm256_mul_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vmulc_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_mul_ps(va01234567, vb);
    __m256 vy89ABCDEF = _mm256_mul_ps(va89ABCDEF, vb);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_mul_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_mul_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vrdivc_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_div_ps(vb, va01234567);
    __m256 vy89ABCDEF = _mm256_div_ps(vb, va89ABCDEF);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_div_ps(vb, va);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_div_ps(vb, va);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vrsubc_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_sub_ps(vb, va01234567);
    __m256 vy89ABCDEF = _mm256_sub_ps(vb, va89ABCDEF);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_sub_ps(vb, va);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_sub_ps(vb, va);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vsqrdiff_ukernel__avx_x16(
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


  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    const __m256 vb01234567 = _mm256_loadu_ps(b);
    const __m256 vb89ABCDEF = _mm256_loadu_ps(b + 8);
    b += 16;

    __m256 vy01234567 = _mm256_sub_ps(va01234567, vb01234567);
    __m256 vy89ABCDEF = _mm256_sub_ps(va89ABCDEF, vb89ABCDEF);

    vy01234567 = _mm256_mul_ps(vy01234567, vy01234567);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vy89ABCDEF);


    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    const __m256 vb = _mm256_loadu_ps(b);
    b += 8;

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_mul_ps(vy, vy);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);
    const __m256 vb = _mm256_maskload_ps(b, vmask);

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_mul_ps(vy, vy);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vsqrdiffc_ukernel__avx_x16(
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


  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_sub_ps(va01234567, vb);
    __m256 vy89ABCDEF = _mm256_sub_ps(va89ABCDEF, vb);

    vy01234567 = _mm256_mul_ps(vy01234567, vy01234567);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vy89ABCDEF);


    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_mul_ps(vy, vy);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_mul_ps(vy, vy);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vsub_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    const __m256 vb01234567 = _mm256_loadu_ps(b);
    const __m256 vb89ABCDEF = _mm256_loadu_ps(b + 8);
    b += 16;

    __m256 vy01234567 = _mm256_sub_ps(va01234567, vb01234567);
    __m256 vy89ABCDEF = _mm256_sub_ps(va89ABCDEF, vb89ABCDEF);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    const __m256 vb = _mm256_loadu_ps(b);
    b += 8;

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);
    const __m256 vb = _mm256_maskload_ps(b, vmask);

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vsubc_minmax_ukernel__avx_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  const __m256 vb = _mm256_broadcast_ss(b);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(a + 8);
    a += 16;

    __m256 vy01234567 = _mm256_sub_ps(va01234567, vb);
    __m256 vy89ABCDEF = _mm256_sub_ps(va89ABCDEF, vb);


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy89ABCDEF = _mm256_max_ps(vy89ABCDEF, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy89ABCDEF = _mm256_min_ps(vy89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(a);
    a += 8;

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 va = _mm256_maskload_ps(a, vmask);

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vclamp_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m256 vacc01234567 = _mm256_loadu_ps(x);
    __m256 vacc89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    vacc01234567 = _mm256_max_ps(vacc01234567, vy_min);
    vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEF, vy_min);

    vacc01234567 = _mm256_min_ps(vacc01234567, vy_max);
    vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vy_max);

    _mm256_storeu_ps(y, vacc01234567);
    _mm256_storeu_ps(y + 8, vacc89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    __m256 vacc = _mm256_loadu_ps(x);
    x += 8;

    vacc = _mm256_max_ps(vacc, vy_min);
    vacc = _mm256_min_ps(vacc, vy_max);

    _mm256_storeu_ps(y, vacc);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    __m256 vacc = _mm256_maskload_ps(x, vmask);
    vacc = _mm256_max_ps(vacc, vy_min);
    vacc = _mm256_min_ps(vacc, vy_max);

    __m128 vacc_lo = _mm256_castps256_ps128(vacc);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vacc_lo);
      vacc_lo = _mm256_extractf128_ps(vacc, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vacc_lo);
      vacc_lo = _mm_movehl_ps(vacc_lo, vacc_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vacc_lo);
    }
  }
}

void xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n % sizeof(float) == 0);

  const __m256 vprescale = _mm256_load_ps(params->avx_rr2_lut4_p4.prescale);
  const __m256 valpha = _mm256_load_ps(params->avx_rr2_lut4_p4.alpha);
  const __m256 vbeta = _mm256_load_ps(params->avx_rr2_lut4_p4.beta);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx_rr2_lut4_p4.sat_cutoff);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_rr2_lut4_p4.magic_bias);
  const __m256 vlog2e = _mm256_load_ps(params->avx_rr2_lut4_p4.log2e);
  const __m256 vindex_mask = _mm256_load_ps((const float*) params->avx_rr2_lut4_p4.index_mask);
  const __m256 vtable = _mm256_load_ps(params->avx_rr2_lut4_p4.table);
  const __m256 vminus_ln2_hi = _mm256_load_ps(params->avx_rr2_lut4_p4.minus_ln2_hi);
  const __m256 vminus_ln2_lo = _mm256_load_ps(params->avx_rr2_lut4_p4.minus_ln2_lo);
  const __m256 vc4 = _mm256_load_ps(params->avx_rr2_lut4_p4.c4);
  const __m256 vc3 = _mm256_load_ps(params->avx_rr2_lut4_p4.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_rr2_lut4_p4.c2);
  const __m256 vone = _mm256_load_ps(params->avx_rr2_lut4_p4.one);

  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    __m256 vx0 = _mm256_loadu_ps(x);
    __m256 vx1 = _mm256_loadu_ps(x + 8);
    __m256 vx2 = _mm256_loadu_ps(x + 16);
    __m256 vx3 = _mm256_loadu_ps(x + 24);
    x += 32;

    const __m256 vz0 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx0, vprescale));
    const __m256 vz1 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx1, vprescale));
    const __m256 vz2 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx2, vprescale));
    const __m256 vz3 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx3, vprescale));

    __m256 vn0 = _mm256_add_ps(_mm256_mul_ps(vz0, vlog2e), vmagic_bias);
    __m256 vn1 = _mm256_add_ps(_mm256_mul_ps(vz1, vlog2e), vmagic_bias);
    __m256 vn2 = _mm256_add_ps(_mm256_mul_ps(vz2, vlog2e), vmagic_bias);
    __m256 vn3 = _mm256_add_ps(_mm256_mul_ps(vz3, vlog2e), vmagic_bias);

    __m256 ven0 = _mm256_andnot_ps(vindex_mask, vn0);
    const __m256 vl0 = _mm256_permutevar_ps(vtable, _mm256_castps_si256(vn0));
    const __m128 ven0_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(ven0)), 21));
    __m256 ven1 = _mm256_andnot_ps(vindex_mask, vn1);
    const __m256 vl1 = _mm256_permutevar_ps(vtable, _mm256_castps_si256(vn1));
    const __m128 ven1_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(ven1)), 21));
    __m256 ven2 = _mm256_andnot_ps(vindex_mask, vn2);
    const __m256 vl2 = _mm256_permutevar_ps(vtable, _mm256_castps_si256(vn2));
    const __m128 ven2_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(ven2)), 21));
    __m256 ven3 = _mm256_andnot_ps(vindex_mask, vn3);
    const __m256 vl3 = _mm256_permutevar_ps(vtable, _mm256_castps_si256(vn3));
    const __m128 ven3_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(ven3)), 21));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    const __m128 ven0_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(ven0, 1)), 21));
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    const __m128 ven1_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(ven1, 1)), 21));
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    const __m128 ven2_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(ven2, 1)), 21));
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    const __m128 ven3_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(ven3, 1)), 21));

    __m256 vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2_hi), vz0);
    ven0 = _mm256_insertf128_ps(_mm256_castps128_ps256(ven0_lo), ven0_hi, 1);
    __m256 vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2_hi), vz1);
    ven1 = _mm256_insertf128_ps(_mm256_castps128_ps256(ven1_lo), ven1_hi, 1);
    __m256 vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2_hi), vz2);
    ven2 = _mm256_insertf128_ps(_mm256_castps128_ps256(ven2_lo), ven2_hi, 1);
    __m256 vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2_hi), vz3);
    ven3 = _mm256_insertf128_ps(_mm256_castps128_ps256(ven3_lo), ven3_hi, 1);

    vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2_lo), vt0);
    __m256 vs0 = _mm256_mul_ps(vl0, ven0);
    vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2_lo), vt1);
    __m256 vs1 = _mm256_mul_ps(vl1, ven1);
    vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2_lo), vt2);
    __m256 vs2 = _mm256_mul_ps(vl2, ven2);
    vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2_lo), vt3);
    __m256 vs3 = _mm256_mul_ps(vl3, ven3);

    __m256 vp0 = _mm256_add_ps(_mm256_mul_ps(vc4, vt0), vc3);
    __m256 vp1 = _mm256_add_ps(_mm256_mul_ps(vc4, vt1), vc3);
    __m256 vp2 = _mm256_add_ps(_mm256_mul_ps(vc4, vt2), vc3);
    __m256 vp3 = _mm256_add_ps(_mm256_mul_ps(vc4, vt3), vc3);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc2);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc2);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc2);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc2);

    vp0 = _mm256_mul_ps(vp0, vt0);
    vp1 = _mm256_mul_ps(vp1, vt1);
    vp2 = _mm256_mul_ps(vp2, vt2);
    vp3 = _mm256_mul_ps(vp3, vt3);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vs0 = _mm256_sub_ps(vs0, vone);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vs1 = _mm256_sub_ps(vs1, vone);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vs2 = _mm256_sub_ps(vs2, vone);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vs3 = _mm256_sub_ps(vs3, vone);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vt0);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vt1);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vt2);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vt3);

    const __m256 ve0 = _mm256_mul_ps(_mm256_add_ps(vp0, vs0), valpha);
    vx0 = _mm256_mul_ps(vx0, vbeta);
    const __m256 ve1 = _mm256_mul_ps(_mm256_add_ps(vp1, vs1), valpha);
    vx1 = _mm256_mul_ps(vx1, vbeta);
    const __m256 ve2 = _mm256_mul_ps(_mm256_add_ps(vp2, vs2), valpha);
    vx2 = _mm256_mul_ps(vx2, vbeta);
    const __m256 ve3 = _mm256_mul_ps(_mm256_add_ps(vp3, vs3), valpha);
    vx3 = _mm256_mul_ps(vx3, vbeta);

    const __m256 vy0 = _mm256_blendv_ps(vx0, ve0, vx0);
    const __m256 vy1 = _mm256_blendv_ps(vx1, ve1, vx1);
    const __m256 vy2 = _mm256_blendv_ps(vx2, ve2, vx2);
    const __m256 vy3 = _mm256_blendv_ps(vx3, ve3, vx3);

    _mm256_storeu_ps(y, vy0);
    _mm256_storeu_ps(y + 8, vy1);
    _mm256_storeu_ps(y + 16, vy2);
    _mm256_storeu_ps(y + 24, vy3);
    y += 32;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);
    __m256 ven = _mm256_andnot_ps(vindex_mask, vn);
    const __m256 vl = _mm256_permutevar_ps(vtable, _mm256_castps_si256(vn));
    const __m128 ven_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(ven)), 21));
    vn = _mm256_sub_ps(vn, vmagic_bias);
    const __m128 ven_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(ven, 1)), 21));

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    ven = _mm256_insertf128_ps(_mm256_castps128_ps256(ven_lo), ven_hi, 1);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);
    __m256 vs = _mm256_mul_ps(vl, ven);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc4, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_sub_ps(vs, vone);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vt);

    const __m256 ve = _mm256_mul_ps(_mm256_add_ps(vp, vs), valpha);
    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx_rr2_p6.mask_table[7] - n));

    __m256 vx = _mm256_maskload_ps(x, vmask);

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);
    __m256 ven = _mm256_andnot_ps(vindex_mask, vn);
    const __m256 vl = _mm256_permutevar_ps(vtable, _mm256_castps_si256(vn));
    const __m128 ven_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(ven)), 21));
    vn = _mm256_sub_ps(vn, vmagic_bias);
    const __m128 ven_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(ven, 1)), 21));

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    ven = _mm256_insertf128_ps(_mm256_castps128_ps256(ven_lo), ven_hi, 1);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);
    __m256 vs = _mm256_mul_ps(vl, ven);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc4, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_sub_ps(vs, vone);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vt);

    const __m256 ve = _mm256_mul_ps(_mm256_add_ps(vp, vs), valpha);
    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vhswish_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m256 vsixth = _mm256_load_ps(params->avx.sixth);
  const __m256 vhalf = _mm256_load_ps(params->avx.half);
  const __m256 vone = _mm256_load_ps(params->avx.one);
  const __m256 vzero = _mm256_setzero_ps();

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    __m256 vacc01234567 = _mm256_mul_ps(vx01234567, vsixth);
    __m256 vacc89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vsixth);

    vacc01234567 = _mm256_add_ps(vacc01234567, vhalf);
    vacc89ABCDEF = _mm256_add_ps(vacc89ABCDEF, vhalf);

    vacc01234567 = _mm256_max_ps(vacc01234567, vzero);
    vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEF, vzero);

    vacc01234567 = _mm256_min_ps(vacc01234567, vone);
    vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vone);

    vacc01234567 = _mm256_mul_ps(vacc01234567, vx01234567);
    vacc89ABCDEF = _mm256_mul_ps(vacc89ABCDEF, vx89ABCDEF);

    _mm256_storeu_ps(y, vacc01234567);
    _mm256_storeu_ps(y + 8, vacc89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;
    __m256 vacc = _mm256_mul_ps(vx, vsixth);
    vacc = _mm256_add_ps(vacc, vhalf);
    vacc = _mm256_max_ps(vacc, vzero);
    vacc = _mm256_min_ps(vacc, vone);
    vacc = _mm256_mul_ps(vacc, vx);
    _mm256_storeu_ps(y, vacc);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    __m256 vacc = _mm256_mul_ps(vx, vsixth);
    vacc = _mm256_add_ps(vacc, vhalf);
    vacc = _mm256_max_ps(vacc, vzero);
    vacc = _mm256_min_ps(vacc, vone);
    vacc = _mm256_mul_ps(vacc, vx);

    __m128 vacc_lo = _mm256_castps256_ps128(vacc);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vacc_lo);
      vacc_lo = _mm256_extractf128_ps(vacc, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vacc_lo);
      vacc_lo = _mm_movehl_ps(vacc_lo, vacc_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vacc_lo);
    }
  }
}

void xnn_f32_vlrelu_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const __m256 vslope = _mm256_load_ps(params->avx.slope);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    __m256 vacc01234567 = _mm256_mul_ps(vx01234567, vslope);
    __m256 vacc89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vslope);

    vacc01234567 = _mm256_blendv_ps(vx01234567, vacc01234567, vx01234567);
    vacc89ABCDEF = _mm256_blendv_ps(vx89ABCDEF, vacc89ABCDEF, vx89ABCDEF);

    _mm256_storeu_ps(y, vacc01234567);
    _mm256_storeu_ps(y + 8, vacc89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;
    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);
    _mm256_storeu_ps(y, vacc);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);

    __m128 vacc_lo = _mm256_castps256_ps128(vacc);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vacc_lo);
      vacc_lo = _mm256_extractf128_ps(vacc, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vacc_lo);
      vacc_lo = _mm_movehl_ps(vacc_lo, vacc_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vacc_lo);
    }
  }
}

void xnn_f32_vrndd_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    const __m256 vy01234567 = _mm256_round_ps(vx01234567, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    const __m256 vy89ABCDEF = _mm256_round_ps(vx89ABCDEF, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vrndne_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    const __m256 vy01234567 = _mm256_round_ps(vx01234567, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vy89ABCDEF = _mm256_round_ps(vx89ABCDEF, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vrndu_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    const __m256 vy01234567 = _mm256_round_ps(vx01234567, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    const __m256 vy89ABCDEF = _mm256_round_ps(vx89ABCDEF, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vrndz_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    const __m256 vy01234567 = _mm256_round_ps(vx01234567, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m256 vy89ABCDEF = _mm256_round_ps(vx89ABCDEF, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_round_ps(vx, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x40(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n % sizeof(float) == 0);

  const __m256 vsign_mask = _mm256_load_ps(params->avx_rr2_p5.sign_mask);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_rr2_p5.magic_bias);
  const __m256 vlog2e = _mm256_load_ps(params->avx_rr2_p5.log2e);
  const __m256 vminus_ln2_hi = _mm256_load_ps(params->avx_rr2_p5.minus_ln2_hi);
  const __m256 vminus_ln2_lo = _mm256_load_ps(params->avx_rr2_p5.minus_ln2_lo);
  const __m256 vc5 = _mm256_load_ps(params->avx_rr2_p5.c5);
  const __m256 vc4 = _mm256_load_ps(params->avx_rr2_p5.c4);
  const __m256 vc3 = _mm256_load_ps(params->avx_rr2_p5.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_rr2_p5.c2);
  const __m256 vc1 = _mm256_load_ps(params->avx_rr2_p5.c1);
  const __m256 vone = _mm256_load_ps(params->avx_rr2_p5.one);
  const __m256 vtwo = _mm256_load_ps(params->avx_rr2_p5.two);
  const __m256 vdenorm_cutoff = _mm256_load_ps(params->avx_rr2_p5.denorm_cutoff);

  for (; n >= 40 * sizeof(float); n -= 40 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(x);
    const __m256 vx1 = _mm256_loadu_ps(x + 8);
    const __m256 vx2 = _mm256_loadu_ps(x + 16);
    const __m256 vx3 = _mm256_loadu_ps(x + 24);
    const __m256 vx4 = _mm256_loadu_ps(x + 32);
    x += 40;

    const __m256 vz0 = _mm256_or_ps(vx0, vsign_mask);
    const __m256 vz1 = _mm256_or_ps(vx1, vsign_mask);
    const __m256 vz2 = _mm256_or_ps(vx2, vsign_mask);
    const __m256 vz3 = _mm256_or_ps(vx3, vsign_mask);
    const __m256 vz4 = _mm256_or_ps(vx4, vsign_mask);

    __m256 vn0 = _mm256_add_ps(_mm256_mul_ps(vz0, vlog2e), vmagic_bias);
    __m256 vn1 = _mm256_add_ps(_mm256_mul_ps(vz1, vlog2e), vmagic_bias);
    __m256 vn2 = _mm256_add_ps(_mm256_mul_ps(vz2, vlog2e), vmagic_bias);
    __m256 vn3 = _mm256_add_ps(_mm256_mul_ps(vz3, vlog2e), vmagic_bias);
    __m256 vn4 = _mm256_add_ps(_mm256_mul_ps(vz4, vlog2e), vmagic_bias);

    const __m128 vs_lo0 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn0)), 23));
    const __m128 vs_hi0 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn0, 1)), 23));
    const __m256 vs0 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo0), vs_hi0, 1);
    const __m128 vs_lo1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn1)), 23));
    const __m128 vs_hi1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn1, 1)), 23));
    const __m256 vs1 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo1), vs_hi1, 1);
    const __m128 vs_lo2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn2)), 23));
    const __m128 vs_hi2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn2, 1)), 23));
    const __m256 vs2 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo2), vs_hi2, 1);
    const __m128 vs_lo3 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn3)), 23));
    const __m128 vs_hi3 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn3, 1)), 23));
    const __m256 vs3 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo3), vs_hi3, 1);
    const __m128 vs_lo4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn4)), 23));
    const __m128 vs_hi4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn4, 1)), 23));
    const __m256 vs4 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo4), vs_hi4, 1);

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);

    __m256 vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2_hi), vz0);
    __m256 vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2_hi), vz1);
    __m256 vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2_hi), vz2);
    __m256 vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2_hi), vz3);
    __m256 vt4 = _mm256_add_ps(_mm256_mul_ps(vn4, vminus_ln2_hi), vz4);

    vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2_lo), vt0);
    vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2_lo), vt1);
    vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2_lo), vt2);
    vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2_lo), vt3);
    vt4 = _mm256_add_ps(_mm256_mul_ps(vn4, vminus_ln2_lo), vt4);

    __m256 vp0 = _mm256_add_ps(_mm256_mul_ps(vc5, vt0), vc4);
    __m256 vp1 = _mm256_add_ps(_mm256_mul_ps(vc5, vt1), vc4);
    __m256 vp2 = _mm256_add_ps(_mm256_mul_ps(vc5, vt2), vc4);
    __m256 vp3 = _mm256_add_ps(_mm256_mul_ps(vc5, vt3), vc4);
    __m256 vp4 = _mm256_add_ps(_mm256_mul_ps(vc5, vt4), vc4);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc3);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc3);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc3);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc3);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc3);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc2);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc2);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc2);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc2);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc2);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc1);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc1);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc1);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc1);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vt4 = _mm256_mul_ps(vt4, vs4);

    const __m256 ve0 = _mm256_add_ps(_mm256_mul_ps(vt0, vp0), vs0);
    const __m256 ve1 = _mm256_add_ps(_mm256_mul_ps(vt1, vp1), vs1);
    const __m256 ve2 = _mm256_add_ps(_mm256_mul_ps(vt2, vp2), vs2);
    const __m256 ve3 = _mm256_add_ps(_mm256_mul_ps(vt3, vp3), vs3);
    const __m256 ve4 = _mm256_add_ps(_mm256_mul_ps(vt4, vp4), vs4);

    const __m256 vd0 = _mm256_add_ps(ve0, vone);
    const __m256 vd1 = _mm256_add_ps(ve1, vone);
    const __m256 vd2 = _mm256_add_ps(ve2, vone);
    const __m256 vd3 = _mm256_add_ps(ve3, vone);
    const __m256 vd4 = _mm256_add_ps(ve4, vone);

    __m256 vr0 = _mm256_rcp_ps(vd0);
    __m256 vr1 = _mm256_rcp_ps(vd1);
    __m256 vr2 = _mm256_rcp_ps(vd2);
    __m256 vr3 = _mm256_rcp_ps(vd3);
    __m256 vr4 = _mm256_rcp_ps(vd4);

    vr0 = _mm256_mul_ps(vr0, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr0, vd0)));
    vr0 = _mm256_mul_ps(vr0, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr0, vd0)));
    vr1 = _mm256_mul_ps(vr1, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr1, vd1)));
    vr1 = _mm256_mul_ps(vr1, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr1, vd1)));
    vr2 = _mm256_mul_ps(vr2, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr2, vd2)));
    vr2 = _mm256_mul_ps(vr2, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr2, vd2)));
    vr3 = _mm256_mul_ps(vr3, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr3, vd3)));
    vr3 = _mm256_mul_ps(vr3, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr3, vd3)));
    vr4 = _mm256_mul_ps(vr4, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr4, vd4)));
    vr4 = _mm256_mul_ps(vr4, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr4, vd4)));

    __m256 vf0 = _mm256_mul_ps(ve0, vr0);
    __m256 vf1 = _mm256_mul_ps(ve1, vr1);
    __m256 vf2 = _mm256_mul_ps(ve2, vr2);
    __m256 vf3 = _mm256_mul_ps(ve3, vr3);
    __m256 vf4 = _mm256_mul_ps(ve4, vr4);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vz0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vz1, vdenorm_cutoff, _CMP_LT_OS), vf1);
    vf2 = _mm256_andnot_ps(_mm256_cmp_ps(vz2, vdenorm_cutoff, _CMP_LT_OS), vf2);
    vf3 = _mm256_andnot_ps(_mm256_cmp_ps(vz3, vdenorm_cutoff, _CMP_LT_OS), vf3);
    vf4 = _mm256_andnot_ps(_mm256_cmp_ps(vz4, vdenorm_cutoff, _CMP_LT_OS), vf4);

    vf0 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf0), vf0, vx0);
    vf1 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf1), vf1, vx1);
    vf2 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf2), vf2, vx2);
    vf3 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf3), vf3, vx3);
    vf4 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf4), vf4, vx4);

    _mm256_storeu_ps(y, vf0);
    _mm256_storeu_ps(y + 8, vf1);
    _mm256_storeu_ps(y + 16, vf2);
    _mm256_storeu_ps(y + 24, vf3);
    _mm256_storeu_ps(y + 32, vf4);
    y += 40;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vs_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 23));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc5, vt), vc4);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc1);

    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_add_ps(_mm256_mul_ps(vt, vp), vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    __m256 vr = _mm256_rcp_ps(vd);
    vr = _mm256_mul_ps(vr, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr, vd)));
    vr = _mm256_mul_ps(vr, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr, vd)));
    __m256 vf = _mm256_mul_ps(ve, vr);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm256_storeu_ps(y, vf);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx_rr2_p5.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);
    const __m128 vs_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 23));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc5, vt), vc4);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc1);

    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_add_ps(_mm256_mul_ps(vt, vp), vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    __m256 vr = _mm256_rcp_ps(vd);
    vr = _mm256_mul_ps(vr, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr, vd)));
    vr = _mm256_mul_ps(vr, _mm256_sub_ps(vtwo, _mm256_mul_ps(vr, vd)));
    __m256 vf = _mm256_mul_ps(ve, vr);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    __m128 vf_lo = _mm256_castps256_ps128(vf);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vf_lo);
      vf_lo = _mm256_extractf128_ps(vf, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vf_lo);
      vf_lo = _mm_movehl_ps(vf_lo, vf_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vf_lo);
    }
  }
}

void xnn_f32_vsqrt_ukernel__avx_sqrt_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;
    const __m256 vy = _mm256_sqrt_ps(vx);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_sqrt_ps(vx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vabs_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_abs_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m256 vnonsign_mask = _mm256_load_ps(params->avx.nonsign_mask);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    const __m256 vy01234567 = _mm256_and_ps(vx01234567, vnonsign_mask);
    const __m256 vy89ABCDEF = _mm256_and_ps(vx89ABCDEF, vnonsign_mask);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;
    const __m256 vy = _mm256_and_ps(vx, vnonsign_mask);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_and_ps(vx, vnonsign_mask);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vneg_ukernel__avx_x16(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_neg_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m256 vsign_mask = _mm256_load_ps(params->sse.sign_mask);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    const __m256 vy01234567 = _mm256_xor_ps(vx01234567, vsign_mask);
    const __m256 vy89ABCDEF = _mm256_xor_ps(vx89ABCDEF, vsign_mask);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;
    const __m256 vy = _mm256_xor_ps(vx, vsign_mask);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_xor_ps(vx, vsign_mask);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_f32_vsqr_ukernel__avx_x16(
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
    const __m256 vx01234567 = _mm256_loadu_ps(x);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(x + 8);
    x += 16;

    const __m256 vy01234567 = _mm256_mul_ps(vx01234567, vx01234567);
    const __m256 vy89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vx89ABCDEF);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;
    const __m256 vy = _mm256_mul_ps(vx, vx);
    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - n));

    const __m256 vx = _mm256_maskload_ps(x, vmask);
    const __m256 vy = _mm256_mul_ps(vx, vx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      i3 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      i4 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      i5 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      i6 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      i7 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      i8 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);


      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
      const __m128i vi9x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i9 + 8));
      const __m128i vxi9x89ABCDEF = _mm_cvtepi8_epi16(vi9x89ABCDEF);
      const __m128i vk9x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vxk9x89ABCDEF = _mm_cvtepi8_epi16(vk9x89ABCDEF);
      i9 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
      const __m128i vi10x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i10 + 8));
      const __m128i vxi10x89ABCDEF = _mm_cvtepi8_epi16(vi10x89ABCDEF);
      const __m128i vk10x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vxk10x89ABCDEF = _mm_cvtepi8_epi16(vk10x89ABCDEF);
      i10 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);


      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
      const __m128i vi11x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i11 + 8));
      const __m128i vxi11x89ABCDEF = _mm_cvtepi8_epi16(vi11x89ABCDEF);
      const __m128i vk11x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      const __m128i vxk11x89ABCDEF = _mm_cvtepi8_epi16(vk11x89ABCDEF);
      i11 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
      const __m128i vi12x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i12 + 8));
      const __m128i vxi12x89ABCDEF = _mm_cvtepi8_epi16(vi12x89ABCDEF);
      const __m128i vk12x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      const __m128i vxk12x89ABCDEF = _mm_cvtepi8_epi16(vk12x89ABCDEF);
      i12 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);


      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
      const __m128i vi13x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i13 + 8));
      const __m128i vxi13x89ABCDEF = _mm_cvtepi8_epi16(vi13x89ABCDEF);
      const __m128i vk13x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(int8_t)));
      const __m128i vxk13x89ABCDEF = _mm_cvtepi8_epi16(vk13x89ABCDEF);
      i13 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t)));
      const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
      const __m128i vi14x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i14 + 8));
      const __m128i vxi14x89ABCDEF = _mm_cvtepi8_epi16(vi14x89ABCDEF);
      const __m128i vk14x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(int8_t)));
      const __m128i vxk14x89ABCDEF = _mm_cvtepi8_epi16(vk14x89ABCDEF);
      i14 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);


      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t)));
      const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
      const __m128i vi15x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i15 + 8));
      const __m128i vxi15x89ABCDEF = _mm_cvtepi8_epi16(vi15x89ABCDEF);
      const __m128i vk15x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(int8_t)));
      const __m128i vxk15x89ABCDEF = _mm_cvtepi8_epi16(vk15x89ABCDEF);
      i15 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t)));
      const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
      const __m128i vi16x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i16 + 8));
      const __m128i vxi16x89ABCDEF = _mm_cvtepi8_epi16(vi16x89ABCDEF);
      const __m128i vk16x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(int8_t)));
      const __m128i vxk16x89ABCDEF = _mm_cvtepi8_epi16(vk16x89ABCDEF);
      i16 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);


      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t)));
      const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
      const __m128i vi17x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i17 + 8));
      const __m128i vxi17x89ABCDEF = _mm_cvtepi8_epi16(vi17x89ABCDEF);
      const __m128i vk17x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(int8_t)));
      const __m128i vxk17x89ABCDEF = _mm_cvtepi8_epi16(vk17x89ABCDEF);
      i17 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t)));
      const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
      const __m128i vi18x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i18 + 8));
      const __m128i vxi18x89ABCDEF = _mm_cvtepi8_epi16(vi18x89ABCDEF);
      const __m128i vk18x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(int8_t)));
      const __m128i vxk18x89ABCDEF = _mm_cvtepi8_epi16(vk18x89ABCDEF);
      i18 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);


      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t)));
      const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
      const __m128i vi19x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i19 + 8));
      const __m128i vxi19x89ABCDEF = _mm_cvtepi8_epi16(vi19x89ABCDEF);
      const __m128i vk19x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(int8_t)));
      const __m128i vxk19x89ABCDEF = _mm_cvtepi8_epi16(vk19x89ABCDEF);
      i19 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t)));
      const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
      const __m128i vi20x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i20 + 8));
      const __m128i vxi20x89ABCDEF = _mm_cvtepi8_epi16(vi20x89ABCDEF);
      const __m128i vk20x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(int8_t)));
      const __m128i vxk20x89ABCDEF = _mm_cvtepi8_epi16(vk20x89ABCDEF);
      i20 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);


      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t)));
      const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
      const __m128i vi21x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i21 + 8));
      const __m128i vxi21x89ABCDEF = _mm_cvtepi8_epi16(vi21x89ABCDEF);
      const __m128i vk21x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(int8_t)));
      const __m128i vxk21x89ABCDEF = _mm_cvtepi8_epi16(vk21x89ABCDEF);
      i21 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t)));
      const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
      const __m128i vi22x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i22 + 8));
      const __m128i vxi22x89ABCDEF = _mm_cvtepi8_epi16(vi22x89ABCDEF);
      const __m128i vk22x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(int8_t)));
      const __m128i vxk22x89ABCDEF = _mm_cvtepi8_epi16(vk22x89ABCDEF);
      i22 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);


      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t)));
      const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
      const __m128i vi23x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i23 + 8));
      const __m128i vxi23x89ABCDEF = _mm_cvtepi8_epi16(vi23x89ABCDEF);
      const __m128i vk23x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(int8_t)));
      const __m128i vxk23x89ABCDEF = _mm_cvtepi8_epi16(vk23x89ABCDEF);
      i23 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t)));
      const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
      const __m128i vi24x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i24 + 8));
      const __m128i vxi24x89ABCDEF = _mm_cvtepi8_epi16(vi24x89ABCDEF);
      const __m128i vk24x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(int8_t)));
      const __m128i vxk24x89ABCDEF = _mm_cvtepi8_epi16(vk24x89ABCDEF);
      i24 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      const __m128 vscale89AB = _mm_loadu_ps((const float*) w + 8);
      const __m128 vscaleCDEF = _mm_loadu_ps((const float*) w + 12);
      w = (const void*) ((const float*) w + 16);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale89AB);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscaleCDEF);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
        i9 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) (k + 160));
        const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
        i10 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) (k + 176));
        const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
        i11 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
        i12 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) (k + 208));
        const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
        i13 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) (k + 224));
        const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
        i14 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) (k + 240));
        const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
        i15 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) (k + 256));
        const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
        i16 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) (k + 272));
        const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
        i17 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) (k + 288));
        const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
        i18 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) (k + 304));
        const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
        i19 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) (k + 320));
        const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
        i20 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) (k + 336));
        const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
        i21 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) (k + 352));
        const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
        i22 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) (k + 368));
        const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
        i23 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) (k + 384));
        const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
        i24 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->sse4.output_min));

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      i3 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      i4 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      i5 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      i6 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      i7 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      i8 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      const __m128 vscale89AB = _mm_loadu_ps((const float*) w + 8);
      const __m128 vscaleCDEF = _mm_loadu_ps((const float*) w + 12);
      w = (const void*) ((const float*) w + 16);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale89AB);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscaleCDEF);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->sse4.output_min));

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
      const __m128i vxb1 = _mm_srai_epi16(_mm_unpackhi_epi8(vb01, vb01), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
      const __m128i vxb3 = _mm_srai_epi16(_mm_unpackhi_epi8(vb23, vb23), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const void*) ((const float*) w + 4);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->sse4.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
      a1 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
      const __m128i vxb1 = _mm_srai_epi16(_mm_unpackhi_epi8(vb01, vb01), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
      const __m128i vxb3 = _mm_srai_epi16(_mm_unpackhi_epi8(vb23, vb23), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const void*) ((const float*) w + 4);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->sse4.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      *((uint32_t*) c1) = (uint32_t) _mm_extract_epi32(vout, 1);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        *((uint16_t*) c1) = (uint16_t) _mm_extract_epi16(vout, 2);
        c1 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpackhi_epi8(vb01, vb01), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpackhi_epi8(vb23, vb23), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const void*) ((const float*) w + 4);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->sse4.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpackhi_epi8(vb01, vb01), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpackhi_epi8(vb23, vb23), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const void*) ((const float*) w + 4);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->sse4.output_min));

    if (nc >= 4) {
      *((uint32_t*) c1) = (uint32_t) _mm_extract_epi32(vout, 1);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c1) = (uint16_t) _mm_extract_epi16(vout, 2);
        c1 += 2;
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      i3 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      i4 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      i5 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      i6 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      i7 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      i8 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);


      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
      const __m128i vi9x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i9 + 8));
      const __m128i vxi9x89ABCDEF = _mm_cvtepi8_epi16(vi9x89ABCDEF);
      const __m128i vk9x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vxk9x89ABCDEF = _mm_cvtepi8_epi16(vk9x89ABCDEF);
      i9 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
      const __m128i vi10x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i10 + 8));
      const __m128i vxi10x89ABCDEF = _mm_cvtepi8_epi16(vi10x89ABCDEF);
      const __m128i vk10x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vxk10x89ABCDEF = _mm_cvtepi8_epi16(vk10x89ABCDEF);
      i10 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);


      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
      const __m128i vi11x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i11 + 8));
      const __m128i vxi11x89ABCDEF = _mm_cvtepi8_epi16(vi11x89ABCDEF);
      const __m128i vk11x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      const __m128i vxk11x89ABCDEF = _mm_cvtepi8_epi16(vk11x89ABCDEF);
      i11 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
      const __m128i vi12x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i12 + 8));
      const __m128i vxi12x89ABCDEF = _mm_cvtepi8_epi16(vi12x89ABCDEF);
      const __m128i vk12x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      const __m128i vxk12x89ABCDEF = _mm_cvtepi8_epi16(vk12x89ABCDEF);
      i12 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);


      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
      const __m128i vi13x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i13 + 8));
      const __m128i vxi13x89ABCDEF = _mm_cvtepi8_epi16(vi13x89ABCDEF);
      const __m128i vk13x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(int8_t)));
      const __m128i vxk13x89ABCDEF = _mm_cvtepi8_epi16(vk13x89ABCDEF);
      i13 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(int8_t)));
      const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
      const __m128i vi14x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i14 + 8));
      const __m128i vxi14x89ABCDEF = _mm_cvtepi8_epi16(vi14x89ABCDEF);
      const __m128i vk14x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(int8_t)));
      const __m128i vxk14x89ABCDEF = _mm_cvtepi8_epi16(vk14x89ABCDEF);
      i14 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);


      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(int8_t)));
      const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
      const __m128i vi15x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i15 + 8));
      const __m128i vxi15x89ABCDEF = _mm_cvtepi8_epi16(vi15x89ABCDEF);
      const __m128i vk15x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(int8_t)));
      const __m128i vxk15x89ABCDEF = _mm_cvtepi8_epi16(vk15x89ABCDEF);
      i15 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(int8_t)));
      const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
      const __m128i vi16x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i16 + 8));
      const __m128i vxi16x89ABCDEF = _mm_cvtepi8_epi16(vi16x89ABCDEF);
      const __m128i vk16x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(int8_t)));
      const __m128i vxk16x89ABCDEF = _mm_cvtepi8_epi16(vk16x89ABCDEF);
      i16 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);


      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(int8_t)));
      const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
      const __m128i vi17x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i17 + 8));
      const __m128i vxi17x89ABCDEF = _mm_cvtepi8_epi16(vi17x89ABCDEF);
      const __m128i vk17x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(int8_t)));
      const __m128i vxk17x89ABCDEF = _mm_cvtepi8_epi16(vk17x89ABCDEF);
      i17 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(int8_t)));
      const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
      const __m128i vi18x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i18 + 8));
      const __m128i vxi18x89ABCDEF = _mm_cvtepi8_epi16(vi18x89ABCDEF);
      const __m128i vk18x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(int8_t)));
      const __m128i vxk18x89ABCDEF = _mm_cvtepi8_epi16(vk18x89ABCDEF);
      i18 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);


      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(int8_t)));
      const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
      const __m128i vi19x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i19 + 8));
      const __m128i vxi19x89ABCDEF = _mm_cvtepi8_epi16(vi19x89ABCDEF);
      const __m128i vk19x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(int8_t)));
      const __m128i vxk19x89ABCDEF = _mm_cvtepi8_epi16(vk19x89ABCDEF);
      i19 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(int8_t)));
      const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
      const __m128i vi20x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i20 + 8));
      const __m128i vxi20x89ABCDEF = _mm_cvtepi8_epi16(vi20x89ABCDEF);
      const __m128i vk20x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(int8_t)));
      const __m128i vxk20x89ABCDEF = _mm_cvtepi8_epi16(vk20x89ABCDEF);
      i20 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);


      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(int8_t)));
      const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
      const __m128i vi21x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i21 + 8));
      const __m128i vxi21x89ABCDEF = _mm_cvtepi8_epi16(vi21x89ABCDEF);
      const __m128i vk21x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(int8_t)));
      const __m128i vxk21x89ABCDEF = _mm_cvtepi8_epi16(vk21x89ABCDEF);
      i21 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(int8_t)));
      const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
      const __m128i vi22x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i22 + 8));
      const __m128i vxi22x89ABCDEF = _mm_cvtepi8_epi16(vi22x89ABCDEF);
      const __m128i vk22x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(int8_t)));
      const __m128i vxk22x89ABCDEF = _mm_cvtepi8_epi16(vk22x89ABCDEF);
      i22 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);


      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(int8_t)));
      const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
      const __m128i vi23x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i23 + 8));
      const __m128i vxi23x89ABCDEF = _mm_cvtepi8_epi16(vi23x89ABCDEF);
      const __m128i vk23x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(int8_t)));
      const __m128i vxk23x89ABCDEF = _mm_cvtepi8_epi16(vk23x89ABCDEF);
      i23 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(int8_t)));
      const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
      const __m128i vi24x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i24 + 8));
      const __m128i vxi24x89ABCDEF = _mm_cvtepi8_epi16(vi24x89ABCDEF);
      const __m128i vk24x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(int8_t)));
      const __m128i vxk24x89ABCDEF = _mm_cvtepi8_epi16(vk24x89ABCDEF);
      i24 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
        i9 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) (k + 160));
        const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
        i10 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) (k + 176));
        const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
        i11 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
        i12 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) (k + 208));
        const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
        i13 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) (k + 224));
        const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
        i14 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) (k + 240));
        const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
        i15 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) (k + 256));
        const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
        i16 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) (k + 272));
        const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
        i17 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) (k + 288));
        const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
        i18 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) (k + 304));
        const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
        i19 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) (k + 320));
        const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
        i20 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) (k + 336));
        const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
        i21 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) (k + 352));
        const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
        i22 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) (k + 368));
        const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
        i23 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) (k + 384));
        const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
        i24 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepi8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk0x89ABCDEF = _mm_cvtepi8_epi16(vk0x89ABCDEF);
      i0 += 16;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      __m128i vprod89ABCDEF = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepi8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk1x89ABCDEF = _mm_cvtepi8_epi16(vk1x89ABCDEF);
      i1 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepi8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk2x89ABCDEF = _mm_cvtepi8_epi16(vk2x89ABCDEF);
      i2 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepi8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk3x89ABCDEF = _mm_cvtepi8_epi16(vk3x89ABCDEF);
      i3 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepi8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk4x89ABCDEF = _mm_cvtepi8_epi16(vk4x89ABCDEF);
      i4 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepi8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk5x89ABCDEF = _mm_cvtepi8_epi16(vk5x89ABCDEF);
      i5 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepi8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk6x89ABCDEF = _mm_cvtepi8_epi16(vk6x89ABCDEF);
      i6 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepi8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk7x89ABCDEF = _mm_cvtepi8_epi16(vk7x89ABCDEF);
      i7 += 16;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));
      vprod89ABCDEF = _mm_add_epi16(vprod89ABCDEF, _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepi8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk8x89ABCDEF = _mm_cvtepi8_epi16(vk8x89ABCDEF);
      i8 += 16;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      vprod89ABCDEF = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vprod89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_srai_epi32(_mm_unpackhi_epi16(vprod89ABCDEF, vprod89ABCDEF), 16));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
        i0 += 8;


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
        i1 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
        i2 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
        i3 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
        i4 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
        i5 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
        i6 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
        i7 += 8;


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
        i8 += 8;


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_f32_vcvt_ukernel__avx_x32(
    size_t n,
    const int8_t* x,
    float* y,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(int8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m128i vminus_zero_point = _mm_load_si128((const __m128i*) params->avx.minus_zero_point);
  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  for (; n >= 32 * sizeof(int8_t); n -= 32 * sizeof(int8_t)) {
    __m128i vx0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(x));
    __m128i vx4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(x + 4));
    __m128i vx89AB = _mm_cvtepi8_epi32(_mm_loadu_si32(x + 8));
    __m128i vxCDEF = _mm_cvtepi8_epi32(_mm_loadu_si32(x + 12));
    __m128i vxGHIJ = _mm_cvtepi8_epi32(_mm_loadu_si32(x + 16));
    __m128i vxKLMN = _mm_cvtepi8_epi32(_mm_loadu_si32(x + 20));
    __m128i vxOPQR = _mm_cvtepi8_epi32(_mm_loadu_si32(x + 24));
    __m128i vxSTUV = _mm_cvtepi8_epi32(_mm_loadu_si32(x + 28));
    x += 32;

    vx0123 = _mm_add_epi32(vx0123, vminus_zero_point);
    vx4567 = _mm_add_epi32(vx4567, vminus_zero_point);
    vx89AB = _mm_add_epi32(vx89AB, vminus_zero_point);
    vxCDEF = _mm_add_epi32(vxCDEF, vminus_zero_point);
    vxGHIJ = _mm_add_epi32(vxGHIJ, vminus_zero_point);
    vxKLMN = _mm_add_epi32(vxKLMN, vminus_zero_point);
    vxOPQR = _mm_add_epi32(vxOPQR, vminus_zero_point);
    vxSTUV = _mm_add_epi32(vxSTUV, vminus_zero_point);

    const __m256i vx01234567 = _mm256_insertf128_si256(_mm256_castsi128_si256(vx0123), vx4567, 1);
    const __m256i vx89ABCDEF = _mm256_insertf128_si256(_mm256_castsi128_si256(vx89AB), vxCDEF, 1);
    const __m256i vxGHIJKLMN = _mm256_insertf128_si256(_mm256_castsi128_si256(vxGHIJ), vxKLMN, 1);
    const __m256i vxOPQRSTUV = _mm256_insertf128_si256(_mm256_castsi128_si256(vxOPQR), vxSTUV, 1);

    __m256 vy01234567 = _mm256_cvtepi32_ps(vx01234567);
    __m256 vy89ABCDEF = _mm256_cvtepi32_ps(vx89ABCDEF);
    __m256 vyGHIJKLMN = _mm256_cvtepi32_ps(vxGHIJKLMN);
    __m256 vyOPQRSTUV = _mm256_cvtepi32_ps(vxOPQRSTUV);

    vy01234567 = _mm256_mul_ps(vy01234567, vscale);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vscale);
    vyGHIJKLMN = _mm256_mul_ps(vyGHIJKLMN, vscale);
    vyOPQRSTUV = _mm256_mul_ps(vyOPQRSTUV, vscale);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    _mm256_storeu_ps(y + 16, vyGHIJKLMN);
    _mm256_storeu_ps(y + 24, vyOPQRSTUV);
    y += 32;
  }
  for (; n >= 4 * sizeof(int8_t); n -= 4 * sizeof(int8_t)) {
    __m128i vx = _mm_cvtepi8_epi32(_mm_loadu_si32(x));
    vx = _mm_add_epi32(vx, vminus_zero_point);
    x += 4;

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, _mm256_castps256_ps128(vscale));

    _mm_storeu_ps(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(int8_t));
    assert(n <= 3 * sizeof(int8_t));

    __m128i vx = _mm_cvtepi8_epi32(_mm_loadu_si32(x));
    vx = _mm_add_epi32(vx, vminus_zero_point);

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, _mm256_castps256_ps128(vscale));

    if (n & (2 * sizeof(int8_t))) {
      _mm_storel_pi((__m64*) y, vy);
      vy = _mm_movehl_ps(vy, vy);
      y += 2;
    }
    if (n & (1 * sizeof(int8_t))) {
      _mm_store_ss(y, vy);
    }
  }
}

void xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
      const __m128i vxb1 = _mm_srai_epi16(_mm_unpackhi_epi8(vb01, vb01), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
      const __m128i vxb3 = _mm_srai_epi16(_mm_unpackhi_epi8(vb23, vb23), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
      a1 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
      const __m128i vxb1 = _mm_srai_epi16(_mm_unpackhi_epi8(vb01, vb01), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
      const __m128i vxb3 = _mm_srai_epi16(_mm_unpackhi_epi8(vb23, vb23), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      *((uint32_t*) c1) = (uint32_t) _mm_extract_epi32(vout, 1);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        *((uint16_t*) c1) = (uint16_t) _mm_extract_epi16(vout, 2);
        c1 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpackhi_epi8(vb01, vb01), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpackhi_epi8(vb23, vb23), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpackhi_epi8(vb01, vb01), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpackhi_epi8(vb23, vb23), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      *((uint32_t*) c1) = (uint32_t) _mm_extract_epi32(vout, 1);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c1) = (uint16_t) _mm_extract_epi16(vout, 2);
        c1 += 2;
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_x8(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_addsub_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse4_mul32.bias);
  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4_mul32.a_multiplier);
  const __m128i vb_multiplier = _mm_load_si128((const __m128i*) params->sse4_mul32.b_multiplier);
  const __m128i vshift = _mm_loadu_si32(params->sse4_mul32.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4_mul32.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4_mul32.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4_mul32.output_max);

  for (; n >= 8 * sizeof(int8_t); n -= 8 * sizeof(int8_t)) {
    const __m128i va0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a));
    const __m128i vb0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b));
    const __m128i va4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 4));
    const __m128i vb4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 4));
    input_a += 8;
    input_b += 8;

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vb0123, vb_multiplier));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vb4567, vb_multiplier));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    {
      const __m128i va0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a));
      const __m128i vb0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b));
      const __m128i va4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 4));
      const __m128i vb4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_b + 4));

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vb0123, vb_multiplier));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vb4567, vb_multiplier));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if (n & (4 * sizeof(int8_t))) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (n & (2 * sizeof(int8_t))) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (n & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qs8_vaddc_minmax_ukernel__avx_mul32_ld32_x8(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_addsub_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4_mul32.a_multiplier);
  const __m128i vshift = _mm_loadu_si32(params->sse4_mul32.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4_mul32.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4_mul32.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4_mul32.output_max);

  __m128i vbias = _mm_cvtsi32_si128(params->sse4_mul32.b_multiplier[0] * (int32_t) *input_b);
  vbias = _mm_shuffle_epi32(vbias, _MM_SHUFFLE(0, 0, 0, 0));
  vbias = _mm_add_epi32(vbias, _mm_load_si128((const __m128i*) params->sse4_mul32.bias));
  for (; n >= 8 * sizeof(int8_t); n -= 8 * sizeof(int8_t)) {
    const __m128i va0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a));
    const __m128i va4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 4));
    input_a += 8;
    input_b += 8;

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    {
      const __m128i va0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a));
      const __m128i va4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(input_a + 4));

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if (n & (4 * sizeof(int8_t))) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (n & (2 * sizeof(int8_t))) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (n & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qs8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS

{
  const __m128i va_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.a_zero_point);
  const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.b_zero_point);
  const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->fp32_sse4.output_max);

  for (; n >= 16 * sizeof(int8_t); n -= 16 * sizeof(int8_t)) {
    const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i vb01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
    const __m128i va89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m128i vb89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    input_a += 16;
    input_b += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);
    const __m128i vxb89ABCDEF = _mm_sub_epi16(vb89ABCDEF, vb_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb89ABCDEF);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb89ABCDEF);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      const __m128i vb01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
      const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(n >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        n -= 8 * sizeof(int8_t);
      } else {
        if (n & (4 * sizeof(int8_t))) {
          *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (n & (2 * sizeof(int8_t))) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (n & (1 * sizeof(int8_t))) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        n = 0;
      }
    } while (n != 0);
  }
}

void xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS

{
  const __m128i va_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.a_zero_point);
  const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->fp32_sse4.output_max);

  __m128i vxb = _mm_sub_epi16(
    _mm_shuffle_epi32(_mm_cvtsi32_si128(UINT32_C(0x00010001) * (uint32_t) (uint16_t) (int16_t) *input_b), 0),
    _mm_load_si128((const __m128i*) params->fp32_sse4.b_zero_point));
  for (; n >= 16 * sizeof(int8_t); n -= 16 * sizeof(int8_t)) {
    const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i va89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    input_a += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      input_a += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(n >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        n -= 8 * sizeof(int8_t);
      } else {
        if (n & (4 * sizeof(int8_t))) {
          *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (n & (2 * sizeof(int8_t))) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (n & (1 * sizeof(int8_t))) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        n = 0;
      }
    } while (n != 0);
  }
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint8_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint8_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint8_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint8_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint8_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint8_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint8_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint8_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint8_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint8_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint8_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint8_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint8_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint8_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint8_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    const __m128i vk_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t)));
      const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x01234567), vk_zero_point);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepu8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(uint8_t)));
      const __m128i vxk0x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x89ABCDEF), vk_zero_point);
      i0 += 16;


      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x89ABCDEFlo = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);
      const __m128i vprod0x89ABCDEFhi = _mm_mulhi_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod0x89ABCDEFlo, vprod0x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod0x89ABCDEFlo, vprod0x89ABCDEFhi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t)));
      const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x01234567), vk_zero_point);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepu8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(uint8_t)));
      const __m128i vxk1x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x89ABCDEF), vk_zero_point);
      i1 += 16;


      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x89ABCDEFlo = _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);
      const __m128i vprod1x89ABCDEFhi = _mm_mulhi_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod1x89ABCDEFlo, vprod1x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod1x89ABCDEFlo, vprod1x89ABCDEFhi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t)));
      const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x01234567), vk_zero_point);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepu8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(uint8_t)));
      const __m128i vxk2x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x89ABCDEF), vk_zero_point);
      i2 += 16;


      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x89ABCDEFlo = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);
      const __m128i vprod2x89ABCDEFhi = _mm_mulhi_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod2x89ABCDEFlo, vprod2x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod2x89ABCDEFlo, vprod2x89ABCDEFhi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t)));
      const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x01234567), vk_zero_point);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepu8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(uint8_t)));
      const __m128i vxk3x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x89ABCDEF), vk_zero_point);
      i3 += 16;


      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x89ABCDEFlo = _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);
      const __m128i vprod3x89ABCDEFhi = _mm_mulhi_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod3x89ABCDEFlo, vprod3x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod3x89ABCDEFlo, vprod3x89ABCDEFhi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t)));
      const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x01234567), vk_zero_point);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepu8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(uint8_t)));
      const __m128i vxk4x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x89ABCDEF), vk_zero_point);
      i4 += 16;


      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x89ABCDEFlo = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);
      const __m128i vprod4x89ABCDEFhi = _mm_mulhi_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod4x89ABCDEFlo, vprod4x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod4x89ABCDEFlo, vprod4x89ABCDEFhi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t)));
      const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x01234567), vk_zero_point);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepu8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(uint8_t)));
      const __m128i vxk5x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x89ABCDEF), vk_zero_point);
      i5 += 16;


      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x89ABCDEFlo = _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);
      const __m128i vprod5x89ABCDEFhi = _mm_mulhi_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod5x89ABCDEFlo, vprod5x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod5x89ABCDEFlo, vprod5x89ABCDEFhi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(uint8_t)));
      const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x01234567), vk_zero_point);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepu8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(uint8_t)));
      const __m128i vxk6x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x89ABCDEF), vk_zero_point);
      i6 += 16;


      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x89ABCDEFlo = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);
      const __m128i vprod6x89ABCDEFhi = _mm_mulhi_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod6x89ABCDEFlo, vprod6x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod6x89ABCDEFlo, vprod6x89ABCDEFhi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepu8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(uint8_t)));
      const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x01234567), vk_zero_point);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepu8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(uint8_t)));
      const __m128i vxk7x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x89ABCDEF), vk_zero_point);
      i7 += 16;


      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x89ABCDEFlo = _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);
      const __m128i vprod7x89ABCDEFhi = _mm_mulhi_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod7x89ABCDEFlo, vprod7x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod7x89ABCDEFlo, vprod7x89ABCDEFhi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepu8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(uint8_t)));
      const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x01234567), vk_zero_point);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepu8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(uint8_t)));
      const __m128i vxk8x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x89ABCDEF), vk_zero_point);
      i8 += 16;


      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x89ABCDEFlo = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);
      const __m128i vprod8x89ABCDEFhi = _mm_mulhi_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod8x89ABCDEFlo, vprod8x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod8x89ABCDEFlo, vprod8x89ABCDEFhi));

      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vxi9x01234567 = _mm_cvtepu8_epi16(vi9x01234567);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(uint8_t)));
      const __m128i vxk9x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk9x01234567), vk_zero_point);
      const __m128i vi9x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i9 + 8));
      const __m128i vxi9x89ABCDEF = _mm_cvtepu8_epi16(vi9x89ABCDEF);
      const __m128i vk9x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 152 * sizeof(uint8_t)));
      const __m128i vxk9x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk9x89ABCDEF), vk_zero_point);
      i9 += 16;


      const __m128i vprod9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
      const __m128i vprod9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);
      const __m128i vprod9x89ABCDEFlo = _mm_mullo_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF);
      const __m128i vprod9x89ABCDEFhi = _mm_mulhi_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod9x01234567lo, vprod9x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod9x01234567lo, vprod9x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod9x89ABCDEFlo, vprod9x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod9x89ABCDEFlo, vprod9x89ABCDEFhi));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vxi10x01234567 = _mm_cvtepu8_epi16(vi10x01234567);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 160 * sizeof(uint8_t)));
      const __m128i vxk10x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk10x01234567), vk_zero_point);
      const __m128i vi10x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i10 + 8));
      const __m128i vxi10x89ABCDEF = _mm_cvtepu8_epi16(vi10x89ABCDEF);
      const __m128i vk10x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 168 * sizeof(uint8_t)));
      const __m128i vxk10x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk10x89ABCDEF), vk_zero_point);
      i10 += 16;


      const __m128i vprod10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      const __m128i vprod10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);
      const __m128i vprod10x89ABCDEFlo = _mm_mullo_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);
      const __m128i vprod10x89ABCDEFhi = _mm_mulhi_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod10x01234567lo, vprod10x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod10x01234567lo, vprod10x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod10x89ABCDEFlo, vprod10x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod10x89ABCDEFlo, vprod10x89ABCDEFhi));

      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vxi11x01234567 = _mm_cvtepu8_epi16(vi11x01234567);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 176 * sizeof(uint8_t)));
      const __m128i vxk11x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk11x01234567), vk_zero_point);
      const __m128i vi11x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i11 + 8));
      const __m128i vxi11x89ABCDEF = _mm_cvtepu8_epi16(vi11x89ABCDEF);
      const __m128i vk11x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 184 * sizeof(uint8_t)));
      const __m128i vxk11x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk11x89ABCDEF), vk_zero_point);
      i11 += 16;


      const __m128i vprod11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
      const __m128i vprod11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);
      const __m128i vprod11x89ABCDEFlo = _mm_mullo_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF);
      const __m128i vprod11x89ABCDEFhi = _mm_mulhi_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod11x01234567lo, vprod11x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod11x01234567lo, vprod11x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod11x89ABCDEFlo, vprod11x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod11x89ABCDEFlo, vprod11x89ABCDEFhi));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vxi12x01234567 = _mm_cvtepu8_epi16(vi12x01234567);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 192 * sizeof(uint8_t)));
      const __m128i vxk12x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk12x01234567), vk_zero_point);
      const __m128i vi12x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i12 + 8));
      const __m128i vxi12x89ABCDEF = _mm_cvtepu8_epi16(vi12x89ABCDEF);
      const __m128i vk12x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 200 * sizeof(uint8_t)));
      const __m128i vxk12x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk12x89ABCDEF), vk_zero_point);
      i12 += 16;


      const __m128i vprod12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      const __m128i vprod12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);
      const __m128i vprod12x89ABCDEFlo = _mm_mullo_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);
      const __m128i vprod12x89ABCDEFhi = _mm_mulhi_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod12x01234567lo, vprod12x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod12x01234567lo, vprod12x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod12x89ABCDEFlo, vprod12x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod12x89ABCDEFlo, vprod12x89ABCDEFhi));

      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vxi13x01234567 = _mm_cvtepu8_epi16(vi13x01234567);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 208 * sizeof(uint8_t)));
      const __m128i vxk13x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk13x01234567), vk_zero_point);
      const __m128i vi13x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i13 + 8));
      const __m128i vxi13x89ABCDEF = _mm_cvtepu8_epi16(vi13x89ABCDEF);
      const __m128i vk13x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 216 * sizeof(uint8_t)));
      const __m128i vxk13x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk13x89ABCDEF), vk_zero_point);
      i13 += 16;


      const __m128i vprod13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
      const __m128i vprod13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);
      const __m128i vprod13x89ABCDEFlo = _mm_mullo_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF);
      const __m128i vprod13x89ABCDEFhi = _mm_mulhi_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod13x01234567lo, vprod13x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod13x01234567lo, vprod13x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod13x89ABCDEFlo, vprod13x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod13x89ABCDEFlo, vprod13x89ABCDEFhi));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vxi14x01234567 = _mm_cvtepu8_epi16(vi14x01234567);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 224 * sizeof(uint8_t)));
      const __m128i vxk14x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk14x01234567), vk_zero_point);
      const __m128i vi14x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i14 + 8));
      const __m128i vxi14x89ABCDEF = _mm_cvtepu8_epi16(vi14x89ABCDEF);
      const __m128i vk14x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 232 * sizeof(uint8_t)));
      const __m128i vxk14x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk14x89ABCDEF), vk_zero_point);
      i14 += 16;


      const __m128i vprod14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      const __m128i vprod14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);
      const __m128i vprod14x89ABCDEFlo = _mm_mullo_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);
      const __m128i vprod14x89ABCDEFhi = _mm_mulhi_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod14x01234567lo, vprod14x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod14x01234567lo, vprod14x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod14x89ABCDEFlo, vprod14x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod14x89ABCDEFlo, vprod14x89ABCDEFhi));

      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vxi15x01234567 = _mm_cvtepu8_epi16(vi15x01234567);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 240 * sizeof(uint8_t)));
      const __m128i vxk15x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk15x01234567), vk_zero_point);
      const __m128i vi15x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i15 + 8));
      const __m128i vxi15x89ABCDEF = _mm_cvtepu8_epi16(vi15x89ABCDEF);
      const __m128i vk15x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 248 * sizeof(uint8_t)));
      const __m128i vxk15x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk15x89ABCDEF), vk_zero_point);
      i15 += 16;


      const __m128i vprod15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
      const __m128i vprod15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);
      const __m128i vprod15x89ABCDEFlo = _mm_mullo_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF);
      const __m128i vprod15x89ABCDEFhi = _mm_mulhi_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod15x01234567lo, vprod15x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod15x01234567lo, vprod15x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod15x89ABCDEFlo, vprod15x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod15x89ABCDEFlo, vprod15x89ABCDEFhi));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vxi16x01234567 = _mm_cvtepu8_epi16(vi16x01234567);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 256 * sizeof(uint8_t)));
      const __m128i vxk16x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk16x01234567), vk_zero_point);
      const __m128i vi16x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i16 + 8));
      const __m128i vxi16x89ABCDEF = _mm_cvtepu8_epi16(vi16x89ABCDEF);
      const __m128i vk16x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 264 * sizeof(uint8_t)));
      const __m128i vxk16x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk16x89ABCDEF), vk_zero_point);
      i16 += 16;


      const __m128i vprod16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      const __m128i vprod16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);
      const __m128i vprod16x89ABCDEFlo = _mm_mullo_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);
      const __m128i vprod16x89ABCDEFhi = _mm_mulhi_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod16x01234567lo, vprod16x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod16x01234567lo, vprod16x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod16x89ABCDEFlo, vprod16x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod16x89ABCDEFlo, vprod16x89ABCDEFhi));

      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vxi17x01234567 = _mm_cvtepu8_epi16(vi17x01234567);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 272 * sizeof(uint8_t)));
      const __m128i vxk17x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk17x01234567), vk_zero_point);
      const __m128i vi17x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i17 + 8));
      const __m128i vxi17x89ABCDEF = _mm_cvtepu8_epi16(vi17x89ABCDEF);
      const __m128i vk17x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 280 * sizeof(uint8_t)));
      const __m128i vxk17x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk17x89ABCDEF), vk_zero_point);
      i17 += 16;


      const __m128i vprod17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
      const __m128i vprod17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);
      const __m128i vprod17x89ABCDEFlo = _mm_mullo_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF);
      const __m128i vprod17x89ABCDEFhi = _mm_mulhi_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod17x01234567lo, vprod17x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod17x01234567lo, vprod17x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod17x89ABCDEFlo, vprod17x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod17x89ABCDEFlo, vprod17x89ABCDEFhi));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vxi18x01234567 = _mm_cvtepu8_epi16(vi18x01234567);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 288 * sizeof(uint8_t)));
      const __m128i vxk18x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk18x01234567), vk_zero_point);
      const __m128i vi18x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i18 + 8));
      const __m128i vxi18x89ABCDEF = _mm_cvtepu8_epi16(vi18x89ABCDEF);
      const __m128i vk18x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 296 * sizeof(uint8_t)));
      const __m128i vxk18x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk18x89ABCDEF), vk_zero_point);
      i18 += 16;


      const __m128i vprod18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      const __m128i vprod18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);
      const __m128i vprod18x89ABCDEFlo = _mm_mullo_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);
      const __m128i vprod18x89ABCDEFhi = _mm_mulhi_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod18x01234567lo, vprod18x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod18x01234567lo, vprod18x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod18x89ABCDEFlo, vprod18x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod18x89ABCDEFlo, vprod18x89ABCDEFhi));

      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vxi19x01234567 = _mm_cvtepu8_epi16(vi19x01234567);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 304 * sizeof(uint8_t)));
      const __m128i vxk19x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk19x01234567), vk_zero_point);
      const __m128i vi19x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i19 + 8));
      const __m128i vxi19x89ABCDEF = _mm_cvtepu8_epi16(vi19x89ABCDEF);
      const __m128i vk19x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 312 * sizeof(uint8_t)));
      const __m128i vxk19x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk19x89ABCDEF), vk_zero_point);
      i19 += 16;


      const __m128i vprod19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
      const __m128i vprod19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);
      const __m128i vprod19x89ABCDEFlo = _mm_mullo_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF);
      const __m128i vprod19x89ABCDEFhi = _mm_mulhi_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod19x01234567lo, vprod19x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod19x01234567lo, vprod19x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod19x89ABCDEFlo, vprod19x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod19x89ABCDEFlo, vprod19x89ABCDEFhi));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vxi20x01234567 = _mm_cvtepu8_epi16(vi20x01234567);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 320 * sizeof(uint8_t)));
      const __m128i vxk20x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk20x01234567), vk_zero_point);
      const __m128i vi20x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i20 + 8));
      const __m128i vxi20x89ABCDEF = _mm_cvtepu8_epi16(vi20x89ABCDEF);
      const __m128i vk20x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 328 * sizeof(uint8_t)));
      const __m128i vxk20x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk20x89ABCDEF), vk_zero_point);
      i20 += 16;


      const __m128i vprod20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      const __m128i vprod20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);
      const __m128i vprod20x89ABCDEFlo = _mm_mullo_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);
      const __m128i vprod20x89ABCDEFhi = _mm_mulhi_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod20x01234567lo, vprod20x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod20x01234567lo, vprod20x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod20x89ABCDEFlo, vprod20x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod20x89ABCDEFlo, vprod20x89ABCDEFhi));

      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vxi21x01234567 = _mm_cvtepu8_epi16(vi21x01234567);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 336 * sizeof(uint8_t)));
      const __m128i vxk21x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk21x01234567), vk_zero_point);
      const __m128i vi21x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i21 + 8));
      const __m128i vxi21x89ABCDEF = _mm_cvtepu8_epi16(vi21x89ABCDEF);
      const __m128i vk21x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 344 * sizeof(uint8_t)));
      const __m128i vxk21x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk21x89ABCDEF), vk_zero_point);
      i21 += 16;


      const __m128i vprod21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
      const __m128i vprod21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);
      const __m128i vprod21x89ABCDEFlo = _mm_mullo_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF);
      const __m128i vprod21x89ABCDEFhi = _mm_mulhi_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod21x01234567lo, vprod21x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod21x01234567lo, vprod21x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod21x89ABCDEFlo, vprod21x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod21x89ABCDEFlo, vprod21x89ABCDEFhi));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vxi22x01234567 = _mm_cvtepu8_epi16(vi22x01234567);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 352 * sizeof(uint8_t)));
      const __m128i vxk22x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk22x01234567), vk_zero_point);
      const __m128i vi22x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i22 + 8));
      const __m128i vxi22x89ABCDEF = _mm_cvtepu8_epi16(vi22x89ABCDEF);
      const __m128i vk22x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 360 * sizeof(uint8_t)));
      const __m128i vxk22x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk22x89ABCDEF), vk_zero_point);
      i22 += 16;


      const __m128i vprod22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      const __m128i vprod22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);
      const __m128i vprod22x89ABCDEFlo = _mm_mullo_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);
      const __m128i vprod22x89ABCDEFhi = _mm_mulhi_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod22x01234567lo, vprod22x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod22x01234567lo, vprod22x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod22x89ABCDEFlo, vprod22x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod22x89ABCDEFlo, vprod22x89ABCDEFhi));

      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vxi23x01234567 = _mm_cvtepu8_epi16(vi23x01234567);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 368 * sizeof(uint8_t)));
      const __m128i vxk23x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk23x01234567), vk_zero_point);
      const __m128i vi23x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i23 + 8));
      const __m128i vxi23x89ABCDEF = _mm_cvtepu8_epi16(vi23x89ABCDEF);
      const __m128i vk23x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 376 * sizeof(uint8_t)));
      const __m128i vxk23x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk23x89ABCDEF), vk_zero_point);
      i23 += 16;


      const __m128i vprod23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
      const __m128i vprod23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);
      const __m128i vprod23x89ABCDEFlo = _mm_mullo_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF);
      const __m128i vprod23x89ABCDEFhi = _mm_mulhi_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod23x01234567lo, vprod23x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod23x01234567lo, vprod23x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod23x89ABCDEFlo, vprod23x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod23x89ABCDEFlo, vprod23x89ABCDEFhi));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vxi24x01234567 = _mm_cvtepu8_epi16(vi24x01234567);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 384 * sizeof(uint8_t)));
      const __m128i vxk24x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk24x01234567), vk_zero_point);
      const __m128i vi24x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i24 + 8));
      const __m128i vxi24x89ABCDEF = _mm_cvtepu8_epi16(vi24x89ABCDEF);
      const __m128i vk24x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 392 * sizeof(uint8_t)));
      const __m128i vxk24x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk24x89ABCDEF), vk_zero_point);
      i24 += 16;


      const __m128i vprod24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      const __m128i vprod24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);
      const __m128i vprod24x89ABCDEFlo = _mm_mullo_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);
      const __m128i vprod24x89ABCDEFhi = _mm_mulhi_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod24x01234567lo, vprod24x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod24x01234567lo, vprod24x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod24x89ABCDEFlo, vprod24x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod24x89ABCDEFlo, vprod24x89ABCDEFhi));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 400 * sizeof(uint8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x01234567), vk_zero_point);
        i0 += 8;


        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x01234567), vk_zero_point);
        i1 += 8;


        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x01234567), vk_zero_point);
        i2 += 8;


        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x01234567), vk_zero_point);
        i3 += 8;


        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x01234567), vk_zero_point);
        i4 += 8;


        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x01234567), vk_zero_point);
        i5 += 8;


        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x01234567), vk_zero_point);
        i6 += 8;


        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepu8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x01234567), vk_zero_point);
        i7 += 8;


        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepu8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x01234567), vk_zero_point);
        i8 += 8;


        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vxi9x01234567 = _mm_cvtepu8_epi16(vi9x01234567);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        const __m128i vxk9x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk9x01234567), vk_zero_point);
        i9 += 8;


        const __m128i vprod9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
        const __m128i vprod9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod9x01234567lo, vprod9x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod9x01234567lo, vprod9x01234567hi));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vxi10x01234567 = _mm_cvtepu8_epi16(vi10x01234567);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) (k + 160));
        const __m128i vxk10x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk10x01234567), vk_zero_point);
        i10 += 8;


        const __m128i vprod10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
        const __m128i vprod10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod10x01234567lo, vprod10x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod10x01234567lo, vprod10x01234567hi));

        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vxi11x01234567 = _mm_cvtepu8_epi16(vi11x01234567);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) (k + 176));
        const __m128i vxk11x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk11x01234567), vk_zero_point);
        i11 += 8;


        const __m128i vprod11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
        const __m128i vprod11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod11x01234567lo, vprod11x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod11x01234567lo, vprod11x01234567hi));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vxi12x01234567 = _mm_cvtepu8_epi16(vi12x01234567);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        const __m128i vxk12x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk12x01234567), vk_zero_point);
        i12 += 8;


        const __m128i vprod12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
        const __m128i vprod12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod12x01234567lo, vprod12x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod12x01234567lo, vprod12x01234567hi));

        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vxi13x01234567 = _mm_cvtepu8_epi16(vi13x01234567);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) (k + 208));
        const __m128i vxk13x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk13x01234567), vk_zero_point);
        i13 += 8;


        const __m128i vprod13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
        const __m128i vprod13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod13x01234567lo, vprod13x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod13x01234567lo, vprod13x01234567hi));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vxi14x01234567 = _mm_cvtepu8_epi16(vi14x01234567);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) (k + 224));
        const __m128i vxk14x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk14x01234567), vk_zero_point);
        i14 += 8;


        const __m128i vprod14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
        const __m128i vprod14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod14x01234567lo, vprod14x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod14x01234567lo, vprod14x01234567hi));

        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vxi15x01234567 = _mm_cvtepu8_epi16(vi15x01234567);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) (k + 240));
        const __m128i vxk15x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk15x01234567), vk_zero_point);
        i15 += 8;


        const __m128i vprod15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
        const __m128i vprod15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod15x01234567lo, vprod15x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod15x01234567lo, vprod15x01234567hi));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vxi16x01234567 = _mm_cvtepu8_epi16(vi16x01234567);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) (k + 256));
        const __m128i vxk16x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk16x01234567), vk_zero_point);
        i16 += 8;


        const __m128i vprod16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
        const __m128i vprod16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod16x01234567lo, vprod16x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod16x01234567lo, vprod16x01234567hi));

        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vxi17x01234567 = _mm_cvtepu8_epi16(vi17x01234567);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) (k + 272));
        const __m128i vxk17x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk17x01234567), vk_zero_point);
        i17 += 8;


        const __m128i vprod17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
        const __m128i vprod17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod17x01234567lo, vprod17x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod17x01234567lo, vprod17x01234567hi));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vxi18x01234567 = _mm_cvtepu8_epi16(vi18x01234567);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) (k + 288));
        const __m128i vxk18x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk18x01234567), vk_zero_point);
        i18 += 8;


        const __m128i vprod18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
        const __m128i vprod18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod18x01234567lo, vprod18x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod18x01234567lo, vprod18x01234567hi));

        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vxi19x01234567 = _mm_cvtepu8_epi16(vi19x01234567);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) (k + 304));
        const __m128i vxk19x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk19x01234567), vk_zero_point);
        i19 += 8;


        const __m128i vprod19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
        const __m128i vprod19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod19x01234567lo, vprod19x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod19x01234567lo, vprod19x01234567hi));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vxi20x01234567 = _mm_cvtepu8_epi16(vi20x01234567);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) (k + 320));
        const __m128i vxk20x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk20x01234567), vk_zero_point);
        i20 += 8;


        const __m128i vprod20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
        const __m128i vprod20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod20x01234567lo, vprod20x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod20x01234567lo, vprod20x01234567hi));

        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vxi21x01234567 = _mm_cvtepu8_epi16(vi21x01234567);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) (k + 336));
        const __m128i vxk21x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk21x01234567), vk_zero_point);
        i21 += 8;


        const __m128i vprod21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
        const __m128i vprod21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod21x01234567lo, vprod21x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod21x01234567lo, vprod21x01234567hi));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vxi22x01234567 = _mm_cvtepu8_epi16(vi22x01234567);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) (k + 352));
        const __m128i vxk22x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk22x01234567), vk_zero_point);
        i22 += 8;


        const __m128i vprod22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
        const __m128i vprod22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod22x01234567lo, vprod22x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod22x01234567lo, vprod22x01234567hi));

        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vxi23x01234567 = _mm_cvtepu8_epi16(vi23x01234567);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) (k + 368));
        const __m128i vxk23x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk23x01234567), vk_zero_point);
        i23 += 8;


        const __m128i vprod23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
        const __m128i vprod23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod23x01234567lo, vprod23x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod23x01234567lo, vprod23x01234567hi));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vxi24x01234567 = _mm_cvtepu8_epi16(vi24x01234567);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) (k + 384));
        const __m128i vxk24x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk24x01234567), vk_zero_point);
        i24 += 8;


        const __m128i vprod24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
        const __m128i vprod24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod24x01234567lo, vprod24x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod24x01234567lo, vprod24x01234567hi));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    const __m128i vk_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    for (; c >= 16; c -= 16) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 8));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 12));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t)));
      const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x01234567), vk_zero_point);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vxi0x89ABCDEF = _mm_cvtepu8_epi16(vi0x89ABCDEF);
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(uint8_t)));
      const __m128i vxk0x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x89ABCDEF), vk_zero_point);
      i0 += 16;


      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x89ABCDEFlo = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);
      const __m128i vprod0x89ABCDEFhi = _mm_mulhi_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod0x89ABCDEFlo, vprod0x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod0x89ABCDEFlo, vprod0x89ABCDEFhi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t)));
      const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x01234567), vk_zero_point);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi1x89ABCDEF = _mm_cvtepu8_epi16(vi1x89ABCDEF);
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(uint8_t)));
      const __m128i vxk1x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x89ABCDEF), vk_zero_point);
      i1 += 16;


      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x89ABCDEFlo = _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);
      const __m128i vprod1x89ABCDEFhi = _mm_mulhi_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod1x89ABCDEFlo, vprod1x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod1x89ABCDEFlo, vprod1x89ABCDEFhi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t)));
      const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x01234567), vk_zero_point);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi2x89ABCDEF = _mm_cvtepu8_epi16(vi2x89ABCDEF);
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(uint8_t)));
      const __m128i vxk2x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x89ABCDEF), vk_zero_point);
      i2 += 16;


      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x89ABCDEFlo = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);
      const __m128i vprod2x89ABCDEFhi = _mm_mulhi_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod2x89ABCDEFlo, vprod2x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod2x89ABCDEFlo, vprod2x89ABCDEFhi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t)));
      const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x01234567), vk_zero_point);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vxi3x89ABCDEF = _mm_cvtepu8_epi16(vi3x89ABCDEF);
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(uint8_t)));
      const __m128i vxk3x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x89ABCDEF), vk_zero_point);
      i3 += 16;


      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x89ABCDEFlo = _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);
      const __m128i vprod3x89ABCDEFhi = _mm_mulhi_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod3x89ABCDEFlo, vprod3x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod3x89ABCDEFlo, vprod3x89ABCDEFhi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t)));
      const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x01234567), vk_zero_point);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vxi4x89ABCDEF = _mm_cvtepu8_epi16(vi4x89ABCDEF);
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(uint8_t)));
      const __m128i vxk4x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x89ABCDEF), vk_zero_point);
      i4 += 16;


      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x89ABCDEFlo = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);
      const __m128i vprod4x89ABCDEFhi = _mm_mulhi_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod4x89ABCDEFlo, vprod4x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod4x89ABCDEFlo, vprod4x89ABCDEFhi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t)));
      const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x01234567), vk_zero_point);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vxi5x89ABCDEF = _mm_cvtepu8_epi16(vi5x89ABCDEF);
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(uint8_t)));
      const __m128i vxk5x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x89ABCDEF), vk_zero_point);
      i5 += 16;


      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x89ABCDEFlo = _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);
      const __m128i vprod5x89ABCDEFhi = _mm_mulhi_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod5x89ABCDEFlo, vprod5x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod5x89ABCDEFlo, vprod5x89ABCDEFhi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(uint8_t)));
      const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x01234567), vk_zero_point);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vxi6x89ABCDEF = _mm_cvtepu8_epi16(vi6x89ABCDEF);
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(uint8_t)));
      const __m128i vxk6x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x89ABCDEF), vk_zero_point);
      i6 += 16;


      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x89ABCDEFlo = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);
      const __m128i vprod6x89ABCDEFhi = _mm_mulhi_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod6x89ABCDEFlo, vprod6x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod6x89ABCDEFlo, vprod6x89ABCDEFhi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepu8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(uint8_t)));
      const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x01234567), vk_zero_point);
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vxi7x89ABCDEF = _mm_cvtepu8_epi16(vi7x89ABCDEF);
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(uint8_t)));
      const __m128i vxk7x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x89ABCDEF), vk_zero_point);
      i7 += 16;


      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x89ABCDEFlo = _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);
      const __m128i vprod7x89ABCDEFhi = _mm_mulhi_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod7x89ABCDEFlo, vprod7x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod7x89ABCDEFlo, vprod7x89ABCDEFhi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepu8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(uint8_t)));
      const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x01234567), vk_zero_point);
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vxi8x89ABCDEF = _mm_cvtepu8_epi16(vi8x89ABCDEF);
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(uint8_t)));
      const __m128i vxk8x89ABCDEF = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x89ABCDEF), vk_zero_point);
      i8 += 16;


      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x89ABCDEFlo = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);
      const __m128i vprod8x89ABCDEFhi = _mm_mulhi_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vprod8x89ABCDEFlo, vprod8x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vprod8x89ABCDEFlo, vprod8x89ABCDEFhi));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(uint8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);
      __m128 vscaled89AB = _mm_cvtepi32_ps(vacc89AB);
      __m128 vscaledCDEF = _mm_cvtepi32_ps(vaccCDEF);

      const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);
      vscaled89AB = _mm_mul_ps(vscaled89AB, vscale);
      vscaledCDEF = _mm_mul_ps(vscaledCDEF, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);
      vscaled89AB = _mm_min_ps(vscaled89AB, voutput_max_less_zero_point);
      vscaledCDEF = _mm_min_ps(vscaledCDEF, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 16);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x01234567), vk_zero_point);
        i0 += 8;


        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 16));
        const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x01234567), vk_zero_point);
        i1 += 8;


        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 32));
        const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x01234567), vk_zero_point);
        i2 += 8;


        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x01234567), vk_zero_point);
        i3 += 8;


        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 64));
        const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x01234567), vk_zero_point);
        i4 += 8;


        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 80));
        const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x01234567), vk_zero_point);
        i5 += 8;


        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x01234567), vk_zero_point);
        i6 += 8;


        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepu8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 112));
        const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x01234567), vk_zero_point);
        i7 += 8;


        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepu8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 128));
        const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x01234567), vk_zero_point);
        i8 += 8;


        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

        k += 8;

        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        w = (const void*) ((const int32_t*) w + 8);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_f32_vcvt_ukernel__avx_x32(
    size_t n,
    const uint8_t* x,
    float* y,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m128i vminus_zero_point = _mm_load_si128((const __m128i*) params->avx.minus_zero_point);
  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  for (; n >= 32 * sizeof(uint8_t); n -= 32 * sizeof(uint8_t)) {
    __m128i vx0123 = _mm_cvtepu8_epi32(_mm_loadu_si32(x));
    __m128i vx4567 = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 4));
    __m128i vx89AB = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 8));
    __m128i vxCDEF = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 12));
    __m128i vxGHIJ = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 16));
    __m128i vxKLMN = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 20));
    __m128i vxOPQR = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 24));
    __m128i vxSTUV = _mm_cvtepu8_epi32(_mm_loadu_si32(x + 28));
    x += 32;

    vx0123 = _mm_add_epi32(vx0123, vminus_zero_point);
    vx4567 = _mm_add_epi32(vx4567, vminus_zero_point);
    vx89AB = _mm_add_epi32(vx89AB, vminus_zero_point);
    vxCDEF = _mm_add_epi32(vxCDEF, vminus_zero_point);
    vxGHIJ = _mm_add_epi32(vxGHIJ, vminus_zero_point);
    vxKLMN = _mm_add_epi32(vxKLMN, vminus_zero_point);
    vxOPQR = _mm_add_epi32(vxOPQR, vminus_zero_point);
    vxSTUV = _mm_add_epi32(vxSTUV, vminus_zero_point);

    const __m256i vx01234567 = _mm256_insertf128_si256(_mm256_castsi128_si256(vx0123), vx4567, 1);
    const __m256i vx89ABCDEF = _mm256_insertf128_si256(_mm256_castsi128_si256(vx89AB), vxCDEF, 1);
    const __m256i vxGHIJKLMN = _mm256_insertf128_si256(_mm256_castsi128_si256(vxGHIJ), vxKLMN, 1);
    const __m256i vxOPQRSTUV = _mm256_insertf128_si256(_mm256_castsi128_si256(vxOPQR), vxSTUV, 1);

    __m256 vy01234567 = _mm256_cvtepi32_ps(vx01234567);
    __m256 vy89ABCDEF = _mm256_cvtepi32_ps(vx89ABCDEF);
    __m256 vyGHIJKLMN = _mm256_cvtepi32_ps(vxGHIJKLMN);
    __m256 vyOPQRSTUV = _mm256_cvtepi32_ps(vxOPQRSTUV);

    vy01234567 = _mm256_mul_ps(vy01234567, vscale);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vscale);
    vyGHIJKLMN = _mm256_mul_ps(vyGHIJKLMN, vscale);
    vyOPQRSTUV = _mm256_mul_ps(vyOPQRSTUV, vscale);

    _mm256_storeu_ps(y, vy01234567);
    _mm256_storeu_ps(y + 8, vy89ABCDEF);
    _mm256_storeu_ps(y + 16, vyGHIJKLMN);
    _mm256_storeu_ps(y + 24, vyOPQRSTUV);
    y += 32;
  }
  for (; n >= 4 * sizeof(uint8_t); n -= 4 * sizeof(uint8_t)) {
    __m128i vx = _mm_cvtepu8_epi32(_mm_loadu_si32(x));
    vx = _mm_add_epi32(vx, vminus_zero_point);
    x += 4;

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, _mm256_castps256_ps128(vscale));

    _mm_storeu_ps(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint8_t));
    assert(n <= 3 * sizeof(uint8_t));

    __m128i vx = _mm_cvtepu8_epi32(_mm_loadu_si32(x));
    vx = _mm_add_epi32(vx, vminus_zero_point);

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, _mm256_castps256_ps128(vscale));

    if (n & (2 * sizeof(uint8_t))) {
      _mm_storel_pi((__m64*) y, vy);
      vy = _mm_movehl_ps(vy, vy);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      _mm_store_ss(y, vy);
    }
  }
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = 0;
    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    const __m128i vzero = _mm_setzero_si128();
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
      a0 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb01, vzero), vb_zero_point);
      const __m128i vxb1 = _mm_sub_epi16(_mm_unpackhi_epi8(vb01, vzero), vb_zero_point);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
      const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb23, vzero), vb_zero_point);
      const __m128i vxb3 = _mm_sub_epi16(_mm_unpackhi_epi8(vb23, vzero), vb_zero_point);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const void*) ((const uint8_t*) w + 32);
      k += 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = 0;
    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    const __m128i vzero = _mm_setzero_si128();
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepu8_epi16(va1);
      a1 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb01, vzero), vb_zero_point);
      const __m128i vxb1 = _mm_sub_epi16(_mm_unpackhi_epi8(vb01, vzero), vb_zero_point);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
      const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb23, vzero), vb_zero_point);
      const __m128i vxb3 = _mm_sub_epi16(_mm_unpackhi_epi8(vb23, vzero), vb_zero_point);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));

      w = (const void*) ((const uint8_t*) w + 32);
      k += 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      *((uint32_t*) c1) = (uint32_t) _mm_extract_epi32(vout, 1);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        *((uint16_t*) c1) = (uint16_t) _mm_extract_epi16(vout, 2);
        c1 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
        *c1 = (uint8_t) _mm_extract_epi8(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128(
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
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  uint8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
      const __m128i vzero = _mm_setzero_si128();
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
        a0 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb01, vzero), vb_zero_point);
        const __m128i vxb1 = _mm_sub_epi16(_mm_unpackhi_epi8(vb01, vzero), vb_zero_point);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
        const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb23, vzero), vb_zero_point);
        const __m128i vxb3 = _mm_sub_epi16(_mm_unpackhi_epi8(vb23, vzero), vb_zero_point);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const uint8_t*) w + 32);
        k += 8 * sizeof(uint8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128(
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
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = 0;
      const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
      const __m128i vzero = _mm_setzero_si128();
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepu8_epi16(va1);
        a1 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb01, vzero), vb_zero_point);
        const __m128i vxb1 = _mm_sub_epi16(_mm_unpackhi_epi8(vb01, vzero), vb_zero_point);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
        const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb23, vzero), vb_zero_point);
        const __m128i vxb3 = _mm_sub_epi16(_mm_unpackhi_epi8(vb23, vzero), vb_zero_point);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));

        w = (const void*) ((const uint8_t*) w + 32);
        k += 8 * sizeof(uint8_t);
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc01x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      *((uint32_t*) c1) = (uint32_t) _mm_extract_epi32(vout, 1);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c1) = (uint16_t) _mm_extract_epi16(vout, 2);
        c1 += 2;
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c1 = (uint8_t) _mm_extract_epi8(vout, 4);
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vadd_minmax_ukernel__avx_mul32_ld32_x8(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_addsub_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse4.bias);
  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4.a_multiplier);
  const __m128i vb_multiplier = _mm_load_si128((const __m128i*) params->sse4.b_multiplier);
  const __m128i vshift = _mm_loadu_si32(params->sse4.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4.output_max);

  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    const __m128i va0123 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_a));
    const __m128i vb0123 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_b));
    const __m128i va4567 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_a + 4));
    const __m128i vb4567 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_b + 4));
    input_a += 8;
    input_b += 8;

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vb0123, vb_multiplier));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vb4567, vb_multiplier));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    {
      const __m128i va0123 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_a));
      const __m128i vb0123 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_b));
      const __m128i va4567 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_a + 4));
      const __m128i vb4567 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_b + 4));

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vb0123, vb_multiplier));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vb4567, vb_multiplier));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (n & (4 * sizeof(uint8_t))) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (n & (2 * sizeof(uint8_t))) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (n & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_x8(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_addsub_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const __m128i va_multiplier = _mm_load_si128((const __m128i*) params->sse4.a_multiplier);
  const __m128i vshift = _mm_loadu_si32(params->sse4.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse4.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse4.output_max);

  __m128i vbias = _mm_cvtsi32_si128(params->sse4.b_multiplier[0] * (int32_t) *input_b);
  vbias = _mm_shuffle_epi32(vbias, _MM_SHUFFLE(0, 0, 0, 0));
  vbias = _mm_add_epi32(vbias, _mm_load_si128((const __m128i*) params->sse4.bias));
  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    const __m128i va0123 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_a));
    const __m128i va4567 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_a + 4));
    input_a += 8;
    input_b += 8;

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    {
      const __m128i va0123 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_a));
      const __m128i va4567 = _mm_cvtepu8_epi32(_mm_loadu_si32(input_a + 4));

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_mullo_epi32(va0123, va_multiplier));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_mullo_epi32(va4567, va_multiplier));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (n & (4 * sizeof(uint8_t))) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (n & (2 * sizeof(uint8_t))) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (n & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS

{
  const __m128i va_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.a_zero_point);
  const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.b_zero_point);
  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->fp32_sse2.output_max);

  for (; n >= 16 * sizeof(uint8_t); n -= 16 * sizeof(uint8_t)) {
    const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i vb01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
    const __m128i va89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m128i vb89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    input_a += 16;
    input_b += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);
    const __m128i vxb89ABCDEF = _mm_sub_epi16(vb89ABCDEF, vb_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb89ABCDEF);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb89ABCDEF);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      const __m128i vb01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
      const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(n >= (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        n -= 8 * sizeof(uint8_t);
      } else {
        if (n & (4 * sizeof(uint8_t))) {
          *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (n & (2 * sizeof(uint8_t))) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (n & (1 * sizeof(uint8_t))) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        n = 0;
      }
    } while (n != 0);
  }
}

void xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS

{
  const __m128i va_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.a_zero_point);
  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->fp32_sse2.output_max);

  __m128i vxb = _mm_sub_epi16(
    _mm_shuffle_epi32(_mm_cvtsi32_si128(UINT32_C(0x00010001) * (uint32_t) (uint16_t) (int16_t) *input_b), 0),
    _mm_load_si128((const __m128i*) params->fp32_sse2.b_zero_point));
  for (; n >= 16 * sizeof(uint8_t); n -= 16 * sizeof(uint8_t)) {
    const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i va89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    input_a += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      input_a += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(n >= (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        n -= 8 * sizeof(uint8_t);
      } else {
        if (n & (4 * sizeof(uint8_t))) {
          *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (n & (2 * sizeof(uint8_t))) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (n & (1 * sizeof(uint8_t))) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        n = 0;
      }
    } while (n != 0);
  }
}

void xnn_x8_lut_ukernel__avx_x64(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const uint8_t t[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(n != 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m128i vt0 = _mm_load_si128((const __m128i*) t);
  const __m128i vt1 = _mm_load_si128((const __m128i*) (t + 16));
  const __m128i vt2 = _mm_load_si128((const __m128i*) (t + 32));
  const __m128i vt3 = _mm_load_si128((const __m128i*) (t + 48));
  const __m128i vt4 = _mm_load_si128((const __m128i*) (t + 64));
  const __m128i vt5 = _mm_load_si128((const __m128i*) (t + 80));
  const __m128i vt6 = _mm_load_si128((const __m128i*) (t + 96));
  const __m128i vt7 = _mm_load_si128((const __m128i*) (t + 112));
  const __m128i vt8 = _mm_load_si128((const __m128i*) (t + 128));
  const __m128i vt9 = _mm_load_si128((const __m128i*) (t + 144));
  const __m128i vtA = _mm_load_si128((const __m128i*) (t + 160));
  const __m128i vtB = _mm_load_si128((const __m128i*) (t + 176));
  const __m128i vtC = _mm_load_si128((const __m128i*) (t + 192));
  const __m128i vtD = _mm_load_si128((const __m128i*) (t + 208));
  const __m128i vtE = _mm_load_si128((const __m128i*) (t + 224));
  const __m128i vtF = _mm_load_si128((const __m128i*) (t + 240));

  const __m128i vtable0 = vt0;
  const __m128i vtable1 = _mm_xor_si128(vt0, vt1);
  const __m128i vtable2 = _mm_xor_si128(vt1, vt2);
  const __m128i vtable3 = _mm_xor_si128(vt2, vt3);
  const __m128i vtable4 = _mm_xor_si128(vt3, vt4);
  const __m128i vtable5 = _mm_xor_si128(vt4, vt5);
  const __m128i vtable6 = _mm_xor_si128(vt5, vt6);
  const __m128i vtable7 = _mm_xor_si128(vt6, vt7);
  const __m128i vtable8 = _mm_xor_si128(_mm_xor_si128(vt7, vt8), vtable0);
  const __m128i vtable9 = _mm_xor_si128(_mm_xor_si128(vt8, vt9), vtable1);
  const __m128i vtableA = _mm_xor_si128(_mm_xor_si128(vt9, vtA), vtable2);
  const __m128i vtableB = _mm_xor_si128(_mm_xor_si128(vtA, vtB), vtable3);
  const __m128i vtableC = _mm_xor_si128(_mm_xor_si128(vtB, vtC), vtable4);
  const __m128i vtableD = _mm_xor_si128(_mm_xor_si128(vtC, vtD), vtable5);
  const __m128i vtableE = _mm_xor_si128(_mm_xor_si128(vtD, vtE), vtable6);
  const __m128i vtableF = _mm_xor_si128(_mm_xor_si128(vtE, vtF), vtable7);

  const __m128i voffset = _mm_set1_epi8(16);
  for (; n >= 64 * sizeof(uint8_t); n -= 64 * sizeof(uint8_t)) {
    __m128i vx0 = _mm_loadu_si128((const __m128i*) x);
    __m128i vx1 = _mm_loadu_si128((const __m128i*) (x + 16));
    __m128i vx2 = _mm_loadu_si128((const __m128i*) (x + 32));
    __m128i vx3 = _mm_loadu_si128((const __m128i*) (x + 48));
    x += 64;

    __m128i vy0 = _mm_shuffle_epi8(vtable0, vx0);
    __m128i vy1 = _mm_shuffle_epi8(vtable0, vx1);
    __m128i vy2 = _mm_shuffle_epi8(vtable0, vx2);
    __m128i vy3 = _mm_shuffle_epi8(vtable0, vx3);

    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vx3 = _mm_sub_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable1, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable1, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable1, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable1, vx3));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vx3 = _mm_sub_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable2, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable2, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable2, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable2, vx3));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vx3 = _mm_sub_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable3, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable3, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable3, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable3, vx3));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vx3 = _mm_sub_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable4, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable4, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable4, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable4, vx3));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vx3 = _mm_sub_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable5, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable5, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable5, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable5, vx3));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vx3 = _mm_sub_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable6, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable6, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable6, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable6, vx3));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vx3 = _mm_sub_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable7, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable7, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable7, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable7, vx3));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vx3 = _mm_sub_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable8, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable8, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable8, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable8, vx3));

    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vx3 = _mm_subs_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable9, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable9, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable9, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtable9, vx3));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vx3 = _mm_subs_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableA, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableA, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableA, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtableA, vx3));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vx3 = _mm_subs_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableB, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableB, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableB, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtableB, vx3));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vx3 = _mm_subs_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableC, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableC, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableC, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtableC, vx3));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vx3 = _mm_subs_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableD, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableD, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableD, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtableD, vx3));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vx3 = _mm_subs_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableE, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableE, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableE, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtableE, vx3));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vx3 = _mm_subs_epi8(vx3, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableF, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableF, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableF, vx2));
    vy3 = _mm_xor_si128(vy3, _mm_shuffle_epi8(vtableF, vx3));

    _mm_storeu_si128((__m128i*) y, vy0);
    _mm_storeu_si128((__m128i*) (y + 16), vy1);
    _mm_storeu_si128((__m128i*) (y + 32), vy2);
    _mm_storeu_si128((__m128i*) (y + 48), vy3);
    y += 64;
  }
  for (; n >= 16 * sizeof(uint8_t); n -= 16 * sizeof(uint8_t)) {
    __m128i vx = _mm_loadu_si128((const __m128i*) x);
    x += 16;

    __m128i vy = _mm_shuffle_epi8(vtable0, vx);

    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable1, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable2, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable3, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable4, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable5, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable6, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable7, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable8, vx));

    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable9, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableA, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableB, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableC, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableD, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableE, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableF, vx));

    _mm_storeu_si128((__m128i*) y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    __m128i vx = _mm_loadu_si128((const __m128i*) x);

    __m128i vy = _mm_shuffle_epi8(vtable0, vx);

    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable1, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable2, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable3, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable4, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable5, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable6, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable7, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable8, vx));

    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable9, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableA, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableB, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableC, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableD, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableE, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableF, vx));

    if (n & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) y, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      y += 8;
    }
    if (n & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(y, vy);
      vy = _mm_srli_epi64(vy, 32);
      y += 4;
    }
    if (n & (2 * sizeof(uint8_t))) {
      *((uint16_t*) y) = (uint16_t) _mm_extract_epi16(vy, 0);
      vy = _mm_srli_epi32(vy, 16);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      *y = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
