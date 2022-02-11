// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/ibilinear.h>
#include <xnnpack/igemm.h>
#include <xnnpack/math.h>
#include <xnnpack/vmulcaddc.h>
#include <xnnpack/vunary.h>


void xnn_f16_dwconv_minmax_ukernel_up16x3__fma3(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      w += 64;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_NO_EXC));
      o += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        *((uint32_t*) o) = (uint32_t) _mm_cvtsi128_si32(vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *((uint16_t*) o) = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_up16x4__fma3(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      const __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      w += 80;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_NO_EXC));
      o += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        *((uint32_t*) o) = (uint32_t) _mm_cvtsi128_si32(vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *((uint16_t*) o) = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_up16x9__fma3(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      const __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 8)));
      i4 += 16;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 80)));
      const __m256 vk4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 8)));
      i5 += 16;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 96)));
      const __m256 vk5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 104)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 8)));
      i6 += 16;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 112)));
      const __m256 vk6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      const __m256 vi7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 8)));
      i7 += 16;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 128)));
      const __m256 vk7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 136)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      const __m256 vi8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 8)));
      i8 += 16;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 144)));
      const __m256 vk8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 152)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x89ABCDEF, vk8x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_NO_EXC));

      w += 160;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_NO_EXC));
      o += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        *((uint32_t*) o) = (uint32_t) _mm_cvtsi128_si32(vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *((uint16_t*) o) = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint16_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint16_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint16_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint16_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint16_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint16_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint16_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint16_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint16_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint16_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint16_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint16_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint16_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint16_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint16_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint16_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint16_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint16_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint16_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint16_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint16_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint16_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint16_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint16_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint16_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint16_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint16_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint16_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint16_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint16_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint16_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint16_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 8)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));
      i9 += 8;

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 80)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));
      i10 += 8;

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));
      i11 += 8;

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 96)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));
      i12 += 8;

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 104)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));
      i13 += 8;

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 112)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));
      i14 += 8;

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));
      i15 += 8;

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 128)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));
      i16 += 8;

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 136)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));
      i17 += 8;

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 144)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));
      i18 += 8;

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 152)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));
      i19 += 8;

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 160)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));
      i20 += 8;

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 168)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));
      i21 += 8;

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 176)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));
      i22 += 8;

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 184)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));
      i23 += 8;

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 192)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));
      i24 += 8;

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 200)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      w += 208;

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_NO_EXC));

      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_NO_EXC));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 104)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 136)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 152)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 168)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 176)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 184)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p1), _MM_FROUND_NO_EXC));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 200)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_NO_EXC));

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_NO_EXC));

      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_NO_EXC);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        *((uint32_t*) o) = (uint32_t) _mm_cvtsi128_si32(vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *((uint16_t*) o) = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_ibilinear_ukernel__fma3_c8(
    size_t output_pixels,
    size_t channels,
    const void**restrict input,
    size_t input_offset,
    const void*restrict weights,
    void*restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = (const uint16_t*) ((uintptr_t) input[0] + input_offset);
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input[1] + input_offset);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input[2] + input_offset);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const __m256 valphahv = _mm256_cvtph_ps(_mm_castps_si128(_mm_broadcast_ss(weights)));
    const __m256 valphah = _mm256_permute_ps(valphahv, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256 valphav = _mm256_permute_ps(valphahv, _MM_SHUFFLE(3, 1, 3, 1));
    weights = (const uint16_t*) weights + 2;

    size_t c = channels;
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const __m256 vtl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vtr = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;
      const __m256 vbl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;
      const __m256 vbr = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vtd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vtr, vtl), _MM_FROUND_NO_EXC));
      const __m256 vbd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vbr, vbl), _MM_FROUND_NO_EXC));

      const __m256 vt = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vtd, valphah, vtl), _MM_FROUND_NO_EXC));
      const __m256 vb = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vbd, valphah, vbl), _MM_FROUND_NO_EXC));

      const __m256 vd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb, vt), _MM_FROUND_NO_EXC));

      const __m128i vo = _mm256_cvtps_ph(_mm256_fmadd_ps(vd, valphav, vt), _MM_FROUND_NO_EXC);

      _mm_storeu_si128((__m128i*) o, vo);
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m256 vtl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vtr = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;
      const __m256 vbl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;
      const __m256 vbr = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vtd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vtr, vtl), _MM_FROUND_NO_EXC));
      const __m256 vbd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vbr, vbl), _MM_FROUND_NO_EXC));

      const __m256 vt = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vtd, valphah, vtl), _MM_FROUND_NO_EXC));
      const __m256 vb = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vbd, valphah, vbl), _MM_FROUND_NO_EXC));

      const __m256 vd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb, vt), _MM_FROUND_NO_EXC));

      __m128i vo = _mm256_cvtps_ph(_mm256_fmadd_ps(vd, valphav, vt), _MM_FROUND_NO_EXC);
      if (c & (4 * sizeof(uint16_t))) {
        _mm_storel_epi64((__m128i*) o, vo);
        vo = _mm_unpackhi_epi64(vo, vo);
        o += 4;
      }
      if (c & (2 * sizeof(uint16_t))) {
        *((uint32_t*) o) = (uint32_t) _mm_cvtsi128_si32(vo);
        vo = _mm_srli_epi64(vo, 32);
        o += 2;
      }
      if (c & (1 * sizeof(uint16_t))) {
        *((uint16_t*) o) = (uint16_t) _mm_extract_epi16(vo, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x(
    size_t rows,
    size_t channels,
    const void*restrict input,
    size_t input_stride,
    const void*restrict weights,
    void*restrict output,
    size_t output_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  const uint16_t* i0 = (const uint16_t*) input;
  uint16_t* o0 = (uint16_t*) output;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const __m256 vmin = _mm256_load_ps(params->avx.min);
  const __m256 vmax = _mm256_load_ps(params->avx.max);
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const uint16_t* w = (const uint16_t*) weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const __m256 vscale = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));

      __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vbias = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 8)));
      w += 16;

      vacc0 = _mm256_fmadd_ps(vacc0, vscale, vbias);
      vacc1 = _mm256_fmadd_ps(vacc1, vscale, vbias);

      vacc0 = _mm256_max_ps(vacc0, vmin);
      vacc1 = _mm256_max_ps(vacc1, vmin);

      vacc0 = _mm256_min_ps(vacc0, vmax);
      vacc1 = _mm256_min_ps(vacc1, vmax);

      _mm_storeu_si128((__m128i*) o0, _mm256_cvtps_ph(vacc0, _MM_FROUND_NO_EXC));
      o0 += 8;
      _mm_storeu_si128((__m128i*) o1, _mm256_cvtps_ph(vacc1, _MM_FROUND_NO_EXC));
      o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m256 vscale = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));

      __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 = (const uint16_t*) ((uintptr_t) i0 + c);
      __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 = (const uint16_t*) ((uintptr_t) i1 + c);

      const __m256 vbias = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 8)));

      vacc0 = _mm256_fmadd_ps(vacc0, vscale, vbias);
      vacc1 = _mm256_fmadd_ps(vacc1, vscale, vbias);

      vacc0 = _mm256_max_ps(vacc0, vmin);
      vacc1 = _mm256_max_ps(vacc1, vmin);

      vacc0 = _mm256_min_ps(vacc0, vmax);
      vacc1 = _mm256_min_ps(vacc1, vmax);

      __m128i vh0 = _mm256_cvtps_ph(vacc0, _MM_FROUND_NO_EXC);
      __m128i vh1 = _mm256_cvtps_ph(vacc1, _MM_FROUND_NO_EXC);

      if (c & (4 * sizeof(uint16_t))) {
        _mm_storel_epi64((__m128i*) o0, vh0);
        _mm_storel_epi64((__m128i*) o1, vh1);

        vh0 = _mm_unpackhi_epi64(vh0, vh0);
        vh1 = _mm_unpackhi_epi64(vh1, vh1);

        o0 += 4;
        o1 += 4;
      }
      if (c & (2 * sizeof(uint16_t))) {
        *((uint32_t*) o0) = (uint32_t) _mm_cvtsi128_si32(vh0);
        *((uint32_t*) o1) = (uint32_t) _mm_cvtsi128_si32(vh1);

        vh0 = _mm_srli_epi64(vh0, 32);
        vh1 = _mm_srli_epi64(vh1, 32);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(uint16_t))) {
        *o0 = (uint16_t) _mm_extract_epi16(vh0, 0);
        *o1 = (uint16_t) _mm_extract_epi16(vh1, 0);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_dwconv_minmax_ukernel_up16x3__fma3(
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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0);

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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);


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

void xnn_f32_dwconv_minmax_ukernel_up16x4__fma3(
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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0);

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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);


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

void xnn_f32_dwconv_minmax_ukernel_up16x9__fma3(
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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      const __m256 vi4x89ABCDEF = _mm256_loadu_ps(i4 + 8);
      i4 += 16;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      const __m256 vk4x89ABCDEF = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      const __m256 vi5x89ABCDEF = _mm256_loadu_ps(i5 + 8);
      i5 += 16;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      const __m256 vk5x89ABCDEF = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      const __m256 vi6x89ABCDEF = _mm256_loadu_ps(i6 + 8);
      i6 += 16;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      const __m256 vk6x89ABCDEF = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      const __m256 vi7x89ABCDEF = _mm256_loadu_ps(i7 + 8);
      i7 += 16;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      const __m256 vk7x89ABCDEF = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      const __m256 vi8x89ABCDEF = _mm256_loadu_ps(i8 + 8);
      i8 += 16;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      const __m256 vk8x89ABCDEF = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi8x89ABCDEF, vk8x89ABCDEF, vacc89ABCDEFp0);

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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);

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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);
      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);

      const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);
      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);
      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);

      const __m256 vi8x01234567 = _mm256_maskload_ps(i8, vmask);
      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);


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

void xnn_f32_dwconv_minmax_ukernel_up8x25__fma3(
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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);

      const __m256 vi9x01234567 = _mm256_loadu_ps(i9);
      i9 += 8;

      const __m256 vk9x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0);

      const __m256 vi10x01234567 = _mm256_loadu_ps(i10);
      i10 += 8;

      const __m256 vk10x01234567 = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0);

      const __m256 vi11x01234567 = _mm256_loadu_ps(i11);
      i11 += 8;

      const __m256 vk11x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0);

      const __m256 vi12x01234567 = _mm256_loadu_ps(i12);
      i12 += 8;

      const __m256 vk12x01234567 = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0);

      const __m256 vi13x01234567 = _mm256_loadu_ps(i13);
      i13 += 8;

      const __m256 vk13x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0);

      const __m256 vi14x01234567 = _mm256_loadu_ps(i14);
      i14 += 8;

      const __m256 vk14x01234567 = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0);

      const __m256 vi15x01234567 = _mm256_loadu_ps(i15);
      i15 += 8;

      const __m256 vk15x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0);

      const __m256 vi16x01234567 = _mm256_loadu_ps(i16);
      i16 += 8;

      const __m256 vk16x01234567 = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0);

      const __m256 vi17x01234567 = _mm256_loadu_ps(i17);
      i17 += 8;

      const __m256 vk17x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0);

      const __m256 vi18x01234567 = _mm256_loadu_ps(i18);
      i18 += 8;

      const __m256 vk18x01234567 = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0);

      const __m256 vi19x01234567 = _mm256_loadu_ps(i19);
      i19 += 8;

      const __m256 vk19x01234567 = _mm256_load_ps(w + 160);
      vacc01234567p0 = _mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0);

      const __m256 vi20x01234567 = _mm256_loadu_ps(i20);
      i20 += 8;

      const __m256 vk20x01234567 = _mm256_load_ps(w + 168);
      vacc01234567p0 = _mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0);

      const __m256 vi21x01234567 = _mm256_loadu_ps(i21);
      i21 += 8;

      const __m256 vk21x01234567 = _mm256_load_ps(w + 176);
      vacc01234567p0 = _mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0);

      const __m256 vi22x01234567 = _mm256_loadu_ps(i22);
      i22 += 8;

      const __m256 vk22x01234567 = _mm256_load_ps(w + 184);
      vacc01234567p0 = _mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0);

      const __m256 vi23x01234567 = _mm256_loadu_ps(i23);
      i23 += 8;

      const __m256 vk23x01234567 = _mm256_load_ps(w + 192);
      vacc01234567p0 = _mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0);

      const __m256 vi24x01234567 = _mm256_loadu_ps(i24);
      i24 += 8;

      const __m256 vk24x01234567 = _mm256_load_ps(w + 200);
      vacc01234567p0 = _mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0);

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
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
      const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);
      const __m256 vk5x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);

      const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);
      const __m256 vk6x01234567 = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);
      const __m256 vk7x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);

      const __m256 vi8x01234567 = _mm256_maskload_ps(i8, vmask);
      const __m256 vk8x01234567 = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);

      const __m256 vi9x01234567 = _mm256_maskload_ps(i9, vmask);
      const __m256 vk9x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0);

      const __m256 vi10x01234567 = _mm256_maskload_ps(i10, vmask);
      const __m256 vk10x01234567 = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0);

      const __m256 vi11x01234567 = _mm256_maskload_ps(i11, vmask);
      const __m256 vk11x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0);

      const __m256 vi12x01234567 = _mm256_maskload_ps(i12, vmask);
      const __m256 vk12x01234567 = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0);

      const __m256 vi13x01234567 = _mm256_maskload_ps(i13, vmask);
      const __m256 vk13x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0);

      const __m256 vi14x01234567 = _mm256_maskload_ps(i14, vmask);
      const __m256 vk14x01234567 = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0);

      const __m256 vi15x01234567 = _mm256_maskload_ps(i15, vmask);
      const __m256 vk15x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0);

      const __m256 vi16x01234567 = _mm256_maskload_ps(i16, vmask);
      const __m256 vk16x01234567 = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0);

      const __m256 vi17x01234567 = _mm256_maskload_ps(i17, vmask);
      const __m256 vk17x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0);

      const __m256 vi18x01234567 = _mm256_maskload_ps(i18, vmask);
      const __m256 vk18x01234567 = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0);

      const __m256 vi19x01234567 = _mm256_maskload_ps(i19, vmask);
      const __m256 vk19x01234567 = _mm256_load_ps(w + 160);
      vacc01234567p0 = _mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0);

      const __m256 vi20x01234567 = _mm256_maskload_ps(i20, vmask);
      const __m256 vk20x01234567 = _mm256_load_ps(w + 168);
      vacc01234567p0 = _mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0);

      const __m256 vi21x01234567 = _mm256_maskload_ps(i21, vmask);
      const __m256 vk21x01234567 = _mm256_load_ps(w + 176);
      vacc01234567p0 = _mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0);

      const __m256 vi22x01234567 = _mm256_maskload_ps(i22, vmask);
      const __m256 vk22x01234567 = _mm256_load_ps(w + 184);
      vacc01234567p0 = _mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0);

      const __m256 vi23x01234567 = _mm256_maskload_ps(i23, vmask);
      const __m256 vk23x01234567 = _mm256_load_ps(w + 192);
      vacc01234567p0 = _mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0);

      const __m256 vi24x01234567 = _mm256_maskload_ps(i24, vmask);
      const __m256 vk24x01234567 = _mm256_load_ps(w + 200);
      vacc01234567p0 = _mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0);


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

void xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast(
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

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);

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

void xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    while (k >= 4 * sizeof(float)) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 += 4;


      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);


      w += 64;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 = (const float*) ((uintptr_t) a0 + k);

      const __m256 vzero = _mm256_setzero_ps();

      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);


      w += 64;
    }

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

void xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    w += 16;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 += 4;
      __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
      a1 += 4;
      __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
      a2 += 4;
      __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
      a3 += 4;


      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c0, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c0, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c0, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc0, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc0, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc0, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c1, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c1, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c1, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc1, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc1, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc1, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c2, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c2, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c2, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc2, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc2, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc2, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c3, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c3, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c3, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc3, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc3, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc3, vacc3x89ABCDEF);


      w += 64;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const __m256 vzero = _mm256_setzero_ps();

      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc3x89ABCDEF);


      w += 64;
    }

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc1x01234567 = _mm256_max_ps(vacc1x01234567, vmin);
    vacc2x01234567 = _mm256_max_ps(vacc2x01234567, vmin);
    vacc3x01234567 = _mm256_max_ps(vacc3x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = _mm256_max_ps(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = _mm256_max_ps(vacc2x89ABCDEF, vmin);
    vacc3x89ABCDEF = _mm256_max_ps(vacc3x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc1x01234567 = _mm256_min_ps(vacc1x01234567, vmax);
    vacc2x01234567 = _mm256_min_ps(vacc2x01234567, vmax);
    vacc3x01234567 = _mm256_min_ps(vacc3x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = _mm256_min_ps(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = _mm256_min_ps(vacc2x89ABCDEF, vmax);
    vacc3x89ABCDEF = _mm256_min_ps(vacc3x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
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

      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast(
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

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567, vacc4x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEF, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEF, vacc4x89ABCDEF);

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

void xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast(
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

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
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

void xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast(
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
      while (k >= 4 * sizeof(float)) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 += 4;


        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);


        w += 64;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 = (const float*) ((uintptr_t) a0 + k);

        const __m256 vzero = _mm256_setzero_ps();

        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);


        w += 64;
      }
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

void xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast(
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
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
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
      a += 4;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 += 4;
        __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
        a1 += 4;
        __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
        a2 += 4;
        __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
        a3 += 4;


        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c0, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c0, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c0, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc0, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc0, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc0, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c1, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c1, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c1, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc1, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc1, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc1, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c2, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c2, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c2, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc2, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc2, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc2, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c3, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c3, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c3, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc3, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc3, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc3, vacc3x89ABCDEF);


        w += 64;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 = (const float*) ((uintptr_t) a0 + k);
        __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
        a1 = (const float*) ((uintptr_t) a1 + k);
        __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
        a2 = (const float*) ((uintptr_t) a2 + k);
        __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
        a3 = (const float*) ((uintptr_t) a3 + k);

        const __m256 vzero = _mm256_setzero_ps();

        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc3x89ABCDEF);


        w += 64;
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
    vacc1x01234567 = _mm256_max_ps(vacc1x01234567, vmin);
    vacc2x01234567 = _mm256_max_ps(vacc2x01234567, vmin);
    vacc3x01234567 = _mm256_max_ps(vacc3x01234567, vmin);
    vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);
    vacc1x89ABCDEF = _mm256_max_ps(vacc1x89ABCDEF, vmin);
    vacc2x89ABCDEF = _mm256_max_ps(vacc2x89ABCDEF, vmin);
    vacc3x89ABCDEF = _mm256_max_ps(vacc3x89ABCDEF, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
    vacc1x01234567 = _mm256_min_ps(vacc1x01234567, vmax);
    vacc2x01234567 = _mm256_min_ps(vacc2x01234567, vmax);
    vacc3x01234567 = _mm256_min_ps(vacc3x01234567, vmax);
    vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);
    vacc1x89ABCDEF = _mm256_min_ps(vacc1x89ABCDEF, vmax);
    vacc2x89ABCDEF = _mm256_min_ps(vacc2x89ABCDEF, vmax);
    vacc3x89ABCDEF = _mm256_min_ps(vacc3x89ABCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
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
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast(
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

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567, vacc2x01234567);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567, vacc3x01234567);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEF, vacc3x89ABCDEF);
        vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567, vacc4x01234567);
        vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEF, vacc4x89ABCDEF);
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

void xnn_f32_vhswish_ukernel__fma3_x16(
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

    __m256 vacc01234567 = _mm256_fmadd_ps(vx01234567, vsixth, vhalf);
    __m256 vacc89ABCDEF = _mm256_fmadd_ps(vx89ABCDEF, vsixth, vhalf);

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
    __m256 vacc = _mm256_fmadd_ps(vx, vsixth, vhalf);
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
    __m256 vacc = _mm256_fmadd_ps(vx, vsixth, vhalf);
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
