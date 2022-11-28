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
#include <xnnpack/vadd.h>
#include <xnnpack/vcvt.h>


void xnn_f16_f32_vcvt_ukernel__avx512skx_x16(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m512 vacc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= 1 /* log2(sizeof(uint16_t)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vacc = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_f16_vcvt_ukernel__avx512skx_x16(
    size_t batch,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vf = _mm512_loadu_ps(input);
    input += 16;

    _mm256_storeu_si256((__m256i*) o, _mm512_cvtps_ph(vf, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vf = _mm512_maskz_loadu_ps(vmask, input);
    const __m256i vh = _mm512_cvtps_ph(vf, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT);
    _mm256_mask_storeu_epi16(o, vmask, vh);
  }
}

void xnn_f32_qs8_vcvt_ukernel__avx512skx_x128(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_load_ps(params->avx2.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->avx512.output_zero_point);
  const __m512i vshuffle512_mask = _mm512_load_si512(params->avx512.shuffle512_mask);
  const __m512i voutput_min = _mm512_load_si512(params->avx512.output_min);
  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    __m512 vx4567 = _mm512_loadu_ps(input + 16);
    __m512 vx89AB = _mm512_loadu_ps(input + 32);
    __m512 vxCDEF = _mm512_loadu_ps(input + 48);
    __m512 vxGHIJ = _mm512_loadu_ps(input + 64);
    __m512 vxKLMN = _mm512_loadu_ps(input + 80);
    __m512 vxOPQR = _mm512_loadu_ps(input + 96);
    __m512 vxSTUV = _mm512_loadu_ps(input + 112);
    input += 128;

    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx4567 = _mm512_mul_ps(vx4567, vscale);
    vx89AB = _mm512_mul_ps(vx89AB, vscale);
    vxCDEF = _mm512_mul_ps(vxCDEF, vscale);
    vxGHIJ = _mm512_mul_ps(vxGHIJ, vscale);
    vxKLMN = _mm512_mul_ps(vxKLMN, vscale);
    vxOPQR = _mm512_mul_ps(vxOPQR, vscale);
    vxSTUV = _mm512_mul_ps(vxSTUV, vscale);

    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm512_min_ps(vx4567, voutput_max_less_zero_point);
    vx89AB = _mm512_min_ps(vx89AB, voutput_max_less_zero_point);
    vxCDEF = _mm512_min_ps(vxCDEF, voutput_max_less_zero_point);
    vxGHIJ = _mm512_min_ps(vxGHIJ, voutput_max_less_zero_point);
    vxKLMN = _mm512_min_ps(vxKLMN, voutput_max_less_zero_point);
    vxOPQR = _mm512_min_ps(vxOPQR, voutput_max_less_zero_point);
    vxSTUV = _mm512_min_ps(vxSTUV, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);
    const __m512i vacc4567 = _mm512_cvtps_epi32(vx4567);
    const __m512i vacc89AB = _mm512_cvtps_epi32(vx89AB);
    const __m512i vaccCDEF = _mm512_cvtps_epi32(vxCDEF);
    const __m512i vaccGHIJ = _mm512_cvtps_epi32(vxGHIJ);
    const __m512i vaccKLMN = _mm512_cvtps_epi32(vxKLMN);
    const __m512i vaccOPQR = _mm512_cvtps_epi32(vxOPQR);
    const __m512i vaccSTUV = _mm512_cvtps_epi32(vxSTUV);

    __m512i vacc04152637 = _mm512_packs_epi32(vacc0123, vacc4567);
    __m512i vacc8C9DAEBF = _mm512_packs_epi32(vacc89AB, vaccCDEF);
    __m512i vaccGKHLIMJN = _mm512_packs_epi32(vaccGHIJ, vaccKLMN);
    __m512i vaccOSPTQURV = _mm512_packs_epi32(vaccOPQR, vaccSTUV);

    vacc04152637 = _mm512_adds_epi16(vacc04152637, voutput_zero_point);
    vacc8C9DAEBF = _mm512_adds_epi16(vacc8C9DAEBF, voutput_zero_point);
    vaccGKHLIMJN = _mm512_adds_epi16(vaccGKHLIMJN, voutput_zero_point);
    vaccOSPTQURV = _mm512_adds_epi16(vaccOSPTQURV, voutput_zero_point);

    __m512i vy048C159D26AE37BF = _mm512_packs_epi16(vacc04152637, vacc8C9DAEBF);
    __m512i vyGKOSHLPTIMQUJNRV = _mm512_packs_epi16(vaccGKHLIMJN, vaccOSPTQURV);

    vy048C159D26AE37BF = _mm512_max_epi8(vy048C159D26AE37BF, voutput_min);
    vyGKOSHLPTIMQUJNRV = _mm512_max_epi8(vyGKOSHLPTIMQUJNRV, voutput_min);

    const __m512i vy0123456789ABCDEF = _mm512_permutexvar_epi32(vshuffle512_mask, vy048C159D26AE37BF);
    const __m512i vyGHIJKLMNOPQRSTUV = _mm512_permutexvar_epi32(vshuffle512_mask, vyGKOSHLPTIMQUJNRV);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    _mm512_storeu_si512(output + 64, vyGHIJKLMNOPQRSTUV);
    output += 128;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    input += 16;

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epi8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_storeu_si128((__m128i*) output, vy0123);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx0123 = _mm512_maskz_loadu_ps(vmask, input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epi8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_mask_storeu_epi8(output, vmask, vy0123);
  }
}

void xnn_f32_qu8_vcvt_ukernel__avx512skx_x128(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_load_ps(params->avx2.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->avx512.output_zero_point);
  const __m512i vshuffle512_mask = _mm512_load_si512(params->avx512.shuffle512_mask);
  const __m512i voutput_min = _mm512_load_si512(params->avx512.output_min);
  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    __m512 vx4567 = _mm512_loadu_ps(input + 16);
    __m512 vx89AB = _mm512_loadu_ps(input + 32);
    __m512 vxCDEF = _mm512_loadu_ps(input + 48);
    __m512 vxGHIJ = _mm512_loadu_ps(input + 64);
    __m512 vxKLMN = _mm512_loadu_ps(input + 80);
    __m512 vxOPQR = _mm512_loadu_ps(input + 96);
    __m512 vxSTUV = _mm512_loadu_ps(input + 112);
    input += 128;

    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx4567 = _mm512_mul_ps(vx4567, vscale);
    vx89AB = _mm512_mul_ps(vx89AB, vscale);
    vxCDEF = _mm512_mul_ps(vxCDEF, vscale);
    vxGHIJ = _mm512_mul_ps(vxGHIJ, vscale);
    vxKLMN = _mm512_mul_ps(vxKLMN, vscale);
    vxOPQR = _mm512_mul_ps(vxOPQR, vscale);
    vxSTUV = _mm512_mul_ps(vxSTUV, vscale);

    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm512_min_ps(vx4567, voutput_max_less_zero_point);
    vx89AB = _mm512_min_ps(vx89AB, voutput_max_less_zero_point);
    vxCDEF = _mm512_min_ps(vxCDEF, voutput_max_less_zero_point);
    vxGHIJ = _mm512_min_ps(vxGHIJ, voutput_max_less_zero_point);
    vxKLMN = _mm512_min_ps(vxKLMN, voutput_max_less_zero_point);
    vxOPQR = _mm512_min_ps(vxOPQR, voutput_max_less_zero_point);
    vxSTUV = _mm512_min_ps(vxSTUV, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);
    const __m512i vacc4567 = _mm512_cvtps_epi32(vx4567);
    const __m512i vacc89AB = _mm512_cvtps_epi32(vx89AB);
    const __m512i vaccCDEF = _mm512_cvtps_epi32(vxCDEF);
    const __m512i vaccGHIJ = _mm512_cvtps_epi32(vxGHIJ);
    const __m512i vaccKLMN = _mm512_cvtps_epi32(vxKLMN);
    const __m512i vaccOPQR = _mm512_cvtps_epi32(vxOPQR);
    const __m512i vaccSTUV = _mm512_cvtps_epi32(vxSTUV);

    __m512i vacc04152637 = _mm512_packs_epi32(vacc0123, vacc4567);
    __m512i vacc8C9DAEBF = _mm512_packs_epi32(vacc89AB, vaccCDEF);
    __m512i vaccGKHLIMJN = _mm512_packs_epi32(vaccGHIJ, vaccKLMN);
    __m512i vaccOSPTQURV = _mm512_packs_epi32(vaccOPQR, vaccSTUV);

    vacc04152637 = _mm512_adds_epi16(vacc04152637, voutput_zero_point);
    vacc8C9DAEBF = _mm512_adds_epi16(vacc8C9DAEBF, voutput_zero_point);
    vaccGKHLIMJN = _mm512_adds_epi16(vaccGKHLIMJN, voutput_zero_point);
    vaccOSPTQURV = _mm512_adds_epi16(vaccOSPTQURV, voutput_zero_point);

    __m512i vy048C159D26AE37BF = _mm512_packus_epi16(vacc04152637, vacc8C9DAEBF);
    __m512i vyGKOSHLPTIMQUJNRV = _mm512_packus_epi16(vaccGKHLIMJN, vaccOSPTQURV);

    vy048C159D26AE37BF = _mm512_max_epu8(vy048C159D26AE37BF, voutput_min);
    vyGKOSHLPTIMQUJNRV = _mm512_max_epu8(vyGKOSHLPTIMQUJNRV, voutput_min);

    const __m512i vy0123456789ABCDEF = _mm512_permutexvar_epi32(vshuffle512_mask, vy048C159D26AE37BF);
    const __m512i vyGHIJKLMNOPQRSTUV = _mm512_permutexvar_epi32(vshuffle512_mask, vyGKOSHLPTIMQUJNRV);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    _mm512_storeu_si512(output + 64, vyGHIJKLMNOPQRSTUV);
    output += 128;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    input += 16;

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packus_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epu8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_storeu_si128((__m128i*) output, vy0123);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx0123 = _mm512_maskz_loadu_ps(vmask, input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packus_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epu8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_mask_storeu_epi8(output, vmask, vy0123);
  }
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

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
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      const __m512i vi9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i9));
      const __m512i vk9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m512i vi9xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m512i vk9xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      i9 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi9xGHIJKLMNOPQRSTUV, vk9xGHIJKLMNOPQRSTUV));

      const __m512i vi10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i10));
      const __m512i vk10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m512i vi10xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m512i vk10xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      i10 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi10xGHIJKLMNOPQRSTUV, vk10xGHIJKLMNOPQRSTUV));

      const __m512i vi11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i11));
      const __m512i vk11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m512i vi11xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m512i vk11xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      i11 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi11xGHIJKLMNOPQRSTUV, vk11xGHIJKLMNOPQRSTUV));

      const __m512i vi12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i12));
      const __m512i vk12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m512i vi12xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m512i vk12xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(int8_t))));
      i12 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi12xGHIJKLMNOPQRSTUV, vk12xGHIJKLMNOPQRSTUV));

      const __m512i vi13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i13));
      const __m512i vk13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t))));
      const __m512i vi13xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m512i vk13xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(int8_t))));
      i13 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi13xGHIJKLMNOPQRSTUV, vk13xGHIJKLMNOPQRSTUV));

      const __m512i vi14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i14));
      const __m512i vk14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t))));
      const __m512i vi14xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m512i vk14xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(int8_t))));
      i14 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi14xGHIJKLMNOPQRSTUV, vk14xGHIJKLMNOPQRSTUV));

      const __m512i vi15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i15));
      const __m512i vk15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t))));
      const __m512i vi15xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m512i vk15xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(int8_t))));
      i15 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi15xGHIJKLMNOPQRSTUV, vk15xGHIJKLMNOPQRSTUV));

      const __m512i vi16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i16));
      const __m512i vk16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t))));
      const __m512i vi16xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m512i vk16xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(int8_t))));
      i16 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi16xGHIJKLMNOPQRSTUV, vk16xGHIJKLMNOPQRSTUV));

      const __m512i vi17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i17));
      const __m512i vk17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t))));
      const __m512i vi17xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m512i vk17xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(int8_t))));
      i17 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi17xGHIJKLMNOPQRSTUV, vk17xGHIJKLMNOPQRSTUV));

      const __m512i vi18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i18));
      const __m512i vk18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t))));
      const __m512i vi18xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m512i vk18xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(int8_t))));
      i18 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi18xGHIJKLMNOPQRSTUV, vk18xGHIJKLMNOPQRSTUV));

      const __m512i vi19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i19));
      const __m512i vk19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t))));
      const __m512i vi19xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m512i vk19xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(int8_t))));
      i19 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi19xGHIJKLMNOPQRSTUV, vk19xGHIJKLMNOPQRSTUV));

      const __m512i vi20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i20));
      const __m512i vk20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t))));
      const __m512i vi20xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m512i vk20xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(int8_t))));
      i20 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi20xGHIJKLMNOPQRSTUV, vk20xGHIJKLMNOPQRSTUV));

      const __m512i vi21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i21));
      const __m512i vk21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t))));
      const __m512i vi21xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m512i vk21xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(int8_t))));
      i21 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi21xGHIJKLMNOPQRSTUV, vk21xGHIJKLMNOPQRSTUV));

      const __m512i vi22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i22));
      const __m512i vk22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t))));
      const __m512i vi22xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m512i vk22xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(int8_t))));
      i22 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi22xGHIJKLMNOPQRSTUV, vk22xGHIJKLMNOPQRSTUV));

      const __m512i vi23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i23));
      const __m512i vk23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t))));
      const __m512i vi23xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m512i vk23xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(int8_t))));
      i23 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi23xGHIJKLMNOPQRSTUV, vk23xGHIJKLMNOPQRSTUV));

      const __m512i vi24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i24));
      const __m512i vk24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t))));
      const __m512i vi24xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m512i vk24xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(int8_t))));
      i24 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi24xGHIJKLMNOPQRSTUV, vk24xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps(w);
      const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const void*) ((uintptr_t) w + 16 * sizeof(float)));
      w = (const void*) ((uintptr_t) w + 32 * sizeof(float));
      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        const __m512i vi9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i9));
        const __m512i vk9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 288)));
        i9 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));

        const __m512i vi10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i10));
        const __m512i vk10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 320)));
        i10 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));

        const __m512i vi11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i11));
        const __m512i vk11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 352)));
        i11 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));

        const __m512i vi12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i12));
        const __m512i vk12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 384)));
        i12 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));

        const __m512i vi13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i13));
        const __m512i vk13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 416)));
        i13 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));

        const __m512i vi14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i14));
        const __m512i vk14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 448)));
        i14 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));

        const __m512i vi15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i15));
        const __m512i vk15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 480)));
        i15 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));

        const __m512i vi16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i16));
        const __m512i vk16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 512)));
        i16 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));

        const __m512i vi17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i17));
        const __m512i vk17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 544)));
        i17 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));

        const __m512i vi18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i18));
        const __m512i vk18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 576)));
        i18 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));

        const __m512i vi19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i19));
        const __m512i vk19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 608)));
        i19 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));

        const __m512i vi20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i20));
        const __m512i vk20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 640)));
        i20 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));

        const __m512i vi21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i21));
        const __m512i vk21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 672)));
        i21 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));

        const __m512i vi22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i22));
        const __m512i vk22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 704)));
        i22 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));

        const __m512i vi23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i23));
        const __m512i vk23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 736)));
        i23 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));

        const __m512i vi24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i24));
        const __m512i vk24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 768)));
        i24 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t)));
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_3p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

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
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps(w);
      const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const void*) ((uintptr_t) w + 16 * sizeof(float)));
      w = (const void*) ((uintptr_t) w + 32 * sizeof(float));
      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t)));
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

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
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps(w);
      const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const void*) ((uintptr_t) w + 16 * sizeof(float)));
      w = (const void*) ((uintptr_t) w + 32 * sizeof(float));
      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t)));
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

      w = (const void*) ((const int8_t*) w + 128);
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const void*) ((const float*) w + 16);
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
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
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    __m512i vacc2x0123 = vacc0x0123;
    __m512i vacc2x4567 = vacc0x4567;
    __m512i vacc2x89AB = vacc0x89AB;
    __m512i vacc2xCDEF = vacc0xCDEF;
    __m512i vacc3x0123 = vacc0x0123;
    __m512i vacc3x4567 = vacc0x4567;
    __m512i vacc3x89AB = vacc0x89AB;
    __m512i vacc3xCDEF = vacc0xCDEF;
    w = (const void*) ((const int32_t*) w + 16);

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

      w = (const void*) ((const int8_t*) w + 128);
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const void*) ((const float*) w + 16);
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));

      a0 = (const int8_t*) ((uintptr_t) a0 - k);
      a1 = (const int8_t*) ((uintptr_t) a1 - k);
      a2 = (const int8_t*) ((uintptr_t) a2 - k);
      a3 = (const int8_t*) ((uintptr_t) a3 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx(
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
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
  int8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

        w = (const void*) ((const int8_t*) w + 128);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const void*) ((const float*) w + 16);
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx(
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
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    __m512i vacc2x0123 = vacc0x0123;
    __m512i vacc2x4567 = vacc0x4567;
    __m512i vacc2x89AB = vacc0x89AB;
    __m512i vacc2xCDEF = vacc0xCDEF;
    __m512i vacc3x0123 = vacc0x0123;
    __m512i vacc3x4567 = vacc0x4567;
    __m512i vacc3x89AB = vacc0x89AB;
    __m512i vacc3xCDEF = vacc0xCDEF;
    w = (const void*) ((const int32_t*) w + 16);

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
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = 0;
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;
        const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
        a1 += 8;
        const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
        a2 += 8;
        const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
        a3 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
        vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
        vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
        vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
        vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
        vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
        vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
        vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
        vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
        vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

        w = (const void*) ((const int8_t*) w + 128);
        k += 8 * sizeof(int8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const void*) ((const float*) w + 16);
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (nc + 48)) - (UINT64_C(1) << 48)));

      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

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
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      const __m512i vi9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i9));
      const __m512i vk9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m512i vi9xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m512i vk9xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      i9 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi9xGHIJKLMNOPQRSTUV, vk9xGHIJKLMNOPQRSTUV));

      const __m512i vi10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i10));
      const __m512i vk10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m512i vi10xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m512i vk10xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      i10 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi10xGHIJKLMNOPQRSTUV, vk10xGHIJKLMNOPQRSTUV));

      const __m512i vi11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i11));
      const __m512i vk11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m512i vi11xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m512i vk11xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      i11 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi11xGHIJKLMNOPQRSTUV, vk11xGHIJKLMNOPQRSTUV));

      const __m512i vi12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i12));
      const __m512i vk12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m512i vi12xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m512i vk12xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(int8_t))));
      i12 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi12xGHIJKLMNOPQRSTUV, vk12xGHIJKLMNOPQRSTUV));

      const __m512i vi13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i13));
      const __m512i vk13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t))));
      const __m512i vi13xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m512i vk13xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(int8_t))));
      i13 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi13xGHIJKLMNOPQRSTUV, vk13xGHIJKLMNOPQRSTUV));

      const __m512i vi14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i14));
      const __m512i vk14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t))));
      const __m512i vi14xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m512i vk14xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(int8_t))));
      i14 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi14xGHIJKLMNOPQRSTUV, vk14xGHIJKLMNOPQRSTUV));

      const __m512i vi15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i15));
      const __m512i vk15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t))));
      const __m512i vi15xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m512i vk15xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(int8_t))));
      i15 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi15xGHIJKLMNOPQRSTUV, vk15xGHIJKLMNOPQRSTUV));

      const __m512i vi16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i16));
      const __m512i vk16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t))));
      const __m512i vi16xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m512i vk16xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(int8_t))));
      i16 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi16xGHIJKLMNOPQRSTUV, vk16xGHIJKLMNOPQRSTUV));

      const __m512i vi17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i17));
      const __m512i vk17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t))));
      const __m512i vi17xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m512i vk17xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(int8_t))));
      i17 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi17xGHIJKLMNOPQRSTUV, vk17xGHIJKLMNOPQRSTUV));

      const __m512i vi18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i18));
      const __m512i vk18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t))));
      const __m512i vi18xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m512i vk18xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(int8_t))));
      i18 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi18xGHIJKLMNOPQRSTUV, vk18xGHIJKLMNOPQRSTUV));

      const __m512i vi19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i19));
      const __m512i vk19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t))));
      const __m512i vi19xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m512i vk19xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(int8_t))));
      i19 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi19xGHIJKLMNOPQRSTUV, vk19xGHIJKLMNOPQRSTUV));

      const __m512i vi20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i20));
      const __m512i vk20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t))));
      const __m512i vi20xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m512i vk20xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(int8_t))));
      i20 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi20xGHIJKLMNOPQRSTUV, vk20xGHIJKLMNOPQRSTUV));

      const __m512i vi21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i21));
      const __m512i vk21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t))));
      const __m512i vi21xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m512i vk21xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(int8_t))));
      i21 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi21xGHIJKLMNOPQRSTUV, vk21xGHIJKLMNOPQRSTUV));

      const __m512i vi22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i22));
      const __m512i vk22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t))));
      const __m512i vi22xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m512i vk22xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(int8_t))));
      i22 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi22xGHIJKLMNOPQRSTUV, vk22xGHIJKLMNOPQRSTUV));

      const __m512i vi23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i23));
      const __m512i vk23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t))));
      const __m512i vi23xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m512i vk23xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(int8_t))));
      i23 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi23xGHIJKLMNOPQRSTUV, vk23xGHIJKLMNOPQRSTUV));

      const __m512i vi24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i24));
      const __m512i vk24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t))));
      const __m512i vi24xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m512i vk24xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(int8_t))));
      i24 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi24xGHIJKLMNOPQRSTUV, vk24xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        const __m512i vi9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i9));
        const __m512i vk9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 288)));
        i9 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));

        const __m512i vi10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i10));
        const __m512i vk10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 320)));
        i10 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));

        const __m512i vi11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i11));
        const __m512i vk11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 352)));
        i11 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));

        const __m512i vi12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i12));
        const __m512i vk12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 384)));
        i12 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));

        const __m512i vi13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i13));
        const __m512i vk13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 416)));
        i13 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));

        const __m512i vi14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i14));
        const __m512i vk14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 448)));
        i14 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));

        const __m512i vi15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i15));
        const __m512i vk15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 480)));
        i15 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));

        const __m512i vi16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i16));
        const __m512i vk16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 512)));
        i16 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));

        const __m512i vi17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i17));
        const __m512i vk17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 544)));
        i17 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));

        const __m512i vi18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i18));
        const __m512i vk18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 576)));
        i18 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));

        const __m512i vi19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i19));
        const __m512i vk19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 608)));
        i19 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));

        const __m512i vi20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i20));
        const __m512i vk20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 640)));
        i20 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));

        const __m512i vi21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i21));
        const __m512i vk21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 672)));
        i21 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));

        const __m512i vi22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i22));
        const __m512i vk22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 704)));
        i22 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));

        const __m512i vi23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i23));
        const __m512i vk23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 736)));
        i23 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));

        const __m512i vi24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i24));
        const __m512i vk24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 768)));
        i24 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

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
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_f32_vcvt_ukernel__avx512skx_x32(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vminus_zero_point = _mm512_load_si512(params->avx512.minus_zero_point);
  const __m512 vscale = _mm512_load_ps(params->avx512.scale);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m512i vx0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input));
    __m512i vxGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (input + 16)));
    input += 32;

    vx0123456789ABCDEF = _mm512_add_epi32(vx0123456789ABCDEF, vminus_zero_point);
    vxGHIJKLMNOPQRSTUV = _mm512_add_epi32(vxGHIJKLMNOPQRSTUV, vminus_zero_point);

    __m512 vy0123456789ABCDEF = _mm512_cvtepi32_ps(vx0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vxGHIJKLMNOPQRSTUV);

    vy0123456789ABCDEF = _mm512_mul_ps(vy0123456789ABCDEF, vscale);
    vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vyGHIJKLMNOPQRSTUV, vscale);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    _mm512_storeu_ps(output + 16, vyGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m512i vx = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input));
    vx = _mm512_add_epi32(vx, vminus_zero_point);
    input += 16;

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    // Prepare mask for valid elements (depends on batch).
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512i vx = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input));
    vx = _mm512_add_epi32(vx, vminus_zero_point);

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(
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

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

      w = (const void*) ((const int8_t*) w + 128);
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx(
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
  assert(mr <= 4);
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
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    __m512i vacc2x0123 = vacc0x0123;
    __m512i vacc2x4567 = vacc0x4567;
    __m512i vacc2x89AB = vacc0x89AB;
    __m512i vacc2xCDEF = vacc0xCDEF;
    __m512i vacc3x0123 = vacc0x0123;
    __m512i vacc3x4567 = vacc0x4567;
    __m512i vacc3x89AB = vacc0x89AB;
    __m512i vacc3xCDEF = vacc0xCDEF;
    w = (const void*) ((const int32_t*) w + 16);

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

      w = (const void*) ((const int8_t*) w + 128);
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));

      a0 = (const int8_t*) ((uintptr_t) a0 - k);
      a1 = (const int8_t*) ((uintptr_t) a1 - k);
      a2 = (const int8_t*) ((uintptr_t) a2 - k);
      a3 = (const int8_t*) ((uintptr_t) a3 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx(
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
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

        w = (const void*) ((const int8_t*) w + 128);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    __m512i vacc2x0123 = vacc0x0123;
    __m512i vacc2x4567 = vacc0x4567;
    __m512i vacc2x89AB = vacc0x89AB;
    __m512i vacc2xCDEF = vacc0xCDEF;
    __m512i vacc3x0123 = vacc0x0123;
    __m512i vacc3x4567 = vacc0x4567;
    __m512i vacc3x89AB = vacc0x89AB;
    __m512i vacc3xCDEF = vacc0xCDEF;
    w = (const void*) ((const int32_t*) w + 16);

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
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = 0;
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;
        const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
        a1 += 8;
        const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
        a2 += 8;
        const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
        a3 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
        vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
        vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
        vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
        vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
        vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
        vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
        vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
        vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
        vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

        w = (const void*) ((const int8_t*) w + 128);
        k += 8 * sizeof(int8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (nc + 48)) - (UINT64_C(1) << 48)));

      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_x16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i vbias = _mm512_load_si512(params->avx512.bias);
  const __m512i va_multiplier = _mm512_load_si512(params->avx512.a_multiplier);
  const __m512i vb_multiplier = _mm512_load_si512(params->avx512.b_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx512.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx512.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx512.output_max);

  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    const __m512i vb0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_b));
    input_a += 16;
    input_b += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));
      const __m512i vb0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input_b));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}

void xnn_qs8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_x16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i va_multiplier = _mm512_load_si512(params->avx512.a_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx512.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx512.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx512.output_max);

  const __m512i vbias = _mm512_add_epi32(
    _mm512_broadcastd_epi32(_mm_cvtsi32_si128(params->avx512.b_multiplier[0] * (int32_t) *input_b)),
    _mm512_load_si512(params->avx512.bias));
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    input_a += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  const __m512i vk_zero_point = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i*) params->fp32_avx512.kernel_zero_point));
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
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(uint8_t)))), vk_zero_point);
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(uint8_t)))), vk_zero_point);
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(uint8_t)))), vk_zero_point);
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(uint8_t)))), vk_zero_point);
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(uint8_t)))), vk_zero_point);
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      const __m512i vi9x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i9));
      const __m512i vk9x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi9xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m512i vk9xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(uint8_t)))), vk_zero_point);
      i9 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi9xGHIJKLMNOPQRSTUV, vk9xGHIJKLMNOPQRSTUV));

      const __m512i vi10x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i10));
      const __m512i vk10x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi10xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m512i vk10xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(uint8_t)))), vk_zero_point);
      i10 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi10xGHIJKLMNOPQRSTUV, vk10xGHIJKLMNOPQRSTUV));

      const __m512i vi11x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i11));
      const __m512i vk11x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi11xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m512i vk11xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(uint8_t)))), vk_zero_point);
      i11 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi11xGHIJKLMNOPQRSTUV, vk11xGHIJKLMNOPQRSTUV));

      const __m512i vi12x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i12));
      const __m512i vk12x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi12xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m512i vk12xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(uint8_t)))), vk_zero_point);
      i12 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi12xGHIJKLMNOPQRSTUV, vk12xGHIJKLMNOPQRSTUV));

      const __m512i vi13x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i13));
      const __m512i vk13x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi13xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m512i vk13xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(uint8_t)))), vk_zero_point);
      i13 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi13xGHIJKLMNOPQRSTUV, vk13xGHIJKLMNOPQRSTUV));

      const __m512i vi14x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i14));
      const __m512i vk14x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi14xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m512i vk14xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(uint8_t)))), vk_zero_point);
      i14 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi14xGHIJKLMNOPQRSTUV, vk14xGHIJKLMNOPQRSTUV));

      const __m512i vi15x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i15));
      const __m512i vk15x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi15xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m512i vk15xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(uint8_t)))), vk_zero_point);
      i15 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi15xGHIJKLMNOPQRSTUV, vk15xGHIJKLMNOPQRSTUV));

      const __m512i vi16x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i16));
      const __m512i vk16x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi16xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m512i vk16xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(uint8_t)))), vk_zero_point);
      i16 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi16xGHIJKLMNOPQRSTUV, vk16xGHIJKLMNOPQRSTUV));

      const __m512i vi17x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i17));
      const __m512i vk17x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi17xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m512i vk17xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(uint8_t)))), vk_zero_point);
      i17 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi17xGHIJKLMNOPQRSTUV, vk17xGHIJKLMNOPQRSTUV));

      const __m512i vi18x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i18));
      const __m512i vk18x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi18xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m512i vk18xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(uint8_t)))), vk_zero_point);
      i18 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi18xGHIJKLMNOPQRSTUV, vk18xGHIJKLMNOPQRSTUV));

      const __m512i vi19x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i19));
      const __m512i vk19x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi19xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m512i vk19xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(uint8_t)))), vk_zero_point);
      i19 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi19xGHIJKLMNOPQRSTUV, vk19xGHIJKLMNOPQRSTUV));

      const __m512i vi20x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i20));
      const __m512i vk20x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi20xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m512i vk20xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(uint8_t)))), vk_zero_point);
      i20 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi20xGHIJKLMNOPQRSTUV, vk20xGHIJKLMNOPQRSTUV));

      const __m512i vi21x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i21));
      const __m512i vk21x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi21xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m512i vk21xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(uint8_t)))), vk_zero_point);
      i21 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi21xGHIJKLMNOPQRSTUV, vk21xGHIJKLMNOPQRSTUV));

      const __m512i vi22x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i22));
      const __m512i vk22x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi22xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m512i vk22xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(uint8_t)))), vk_zero_point);
      i22 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi22xGHIJKLMNOPQRSTUV, vk22xGHIJKLMNOPQRSTUV));

      const __m512i vi23x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i23));
      const __m512i vk23x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi23xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m512i vk23xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(uint8_t)))), vk_zero_point);
      i23 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi23xGHIJKLMNOPQRSTUV, vk23xGHIJKLMNOPQRSTUV));

      const __m512i vi24x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i24));
      const __m512i vk24x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi24xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m512i vk24xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(uint8_t)))), vk_zero_point);
      i24 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi24xGHIJKLMNOPQRSTUV, vk24xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(uint8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packus_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packus_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epu8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epu8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const uint8_t* k = (const uint8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) k)), vk_zero_point);
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 32))), vk_zero_point);
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 64))), vk_zero_point);
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 96))), vk_zero_point);
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 128))), vk_zero_point);
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 160))), vk_zero_point);
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 192))), vk_zero_point);
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 224))), vk_zero_point);
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 256))), vk_zero_point);
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        const __m512i vi9x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i9));
        const __m512i vk9x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 288))), vk_zero_point);
        i9 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));

        const __m512i vi10x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i10));
        const __m512i vk10x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 320))), vk_zero_point);
        i10 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));

        const __m512i vi11x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i11));
        const __m512i vk11x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 352))), vk_zero_point);
        i11 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));

        const __m512i vi12x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i12));
        const __m512i vk12x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 384))), vk_zero_point);
        i12 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));

        const __m512i vi13x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i13));
        const __m512i vk13x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 416))), vk_zero_point);
        i13 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));

        const __m512i vi14x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i14));
        const __m512i vk14x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 448))), vk_zero_point);
        i14 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));

        const __m512i vi15x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i15));
        const __m512i vk15x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 480))), vk_zero_point);
        i15 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));

        const __m512i vi16x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i16));
        const __m512i vk16x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 512))), vk_zero_point);
        i16 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));

        const __m512i vi17x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i17));
        const __m512i vk17x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 544))), vk_zero_point);
        i17 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));

        const __m512i vi18x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i18));
        const __m512i vk18x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 576))), vk_zero_point);
        i18 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));

        const __m512i vi19x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i19));
        const __m512i vk19x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 608))), vk_zero_point);
        i19 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));

        const __m512i vi20x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i20));
        const __m512i vk20x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 640))), vk_zero_point);
        i20 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));

        const __m512i vi21x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i21));
        const __m512i vk21x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 672))), vk_zero_point);
        i21 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));

        const __m512i vi22x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i22));
        const __m512i vk22x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 704))), vk_zero_point);
        i22 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));

        const __m512i vi23x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i23));
        const __m512i vk23x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 736))), vk_zero_point);
        i23 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));

        const __m512i vi24x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i24));
        const __m512i vk24x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 768))), vk_zero_point);
        i24 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (uint8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  const __m512i vk_zero_point = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i*) params->fp32_avx512.kernel_zero_point));
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
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(uint8_t)))), vk_zero_point);
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(uint8_t)))), vk_zero_point);
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(uint8_t)))), vk_zero_point);
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(uint8_t)))), vk_zero_point);
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(uint8_t)))), vk_zero_point);
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(uint8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packus_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packus_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epu8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epu8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const uint8_t* k = (const uint8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) k)), vk_zero_point);
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 32))), vk_zero_point);
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 64))), vk_zero_point);
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 96))), vk_zero_point);
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 128))), vk_zero_point);
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 160))), vk_zero_point);
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 192))), vk_zero_point);
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 224))), vk_zero_point);
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 256))), vk_zero_point);
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (uint8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_f32_vcvt_ukernel__avx512skx_x32(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vminus_zero_point = _mm512_load_si512(params->avx512.minus_zero_point);
  const __m512 vscale = _mm512_load_ps(params->avx512.scale);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    __m512i vx0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input));
    __m512i vxGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (input + 16)));
    input += 32;

    vx0123456789ABCDEF = _mm512_add_epi32(vx0123456789ABCDEF, vminus_zero_point);
    vxGHIJKLMNOPQRSTUV = _mm512_add_epi32(vxGHIJKLMNOPQRSTUV, vminus_zero_point);

    __m512 vy0123456789ABCDEF = _mm512_cvtepi32_ps(vx0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vxGHIJKLMNOPQRSTUV);

    vy0123456789ABCDEF = _mm512_mul_ps(vy0123456789ABCDEF, vscale);
    vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vyGHIJKLMNOPQRSTUV, vscale);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    _mm512_storeu_ps(output + 16, vyGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m512i vx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input));
    vx = _mm512_add_epi32(vx, vminus_zero_point);
    input += 16;

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    // Prepare mask for valid elements (depends on batch).
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512i vx = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input));
    vx = _mm512_add_epi32(vx, vminus_zero_point);

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(
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

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t k = 0;
    const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;

      const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

      w = (const void*) ((const uint8_t*) w + 128);
      k += 8 * sizeof(uint8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packus_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epu8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      a0 = (const uint8_t*) ((uintptr_t) a0 - k);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx(
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
  assert(mr <= 4);
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
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    __m512i vacc2x0123 = vacc0x0123;
    __m512i vacc2x4567 = vacc0x4567;
    __m512i vacc2x89AB = vacc0x89AB;
    __m512i vacc2xCDEF = vacc0xCDEF;
    __m512i vacc3x0123 = vacc0x0123;
    __m512i vacc3x4567 = vacc0x4567;
    __m512i vacc3x89AB = vacc0x89AB;
    __m512i vacc3xCDEF = vacc0xCDEF;
    w = (const void*) ((const int32_t*) w + 16);

    size_t k = 0;
    const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;

      const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

      w = (const void*) ((const uint8_t*) w + 128);
      k += 8 * sizeof(uint8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packus_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epu8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));

      a0 = (const uint8_t*) ((uintptr_t) a0 - k);
      a1 = (const uint8_t*) ((uintptr_t) a1 - k);
      a2 = (const uint8_t*) ((uintptr_t) a2 - k);
      a3 = (const uint8_t*) ((uintptr_t) a3 - k);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx(
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
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  uint8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;

        const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

        w = (const void*) ((const uint8_t*) w + 128);
        k += 8 * sizeof(uint8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packus_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epu8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    __m512i vacc2x0123 = vacc0x0123;
    __m512i vacc2x4567 = vacc0x4567;
    __m512i vacc2x89AB = vacc0x89AB;
    __m512i vacc2xCDEF = vacc0xCDEF;
    __m512i vacc3x0123 = vacc0x0123;
    __m512i vacc3x4567 = vacc0x4567;
    __m512i vacc3x89AB = vacc0x89AB;
    __m512i vacc3xCDEF = vacc0xCDEF;
    w = (const void*) ((const int32_t*) w + 16);

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
      const uint8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint8_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = 0;
      const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;
        const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
        a1 += 8;
        const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
        a2 += 8;
        const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
        a3 += 8;

        const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
        vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
        vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
        const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
        vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
        vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
        const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
        vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
        vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
        const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
        vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
        vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
        vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

        w = (const void*) ((const uint8_t*) w + 128);
        k += 8 * sizeof(uint8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packus_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epu8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));

      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (nc + 48)) - (UINT64_C(1) << 48)));

      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vadd_minmax_ukernel__avx512skx_mul32_ld128_x16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i vbias = _mm512_load_si512(params->avx512.bias);
  const __m512i va_multiplier = _mm512_load_si512(params->avx512.a_multiplier);
  const __m512i vb_multiplier = _mm512_load_si512(params->avx512.b_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx512.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx512.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx512.output_max);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    const __m512i vb0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input_b));
    input_a += 16;
    input_b += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));
      const __m512i vb0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input_b));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}

void xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_x16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i va_multiplier = _mm512_load_si512(params->avx512.a_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx512.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx512.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx512.output_max);

  const __m512i vbias = _mm512_add_epi32(
    _mm512_broadcastd_epi32(_mm_cvtsi32_si128(params->avx512.b_multiplier[0] * (int32_t) *input_b)),
    _mm512_load_si512(params->avx512.bias));
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    input_a += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}

void xnn_x8_lut_ukernel__avx512skx_vpshufb_x64(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vt0 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) table));
  const __m512i vt1 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 16)));
  const __m512i vt2 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 32)));
  const __m512i vt3 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 48)));
  const __m512i vt4 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 64)));
  const __m512i vt5 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 80)));
  const __m512i vt6 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 96)));
  const __m512i vt7 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 112)));
  const __m512i vt8 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 128)));
  const __m512i vt9 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 144)));
  const __m512i vtA = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 160)));
  const __m512i vtB = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 176)));
  const __m512i vtC = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 192)));
  const __m512i vtD = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 208)));
  const __m512i vtE = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 224)));
  const __m512i vtF = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 240)));

  const __m512i vtable0 = vt0;
  const __m512i vtable1 = _mm512_xor_si512(vt0, vt1);
  const __m512i vtable2 = _mm512_xor_si512(vt1, vt2);
  const __m512i vtable3 = _mm512_xor_si512(vt2, vt3);
  const __m512i vtable4 = _mm512_xor_si512(vt3, vt4);
  const __m512i vtable5 = _mm512_xor_si512(vt4, vt5);
  const __m512i vtable6 = _mm512_xor_si512(vt5, vt6);
  const __m512i vtable7 = _mm512_xor_si512(vt6, vt7);
  const __m512i vtable8 = _mm512_xor_si512(_mm512_xor_si512(vt7, vt8), vtable0);
  const __m512i vtable9 = _mm512_xor_si512(_mm512_xor_si512(vt8, vt9), vtable1);
  const __m512i vtableA = _mm512_xor_si512(_mm512_xor_si512(vt9, vtA), vtable2);
  const __m512i vtableB = _mm512_xor_si512(_mm512_xor_si512(vtA, vtB), vtable3);
  const __m512i vtableC = _mm512_xor_si512(_mm512_xor_si512(vtB, vtC), vtable4);
  const __m512i vtableD = _mm512_xor_si512(_mm512_xor_si512(vtC, vtD), vtable5);
  const __m512i vtableE = _mm512_xor_si512(_mm512_xor_si512(vtD, vtE), vtable6);
  const __m512i vtableF = _mm512_xor_si512(_mm512_xor_si512(vtE, vtF), vtable7);

  const __m512i voffset = _mm512_set1_epi8(16);
  for (; batch >= 64 * sizeof(uint8_t); batch -= 64 * sizeof(uint8_t)) {
    __m512i vx = _mm512_loadu_si512(input);
    input += 64;

    __m512i vy = _mm512_shuffle_epi8(vtable0, vx);

    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable1, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable2, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable3, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable4, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable5, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable6, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable7, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable8, vx));

    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable9, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableA, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableB, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableC, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableD, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableE, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableF, vx));

    _mm512_storeu_si512(output, vy);
    output += 64;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch < 64);
    const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << batch) - UINT64_C(1)));

    __m512i vx = _mm512_maskz_loadu_epi8(vmask, input);

    __m512i vy = _mm512_shuffle_epi8(vtable0, vx);

    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable1, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable2, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable3, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable4, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable5, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable6, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable7, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable8, vx));

    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable9, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableA, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableB, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableC, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableD, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableE, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableF, vx));

    _mm512_mask_storeu_epi8(output, vmask, vy);
  }
}
