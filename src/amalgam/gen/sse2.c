// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#ifndef M_LN2
#define M_LN2 0.69314718055994531
#endif  // M_LN2

#include <immintrin.h>

#include "xnnpack/argmaxpool.h"
#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/fill.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/gemm.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams.h"
#include "xnnpack/packw.h"
#include "xnnpack/pad.h"
#include "xnnpack/prelu.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "xnnpack/reduce.h"
#include "xnnpack/simd/f32-sse2.h"
#include "xnnpack/transpose.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/unpool.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vlrelu.h"
#include "xnnpack/vunary.h"
#include "xnnpack/zip.h"


void xnn_f16_f32_vcvt_ukernel__sse2_int16_u32(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vsign_mask = _mm_set1_epi16(UINT16_C(0x8000));
  const __m128i vexp_offset = _mm_set1_epi16(UINT16_C(0x7000));
  const __m128 vexp_scale = _mm_set1_ps(0x1.0p-112f);
  const __m128i vmagic_mask = _mm_set1_epi16(UINT16_C(0x3F00));
  const __m128 vmagic_bias = _mm_set1_ps(0.5f);
  const __m128i vdenorm_cutoff = _mm_set1_epi16(UINT16_C(0x0400));

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vexp_offset);
  XNN_FORCE_REALIZATION(vexp_scale);
  XNN_FORCE_REALIZATION(vmagic_mask);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m128i vh0 = _mm_loadu_si128((const __m128i*) i);
    const __m128i vh1 = _mm_loadu_si128((const __m128i*) (i + 8));
    const __m128i vh2 = _mm_loadu_si128((const __m128i*) (i + 16));
    const __m128i vh3 = _mm_loadu_si128((const __m128i*) (i + 24));
    i += 32;

    const __m128i vsign0 = _mm_and_si128(vh0, vsign_mask);
    const __m128i vsign1 = _mm_and_si128(vh1, vsign_mask);
    const __m128i vsign2 = _mm_and_si128(vh2, vsign_mask);
    const __m128i vsign3 = _mm_and_si128(vh3, vsign_mask);

    const __m128i vnonsign0 = _mm_xor_si128(vh0, vsign0);
    const __m128i vnonsign1 = _mm_xor_si128(vh1, vsign1);
    const __m128i vnonsign2 = _mm_xor_si128(vh2, vsign2);
    const __m128i vnonsign3 = _mm_xor_si128(vh3, vsign3);

    const __m128i vprenorm0 = _mm_slli_epi16(vnonsign0, 13);
    const __m128i vprenorm1 = _mm_add_epi16(_mm_srli_epi16(vnonsign0, 3), vexp_offset);
    const __m128i vprenorm2 = _mm_slli_epi16(vnonsign1, 13);
    const __m128i vprenorm3 = _mm_add_epi16(_mm_srli_epi16(vnonsign1, 3), vexp_offset);
    const __m128i vprenorm4 = _mm_slli_epi16(vnonsign2, 13);
    const __m128i vprenorm5 = _mm_add_epi16(_mm_srli_epi16(vnonsign2, 3), vexp_offset);
    const __m128i vprenorm6 = _mm_slli_epi16(vnonsign3, 13);
    const __m128i vprenorm7 = _mm_add_epi16(_mm_srli_epi16(vnonsign3, 3), vexp_offset);

    const __m128i vnorm0 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm0, vprenorm1)), vexp_scale));
    const __m128i vnorm1 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm0, vprenorm1)), vexp_scale));
    const __m128i vnorm2 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm2, vprenorm3)), vexp_scale));
    const __m128i vnorm3 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm2, vprenorm3)), vexp_scale));
    const __m128i vnorm4 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm4, vprenorm5)), vexp_scale));
    const __m128i vnorm5 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm4, vprenorm5)), vexp_scale));
    const __m128i vnorm6 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm6, vprenorm7)), vexp_scale));
    const __m128i vnorm7 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm6, vprenorm7)), vexp_scale));

    const __m128i vdenorm0 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign0, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm1 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign0, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm2 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign1, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm3 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign1, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm4 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign2, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm5 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign2, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm6 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign3, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm7 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign3, vmagic_mask)), vmagic_bias));

    const __m128i vmask0 = _mm_cmpgt_epi16(vnonsign0, vdenorm_cutoff);
    const __m128i vmask1 = _mm_cmpgt_epi16(vnonsign1, vdenorm_cutoff);
    const __m128i vmask2 = _mm_cmpgt_epi16(vnonsign2, vdenorm_cutoff);
    const __m128i vmask3 = _mm_cmpgt_epi16(vnonsign3, vdenorm_cutoff);

    const __m128i vxmask0 = _mm_unpacklo_epi16(vmask0, vmask0);
    const __m128i vf0 = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign0),
      _mm_or_si128(_mm_and_si128(vxmask0, vnorm0), _mm_andnot_si128(vxmask0, vdenorm0)));
    const __m128i vxmask1 = _mm_unpackhi_epi16(vmask0, vmask0);
    const __m128i vf1 = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign0),
      _mm_or_si128(_mm_and_si128(vxmask1, vnorm1), _mm_andnot_si128(vxmask1, vdenorm1)));
    const __m128i vxmask2 = _mm_unpacklo_epi16(vmask1, vmask1);
    const __m128i vf2 = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign1),
      _mm_or_si128(_mm_and_si128(vxmask2, vnorm2), _mm_andnot_si128(vxmask2, vdenorm2)));
    const __m128i vxmask3 = _mm_unpackhi_epi16(vmask1, vmask1);
    const __m128i vf3 = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign1),
      _mm_or_si128(_mm_and_si128(vxmask3, vnorm3), _mm_andnot_si128(vxmask3, vdenorm3)));
    const __m128i vxmask4 = _mm_unpacklo_epi16(vmask2, vmask2);
    const __m128i vf4 = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign2),
      _mm_or_si128(_mm_and_si128(vxmask4, vnorm4), _mm_andnot_si128(vxmask4, vdenorm4)));
    const __m128i vxmask5 = _mm_unpackhi_epi16(vmask2, vmask2);
    const __m128i vf5 = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign2),
      _mm_or_si128(_mm_and_si128(vxmask5, vnorm5), _mm_andnot_si128(vxmask5, vdenorm5)));
    const __m128i vxmask6 = _mm_unpacklo_epi16(vmask3, vmask3);
    const __m128i vf6 = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign3),
      _mm_or_si128(_mm_and_si128(vxmask6, vnorm6), _mm_andnot_si128(vxmask6, vdenorm6)));
    const __m128i vxmask7 = _mm_unpackhi_epi16(vmask3, vmask3);
    const __m128i vf7 = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign3),
      _mm_or_si128(_mm_and_si128(vxmask7, vnorm7), _mm_andnot_si128(vxmask7, vdenorm7)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf0));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf1));
    _mm_storeu_ps(output + 8, _mm_castsi128_ps(vf2));
    _mm_storeu_ps(output + 12, _mm_castsi128_ps(vf3));
    _mm_storeu_ps(output + 16, _mm_castsi128_ps(vf4));
    _mm_storeu_ps(output + 20, _mm_castsi128_ps(vf5));
    _mm_storeu_ps(output + 24, _mm_castsi128_ps(vf6));
    _mm_storeu_ps(output + 28, _mm_castsi128_ps(vf7));
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
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

    const __m128i vxmask_lo = _mm_unpacklo_epi16(vmask, vmask);
    const __m128i vf_lo = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_or_si128(_mm_and_si128(vxmask_lo, vnorm_lo), _mm_andnot_si128(vxmask_lo, vdenorm_lo)));

    const __m128i vxmask_hi = _mm_unpackhi_epi16(vmask, vmask);
    const __m128i vf_hi = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
      _mm_or_si128(_mm_and_si128(vxmask_hi, vnorm_hi), _mm_andnot_si128(vxmask_hi, vdenorm_hi)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf_lo));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf_hi));
    output += 8;
  }
  if XNN_UNPREDICTABLE(batch != 0) {
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

    const __m128i vxmask_lo = _mm_unpacklo_epi16(vmask, vmask);
    __m128i vf = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_or_si128(_mm_and_si128(vxmask_lo, vnorm_lo), _mm_andnot_si128(vxmask_lo, vdenorm_lo)));

    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storeu_ps(output, _mm_castsi128_ps(vf));
      output += 4;

      const __m128i vxmask_hi = _mm_unpackhi_epi16(vmask, vmask);
      vf = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
        _mm_or_si128(_mm_and_si128(vxmask_hi, vnorm_hi), _mm_andnot_si128(vxmask_hi, vdenorm_hi)));
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storel_pi((__m64*) output, _mm_castsi128_ps(vf));
      output += 2;

      vf = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(vf), _mm_castsi128_ps(vf)));
    }
    if (batch & (1 * sizeof(uint16_t))) {
      _mm_store_ss(output, _mm_castsi128_ps(vf));
    }
  }
}

void xnn_f16_vabs_ukernel__sse2_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const __m128i vnonsign_mask = _mm_set1_epi16(0x7FFF);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) i);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) (i + 8));
    i += 16;

    vacc0 = _mm_and_si128(vacc0, vnonsign_mask);
    vacc1 = _mm_and_si128(vacc1, vnonsign_mask);

    _mm_storeu_si128((__m128i*) o, vacc0);
    _mm_storeu_si128((__m128i*) (o + 8), vacc1);
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) i);
    i += 8;
    vacc = _mm_and_si128(vacc, vnonsign_mask);
    _mm_storeu_si128((__m128i*) o, vacc);
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) i);
    vacc = _mm_and_si128(vacc, vnonsign_mask);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vacc);
      o += 4;
      vacc = _mm_unpackhi_epi64(vacc, vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vacc));
      o += 2;
      vacc = _mm_srli_epi64(vacc, 32);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vacc, 0);
    }
  }
}

void xnn_f16_vneg_ukernel__sse2_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const __m128i vsign_mask = _mm_set1_epi16(0x8000);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) i);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) (i + 8));
    i += 16;

    vacc0 = _mm_xor_si128(vacc0, vsign_mask);
    vacc1 = _mm_xor_si128(vacc1, vsign_mask);

    _mm_storeu_si128((__m128i*) o, vacc0);
    _mm_storeu_si128((__m128i*) (o + 8), vacc1);
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) i);
    i += 8;
    vacc = _mm_xor_si128(vacc, vsign_mask);
    _mm_storeu_si128((__m128i*) o, vacc);
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) i);
    vacc = _mm_xor_si128(vacc, vsign_mask);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vacc);
      o += 4;
      vacc = _mm_unpackhi_epi64(vacc, vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vacc));
      o += 2;
      vacc = _mm_srli_epi64(vacc, 32);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vacc, 0);
    }
  }
}

void xnn_f32_argmaxpool_ukernel_4x__sse2_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 4);
  assert(channels != 0);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    i0 = (const float*) ((uintptr_t) i0 + input_offset);
    i1 = (const float*) ((uintptr_t) i1 + input_offset);
    i2 = (const float*) ((uintptr_t) i2 + input_offset);
    i3 = (const float*) ((uintptr_t) i3 + input_offset);
    if (pooling_elements < 2) {
      i1 = i0;
    }
    if (pooling_elements <= 2) {
      i2 = i0;
    }
    if (pooling_elements != 4) {
      i3 = i0;
    }

    size_t c = channels;
    for (; c >= 4; c -= 4) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1 = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2 = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3 = _mm_loadu_ps(i3);
      i3 += 4;

      __m128 vmax = vi0;
      __m128i vidx = _mm_setzero_si128();

      const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
      vmax = _mm_max_ps(vi1, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, _mm_set1_epi32(1)));

      const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
      vmax = _mm_max_ps(vi2, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, _mm_set1_epi32(2)));

      const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
      vmax = _mm_max_ps(vi3, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, _mm_set1_epi32(3)));

      _mm_storeu_ps(output, vmax);
      output += 4;
      _mm_storeu_si128((__m128i*) index, vidx);
      index += 4;
    }
    if (c != 0) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      const __m128 vi1 = _mm_loadu_ps(i1);
      const __m128 vi2 = _mm_loadu_ps(i2);
      const __m128 vi3 = _mm_loadu_ps(i3);

      __m128 vmax = vi0;
      __m128i vidx = _mm_setzero_si128();

      const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
      vmax = _mm_max_ps(vi1, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, _mm_set1_epi32(1)));

      const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
      vmax = _mm_max_ps(vi2, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, _mm_set1_epi32(2)));

      const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
      vmax = _mm_max_ps(vi3, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, _mm_set1_epi32(3)));

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vmax);
        _mm_storel_epi64((__m128i*) index, vidx);
        vmax = _mm_movehl_ps(vmax, vmax);
        vidx = _mm_unpackhi_epi64(vidx, vidx);
        output += 2;
        index += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vmax);
        *index = (uint32_t) _mm_cvtsi128_si32(vidx);
        output += 1;
        index += 1;
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* accumulation_buffer,
    uint32_t* index_buffer,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements > 9);
  assert(channels != 0);

  do {
    {
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

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

      for (size_t c = 0; c < channels; c += 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vi8 = _mm_loadu_ps(i8);
        i8 += 4;

        __m128 vmax = vi0;
        __m128i vidx = _mm_setzero_si128();

        const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
        vmax = _mm_max_ps(vi1, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, _mm_set1_epi32(1)));

        const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
        vmax = _mm_max_ps(vi2, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, _mm_set1_epi32(2)));

        const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
        vmax = _mm_max_ps(vi3, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, _mm_set1_epi32(3)));

        const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
        vmax = _mm_max_ps(vi4, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, _mm_set1_epi32(4)));

        const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
        vmax = _mm_max_ps(vi5, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, _mm_set1_epi32(5)));

        const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
        vmax = _mm_max_ps(vi6, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, _mm_set1_epi32(6)));

        const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
        vmax = _mm_max_ps(vi7, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, _mm_set1_epi32(7)));

        const __m128i vm8 = _mm_castps_si128(_mm_cmpgt_ps(vi8, vmax));
        vmax = _mm_max_ps(vi8, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm8, vidx), _mm_and_si128(vm8, _mm_set1_epi32(8)));

        _mm_store_ps(ab, vmax);
        ab += 4;
        _mm_store_si128((__m128i*) ib, vidx);
        ib += 4;
      }
    }
    const __m128i v1 = _mm_set1_epi32(1);
    const __m128i v8 = _mm_set1_epi32(8);
    __m128i vidx0 = _mm_add_epi32(v1, v8);

    size_t k = pooling_elements;
    for (k -= 9; k > 8; k -= 8) {
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

      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

      for (size_t c = 0; c < channels; c += 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;

        __m128 vmax = _mm_load_ps(ab);
        __m128i vidx = _mm_load_si128((const __m128i*) ib);

        const __m128i vm0 = _mm_castps_si128(_mm_cmpgt_ps(vi0, vmax));
        vmax = _mm_max_ps(vi0, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm0, vidx), _mm_and_si128(vm0, vidx0));

        const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
        const __m128i vidx1 = _mm_add_epi32(vidx0, v1);
        vmax = _mm_max_ps(vi1, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, vidx1));

        const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
        const __m128i vidx2 = _mm_add_epi32(vidx1, v1);
        vmax = _mm_max_ps(vi2, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, vidx2));

        const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
        const __m128i vidx3 = _mm_add_epi32(vidx2, v1);
        vmax = _mm_max_ps(vi3, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, vidx3));

        const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
        const __m128i vidx4 = _mm_add_epi32(vidx3, v1);
        vmax = _mm_max_ps(vi4, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, vidx4));

        const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
        const __m128i vidx5 = _mm_add_epi32(vidx4, v1);
        vmax = _mm_max_ps(vi5, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, vidx5));

        const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
        const __m128i vidx6 = _mm_add_epi32(vidx5, v1);
        vmax = _mm_max_ps(vi6, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, vidx6));

        const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
        const __m128i vidx7 = _mm_add_epi32(vidx6, v1);
        vmax = _mm_max_ps(vi7, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, vidx7));

        _mm_store_ps(ab, vmax);
        ab += 4;
        _mm_store_si128((__m128i*) ib, vidx);
        ib += 4;
      }
      vidx0 = _mm_add_epi32(vidx0, v8);
    }

    float* o = output;
    uint32_t* i = index;
    {
      const float* i0 = input[0];
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      input = (const float**) ((uintptr_t) input + input_increment);
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
      if (k != 8) {
        i7 = i0;
      }

      size_t c = channels;
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;
      for (; c >= 4; c -= 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;

        __m128 vmax = _mm_load_ps(ab);
        ab += 4;
        __m128i vidx = _mm_load_si128((const __m128i*) ib);
        ib += 4;

        const __m128i vm0 = _mm_castps_si128(_mm_cmpgt_ps(vi0, vmax));
        vmax = _mm_max_ps(vi0, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm0, vidx), _mm_and_si128(vm0, vidx0));

        const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
        const __m128i vidx1 = _mm_add_epi32(vidx0, v1);
        vmax = _mm_max_ps(vi1, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, vidx1));

        const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
        const __m128i vidx2 = _mm_add_epi32(vidx1, v1);
        vmax = _mm_max_ps(vi2, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, vidx2));

        const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
        const __m128i vidx3 = _mm_add_epi32(vidx2, v1);
        vmax = _mm_max_ps(vi3, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, vidx3));

        const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
        const __m128i vidx4 = _mm_add_epi32(vidx3, v1);
        vmax = _mm_max_ps(vi4, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, vidx4));

        const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
        const __m128i vidx5 = _mm_add_epi32(vidx4, v1);
        vmax = _mm_max_ps(vi5, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, vidx5));

        const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
        const __m128i vidx6 = _mm_add_epi32(vidx5, v1);
        vmax = _mm_max_ps(vi6, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, vidx6));

        const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
        const __m128i vidx7 = _mm_add_epi32(vidx6, v1);
        vmax = _mm_max_ps(vi7, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, vidx7));

        _mm_storeu_ps(o, vmax);
        o += 4;
        _mm_storeu_si128((__m128i*) i, vidx);
        i += 4;
      }
      if (c != 0) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        const __m128 vi1 = _mm_loadu_ps(i1);
        const __m128 vi2 = _mm_loadu_ps(i2);
        const __m128 vi3 = _mm_loadu_ps(i3);
        const __m128 vi4 = _mm_loadu_ps(i4);
        const __m128 vi5 = _mm_loadu_ps(i5);
        const __m128 vi6 = _mm_loadu_ps(i6);
        const __m128 vi7 = _mm_loadu_ps(i7);

        __m128 vmax = _mm_load_ps(ab);
        __m128i vidx = _mm_load_si128((const __m128i*) ib);

        const __m128i vm0 = _mm_castps_si128(_mm_cmpgt_ps(vi0, vmax));
        vmax = _mm_max_ps(vi0, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm0, vidx), _mm_and_si128(vm0, vidx0));

        const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
        const __m128i vidx1 = _mm_add_epi32(vidx0, v1);
        vmax = _mm_max_ps(vi1, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, vidx1));

        const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
        const __m128i vidx2 = _mm_add_epi32(vidx1, v1);
        vmax = _mm_max_ps(vi2, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, vidx2));

        const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
        const __m128i vidx3 = _mm_add_epi32(vidx2, v1);
        vmax = _mm_max_ps(vi3, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, vidx3));

        const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
        const __m128i vidx4 = _mm_add_epi32(vidx3, v1);
        vmax = _mm_max_ps(vi4, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, vidx4));

        const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
        const __m128i vidx5 = _mm_add_epi32(vidx4, v1);
        vmax = _mm_max_ps(vi5, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, vidx5));

        const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
        const __m128i vidx6 = _mm_add_epi32(vidx5, v1);
        vmax = _mm_max_ps(vi6, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, vidx6));

        const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
        const __m128i vidx7 = _mm_add_epi32(vidx6, v1);
        vmax = _mm_max_ps(vi7, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, vidx7));

        if (c & 2) {
          _mm_storel_pi((__m64*) o, vmax);
          _mm_storel_epi64((__m128i*) i, vidx);
          vmax = _mm_movehl_ps(vmax, vmax);
          vidx = _mm_unpackhi_epi64(vidx, vidx);
          o += 2;
          i += 2;
        }
        if (c & 1) {
          _mm_store_ss(o, vmax);
          *i = (uint32_t) _mm_cvtsi128_si32(vidx);
          o += 1;
          i += 1;
        }
      }
    }

    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--output_pixels != 0);
}

void xnn_f32_argmaxpool_ukernel_9x__sse2_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 9);
  assert(channels != 0);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    i0 = (const float*) ((uintptr_t) i0 + input_offset);
    i1 = (const float*) ((uintptr_t) i1 + input_offset);
    i2 = (const float*) ((uintptr_t) i2 + input_offset);
    i3 = (const float*) ((uintptr_t) i3 + input_offset);
    i4 = (const float*) ((uintptr_t) i4 + input_offset);
    i5 = (const float*) ((uintptr_t) i5 + input_offset);
    i6 = (const float*) ((uintptr_t) i6 + input_offset);
    i7 = (const float*) ((uintptr_t) i7 + input_offset);
    i8 = (const float*) ((uintptr_t) i8 + input_offset);
    if (pooling_elements < 2) {
      i1 = i0;
    }
    if (pooling_elements <= 2) {
      i2 = i0;
    }
    if (pooling_elements < 4) {
      i3 = i0;
    }
    if (pooling_elements <= 4) {
      i4 = i0;
    }
    if (pooling_elements < 6) {
      i5 = i0;
    }
    if (pooling_elements <= 6) {
      i6 = i0;
    }
    if (pooling_elements < 8) {
      i7 = i0;
    }
    if (pooling_elements <= 8) {
      i8 = i0;
    }

    size_t c = channels;
    for (; c >= 4; c -= 4) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1 = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2 = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3 = _mm_loadu_ps(i3);
      i3 += 4;
      const __m128 vi4 = _mm_loadu_ps(i4);
      i4 += 4;
      const __m128 vi5 = _mm_loadu_ps(i5);
      i5 += 4;
      const __m128 vi6 = _mm_loadu_ps(i6);
      i6 += 4;
      const __m128 vi7 = _mm_loadu_ps(i7);
      i7 += 4;
      const __m128 vi8 = _mm_loadu_ps(i8);
      i8 += 4;

      __m128 vmax = vi0;
      __m128i vidx = _mm_setzero_si128();

      const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
      vmax = _mm_max_ps(vi1, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, _mm_set1_epi32(1)));

      const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
      vmax = _mm_max_ps(vi2, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, _mm_set1_epi32(2)));

      const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
      vmax = _mm_max_ps(vi3, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, _mm_set1_epi32(3)));

      const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
      vmax = _mm_max_ps(vi4, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, _mm_set1_epi32(4)));

      const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
      vmax = _mm_max_ps(vi5, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, _mm_set1_epi32(5)));

      const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
      vmax = _mm_max_ps(vi6, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, _mm_set1_epi32(6)));

      const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
      vmax = _mm_max_ps(vi7, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, _mm_set1_epi32(7)));

      const __m128i vm8 = _mm_castps_si128(_mm_cmpgt_ps(vi8, vmax));
      vmax = _mm_max_ps(vi8, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm8, vidx), _mm_and_si128(vm8, _mm_set1_epi32(8)));

      _mm_storeu_ps(output, vmax);
      output += 4;
      _mm_storeu_si128((__m128i*) index, vidx);
      index += 4;
    }
    if (c != 0) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      const __m128 vi1 = _mm_loadu_ps(i1);
      const __m128 vi2 = _mm_loadu_ps(i2);
      const __m128 vi3 = _mm_loadu_ps(i3);
      const __m128 vi4 = _mm_loadu_ps(i4);
      const __m128 vi5 = _mm_loadu_ps(i5);
      const __m128 vi6 = _mm_loadu_ps(i6);
      const __m128 vi7 = _mm_loadu_ps(i7);
      const __m128 vi8 = _mm_loadu_ps(i8);

      __m128 vmax = vi0;
      __m128i vidx = _mm_setzero_si128();

      const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
      vmax = _mm_max_ps(vi1, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, _mm_set1_epi32(1)));

      const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
      vmax = _mm_max_ps(vi2, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, _mm_set1_epi32(2)));

      const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
      vmax = _mm_max_ps(vi3, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, _mm_set1_epi32(3)));

      const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
      vmax = _mm_max_ps(vi4, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, _mm_set1_epi32(4)));

      const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
      vmax = _mm_max_ps(vi5, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, _mm_set1_epi32(5)));

      const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
      vmax = _mm_max_ps(vi6, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, _mm_set1_epi32(6)));

      const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
      vmax = _mm_max_ps(vi7, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, _mm_set1_epi32(7)));

      const __m128i vm8 = _mm_castps_si128(_mm_cmpgt_ps(vi8, vmax));
      vmax = _mm_max_ps(vi8, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm8, vidx), _mm_and_si128(vm8, _mm_set1_epi32(8)));

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vmax);
        _mm_storel_epi64((__m128i*) index, vidx);
        vmax = _mm_movehl_ps(vmax, vmax);
        vidx = _mm_unpackhi_epi64(vidx, vidx);
        output += 2;
        index += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vmax);
        *index = (uint32_t) _mm_cvtsi128_si32(vidx);
        output += 1;
        index += 1;
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
    index = (uint32_t*) index;
  } while (--output_pixels != 0);
}

void xnn_f32_f16_vcvt_ukernel__sse2_u16(
    size_t batch,
    const float* input,
    void* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vnonsign_mask = _mm_castsi128_ps(_mm_set1_epi32(UINT32_C(0x7FFFFFFF)));
  const __m128i vexp_bias = _mm_set1_epi32(UINT32_C(0x07800000));
  const __m128 vscale_to_inf = _mm_set1_ps(0x1.0p+112f);
  const __m128i vexpw_max = _mm_set1_epi32(UINT32_C(0x7F800000));
  const __m128 vscale_to_zero = _mm_set1_ps(0x1.0p-110f);
  const __m128i vbias_min = _mm_set1_epi32(UINT32_C(0x40008000));  // 0x8000, 0x4000, 0x8000, 0x4000, ...
  const __m128i vmanth_mask = _mm_set1_epi32(UINT32_C(0x00000FFF));
  const __m128i vexph_mask = _mm_set1_epi32(UINT32_C(0x00007C00));
  const __m128i vnanh = _mm_set1_epi16(UINT16_C(0x7E00));

  XNN_FORCE_REALIZATION(vnonsign_mask);
  XNN_FORCE_REALIZATION(vexp_bias);
  XNN_FORCE_REALIZATION(vscale_to_inf);
  XNN_FORCE_REALIZATION(vexpw_max);
  XNN_FORCE_REALIZATION(vscale_to_zero);
  XNN_FORCE_REALIZATION(vbias_min);
  XNN_FORCE_REALIZATION(vmanth_mask);
  XNN_FORCE_REALIZATION(vexph_mask);
  XNN_FORCE_REALIZATION(vnanh);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m128 vx0 = _mm_loadu_ps(input);
    const __m128 vx1 = _mm_loadu_ps(input + 4);
    const __m128 vx2 = _mm_loadu_ps(input + 8);
    const __m128 vx3 = _mm_loadu_ps(input + 12);
    input += 16;

    const __m128 vabsx0 = _mm_and_ps(vx0, vnonsign_mask);
    const __m128 vabsx1 = _mm_and_ps(vx1, vnonsign_mask);
    const __m128 vabsx2 = _mm_and_ps(vx2, vnonsign_mask);
    const __m128 vabsx3 = _mm_and_ps(vx3, vnonsign_mask);

    const __m128 vsignx0 = _mm_xor_ps(vx0, vabsx0);
    const __m128 vsignx1 = _mm_xor_ps(vx1, vabsx1);
    const __m128 vsignx2 = _mm_xor_ps(vx2, vabsx2);
    const __m128 vsignx3 = _mm_xor_ps(vx3, vabsx3);

    __m128i vbias0 = _mm_add_epi32(_mm_castps_si128(vabsx0), vexp_bias);
    __m128i vbias1 = _mm_add_epi32(_mm_castps_si128(vabsx1), vexp_bias);
    __m128i vbias2 = _mm_add_epi32(_mm_castps_si128(vabsx2), vexp_bias);
    __m128i vbias3 = _mm_add_epi32(_mm_castps_si128(vabsx3), vexp_bias);

    __m128 vf0 = _mm_mul_ps(vabsx0, vscale_to_inf);
    __m128 vf1 = _mm_mul_ps(vabsx1, vscale_to_inf);
    __m128 vf2 = _mm_mul_ps(vabsx2, vscale_to_inf);
    __m128 vf3 = _mm_mul_ps(vabsx3, vscale_to_inf);

    const __m128i vnanmaskw0 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx0), vexpw_max);
    const __m128i vnanmaskw1 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx1), vexpw_max);
    const __m128i vnanmaskw2 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx2), vexpw_max);
    const __m128i vnanmaskw3 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx3), vexpw_max);

    vbias0 = _mm_and_si128(vbias0, vexpw_max);
    vbias1 = _mm_and_si128(vbias1, vexpw_max);
    vbias2 = _mm_and_si128(vbias2, vexpw_max);
    vbias3 = _mm_and_si128(vbias3, vexpw_max);

    vf0 = _mm_mul_ps(vf0, vscale_to_zero);
    vf1 = _mm_mul_ps(vf1, vscale_to_zero);
    vf2 = _mm_mul_ps(vf2, vscale_to_zero);
    vf3 = _mm_mul_ps(vf3, vscale_to_zero);

    const __m128i vnanmaskh0 = _mm_packs_epi32(vnanmaskw0, vnanmaskw1);
    const __m128i vnanmaskh1 = _mm_packs_epi32(vnanmaskw2, vnanmaskw3);

    const __m128i vsignh0 = _mm_packs_epi32(_mm_castps_si128(vsignx0), _mm_castps_si128(vsignx1));
    const __m128i vsignh1 = _mm_packs_epi32(_mm_castps_si128(vsignx2), _mm_castps_si128(vsignx3));

    vbias0 = _mm_max_epi16(vbias0, vbias_min);
    vbias1 = _mm_max_epi16(vbias1, vbias_min);
    vbias2 = _mm_max_epi16(vbias2, vbias_min);
    vbias3 = _mm_max_epi16(vbias3, vbias_min);

    __m128i vh0 = _mm_and_si128(vnanh, vnanmaskh0);
    __m128i vh1 = _mm_and_si128(vnanh, vnanmaskh1);

    vf0 = _mm_add_ps(vf0, _mm_castsi128_ps(vbias0));
    vf1 = _mm_add_ps(vf1, _mm_castsi128_ps(vbias1));
    vf2 = _mm_add_ps(vf2, _mm_castsi128_ps(vbias2));
    vf3 = _mm_add_ps(vf3, _mm_castsi128_ps(vbias3));

    vh0 = _mm_or_si128(vh0, vsignh0);
    vh1 = _mm_or_si128(vh1, vsignh1);

    __m128i vexpw0 = _mm_srli_epi32(_mm_castps_si128(vf0), 13);
    __m128i vexpw1 = _mm_srli_epi32(_mm_castps_si128(vf1), 13);
    __m128i vexpw2 = _mm_srli_epi32(_mm_castps_si128(vf2), 13);
    __m128i vexpw3 = _mm_srli_epi32(_mm_castps_si128(vf3), 13);

    const __m128i vmantw0 = _mm_and_si128(_mm_castps_si128(vf0), vmanth_mask);
    const __m128i vmantw1 = _mm_and_si128(_mm_castps_si128(vf1), vmanth_mask);
    const __m128i vmantw2 = _mm_and_si128(_mm_castps_si128(vf2), vmanth_mask);
    const __m128i vmantw3 = _mm_and_si128(_mm_castps_si128(vf3), vmanth_mask);

    vexpw0 = _mm_and_si128(vexpw0, vexph_mask);
    vexpw1 = _mm_and_si128(vexpw1, vexph_mask);
    vexpw2 = _mm_and_si128(vexpw2, vexph_mask);
    vexpw3 = _mm_and_si128(vexpw3, vexph_mask);

    const __m128i vnonsignw0 = _mm_add_epi32(vmantw0, vexpw0);
    const __m128i vnonsignw1 = _mm_add_epi32(vmantw1, vexpw1);
    const __m128i vnonsignw2 = _mm_add_epi32(vmantw2, vexpw2);
    const __m128i vnonsignw3 = _mm_add_epi32(vmantw3, vexpw3);

    const __m128i vnonsignh0 = _mm_packs_epi32(vnonsignw0, vnonsignw1);
    const __m128i vnonsignh1 = _mm_packs_epi32(vnonsignw2, vnonsignw3);

    vh0 = _mm_or_si128(vh0, _mm_andnot_si128(vnanmaskh0, vnonsignh0));
    vh1 = _mm_or_si128(vh1, _mm_andnot_si128(vnanmaskh1, vnonsignh1));

    _mm_storeu_si128((__m128i*) o, vh0);
    _mm_storeu_si128((__m128i*) (o + 8), vh1);
    o += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
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
    __m128i vh = _mm_and_si128(vnanh, vnanmaskh);

    vf_lo = _mm_add_ps(vf_lo, _mm_castsi128_ps(vbias_lo));
    vf_hi = _mm_add_ps(vf_hi, _mm_castsi128_ps(vbias_hi));
    vh = _mm_or_si128(vh, vsignh);

    __m128i vexpw_lo = _mm_srli_epi32(_mm_castps_si128(vf_lo), 13);
    __m128i vexpw_hi = _mm_srli_epi32(_mm_castps_si128(vf_hi), 13);
    const __m128i vmantw_lo = _mm_and_si128(_mm_castps_si128(vf_lo), vmanth_mask);
    const __m128i vmantw_hi = _mm_and_si128(_mm_castps_si128(vf_hi), vmanth_mask);

    vexpw_lo = _mm_and_si128(vexpw_lo, vexph_mask);
    vexpw_hi = _mm_and_si128(vexpw_hi, vexph_mask);

    const __m128i vnonsignw_lo = _mm_add_epi32(vmantw_lo, vexpw_lo);
    const __m128i vnonsignw_hi = _mm_add_epi32(vmantw_hi, vexpw_hi);

    const __m128i vnonsignh = _mm_packs_epi32(vnonsignw_lo, vnonsignw_hi);

    vh = _mm_or_si128(vh, _mm_andnot_si128(vnanmaskh, vnonsignh));

    _mm_storeu_si128((__m128i*) o, vh);
    o += 8;
  }
  if XNN_UNPREDICTABLE(batch != 0) {
    const __m128 vx_lo = _mm_loadu_ps(input);
    const float* input_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
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
    __m128i vh = _mm_and_si128(vnanh, vnanmaskh);

    vf_lo = _mm_add_ps(vf_lo, _mm_castsi128_ps(vbias_lo));
    vf_hi = _mm_add_ps(vf_hi, _mm_castsi128_ps(vbias_hi));
    vh = _mm_or_si128(vh, vsignh);

    __m128i vexpw_lo = _mm_srli_epi32(_mm_castps_si128(vf_lo), 13);
    __m128i vexpw_hi = _mm_srli_epi32(_mm_castps_si128(vf_hi), 13);
    const __m128i vmantw_lo = _mm_and_si128(_mm_castps_si128(vf_lo), vmanth_mask);
    const __m128i vmantw_hi = _mm_and_si128(_mm_castps_si128(vf_hi), vmanth_mask);

    vexpw_lo = _mm_and_si128(vexpw_lo, vexph_mask);
    vexpw_hi = _mm_and_si128(vexpw_hi, vexph_mask);

    const __m128i vnonsignw_lo = _mm_add_epi32(vmantw_lo, vexpw_lo);
    const __m128i vnonsignw_hi = _mm_add_epi32(vmantw_hi, vexpw_hi);

    const __m128i vnonsignh = _mm_packs_epi32(vnonsignw_lo, vnonsignw_hi);

    vh = _mm_or_si128(vh, _mm_andnot_si128(vnanmaskh, vnonsignh));

    if (batch & (4 * sizeof(float))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(float))) {
      unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vh));
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(float))) {
      *o = (uint16_t) _mm_cvtsi128_si32(vh);
    }
  }
}

void xnn_f32_prelu_ukernel__sse2_2x8(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride) XNN_OOB_READS
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
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const __m128 vw0123 = _mm_load_ps(w);
      const __m128 vw4567 = _mm_load_ps(w + 4);
      w += 8;

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
      i0 += 8;
      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
      i1 += 8;

      const __m128 vprod0x0123 = _mm_mul_ps(vi0x0123, vw0123);
      const __m128 vmask0x0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vi0x0123)));
      const __m128 vprod0x4567 = _mm_mul_ps(vi0x4567, vw4567);
      const __m128 vmask0x4567 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vi0x4567)));
      const __m128 vprod1x0123 = _mm_mul_ps(vi1x0123, vw0123);
      const __m128 vmask1x0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vi1x0123)));
      const __m128 vprod1x4567 = _mm_mul_ps(vi1x4567, vw4567);
      const __m128 vmask1x4567 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vi1x4567)));

      const __m128 vacc0x0123 = _mm_or_ps(_mm_and_ps(vprod0x0123, vmask0x0123), _mm_andnot_ps(vmask0x0123, vi0x0123));
      const __m128 vacc0x4567 = _mm_or_ps(_mm_and_ps(vprod0x4567, vmask0x4567), _mm_andnot_ps(vmask0x4567, vi0x4567));
      const __m128 vacc1x0123 = _mm_or_ps(_mm_and_ps(vprod1x0123, vmask1x0123), _mm_andnot_ps(vmask1x0123, vi1x0123));
      const __m128 vacc1x4567 = _mm_or_ps(_mm_and_ps(vprod1x4567, vmask1x4567), _mm_andnot_ps(vmask1x4567, vi1x4567));

      _mm_storeu_ps(o0, vacc0x0123);
      _mm_storeu_ps(o0 + 4, vacc0x4567);
      o0 += 8;
      _mm_storeu_ps(o1, vacc1x0123);
      _mm_storeu_ps(o1 + 4, vacc1x4567);
      o1 += 8;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const __m128 vw0123 = _mm_load_ps(w);
      w += 4;

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      const __m128 vprod0x0123 = _mm_mul_ps(vi0x0123, vw0123);
      const __m128 vmask0x0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vi0x0123)));
      const __m128 vprod1x0123 = _mm_mul_ps(vi1x0123, vw0123);
      const __m128 vmask1x0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vi1x0123)));

      __m128 vacc0x0123 = _mm_or_ps(_mm_and_ps(vprod0x0123, vmask0x0123), _mm_andnot_ps(vmask0x0123, vi0x0123));
      __m128 vacc1x0123 = _mm_or_ps(_mm_and_ps(vprod1x0123, vmask1x0123), _mm_andnot_ps(vmask1x0123, vi1x0123));

      _mm_storeu_ps(o0, vacc0x0123);
      o0 += 4;
      _mm_storeu_ps(o1, vacc1x0123);
      o1 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m128 vw0123 = _mm_load_ps(w);
      w = (const float*) ((uintptr_t) w + c);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const __m128 vprod0x0123 = _mm_mul_ps(vi0x0123, vw0123);
      const __m128 vmask0x0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vi0x0123)));
      const __m128 vprod1x0123 = _mm_mul_ps(vi1x0123, vw0123);
      const __m128 vmask1x0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vi1x0123)));

      __m128 vacc0x0123 = _mm_or_ps(_mm_and_ps(vprod0x0123, vmask0x0123), _mm_andnot_ps(vmask0x0123, vi0x0123));
      __m128 vacc1x0123 = _mm_or_ps(_mm_and_ps(vprod1x0123, vmask1x0123), _mm_andnot_ps(vmask1x0123, vi1x0123));

      if (c & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) o0, vacc0x0123);
        _mm_storel_pi((__m64*) o1, vacc1x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        _mm_store_ss(o0, vacc0x0123);
        _mm_store_ss(o1, vacc1x0123);

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

void xnn_f32_qs8_vcvt_ukernel__sse2_u32(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128 voutput_max_less_zero_point = _mm_set1_ps((float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point));
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi16(params->scalar.output_min);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vx4567 = _mm_loadu_ps(input + 4);
    __m128 vx89AB = _mm_loadu_ps(input + 8);
    __m128 vxCDEF = _mm_loadu_ps(input + 12);
    __m128 vxGHIJ = _mm_loadu_ps(input + 16);
    __m128 vxKLMN = _mm_loadu_ps(input + 20);
    __m128 vxOPQR = _mm_loadu_ps(input + 24);
    __m128 vxSTUV = _mm_loadu_ps(input + 28);
    input += 32;

    vx0123 = _mm_mul_ps(vx0123, vscale);
    vx4567 = _mm_mul_ps(vx4567, vscale);
    vx89AB = _mm_mul_ps(vx89AB, vscale);
    vxCDEF = _mm_mul_ps(vxCDEF, vscale);
    vxGHIJ = _mm_mul_ps(vxGHIJ, vscale);
    vxKLMN = _mm_mul_ps(vxKLMN, vscale);
    vxOPQR = _mm_mul_ps(vxOPQR, vscale);
    vxSTUV = _mm_mul_ps(vxSTUV, vscale);

    vx0123 = _mm_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm_min_ps(vx4567, voutput_max_less_zero_point);
    vx89AB = _mm_min_ps(vx89AB, voutput_max_less_zero_point);
    vxCDEF = _mm_min_ps(vxCDEF, voutput_max_less_zero_point);
    vxGHIJ = _mm_min_ps(vxGHIJ, voutput_max_less_zero_point);
    vxKLMN = _mm_min_ps(vxKLMN, voutput_max_less_zero_point);
    vxOPQR = _mm_min_ps(vxOPQR, voutput_max_less_zero_point);
    vxSTUV = _mm_min_ps(vxSTUV, voutput_max_less_zero_point);

    const __m128i vy0123 = _mm_cvtps_epi32(vx0123);
    const __m128i vy4567 = _mm_cvtps_epi32(vx4567);
    const __m128i vy89AB = _mm_cvtps_epi32(vx89AB);
    const __m128i vyCDEF = _mm_cvtps_epi32(vxCDEF);
    const __m128i vyGHIJ = _mm_cvtps_epi32(vxGHIJ);
    const __m128i vyKLMN = _mm_cvtps_epi32(vxKLMN);
    const __m128i vyOPQR = _mm_cvtps_epi32(vxOPQR);
    const __m128i vySTUV = _mm_cvtps_epi32(vxSTUV);

    __m128i vy01234567 = _mm_packs_epi32(vy0123, vy4567);
    __m128i vy89ABCDEF = _mm_packs_epi32(vy89AB, vyCDEF);
    __m128i vyGHIJKLMN = _mm_packs_epi32(vyGHIJ, vyKLMN);
    __m128i vyOPQRSTUV = _mm_packs_epi32(vyOPQR, vySTUV);

    vy01234567 = _mm_adds_epi16(vy01234567, voutput_zero_point);
    vy89ABCDEF = _mm_adds_epi16(vy89ABCDEF, voutput_zero_point);
    vyGHIJKLMN = _mm_adds_epi16(vyGHIJKLMN, voutput_zero_point);
    vyOPQRSTUV = _mm_adds_epi16(vyOPQRSTUV, voutput_zero_point);

    vy01234567 = _mm_max_epi16(vy01234567, voutput_min);
    vy89ABCDEF = _mm_max_epi16(vy89ABCDEF, voutput_min);
    vyGHIJKLMN = _mm_max_epi16(vyGHIJKLMN, voutput_min);
    vyOPQRSTUV = _mm_max_epi16(vyOPQRSTUV, voutput_min);

    __m128i vy0123456789ABCDEF = _mm_packs_epi16(vy01234567, vy89ABCDEF);
    __m128i vyGHIJKLMNOPQRSTUV = _mm_packs_epi16(vyGHIJKLMN, vyOPQRSTUV);


    _mm_storeu_si128((__m128i*) output, vy0123456789ABCDEF);
    _mm_storeu_si128((__m128i*) (output + 16), vyGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m128 vx_lo = _mm_loadu_ps(input);
    __m128 vx_hi = _mm_loadu_ps(input + 4);
    input += 8;

    vx_lo = _mm_mul_ps(vx_lo, vscale);
    vx_hi = _mm_mul_ps(vx_hi, vscale);

    vx_lo = _mm_min_ps(vx_lo, voutput_max_less_zero_point);
    vx_hi = _mm_min_ps(vx_hi, voutput_max_less_zero_point);

    const __m128i vy_lo = _mm_cvtps_epi32(vx_lo);
    const __m128i vy_hi = _mm_cvtps_epi32(vx_hi);

    __m128i vy = _mm_packs_epi32(vy_lo, vy_hi);
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_max_epi16(vy, voutput_min);
    vy = _mm_packs_epi16(vy, vy);

    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx_lo = _mm_loadu_ps(input);
    const float* x_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    __m128 vx_hi = _mm_loadu_ps(x_hi);

    vx_lo = _mm_mul_ps(vx_lo, vscale);
    vx_hi = _mm_mul_ps(vx_hi, vscale);

    vx_lo = _mm_min_ps(vx_lo, voutput_max_less_zero_point);
    vx_hi = _mm_min_ps(vx_hi, voutput_max_less_zero_point);

    const __m128i vy_lo = _mm_cvtps_epi32(vx_lo);
    const __m128i vy_hi = _mm_cvtps_epi32(vx_hi);

    __m128i vy = _mm_packs_epi32(vy_lo, vy_hi);
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_max_epi16(vy, voutput_min);
    vy = _mm_packs_epi16(vy, vy);

    if (batch & (4 * sizeof(float))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      output += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    {
      uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
      if (batch & (2 * sizeof(float))) {
        unaligned_store_u16(output, (uint16_t) vy_lo);
        output += 2;
        vy_lo >>= 16;
      }
      if (batch & (1 * sizeof(float))) {
        *output = (int8_t) vy_lo;
      }
    }
  }
}

void xnn_f32_qu8_vcvt_ukernel__sse2_u32(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128 voutput_max_less_zero_point = _mm_set1_ps((float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point));
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vx4567 = _mm_loadu_ps(input + 4);
    __m128 vx89AB = _mm_loadu_ps(input + 8);
    __m128 vxCDEF = _mm_loadu_ps(input + 12);
    __m128 vxGHIJ = _mm_loadu_ps(input + 16);
    __m128 vxKLMN = _mm_loadu_ps(input + 20);
    __m128 vxOPQR = _mm_loadu_ps(input + 24);
    __m128 vxSTUV = _mm_loadu_ps(input + 28);
    input += 32;

    vx0123 = _mm_mul_ps(vx0123, vscale);
    vx4567 = _mm_mul_ps(vx4567, vscale);
    vx89AB = _mm_mul_ps(vx89AB, vscale);
    vxCDEF = _mm_mul_ps(vxCDEF, vscale);
    vxGHIJ = _mm_mul_ps(vxGHIJ, vscale);
    vxKLMN = _mm_mul_ps(vxKLMN, vscale);
    vxOPQR = _mm_mul_ps(vxOPQR, vscale);
    vxSTUV = _mm_mul_ps(vxSTUV, vscale);

    vx0123 = _mm_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm_min_ps(vx4567, voutput_max_less_zero_point);
    vx89AB = _mm_min_ps(vx89AB, voutput_max_less_zero_point);
    vxCDEF = _mm_min_ps(vxCDEF, voutput_max_less_zero_point);
    vxGHIJ = _mm_min_ps(vxGHIJ, voutput_max_less_zero_point);
    vxKLMN = _mm_min_ps(vxKLMN, voutput_max_less_zero_point);
    vxOPQR = _mm_min_ps(vxOPQR, voutput_max_less_zero_point);
    vxSTUV = _mm_min_ps(vxSTUV, voutput_max_less_zero_point);

    const __m128i vy0123 = _mm_cvtps_epi32(vx0123);
    const __m128i vy4567 = _mm_cvtps_epi32(vx4567);
    const __m128i vy89AB = _mm_cvtps_epi32(vx89AB);
    const __m128i vyCDEF = _mm_cvtps_epi32(vxCDEF);
    const __m128i vyGHIJ = _mm_cvtps_epi32(vxGHIJ);
    const __m128i vyKLMN = _mm_cvtps_epi32(vxKLMN);
    const __m128i vyOPQR = _mm_cvtps_epi32(vxOPQR);
    const __m128i vySTUV = _mm_cvtps_epi32(vxSTUV);

    __m128i vy01234567 = _mm_packs_epi32(vy0123, vy4567);
    __m128i vy89ABCDEF = _mm_packs_epi32(vy89AB, vyCDEF);
    __m128i vyGHIJKLMN = _mm_packs_epi32(vyGHIJ, vyKLMN);
    __m128i vyOPQRSTUV = _mm_packs_epi32(vyOPQR, vySTUV);

    vy01234567 = _mm_adds_epi16(vy01234567, voutput_zero_point);
    vy89ABCDEF = _mm_adds_epi16(vy89ABCDEF, voutput_zero_point);
    vyGHIJKLMN = _mm_adds_epi16(vyGHIJKLMN, voutput_zero_point);
    vyOPQRSTUV = _mm_adds_epi16(vyOPQRSTUV, voutput_zero_point);


    __m128i vy0123456789ABCDEF = _mm_packus_epi16(vy01234567, vy89ABCDEF);
    __m128i vyGHIJKLMNOPQRSTUV = _mm_packus_epi16(vyGHIJKLMN, vyOPQRSTUV);

    vy0123456789ABCDEF = _mm_max_epu8(vy0123456789ABCDEF, voutput_min);
    vyGHIJKLMNOPQRSTUV = _mm_max_epu8(vyGHIJKLMNOPQRSTUV, voutput_min);

    _mm_storeu_si128((__m128i*) output, vy0123456789ABCDEF);
    _mm_storeu_si128((__m128i*) (output + 16), vyGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m128 vx_lo = _mm_loadu_ps(input);
    __m128 vx_hi = _mm_loadu_ps(input + 4);
    input += 8;

    vx_lo = _mm_mul_ps(vx_lo, vscale);
    vx_hi = _mm_mul_ps(vx_hi, vscale);

    vx_lo = _mm_min_ps(vx_lo, voutput_max_less_zero_point);
    vx_hi = _mm_min_ps(vx_hi, voutput_max_less_zero_point);

    const __m128i vy_lo = _mm_cvtps_epi32(vx_lo);
    const __m128i vy_hi = _mm_cvtps_epi32(vx_hi);

    __m128i vy = _mm_packs_epi32(vy_lo, vy_hi);
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packus_epi16(vy, vy);
    vy = _mm_max_epu8(vy, voutput_min);

    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx_lo = _mm_loadu_ps(input);
    const float* x_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    __m128 vx_hi = _mm_loadu_ps(x_hi);

    vx_lo = _mm_mul_ps(vx_lo, vscale);
    vx_hi = _mm_mul_ps(vx_hi, vscale);

    vx_lo = _mm_min_ps(vx_lo, voutput_max_less_zero_point);
    vx_hi = _mm_min_ps(vx_hi, voutput_max_less_zero_point);

    const __m128i vy_lo = _mm_cvtps_epi32(vx_lo);
    const __m128i vy_hi = _mm_cvtps_epi32(vx_hi);

    __m128i vy = _mm_packs_epi32(vy_lo, vy_hi);
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packus_epi16(vy, vy);
    vy = _mm_max_epu8(vy, voutput_min);

    if (batch & (4 * sizeof(float))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      output += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    {
      uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
      if (batch & (2 * sizeof(float))) {
        unaligned_store_u16(output, (uint16_t) vy_lo);
        output += 2;
        vy_lo >>= 16;
      }
      if (batch & (1 * sizeof(float))) {
        *output = (uint8_t) vy_lo;
      }
    }
  }
}

void xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc2(
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

  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.8000FEp23f);
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  const __m128 vc5 = _mm_set1_ps(0x1.0F9F9Cp-7f);
  const __m128 vc4 = _mm_set1_ps(0x1.573A1Ap-5f);
  const __m128 vc3 = _mm_set1_ps(0x1.555A80p-3f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFDC6p-2f);
  const __m128 vc1 = _mm_set1_ps(0x1.FFFFF6p-1f);
  const __m128 vdenorm_cutoff = _mm_set1_ps(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const __m128 vi_max = _mm_load1_ps(max);

  __m128 vacc0 = _mm_setzero_ps();
  __m128 vacc1 = _mm_setzero_ps();
  for (; batch >= 20 * sizeof(float); batch -= 20 * sizeof(float)) {
    // Load 20 (5x4) inputs at a time.
    const __m128 vi0123 = _mm_loadu_ps(input);
    const __m128 vi4567 = _mm_loadu_ps(input + 4);
    const __m128 vi89AB = _mm_loadu_ps(input + 8);
    const __m128 viCDEF = _mm_loadu_ps(input + 12);
    const __m128 viGHIJ = _mm_loadu_ps(input + 16);
    input += 20;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx0123 = _mm_sub_ps(vi0123, vi_max);
    const __m128 vx4567 = _mm_sub_ps(vi4567, vi_max);
    const __m128 vx89AB = _mm_sub_ps(vi89AB, vi_max);
    const __m128 vxCDEF = _mm_sub_ps(viCDEF, vi_max);
    const __m128 vxGHIJ = _mm_sub_ps(viGHIJ, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vx0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vx4567, vlog2e), vmagic_bias);
    __m128 vn89AB = _mm_add_ps(_mm_mul_ps(vx89AB, vlog2e), vmagic_bias);
    __m128 vnCDEF = _mm_add_ps(_mm_mul_ps(vxCDEF, vlog2e), vmagic_bias);
    __m128 vnGHIJ = _mm_add_ps(_mm_mul_ps(vxGHIJ, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn0123), 23));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn4567), 23));
    const __m128 vs89AB = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn89AB), 23));
    const __m128 vsCDEF = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnCDEF), 23));
    const __m128 vsGHIJ = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnGHIJ), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);
    vn89AB = _mm_sub_ps(vn89AB, vmagic_bias);
    vnCDEF = _mm_sub_ps(vnCDEF, vmagic_bias);
    vnGHIJ = _mm_sub_ps(vnGHIJ, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_hi), vx0123);
    __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_hi), vx4567);
    __m128 vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_hi), vx89AB);
    __m128 vtCDEF = _mm_add_ps(_mm_mul_ps(vnCDEF, vminus_ln2_hi), vxCDEF);
    __m128 vtGHIJ = _mm_add_ps(_mm_mul_ps(vnGHIJ, vminus_ln2_hi), vxGHIJ);

    vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_lo), vt4567);
    vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_lo), vt89AB);
    vtCDEF = _mm_add_ps(_mm_mul_ps(vnCDEF, vminus_ln2_lo), vtCDEF);
    vtGHIJ = _mm_add_ps(_mm_mul_ps(vnGHIJ, vminus_ln2_lo), vtGHIJ);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc5, vt0123), vc4);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc5, vt4567), vc4);
    __m128 vp89AB = _mm_add_ps(_mm_mul_ps(vc5, vt89AB), vc4);
    __m128 vpCDEF = _mm_add_ps(_mm_mul_ps(vc5, vtCDEF), vc4);
    __m128 vpGHIJ = _mm_add_ps(_mm_mul_ps(vc5, vtGHIJ), vc4);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc3);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc3);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc3);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc3);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc3);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc2);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc2);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc2);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc2);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc2);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc1);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc1);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc1);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc1);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0123 = _mm_mul_ps(vt0123, vs0123);
    vt4567 = _mm_mul_ps(vt4567, vs4567);
    vt89AB = _mm_mul_ps(vt89AB, vs89AB);
    vtCDEF = _mm_mul_ps(vtCDEF, vsCDEF);
    vtGHIJ = _mm_mul_ps(vtGHIJ, vsGHIJ);

    __m128 vf0123 = _mm_add_ps(_mm_mul_ps(vt0123, vp0123), vs0123);
    __m128 vf4567 = _mm_add_ps(_mm_mul_ps(vt4567, vp4567), vs4567);
    __m128 vf89AB = _mm_add_ps(_mm_mul_ps(vt89AB, vp89AB), vs89AB);
    __m128 vfCDEF = _mm_add_ps(_mm_mul_ps(vtCDEF, vpCDEF), vsCDEF);
    __m128 vfGHIJ = _mm_add_ps(_mm_mul_ps(vtGHIJ, vpGHIJ), vsGHIJ);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = _mm_andnot_ps(_mm_cmplt_ps(vx0123, vdenorm_cutoff), vf0123);
    vf4567 = _mm_andnot_ps(_mm_cmplt_ps(vx4567, vdenorm_cutoff), vf4567);
    vf89AB = _mm_andnot_ps(_mm_cmplt_ps(vx89AB, vdenorm_cutoff), vf89AB);
    vfCDEF = _mm_andnot_ps(_mm_cmplt_ps(vxCDEF, vdenorm_cutoff), vfCDEF);
    vfGHIJ = _mm_andnot_ps(_mm_cmplt_ps(vxGHIJ, vdenorm_cutoff), vfGHIJ);

    // Store 20 (5x4) outputs at a time.
    _mm_storeu_ps(output, vf0123);
    _mm_storeu_ps(output + 4, vf4567);
    _mm_storeu_ps(output + 8, vf89AB);
    _mm_storeu_ps(output + 12, vfCDEF);
    _mm_storeu_ps(output + 16, vfGHIJ);
    output += 20;

    // Accumulate computed exponents.
    vacc0 = _mm_add_ps(vacc0, vf0123);
    vacc0 = _mm_add_ps(vacc0, vf4567);
    vacc0 = _mm_add_ps(vacc0, vf89AB);
    vacc0 = _mm_add_ps(vacc0, vfCDEF);
    vacc0 = _mm_add_ps(vacc0, vfGHIJ);
  }
  // Add up all accumulators to vacc0
  vacc0 = _mm_add_ps(vacc0, vacc1);

  __m128 vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    // Load 4 inputs at a time.
    const __m128 vi = _mm_loadu_ps(input);
    input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx = _mm_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vx);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm_mul_ps(vt, vs);
    __m128 vf = _mm_add_ps(_mm_mul_ps(vt, vp), vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vx, vdenorm_cutoff), vf);

    // Store 4 outputs at a time.
    _mm_storeu_ps(output, vf);
    output += 4;

    // Accumulate computed exponents.
    vacc = _mm_add_ps(vacc, vf);
  }
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    // Load 4 inputs at a time.
    const __m128 vi = _mm_loadu_ps(input);

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx = _mm_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vx);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm_mul_ps(vt, vs);
    __m128 vf = _mm_add_ps(_mm_mul_ps(vt, vp), vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vx, vdenorm_cutoff), vf);

    if (batch & (2 * sizeof(float))) {
      // Store 2 outputs at a time.
      _mm_storel_pi((__m64*) output, vf);
      output += 2;

      // Accumulate 2 computed exponents.
      vacc = _mm_add_ps(vacc, _mm_movelh_ps(vf, _mm_setzero_ps()));

      vf = _mm_movehl_ps(vf, vf);
    }
    if (batch & (1 * sizeof(float))) {
      // Store 1 output at a time.
      _mm_store_ss(output, vf);

      // Accumulate 1 computed exponent.
      vacc = _mm_add_ss(vacc, vf);
    }
  }
  // Reduce 4 batch in the SIMD register
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_shuffle_ps(vacc, vacc, _MM_SHUFFLE(2, 3, 0, 1)));
  _mm_store_ss(sum, vacc);
}

extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u12(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsat_cutoff = _mm_set1_ps(-0x1.154246p+4f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.800000p19f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  const __m128i vindex_mask = _mm_set1_epi32(UINT32_C(0xF));
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  const __m128 vc3 = _mm_set1_ps(0x1.55561Cp-3f);
  const __m128 vc2 = _mm_set1_ps(0x1.0001ECp-1f);
  const __m128 vone = _mm_set1_ps(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);

  const __m128 vprescale = _mm_set1_ps(params->scalar.prescale);
  const __m128 valpha = _mm_set1_ps(params->scalar.alpha);
  const __m128 vbeta = _mm_set1_ps(params->scalar.beta);

  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vx4567 = _mm_loadu_ps(input + 4);
    __m128 vx89AB = _mm_loadu_ps(input + 8);
    input += 12;

    const __m128 vz0123 = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx0123, vprescale));
    const __m128 vz4567 = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx4567, vprescale));
    const __m128 vz89AB = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx89AB, vprescale));

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);
    __m128 vn89AB = _mm_add_ps(_mm_mul_ps(vz89AB, vlog2e), vmagic_bias);

    const __m128i vidx0123 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn0123), vindex_mask), 2);
    const __m128i ven0123 = _mm_slli_epi32(_mm_castps_si128(vn0123), 19);
    const __m128i vidx4567 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn4567), vindex_mask), 2);
    const __m128i ven4567 = _mm_slli_epi32(_mm_castps_si128(vn4567), 19);
    const __m128i vidx89AB = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn89AB), vindex_mask), 2);
    const __m128i ven89AB = _mm_slli_epi32(_mm_castps_si128(vn89AB), 19);

    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      const uint64_t vidx23 = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx0123, vidx0123));
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx01)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx23)));
      const __m128i vl1 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx01 >> 32))));
      const __m128i vl01 = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl3 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx23 >> 32))));
      const __m128i vl23 = _mm_unpacklo_epi32(vl2, vl3);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx4567);
      const uint64_t vidx67 = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx4567, vidx4567));
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx45)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx67)));
      const __m128i vl5 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx45 >> 32))));
      const __m128i vl45 = _mm_unpacklo_epi32(vl4, vl5);
      const __m128i vl7 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx67 >> 32))));
      const __m128i vl67 = _mm_unpacklo_epi32(vl6, vl7);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
      const uint64_t vidx89 = (uint64_t) _mm_cvtsi128_si64(vidx89AB);
      const uint64_t vidxAB = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx89AB, vidx89AB));
      const __m128i vl8   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx89)));
      const __m128i vlA = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxAB)));
      const __m128i vl9 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx89 >> 32))));
      const __m128i vl89 = _mm_unpacklo_epi32(vl8, vl9);
      const __m128i vlB = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxAB >> 32))));
      const __m128i vlAB = _mm_unpacklo_epi32(vlA, vlB);
      const __m128i vl89AB = _mm_unpacklo_epi64(vl89, vlAB);
    #else  // !XNN_ARCH_X86_64
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx0123);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx0123, 2);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx0123, 4);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx0123, 6);
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx0)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx2)));
      const __m128i vl1 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx1)));
      const __m128i vl01 = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl3 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx3)));
      const __m128i vl23 = _mm_unpacklo_epi32(vl2, vl3);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx4567);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi16(vidx4567, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi16(vidx4567, 4);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi16(vidx4567, 6);
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx4)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx6)));
      const __m128i vl5 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx5)));
      const __m128i vl45 = _mm_unpacklo_epi32(vl4, vl5);
      const __m128i vl7 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx7)));
      const __m128i vl67 = _mm_unpacklo_epi32(vl6, vl7);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
      const uint32_t vidx8 = (uint32_t) _mm_cvtsi128_si32(vidx89AB);
      const uint32_t vidx9 = (uint32_t) _mm_extract_epi16(vidx89AB, 2);
      const uint32_t vidxA = (uint32_t) _mm_extract_epi16(vidx89AB, 4);
      const uint32_t vidxB = (uint32_t) _mm_extract_epi16(vidx89AB, 6);
      const __m128i vl8   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx8)));
      const __m128i vlA = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidxA)));
      const __m128i vl9 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx9)));
      const __m128i vl89 = _mm_unpacklo_epi32(vl8, vl9);
      const __m128i vlB = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidxB)));
      const __m128i vlAB = _mm_unpacklo_epi32(vlA, vlB);
      const __m128i vl89AB = _mm_unpacklo_epi64(vl89, vlAB);
    #endif  // XNN_ARCH_X86_64

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    __m128 vs0123 = _mm_castsi128_ps(_mm_add_epi32(vl0123, ven0123));
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);
    __m128 vs4567 = _mm_castsi128_ps(_mm_add_epi32(vl4567, ven4567));
    vn89AB = _mm_sub_ps(vn89AB, vmagic_bias);
    __m128 vs89AB = _mm_castsi128_ps(_mm_add_epi32(vl89AB, ven89AB));

    __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_hi), vz0123);
    __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_hi), vz4567);
    __m128 vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_hi), vz89AB);

    vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_lo), vt4567);
    vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_lo), vt89AB);

    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc3, vt0123), vc2);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc3, vt4567), vc2);
    __m128 vp89AB = _mm_add_ps(_mm_mul_ps(vc3, vt89AB), vc2);

    vp0123 = _mm_mul_ps(vp0123, vt0123);
    vp4567 = _mm_mul_ps(vp4567, vt4567);
    vp89AB = _mm_mul_ps(vp89AB, vt89AB);

    vt0123 = _mm_mul_ps(vt0123, vs0123);
    vs0123 = _mm_sub_ps(vs0123, vone);
    vt4567 = _mm_mul_ps(vt4567, vs4567);
    vs4567 = _mm_sub_ps(vs4567, vone);
    vt89AB = _mm_mul_ps(vt89AB, vs89AB);
    vs89AB = _mm_sub_ps(vs89AB, vone);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vt0123);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vt4567);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vt89AB);

    const __m128 ve0123 = _mm_mul_ps(_mm_add_ps(vp0123, vs0123), valpha);
    const __m128 ve4567 = _mm_mul_ps(_mm_add_ps(vp4567, vs4567), valpha);
    const __m128 ve89AB = _mm_mul_ps(_mm_add_ps(vp89AB, vs89AB), valpha);

    const __m128 vm0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx0123)));
    vx0123 = _mm_mul_ps(vx0123, vbeta);
    const __m128 vm4567 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx4567)));
    vx4567 = _mm_mul_ps(vx4567, vbeta);
    const __m128 vm89AB = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx89AB)));
    vx89AB = _mm_mul_ps(vx89AB, vbeta);

    const __m128 vy0123 = _mm_or_ps(_mm_and_ps(ve0123, vm0123), _mm_andnot_ps(vm0123, vx0123));
    const __m128 vy4567 = _mm_or_ps(_mm_and_ps(ve4567, vm4567), _mm_andnot_ps(vm4567, vx4567));
    const __m128 vy89AB = _mm_or_ps(_mm_and_ps(ve89AB, vm89AB), _mm_andnot_ps(vm89AB, vx89AB));

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    output += 12;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vz = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx, vprescale));

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ven = _mm_slli_epi32(_mm_castps_si128(vn), 19);
    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx, vidx));
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi)));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 2))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 6))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);
    __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ven));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc3, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    vt = _mm_mul_ps(vt, vs);
    vs = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vt);
    const __m128 ve = _mm_mul_ps(_mm_add_ps(vp, vs), valpha);

    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vx = _mm_mul_ps(vx, vbeta);
    const __m128 vy = _mm_or_ps(_mm_and_ps(ve, vm), _mm_andnot_ps(vm, vx));

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx, vprescale));

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ven = _mm_slli_epi32(_mm_castps_si128(vn), 19);
    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx, vidx));
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi)));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 2))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 6))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);
    __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ven));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc3, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    vt = _mm_mul_ps(vt, vs);
    vs = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vt);
    const __m128 ve = _mm_mul_ps(_mm_add_ps(vp, vs), valpha);

    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vx = _mm_mul_ps(vx, vbeta);
    __m128 vy = _mm_or_ps(_mm_and_ps(ve, vm), _mm_andnot_ps(vm, vx));

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_f32_vrndd_ukernel__sse2_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmagic = _mm_castps_si128(_mm_set1_ps(-0.0f));
  const __m128 vone = _mm_set1_ps(1.0f);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128i vintx0123 = _mm_cvttps_epi32(vx0123);
    const __m128i vintx4567 = _mm_cvttps_epi32(vx4567);

    const __m128 vrndmask0123 = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx0123, vmagic)));
    const __m128 vrndmask4567 = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx4567, vmagic)));

    const __m128 vprerndx0123 = _mm_cvtepi32_ps(vintx0123);
    const __m128 vprerndx4567 = _mm_cvtepi32_ps(vintx4567);

    const __m128 vrndx0123 = _mm_or_ps(_mm_and_ps(vx0123, vrndmask0123), _mm_andnot_ps(vrndmask0123, vprerndx0123));
    const __m128 vrndx4567 = _mm_or_ps(_mm_and_ps(vx4567, vrndmask4567), _mm_andnot_ps(vrndmask4567, vprerndx4567));

    const __m128 vy0123 = _mm_sub_ps(vrndx0123, _mm_and_ps(_mm_cmpgt_ps(vrndx0123, vx0123), vone));
    const __m128 vy4567 = _mm_sub_ps(vrndx4567, _mm_and_ps(_mm_cmpgt_ps(vrndx4567, vx4567), vone));

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128i vintx = _mm_cvttps_epi32(vx);
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));
    const __m128 vprerndx = _mm_cvtepi32_ps(vintx);
    const __m128 vrndx = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vprerndx));
    const __m128 vy = _mm_sub_ps(vrndx, _mm_and_ps(_mm_cmpgt_ps(vrndx, vx), vone));

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    const __m128i vintx = _mm_cvttps_epi32(vx);
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));
    const __m128 vprerndx = _mm_cvtepi32_ps(vintx);
    const __m128 vrndx = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vprerndx));
    __m128 vy = _mm_sub_ps(vrndx, _mm_and_ps(_mm_cmpgt_ps(vrndx, vx), vone));
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_f32_vrndne_ukernel__sse2_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmagic = _mm_castps_si128(_mm_set1_ps(-0.0f));
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128i vintx0123 = _mm_cvtps_epi32(vx0123);
    const __m128i vintx4567 = _mm_cvtps_epi32(vx4567);

    const __m128 vrndmask0123 = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx0123, vmagic)));
    const __m128 vrndmask4567 = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx4567, vmagic)));

    const __m128 vrndx0123 = _mm_cvtepi32_ps(vintx0123);
    const __m128 vrndx4567 = _mm_cvtepi32_ps(vintx4567);

    const __m128 vy0123 = _mm_or_ps(_mm_and_ps(vx0123, vrndmask0123), _mm_andnot_ps(vrndmask0123, vrndx0123));
    const __m128 vy4567 = _mm_or_ps(_mm_and_ps(vx4567, vrndmask4567), _mm_andnot_ps(vrndmask4567, vrndx4567));

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128i vintx = _mm_cvtps_epi32(vx);
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));
    const __m128 vrndx = _mm_cvtepi32_ps(vintx);
    const __m128 vy = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vrndx));

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    const __m128i vintx = _mm_cvtps_epi32(vx);
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));
    const __m128 vrndx = _mm_cvtepi32_ps(vintx);
    __m128 vy = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vrndx));
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_f32_vrndu_ukernel__sse2_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmagic = _mm_castps_si128(_mm_set1_ps(-0.0f));
  const __m128 vone = _mm_set1_ps(1.0f);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128i vintx0123 = _mm_cvttps_epi32(vx0123);
    const __m128i vintx4567 = _mm_cvttps_epi32(vx4567);

    const __m128 vrndmask0123 = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx0123, vmagic)));
    const __m128 vrndmask4567 = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx4567, vmagic)));

    const __m128 vprerndx0123 = _mm_cvtepi32_ps(vintx0123);
    const __m128 vprerndx4567 = _mm_cvtepi32_ps(vintx4567);

    const __m128 vrndx0123 = _mm_or_ps(_mm_and_ps(vx0123, vrndmask0123), _mm_andnot_ps(vrndmask0123, vprerndx0123));
    const __m128 vrndx4567 = _mm_or_ps(_mm_and_ps(vx4567, vrndmask4567), _mm_andnot_ps(vrndmask4567, vprerndx4567));

    const __m128 vadjmask0123 = _mm_or_ps(_mm_cmpge_ps(vrndx0123, vx0123), _mm_castsi128_ps(vmagic));
    const __m128 vadjmask4567 = _mm_or_ps(_mm_cmpge_ps(vrndx4567, vx4567), _mm_castsi128_ps(vmagic));

    const __m128 vadjrndx0123 = _mm_add_ps(vrndx0123, vone);
    const __m128 vadjrndx4567 = _mm_add_ps(vrndx4567, vone);

    const __m128 vy0123 = _mm_or_ps(_mm_and_ps(vrndx0123, vadjmask0123), _mm_andnot_ps(vadjmask0123, vadjrndx0123));
    const __m128 vy4567 = _mm_or_ps(_mm_and_ps(vrndx4567, vadjmask4567), _mm_andnot_ps(vadjmask4567, vadjrndx4567));

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128i vintx = _mm_cvttps_epi32(vx);
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));
    const __m128 vprerndx = _mm_cvtepi32_ps(vintx);
    const __m128 vrndx = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vprerndx));
    const __m128 vadjmask = _mm_or_ps(_mm_cmpge_ps(vrndx, vx), _mm_castsi128_ps(vmagic));
    const __m128 vadjrndx = _mm_add_ps(vrndx, vone);
    const __m128 vy = _mm_or_ps(_mm_and_ps(vrndx, vadjmask), _mm_andnot_ps(vadjmask, vadjrndx));

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    const __m128i vintx = _mm_cvttps_epi32(vx);
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));
    const __m128 vprerndx = _mm_cvtepi32_ps(vintx);
    const __m128 vrndx = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vprerndx));
    const __m128 vadjmask = _mm_or_ps(_mm_cmpge_ps(vrndx, vx), _mm_castsi128_ps(vmagic));
    const __m128 vadjrndx = _mm_add_ps(vrndx, vone);
    __m128 vy = _mm_or_ps(_mm_and_ps(vrndx, vadjmask), _mm_andnot_ps(vadjmask, vadjrndx));
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_f32_vrndz_ukernel__sse2_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmagic = _mm_castps_si128(_mm_set1_ps(-0.0f));
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128i vintx0123 = _mm_cvttps_epi32(vx0123);
    const __m128i vintx4567 = _mm_cvttps_epi32(vx4567);

    const __m128 vrndmask0123 = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx0123, vmagic)));
    const __m128 vrndmask4567 = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx4567, vmagic)));

    const __m128 vrndx0123 = _mm_cvtepi32_ps(vintx0123);
    const __m128 vrndx4567 = _mm_cvtepi32_ps(vintx4567);

    const __m128 vy0123 = _mm_or_ps(_mm_and_ps(vx0123, vrndmask0123), _mm_andnot_ps(vrndmask0123, vrndx0123));
    const __m128 vy4567 = _mm_or_ps(_mm_and_ps(vx4567, vrndmask4567), _mm_andnot_ps(vrndmask4567, vrndx4567));

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128i vintx = _mm_cvttps_epi32(vx);
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));
    const __m128 vrndx = _mm_cvtepi32_ps(vintx);
    const __m128 vy = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vrndx));

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    const __m128i vintx = _mm_cvttps_epi32(vx);
    const __m128 vrndmask = _mm_castsi128_ps(_mm_or_si128(vmagic, _mm_cmpeq_epi32(vintx, vmagic)));
    const __m128 vrndx = _mm_cvtepi32_ps(vintx);
    __m128 vy = _mm_or_ps(_mm_and_ps(vx, vrndmask), _mm_andnot_ps(vrndmask, vrndx));
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}

extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64];

void xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsign_mask = _mm_set1_ps(-0.0f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.800000p17f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p0f);
  const __m128i vindex_mask = _mm_set1_epi32(UINT32_C(0x3F));
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.630000p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(0x1.BD0106p-13f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFF0Ap-2f);
  const __m128 vone = _mm_set1_ps(1.0f);
  const __m128 vdenorm_cutoff = _mm_set1_ps(-0x1.5D589Ep+6f);

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vz0123 = _mm_or_ps(vx0123, vsign_mask);
    const __m128 vz4567 = _mm_or_ps(vx4567, vsign_mask);

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);

    const __m128i ve0123 = _mm_slli_epi32(_mm_castps_si128(vn0123), 17);
    const __m128i ve4567 = _mm_slli_epi32(_mm_castps_si128(vn4567), 17);

    const __m128i vidx0123 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn0123), vindex_mask), 2);
    const __m128i vidx4567 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn4567), vindex_mask), 2);

    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      const uint64_t vidx23 = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx0123, vidx0123));
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx01)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx23)));
      const __m128i vl1 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx01 >> 32))));
      const __m128i vl01 = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl3 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx23 >> 32))));
      const __m128i vl23 = _mm_unpacklo_epi32(vl2, vl3);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx4567);
      const uint64_t vidx67 = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx4567, vidx4567));
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx45)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx67)));
      const __m128i vl5 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx45 >> 32))));
      const __m128i vl45 = _mm_unpacklo_epi32(vl4, vl5);
      const __m128i vl7 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx67 >> 32))));
      const __m128i vl67 = _mm_unpacklo_epi32(vl6, vl7);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
    #else  // !XNN_ARCH_X86_64
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx0123);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx0123, 2);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx0123, 4);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx0123, 6);
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx0)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx2)));
      const __m128i vl1 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx1)));
      const __m128i vl01 = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl3 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx3)));
      const __m128i vl23 = _mm_unpacklo_epi32(vl2, vl3);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx4567);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi16(vidx4567, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi16(vidx4567, 4);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi16(vidx4567, 6);
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx4)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx6)));
      const __m128i vl5 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx5)));
      const __m128i vl45 = _mm_unpacklo_epi32(vl4, vl5);
      const __m128i vl7 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx7)));
      const __m128i vl67 = _mm_unpacklo_epi32(vl6, vl7);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
    #endif  // XNN_ARCH_X86_64

    const __m128 vs0123 = _mm_castsi128_ps(_mm_add_epi32(vl0123, ve0123));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_add_epi32(vl4567, ve4567));

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);

    __m128 vt0123 = _mm_add_ps(vz0123, _mm_mul_ps(vn0123, vminus_ln2_hi));
    __m128 vt4567 = _mm_add_ps(vz4567, _mm_mul_ps(vn4567, vminus_ln2_hi));

    vt0123 = _mm_add_ps(vt0123, _mm_mul_ps(vn0123, vminus_ln2_lo));
    vt4567 = _mm_add_ps(vt4567, _mm_mul_ps(vn4567, vminus_ln2_lo));

    __m128 vp0123 = _mm_mul_ps(vt0123, vc2);
    __m128 vp4567 = _mm_mul_ps(vt4567, vc2);

    vp0123 = _mm_add_ps(vt0123, _mm_mul_ps(vp0123, vt0123));
    vp4567 = _mm_add_ps(vt4567, _mm_mul_ps(vp4567, vt4567));

    const __m128 vy0123 = _mm_add_ps(vs0123, _mm_mul_ps(vs0123, vp0123));
    const __m128 vy4567 = _mm_add_ps(vs4567, _mm_mul_ps(vs4567, vp4567));

    __m128 vf0123 = _mm_div_ps(vy0123, _mm_add_ps(vy0123, vone));
    __m128 vf4567 = _mm_div_ps(vy4567, _mm_add_ps(vy4567, vone));

    vf0123 = _mm_andnot_ps(_mm_cmplt_ps(vz0123, vdenorm_cutoff), vf0123);
    vf4567 = _mm_andnot_ps(_mm_cmplt_ps(vz4567, vdenorm_cutoff), vf4567);

    const __m128 vm0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx0123)));
    const __m128 vm4567 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx4567)));

    vf0123 = _mm_or_ps(_mm_and_ps(vf0123, vm0123), _mm_andnot_ps(vm0123, _mm_sub_ps(vone, vf0123)));
    vf4567 = _mm_or_ps(_mm_and_ps(vf4567, vm4567), _mm_andnot_ps(vm4567, _mm_sub_ps(vone, vf4567)));

    _mm_storeu_ps(output, vf0123);
    _mm_storeu_ps(output + 4, vf4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vz = _mm_or_ps(vx, vsign_mask);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);
    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 17);

    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx, vidx));
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_hi)));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_lo >> 32))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_hi >> 32))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 2))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 6))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);

    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(vz, _mm_mul_ps(vn, vminus_ln2_hi));
    vt = _mm_add_ps(vt, _mm_mul_ps(vn, vminus_ln2_lo));

    __m128 vp = _mm_mul_ps(vt, vc2);
    vp = _mm_add_ps(vt, _mm_mul_ps(vp, vt));

    const __m128 vy = _mm_add_ps(vs, _mm_mul_ps(vs, vp));

    __m128 vf = _mm_div_ps(vy, _mm_add_ps(vy, vone));
    vf = _mm_andnot_ps(_mm_cmplt_ps(vz, vdenorm_cutoff), vf);
    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vf = _mm_or_ps(_mm_and_ps(vf, vm), _mm_andnot_ps(vm, _mm_sub_ps(vone, vf)));

    _mm_storeu_ps(output, vf);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_or_ps(vx, vsign_mask);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);
    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 17);

    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx, vidx));
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_hi)));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_lo >> 32))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_hi >> 32))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 2))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 6))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);

    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(vz, _mm_mul_ps(vn, vminus_ln2_hi));
    vt = _mm_add_ps(vt, _mm_mul_ps(vn, vminus_ln2_lo));

    __m128 vp = _mm_mul_ps(vt, vc2);
    vp = _mm_add_ps(vt, _mm_mul_ps(vp, vt));

    const __m128 vy = _mm_add_ps(vs, _mm_mul_ps(vs, vp));

    __m128 vf = _mm_div_ps(vy, _mm_add_ps(vy, vone));
    vf = _mm_andnot_ps(_mm_cmplt_ps(vz, vdenorm_cutoff), vf);
    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vf = _mm_or_ps(_mm_and_ps(vf, vm), _mm_andnot_ps(vm, _mm_sub_ps(vone, vf)));

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vf);
      vf = _mm_movehl_ps(vf, vf);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vf);
    }
  }
}

void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->sse.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);
  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128 vksum = _mm_load_ps((const float*) w);
    __m128i vinput_zero_point0 = _mm_cvtsi32_si128(*((const int*) &quantization_params[0].zero_point));
    vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point0, _MM_SHUFFLE(0, 0, 0, 0));

    __m128 vinput_zero_point0_float = _mm_cvtepi32_ps(vinput_zero_point0);
    __m128 vout0x0123 = _mm_mul_ps(vksum, vinput_zero_point0_float);
    w = (const int32_t*) w + 4;

    for (size_t kb=0; kb < kc; kb += bl) {
      __m128i vacc0x0 = _mm_setzero_si128();
      __m128i vacc0x1 = _mm_setzero_si128();
      __m128i vacc0x2 = _mm_setzero_si128();
      __m128i vacc0x3 = _mm_setzero_si128();
      size_t k = bl;

      while (k >= 16 * sizeof(int8_t)) {
        const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0c0, va0c0), 8);
        a0 += 8;

        const __m128i vb01c01 = _mm_loadu_si128((const __m128i*) w);
        const __m128i vbs01c0 = _mm_slli_epi32(vb01c01, 4);
        const __m128i vb01c0 = _mm_and_si128(vbs01c0, vmask);
        const __m128i vsb01c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c0);
        const __m128i vxb0c0 = _mm_unpacklo_epi8(vb01c0, vsb01c0);
        const __m128i vxb1c0 = _mm_unpackhi_epi8(vb01c0, vsb01c0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
        const __m128i vb23c01 = _mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vbs23c0 = _mm_slli_epi32(vb23c01, 4);
        const __m128i vb23c0 = _mm_and_si128(vbs23c0, vmask);
        const __m128i vsb23c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c0);
        const __m128i vxb2c0 = _mm_unpacklo_epi8(vb23c0, vsb23c0);
        const __m128i vxb3c0 = _mm_unpackhi_epi8(vb23c0, vsb23c0);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));

        const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va0c1, va0c1), 8);
        a0 += 8;

        const __m128i vb01c1 = _mm_and_si128(vb01c01, vmask);
        const __m128i vsb01c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c1);
        const __m128i vxb0c1 = _mm_unpacklo_epi8(vb01c1, vsb01c1);
        const __m128i vxb1c1 = _mm_unpackhi_epi8(vb01c1, vsb01c1);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
        const __m128i vb23c1 = _mm_and_si128(vb23c01, vmask);
        const __m128i vsb23c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c1);
        const __m128i vxb2c1 = _mm_unpacklo_epi8(vb23c1, vsb23c1);
        const __m128i vxb3c1 = _mm_unpackhi_epi8(vb23c1, vsb23c1);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));


        w = (const int8_t*) w + 32;
        k -= 16 * sizeof(int8_t);
      }

      while (k >= 8 * sizeof(int8_t)) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;

        __m128i vb01 = _mm_loadu_si128((const __m128i*) w);
        vb01 = _mm_slli_epi32(vb01, 4);
        vb01 = _mm_and_si128(vb01, vmask);

        const __m128i vsb01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01);
        const __m128i vxb0 = _mm_unpacklo_epi8(vb01, vsb01);
        const __m128i vxb1 = _mm_unpackhi_epi8(vb01, vsb01);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        __m128i vb23 = _mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16));
        vb23 = _mm_slli_epi32(vb23, 4);
        vb23 = _mm_and_si128(vb23, vmask);

        const __m128i vsb23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23);
        const __m128i vxb2 = _mm_unpacklo_epi8(vb23, vsb23);
        const __m128i vxb3 = _mm_unpackhi_epi8(vb23, vsb23);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const int8_t*) w + 32;
        k -= 8 * sizeof(int8_t);
      }
      // accumulate float
      const __m128 vfilter_output_scale0123 = _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), _mm_loadl_epi64((const __m128i*) w)));
      w = (const uint16_t*) w + 4;

      const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
      const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

      __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

      vout0x0123 = _mm_add_ps(vout0x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc0x0123), vfilter_output_scale0123));
    }

    const __m128 vinput_scale0 = _mm_load1_ps(&quantization_params[0].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);


    const __m128 vbias0123 = _mm_loadu_ps((const float*) w);
    w = (const float*) w + 4;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->sse.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);
  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128 vksum = _mm_load_ps((const float*) w);
    const __m128i vinput_zero_point01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128i vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point1 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128i vinput_zero_point23 = _mm_loadu_si128((const __m128i*) &quantization_params[2]);
    const __m128i vinput_zero_point2 = _mm_shuffle_epi32(vinput_zero_point23, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point3 = _mm_shuffle_epi32(vinput_zero_point23, _MM_SHUFFLE(2, 2, 2, 2));

    __m128 vinput_zero_point0_float = _mm_cvtepi32_ps(vinput_zero_point0);
    __m128 vout0x0123 = _mm_mul_ps(vksum, vinput_zero_point0_float);
    __m128 vinput_zero_point1_float = _mm_cvtepi32_ps(vinput_zero_point1);
    __m128 vout1x0123 = _mm_mul_ps(vksum, vinput_zero_point1_float);
    __m128 vinput_zero_point2_float = _mm_cvtepi32_ps(vinput_zero_point2);
    __m128 vout2x0123 = _mm_mul_ps(vksum, vinput_zero_point2_float);
    __m128 vinput_zero_point3_float = _mm_cvtepi32_ps(vinput_zero_point3);
    __m128 vout3x0123 = _mm_mul_ps(vksum, vinput_zero_point3_float);
    w = (const int32_t*) w + 4;

    for (size_t kb=0; kb < kc; kb += bl) {
      __m128i vacc0x0 = _mm_setzero_si128();
      __m128i vacc0x1 = _mm_setzero_si128();
      __m128i vacc0x2 = _mm_setzero_si128();
      __m128i vacc0x3 = _mm_setzero_si128();
      __m128i vacc1x0 = _mm_setzero_si128();
      __m128i vacc1x1 = _mm_setzero_si128();
      __m128i vacc1x2 = _mm_setzero_si128();
      __m128i vacc1x3 = _mm_setzero_si128();
      __m128i vacc2x0 = _mm_setzero_si128();
      __m128i vacc2x1 = _mm_setzero_si128();
      __m128i vacc2x2 = _mm_setzero_si128();
      __m128i vacc2x3 = _mm_setzero_si128();
      __m128i vacc3x0 = _mm_setzero_si128();
      __m128i vacc3x1 = _mm_setzero_si128();
      __m128i vacc3x2 = _mm_setzero_si128();
      __m128i vacc3x3 = _mm_setzero_si128();
      size_t k = bl;

      while (k >= 16 * sizeof(int8_t)) {
        const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0c0, va0c0), 8);
        a0 += 8;
        const __m128i va1c0 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va1c0, va1c0), 8);
        a1 += 8;
        const __m128i va2c0 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va2c0, va2c0), 8);
        a2 += 8;
        const __m128i va3c0 = _mm_loadl_epi64((const __m128i*) a3);
        const __m128i vxa3c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va3c0, va3c0), 8);
        a3 += 8;

        const __m128i vb01c01 = _mm_loadu_si128((const __m128i*) w);
        const __m128i vbs01c0 = _mm_slli_epi32(vb01c01, 4);
        const __m128i vb01c0 = _mm_and_si128(vbs01c0, vmask);
        const __m128i vsb01c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c0);
        const __m128i vxb0c0 = _mm_unpacklo_epi8(vb01c0, vsb01c0);
        const __m128i vxb1c0 = _mm_unpackhi_epi8(vb01c0, vsb01c0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c0, vxb0c0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c0, vxb1c0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c0, vxb0c0));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c0, vxb1c0));
        vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3c0, vxb0c0));
        vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3c0, vxb1c0));
        const __m128i vb23c01 = _mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vbs23c0 = _mm_slli_epi32(vb23c01, 4);
        const __m128i vb23c0 = _mm_and_si128(vbs23c0, vmask);
        const __m128i vsb23c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c0);
        const __m128i vxb2c0 = _mm_unpacklo_epi8(vb23c0, vsb23c0);
        const __m128i vxb3c0 = _mm_unpackhi_epi8(vb23c0, vsb23c0);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c0, vxb2c0));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c0, vxb3c0));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c0, vxb2c0));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c0, vxb3c0));
        vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3c0, vxb2c0));
        vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3c0, vxb3c0));

        const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va0c1, va0c1), 8);
        a0 += 8;
        const __m128i va1c1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1c1, va1c1), 8);
        a1 += 8;
        const __m128i va2c1 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va2c1, va2c1), 8);
        a2 += 8;
        const __m128i va3c1 = _mm_loadl_epi64((const __m128i*) a3);
        const __m128i vxa3c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va3c1, va3c1), 8);
        a3 += 8;

        const __m128i vb01c1 = _mm_and_si128(vb01c01, vmask);
        const __m128i vsb01c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c1);
        const __m128i vxb0c1 = _mm_unpacklo_epi8(vb01c1, vsb01c1);
        const __m128i vxb1c1 = _mm_unpackhi_epi8(vb01c1, vsb01c1);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c1, vxb0c1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c1, vxb1c1));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c1, vxb0c1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c1, vxb1c1));
        vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3c1, vxb0c1));
        vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3c1, vxb1c1));
        const __m128i vb23c1 = _mm_and_si128(vb23c01, vmask);
        const __m128i vsb23c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c1);
        const __m128i vxb2c1 = _mm_unpacklo_epi8(vb23c1, vsb23c1);
        const __m128i vxb3c1 = _mm_unpackhi_epi8(vb23c1, vsb23c1);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c1, vxb2c1));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c1, vxb3c1));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c1, vxb2c1));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c1, vxb3c1));
        vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3c1, vxb2c1));
        vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3c1, vxb3c1));


        w = (const int8_t*) w + 32;
        k -= 16 * sizeof(int8_t);
      }

      while (k >= 8 * sizeof(int8_t)) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_srai_epi16(_mm_unpacklo_epi8(va2, va2), 8);
        a2 += 8;
        const __m128i va3 = _mm_loadl_epi64((const __m128i*) a3);
        const __m128i vxa3 = _mm_srai_epi16(_mm_unpacklo_epi8(va3, va3), 8);
        a3 += 8;

        __m128i vb01 = _mm_loadu_si128((const __m128i*) w);
        vb01 = _mm_slli_epi32(vb01, 4);
        vb01 = _mm_and_si128(vb01, vmask);

        const __m128i vsb01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01);
        const __m128i vxb0 = _mm_unpacklo_epi8(vb01, vsb01);
        const __m128i vxb1 = _mm_unpackhi_epi8(vb01, vsb01);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3, vxb0));
        vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3, vxb1));
        __m128i vb23 = _mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16));
        vb23 = _mm_slli_epi32(vb23, 4);
        vb23 = _mm_and_si128(vb23, vmask);

        const __m128i vsb23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23);
        const __m128i vxb2 = _mm_unpacklo_epi8(vb23, vsb23);
        const __m128i vxb3 = _mm_unpackhi_epi8(vb23, vsb23);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));
        vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3, vxb2));
        vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3, vxb3));

        w = (const int8_t*) w + 32;
        k -= 8 * sizeof(int8_t);
      }
      // accumulate float
      const __m128 vfilter_output_scale0123 = _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), _mm_loadl_epi64((const __m128i*) w)));
      w = (const uint16_t*) w + 4;

      const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
      const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
      const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
      const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
      const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
      const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));
      const __m128i vacc3x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x0, vacc3x2), _mm_unpackhi_epi32(vacc3x0, vacc3x2));
      const __m128i vacc3x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x1, vacc3x3), _mm_unpackhi_epi32(vacc3x1, vacc3x3));

      __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
      __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
      __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));
      __m128i vacc3x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x02, vacc3x13), _mm_unpackhi_epi32(vacc3x02, vacc3x13));

      vout0x0123 = _mm_add_ps(vout0x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc0x0123), vfilter_output_scale0123));
      vout1x0123 = _mm_add_ps(vout1x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc1x0123), vfilter_output_scale0123));
      vout2x0123 = _mm_add_ps(vout2x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc2x0123), vfilter_output_scale0123));
      vout3x0123 = _mm_add_ps(vout3x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc3x0123), vfilter_output_scale0123));
    }

    const __m128i vinput_scale01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128 vinput_scale0 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale1 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(3, 3, 3, 3)));
    const __m128i vinput_scale23 = _mm_loadu_si128((const __m128i*) &quantization_params[2]);
    const __m128 vinput_scale2 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale23, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale3 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale23, _MM_SHUFFLE(3, 3, 3, 3)));

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale1);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale2);
    vout3x0123 = _mm_mul_ps(vout3x0123, vinput_scale3);


    const __m128 vbias0123 = _mm_loadu_ps((const float*) w);
    w = (const float*) w + 4;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);
    vout3x0123 = _mm_add_ps(vout3x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);
    vout3x0123 = _mm_max_ps(vout3x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);
    vout3x0123 = _mm_min_ps(vout3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c2, vout2x0123);
      _mm_storeu_ps(c3, vout3x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
        _mm_storel_pi((__m64*) c3, vout3x0123);
        vout3x0123 = _mm_unpackhi_ps(vout3x0123, vout3x0123);
        c3 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
        _mm_store_ss(c3, vout3x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    __m128i vinput_zero_point0 = _mm_cvtsi32_si128(*((const int*) &quantization_params[0].zero_point));
    vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point0, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vksum_lo = _mm_srli_epi32(_mm_slli_epi32(vksum, 16), 16);
    const __m128i vksum_hi = _mm_srli_epi32(vksum, 16);

    __m128i vzpprodksumhi0 = _mm_mulhi_epu16(vinput_zero_point0, vksum_lo);
    const __m128i vzpprodksumlo0 = _mm_mullo_epi16(vinput_zero_point0, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point0, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point0, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    w = (const int32_t*) w + 4;

    size_t k = kc;

    while (k >= 16 * sizeof(int8_t)) {
      const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0c0, va0c0), 8);
      a0 += 8;

      const __m128i vb01c01 = _mm_load_si128((const __m128i*) w);
      const __m128i vbs01c0 = _mm_slli_epi32(vb01c01, 4);
      const __m128i vb01c0 = _mm_and_si128(vbs01c0, vmask);
      const __m128i vsb01c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c0);
      const __m128i vxb0c0 = _mm_unpacklo_epi8(vb01c0, vsb01c0);
      const __m128i vxb1c0 = _mm_unpackhi_epi8(vb01c0, vsb01c0);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
      const __m128i vb23c01 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbs23c0 = _mm_slli_epi32(vb23c01, 4);
      const __m128i vb23c0 = _mm_and_si128(vbs23c0, vmask);
      const __m128i vsb23c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c0);
      const __m128i vxb2c0 = _mm_unpacklo_epi8(vb23c0, vsb23c0);
      const __m128i vxb3c0 = _mm_unpackhi_epi8(vb23c0, vsb23c0);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));

      const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va0c1, va0c1), 8);
      a0 += 8;

      const __m128i vb01c1 = _mm_and_si128(vb01c01, vmask);
      const __m128i vsb01c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c1);
      const __m128i vxb0c1 = _mm_unpacklo_epi8(vb01c1, vsb01c1);
      const __m128i vxb1c1 = _mm_unpackhi_epi8(vb01c1, vsb01c1);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
      const __m128i vb23c1 = _mm_and_si128(vb23c01, vmask);
      const __m128i vsb23c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c1);
      const __m128i vxb2c1 = _mm_unpacklo_epi8(vb23c1, vsb23c1);
      const __m128i vxb3c1 = _mm_unpackhi_epi8(vb23c1, vsb23c1);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));


      w = (const int8_t*) w + 32;
      k -= 16 * sizeof(int8_t);
    }

    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
      a0 += 8;

      __m128i vb01 = _mm_load_si128((const __m128i*) w);
      vb01 = _mm_slli_epi32(vb01, 4);
      vb01 = _mm_and_si128(vb01, vmask);

      const __m128i vsb01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01);
      const __m128i vxb0 = _mm_unpacklo_epi8(vb01, vsb01);
      const __m128i vxb1 = _mm_unpackhi_epi8(vb01, vsb01);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      vb23 = _mm_slli_epi32(vb23, 4);
      vb23 = _mm_and_si128(vb23, vmask);

      const __m128i vsb23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23);
      const __m128i vxb2 = _mm_unpacklo_epi8(vb23, vsb23);
      const __m128i vxb3 = _mm_unpackhi_epi8(vb23, vsb23);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

    vacc0x0123 = _mm_srai_epi32(vacc0x0123, 4);
    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vinput_scale0 = _mm_load1_ps(&quantization_params[0].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vinput_zero_point01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128i vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point1 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128i vinput_zero_point23 = _mm_loadu_si128((const __m128i*) &quantization_params[2]);
    const __m128i vinput_zero_point2 = _mm_shuffle_epi32(vinput_zero_point23, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point3 = _mm_shuffle_epi32(vinput_zero_point23, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vksum_lo = _mm_srli_epi32(_mm_slli_epi32(vksum, 16), 16);
    const __m128i vksum_hi = _mm_srli_epi32(vksum, 16);

    __m128i vzpprodksumhi0 = _mm_mulhi_epu16(vinput_zero_point0, vksum_lo);
    const __m128i vzpprodksumlo0 = _mm_mullo_epi16(vinput_zero_point0, vksum_lo);
    __m128i vzpprodksumhi1 = _mm_mulhi_epu16(vinput_zero_point1, vksum_lo);
    const __m128i vzpprodksumlo1 = _mm_mullo_epi16(vinput_zero_point1, vksum_lo);
    __m128i vzpprodksumhi2 = _mm_mulhi_epu16(vinput_zero_point2, vksum_lo);
    const __m128i vzpprodksumlo2 = _mm_mullo_epi16(vinput_zero_point2, vksum_lo);
    __m128i vzpprodksumhi3 = _mm_mulhi_epu16(vinput_zero_point3, vksum_lo);
    const __m128i vzpprodksumlo3 = _mm_mullo_epi16(vinput_zero_point3, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point0, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point0, 15), vksum_lo));
    vzpprodksumhi1 = _mm_add_epi16(vzpprodksumhi1, _mm_mullo_epi16(vinput_zero_point1, vksum_hi));
    vzpprodksumhi1 = _mm_sub_epi16(vzpprodksumhi1, _mm_and_si128(_mm_srai_epi16(vinput_zero_point1, 15), vksum_lo));
    vzpprodksumhi2 = _mm_add_epi16(vzpprodksumhi2, _mm_mullo_epi16(vinput_zero_point2, vksum_hi));
    vzpprodksumhi2 = _mm_sub_epi16(vzpprodksumhi2, _mm_and_si128(_mm_srai_epi16(vinput_zero_point2, 15), vksum_lo));
    vzpprodksumhi3 = _mm_add_epi16(vzpprodksumhi3, _mm_mullo_epi16(vinput_zero_point3, vksum_hi));
    vzpprodksumhi3 = _mm_sub_epi16(vzpprodksumhi3, _mm_and_si128(_mm_srai_epi16(vinput_zero_point3, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);
    vzpprodksumhi1 = _mm_slli_si128(vzpprodksumhi1, 2);
    vzpprodksumhi2 = _mm_slli_si128(vzpprodksumhi2, 2);
    vzpprodksumhi3 = _mm_slli_si128(vzpprodksumhi3, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);
    const __m128i vksumzp1 = _mm_or_si128(vzpprodksumhi1, vzpprodksumlo1);
    const __m128i vksumzp2 = _mm_or_si128(vzpprodksumhi2, vzpprodksumlo2);
    const __m128i vksumzp3 = _mm_or_si128(vzpprodksumhi3, vzpprodksumlo3);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);
    const __m128i vksum011 = _mm_unpacklo_epi32(vksumzp1, vzero);
    const __m128i vksum231 = _mm_unpackhi_epi32(vksumzp1, vzero);
    const __m128i vksum012 = _mm_unpacklo_epi32(vksumzp2, vzero);
    const __m128i vksum232 = _mm_unpackhi_epi32(vksumzp2, vzero);
    const __m128i vksum013 = _mm_unpacklo_epi32(vksumzp3, vzero);
    const __m128i vksum233 = _mm_unpackhi_epi32(vksumzp3, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);
    __m128i vacc1x0 = _mm_unpacklo_epi64(vksum011, vzero);
    __m128i vacc1x1 = _mm_unpackhi_epi64(vksum011, vzero);
    __m128i vacc2x0 = _mm_unpacklo_epi64(vksum012, vzero);
    __m128i vacc2x1 = _mm_unpackhi_epi64(vksum012, vzero);
    __m128i vacc3x0 = _mm_unpacklo_epi64(vksum013, vzero);
    __m128i vacc3x1 = _mm_unpackhi_epi64(vksum013, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    __m128i vacc1x2 = _mm_unpacklo_epi64(vksum231, vzero);
    __m128i vacc1x3 = _mm_unpackhi_epi64(vksum231, vzero);
    __m128i vacc2x2 = _mm_unpacklo_epi64(vksum232, vzero);
    __m128i vacc2x3 = _mm_unpackhi_epi64(vksum232, vzero);
    __m128i vacc3x2 = _mm_unpacklo_epi64(vksum233, vzero);
    __m128i vacc3x3 = _mm_unpackhi_epi64(vksum233, vzero);
    w = (const int32_t*) w + 4;

    size_t k = kc;

    while (k >= 16 * sizeof(int8_t)) {
      const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0c0, va0c0), 8);
      a0 += 8;
      const __m128i va1c0 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va1c0, va1c0), 8);
      a1 += 8;
      const __m128i va2c0 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va2c0, va2c0), 8);
      a2 += 8;
      const __m128i va3c0 = _mm_loadl_epi64((const __m128i*) a3);
      const __m128i vxa3c0 = _mm_srai_epi16(_mm_unpacklo_epi8(va3c0, va3c0), 8);
      a3 += 8;

      const __m128i vb01c01 = _mm_load_si128((const __m128i*) w);
      const __m128i vbs01c0 = _mm_slli_epi32(vb01c01, 4);
      const __m128i vb01c0 = _mm_and_si128(vbs01c0, vmask);
      const __m128i vsb01c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c0);
      const __m128i vxb0c0 = _mm_unpacklo_epi8(vb01c0, vsb01c0);
      const __m128i vxb1c0 = _mm_unpackhi_epi8(vb01c0, vsb01c0);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c0, vxb0c0));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c0, vxb1c0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c0, vxb0c0));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c0, vxb1c0));
      vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3c0, vxb0c0));
      vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3c0, vxb1c0));
      const __m128i vb23c01 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbs23c0 = _mm_slli_epi32(vb23c01, 4);
      const __m128i vb23c0 = _mm_and_si128(vbs23c0, vmask);
      const __m128i vsb23c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c0);
      const __m128i vxb2c0 = _mm_unpacklo_epi8(vb23c0, vsb23c0);
      const __m128i vxb3c0 = _mm_unpackhi_epi8(vb23c0, vsb23c0);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c0, vxb2c0));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c0, vxb3c0));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c0, vxb2c0));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c0, vxb3c0));
      vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3c0, vxb2c0));
      vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3c0, vxb3c0));

      const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va0c1, va0c1), 8);
      a0 += 8;
      const __m128i va1c1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1c1, va1c1), 8);
      a1 += 8;
      const __m128i va2c1 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va2c1, va2c1), 8);
      a2 += 8;
      const __m128i va3c1 = _mm_loadl_epi64((const __m128i*) a3);
      const __m128i vxa3c1 = _mm_srai_epi16(_mm_unpacklo_epi8(va3c1, va3c1), 8);
      a3 += 8;

      const __m128i vb01c1 = _mm_and_si128(vb01c01, vmask);
      const __m128i vsb01c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c1);
      const __m128i vxb0c1 = _mm_unpacklo_epi8(vb01c1, vsb01c1);
      const __m128i vxb1c1 = _mm_unpackhi_epi8(vb01c1, vsb01c1);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c1, vxb0c1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c1, vxb1c1));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c1, vxb0c1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c1, vxb1c1));
      vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3c1, vxb0c1));
      vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3c1, vxb1c1));
      const __m128i vb23c1 = _mm_and_si128(vb23c01, vmask);
      const __m128i vsb23c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c1);
      const __m128i vxb2c1 = _mm_unpacklo_epi8(vb23c1, vsb23c1);
      const __m128i vxb3c1 = _mm_unpackhi_epi8(vb23c1, vsb23c1);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c1, vxb2c1));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c1, vxb3c1));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c1, vxb2c1));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c1, vxb3c1));
      vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3c1, vxb2c1));
      vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3c1, vxb3c1));


      w = (const int8_t*) w + 32;
      k -= 16 * sizeof(int8_t);
    }

    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_srai_epi16(_mm_unpacklo_epi8(va2, va2), 8);
      a2 += 8;
      const __m128i va3 = _mm_loadl_epi64((const __m128i*) a3);
      const __m128i vxa3 = _mm_srai_epi16(_mm_unpacklo_epi8(va3, va3), 8);
      a3 += 8;

      __m128i vb01 = _mm_load_si128((const __m128i*) w);
      vb01 = _mm_slli_epi32(vb01, 4);
      vb01 = _mm_and_si128(vb01, vmask);

      const __m128i vsb01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01);
      const __m128i vxb0 = _mm_unpacklo_epi8(vb01, vsb01);
      const __m128i vxb1 = _mm_unpackhi_epi8(vb01, vsb01);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3, vxb0));
      vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3, vxb1));
      __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      vb23 = _mm_slli_epi32(vb23, 4);
      vb23 = _mm_and_si128(vb23, vmask);

      const __m128i vsb23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23);
      const __m128i vxb2 = _mm_unpacklo_epi8(vb23, vsb23);
      const __m128i vxb3 = _mm_unpackhi_epi8(vb23, vsb23);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));
      vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3, vxb2));
      vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));
    const __m128i vacc3x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x0, vacc3x2), _mm_unpackhi_epi32(vacc3x0, vacc3x2));
    const __m128i vacc3x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x1, vacc3x3), _mm_unpackhi_epi32(vacc3x1, vacc3x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));
    __m128i vacc3x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc3x02, vacc3x13), _mm_unpackhi_epi32(vacc3x02, vacc3x13));

    vacc0x0123 = _mm_srai_epi32(vacc0x0123, 4);
    vacc1x0123 = _mm_srai_epi32(vacc1x0123, 4);
    vacc2x0123 = _mm_srai_epi32(vacc2x0123, 4);
    vacc3x0123 = _mm_srai_epi32(vacc3x0123, 4);
    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);
    __m128 vout3x0123 = _mm_cvtepi32_ps(vacc3x0123);

    const __m128i vinput_scale01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128 vinput_scale0 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale1 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(3, 3, 3, 3)));
    const __m128i vinput_scale23 = _mm_loadu_si128((const __m128i*) &quantization_params[2]);
    const __m128 vinput_scale2 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale23, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale3 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale23, _MM_SHUFFLE(3, 3, 3, 3)));

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale1);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale2);
    vout3x0123 = _mm_mul_ps(vout3x0123, vinput_scale3);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);
    vout3x0123 = _mm_mul_ps(vout3x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);
    vout3x0123 = _mm_add_ps(vout3x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);
    vout3x0123 = _mm_max_ps(vout3x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);
    vout3x0123 = _mm_min_ps(vout3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c2, vout2x0123);
      _mm_storeu_ps(c3, vout3x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
        _mm_storel_pi((__m64*) c3, vout3x0123);
        vout3x0123 = _mm_unpackhi_ps(vout3x0123, vout3x0123);
        c3 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
        _mm_store_ss(c3, vout3x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    __m128i vinput_zero_point0 = _mm_cvtsi32_si128(*((const int*) &quantization_params[0].zero_point));
    vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point0, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vksum_lo = _mm_srli_epi32(_mm_slli_epi32(vksum, 16), 16);
    const __m128i vksum_hi = _mm_srli_epi32(vksum, 16);

    __m128i vzpprodksumhi0 = _mm_mulhi_epu16(vinput_zero_point0, vksum_lo);
    const __m128i vzpprodksumlo0 = _mm_mullo_epi16(vinput_zero_point0, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point0, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point0, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vinput_scale0 = _mm_load1_ps(&quantization_params[0].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vinput_zero_point01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128i vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point1 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(2, 2, 2, 2));
    __m128i vinput_zero_point2 = _mm_cvtsi32_si128(*((const int*) &quantization_params[2].zero_point));
    vinput_zero_point2 = _mm_shuffle_epi32(vinput_zero_point2, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vksum_lo = _mm_srli_epi32(_mm_slli_epi32(vksum, 16), 16);
    const __m128i vksum_hi = _mm_srli_epi32(vksum, 16);

    __m128i vzpprodksumhi0 = _mm_mulhi_epu16(vinput_zero_point0, vksum_lo);
    const __m128i vzpprodksumlo0 = _mm_mullo_epi16(vinput_zero_point0, vksum_lo);
    __m128i vzpprodksumhi1 = _mm_mulhi_epu16(vinput_zero_point1, vksum_lo);
    const __m128i vzpprodksumlo1 = _mm_mullo_epi16(vinput_zero_point1, vksum_lo);
    __m128i vzpprodksumhi2 = _mm_mulhi_epu16(vinput_zero_point2, vksum_lo);
    const __m128i vzpprodksumlo2 = _mm_mullo_epi16(vinput_zero_point2, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point0, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point0, 15), vksum_lo));
    vzpprodksumhi1 = _mm_add_epi16(vzpprodksumhi1, _mm_mullo_epi16(vinput_zero_point1, vksum_hi));
    vzpprodksumhi1 = _mm_sub_epi16(vzpprodksumhi1, _mm_and_si128(_mm_srai_epi16(vinput_zero_point1, 15), vksum_lo));
    vzpprodksumhi2 = _mm_add_epi16(vzpprodksumhi2, _mm_mullo_epi16(vinput_zero_point2, vksum_hi));
    vzpprodksumhi2 = _mm_sub_epi16(vzpprodksumhi2, _mm_and_si128(_mm_srai_epi16(vinput_zero_point2, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);
    vzpprodksumhi1 = _mm_slli_si128(vzpprodksumhi1, 2);
    vzpprodksumhi2 = _mm_slli_si128(vzpprodksumhi2, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);
    const __m128i vksumzp1 = _mm_or_si128(vzpprodksumhi1, vzpprodksumlo1);
    const __m128i vksumzp2 = _mm_or_si128(vzpprodksumhi2, vzpprodksumlo2);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);
    const __m128i vksum011 = _mm_unpacklo_epi32(vksumzp1, vzero);
    const __m128i vksum231 = _mm_unpackhi_epi32(vksumzp1, vzero);
    const __m128i vksum012 = _mm_unpacklo_epi32(vksumzp2, vzero);
    const __m128i vksum232 = _mm_unpackhi_epi32(vksumzp2, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);
    __m128i vacc1x0 = _mm_unpacklo_epi64(vksum011, vzero);
    __m128i vacc1x1 = _mm_unpackhi_epi64(vksum011, vzero);
    __m128i vacc2x0 = _mm_unpacklo_epi64(vksum012, vzero);
    __m128i vacc2x1 = _mm_unpackhi_epi64(vksum012, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    __m128i vacc1x2 = _mm_unpacklo_epi64(vksum231, vzero);
    __m128i vacc1x3 = _mm_unpackhi_epi64(vksum231, vzero);
    __m128i vacc2x2 = _mm_unpacklo_epi64(vksum232, vzero);
    __m128i vacc2x3 = _mm_unpackhi_epi64(vksum232, vzero);
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_srai_epi16(_mm_unpacklo_epi8(va2, va2), 8);
      a2 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128i vinput_scale01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128 vinput_scale0 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale1 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(3, 3, 3, 3)));
    const __m128 vinput_scale2 = _mm_load1_ps(&quantization_params[2].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale1);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale2);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c2, vout2x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse2_ld64(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  __m128i vinput_zero_point = _mm_cvtsi32_si128(*((const int*) &quantization_params->zero_point));
  vinput_zero_point = _mm_shuffle_epi32(vinput_zero_point, _MM_SHUFFLE(0, 0, 0, 0));
  const __m128 vinput_scale = _mm_load1_ps(&quantization_params->inv_scale);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vksum_lo = _mm_srli_epi32(_mm_slli_epi32(vksum, 16), 16);
    const __m128i vksum_hi = _mm_srli_epi32(vksum, 16);

    __m128i vzpprodksumhi0 = _mm_mulhi_epu16(vinput_zero_point, vksum_lo);
    const __m128i vzpprodksumlo0 = _mm_mullo_epi16(vinput_zero_point, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse2_ld64(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  __m128i vinput_zero_point = _mm_cvtsi32_si128(*((const int*) &quantization_params->zero_point));
  vinput_zero_point = _mm_shuffle_epi32(vinput_zero_point, _MM_SHUFFLE(0, 0, 0, 0));
  const __m128 vinput_scale = _mm_load1_ps(&quantization_params->inv_scale);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vksum_lo = _mm_srli_epi32(_mm_slli_epi32(vksum, 16), 16);
    const __m128i vksum_hi = _mm_srli_epi32(vksum, 16);

    __m128i vzpprodksumhi0 = _mm_mulhi_epu16(vinput_zero_point, vksum_lo);
    const __m128i vzpprodksumlo0 = _mm_mullo_epi16(vinput_zero_point, vksum_lo);
    __m128i vzpprodksumhi1 = _mm_mulhi_epu16(vinput_zero_point, vksum_lo);
    const __m128i vzpprodksumlo1 = _mm_mullo_epi16(vinput_zero_point, vksum_lo);
    __m128i vzpprodksumhi2 = _mm_mulhi_epu16(vinput_zero_point, vksum_lo);
    const __m128i vzpprodksumlo2 = _mm_mullo_epi16(vinput_zero_point, vksum_lo);

    vzpprodksumhi0 = _mm_add_epi16(vzpprodksumhi0, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi0 = _mm_sub_epi16(vzpprodksumhi0, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));
    vzpprodksumhi1 = _mm_add_epi16(vzpprodksumhi1, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi1 = _mm_sub_epi16(vzpprodksumhi1, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));
    vzpprodksumhi2 = _mm_add_epi16(vzpprodksumhi2, _mm_mullo_epi16(vinput_zero_point, vksum_hi));
    vzpprodksumhi2 = _mm_sub_epi16(vzpprodksumhi2, _mm_and_si128(_mm_srai_epi16(vinput_zero_point, 15), vksum_lo));

    vzpprodksumhi0 = _mm_slli_si128(vzpprodksumhi0, 2);
    vzpprodksumhi1 = _mm_slli_si128(vzpprodksumhi1, 2);
    vzpprodksumhi2 = _mm_slli_si128(vzpprodksumhi2, 2);

    const __m128i vksumzp0 = _mm_or_si128(vzpprodksumhi0, vzpprodksumlo0);
    const __m128i vksumzp1 = _mm_or_si128(vzpprodksumhi1, vzpprodksumlo1);
    const __m128i vksumzp2 = _mm_or_si128(vzpprodksumhi2, vzpprodksumlo2);

    const __m128i vksum010 = _mm_unpacklo_epi32(vksumzp0, vzero);
    const __m128i vksum230 = _mm_unpackhi_epi32(vksumzp0, vzero);
    const __m128i vksum011 = _mm_unpacklo_epi32(vksumzp1, vzero);
    const __m128i vksum231 = _mm_unpackhi_epi32(vksumzp1, vzero);
    const __m128i vksum012 = _mm_unpacklo_epi32(vksumzp2, vzero);
    const __m128i vksum232 = _mm_unpackhi_epi32(vksumzp2, vzero);

    __m128i vacc0x0 = _mm_unpacklo_epi64(vksum010, vzero);
    __m128i vacc0x1 = _mm_unpackhi_epi64(vksum010, vzero);
    __m128i vacc1x0 = _mm_unpacklo_epi64(vksum011, vzero);
    __m128i vacc1x1 = _mm_unpackhi_epi64(vksum011, vzero);
    __m128i vacc2x0 = _mm_unpacklo_epi64(vksum012, vzero);
    __m128i vacc2x1 = _mm_unpackhi_epi64(vksum012, vzero);

    __m128i vacc0x2 = _mm_unpacklo_epi64(vksum230, vzero);
    __m128i vacc0x3 = _mm_unpackhi_epi64(vksum230, vzero);
    __m128i vacc1x2 = _mm_unpacklo_epi64(vksum231, vzero);
    __m128i vacc1x3 = _mm_unpackhi_epi64(vksum231, vzero);
    __m128i vacc2x2 = _mm_unpacklo_epi64(vksum232, vzero);
    __m128i vacc2x3 = _mm_unpackhi_epi64(vksum232, vzero);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      a += 3;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_srai_epi16(_mm_unpacklo_epi8(va2, va2), 8);
        a2 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c2, vout2x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c0, vout0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c2, vout2x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs16_qs8_vcvt_ukernel__sse2_u16(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_bias = _mm_set1_epi16(UINT16_C(0x8000));
  const __m128i vmultiplier = _mm_set1_epi32(params->scalar.multiplier);
  const __m128i vbias = _mm_set1_epi64x(
      (int64_t) ((uint64_t) params->scalar.output_zero_point << 32) + 
      INT64_C(0x80000000) -
      (INT64_C(0x80000000) * (int64_t) params->scalar.multiplier));
  const __m128i vzero = _mm_setzero_si128();
  XNN_FORCE_REALIZATION(vinput_bias);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  for (; batch >= 16 * sizeof(int16_t); batch -= 16 * sizeof(int16_t)) {
    __m128i vx0 = _mm_loadu_si128((const __m128i*) input); input += 8;
    __m128i vx2 = _mm_loadu_si128((const __m128i*) input); input += 8;

    // Add 0x8000 to convert signed inputs to unsigned.
    vx0 = _mm_xor_si128(vx0, vinput_bias);
    vx2 = _mm_xor_si128(vx2, vinput_bias);

    // Move int16 to upper part of int32
    __m128i vacce0 = _mm_unpacklo_epi16(vzero, vx0);
    __m128i vacce1 = _mm_unpackhi_epi16(vzero, vx0);
    __m128i vacce2 = _mm_unpacklo_epi16(vzero, vx2);
    __m128i vacce3 = _mm_unpackhi_epi16(vzero, vx2);

    __m128i vacco0 = _mm_shuffle_epi32(vacce0, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco1 = _mm_shuffle_epi32(vacce1, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco2 = _mm_shuffle_epi32(vacce2, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i vacco3 = _mm_shuffle_epi32(vacce3, _MM_SHUFFLE(3, 3, 1, 1));

    vacce0 = _mm_mul_epu32(vacce0, vmultiplier);
    vacco0 = _mm_mul_epu32(vacco0, vmultiplier);
    vacce1 = _mm_mul_epu32(vacce1, vmultiplier);
    vacco1 = _mm_mul_epu32(vacco1, vmultiplier);
    vacce2 = _mm_mul_epu32(vacce2, vmultiplier);
    vacco2 = _mm_mul_epu32(vacco2, vmultiplier);
    vacce3 = _mm_mul_epu32(vacce3, vmultiplier);
    vacco3 = _mm_mul_epu32(vacco3, vmultiplier);

    vacce0 = _mm_add_epi64(vacce0, vbias);
    vacco0 = _mm_add_epi64(vacco0, vbias);
    vacce1 = _mm_add_epi64(vacce1, vbias);
    vacco1 = _mm_add_epi64(vacco1, vbias);
    vacce2 = _mm_add_epi64(vacce2, vbias);
    vacco2 = _mm_add_epi64(vacco2, vbias);
    vacce3 = _mm_add_epi64(vacce3, vbias);
    vacco3 = _mm_add_epi64(vacco3, vbias);

    __m128i vacc0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce0),
                                                            _mm_castsi128_ps(vacco0),
                                                            _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce1),
                                                            _mm_castsi128_ps(vacco1),
                                                            _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce2),
                                                            _mm_castsi128_ps(vacco2),
                                                            _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce3),
                                                            _mm_castsi128_ps(vacco3),
                                                            _MM_SHUFFLE(3, 1, 3, 1)));

    // Shuffle order from 3,1,2,0 to 3,2,1,0
    vacc0 = _mm_shuffle_epi32(vacc0, _MM_SHUFFLE(3, 1, 2, 0));
    vacc1 = _mm_shuffle_epi32(vacc1, _MM_SHUFFLE(3, 1, 2, 0));
    vacc2 = _mm_shuffle_epi32(vacc2, _MM_SHUFFLE(3, 1, 2, 0));
    vacc3 = _mm_shuffle_epi32(vacc3, _MM_SHUFFLE(3, 1, 2, 0));

    // Pack 8 ints into 8 shorts
    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc2 = _mm_packs_epi32(vacc2, vacc3);

    // Pack 16 shorts into 16 bytes
    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc2);

    _mm_storeu_si128((__m128i*) output, vy0); output += 16;
  }

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    __m128i vx = _mm_loadl_epi64((const __m128i*) input); input += 4;
    vx = _mm_xor_si128(vx, vinput_bias);
    __m128i vacce = _mm_unpacklo_epi16(vzero, vx);
    __m128i vacco = _mm_shuffle_epi32(vacce, _MM_SHUFFLE(3, 3, 1, 1));
    vacce = _mm_mul_epu32(vacce, vmultiplier);
    vacco = _mm_mul_epu32(vacco, vmultiplier);
    vacce = _mm_add_epi64(vacce, vbias);
    vacco = _mm_add_epi64(vacco, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce), _mm_castsi128_ps(vacco),
                                                   _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_shuffle_epi32(vacc, _MM_SHUFFLE(3, 1, 2, 0));
    vacc = _mm_packs_epi32(vacc, vacc);
    const __m128i vy = _mm_packs_epi16(vacc, vacc);
    _mm_storeu_si32(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int16_t));
    assert(batch <= 3 * sizeof(int16_t));

    __m128i vx = _mm_loadl_epi64((const __m128i*) input);
    vx = _mm_xor_si128(vx, vinput_bias);
    __m128i vacce = _mm_unpacklo_epi16(vzero, vx);
    __m128i vacco = _mm_shuffle_epi32(vacce, _MM_SHUFFLE(3, 3, 1, 1));
    vacce = _mm_mul_epu32(vacce, vmultiplier);
    vacco = _mm_mul_epu32(vacco, vmultiplier);
    vacce = _mm_add_epi64(vacce, vbias);
    vacco = _mm_add_epi64(vacco, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacce), _mm_castsi128_ps(vacco),
                                                   _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_shuffle_epi32(vacc, _MM_SHUFFLE(3, 1, 2, 0));
    vacc = _mm_packs_epi32(vacc, vacc);
    __m128i vy = _mm_packs_epi16(vacc, vacc);

    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int16_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int16_t))) {
      *output = (int8_t) vy_lo;
    }
  }
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
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
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      i0 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

      const __m128i vsignprod1x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod1x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod1x01234567));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i2 += 8;

      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      i3 += 8;

      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

      const __m128i vsignprod3x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod3x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod3x01234567));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      i4 += 8;

      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i5 += 8;

      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
      const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

      const __m128i vsignprod5x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod5x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod5x01234567));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      i6 += 8;

      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
      const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      i7 += 8;

      const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
      const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

      const __m128i vsignprod7x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod7x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod7x01234567));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      i8 += 8;

      const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
      const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      i9 += 8;

      const __m128i vxi9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi9x01234567, vi9x01234567), 8);
      const __m128i vxk9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk9x01234567, vk9x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));

      const __m128i vsignprod9x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod9x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod9x01234567));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      i10 += 8;

      const __m128i vxi10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi10x01234567, vi10x01234567), 8);
      const __m128i vxk10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk10x01234567, vk10x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      i11 += 8;

      const __m128i vxi11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi11x01234567, vi11x01234567), 8);
      const __m128i vxk11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk11x01234567, vk11x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));

      const __m128i vsignprod11x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod11x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod11x01234567));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      i12 += 8;

      const __m128i vxi12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi12x01234567, vi12x01234567), 8);
      const __m128i vxk12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk12x01234567, vk12x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      i13 += 8;

      const __m128i vxi13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi13x01234567, vi13x01234567), 8);
      const __m128i vxk13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk13x01234567, vk13x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));

      const __m128i vsignprod13x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod13x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod13x01234567));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      i14 += 8;

      const __m128i vxi14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi14x01234567, vi14x01234567), 8);
      const __m128i vxk14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk14x01234567, vk14x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      i15 += 8;

      const __m128i vxi15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi15x01234567, vi15x01234567), 8);
      const __m128i vxk15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk15x01234567, vk15x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));

      const __m128i vsignprod15x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod15x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod15x01234567));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      i16 += 8;

      const __m128i vxi16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi16x01234567, vi16x01234567), 8);
      const __m128i vxk16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk16x01234567, vk16x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      i17 += 8;

      const __m128i vxi17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi17x01234567, vi17x01234567), 8);
      const __m128i vxk17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk17x01234567, vk17x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));

      const __m128i vsignprod17x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod17x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod17x01234567));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      i18 += 8;

      const __m128i vxi18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi18x01234567, vi18x01234567), 8);
      const __m128i vxk18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk18x01234567, vk18x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      i19 += 8;

      const __m128i vxi19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi19x01234567, vi19x01234567), 8);
      const __m128i vxk19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk19x01234567, vk19x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));

      const __m128i vsignprod19x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod19x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod19x01234567));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      i20 += 8;

      const __m128i vxi20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi20x01234567, vi20x01234567), 8);
      const __m128i vxk20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk20x01234567, vk20x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      i21 += 8;

      const __m128i vxi21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi21x01234567, vi21x01234567), 8);
      const __m128i vxk21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk21x01234567, vk21x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));

      const __m128i vsignprod21x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod21x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod21x01234567));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      i22 += 8;

      const __m128i vxi22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi22x01234567, vi22x01234567), 8);
      const __m128i vxk22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk22x01234567, vk22x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      i23 += 8;

      const __m128i vxi23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi23x01234567, vi23x01234567), 8);
      const __m128i vxk23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk23x01234567, vk23x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));

      const __m128i vsignprod23x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod23x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod23x01234567));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      i24 += 8;

      const __m128i vxi24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi24x01234567, vi24x01234567), 8);
      const __m128i vxk24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk24x01234567, vk24x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

      const __m128i vsignprod24x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod24x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod24x01234567));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t));

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

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));

        const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
        const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));

        const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
        const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        const __m128i vsignprod1x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod1x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod1x01234567));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));

        const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
        const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));

        const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
        const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        const __m128i vsignprod3x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod3x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod3x01234567));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));

        const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
        const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));

        const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
        const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        const __m128i vsignprod5x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod5x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod5x01234567));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));

        const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
        const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));

        const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
        const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        const __m128i vsignprod7x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod7x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod7x01234567));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));

        const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
        const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));

        const __m128i vxi9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi9x01234567, vi9x01234567), 8);
        const __m128i vxk9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk9x01234567, vk9x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));

        const __m128i vsignprod9x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod9x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod9x01234567));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t)));

        const __m128i vxi10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi10x01234567, vi10x01234567), 8);
        const __m128i vxk10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk10x01234567, vk10x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t)));

        const __m128i vxi11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi11x01234567, vi11x01234567), 8);
        const __m128i vxk11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk11x01234567, vk11x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));

        const __m128i vsignprod11x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod11x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod11x01234567));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t)));

        const __m128i vxi12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi12x01234567, vi12x01234567), 8);
        const __m128i vxk12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk12x01234567, vk12x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t)));

        const __m128i vxi13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi13x01234567, vi13x01234567), 8);
        const __m128i vxk13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk13x01234567, vk13x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));

        const __m128i vsignprod13x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod13x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod13x01234567));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t)));

        const __m128i vxi14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi14x01234567, vi14x01234567), 8);
        const __m128i vxk14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk14x01234567, vk14x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t)));

        const __m128i vxi15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi15x01234567, vi15x01234567), 8);
        const __m128i vxk15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk15x01234567, vk15x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));

        const __m128i vsignprod15x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod15x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod15x01234567));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t)));

        const __m128i vxi16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi16x01234567, vi16x01234567), 8);
        const __m128i vxk16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk16x01234567, vk16x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t)));

        const __m128i vxi17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi17x01234567, vi17x01234567), 8);
        const __m128i vxk17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk17x01234567, vk17x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));

        const __m128i vsignprod17x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod17x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod17x01234567));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));

        const __m128i vxi18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi18x01234567, vi18x01234567), 8);
        const __m128i vxk18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk18x01234567, vk18x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t)));

        const __m128i vxi19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi19x01234567, vi19x01234567), 8);
        const __m128i vxk19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk19x01234567, vk19x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));

        const __m128i vsignprod19x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod19x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod19x01234567));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t)));

        const __m128i vxi20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi20x01234567, vi20x01234567), 8);
        const __m128i vxk20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk20x01234567, vk20x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t)));

        const __m128i vxi21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi21x01234567, vi21x01234567), 8);
        const __m128i vxk21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk21x01234567, vk21x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));

        const __m128i vsignprod21x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod21x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod21x01234567));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t)));

        const __m128i vxi22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi22x01234567, vi22x01234567), 8);
        const __m128i vxk22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk22x01234567, vk22x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t)));

        const __m128i vxi23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi23x01234567, vi23x01234567), 8);
        const __m128i vxk23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk23x01234567, vk23x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));

        const __m128i vsignprod23x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod23x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod23x01234567));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t)));

        const __m128i vxi24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi24x01234567, vi24x01234567), 8);
        const __m128i vxk24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk24x01234567, vk24x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

        const __m128i vsignprod24x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod24x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod24x01234567));


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


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        vout01234567 = _mm_max_epi16(vout01234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
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
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      i0 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

      const __m128i vsignprod1x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod1x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod1x01234567));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i2 += 8;

      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      i3 += 8;

      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

      const __m128i vsignprod3x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod3x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod3x01234567));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      i4 += 8;

      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i5 += 8;

      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
      const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

      const __m128i vsignprod5x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod5x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod5x01234567));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      i6 += 8;

      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
      const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      i7 += 8;

      const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
      const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

      const __m128i vsignprod7x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod7x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod7x01234567));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      i8 += 8;

      const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
      const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

      const __m128i vsignprod8x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod8x01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod8x01234567));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t));

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

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));

        const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
        const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));

        const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
        const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        const __m128i vsignprod1x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod1x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod1x01234567));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));

        const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
        const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));

        const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
        const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        const __m128i vsignprod3x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod3x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod3x01234567));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));

        const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
        const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));

        const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
        const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        const __m128i vsignprod5x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod5x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod5x01234567));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));

        const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
        const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));

        const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
        const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        const __m128i vsignprod7x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod7x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod7x01234567));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));

        const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
        const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        const __m128i vsignprod8x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vprod01234567);
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod01234567, vsignprod8x01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod01234567, vsignprod8x01234567));


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


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        vout01234567 = _mm_max_epi16(vout01234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_f32_vcvt_ukernel__sse2_u32(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vsign_mask = _mm_set1_epi8(UINT8_C(0x80));
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m128i vmagic_exp = _mm_set1_epi16(UINT16_C(0x4B00));
  const __m128 vmagic_bias = _mm_set1_ps((float) (INT32_C(0x00800080) + (int32_t) params->scalar.zero_point));
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i vzero = _mm_setzero_si128();
  XNN_FORCE_REALIZATION(vmagic_exp);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m128i vx01234567 = _mm_loadl_epi64((const __m128i*) input);
    __m128i vx89ABCDEF = _mm_loadl_epi64((const __m128i*) (input + 8));
    __m128i vxGHIJKLMN = _mm_loadl_epi64((const __m128i*) (input + 16));
    __m128i vxOPQRSTUV = _mm_loadl_epi64((const __m128i*) (input + 24));
    input += 32;

    vx01234567 = _mm_xor_si128(vx01234567, vsign_mask);
    vx89ABCDEF = _mm_xor_si128(vx89ABCDEF, vsign_mask);
    vxGHIJKLMN = _mm_xor_si128(vxGHIJKLMN, vsign_mask);
    vxOPQRSTUV = _mm_xor_si128(vxOPQRSTUV, vsign_mask);

    vx01234567 = _mm_unpacklo_epi8(vx01234567, vzero);
    vx89ABCDEF = _mm_unpacklo_epi8(vx89ABCDEF, vzero);
    vxGHIJKLMN = _mm_unpacklo_epi8(vxGHIJKLMN, vzero);
    vxOPQRSTUV = _mm_unpacklo_epi8(vxOPQRSTUV, vzero);

    __m128 vy0123 = _mm_castsi128_ps(_mm_unpacklo_epi16(vx01234567, vmagic_exp));
    __m128 vy4567 = _mm_castsi128_ps(_mm_unpackhi_epi16(vx01234567, vmagic_exp));
    __m128 vy89AB = _mm_castsi128_ps(_mm_unpacklo_epi16(vx89ABCDEF, vmagic_exp));
    __m128 vyCDEF = _mm_castsi128_ps(_mm_unpackhi_epi16(vx89ABCDEF, vmagic_exp));
    __m128 vyGHIJ = _mm_castsi128_ps(_mm_unpacklo_epi16(vxGHIJKLMN, vmagic_exp));
    __m128 vyKLMN = _mm_castsi128_ps(_mm_unpackhi_epi16(vxGHIJKLMN, vmagic_exp));
    __m128 vyOPQR = _mm_castsi128_ps(_mm_unpacklo_epi16(vxOPQRSTUV, vmagic_exp));
    __m128 vySTUV = _mm_castsi128_ps(_mm_unpackhi_epi16(vxOPQRSTUV, vmagic_exp));

    vy0123 = _mm_sub_ps(vy0123, vmagic_bias);
    vy4567 = _mm_sub_ps(vy4567, vmagic_bias);
    vy89AB = _mm_sub_ps(vy89AB, vmagic_bias);
    vyCDEF = _mm_sub_ps(vyCDEF, vmagic_bias);
    vyGHIJ = _mm_sub_ps(vyGHIJ, vmagic_bias);
    vyKLMN = _mm_sub_ps(vyKLMN, vmagic_bias);
    vyOPQR = _mm_sub_ps(vyOPQR, vmagic_bias);
    vySTUV = _mm_sub_ps(vySTUV, vmagic_bias);

    vy0123 = _mm_mul_ps(vy0123, vscale);
    vy4567 = _mm_mul_ps(vy4567, vscale);
    vy89AB = _mm_mul_ps(vy89AB, vscale);
    vyCDEF = _mm_mul_ps(vyCDEF, vscale);
    vyGHIJ = _mm_mul_ps(vyGHIJ, vscale);
    vyKLMN = _mm_mul_ps(vyKLMN, vscale);
    vyOPQR = _mm_mul_ps(vyOPQR, vscale);
    vySTUV = _mm_mul_ps(vySTUV, vscale);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    _mm_storeu_ps(output + 16, vyGHIJ);
    _mm_storeu_ps(output + 20, vyKLMN);
    _mm_storeu_ps(output + 24, vyOPQR);
    _mm_storeu_ps(output + 28, vySTUV);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i vx = _mm_loadl_epi64((const __m128i*) input);
    vx = _mm_xor_si128(vx, vsign_mask);
    vx = _mm_unpacklo_epi8(vx, vzero);
    input += 8;

    __m128 vy_lo = _mm_castsi128_ps(_mm_unpacklo_epi16(vx, vmagic_exp));
    __m128 vy_hi = _mm_castsi128_ps(_mm_unpackhi_epi16(vx, vmagic_exp));

    vy_lo = _mm_sub_ps(vy_lo, vmagic_bias);
    vy_hi = _mm_sub_ps(vy_hi, vmagic_bias);

    vy_lo = _mm_mul_ps(vy_lo, vscale);
    vy_hi = _mm_mul_ps(vy_hi, vscale);

    _mm_storeu_ps(output, vy_lo);
    _mm_storeu_ps(output + 4, vy_hi);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m128i vx = _mm_loadl_epi64((const __m128i*) input);
    vx = _mm_xor_si128(vx, vsign_mask);
    vx = _mm_unpacklo_epi8(vx, vzero);

    __m128 vy = _mm_castsi128_ps(_mm_unpacklo_epi16(vx, vmagic_exp));
    vy = _mm_sub_ps(vy, vmagic_bias);
    vy = _mm_mul_ps(vy, vscale);

    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_ps(output, vy);
      vy = _mm_castsi128_ps(_mm_unpackhi_epi16(vx, vmagic_exp));
      vy = _mm_sub_ps(vy, vmagic_bias);
      vy = _mm_mul_ps(vy, vscale);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(int8_t);

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse2.init_bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;

    const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;

    const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    _mm_store_si128((__m128i*) b, vacc0123);
    _mm_store_si128((__m128i*) (b + 4), vacc4567);
    b += 8;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {

      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
      __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) b));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (b + 4)));

      _mm_store_si128((__m128i*) b, vacc0123);
      _mm_store_si128((__m128i*) (b + 4), vacc4567);
      b += 8;
    }
  }

  i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
  for (; channels >= 8; channels -= 8) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;

    const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;

    const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);

    vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
    buffer += 8;

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {

      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
      __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
      buffer += 8;

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      uint32_t vout0123 = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) vout0123);
        vout0123 >>= 16;
        output += 2;
      }
      if (channels & 1) {
        *output = (int8_t) vout0123;
      }
    }
  }
}

void xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse2.init_bias);
  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
  for (; channels >= 8; channels -= 8) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;

    const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;

    const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {

      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
      __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);

      vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
      vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      uint32_t vout0123 = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) vout0123);
        vout0123 >>= 16;
        output += 2;
      }
      if (channels & 1) {
        *output = (int8_t) vout0123;
      }
    }
  }
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      i0 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i2 += 8;

      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      i3 += 8;

      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      i4 += 8;

      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i5 += 8;

      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
      const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      i6 += 8;

      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
      const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      i7 += 8;

      const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
      const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      i8 += 8;

      const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
      const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      i9 += 8;

      const __m128i vxi9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi9x01234567, vi9x01234567), 8);
      const __m128i vxk9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk9x01234567, vk9x01234567), 8);

      const __m128i vprod9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
      const __m128i vprod9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod9x01234567lo, vprod9x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod9x01234567lo, vprod9x01234567hi));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      i10 += 8;

      const __m128i vxi10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi10x01234567, vi10x01234567), 8);
      const __m128i vxk10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk10x01234567, vk10x01234567), 8);

      const __m128i vprod10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      const __m128i vprod10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod10x01234567lo, vprod10x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod10x01234567lo, vprod10x01234567hi));

      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      i11 += 8;

      const __m128i vxi11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi11x01234567, vi11x01234567), 8);
      const __m128i vxk11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk11x01234567, vk11x01234567), 8);

      const __m128i vprod11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
      const __m128i vprod11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod11x01234567lo, vprod11x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod11x01234567lo, vprod11x01234567hi));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      i12 += 8;

      const __m128i vxi12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi12x01234567, vi12x01234567), 8);
      const __m128i vxk12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk12x01234567, vk12x01234567), 8);

      const __m128i vprod12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      const __m128i vprod12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod12x01234567lo, vprod12x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod12x01234567lo, vprod12x01234567hi));

      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      i13 += 8;

      const __m128i vxi13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi13x01234567, vi13x01234567), 8);
      const __m128i vxk13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk13x01234567, vk13x01234567), 8);

      const __m128i vprod13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
      const __m128i vprod13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod13x01234567lo, vprod13x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod13x01234567lo, vprod13x01234567hi));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      i14 += 8;

      const __m128i vxi14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi14x01234567, vi14x01234567), 8);
      const __m128i vxk14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk14x01234567, vk14x01234567), 8);

      const __m128i vprod14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      const __m128i vprod14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod14x01234567lo, vprod14x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod14x01234567lo, vprod14x01234567hi));

      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      i15 += 8;

      const __m128i vxi15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi15x01234567, vi15x01234567), 8);
      const __m128i vxk15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk15x01234567, vk15x01234567), 8);

      const __m128i vprod15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
      const __m128i vprod15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod15x01234567lo, vprod15x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod15x01234567lo, vprod15x01234567hi));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      i16 += 8;

      const __m128i vxi16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi16x01234567, vi16x01234567), 8);
      const __m128i vxk16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk16x01234567, vk16x01234567), 8);

      const __m128i vprod16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      const __m128i vprod16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod16x01234567lo, vprod16x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod16x01234567lo, vprod16x01234567hi));

      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      i17 += 8;

      const __m128i vxi17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi17x01234567, vi17x01234567), 8);
      const __m128i vxk17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk17x01234567, vk17x01234567), 8);

      const __m128i vprod17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
      const __m128i vprod17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod17x01234567lo, vprod17x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod17x01234567lo, vprod17x01234567hi));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      i18 += 8;

      const __m128i vxi18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi18x01234567, vi18x01234567), 8);
      const __m128i vxk18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk18x01234567, vk18x01234567), 8);

      const __m128i vprod18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      const __m128i vprod18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod18x01234567lo, vprod18x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod18x01234567lo, vprod18x01234567hi));

      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      i19 += 8;

      const __m128i vxi19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi19x01234567, vi19x01234567), 8);
      const __m128i vxk19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk19x01234567, vk19x01234567), 8);

      const __m128i vprod19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
      const __m128i vprod19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod19x01234567lo, vprod19x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod19x01234567lo, vprod19x01234567hi));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      i20 += 8;

      const __m128i vxi20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi20x01234567, vi20x01234567), 8);
      const __m128i vxk20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk20x01234567, vk20x01234567), 8);

      const __m128i vprod20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      const __m128i vprod20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod20x01234567lo, vprod20x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod20x01234567lo, vprod20x01234567hi));

      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      i21 += 8;

      const __m128i vxi21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi21x01234567, vi21x01234567), 8);
      const __m128i vxk21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk21x01234567, vk21x01234567), 8);

      const __m128i vprod21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
      const __m128i vprod21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod21x01234567lo, vprod21x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod21x01234567lo, vprod21x01234567hi));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      i22 += 8;

      const __m128i vxi22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi22x01234567, vi22x01234567), 8);
      const __m128i vxk22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk22x01234567, vk22x01234567), 8);

      const __m128i vprod22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      const __m128i vprod22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod22x01234567lo, vprod22x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod22x01234567lo, vprod22x01234567hi));

      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      i23 += 8;

      const __m128i vxi23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi23x01234567, vi23x01234567), 8);
      const __m128i vxk23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk23x01234567, vk23x01234567), 8);

      const __m128i vprod23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
      const __m128i vprod23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod23x01234567lo, vprod23x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod23x01234567lo, vprod23x01234567hi));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      i24 += 8;

      const __m128i vxi24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi24x01234567, vi24x01234567), 8);
      const __m128i vxk24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk24x01234567, vk24x01234567), 8);

      const __m128i vprod24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      const __m128i vprod24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod24x01234567lo, vprod24x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod24x01234567lo, vprod24x01234567hi));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      w = (const void*) ((const float*) w + 8);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));

        const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
        const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));

        const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
        const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));

        const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
        const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));

        const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
        const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));

        const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
        const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));

        const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
        const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));

        const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
        const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));

        const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
        const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));

        const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
        const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));

        const __m128i vxi9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi9x01234567, vi9x01234567), 8);
        const __m128i vxk9x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk9x01234567, vk9x01234567), 8);

        const __m128i vprod9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
        const __m128i vprod9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod9x01234567lo, vprod9x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod9x01234567lo, vprod9x01234567hi));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t)));

        const __m128i vxi10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi10x01234567, vi10x01234567), 8);
        const __m128i vxk10x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk10x01234567, vk10x01234567), 8);

        const __m128i vprod10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
        const __m128i vprod10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod10x01234567lo, vprod10x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod10x01234567lo, vprod10x01234567hi));

        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t)));

        const __m128i vxi11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi11x01234567, vi11x01234567), 8);
        const __m128i vxk11x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk11x01234567, vk11x01234567), 8);

        const __m128i vprod11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
        const __m128i vprod11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod11x01234567lo, vprod11x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod11x01234567lo, vprod11x01234567hi));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t)));

        const __m128i vxi12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi12x01234567, vi12x01234567), 8);
        const __m128i vxk12x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk12x01234567, vk12x01234567), 8);

        const __m128i vprod12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
        const __m128i vprod12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod12x01234567lo, vprod12x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod12x01234567lo, vprod12x01234567hi));

        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t)));

        const __m128i vxi13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi13x01234567, vi13x01234567), 8);
        const __m128i vxk13x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk13x01234567, vk13x01234567), 8);

        const __m128i vprod13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
        const __m128i vprod13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod13x01234567lo, vprod13x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod13x01234567lo, vprod13x01234567hi));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t)));

        const __m128i vxi14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi14x01234567, vi14x01234567), 8);
        const __m128i vxk14x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk14x01234567, vk14x01234567), 8);

        const __m128i vprod14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
        const __m128i vprod14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod14x01234567lo, vprod14x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod14x01234567lo, vprod14x01234567hi));

        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t)));

        const __m128i vxi15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi15x01234567, vi15x01234567), 8);
        const __m128i vxk15x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk15x01234567, vk15x01234567), 8);

        const __m128i vprod15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
        const __m128i vprod15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod15x01234567lo, vprod15x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod15x01234567lo, vprod15x01234567hi));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t)));

        const __m128i vxi16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi16x01234567, vi16x01234567), 8);
        const __m128i vxk16x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk16x01234567, vk16x01234567), 8);

        const __m128i vprod16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
        const __m128i vprod16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod16x01234567lo, vprod16x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod16x01234567lo, vprod16x01234567hi));

        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t)));

        const __m128i vxi17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi17x01234567, vi17x01234567), 8);
        const __m128i vxk17x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk17x01234567, vk17x01234567), 8);

        const __m128i vprod17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
        const __m128i vprod17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod17x01234567lo, vprod17x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod17x01234567lo, vprod17x01234567hi));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));

        const __m128i vxi18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi18x01234567, vi18x01234567), 8);
        const __m128i vxk18x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk18x01234567, vk18x01234567), 8);

        const __m128i vprod18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
        const __m128i vprod18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod18x01234567lo, vprod18x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod18x01234567lo, vprod18x01234567hi));

        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t)));

        const __m128i vxi19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi19x01234567, vi19x01234567), 8);
        const __m128i vxk19x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk19x01234567, vk19x01234567), 8);

        const __m128i vprod19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
        const __m128i vprod19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod19x01234567lo, vprod19x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod19x01234567lo, vprod19x01234567hi));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t)));

        const __m128i vxi20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi20x01234567, vi20x01234567), 8);
        const __m128i vxk20x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk20x01234567, vk20x01234567), 8);

        const __m128i vprod20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
        const __m128i vprod20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod20x01234567lo, vprod20x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod20x01234567lo, vprod20x01234567hi));

        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t)));

        const __m128i vxi21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi21x01234567, vi21x01234567), 8);
        const __m128i vxk21x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk21x01234567, vk21x01234567), 8);

        const __m128i vprod21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
        const __m128i vprod21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod21x01234567lo, vprod21x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod21x01234567lo, vprod21x01234567hi));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t)));

        const __m128i vxi22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi22x01234567, vi22x01234567), 8);
        const __m128i vxk22x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk22x01234567, vk22x01234567), 8);

        const __m128i vprod22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
        const __m128i vprod22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod22x01234567lo, vprod22x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod22x01234567lo, vprod22x01234567hi));

        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t)));

        const __m128i vxi23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi23x01234567, vi23x01234567), 8);
        const __m128i vxk23x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk23x01234567, vk23x01234567), 8);

        const __m128i vprod23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
        const __m128i vprod23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod23x01234567lo, vprod23x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod23x01234567lo, vprod23x01234567hi));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t)));

        const __m128i vxi24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi24x01234567, vi24x01234567), 8);
        const __m128i vxk24x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk24x01234567, vk24x01234567), 8);

        const __m128i vprod24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
        const __m128i vprod24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod24x01234567lo, vprod24x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod24x01234567lo, vprod24x01234567hi));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        vout01234567 = _mm_max_epi16(vout01234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__sse2_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      i0 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i2 += 8;

      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      w = (const void*) ((const float*) w + 8);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));

        const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
        const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));

        const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
        const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));

        const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
        const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        vout01234567 = _mm_max_epi16(vout01234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      i0 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i2 += 8;

      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      i3 += 8;

      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      i4 += 8;

      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i5 += 8;

      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
      const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      i6 += 8;

      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
      const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      i7 += 8;

      const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
      const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      i8 += 8;

      const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
      const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      w = (const void*) ((const float*) w + 8);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));

        const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
        const __m128i vxk0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk0x01234567, vk0x01234567), 8);

        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));

        const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
        const __m128i vxk1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk1x01234567, vk1x01234567), 8);

        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));

        const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
        const __m128i vxk2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk2x01234567, vk2x01234567), 8);

        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));

        const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
        const __m128i vxk3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk3x01234567, vk3x01234567), 8);

        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));

        const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
        const __m128i vxk4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk4x01234567, vk4x01234567), 8);

        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));

        const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
        const __m128i vxk5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk5x01234567, vk5x01234567), 8);

        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));

        const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
        const __m128i vxk6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk6x01234567, vk6x01234567), 8);

        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));

        const __m128i vxi7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi7x01234567, vi7x01234567), 8);
        const __m128i vxk7x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk7x01234567, vk7x01234567), 8);

        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));

        const __m128i vxi8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi8x01234567, vi8x01234567), 8);
        const __m128i vxk8x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vk8x01234567, vk8x01234567), 8);

        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        vout01234567 = _mm_max_epi16(vout01234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
    vacc00x0123 = _mm_max_epi16(vacc00x0123, voutput_min);

    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);


    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_cvtsi128_si32(vout);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
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


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    __m128i vacc2x0 = vacc0x0;
    __m128i vacc2x1 = vacc0x1;
    __m128i vacc2x2 = vacc0x2;
    __m128i vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_srai_epi16(_mm_unpacklo_epi8(va2, va2), 8);
      a2 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vscaled2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);
    vscaled2x0123 = _mm_mul_ps(vscaled2x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm_min_ps(vscaled2x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm_cvtps_epi32(vscaled2x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vacc22x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc2x0123, vacc2x0123), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
    vacc01x0123 = _mm_max_epi16(vacc01x0123, voutput_min);
    vacc22x0123 = _mm_max_epi16(vacc22x0123, voutput_min);

    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc22x0123);


    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      vout = _mm_srli_si128(vout, 4);
      unaligned_store_u32(c1, (uint32_t) _mm_cvtsi128_si32(vout));
      vout = _mm_srli_si128(vout, 4);
      unaligned_store_u32(c2, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout, 4));
        c2 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_cvtsi128_si32(vout);
        *c1 = (int8_t) _mm_extract_epi16(vout, 2);
        *c2 = (int8_t) _mm_extract_epi16(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

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
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
    vacc00x0123 = _mm_max_epi16(vacc00x0123, voutput_min);

    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);


    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_cvtsi128_si32(vout);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    __m128i vacc2x0 = vacc0x0;
    __m128i vacc2x1 = vacc0x1;
    __m128i vacc2x2 = vacc0x2;
    __m128i vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

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
      a += 3;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_srai_epi16(_mm_unpacklo_epi8(va2, va2), 8);
        a2 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vscaled2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);
    vscaled2x0123 = _mm_mul_ps(vscaled2x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm_min_ps(vscaled2x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm_cvtps_epi32(vscaled2x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vacc22x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc2x0123, vacc2x0123), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
    vacc01x0123 = _mm_max_epi16(vacc01x0123, voutput_min);
    vacc22x0123 = _mm_max_epi16(vacc22x0123, voutput_min);

    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc22x0123);


    if (nc >= 4) {
      unaligned_store_u32(c2, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(vout, _MM_SHUFFLE(2, 2, 2, 2))));
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      unaligned_store_u32(c1, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(vout, _MM_SHUFFLE(1, 1, 1, 1))));
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout, 4));
        c2 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c2 = (int8_t) _mm_extract_epi16(vout, 4);
        *c1 = (int8_t) _mm_extract_epi16(vout, 2);
        *c0 = (int8_t) _mm_cvtsi128_si32(vout);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.bias);
  const __m128i va_multiplier_lo = _mm_set1_epi16(params->scalar.a_multiplier);
  const __m128i va_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.a_multiplier >> 16);
  const __m128i vb_multiplier_lo = _mm_set1_epi16(params->scalar.b_multiplier);
  const __m128i vb_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.b_multiplier >> 16);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi16(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi16(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier_lo);
  XNN_FORCE_REALIZATION(va_multiplier_hi);
  XNN_FORCE_REALIZATION(vb_multiplier_lo);
  XNN_FORCE_REALIZATION(vb_multiplier_hi);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);
    input_a += 8;
    input_b += 8;

    va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);
    vb01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vb01234567, vb01234567), 8);

    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
    const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
    vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));

    vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));
    vbprod01234567hi = _mm_sub_epi16(vbprod01234567hi, _mm_and_si128(_mm_srai_epi16(vb01234567, 15), vb_multiplier_lo));

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

    vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
      __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);

      va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);
      vb01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vb01234567, vb01234567), 8);

      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
      const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
      vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));

      vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));
      vbprod01234567hi = _mm_sub_epi16(vbprod01234567hi, _mm_and_si128(_mm_srai_epi16(vb01234567, 15), vb_multiplier_lo));

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (batch & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.b_multiplier * (int32_t) *input_b + params->scalar.bias);
  const __m128i va_multiplier_lo = _mm_set1_epi16(params->scalar.a_multiplier);
  const __m128i va_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.a_multiplier >> 16);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi16(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi16(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier_lo);
  XNN_FORCE_REALIZATION(va_multiplier_hi);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    input_a += 8;

    va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);

    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));

    vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

    vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);

      va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);

      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));

      vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (batch & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_qs8_vcvt_ukernel__sse2_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmultiplier = _mm_set1_epi16(-params->scalar.multiplier);
  const __m128i vbias = _mm_set1_epi32(
      ((int32_t) params->scalar.output_zero_point << 8) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point + 
      INT32_C(0x80));
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    const __m128i vm0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx0);
    const __m128i vextx0 = _mm_unpacklo_epi8(vx0, vm0);
    const __m128i vextx1 = _mm_unpackhi_epi8(vx0, vm0);
    const __m128i vm1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx1);
    const __m128i vextx2 = _mm_unpacklo_epi8(vx1, vm1);
    const __m128i vextx3 = _mm_unpackhi_epi8(vx1, vm1);

    const __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier);
    const __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vmultiplier);
    const __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier);
    const __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vmultiplier);
    const __m128i vprodlo2 = _mm_mullo_epi16(vextx2, vmultiplier);
    const __m128i vprodhi2 = _mm_mulhi_epi16(vextx2, vmultiplier);
    const __m128i vprodlo3 = _mm_mullo_epi16(vextx3, vmultiplier);
    const __m128i vprodhi3 = _mm_mulhi_epi16(vextx3, vmultiplier);

    __m128i vacc0 = _mm_unpacklo_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_unpackhi_epi16(vprodlo0, vprodhi0);
    __m128i vacc2 = _mm_unpacklo_epi16(vprodlo1, vprodhi1);
    __m128i vacc3 = _mm_unpackhi_epi16(vprodlo1, vprodhi1);
    __m128i vacc4 = _mm_unpacklo_epi16(vprodlo2, vprodhi2);
    __m128i vacc5 = _mm_unpackhi_epi16(vprodlo2, vprodhi2);
    __m128i vacc6 = _mm_unpacklo_epi16(vprodlo3, vprodhi3);
    __m128i vacc7 = _mm_unpackhi_epi16(vprodlo3, vprodhi3);

    vacc0 = _mm_sub_epi32(vbias, vacc0);
    vacc1 = _mm_sub_epi32(vbias, vacc1);
    vacc2 = _mm_sub_epi32(vbias, vacc2);
    vacc3 = _mm_sub_epi32(vbias, vacc3);
    vacc4 = _mm_sub_epi32(vbias, vacc4);
    vacc5 = _mm_sub_epi32(vbias, vacc5);
    vacc6 = _mm_sub_epi32(vbias, vacc6);
    vacc7 = _mm_sub_epi32(vbias, vacc7);

    vacc0 = _mm_srai_epi32(vacc0, 8);
    vacc1 = _mm_srai_epi32(vacc1, 8);
    vacc2 = _mm_srai_epi32(vacc2, 8);
    vacc3 = _mm_srai_epi32(vacc3, 8);
    vacc4 = _mm_srai_epi32(vacc4, 8);
    vacc5 = _mm_srai_epi32(vacc5, 8);
    vacc6 = _mm_srai_epi32(vacc6, 8);
    vacc7 = _mm_srai_epi32(vacc7, 8);

    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc1 = _mm_packs_epi32(vacc2, vacc3);
    vacc2 = _mm_packs_epi32(vacc4, vacc5);
    vacc3 = _mm_packs_epi32(vacc6, vacc7);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    const __m128i vextx_lo = _mm_unpacklo_epi8(vx, vm);
    const __m128i vextx_hi = _mm_unpackhi_epi8(vx, vm);

    const __m128i vprodlo_lo = _mm_mullo_epi16(vextx_lo, vmultiplier);
    const __m128i vprodlo_hi = _mm_mullo_epi16(vextx_hi, vmultiplier);
    const __m128i vprodhi_lo = _mm_mulhi_epi16(vextx_lo, vmultiplier);
    const __m128i vprodhi_hi = _mm_mulhi_epi16(vextx_hi, vmultiplier);

    __m128i vacc_ll = _mm_unpacklo_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_lh = _mm_unpackhi_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_hl = _mm_unpacklo_epi16(vprodlo_hi, vprodhi_hi);
    __m128i vacc_hh = _mm_unpackhi_epi16(vprodlo_hi, vprodhi_hi);

    vacc_ll = _mm_sub_epi32(vbias, vacc_ll);
    vacc_lh = _mm_sub_epi32(vbias, vacc_lh);
    vacc_hl = _mm_sub_epi32(vbias, vacc_hl);
    vacc_hh = _mm_sub_epi32(vbias, vacc_hh);

    vacc_ll = _mm_srai_epi32(vacc_ll, 8);
    vacc_lh = _mm_srai_epi32(vacc_lh, 8);
    vacc_hl = _mm_srai_epi32(vacc_hl, 8);
    vacc_hh = _mm_srai_epi32(vacc_hh, 8);

    const __m128i vacc_lo = _mm_packs_epi32(vacc_ll, vacc_lh);
    const __m128i vacc_hi = _mm_packs_epi32(vacc_hl, vacc_hh);

    const __m128i vy = _mm_packs_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    const __m128i vextx_lo = _mm_unpacklo_epi8(vx, vm);
    const __m128i vextx_hi = _mm_unpackhi_epi8(vx, vm);

    const __m128i vprodlo_lo = _mm_mullo_epi16(vextx_lo, vmultiplier);
    const __m128i vprodlo_hi = _mm_mullo_epi16(vextx_hi, vmultiplier);
    const __m128i vprodhi_lo = _mm_mulhi_epi16(vextx_lo, vmultiplier);
    const __m128i vprodhi_hi = _mm_mulhi_epi16(vextx_hi, vmultiplier);

    __m128i vacc_ll = _mm_unpacklo_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_lh = _mm_unpackhi_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_hl = _mm_unpacklo_epi16(vprodlo_hi, vprodhi_hi);
    __m128i vacc_hh = _mm_unpackhi_epi16(vprodlo_hi, vprodhi_hi);

    vacc_ll = _mm_sub_epi32(vbias, vacc_ll);
    vacc_lh = _mm_sub_epi32(vbias, vacc_lh);
    vacc_hl = _mm_sub_epi32(vbias, vacc_hl);
    vacc_hh = _mm_sub_epi32(vbias, vacc_hh);

    vacc_ll = _mm_srai_epi32(vacc_ll, 8);
    vacc_lh = _mm_srai_epi32(vacc_lh, 8);
    vacc_hl = _mm_srai_epi32(vacc_hl, 8);
    vacc_hh = _mm_srai_epi32(vacc_hh, 8);

    const __m128i vacc_lo = _mm_packs_epi32(vacc_ll, vacc_lh);
    const __m128i vacc_hi = _mm_packs_epi32(vacc_hl, vacc_hh);

    __m128i vy = _mm_packs_epi16(vacc_lo, vacc_hi);
    if (batch & (8 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(int8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) vy_lo;
    }
  }
}

void xnn_qs8_vlrelu_ukernel__sse2_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier_diff = _mm_set1_epi16(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const __m128i vmultiplier_base = _mm_set1_epi16(-params->scalar.negative_multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i vzero = _mm_setzero_si128();
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier_diff);
  XNN_FORCE_REALIZATION(vmultiplier_base);
  XNN_FORCE_REALIZATION(voutput_zero_point);

  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    const __m128i vm0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx0);
    __m128i vextx0 = _mm_unpacklo_epi8(vx0, vm0);
    __m128i vextx1 = _mm_unpackhi_epi8(vx0, vm0);
    const __m128i vm1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx1);
    __m128i vextx2 = _mm_unpacklo_epi8(vx1, vm1);
    __m128i vextx3 = _mm_unpackhi_epi8(vx1, vm1);

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vextx0, vinput_zero_point);
    vextx0 = _mm_sub_epi16(vinput_zero_point, vextx0);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vextx1, vinput_zero_point);
    vextx1 = _mm_sub_epi16(vinput_zero_point, vextx1);
    __m128i vmultiplier2 = _mm_cmpgt_epi16(vextx2, vinput_zero_point);
    vextx2 = _mm_sub_epi16(vinput_zero_point, vextx2);
    __m128i vmultiplier3 = _mm_cmpgt_epi16(vextx3, vinput_zero_point);
    vextx3 = _mm_sub_epi16(vinput_zero_point, vextx3);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);
    vmultiplier2 = _mm_and_si128(vmultiplier2, vmultiplier_diff);
    vmultiplier3 = _mm_and_si128(vmultiplier3, vmultiplier_diff);

    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);
    vmultiplier2 = _mm_xor_si128(vmultiplier2, vmultiplier_base);
    vmultiplier3 = _mm_xor_si128(vmultiplier3, vmultiplier_base);

    __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier0);
    __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier1);
    __m128i vprodlo2 = _mm_mullo_epi16(vextx2, vmultiplier2);
    __m128i vprodlo3 = _mm_mullo_epi16(vextx3, vmultiplier3);

    vprodlo0 = _mm_srli_epi16(vprodlo0, 7);
    __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vmultiplier0);
    vprodlo1 = _mm_srli_epi16(vprodlo1, 7);
    __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vmultiplier1);
    vprodlo2 = _mm_srli_epi16(vprodlo2, 7);
    __m128i vprodhi2 = _mm_mulhi_epi16(vextx2, vmultiplier2);
    vprodlo3 = _mm_srli_epi16(vprodlo3, 7);
    __m128i vprodhi3 = _mm_mulhi_epi16(vextx3, vmultiplier3);

    vprodhi0 = _mm_slli_epi16(vprodhi0, 8);
    vprodlo0 = _mm_avg_epu16(vprodlo0, vzero);
    vprodhi1 = _mm_slli_epi16(vprodhi1, 8);
    vprodlo1 = _mm_avg_epu16(vprodlo1, vzero);
    vprodhi2 = _mm_slli_epi16(vprodhi2, 8);
    vprodlo2 = _mm_avg_epu16(vprodlo2, vzero);
    vprodhi3 = _mm_slli_epi16(vprodhi3, 8);
    vprodlo3 = _mm_avg_epu16(vprodlo3, vzero);

    __m128i vacc0 = _mm_add_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_add_epi16(vprodlo1, vprodhi1);
    __m128i vacc2 = _mm_add_epi16(vprodlo2, vprodhi2);
    __m128i vacc3 = _mm_add_epi16(vprodlo3, vprodhi3);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vextx0 = _mm_unpacklo_epi8(vx, vm);
    __m128i vextx1 = _mm_unpackhi_epi8(vx, vm);

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vextx0, vinput_zero_point);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vextx1, vinput_zero_point);
    vextx0 = _mm_sub_epi16(vinput_zero_point, vextx0);
    vextx1 = _mm_sub_epi16(vinput_zero_point, vextx1);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);

    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);

    __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier0);
    __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier1);

    vprodlo0 = _mm_srli_epi16(vprodlo0, 7);
    vprodlo1 = _mm_srli_epi16(vprodlo1, 7);
    __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vmultiplier0);
    __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vmultiplier1);

    vprodhi0 = _mm_slli_epi16(vprodhi0, 8);
    vprodhi1 = _mm_slli_epi16(vprodhi1, 8);
    vprodlo0 = _mm_avg_epu16(vprodlo0, vzero);
    vprodlo1 = _mm_avg_epu16(vprodlo1, vzero);

    __m128i vacc0 = _mm_add_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_add_epi16(vprodlo1, vprodhi1);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);

    const __m128i vy = _mm_packs_epi16(vacc0, vacc1);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vextx0 = _mm_unpacklo_epi8(vx, vm);
    __m128i vextx1 = _mm_unpackhi_epi8(vx, vm);

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vextx0, vinput_zero_point);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vextx1, vinput_zero_point);
    vextx0 = _mm_sub_epi16(vinput_zero_point, vextx0);
    vextx1 = _mm_sub_epi16(vinput_zero_point, vextx1);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);

    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);

    __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier0);
    __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier1);

    vprodlo0 = _mm_srli_epi16(vprodlo0, 7);
    vprodlo1 = _mm_srli_epi16(vprodlo1, 7);
    __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vmultiplier0);
    __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vmultiplier1);

    vprodhi0 = _mm_slli_epi16(vprodhi0, 8);
    vprodhi1 = _mm_slli_epi16(vprodhi1, 8);
    vprodlo0 = _mm_avg_epu16(vprodlo0, vzero);
    vprodlo1 = _mm_avg_epu16(vprodlo1, vzero);

    __m128i vacc0 = _mm_add_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_add_epi16(vprodlo1, vprodhi1);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);

    __m128i vy = _mm_packs_epi16(vacc0, vacc1);
    if (batch & (8 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(int8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy0 = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      unaligned_store_u16(output, (uint16_t) vy0);
      vy0 >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) vy0;
    }
  }
}

void xnn_qs8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128i vb_zero_point = _mm_set1_epi16(params->scalar.b_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi16(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi16(params->scalar.output_max);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vb_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);
    input_a += 8;
    input_b += 8;

    va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);
    vb01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vb01234567, vb01234567), 8);

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

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

    vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
      __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);

      va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);
      vb01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vb01234567, vb01234567), 8);

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
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (batch & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi16(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi16(params->scalar.output_max);

  __m128i vxb = _mm_set1_epi16((UINT32_C(0x00010001) * (uint32_t) (uint16_t) (int16_t) *input_b) - params->scalar.b_zero_point);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    input_a += 8;

    va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);

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

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

    vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);

      va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);

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
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (batch & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const __m128i vzero = _mm_setzero_si128();
  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse2.init_bias);
  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);

  do {
    {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint8_t* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
      }

      int32_t* b = buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
        const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
        const __m128i vi8 = _mm_loadl_epi64((const __m128i*) i8); i8 += 8;

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
        const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);
        const __m128i vxi8 = _mm_unpacklo_epi8(vi8, vzero);

        const __m128i vsum018 = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1), vxi8);
        const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
        const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
        const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

        const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);
        const __m128i vsum01678 = _mm_add_epi16(vsum018, vsum67);
        const __m128i vsum = _mm_add_epi16(vsum2345, vsum01678);

        const __m128i vacc_lo = _mm_add_epi32(vinit_bias, _mm_unpacklo_epi16(vsum, vzero));
        const __m128i vacc_hi = _mm_add_epi32(vinit_bias, _mm_unpackhi_epi16(vsum, vzero));

        _mm_store_si128((__m128i*) b, vacc_lo);
        _mm_store_si128((__m128i*) b + 1, vacc_hi);
        b += 8;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      int32_t* b = buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
        const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
        __m128i vacc_lo = _mm_load_si128((const __m128i*) b);
        __m128i vacc_hi = _mm_load_si128((const __m128i*) b + 1);

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
        const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);

        const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);
        const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
        const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
        const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

        const __m128i vsum0123 = _mm_add_epi16(vsum01, vsum23);
        const __m128i vsum4567 = _mm_add_epi16(vsum45, vsum67);
        const __m128i vsum = _mm_add_epi16(vsum0123, vsum4567);

        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));

        _mm_store_si128((__m128i*) b, vacc_lo);
        _mm_store_si128((__m128i*) b + 1, vacc_hi);
        b += 8;
      }
    }

    {
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      const uint8_t* i1 = input[1];
      const uint8_t* i2 = input[2];
      const uint8_t* i3 = input[3];
      const uint8_t* i4 = input[4];
      const uint8_t* i5 = input[5];
      const uint8_t* i6 = input[6];
      const uint8_t* i7 = input[7];
      input = (const uint8_t**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      int32_t* b = buffer;
      while (c >= 8) {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
        const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
        __m128i vacc_lo = _mm_load_si128((const __m128i*) b);
        __m128i vacc_hi = _mm_load_si128((const __m128i*) b + 1);
        b += 8;

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
        const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);

        const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);
        const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
        const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
        const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

        const __m128i vsum0123 = _mm_add_epi16(vsum01, vsum23);
        const __m128i vsum4567 = _mm_add_epi16(vsum45, vsum67);
        const __m128i vsum = _mm_add_epi16(vsum0123, vsum4567);

        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));

        __m128 vfpacc_lo = _mm_cvtepi32_ps(vacc_lo);
        __m128 vfpacc_hi = _mm_cvtepi32_ps(vacc_hi);

        vfpacc_lo = _mm_mul_ps(vfpacc_lo, vscale);
        vfpacc_hi = _mm_mul_ps(vfpacc_hi, vscale);

        vfpacc_lo = _mm_min_ps(vfpacc_lo, voutput_max_less_zero_point);
        vfpacc_hi = _mm_min_ps(vfpacc_hi, voutput_max_less_zero_point);

        vacc_lo = _mm_cvtps_epi32(vfpacc_lo);
        vacc_hi = _mm_cvtps_epi32(vfpacc_hi);

        __m128i vout = _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), voutput_zero_point);
        vout = _mm_packus_epi16(vout, vout);
        vout = _mm_max_epu8(vout, voutput_min);

        _mm_storel_epi64((__m128i*) output, vout);
        output += 8;

        c -= 8;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7);
        __m128i vacc_lo = _mm_load_si128((const __m128i*) b);
        __m128i vacc_hi = _mm_load_si128((const __m128i*) b + 1);

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
        const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);

        const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);
        const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
        const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
        const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

        const __m128i vsum0123 = _mm_add_epi16(vsum01, vsum23);
        const __m128i vsum4567 = _mm_add_epi16(vsum45, vsum67);
        const __m128i vsum = _mm_add_epi16(vsum0123, vsum4567);

        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));

        __m128 vfpacc_lo = _mm_cvtepi32_ps(vacc_lo);
        __m128 vfpacc_hi = _mm_cvtepi32_ps(vacc_hi);

        vfpacc_lo = _mm_mul_ps(vfpacc_lo, vscale);
        vfpacc_hi = _mm_mul_ps(vfpacc_hi, vscale);

        vfpacc_lo = _mm_min_ps(vfpacc_lo, voutput_max_less_zero_point);
        vfpacc_hi = _mm_min_ps(vfpacc_hi, voutput_max_less_zero_point);

        vacc_lo = _mm_cvtps_epi32(vfpacc_lo);
        vacc_hi = _mm_cvtps_epi32(vfpacc_hi);

        __m128i vout = _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), voutput_zero_point);
        vout = _mm_packus_epi16(vout, vout);
        vout = _mm_max_epu8(vout, voutput_min);

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout));
          output += 4;
          vout = _mm_srli_epi64(vout, 32);
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout, 0));
          output += 2;
          vout = _mm_srli_epi32(vout, 16);
        }
        if (c & 1) {
          *output = (uint8_t) _mm_cvtsi128_si32(vout);
          output += 1;
        }
      }
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const __m128i vzero = _mm_setzero_si128();
  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse2.init_bias);
  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);

  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    while (c >= 8) {
      const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
      const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
      const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
      const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
      const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
      const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
      const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
      const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
      const __m128i vi8 = _mm_loadl_epi64((const __m128i*) i8); i8 += 8;

      const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
      const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
      const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
      const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
      const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
      const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
      const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
      const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);
      const __m128i vxi8 = _mm_unpacklo_epi8(vi8, vzero);

      const __m128i vsum018 = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1), vxi8);
      const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
      const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
      const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

      const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);
      const __m128i vsum01678 = _mm_add_epi16(vsum018, vsum67);
      const __m128i vsum = _mm_add_epi16(vsum2345, vsum01678);

      __m128i vacc_lo = _mm_add_epi32(vinit_bias, _mm_unpacklo_epi16(vsum, vzero));
      __m128i vacc_hi = _mm_add_epi32(vinit_bias, _mm_unpackhi_epi16(vsum, vzero));

      __m128 vfpacc_lo = _mm_cvtepi32_ps(vacc_lo);
      __m128 vfpacc_hi = _mm_cvtepi32_ps(vacc_hi);

      vfpacc_lo = _mm_mul_ps(vfpacc_lo, vscale);
      vfpacc_hi = _mm_mul_ps(vfpacc_hi, vscale);

      vfpacc_lo = _mm_min_ps(vfpacc_lo, voutput_max_less_zero_point);
      vfpacc_hi = _mm_min_ps(vfpacc_hi, voutput_max_less_zero_point);

      vacc_lo = _mm_cvtps_epi32(vfpacc_lo);
      vacc_hi = _mm_cvtps_epi32(vfpacc_hi);

      __m128i vout = _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), voutput_zero_point);
      vout = _mm_packus_epi16(vout, vout);
      vout = _mm_max_epu8(vout, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout);
      output += 8;

      c -= 8;
    }
    if (c != 0) {
      const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vi8 = _mm_loadl_epi64((const __m128i*) i8);

      const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
      const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
      const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
      const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
      const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
      const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
      const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
      const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);
      const __m128i vxi8 = _mm_unpacklo_epi8(vi8, vzero);

      const __m128i vsum018 = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1), vxi8);
      const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
      const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
      const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

      const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);
      const __m128i vsum01678 = _mm_add_epi16(vsum018, vsum67);
      const __m128i vsum = _mm_add_epi16(vsum2345, vsum01678);

      __m128i vacc_lo = _mm_add_epi32(vinit_bias, _mm_unpacklo_epi16(vsum, vzero));
      __m128i vacc_hi = _mm_add_epi32(vinit_bias, _mm_unpackhi_epi16(vsum, vzero));

      __m128 vfpacc_lo = _mm_cvtepi32_ps(vacc_lo);
      __m128 vfpacc_hi = _mm_cvtepi32_ps(vacc_hi);

      vfpacc_lo = _mm_mul_ps(vfpacc_lo, vscale);
      vfpacc_hi = _mm_mul_ps(vfpacc_hi, vscale);

      vfpacc_lo = _mm_min_ps(vfpacc_lo, voutput_max_less_zero_point);
      vfpacc_hi = _mm_min_ps(vfpacc_hi, voutput_max_less_zero_point);

      vacc_lo = _mm_cvtps_epi32(vfpacc_lo);
      vacc_hi = _mm_cvtps_epi32(vfpacc_hi);

      __m128i vout = _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), voutput_zero_point);
      vout = _mm_packus_epi16(vout, vout);
      vout = _mm_max_epu8(vout, voutput_min);

      if (c & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout));
        output += 4;
        vout = _mm_srli_epi64(vout, 32);
      }
      if (c & 2) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout, 0));
        output += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (c & 1) {
        *output = (uint8_t) _mm_cvtsi128_si32(vout);
        output += 1;
      }
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
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
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(uint8_t)));
      i0 += 8;

      const __m128i vzero = _mm_setzero_si128();
      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
      const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk0x01234567, vzero), vk_zero_point);

      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)));
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
      const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk1x01234567, vzero), vk_zero_point);

      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)));
      i2 += 8;

      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
      const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk2x01234567, vzero), vk_zero_point);

      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)));
      i3 += 8;

      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
      const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk3x01234567, vzero), vk_zero_point);

      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)));
      i4 += 8;

      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
      const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk4x01234567, vzero), vk_zero_point);

      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t)));
      i5 += 8;

      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);
      const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk5x01234567, vzero), vk_zero_point);

      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(uint8_t)));
      i6 += 8;

      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);
      const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk6x01234567, vzero), vk_zero_point);

      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(uint8_t)));
      i7 += 8;

      const __m128i vxi7x01234567 = _mm_unpacklo_epi8(vi7x01234567, vzero);
      const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk7x01234567, vzero), vk_zero_point);

      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(uint8_t)));
      i8 += 8;

      const __m128i vxi8x01234567 = _mm_unpacklo_epi8(vi8x01234567, vzero);
      const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk8x01234567, vzero), vk_zero_point);

      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(uint8_t)));
      i9 += 8;

      const __m128i vxi9x01234567 = _mm_unpacklo_epi8(vi9x01234567, vzero);
      const __m128i vxk9x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk9x01234567, vzero), vk_zero_point);

      const __m128i vprod9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
      const __m128i vprod9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod9x01234567lo, vprod9x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod9x01234567lo, vprod9x01234567hi));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(uint8_t)));
      i10 += 8;

      const __m128i vxi10x01234567 = _mm_unpacklo_epi8(vi10x01234567, vzero);
      const __m128i vxk10x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk10x01234567, vzero), vk_zero_point);

      const __m128i vprod10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      const __m128i vprod10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod10x01234567lo, vprod10x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod10x01234567lo, vprod10x01234567hi));

      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(uint8_t)));
      i11 += 8;

      const __m128i vxi11x01234567 = _mm_unpacklo_epi8(vi11x01234567, vzero);
      const __m128i vxk11x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk11x01234567, vzero), vk_zero_point);

      const __m128i vprod11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
      const __m128i vprod11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod11x01234567lo, vprod11x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod11x01234567lo, vprod11x01234567hi));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(uint8_t)));
      i12 += 8;

      const __m128i vxi12x01234567 = _mm_unpacklo_epi8(vi12x01234567, vzero);
      const __m128i vxk12x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk12x01234567, vzero), vk_zero_point);

      const __m128i vprod12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      const __m128i vprod12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod12x01234567lo, vprod12x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod12x01234567lo, vprod12x01234567hi));

      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(uint8_t)));
      i13 += 8;

      const __m128i vxi13x01234567 = _mm_unpacklo_epi8(vi13x01234567, vzero);
      const __m128i vxk13x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk13x01234567, vzero), vk_zero_point);

      const __m128i vprod13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
      const __m128i vprod13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod13x01234567lo, vprod13x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod13x01234567lo, vprod13x01234567hi));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(uint8_t)));
      i14 += 8;

      const __m128i vxi14x01234567 = _mm_unpacklo_epi8(vi14x01234567, vzero);
      const __m128i vxk14x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk14x01234567, vzero), vk_zero_point);

      const __m128i vprod14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      const __m128i vprod14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod14x01234567lo, vprod14x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod14x01234567lo, vprod14x01234567hi));

      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(uint8_t)));
      i15 += 8;

      const __m128i vxi15x01234567 = _mm_unpacklo_epi8(vi15x01234567, vzero);
      const __m128i vxk15x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk15x01234567, vzero), vk_zero_point);

      const __m128i vprod15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
      const __m128i vprod15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod15x01234567lo, vprod15x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod15x01234567lo, vprod15x01234567hi));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(uint8_t)));
      i16 += 8;

      const __m128i vxi16x01234567 = _mm_unpacklo_epi8(vi16x01234567, vzero);
      const __m128i vxk16x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk16x01234567, vzero), vk_zero_point);

      const __m128i vprod16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      const __m128i vprod16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod16x01234567lo, vprod16x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod16x01234567lo, vprod16x01234567hi));

      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(uint8_t)));
      i17 += 8;

      const __m128i vxi17x01234567 = _mm_unpacklo_epi8(vi17x01234567, vzero);
      const __m128i vxk17x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk17x01234567, vzero), vk_zero_point);

      const __m128i vprod17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
      const __m128i vprod17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod17x01234567lo, vprod17x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod17x01234567lo, vprod17x01234567hi));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(uint8_t)));
      i18 += 8;

      const __m128i vxi18x01234567 = _mm_unpacklo_epi8(vi18x01234567, vzero);
      const __m128i vxk18x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk18x01234567, vzero), vk_zero_point);

      const __m128i vprod18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      const __m128i vprod18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod18x01234567lo, vprod18x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod18x01234567lo, vprod18x01234567hi));

      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(uint8_t)));
      i19 += 8;

      const __m128i vxi19x01234567 = _mm_unpacklo_epi8(vi19x01234567, vzero);
      const __m128i vxk19x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk19x01234567, vzero), vk_zero_point);

      const __m128i vprod19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
      const __m128i vprod19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod19x01234567lo, vprod19x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod19x01234567lo, vprod19x01234567hi));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(uint8_t)));
      i20 += 8;

      const __m128i vxi20x01234567 = _mm_unpacklo_epi8(vi20x01234567, vzero);
      const __m128i vxk20x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk20x01234567, vzero), vk_zero_point);

      const __m128i vprod20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      const __m128i vprod20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod20x01234567lo, vprod20x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod20x01234567lo, vprod20x01234567hi));

      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(uint8_t)));
      i21 += 8;

      const __m128i vxi21x01234567 = _mm_unpacklo_epi8(vi21x01234567, vzero);
      const __m128i vxk21x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk21x01234567, vzero), vk_zero_point);

      const __m128i vprod21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
      const __m128i vprod21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod21x01234567lo, vprod21x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod21x01234567lo, vprod21x01234567hi));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(uint8_t)));
      i22 += 8;

      const __m128i vxi22x01234567 = _mm_unpacklo_epi8(vi22x01234567, vzero);
      const __m128i vxk22x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk22x01234567, vzero), vk_zero_point);

      const __m128i vprod22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      const __m128i vprod22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod22x01234567lo, vprod22x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod22x01234567lo, vprod22x01234567hi));

      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(uint8_t)));
      i23 += 8;

      const __m128i vxi23x01234567 = _mm_unpacklo_epi8(vi23x01234567, vzero);
      const __m128i vxk23x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk23x01234567, vzero), vk_zero_point);

      const __m128i vprod23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
      const __m128i vprod23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod23x01234567lo, vprod23x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod23x01234567lo, vprod23x01234567hi));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(uint8_t)));
      i24 += 8;

      const __m128i vxi24x01234567 = _mm_unpacklo_epi8(vi24x01234567, vzero);
      const __m128i vxk24x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk24x01234567, vzero), vk_zero_point);

      const __m128i vprod24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      const __m128i vprod24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod24x01234567lo, vprod24x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod24x01234567lo, vprod24x01234567hi));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(uint8_t));

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

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(uint8_t)));

        const __m128i vzero = _mm_setzero_si128();
        const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
        const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk0x01234567, vzero), vk_zero_point);

        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)));

        const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
        const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk1x01234567, vzero), vk_zero_point);

        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)));

        const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
        const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk2x01234567, vzero), vk_zero_point);

        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)));

        const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
        const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk3x01234567, vzero), vk_zero_point);

        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)));

        const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
        const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk4x01234567, vzero), vk_zero_point);

        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t)));

        const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);
        const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk5x01234567, vzero), vk_zero_point);

        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(uint8_t)));

        const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);
        const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk6x01234567, vzero), vk_zero_point);

        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(uint8_t)));

        const __m128i vxi7x01234567 = _mm_unpacklo_epi8(vi7x01234567, vzero);
        const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk7x01234567, vzero), vk_zero_point);

        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(uint8_t)));

        const __m128i vxi8x01234567 = _mm_unpacklo_epi8(vi8x01234567, vzero);
        const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk8x01234567, vzero), vk_zero_point);

        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(uint8_t)));

        const __m128i vxi9x01234567 = _mm_unpacklo_epi8(vi9x01234567, vzero);
        const __m128i vxk9x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk9x01234567, vzero), vk_zero_point);

        const __m128i vprod9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
        const __m128i vprod9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod9x01234567lo, vprod9x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod9x01234567lo, vprod9x01234567hi));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(uint8_t)));

        const __m128i vxi10x01234567 = _mm_unpacklo_epi8(vi10x01234567, vzero);
        const __m128i vxk10x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk10x01234567, vzero), vk_zero_point);

        const __m128i vprod10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
        const __m128i vprod10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod10x01234567lo, vprod10x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod10x01234567lo, vprod10x01234567hi));

        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(uint8_t)));

        const __m128i vxi11x01234567 = _mm_unpacklo_epi8(vi11x01234567, vzero);
        const __m128i vxk11x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk11x01234567, vzero), vk_zero_point);

        const __m128i vprod11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
        const __m128i vprod11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod11x01234567lo, vprod11x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod11x01234567lo, vprod11x01234567hi));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(uint8_t)));

        const __m128i vxi12x01234567 = _mm_unpacklo_epi8(vi12x01234567, vzero);
        const __m128i vxk12x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk12x01234567, vzero), vk_zero_point);

        const __m128i vprod12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
        const __m128i vprod12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod12x01234567lo, vprod12x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod12x01234567lo, vprod12x01234567hi));

        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(uint8_t)));

        const __m128i vxi13x01234567 = _mm_unpacklo_epi8(vi13x01234567, vzero);
        const __m128i vxk13x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk13x01234567, vzero), vk_zero_point);

        const __m128i vprod13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
        const __m128i vprod13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod13x01234567lo, vprod13x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod13x01234567lo, vprod13x01234567hi));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(uint8_t)));

        const __m128i vxi14x01234567 = _mm_unpacklo_epi8(vi14x01234567, vzero);
        const __m128i vxk14x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk14x01234567, vzero), vk_zero_point);

        const __m128i vprod14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
        const __m128i vprod14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod14x01234567lo, vprod14x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod14x01234567lo, vprod14x01234567hi));

        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(uint8_t)));

        const __m128i vxi15x01234567 = _mm_unpacklo_epi8(vi15x01234567, vzero);
        const __m128i vxk15x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk15x01234567, vzero), vk_zero_point);

        const __m128i vprod15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
        const __m128i vprod15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod15x01234567lo, vprod15x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod15x01234567lo, vprod15x01234567hi));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(uint8_t)));

        const __m128i vxi16x01234567 = _mm_unpacklo_epi8(vi16x01234567, vzero);
        const __m128i vxk16x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk16x01234567, vzero), vk_zero_point);

        const __m128i vprod16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
        const __m128i vprod16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod16x01234567lo, vprod16x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod16x01234567lo, vprod16x01234567hi));

        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(uint8_t)));

        const __m128i vxi17x01234567 = _mm_unpacklo_epi8(vi17x01234567, vzero);
        const __m128i vxk17x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk17x01234567, vzero), vk_zero_point);

        const __m128i vprod17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
        const __m128i vprod17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod17x01234567lo, vprod17x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod17x01234567lo, vprod17x01234567hi));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(uint8_t)));

        const __m128i vxi18x01234567 = _mm_unpacklo_epi8(vi18x01234567, vzero);
        const __m128i vxk18x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk18x01234567, vzero), vk_zero_point);

        const __m128i vprod18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
        const __m128i vprod18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod18x01234567lo, vprod18x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod18x01234567lo, vprod18x01234567hi));

        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(uint8_t)));

        const __m128i vxi19x01234567 = _mm_unpacklo_epi8(vi19x01234567, vzero);
        const __m128i vxk19x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk19x01234567, vzero), vk_zero_point);

        const __m128i vprod19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
        const __m128i vprod19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod19x01234567lo, vprod19x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod19x01234567lo, vprod19x01234567hi));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(uint8_t)));

        const __m128i vxi20x01234567 = _mm_unpacklo_epi8(vi20x01234567, vzero);
        const __m128i vxk20x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk20x01234567, vzero), vk_zero_point);

        const __m128i vprod20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
        const __m128i vprod20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod20x01234567lo, vprod20x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod20x01234567lo, vprod20x01234567hi));

        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(uint8_t)));

        const __m128i vxi21x01234567 = _mm_unpacklo_epi8(vi21x01234567, vzero);
        const __m128i vxk21x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk21x01234567, vzero), vk_zero_point);

        const __m128i vprod21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
        const __m128i vprod21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod21x01234567lo, vprod21x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod21x01234567lo, vprod21x01234567hi));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(uint8_t)));

        const __m128i vxi22x01234567 = _mm_unpacklo_epi8(vi22x01234567, vzero);
        const __m128i vxk22x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk22x01234567, vzero), vk_zero_point);

        const __m128i vprod22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
        const __m128i vprod22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod22x01234567lo, vprod22x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod22x01234567lo, vprod22x01234567hi));

        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(uint8_t)));

        const __m128i vxi23x01234567 = _mm_unpacklo_epi8(vi23x01234567, vzero);
        const __m128i vxk23x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk23x01234567, vzero), vk_zero_point);

        const __m128i vprod23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
        const __m128i vprod23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod23x01234567lo, vprod23x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod23x01234567lo, vprod23x01234567hi));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(uint8_t)));

        const __m128i vxi24x01234567 = _mm_unpacklo_epi8(vi24x01234567, vzero);
        const __m128i vxk24x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk24x01234567, vzero), vk_zero_point);

        const __m128i vprod24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
        const __m128i vprod24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod24x01234567lo, vprod24x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod24x01234567lo, vprod24x01234567hi));


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


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (uint8_t) _mm_cvtsi128_si32(vout0123456701234567);
          output += 1;
        }
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
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
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(uint8_t)));
      i0 += 8;

      const __m128i vzero = _mm_setzero_si128();
      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
      const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk0x01234567, vzero), vk_zero_point);

      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)));
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
      const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk1x01234567, vzero), vk_zero_point);

      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)));
      i2 += 8;

      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
      const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk2x01234567, vzero), vk_zero_point);

      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)));
      i3 += 8;

      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
      const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk3x01234567, vzero), vk_zero_point);

      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)));
      i4 += 8;

      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
      const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk4x01234567, vzero), vk_zero_point);

      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t)));
      i5 += 8;

      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);
      const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk5x01234567, vzero), vk_zero_point);

      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(uint8_t)));
      i6 += 8;

      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);
      const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk6x01234567, vzero), vk_zero_point);

      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(uint8_t)));
      i7 += 8;

      const __m128i vxi7x01234567 = _mm_unpacklo_epi8(vi7x01234567, vzero);
      const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk7x01234567, vzero), vk_zero_point);

      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(uint8_t)));
      i8 += 8;

      const __m128i vxi8x01234567 = _mm_unpacklo_epi8(vi8x01234567, vzero);
      const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk8x01234567, vzero), vk_zero_point);

      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(uint8_t));

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

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(uint8_t)));

        const __m128i vzero = _mm_setzero_si128();
        const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
        const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk0x01234567, vzero), vk_zero_point);

        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)));

        const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
        const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk1x01234567, vzero), vk_zero_point);

        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)));

        const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
        const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk2x01234567, vzero), vk_zero_point);

        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)));

        const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
        const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk3x01234567, vzero), vk_zero_point);

        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)));

        const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
        const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk4x01234567, vzero), vk_zero_point);

        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t)));

        const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);
        const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk5x01234567, vzero), vk_zero_point);

        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(uint8_t)));

        const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);
        const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk6x01234567, vzero), vk_zero_point);

        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(uint8_t)));

        const __m128i vxi7x01234567 = _mm_unpacklo_epi8(vi7x01234567, vzero);
        const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk7x01234567, vzero), vk_zero_point);

        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(uint8_t)));

        const __m128i vxi8x01234567 = _mm_unpacklo_epi8(vi8x01234567, vzero);
        const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_unpacklo_epi8(vk8x01234567, vzero), vk_zero_point);

        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));


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


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (uint8_t) _mm_cvtsi128_si32(vout0123456701234567);
          output += 1;
        }
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_f32_vcvt_ukernel__sse2_u32(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmagic_exp = _mm_set1_epi16(UINT16_C(0x4B00));
  const __m128 vmagic_bias = _mm_set1_ps((float) (INT32_C(0x00800000) + (int32_t) params->scalar.zero_point));
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i vzero = _mm_setzero_si128();
  XNN_FORCE_REALIZATION(vmagic_exp);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    __m128i vx01234567 = _mm_loadl_epi64((const __m128i*) input);
    __m128i vx89ABCDEF = _mm_loadl_epi64((const __m128i*) (input + 8));
    __m128i vxGHIJKLMN = _mm_loadl_epi64((const __m128i*) (input + 16));
    __m128i vxOPQRSTUV = _mm_loadl_epi64((const __m128i*) (input + 24));
    input += 32;


    vx01234567 = _mm_unpacklo_epi8(vx01234567, vzero);
    vx89ABCDEF = _mm_unpacklo_epi8(vx89ABCDEF, vzero);
    vxGHIJKLMN = _mm_unpacklo_epi8(vxGHIJKLMN, vzero);
    vxOPQRSTUV = _mm_unpacklo_epi8(vxOPQRSTUV, vzero);

    __m128 vy0123 = _mm_castsi128_ps(_mm_unpacklo_epi16(vx01234567, vmagic_exp));
    __m128 vy4567 = _mm_castsi128_ps(_mm_unpackhi_epi16(vx01234567, vmagic_exp));
    __m128 vy89AB = _mm_castsi128_ps(_mm_unpacklo_epi16(vx89ABCDEF, vmagic_exp));
    __m128 vyCDEF = _mm_castsi128_ps(_mm_unpackhi_epi16(vx89ABCDEF, vmagic_exp));
    __m128 vyGHIJ = _mm_castsi128_ps(_mm_unpacklo_epi16(vxGHIJKLMN, vmagic_exp));
    __m128 vyKLMN = _mm_castsi128_ps(_mm_unpackhi_epi16(vxGHIJKLMN, vmagic_exp));
    __m128 vyOPQR = _mm_castsi128_ps(_mm_unpacklo_epi16(vxOPQRSTUV, vmagic_exp));
    __m128 vySTUV = _mm_castsi128_ps(_mm_unpackhi_epi16(vxOPQRSTUV, vmagic_exp));

    vy0123 = _mm_sub_ps(vy0123, vmagic_bias);
    vy4567 = _mm_sub_ps(vy4567, vmagic_bias);
    vy89AB = _mm_sub_ps(vy89AB, vmagic_bias);
    vyCDEF = _mm_sub_ps(vyCDEF, vmagic_bias);
    vyGHIJ = _mm_sub_ps(vyGHIJ, vmagic_bias);
    vyKLMN = _mm_sub_ps(vyKLMN, vmagic_bias);
    vyOPQR = _mm_sub_ps(vyOPQR, vmagic_bias);
    vySTUV = _mm_sub_ps(vySTUV, vmagic_bias);

    vy0123 = _mm_mul_ps(vy0123, vscale);
    vy4567 = _mm_mul_ps(vy4567, vscale);
    vy89AB = _mm_mul_ps(vy89AB, vscale);
    vyCDEF = _mm_mul_ps(vyCDEF, vscale);
    vyGHIJ = _mm_mul_ps(vyGHIJ, vscale);
    vyKLMN = _mm_mul_ps(vyKLMN, vscale);
    vyOPQR = _mm_mul_ps(vyOPQR, vscale);
    vySTUV = _mm_mul_ps(vySTUV, vscale);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    _mm_storeu_ps(output + 16, vyGHIJ);
    _mm_storeu_ps(output + 20, vyKLMN);
    _mm_storeu_ps(output + 24, vyOPQR);
    _mm_storeu_ps(output + 28, vySTUV);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i vx = _mm_loadl_epi64((const __m128i*) input);
    vx = _mm_unpacklo_epi8(vx, vzero);
    input += 8;

    __m128 vy_lo = _mm_castsi128_ps(_mm_unpacklo_epi16(vx, vmagic_exp));
    __m128 vy_hi = _mm_castsi128_ps(_mm_unpackhi_epi16(vx, vmagic_exp));

    vy_lo = _mm_sub_ps(vy_lo, vmagic_bias);
    vy_hi = _mm_sub_ps(vy_hi, vmagic_bias);

    vy_lo = _mm_mul_ps(vy_lo, vscale);
    vy_hi = _mm_mul_ps(vy_hi, vscale);

    _mm_storeu_ps(output, vy_lo);
    _mm_storeu_ps(output + 4, vy_hi);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    __m128i vx = _mm_loadl_epi64((const __m128i*) input);
    vx = _mm_unpacklo_epi8(vx, vzero);

    __m128 vy = _mm_castsi128_ps(_mm_unpacklo_epi16(vx, vmagic_exp));
    vy = _mm_sub_ps(vy, vmagic_bias);
    vy = _mm_mul_ps(vy, vscale);

    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_ps(output, vy);
      vy = _mm_castsi128_ps(_mm_unpackhi_epi16(vx, vmagic_exp));
      vy = _mm_sub_ps(vy, vmagic_bias);
      vy = _mm_mul_ps(vy, vscale);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(uint8_t);

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse2.init_bias);
  const __m128i vzero = _mm_setzero_si128();
  int32_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;

    const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;

    const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vzero);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    _mm_store_si128((__m128i*) b, vacc0123);
    _mm_store_si128((__m128i*) (b + 4), vacc4567);
    b += 8;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {

      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;

      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vzero);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) b));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (b + 4)));

      _mm_store_si128((__m128i*) b, vacc0123);
      _mm_store_si128((__m128i*) (b + 4), vacc4567);
      b += 8;
    }
  }

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
  for (; channels >= 8; channels -= 8) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;

    const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;

    const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vzero);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

    vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
    buffer += 8;

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {

      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;

      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vzero);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
      buffer += 8;

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      uint32_t vout0123 = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) vout0123);
        vout0123 >>= 16;
        output += 2;
      }
      if (channels & 1) {
        *output = (uint8_t) vout0123;
      }
    }
  }
}

void xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse2.init_bias);
  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
  const __m128i vzero = _mm_setzero_si128();
  for (; channels >= 8; channels -= 8) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;

    const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;

    const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vzero);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {

      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, vzero);
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;

      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, vzero);
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, vzero);
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, vzero);
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, vzero);
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, vzero);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, vzero);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vzero);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

      vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
      vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      uint32_t vout0123 = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) vout0123);
        vout0123 >>= 16;
        output += 2;
      }
      if (channels & 1) {
        *output = (uint8_t) vout0123;
      }
    }
  }
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64(
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

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    const __m128i vzero = _mm_setzero_si128();
    size_t k = kc;


    while (k >= 8 * sizeof(uint8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_unpacklo_epi8(va0, vzero);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));

      const __m128i vxb1 = _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));

      const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));

      const __m128i vxb3 = _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const uint8_t*) w + 32;
      k -= 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

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
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_cvtsi128_si32(vout);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64(
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
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
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


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    __m128i vacc2x0 = vacc0x0;
    __m128i vacc2x1 = vacc0x1;
    __m128i vacc2x2 = vacc0x2;
    __m128i vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    const __m128i vzero = _mm_setzero_si128();
    size_t k = kc;


    while (k >= 8 * sizeof(uint8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_unpacklo_epi8(va0, vzero);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_unpacklo_epi8(va1, vzero);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_unpacklo_epi8(va2, vzero);
      a2 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));

      const __m128i vxb1 = _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));

      const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));

      const __m128i vxb3 = _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

      w = (const uint8_t*) w + 32;
      k -= 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vscaled2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);
    vscaled2x0123 = _mm_mul_ps(vscaled2x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm_min_ps(vscaled2x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm_cvtps_epi32(vscaled2x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vacc22x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc2x0123, vacc2x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc22x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      vout = _mm_srli_si128(vout, 4);
      unaligned_store_u32(c1, (uint32_t) _mm_cvtsi128_si32(vout));
      vout = _mm_srli_si128(vout, 4);
      unaligned_store_u32(c2, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout, 4));
        c2 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_cvtsi128_si32(vout);
        *c1 = (uint8_t) _mm_extract_epi16(vout, 2);
        *c2 = (uint8_t) _mm_extract_epi16(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64(
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

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  uint8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

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
        const __m128i vxa0 = _mm_unpacklo_epi8(va0, vzero);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));
        const __m128i vxb1 = _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));
        const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));
        const __m128i vxb3 = _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const uint8_t*) w + 32);
        k += 8 * sizeof(uint8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));

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
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_cvtsi128_si32(vout);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64(
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
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    __m128i vacc2x0 = vacc0x0;
    __m128i vacc2x1 = vacc0x1;
    __m128i vacc2x2 = vacc0x2;
    __m128i vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

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
      a += 3;

      size_t k = 0;
      const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
      const __m128i vzero = _mm_setzero_si128();
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_unpacklo_epi8(va0, vzero);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_unpacklo_epi8(va1, vzero);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_unpacklo_epi8(va2, vzero);
        a2 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));
        const __m128i vxb1 = _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));
        const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));
        const __m128i vxb3 = _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

        w = (const void*) ((const uint8_t*) w + 32);
        k += 8 * sizeof(uint8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));
    const __m128i vacc2x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x0, vacc2x2), _mm_unpackhi_epi32(vacc2x0, vacc2x2));
    const __m128i vacc2x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x1, vacc2x3), _mm_unpackhi_epi32(vacc2x1, vacc2x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));
    __m128i vacc2x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc2x02, vacc2x13), _mm_unpackhi_epi32(vacc2x02, vacc2x13));

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vscaled2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);
    vscaled2x0123 = _mm_mul_ps(vscaled2x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm_min_ps(vscaled2x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm_cvtps_epi32(vscaled2x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vacc22x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc2x0123, vacc2x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc22x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c2, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(vout, _MM_SHUFFLE(2, 2, 2, 2))));
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      unaligned_store_u32(c1, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(vout, _MM_SHUFFLE(1, 1, 1, 1))));
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout, 4));
        c2 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c2 = (uint8_t) _mm_extract_epi16(vout, 4);
        *c1 = (uint8_t) _mm_extract_epi16(vout, 2);
        *c0 = (uint8_t) _mm_cvtsi128_si32(vout);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vadd_minmax_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.bias);
  const __m128i va_multiplier_lo = _mm_set1_epi16(params->scalar.a_multiplier);
  const __m128i va_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.a_multiplier >> 16);
  const __m128i vb_multiplier_lo = _mm_set1_epi16(params->scalar.b_multiplier);
  const __m128i vb_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.b_multiplier >> 16);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier_lo);
  XNN_FORCE_REALIZATION(va_multiplier_hi);
  XNN_FORCE_REALIZATION(vb_multiplier_lo);
  XNN_FORCE_REALIZATION(vb_multiplier_hi);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);
    input_a += 8;
    input_b += 8;

    const __m128i vzero = _mm_setzero_si128();
    va01234567 = _mm_unpacklo_epi8(va01234567, vzero);
    vb01234567 = _mm_unpacklo_epi8(vb01234567, vzero);

    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
    const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
    vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));


    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
      __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);

      const __m128i vzero = _mm_setzero_si128();
      va01234567 = _mm_unpacklo_epi8(va01234567, vzero);
      vb01234567 = _mm_unpacklo_epi8(vb01234567, vzero);

      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
      const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
      vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));


      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.b_multiplier * (int32_t) *input_b + params->scalar.bias);
  const __m128i va_multiplier_lo = _mm_set1_epi16(params->scalar.a_multiplier);
  const __m128i va_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.a_multiplier >> 16);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier_lo);
  XNN_FORCE_REALIZATION(va_multiplier_hi);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    input_a += 8;

    const __m128i vzero = _mm_setzero_si128();
    va01234567 = _mm_unpacklo_epi8(va01234567, vzero);

    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));


    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);

      va01234567 = _mm_unpacklo_epi8(va01234567, _mm_setzero_si128());

      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));


      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_qu8_vcvt_ukernel__sse2_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vmultiplier = _mm_set1_epi16(params->scalar.multiplier);
  const __m128i vbias = _mm_set1_epi32(
      ((int32_t) params->scalar.output_zero_point << 8) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point + 
      INT32_C(0x80));
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  const __m128i vzero = _mm_setzero_si128();
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    const __m128i vextx0 = _mm_unpacklo_epi8(vx0, vzero);
    const __m128i vextx1 = _mm_unpackhi_epi8(vx0, vzero);
    const __m128i vextx2 = _mm_unpacklo_epi8(vx1, vzero);
    const __m128i vextx3 = _mm_unpackhi_epi8(vx1, vzero);

    const __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier);
    const __m128i vprodhi0 = _mm_mulhi_epu16(vextx0, vmultiplier);
    const __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier);
    const __m128i vprodhi1 = _mm_mulhi_epu16(vextx1, vmultiplier);
    const __m128i vprodlo2 = _mm_mullo_epi16(vextx2, vmultiplier);
    const __m128i vprodhi2 = _mm_mulhi_epu16(vextx2, vmultiplier);
    const __m128i vprodlo3 = _mm_mullo_epi16(vextx3, vmultiplier);
    const __m128i vprodhi3 = _mm_mulhi_epu16(vextx3, vmultiplier);

    __m128i vacc0 = _mm_unpacklo_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_unpackhi_epi16(vprodlo0, vprodhi0);
    __m128i vacc2 = _mm_unpacklo_epi16(vprodlo1, vprodhi1);
    __m128i vacc3 = _mm_unpackhi_epi16(vprodlo1, vprodhi1);
    __m128i vacc4 = _mm_unpacklo_epi16(vprodlo2, vprodhi2);
    __m128i vacc5 = _mm_unpackhi_epi16(vprodlo2, vprodhi2);
    __m128i vacc6 = _mm_unpacklo_epi16(vprodlo3, vprodhi3);
    __m128i vacc7 = _mm_unpackhi_epi16(vprodlo3, vprodhi3);

    vacc0 = _mm_add_epi32(vacc0, vbias);
    vacc1 = _mm_add_epi32(vacc1, vbias);
    vacc2 = _mm_add_epi32(vacc2, vbias);
    vacc3 = _mm_add_epi32(vacc3, vbias);
    vacc4 = _mm_add_epi32(vacc4, vbias);
    vacc5 = _mm_add_epi32(vacc5, vbias);
    vacc6 = _mm_add_epi32(vacc6, vbias);
    vacc7 = _mm_add_epi32(vacc7, vbias);

    vacc0 = _mm_srai_epi32(vacc0, 8);
    vacc1 = _mm_srai_epi32(vacc1, 8);
    vacc2 = _mm_srai_epi32(vacc2, 8);
    vacc3 = _mm_srai_epi32(vacc3, 8);
    vacc4 = _mm_srai_epi32(vacc4, 8);
    vacc5 = _mm_srai_epi32(vacc5, 8);
    vacc6 = _mm_srai_epi32(vacc6, 8);
    vacc7 = _mm_srai_epi32(vacc7, 8);

    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc1 = _mm_packs_epi32(vacc2, vacc3);
    vacc2 = _mm_packs_epi32(vacc4, vacc5);
    vacc3 = _mm_packs_epi32(vacc6, vacc7);

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packus_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vextx_lo = _mm_unpacklo_epi8(vx, vzero);
    const __m128i vextx_hi = _mm_unpackhi_epi8(vx, vzero);

    const __m128i vprodlo_lo = _mm_mullo_epi16(vextx_lo, vmultiplier);
    const __m128i vprodlo_hi = _mm_mullo_epi16(vextx_hi, vmultiplier);
    const __m128i vprodhi_lo = _mm_mulhi_epu16(vextx_lo, vmultiplier);
    const __m128i vprodhi_hi = _mm_mulhi_epu16(vextx_hi, vmultiplier);

    __m128i vacc_ll = _mm_unpacklo_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_lh = _mm_unpackhi_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_hl = _mm_unpacklo_epi16(vprodlo_hi, vprodhi_hi);
    __m128i vacc_hh = _mm_unpackhi_epi16(vprodlo_hi, vprodhi_hi);

    vacc_ll = _mm_add_epi32(vacc_ll, vbias);
    vacc_lh = _mm_add_epi32(vacc_lh, vbias);
    vacc_hl = _mm_add_epi32(vacc_hl, vbias);
    vacc_hh = _mm_add_epi32(vacc_hh, vbias);

    vacc_ll = _mm_srai_epi32(vacc_ll, 8);
    vacc_lh = _mm_srai_epi32(vacc_lh, 8);
    vacc_hl = _mm_srai_epi32(vacc_hl, 8);
    vacc_hh = _mm_srai_epi32(vacc_hh, 8);

    const __m128i vacc_lo = _mm_packs_epi32(vacc_ll, vacc_lh);
    const __m128i vacc_hi = _mm_packs_epi32(vacc_hl, vacc_hh);

    const __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    const __m128i vextx_lo = _mm_unpacklo_epi8(vx, vzero);
    const __m128i vextx_hi = _mm_unpackhi_epi8(vx, vzero);

    const __m128i vprodlo_lo = _mm_mullo_epi16(vextx_lo, vmultiplier);
    const __m128i vprodlo_hi = _mm_mullo_epi16(vextx_hi, vmultiplier);
    const __m128i vprodhi_lo = _mm_mulhi_epu16(vextx_lo, vmultiplier);
    const __m128i vprodhi_hi = _mm_mulhi_epu16(vextx_hi, vmultiplier);

    __m128i vacc_ll = _mm_unpacklo_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_lh = _mm_unpackhi_epi16(vprodlo_lo, vprodhi_lo);
    __m128i vacc_hl = _mm_unpacklo_epi16(vprodlo_hi, vprodhi_hi);
    __m128i vacc_hh = _mm_unpackhi_epi16(vprodlo_hi, vprodhi_hi);

    vacc_ll = _mm_add_epi32(vacc_ll, vbias);
    vacc_lh = _mm_add_epi32(vacc_lh, vbias);
    vacc_hl = _mm_add_epi32(vacc_hl, vbias);
    vacc_hh = _mm_add_epi32(vacc_hh, vbias);

    vacc_ll = _mm_srai_epi32(vacc_ll, 8);
    vacc_lh = _mm_srai_epi32(vacc_lh, 8);
    vacc_hl = _mm_srai_epi32(vacc_hl, 8);
    vacc_hh = _mm_srai_epi32(vacc_hh, 8);

    const __m128i vacc_lo = _mm_packs_epi32(vacc_ll, vacc_lh);
    const __m128i vacc_hi = _mm_packs_epi32(vacc_hl, vacc_hh);

    __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) vy_lo;
    }
  }
}

void xnn_qu8_vlrelu_ukernel__sse2_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier_diff = _mm_set1_epi16(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const __m128i vmultiplier_base = _mm_set1_epi16(-params->scalar.negative_multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i vzero = _mm_setzero_si128();
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier_diff);
  XNN_FORCE_REALIZATION(vmultiplier_base);
  XNN_FORCE_REALIZATION(voutput_zero_point);

  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    __m128i vextx0 = _mm_unpacklo_epi8(vx0, vzero);
    __m128i vextx1 = _mm_unpackhi_epi8(vx0, vzero);
    __m128i vextx2 = _mm_unpacklo_epi8(vx1, vzero);
    __m128i vextx3 = _mm_unpackhi_epi8(vx1, vzero);

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vextx0, vinput_zero_point);
    vextx0 = _mm_sub_epi16(vinput_zero_point, vextx0);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vextx1, vinput_zero_point);
    vextx1 = _mm_sub_epi16(vinput_zero_point, vextx1);
    __m128i vmultiplier2 = _mm_cmpgt_epi16(vextx2, vinput_zero_point);
    vextx2 = _mm_sub_epi16(vinput_zero_point, vextx2);
    __m128i vmultiplier3 = _mm_cmpgt_epi16(vextx3, vinput_zero_point);
    vextx3 = _mm_sub_epi16(vinput_zero_point, vextx3);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);
    vmultiplier2 = _mm_and_si128(vmultiplier2, vmultiplier_diff);
    vmultiplier3 = _mm_and_si128(vmultiplier3, vmultiplier_diff);

    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);
    vmultiplier2 = _mm_xor_si128(vmultiplier2, vmultiplier_base);
    vmultiplier3 = _mm_xor_si128(vmultiplier3, vmultiplier_base);

    __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier0);
    __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier1);
    __m128i vprodlo2 = _mm_mullo_epi16(vextx2, vmultiplier2);
    __m128i vprodlo3 = _mm_mullo_epi16(vextx3, vmultiplier3);

    vprodlo0 = _mm_srli_epi16(vprodlo0, 7);
    __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vmultiplier0);
    vprodlo1 = _mm_srli_epi16(vprodlo1, 7);
    __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vmultiplier1);
    vprodlo2 = _mm_srli_epi16(vprodlo2, 7);
    __m128i vprodhi2 = _mm_mulhi_epi16(vextx2, vmultiplier2);
    vprodlo3 = _mm_srli_epi16(vprodlo3, 7);
    __m128i vprodhi3 = _mm_mulhi_epi16(vextx3, vmultiplier3);

    vprodhi0 = _mm_slli_epi16(vprodhi0, 8);
    vprodlo0 = _mm_avg_epu16(vprodlo0, vzero);
    vprodhi1 = _mm_slli_epi16(vprodhi1, 8);
    vprodlo1 = _mm_avg_epu16(vprodlo1, vzero);
    vprodhi2 = _mm_slli_epi16(vprodhi2, 8);
    vprodlo2 = _mm_avg_epu16(vprodlo2, vzero);
    vprodhi3 = _mm_slli_epi16(vprodhi3, 8);
    vprodlo3 = _mm_avg_epu16(vprodlo3, vzero);

    __m128i vacc0 = _mm_add_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_add_epi16(vprodlo1, vprodhi1);
    __m128i vacc2 = _mm_add_epi16(vprodlo2, vprodhi2);
    __m128i vacc3 = _mm_add_epi16(vprodlo3, vprodhi3);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packus_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    __m128i vextx0 = _mm_unpacklo_epi8(vx, vzero);
    __m128i vextx1 = _mm_unpackhi_epi8(vx, vzero);

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vextx0, vinput_zero_point);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vextx1, vinput_zero_point);
    vextx0 = _mm_sub_epi16(vinput_zero_point, vextx0);
    vextx1 = _mm_sub_epi16(vinput_zero_point, vextx1);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);

    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);

    __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier0);
    __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier1);

    vprodlo0 = _mm_srli_epi16(vprodlo0, 7);
    vprodlo1 = _mm_srli_epi16(vprodlo1, 7);
    __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vmultiplier0);
    __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vmultiplier1);

    vprodhi0 = _mm_slli_epi16(vprodhi0, 8);
    vprodhi1 = _mm_slli_epi16(vprodhi1, 8);
    vprodlo0 = _mm_avg_epu16(vprodlo0, vzero);
    vprodlo1 = _mm_avg_epu16(vprodlo1, vzero);

    __m128i vacc0 = _mm_add_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_add_epi16(vprodlo1, vprodhi1);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);

    const __m128i vy = _mm_packus_epi16(vacc0, vacc1);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    __m128i vextx0 = _mm_unpacklo_epi8(vx, vzero);
    __m128i vextx1 = _mm_unpackhi_epi8(vx, vzero);

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vextx0, vinput_zero_point);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vextx1, vinput_zero_point);
    vextx0 = _mm_sub_epi16(vinput_zero_point, vextx0);
    vextx1 = _mm_sub_epi16(vinput_zero_point, vextx1);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);

    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);

    __m128i vprodlo0 = _mm_mullo_epi16(vextx0, vmultiplier0);
    __m128i vprodlo1 = _mm_mullo_epi16(vextx1, vmultiplier1);

    vprodlo0 = _mm_srli_epi16(vprodlo0, 7);
    vprodlo1 = _mm_srli_epi16(vprodlo1, 7);
    __m128i vprodhi0 = _mm_mulhi_epi16(vextx0, vmultiplier0);
    __m128i vprodhi1 = _mm_mulhi_epi16(vextx1, vmultiplier1);

    vprodhi0 = _mm_slli_epi16(vprodhi0, 8);
    vprodhi1 = _mm_slli_epi16(vprodhi1, 8);
    vprodlo0 = _mm_avg_epu16(vprodlo0, vzero);
    vprodlo1 = _mm_avg_epu16(vprodlo1, vzero);

    __m128i vacc0 = _mm_add_epi16(vprodlo0, vprodhi0);
    __m128i vacc1 = _mm_add_epi16(vprodlo1, vprodhi1);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);

    __m128i vy = _mm_packus_epi16(vacc0, vacc1);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy0 = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      unaligned_store_u16(output, (uint16_t) vy0);
      vy0 >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) vy0;
    }
  }
}

void xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128i vb_zero_point = _mm_set1_epi16(params->scalar.b_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vb_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);
    input_a += 8;
    input_b += 8;

    const __m128i vzero = _mm_setzero_si128();
    va01234567 = _mm_unpacklo_epi8(va01234567, vzero);
    vb01234567 = _mm_unpacklo_epi8(vb01234567, vzero);

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

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
      __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);

      const __m128i vzero = _mm_setzero_si128();
      va01234567 = _mm_unpacklo_epi8(va01234567, vzero);
      vb01234567 = _mm_unpacklo_epi8(vb01234567, vzero);

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

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  __m128i vxb = _mm_set1_epi16((UINT32_C(0x00010001) * (uint32_t) (uint16_t) (int16_t) *input_b) - params->scalar.b_zero_point);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    input_a += 8;

    const __m128i vzero = _mm_setzero_si128();
    va01234567 = _mm_unpacklo_epi8(va01234567, vzero);

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

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);

      const __m128i vzero = _mm_setzero_si128();
      va01234567 = _mm_unpacklo_epi8(va01234567, vzero);

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

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_s8_ibilinear_ukernel__sse2_c8(
    size_t output_pixels,
    size_t channels,
    const int8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    int8_t* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const int8_t* i0 = (const int8_t*) ((uintptr_t) input[0] + input_offset);
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input[1] + input_offset);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input[2] + input_offset);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const __m128i valpha = _mm_cvtsi32_si128(*((const int*) weights));
    weights += 2;
    __m128i valphah = _mm_shufflelo_epi16(valpha, _MM_SHUFFLE(0, 0, 0, 0));
    valphah = _mm_unpacklo_epi64(valphah, valphah);
    __m128i valphav = _mm_shufflelo_epi16(valpha, _MM_SHUFFLE(1, 1, 1, 1));
    valphav = _mm_unpacklo_epi64(valphav, valphav);

    valphah = _mm_xor_si128(valphah, _mm_set1_epi32(0xFFFF0000));
    valphah = _mm_add_epi16(valphah, _mm_set1_epi32(0x08010000));

    const __m128i vrounding = _mm_set1_epi32(0x00200000);

    size_t c = channels;
    for (; c >= 8 * sizeof(int8_t); c -= 8 * sizeof(int8_t)) {
      __m128i vtl01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;
      __m128i vtr01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;
      __m128i vbl01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;
      __m128i vbr01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      vtl01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vtl01234567, vtl01234567), 8);
      vtr01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vtr01234567, vtr01234567), 8);
      vbl01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vbl01234567, vbl01234567), 8);
      vbr01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vbr01234567, vbr01234567), 8);

      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_slli_epi32(_mm_mulhi_epu16(vd0123, valphav), 16);
      __m128i vacc4567 = _mm_slli_epi32(_mm_mulhi_epu16(vd4567, valphav), 16);

      vacc0123 = _mm_add_epi16(_mm_mullo_epi16(vd0123, valphav), vacc0123);
      vacc4567 = _mm_add_epi16(_mm_mullo_epi16(vd4567, valphav), vacc4567);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srai_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srai_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      const __m128i vo01234567 = _mm_packs_epi16(vacc01234567, vacc01234567);

      _mm_storel_epi64((__m128i*) output, vo01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128i vtl01234567 = _mm_loadl_epi64((const __m128i*) i0);
      __m128i vtr01234567 = _mm_loadl_epi64((const __m128i*) i1);
      __m128i vbl01234567 = _mm_loadl_epi64((const __m128i*) i2);
      __m128i vbr01234567 = _mm_loadl_epi64((const __m128i*) i3);

      vtl01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vtl01234567, vtl01234567), 8);
      vtr01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vtr01234567, vtr01234567), 8);
      vbl01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vbl01234567, vbl01234567), 8);
      vbr01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vbr01234567, vbr01234567), 8);

      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_slli_epi32(_mm_mulhi_epu16(vd0123, valphav), 16);
      __m128i vacc4567 = _mm_slli_epi32(_mm_mulhi_epu16(vd4567, valphav), 16);

      vacc0123 = _mm_add_epi16(_mm_mullo_epi16(vd0123, valphav), vacc0123);
      vacc4567 = _mm_add_epi16(_mm_mullo_epi16(vd4567, valphav), vacc4567);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srai_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srai_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      __m128i vo01234567 = _mm_packs_epi16(vacc01234567, vacc01234567);

      if (c & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vo01234567));
        output += 4;
        vo01234567 = _mm_srli_epi64(vo01234567, 32);
      }
      uint32_t vo0123 = (uint32_t) _mm_cvtsi128_si32(vo01234567);
      if (c & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) vo0123);
        output += 2;
        vo0123 >>= 16;
      }
      if (c & (1 * sizeof(int8_t))) {
        *output++ = (uint8_t) vo0123;
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_s8_maxpool_minmax_ukernel_9p8x__sse2_c16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const __m128i vbias = _mm_set1_epi8(UINT8_C(0x80));
  const __m128i voutput_max_with_bias = _mm_set1_epi8(UINT8_C(0x80) ^ params->scalar.max);
  const __m128i voutput_min_with_bias = _mm_set1_epi8(UINT8_C(0x80) ^ params->scalar.min);
  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(voutput_max_with_bias);
  XNN_FORCE_REALIZATION(voutput_min_with_bias);

  do {
    int8_t* o = output;
    {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      const int8_t* i8 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
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
      for (; c >= 16; c -= 16) {
        const __m128i vi0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i0), vbias);
        i0 += 16;
        const __m128i vi1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i1), vbias);
        i1 += 16;
        const __m128i vi2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i2), vbias);
        i2 += 16;
        const __m128i vi3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i3), vbias);
        i3 += 16;
        const __m128i vi4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i4), vbias);
        i4 += 16;
        const __m128i vi5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i5), vbias);
        i5 += 16;
        const __m128i vi6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i6), vbias);
        i6 += 16;
        const __m128i vi7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i7), vbias);
        i7 += 16;
        const __m128i vi8 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i8), vbias);
        i8 += 16;

        const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax01678);
        vout = _mm_max_epu8(vout, voutput_min_with_bias);
        vout = _mm_min_epu8(vout, voutput_max_with_bias);
        vout = _mm_xor_si128(vout, vbias);

        _mm_storeu_si128((__m128i*) o, vout); o += 16;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i0), vbias);
        const __m128i vi1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i1), vbias);
        const __m128i vi2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i2), vbias);
        const __m128i vi3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i3), vbias);
        const __m128i vi4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i4), vbias);
        const __m128i vi5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i5), vbias);
        const __m128i vi6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i6), vbias);
        const __m128i vi7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i7), vbias);
        const __m128i vi8 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i8), vbias);

        const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax01678);
        vout = _mm_max_epu8(vout, voutput_min_with_bias);
        vout = _mm_min_epu8(vout, voutput_max_with_bias);
        vout = _mm_xor_si128(vout, vbias);

        if (c & 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          vout = _mm_unpackhi_epi64(vout, vout);
          o += 8;
        }
        if (c & 4) {
          unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
          vout = _mm_srli_epi64(vout, 32);
          o += 4;
        }
        if (c & 2) {
          unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
          vout = _mm_srli_epi32(vout, 16);
          o += 2;
        }
        if (c & 1) {
          *((int8_t*) o) = (int8_t) _mm_cvtsi128_si32(vout);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
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
      for (; c >= 16; c -= 16) {
        const __m128i vi0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i0), vbias);
        i0 += 16;
        const __m128i vi1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i1), vbias);
        i1 += 16;
        const __m128i vi2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i2), vbias);
        i2 += 16;
        const __m128i vi3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i3), vbias);
        i3 += 16;
        const __m128i vi4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i4), vbias);
        i4 += 16;
        const __m128i vi5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i5), vbias);
        i5 += 16;
        const __m128i vi6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i6), vbias);
        i6 += 16;
        const __m128i vi7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i7), vbias);
        i7 += 16;
        const __m128i vo = _mm_xor_si128(_mm_loadu_si128((const __m128i*) o), vbias);

        const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax0167);
        vout = _mm_max_epu8(vout, voutput_min_with_bias);
        vout = _mm_min_epu8(vout, voutput_max_with_bias);
        vout = _mm_xor_si128(vout, vbias);

        _mm_storeu_si128((__m128i*) o, vout);
        o += 16;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i0), vbias);
        const __m128i vi1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i1), vbias);
        const __m128i vi2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i2), vbias);
        const __m128i vi3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i3), vbias);
        const __m128i vi4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i4), vbias);
        const __m128i vi5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i5), vbias);
        const __m128i vi6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i6), vbias);
        const __m128i vi7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i7), vbias);
        const __m128i vo = _mm_xor_si128(_mm_loadu_si128((const __m128i*) o), vbias);

        const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax0167);
        vout = _mm_max_epu8(vout, voutput_min_with_bias);
        vout = _mm_min_epu8(vout, voutput_max_with_bias);
        vout = _mm_xor_si128(vout, vbias);

        if (c & 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          vout = _mm_unpackhi_epi64(vout, vout);
          o += 8;
        }
        if (c & 4) {
          unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
          vout = _mm_srli_epi64(vout, 32);
          o += 4;
        }
        if (c & 2) {
          unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
          vout = _mm_srli_epi32(vout, 16);
          o += 2;
        }
        if (c & 1) {
          *o = (int8_t) _mm_cvtsi128_si32(vout);
          o += 1;
        }
      }
    }
    input = (const int8_t**) ((uintptr_t) input + input_increment);
    output = (int8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_s8_vclamp_ukernel__sse2_u64(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi8(UINT8_C(0x80));
  const __m128i voutput_max_with_bias = _mm_set1_epi8(UINT8_C(0x80) ^ params->scalar.max);
  const __m128i voutput_min_with_bias = _mm_set1_epi8(UINT8_C(0x80) ^ params->scalar.min);
  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(voutput_max_with_bias);
  XNN_FORCE_REALIZATION(voutput_min_with_bias);

  for (; batch >= 64; batch -= 64) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) input);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) input + 1);
    __m128i vacc2 = _mm_loadu_si128((const __m128i*) input + 2);
    __m128i vacc3 = _mm_loadu_si128((const __m128i*) input + 3);
    input += 64;

    vacc0 = _mm_xor_si128(vacc0, vbias);
    vacc1 = _mm_xor_si128(vacc1, vbias);
    vacc2 = _mm_xor_si128(vacc2, vbias);
    vacc3 = _mm_xor_si128(vacc3, vbias);

    vacc0 = _mm_max_epu8(vacc0, voutput_min_with_bias);
    vacc1 = _mm_max_epu8(vacc1, voutput_min_with_bias);
    vacc2 = _mm_max_epu8(vacc2, voutput_min_with_bias);
    vacc3 = _mm_max_epu8(vacc3, voutput_min_with_bias);

    vacc0 = _mm_min_epu8(vacc0, voutput_max_with_bias);
    vacc1 = _mm_min_epu8(vacc1, voutput_max_with_bias);
    vacc2 = _mm_min_epu8(vacc2, voutput_max_with_bias);
    vacc3 = _mm_min_epu8(vacc3, voutput_max_with_bias);

    vacc0 = _mm_xor_si128(vacc0, vbias);
    vacc1 = _mm_xor_si128(vacc1, vbias);
    vacc2 = _mm_xor_si128(vacc2, vbias);
    vacc3 = _mm_xor_si128(vacc3, vbias);

    _mm_storeu_si128((__m128i*) output, vacc0);
    _mm_storeu_si128((__m128i*) output + 1, vacc1);
    _mm_storeu_si128((__m128i*) output + 2, vacc2);
    _mm_storeu_si128((__m128i*) output + 3, vacc3);
    output += 64;
  }
  for (; batch >= 16; batch -= 16) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    vacc = _mm_xor_si128(vacc, vbias);
    vacc = _mm_min_epu8(vacc, voutput_max_with_bias);
    vacc = _mm_max_epu8(vacc, voutput_min_with_bias);
    vacc = _mm_xor_si128(vacc, vbias);

    _mm_storeu_si128((__m128i*) output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);

    vacc = _mm_xor_si128(vacc, vbias);
    vacc = _mm_min_epu8(vacc, voutput_max_with_bias);
    vacc = _mm_max_epu8(vacc, voutput_min_with_bias);
    vacc = _mm_xor_si128(vacc, vbias);

    if (batch & 8) {
      _mm_storel_epi64((__m128i*) output, vacc);
      output += 8;
      vacc = _mm_unpackhi_epi64(vacc, vacc);
    }
    if (batch & 4) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vacc));
      output += 4;
      vacc = _mm_srli_epi64(vacc, 32);
    }
    if (batch & 2) {
      unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vacc));
      output += 2;
      vacc = _mm_srli_epi32(vacc, 16);
    }
    if (batch & 1) {
      *output = (int8_t) _mm_cvtsi128_si32(vacc);
    }
  }
}

void xnn_u8_ibilinear_ukernel__sse2_c8(
    size_t output_pixels,
    size_t channels,
    const uint8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    uint8_t* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const uint8_t* i0 = (const uint8_t*) ((uintptr_t) input[0] + input_offset);
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input[1] + input_offset);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input[2] + input_offset);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const __m128i valpha = _mm_cvtsi32_si128(*((const int*) weights));
    weights += 2;
    __m128i valphah = _mm_shufflelo_epi16(valpha, _MM_SHUFFLE(0, 0, 0, 0));
    valphah = _mm_unpacklo_epi64(valphah, valphah);
    __m128i valphav = _mm_shufflelo_epi16(valpha, _MM_SHUFFLE(1, 1, 1, 1));
    valphav = _mm_unpacklo_epi64(valphav, valphav);

    valphah = _mm_xor_si128(valphah, _mm_set1_epi32(0xFFFF0000));
    valphah = _mm_add_epi16(valphah, _mm_set1_epi32(0x08010000));

    const __m128i vrounding = _mm_set1_epi32(0x00200000);

    size_t c = channels;
    for (; c >= 8 * sizeof(uint8_t); c -= 8 * sizeof(uint8_t)) {
      __m128i vtl01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;
      __m128i vtr01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;
      __m128i vbl01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;
      __m128i vbr01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      __m128i vzero = _mm_setzero_si128();
      vtl01234567 = _mm_unpacklo_epi8(vtl01234567, vzero);
      vtr01234567 = _mm_unpacklo_epi8(vtr01234567, vzero);
      vbl01234567 = _mm_unpacklo_epi8(vbl01234567, vzero);
      vbr01234567 = _mm_unpacklo_epi8(vbr01234567, vzero);

      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_slli_epi32(_mm_mulhi_epu16(vd0123, valphav), 16);
      __m128i vacc4567 = _mm_slli_epi32(_mm_mulhi_epu16(vd4567, valphav), 16);

      vacc0123 = _mm_add_epi16(_mm_mullo_epi16(vd0123, valphav), vacc0123);
      vacc4567 = _mm_add_epi16(_mm_mullo_epi16(vd4567, valphav), vacc4567);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srli_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srli_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      const __m128i vo01234567 = _mm_packus_epi16(vacc01234567, vacc01234567);

      _mm_storel_epi64((__m128i*) output, vo01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128i vtl01234567 = _mm_loadl_epi64((const __m128i*) i0);
      __m128i vtr01234567 = _mm_loadl_epi64((const __m128i*) i1);
      __m128i vbl01234567 = _mm_loadl_epi64((const __m128i*) i2);
      __m128i vbr01234567 = _mm_loadl_epi64((const __m128i*) i3);

      __m128i vzero = _mm_setzero_si128();
      vtl01234567 = _mm_unpacklo_epi8(vtl01234567, vzero);
      vtr01234567 = _mm_unpacklo_epi8(vtr01234567, vzero);
      vbl01234567 = _mm_unpacklo_epi8(vbl01234567, vzero);
      vbr01234567 = _mm_unpacklo_epi8(vbr01234567, vzero);

      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_slli_epi32(_mm_mulhi_epu16(vd0123, valphav), 16);
      __m128i vacc4567 = _mm_slli_epi32(_mm_mulhi_epu16(vd4567, valphav), 16);

      vacc0123 = _mm_add_epi16(_mm_mullo_epi16(vd0123, valphav), vacc0123);
      vacc4567 = _mm_add_epi16(_mm_mullo_epi16(vd4567, valphav), vacc4567);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srli_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srli_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      __m128i vo01234567 = _mm_packus_epi16(vacc01234567, vacc01234567);

      if (c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vo01234567));
        output += 4;
        vo01234567 = _mm_srli_epi64(vo01234567, 32);
      }
      uint32_t vo0123 = (uint32_t) _mm_cvtsi128_si32(vo01234567);
      if (c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vo0123);
        output += 2;
        vo0123 >>= 16;
      }
      if (c & (1 * sizeof(uint8_t))) {
        *output++ = (uint8_t) vo0123;
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const __m128i voutput_max = _mm_set1_epi8(params->scalar.max);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.min);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

  do {
    uint8_t* o = output;
    {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      const uint8_t* i8 = *input++;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
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
      for (; c >= 16; c -= 16) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*) i0); i0 += 16;
        const __m128i vi1 = _mm_loadu_si128((const __m128i*) i1); i1 += 16;
        const __m128i vi2 = _mm_loadu_si128((const __m128i*) i2); i2 += 16;
        const __m128i vi3 = _mm_loadu_si128((const __m128i*) i3); i3 += 16;
        const __m128i vi4 = _mm_loadu_si128((const __m128i*) i4); i4 += 16;
        const __m128i vi5 = _mm_loadu_si128((const __m128i*) i5); i5 += 16;
        const __m128i vi6 = _mm_loadu_si128((const __m128i*) i6); i6 += 16;
        const __m128i vi7 = _mm_loadu_si128((const __m128i*) i7); i7 += 16;
        const __m128i vi8 = _mm_loadu_si128((const __m128i*) i8); i8 += 16;

        const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax01678);
        vout = _mm_max_epu8(vout, voutput_min);
        vout = _mm_min_epu8(vout, voutput_max);

        _mm_storeu_si128((__m128i*) o, vout); o += 16;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*) i0);
        const __m128i vi1 = _mm_loadu_si128((const __m128i*) i1);
        const __m128i vi2 = _mm_loadu_si128((const __m128i*) i2);
        const __m128i vi3 = _mm_loadu_si128((const __m128i*) i3);
        const __m128i vi4 = _mm_loadu_si128((const __m128i*) i4);
        const __m128i vi5 = _mm_loadu_si128((const __m128i*) i5);
        const __m128i vi6 = _mm_loadu_si128((const __m128i*) i6);
        const __m128i vi7 = _mm_loadu_si128((const __m128i*) i7);
        const __m128i vi8 = _mm_loadu_si128((const __m128i*) i8);

        const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax01678);
        vout = _mm_max_epu8(vout, voutput_min);
        vout = _mm_min_epu8(vout, voutput_max);

        if (c & 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          vout = _mm_unpackhi_epi64(vout, vout);
          o += 8;
        }
        if (c & 4) {
          unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
          vout = _mm_srli_epi64(vout, 32);
          o += 4;
        }
        if (c & 2) {
          unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
          vout = _mm_srli_epi32(vout, 16);
          o += 2;
        }
        if (c & 1) {
          *o = (uint8_t) _mm_cvtsi128_si32(vout);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
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
      for (; c >= 16; c -= 16) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*) i0); i0 += 16;
        const __m128i vi1 = _mm_loadu_si128((const __m128i*) i1); i1 += 16;
        const __m128i vi2 = _mm_loadu_si128((const __m128i*) i2); i2 += 16;
        const __m128i vi3 = _mm_loadu_si128((const __m128i*) i3); i3 += 16;
        const __m128i vi4 = _mm_loadu_si128((const __m128i*) i4); i4 += 16;
        const __m128i vi5 = _mm_loadu_si128((const __m128i*) i5); i5 += 16;
        const __m128i vi6 = _mm_loadu_si128((const __m128i*) i6); i6 += 16;
        const __m128i vi7 = _mm_loadu_si128((const __m128i*) i7); i7 += 16;
        const __m128i vo = _mm_loadu_si128((const __m128i*) o);

        const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax0167);
        vout = _mm_max_epu8(vout, voutput_min);
        vout = _mm_min_epu8(vout, voutput_max);

        _mm_storeu_si128((__m128i*) o, vout);
        o += 16;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*) i0);
        const __m128i vi1 = _mm_loadu_si128((const __m128i*) i1);
        const __m128i vi2 = _mm_loadu_si128((const __m128i*) i2);
        const __m128i vi3 = _mm_loadu_si128((const __m128i*) i3);
        const __m128i vi4 = _mm_loadu_si128((const __m128i*) i4);
        const __m128i vi5 = _mm_loadu_si128((const __m128i*) i5);
        const __m128i vi6 = _mm_loadu_si128((const __m128i*) i6);
        const __m128i vi7 = _mm_loadu_si128((const __m128i*) i7);
        const __m128i vo = _mm_loadu_si128((const __m128i*) o);

        const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax0167);
        vout = _mm_max_epu8(vout, voutput_min);
        vout = _mm_min_epu8(vout, voutput_max);

        if (c & 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          vout = _mm_unpackhi_epi64(vout, vout);
          o += 8;
        }
        if (c & 4) {
          unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
          vout = _mm_srli_epi64(vout, 32);
          o += 4;
        }
        if (c & 2) {
          unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
          vout = _mm_srli_epi32(vout, 16);
          o += 2;
        }
        if (c & 1) {
          *o = (uint8_t) _mm_cvtsi128_si32(vout);
          o += 1;
        }
      }
    }
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    output = (uint8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_u8_rmax_ukernel__sse2_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  if XNN_LIKELY(batch >= 16) {
    __m128i vmax = _mm_setzero_si128();
    do {
      const __m128i vx = _mm_loadu_si128((const __m128i*) input);
      input += 16;
      vmax = _mm_max_epu8(vmax, vx);
      batch -= 16;
    } while (batch >= 16);
    if (batch != 0) {
      const size_t x_increment = batch - 16;
      input = (const uint8_t*) ((uintptr_t) input + x_increment);
      const __m128i vx = _mm_loadu_si128((const __m128i*) input);
      vmax = _mm_max_epu8(vmax, vx);
    }
    vmax = _mm_max_epu8(vmax, _mm_unpackhi_epi64(vmax, vmax));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi64(vmax, 32));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi32(vmax, 16));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi16(vmax, 8));
    *output = (uint8_t) _mm_cvtsi128_si32(vmax);
  } else {
    uint8_t vmax = 0;
    do {
      const uint8_t vx = *input++;
      vmax = vx > vmax ? vx : vmax;
    } while (--batch != 0);
    *output = vmax;
  }
}

void xnn_u8_vclamp_ukernel__sse2_u64(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i voutput_max = _mm_set1_epi8(params->scalar.max);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.min);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

  for (; batch >= 64; batch -= 64) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) input);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) input + 1);
    __m128i vacc2 = _mm_loadu_si128((const __m128i*) input + 2);
    __m128i vacc3 = _mm_loadu_si128((const __m128i*) input + 3);
    input += 64;

    vacc0 = _mm_max_epu8(vacc0, voutput_min);
    vacc1 = _mm_max_epu8(vacc1, voutput_min);
    vacc2 = _mm_max_epu8(vacc2, voutput_min);
    vacc3 = _mm_max_epu8(vacc3, voutput_min);

    vacc0 = _mm_min_epu8(vacc0, voutput_max);
    vacc1 = _mm_min_epu8(vacc1, voutput_max);
    vacc2 = _mm_min_epu8(vacc2, voutput_max);
    vacc3 = _mm_min_epu8(vacc3, voutput_max);

    _mm_storeu_si128((__m128i*) output, vacc0);
    _mm_storeu_si128((__m128i*) output + 1, vacc1);
    _mm_storeu_si128((__m128i*) output + 2, vacc2);
    _mm_storeu_si128((__m128i*) output + 3, vacc3);
    output += 64;
  }
  for (; batch >= 16; batch -= 16) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    vacc = _mm_min_epu8(vacc, voutput_max);
    vacc = _mm_max_epu8(vacc, voutput_min);

    _mm_storeu_si128((__m128i*) output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);

    vacc = _mm_min_epu8(vacc, voutput_max);
    vacc = _mm_max_epu8(vacc, voutput_min);

    if (batch & 8) {
      _mm_storel_epi64((__m128i*) output, vacc);
      output += 8;
      vacc = _mm_unpackhi_epi64(vacc, vacc);
    }
    if (batch & 4) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vacc));
      output += 4;
      vacc = _mm_srli_epi64(vacc, 32);
    }
    if (batch & 2) {
      unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vacc));
      output += 2;
      vacc = _mm_srli_epi32(vacc, 16);
    }
    if (batch & 1) {
      *output = (uint8_t) _mm_cvtsi128_si32(vacc);
    }
  }
}

void xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2(
    const uint16_t* input,
    uint16_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint16_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint16_t));

  const size_t tile_height = 8;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t);

  const uint16_t* i0 = input;
  uint16_t* o0 = (uint16_t*) output;
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);
  uint16_t* o2 = (uint16_t*) ((uintptr_t) o1 + output_stride);
  uint16_t* o3 = (uint16_t*) ((uintptr_t) o2 + output_stride);
  uint16_t* o4 = (uint16_t*) ((uintptr_t) o3 + output_stride);
  uint16_t* o5 = (uint16_t*) ((uintptr_t) o4 + output_stride);
  uint16_t* o6 = (uint16_t*) ((uintptr_t) o5 + output_stride);
  uint16_t* o7 = (uint16_t*) ((uintptr_t) o6 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 4) {
      o4 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 6) {
      o5 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 6) {
      o6 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 8) {
      o7 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 8; bh -= 8) {
      const __m128i v3_0 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_1 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_2 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_3 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_4 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_5 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_6 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_7 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      const __m128i v2_0 = _mm_unpacklo_epi16(v3_0, v3_1);
      const __m128i v2_1 = _mm_unpackhi_epi16(v3_0, v3_1);
      const __m128i v2_2 = _mm_unpacklo_epi16(v3_2, v3_3);
      const __m128i v2_3 = _mm_unpackhi_epi16(v3_2, v3_3);
      const __m128i v2_4 = _mm_unpacklo_epi16(v3_4, v3_5);
      const __m128i v2_5 = _mm_unpackhi_epi16(v3_4, v3_5);
      const __m128i v2_6 = _mm_unpacklo_epi16(v3_6, v3_7);
      const __m128i v2_7 = _mm_unpackhi_epi16(v3_6, v3_7);

      const __m128i v1_0 = _mm_unpacklo_epi32(v2_0, v2_2);
      const __m128i v1_1 = _mm_unpackhi_epi32(v2_0, v2_2);
      const __m128i v1_2 = _mm_unpacklo_epi32(v2_1, v2_3);
      const __m128i v1_3 = _mm_unpackhi_epi32(v2_1, v2_3);
      const __m128i v1_4 = _mm_unpacklo_epi32(v2_4, v2_6);
      const __m128i v1_5 = _mm_unpackhi_epi32(v2_4, v2_6);
      const __m128i v1_6 = _mm_unpacklo_epi32(v2_5, v2_7);
      const __m128i v1_7 = _mm_unpackhi_epi32(v2_5, v2_7);

      const __m128i v0_0 = _mm_unpacklo_epi64(v1_0, v1_4);
      const __m128i v0_1 = _mm_unpackhi_epi64(v1_0, v1_4);
      const __m128i v0_2 = _mm_unpacklo_epi64(v1_1, v1_5);
      const __m128i v0_3 = _mm_unpackhi_epi64(v1_1, v1_5);
      const __m128i v0_4 = _mm_unpacklo_epi64(v1_2, v1_6);
      const __m128i v0_5 = _mm_unpackhi_epi64(v1_2, v1_6);
      const __m128i v0_6 = _mm_unpacklo_epi64(v1_3, v1_7);
      const __m128i v0_7 = _mm_unpackhi_epi64(v1_3, v1_7);


      _mm_storeu_si128((__m128i*) o7, v0_7);
      o7 = (uint16_t*) ((uintptr_t) o7 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o6, v0_6);
      o6 = (uint16_t*) ((uintptr_t) o6 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o5, v0_5);
      o5 = (uint16_t*) ((uintptr_t) o5 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o4, v0_4);
      o4 = (uint16_t*) ((uintptr_t) o4 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o3, v0_3);
      o3 = (uint16_t*) ((uintptr_t) o3 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o2, v0_2);
      o2 = (uint16_t*) ((uintptr_t) o2 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o1, v0_1);
      o1 = (uint16_t*) ((uintptr_t) o1 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o0, v0_0);
      o0 = (uint16_t*) ((uintptr_t) o0 + tile_hbytes);
    }
    if (bh != 0) {
      const __m128i v3_0 = _mm_loadu_si128((const __m128i*) i0);
      const uint16_t *i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m128i v3_1 = _mm_loadu_si128((const __m128i*) i1);
      const uint16_t *i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m128i v3_2 = _mm_loadu_si128((const __m128i*) i2);
      const uint16_t *i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m128i v3_3 = _mm_loadu_si128((const __m128i*) i3);
      const uint16_t *i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m128i v3_4 = _mm_loadu_si128((const __m128i*) i4);
      const uint16_t *i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m128i v3_5 = _mm_loadu_si128((const __m128i*) i5);
      const uint16_t *i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m128i v3_6 = _mm_loadu_si128((const __m128i*) i6);
      const __m128i v3_7 = _mm_undefined_si128();

      const __m128i v2_0 = _mm_unpacklo_epi16(v3_0, v3_1);
      const __m128i v2_1 = _mm_unpackhi_epi16(v3_0, v3_1);
      const __m128i v2_2 = _mm_unpacklo_epi16(v3_2, v3_3);
      const __m128i v2_3 = _mm_unpackhi_epi16(v3_2, v3_3);
      const __m128i v2_4 = _mm_unpacklo_epi16(v3_4, v3_5);
      const __m128i v2_5 = _mm_unpackhi_epi16(v3_4, v3_5);
      const __m128i v2_6 = _mm_unpacklo_epi16(v3_6, v3_7);
      const __m128i v2_7 = _mm_unpackhi_epi16(v3_6, v3_7);

      const __m128i v1_0 = _mm_unpacklo_epi32(v2_0, v2_2);
      const __m128i v1_1 = _mm_unpackhi_epi32(v2_0, v2_2);
      const __m128i v1_2 = _mm_unpacklo_epi32(v2_1, v2_3);
      const __m128i v1_3 = _mm_unpackhi_epi32(v2_1, v2_3);
      const __m128i v1_4 = _mm_unpacklo_epi32(v2_4, v2_6);
      const __m128i v1_5 = _mm_unpackhi_epi32(v2_4, v2_6);
      const __m128i v1_6 = _mm_unpacklo_epi32(v2_5, v2_7);
      const __m128i v1_7 = _mm_unpackhi_epi32(v2_5, v2_7);

      __m128i v0_0 = _mm_unpacklo_epi64(v1_0, v1_4);
      __m128i v0_1 = _mm_unpackhi_epi64(v1_0, v1_4);
      __m128i v0_2 = _mm_unpacklo_epi64(v1_1, v1_5);
      __m128i v0_3 = _mm_unpackhi_epi64(v1_1, v1_5);
      __m128i v0_4 = _mm_unpacklo_epi64(v1_2, v1_6);
      __m128i v0_5 = _mm_unpackhi_epi64(v1_2, v1_6);
      __m128i v0_6 = _mm_unpacklo_epi64(v1_3, v1_7);
      __m128i v0_7 = _mm_unpackhi_epi64(v1_3, v1_7);


      if (bh & 4) {
        _mm_storel_epi64((__m128i*) o7, v0_7);
        o7 += 4;
        _mm_storel_epi64((__m128i*) o6, v0_6);
        o6 += 4;
        _mm_storel_epi64((__m128i*) o5, v0_5);
        o5 += 4;
        _mm_storel_epi64((__m128i*) o4, v0_4);
        o4 += 4;
        _mm_storel_epi64((__m128i*) o3, v0_3);
        o3 += 4;
        _mm_storel_epi64((__m128i*) o2, v0_2);
        o2 += 4;
        _mm_storel_epi64((__m128i*) o1, v0_1);
        o1 += 4;
        _mm_storel_epi64((__m128i*) o0, v0_0);
        o0 += 4;
        v0_0 = _mm_unpackhi_epi64(v0_0, v0_0);
        v0_1 = _mm_unpackhi_epi64(v0_1, v0_1);
        v0_2 = _mm_unpackhi_epi64(v0_2, v0_2);
        v0_3 = _mm_unpackhi_epi64(v0_3, v0_3);
        v0_4 = _mm_unpackhi_epi64(v0_4, v0_4);
        v0_5 = _mm_unpackhi_epi64(v0_5, v0_5);
        v0_6 = _mm_unpackhi_epi64(v0_6, v0_6);
        v0_7 = _mm_unpackhi_epi64(v0_7, v0_7);
      }

      if (bh & 2) {
        unaligned_store_u32(o7, (uint32_t) _mm_cvtsi128_si32(v0_7));
        o7 += 2;
        unaligned_store_u32(o6, (uint32_t) _mm_cvtsi128_si32(v0_6));
        o6 += 2;
        unaligned_store_u32(o5, (uint32_t) _mm_cvtsi128_si32(v0_5));
        o5 += 2;
        unaligned_store_u32(o4, (uint32_t) _mm_cvtsi128_si32(v0_4));
        o4 += 2;
        unaligned_store_u32(o3, (uint32_t) _mm_cvtsi128_si32(v0_3));
        o3 += 2;
        unaligned_store_u32(o2, (uint32_t) _mm_cvtsi128_si32(v0_2));
        o2 += 2;
        unaligned_store_u32(o1, (uint32_t) _mm_cvtsi128_si32(v0_1));
        o1 += 2;
        unaligned_store_u32(o0, (uint32_t) _mm_cvtsi128_si32(v0_0));
        o0 += 2;
        v0_0 = _mm_srli_epi64(v0_0, 32);
        v0_1 = _mm_srli_epi64(v0_1, 32);
        v0_2 = _mm_srli_epi64(v0_2, 32);
        v0_3 = _mm_srli_epi64(v0_3, 32);
        v0_4 = _mm_srli_epi64(v0_4, 32);
        v0_5 = _mm_srli_epi64(v0_5, 32);
        v0_6 = _mm_srli_epi64(v0_6, 32);
        v0_7 = _mm_srli_epi64(v0_7, 32);
      }
      if (bh & 1) {
        unaligned_store_u16(o7, (uint16_t) _mm_cvtsi128_si32(v0_7));
        unaligned_store_u16(o6, (uint16_t) _mm_cvtsi128_si32(v0_6));
        unaligned_store_u16(o5, (uint16_t) _mm_cvtsi128_si32(v0_5));
        unaligned_store_u16(o4, (uint16_t) _mm_cvtsi128_si32(v0_4));
        unaligned_store_u16(o3, (uint16_t) _mm_cvtsi128_si32(v0_3));
        unaligned_store_u16(o2, (uint16_t) _mm_cvtsi128_si32(v0_2));
        unaligned_store_u16(o1, (uint16_t) _mm_cvtsi128_si32(v0_1));
        unaligned_store_u16(o0, (uint16_t) _mm_cvtsi128_si32(v0_0));
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint16_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint16_t*) ((uintptr_t) o3 + output_reset);
    o4 = (uint16_t*) ((uintptr_t) o4 + output_reset);
    o5 = (uint16_t*) ((uintptr_t) o5 + output_reset);
    o6 = (uint16_t*) ((uintptr_t) o6 + output_reset);
    o7 = (uint16_t*) ((uintptr_t) o7 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4(
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
  assert(nr == 2);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  __m128 v0;
  __m128 v1;

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 2
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 2; n -= 2) {
      if XNN_LIKELY(b != NULL) {
        packed_w[0] = b[0];
        packed_w[1] = b[1];
        b += 2;
      } else {
        packed_w[0] = 0.0f;
        packed_w[1] = 0.0f;
      }
      packed_w += 2;

      const float* w1 = w0 + kc;

      // KC main loop multiple of 2x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 2x4
        // a b c d
        // e f g h
        v0 = _mm_loadu_ps(w0);
        w0 += 4;
        v1 = _mm_loadu_ps(w1);
        w1 += 4;
        _mm_storeu_ps(packed_w, v0);
        _mm_storeu_ps(packed_w + 4, v1);
        packed_w += 8;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
            // Read blocks of 2x1
            // a
            // e
            v0 = _mm_load_ss(w0);
            w0 += 1;
            v1 = _mm_load_ss(w1);
            w1 += 1;
            break;
          case 2:
            // Read blocks of 2x2
            // a b
            // e f
            v0 = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            w0 += 2;
            v1 = _mm_castpd_ps(_mm_load_sd((const double*) w1));
            w1 += 2;
            break;
          case 3:
          {
            // Read blocks of 2x3
            // a b c
            // e f g
            const __m128 v0lo = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            const __m128 v0hi = _mm_load_ss(w0 + 2);
            v0 = _mm_movelh_ps(v0lo, v0hi);
            w0 += 3;
            const __m128 v1lo = _mm_castpd_ps(_mm_load_sd((const double*) w1));
            const __m128 v1hi = _mm_load_ss(w1 + 2);
            v1 = _mm_movelh_ps(v1lo, v1hi);
            w1 += 3;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
        _mm_storeu_ps(packed_w, v0);
        _mm_storeu_ps(packed_w + 4, v1);
        packed_w += 8;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w1;
    }

    // NC remainder (1..1)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 1);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (2 - n);
      } else {
        packed_w[0] = 0.0f;
        packed_w[1] = 0.0f;
        packed_w += 2;
      }

      // NR remainder has less than 2 rows so last row is not loaded

      // KC main loop multiple of 2x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 2x4
        // a b c d
        // e f g h
        v0 = _mm_loadu_ps(w0);
        w0 += 4;
        _mm_storeu_ps(packed_w, v0);
        _mm_storeu_ps(packed_w + 4, v0);
        packed_w += 8;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
            // Read blocks of 1x1
            // a
            v0 = _mm_load_ss(w0);
            w0 += 1;
            break;
          case 2:
            // Read blocks of 1x2
            // a b
            v0 = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            w0 += 2;
            break;
          case 3:
          {
            // Read blocks of 1x3
            // a b c
            const __m128 v0lo = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            const __m128 v0hi = _mm_load_ss(w0 + 2);
            v0 = _mm_movelh_ps(v0lo, v0hi);
            w0 += 3;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
        _mm_storeu_ps(packed_w, v0);
        _mm_storeu_ps(packed_w + 4, v0);
        packed_w += 8;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}

void xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4(
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
  assert(nr == 8);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 8
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 8; n -= 8) {
      if XNN_LIKELY(b != NULL) {
        const __m128 vb0123 = _mm_loadu_ps(b);
        const __m128 vb4567 = _mm_loadu_ps(b + 4);
        b += 8;

        _mm_store_ps(packed_w, vb0123);
        _mm_store_ps(packed_w + 4, vb4567);
      } else {
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
      }
      packed_w += 8;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;
      const float* w4 = w3 + kc;
      const float* w5 = w4 + kc;
      const float* w6 = w5 + kc;
      const float* w7 = w6 + kc;

      size_t k = kc;

      // KC multiple of 4
      for (; k >= 4; k -= 4) {
        const __m128 v0x0123 = _mm_loadu_ps(w0);
        w0 += 4;
        const __m128 v1x0123 = _mm_loadu_ps(w1);
        w1 += 4;
        const __m128 v2x0123 = _mm_loadu_ps(w2);
        w2 += 4;
        const __m128 v3x0123 = _mm_loadu_ps(w3);
        w3 += 4;
        const __m128 v4x0123 = _mm_loadu_ps(w4);
        w4 += 4;
        const __m128 v5x0123 = _mm_loadu_ps(w5);
        w5 += 4;
        const __m128 v6x0123 = _mm_loadu_ps(w6);
        w6 += 4;
        const __m128 v7x0123 = _mm_loadu_ps(w7);
        w7 += 4;

        const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x0123, v1x0123);
        const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x0123, v3x0123);
        const __m128 v01x2_01x3 = _mm_unpackhi_ps(v0x0123, v1x0123);
        const __m128 v23x2_23x3 = _mm_unpackhi_ps(v2x0123, v3x0123);
        const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x0123, v5x0123);
        const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x0123, v7x0123);
        const __m128 v45x2_45x3 = _mm_unpackhi_ps(v4x0123, v5x0123);
        const __m128 v67x2_67x3 = _mm_unpackhi_ps(v6x0123, v7x0123);
        const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
        const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
        const __m128 v0123x2 = _mm_movelh_ps(v01x2_01x3, v23x2_23x3);
        const __m128 v0123x3 = _mm_movehl_ps(v23x2_23x3, v01x2_01x3);
        const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
        const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
        const __m128 v4567x2 = _mm_movelh_ps(v45x2_45x3, v67x2_67x3);
        const __m128 v4567x3 = _mm_movehl_ps(v67x2_67x3, v45x2_45x3);

        _mm_store_ps(packed_w, v0123x0);
        _mm_store_ps(packed_w + 4, v4567x0);
        _mm_store_ps(packed_w + 8, v0123x1);
        _mm_store_ps(packed_w + 12, v4567x1);
        _mm_store_ps(packed_w + 16, v0123x2);
        _mm_store_ps(packed_w + 20, v4567x2);
        _mm_store_ps(packed_w + 24, v0123x3);
        _mm_store_ps(packed_w + 28, v4567x3);
        packed_w += 32;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            const __m128 v0x0 = _mm_load_ss(w0);
            w0 += 1;
            const __m128 v1x0 = _mm_load_ss(w1);
            w1 += 1;
            const __m128 v2x0 = _mm_load_ss(w2);
            w2 += 1;
            const __m128 v3x0 = _mm_load_ss(w3);
            w3 += 1;
            const __m128 v4x0 = _mm_load_ss(w4);
            w4 += 1;
            const __m128 v5x0 = _mm_load_ss(w5);
            w5 += 1;
            const __m128 v6x0 = _mm_load_ss(w6);
            w6 += 1;
            const __m128 v7x0 = _mm_load_ss(w7);
            w7 += 1;

            const __m128 v01x0 = _mm_unpacklo_ps(v0x0, v1x0);
            const __m128 v23x0 = _mm_unpacklo_ps(v2x0, v3x0);
            const __m128 v45x0 = _mm_unpacklo_ps(v4x0, v5x0);
            const __m128 v67x0 = _mm_unpacklo_ps(v6x0, v7x0);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0, v23x0);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0, v67x0);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            packed_w += 8;
            break;
          }
          case 2:
          {
            const __m128 v0x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
            w0 += 2;
            const __m128 v1x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
            w1 += 2;
            const __m128 v2x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
            w2 += 2;
            const __m128 v3x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
            w3 += 2;
            const __m128 v4x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
            w4 += 2;
            const __m128 v5x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
            w5 += 2;
            const __m128 v6x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
            w6 += 2;
            const __m128 v7x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
            w7 += 2;

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x01, v1x01);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x01, v3x01);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x01, v5x01);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x01, v7x01);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v0123x1);
            _mm_store_ps(packed_w + 12, v4567x1);
            packed_w += 16;
            break;
          }
          case 3:
          {
            __m128 v0x012 = _mm_load_ss(w0 + 2);
            __m128 v1x012 = _mm_load_ss(w1 + 2);
            __m128 v2x012 = _mm_load_ss(w2 + 2);
            __m128 v3x012 = _mm_load_ss(w3 + 2);
            __m128 v4x012 = _mm_load_ss(w4 + 2);
            __m128 v5x012 = _mm_load_ss(w5 + 2);
            __m128 v6x012 = _mm_load_ss(w6 + 2);
            __m128 v7x012 = _mm_load_ss(w7 + 2);

            v0x012 = _mm_movelh_ps(v0x012, v0x012);
            v1x012 = _mm_movelh_ps(v1x012, v1x012);
            v2x012 = _mm_movelh_ps(v2x012, v2x012);
            v3x012 = _mm_movelh_ps(v3x012, v3x012);
            v4x012 = _mm_movelh_ps(v4x012, v4x012);
            v5x012 = _mm_movelh_ps(v5x012, v5x012);
            v6x012 = _mm_movelh_ps(v6x012, v6x012);
            v7x012 = _mm_movelh_ps(v7x012, v7x012);

            v0x012 = _mm_loadl_pi(v0x012, (const __m64*) w0);
            w0 += 3;
            v1x012 = _mm_loadl_pi(v1x012, (const __m64*) w1);
            w1 += 3;
            v2x012 = _mm_loadl_pi(v2x012, (const __m64*) w2);
            w2 += 3;
            v3x012 = _mm_loadl_pi(v3x012, (const __m64*) w3);
            w3 += 3;
            v4x012 = _mm_loadl_pi(v4x012, (const __m64*) w4);
            w4 += 3;
            v5x012 = _mm_loadl_pi(v5x012, (const __m64*) w5);
            w5 += 3;
            v6x012 = _mm_loadl_pi(v6x012, (const __m64*) w6);
            w6 += 3;
            v7x012 = _mm_loadl_pi(v7x012, (const __m64*) w7);
            w7 += 3;

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x012, v1x012);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x012, v3x012);
            const __m128 v01x2 = _mm_unpackhi_ps(v0x012, v1x012);
            const __m128 v23x2 = _mm_unpackhi_ps(v2x012, v3x012);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x012, v5x012);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x012, v7x012);
            const __m128 v45x2 = _mm_unpackhi_ps(v4x012, v5x012);
            const __m128 v67x2 = _mm_unpackhi_ps(v6x012, v7x012);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v0123x2 = _mm_movelh_ps(v01x2, v23x2);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
            const __m128 v4567x2 = _mm_movelh_ps(v45x2, v67x2);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v0123x1);
            _mm_store_ps(packed_w + 12, v4567x1);
            _mm_store_ps(packed_w + 16, v0123x2);
            _mm_store_ps(packed_w + 20, v4567x2);
            packed_w += 24;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (8 - n);
      } else {
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
        packed_w += 8;
      }

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

      size_t k = kc;

      // KC multiple of 4
      for (; k >= 4; k -= 4) {
        const __m128 v0x0123 = _mm_loadu_ps(w0);
        w0 += 4;
        const __m128 v1x0123 = _mm_loadu_ps(w1);
        w1 += 4;
        const __m128 v2x0123 = _mm_loadu_ps(w2);
        w2 += 4;
        const __m128 v3x0123 = _mm_loadu_ps(w3);
        w3 += 4;
        const __m128 v4x0123 = _mm_loadu_ps(w4);
        w4 += 4;
        const __m128 v5x0123 = _mm_loadu_ps(w5);
        w5 += 4;
        const __m128 v6x0123 = _mm_loadu_ps(w6);
        w6 += 4;

        const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x0123, v1x0123);
        const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x0123, v3x0123);
        const __m128 v01x2_01x3 = _mm_unpackhi_ps(v0x0123, v1x0123);
        const __m128 v23x2_23x3 = _mm_unpackhi_ps(v2x0123, v3x0123);
        const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x0123, v5x0123);
        const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x0123, v6x0123);
        const __m128 v45x2_45x3 = _mm_unpackhi_ps(v4x0123, v5x0123);
        const __m128 v67x2_67x3 = _mm_unpackhi_ps(v6x0123, v6x0123);
        const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
        const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
        const __m128 v0123x2 = _mm_movelh_ps(v01x2_01x3, v23x2_23x3);
        const __m128 v0123x3 = _mm_movehl_ps(v23x2_23x3, v01x2_01x3);
        const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
        const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
        const __m128 v4567x2 = _mm_movelh_ps(v45x2_45x3, v67x2_67x3);
        const __m128 v4567x3 = _mm_movehl_ps(v67x2_67x3, v45x2_45x3);

        _mm_store_ps(packed_w, v0123x0);
        _mm_store_ps(packed_w + 4, v4567x0);
        _mm_store_ps(packed_w + 8, v0123x1);
        _mm_store_ps(packed_w + 12, v4567x1);
        _mm_store_ps(packed_w + 16, v0123x2);
        _mm_store_ps(packed_w + 20, v4567x2);
        _mm_store_ps(packed_w + 24, v0123x3);
        _mm_store_ps(packed_w + 28, v4567x3);
        packed_w += 32;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            const __m128 v0x0 = _mm_load_ss(w0);
            const __m128 v1x0 = _mm_load_ss(w1);
            const __m128 v2x0 = _mm_load_ss(w2);
            const __m128 v3x0 = _mm_load_ss(w3);
            const __m128 v4x0 = _mm_load_ss(w4);
            const __m128 v5x0 = _mm_load_ss(w5);
            const __m128 v6x0 = _mm_load_ss(w6);

            const __m128 v01x0 = _mm_unpacklo_ps(v0x0, v1x0);
            const __m128 v23x0 = _mm_unpacklo_ps(v2x0, v3x0);
            const __m128 v45x0 = _mm_unpacklo_ps(v4x0, v5x0);
            const __m128 v67x0 = _mm_unpacklo_ps(v6x0, v6x0);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0, v23x0);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0, v67x0);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            packed_w += 8;
            break;
          }
          case 2:
          {
            const __m128 v0x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
            const __m128 v1x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
            const __m128 v2x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
            const __m128 v3x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
            const __m128 v4x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
            const __m128 v5x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
            const __m128 v6x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x01, v1x01);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x01, v3x01);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x01, v5x01);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x01, v6x01);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v0123x1);
            _mm_store_ps(packed_w + 12, v4567x1);
            packed_w += 16;
            break;
          }
          case 3:
          {
            __m128 v0x012 = _mm_load_ss(w0 + 2);
            __m128 v1x012 = _mm_load_ss(w1 + 2);
            __m128 v2x012 = _mm_load_ss(w2 + 2);
            __m128 v3x012 = _mm_load_ss(w3 + 2);
            __m128 v4x012 = _mm_load_ss(w4 + 2);
            __m128 v5x012 = _mm_load_ss(w5 + 2);
            __m128 v6x012 = _mm_load_ss(w6 + 2);

            v0x012 = _mm_movelh_ps(v0x012, v0x012);
            v1x012 = _mm_movelh_ps(v1x012, v1x012);
            v2x012 = _mm_movelh_ps(v2x012, v2x012);
            v3x012 = _mm_movelh_ps(v3x012, v3x012);
            v4x012 = _mm_movelh_ps(v4x012, v4x012);
            v5x012 = _mm_movelh_ps(v5x012, v5x012);
            v6x012 = _mm_movelh_ps(v6x012, v6x012);

            v0x012 = _mm_loadl_pi(v0x012, (const __m64*) w0);
            v1x012 = _mm_loadl_pi(v1x012, (const __m64*) w1);
            v2x012 = _mm_loadl_pi(v2x012, (const __m64*) w2);
            v3x012 = _mm_loadl_pi(v3x012, (const __m64*) w3);
            v4x012 = _mm_loadl_pi(v4x012, (const __m64*) w4);
            v5x012 = _mm_loadl_pi(v5x012, (const __m64*) w5);
            v6x012 = _mm_loadl_pi(v6x012, (const __m64*) w6);

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x012, v1x012);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x012, v3x012);
            const __m128 v01x2 = _mm_unpackhi_ps(v0x012, v1x012);
            const __m128 v23x2 = _mm_unpackhi_ps(v2x012, v3x012);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x012, v5x012);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x012, v6x012);
            const __m128 v45x2 = _mm_unpackhi_ps(v4x012, v5x012);
            const __m128 v67x2 = _mm_unpackhi_ps(v6x012, v6x012);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v0123x2 = _mm_movelh_ps(v01x2, v23x2);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
            const __m128 v4567x2 = _mm_movelh_ps(v45x2, v67x2);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v0123x1);
            _mm_store_ps(packed_w + 12, v4567x1);
            _mm_store_ps(packed_w + 16, v0123x2);
            _mm_store_ps(packed_w + 20, v4567x2);
            packed_w += 24;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}

void xnn_x32_unpool_ukernel__sse2(
    size_t kernel_elements,
    size_t channels,
    uint32_t fill,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output)
{
  // Pre-initialize outputs with constant.
  const __m128i vfill = _mm_set1_epi32((int) fill);
  uint32_t** os = output;
  do {
    uint32_t* o = *os++;
    size_t c = channels;
    for (; c >= 4; c -= 4) {
      _mm_storeu_si128((__m128i*) o, vfill);
      o += 4;
    }
    if (c != 0) {
      if (c & 2) {
        _mm_storel_epi64((__m128i*) o, vfill);
        o += 2;
      }
      if (c & 1) {
        *((int*) o) = _mm_cvtsi128_si32(vfill);
      }
    }
  } while (--kernel_elements != 0);

  // Copy indexed elements to output.
  size_t offset = 0;
  do {
    const uint32_t i = *index++;
    *((uint32_t*) ((uintptr_t) output[i] + offset)) = *input++;
    offset += sizeof(uint32_t);
  } while (--channels != 0);
}

void xnn_x32_zip_x2_ukernel__sse2(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  uint32_t* o = output;

  while (n >= 16) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) x);
    x += 4;
    const __m128i vy = _mm_loadu_si128((const __m128i*) y);
    y += 4;
    const __m128i vxy_lo = _mm_unpacklo_epi32(vx, vy);
    const __m128i vxy_hi = _mm_unpackhi_epi32(vx, vy);
    _mm_storeu_si128((__m128i*) o, vxy_lo);
    _mm_storeu_si128((__m128i*) (o + 4), vxy_hi);
    o += 8;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      const __m128i vx = _mm_loadl_epi64((const __m128i*) x);
      x += 2;
      const __m128i vy = _mm_loadl_epi64((const __m128i*) y);
      y += 2;
      const __m128i vxy = _mm_unpacklo_epi32(vx, vy);
      _mm_storeu_si128((__m128i*) o, vxy);
      o += 4;
    }
    if (n & 4) {
      const uint32_t vx = *x;
      const uint32_t vy = *y;
      o[0] = vx;
      o[1] = vy;
    }
  }
}

void xnn_x32_zip_x3_ukernel__sse2(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const float* x = (const float*) input;
  const float* y = (const float*) ((uintptr_t) x + n);
  const float* z = (const float*) ((uintptr_t) y + n);
  float* o = (float*) output;

  while (n >= 16) {
    // vx = ( x3, x2, x1, x0 )
    const __m128 vx = _mm_loadu_ps(x);
    x += 4;
    // vy = ( y3, y2, y1, y0 )
    const __m128 vy = _mm_loadu_ps(y);
    y += 4;
    // vz = ( z3, z2, z1, z0 )
    const __m128 vz = _mm_loadu_ps(z);
    z += 4;

    // vxy = ( y2, y0, x2, x0 )
    const __m128 vxy = _mm_shuffle_ps(vx, vy, _MM_SHUFFLE(2, 0, 2, 0));
    // vyz = ( z3, z1, y3, y1 )
    const __m128 vyz = _mm_shuffle_ps(vy, vz, _MM_SHUFFLE(3, 1, 3, 1));
    // vzx = ( x3, x1, z2, z0 )
    const __m128 vzx = _mm_shuffle_ps(vz, vx, _MM_SHUFFLE(3, 1, 2, 0));

    // vxyz0 = ( x1, z0, y0, x0 )
    const __m128 vxyz0 = _mm_shuffle_ps(vxy, vzx, _MM_SHUFFLE(2, 0, 2, 0));
    // vxyz1 = ( y2, x2, z1, y1 )
    const __m128 vxyz1 = _mm_shuffle_ps(vyz, vxy, _MM_SHUFFLE(3, 1, 2, 0));
    // vxyz2 = ( z3, y3, x3, z2 )
    const __m128 vxyz2 = _mm_shuffle_ps(vzx, vyz, _MM_SHUFFLE(3, 1, 3, 1));

    _mm_storeu_ps(o, vxyz0);
    _mm_storeu_ps(o + 4, vxyz1);
    _mm_storeu_ps(o + 8, vxyz2);
    o += 12;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      // vx = ( -, -, x1, x0 )
      const __m128 vx = _mm_castpd_ps(_mm_load_sd((const double*) x));
      x += 2;
      // vy = ( -, -, y1, y0 )
      const __m128 vy = _mm_castpd_ps(_mm_load_sd((const double*) y));
      y += 2;
      // vz = ( -, -, z1, z0 )
      const __m128 vz = _mm_castpd_ps(_mm_load_sd((const double*) z));
      z += 2;

      // vxy = ( y1, x1, y0, x0 )
      const __m128 vxy = _mm_unpacklo_ps(vx, vy);
      // vzx = ( x1, z1, x0, z0 )
      const __m128 vzx = _mm_unpacklo_ps(vz, vx);
      // vyz = ( z1, y1, z0, y0 )
      const __m128 vyz = _mm_unpacklo_ps(vy, vz);

      _mm_storeu_ps(o, _mm_shuffle_ps(vxy, vzx, _MM_SHUFFLE(3, 0, 1, 0)));
      _mm_storeh_pi((__m64*) (o + 4), vyz);
      o += 6;
    }
    if (n & 4) {
      const __m128 vx = _mm_load_ss(x);
      const __m128 vy = _mm_load_ss(y);
      const __m128 vz = _mm_load_ss(z);
      _mm_store_ss(o, vx);
      _mm_store_ss(o + 1, vy);
      _mm_store_ss(o + 2, vz);
    }
  }
}

void xnn_x32_zip_x4_ukernel__sse2(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  const uint32_t* z = (const uint32_t*) ((uintptr_t) y + n);
  const uint32_t* w = (const uint32_t*) ((uintptr_t) z + n);
  uint32_t* o = output;

  while (n >= 16) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) x);
    x += 4;
    const __m128i vy = _mm_loadu_si128((const __m128i*) y);
    y += 4;
    const __m128i vz = _mm_loadu_si128((const __m128i*) z);
    z += 4;
    const __m128i vw = _mm_loadu_si128((const __m128i*) w);
    w += 4;

    const __m128i vxy_lo = _mm_unpacklo_epi32(vx, vy);
    const __m128i vxy_hi = _mm_unpackhi_epi32(vx, vy);
    const __m128i vzw_lo = _mm_unpacklo_epi32(vz, vw);
    const __m128i vzw_hi = _mm_unpackhi_epi32(vz, vw);

    const __m128i vxyzw0 = _mm_unpacklo_epi64(vxy_lo, vzw_lo);
    const __m128i vxyzw1 = _mm_unpackhi_epi64(vxy_lo, vzw_lo);
    const __m128i vxyzw2 = _mm_unpacklo_epi64(vxy_hi, vzw_hi);
    const __m128i vxyzw3 = _mm_unpackhi_epi64(vxy_hi, vzw_hi);

    _mm_storeu_si128((__m128i*) o, vxyzw0);
    _mm_storeu_si128((__m128i*) (o + 4), vxyzw1);
    _mm_storeu_si128((__m128i*) (o + 8), vxyzw2);
    _mm_storeu_si128((__m128i*) (o + 12), vxyzw3);
    o += 16;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      const __m128i vx = _mm_loadl_epi64((const __m128i*) x);
      x += 2;
      const __m128i vy = _mm_loadl_epi64((const __m128i*) y);
      y += 2;
      const __m128i vz = _mm_loadl_epi64((const __m128i*) z);
      z += 2;
      const __m128i vw = _mm_loadl_epi64((const __m128i*) w);
      w += 2;

      const __m128i vxy = _mm_unpacklo_epi32(vx, vy);
      const __m128i vzw = _mm_unpacklo_epi32(vz, vw);

      const __m128i vxyzw_lo = _mm_unpacklo_epi64(vxy, vzw);
      const __m128i vxyzw_hi = _mm_unpackhi_epi64(vxy, vzw);

      _mm_storeu_si128((__m128i*) o, vxyzw_lo);
      _mm_storeu_si128((__m128i*) (o + 4), vxyzw_hi);
      o += 8;
    }
    if (n & 4) {
      const uint32_t vx = *x;
      const uint32_t vy = *y;
      const uint32_t vz = *z;
      const uint32_t vw = *w;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o[3] = vw;
    }
  }
}

void xnn_x32_zip_xm_ukernel__sse2(
    size_t n,
    size_t m,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);
  assert(m >= 4);

  const uint32_t* w = input;
  const size_t group_increment = m * 4;
  const size_t input_increment = n * 3;
  const size_t output_increment = 16 - m * n;
  const uint32_t* last_input = (const uint32_t*) ((uintptr_t) input + n * (m - 1));
  uint32_t* last_output = (uint32_t*) ((uintptr_t) output + (m * 4 - 16));

  for (size_t i = 0; i < m; i += 4) {
    w = (const uint32_t*) ((uintptr_t) w + input_increment);
    if (w >= last_input) {
      w = last_input;
    }
    const uint32_t* z = (const uint32_t*) ((uintptr_t) w - n);
    const uint32_t* y = (const uint32_t*) ((uintptr_t) z - n);
    const uint32_t* x = (const uint32_t*) ((uintptr_t) y - n);

    size_t k = n;
    while (k >= 16) {
      const __m128i vx = _mm_loadu_si128((const __m128i*) x);
      x += 4;
      const __m128i vy = _mm_loadu_si128((const __m128i*) y);
      y += 4;
      const __m128i vz = _mm_loadu_si128((const __m128i*) z);
      z += 4;
      const __m128i vw = _mm_loadu_si128((const __m128i*) w);
      w += 4;

      const __m128i vxy_lo = _mm_unpacklo_epi32(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi32(vx, vy);
      const __m128i vzw_lo = _mm_unpacklo_epi32(vz, vw);
      const __m128i vzw_hi = _mm_unpackhi_epi32(vz, vw);

      const __m128i vxyzw0 = _mm_unpacklo_epi64(vxy_lo, vzw_lo);
      const __m128i vxyzw1 = _mm_unpackhi_epi64(vxy_lo, vzw_lo);
      const __m128i vxyzw2 = _mm_unpacklo_epi64(vxy_hi, vzw_hi);
      const __m128i vxyzw3 = _mm_unpackhi_epi64(vxy_hi, vzw_hi);

      _mm_storeu_si128((__m128i*) output, vxyzw0);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      _mm_storeu_si128((__m128i*) output, vxyzw1);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      _mm_storeu_si128((__m128i*) output, vxyzw2);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      _mm_storeu_si128((__m128i*) output, vxyzw3);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      k -= 16;
    }
    if XNN_UNLIKELY(k != 0) {
      if (k & 8) {
        const __m128i vx = _mm_loadl_epi64((const __m128i*) x);
        x += 2;
        const __m128i vy = _mm_loadl_epi64((const __m128i*) y);
        y += 2;
        const __m128i vz = _mm_loadl_epi64((const __m128i*) z);
        z += 2;
        const __m128i vw = _mm_loadl_epi64((const __m128i*) w);
        w += 2;

        const __m128i vxy = _mm_unpacklo_epi32(vx, vy);
        const __m128i vzw = _mm_unpacklo_epi32(vz, vw);

        const __m128i vxyzw_lo = _mm_unpacklo_epi64(vxy, vzw);
        const __m128i vxyzw_hi = _mm_unpackhi_epi64(vxy, vzw);

        _mm_storeu_si128((__m128i*) output, vxyzw_lo);
        output = (uint32_t*) ((uintptr_t) output + group_increment);

        _mm_storeu_si128((__m128i*) output, vxyzw_hi);
        output = (uint32_t*) ((uintptr_t) output + group_increment);
      }
      if (k & 4) {
        const uint32_t vx = *x;
        const uint32_t vy = *y;
        const uint32_t vz = *z;
        const uint32_t vw = *w++;

        output[0] = vx;
        output[1] = vy;
        output[2] = vz;
        output[3] = vw;
        output = (uint32_t*) ((uintptr_t) output + group_increment);
      }
    }
    output = (uint32_t*) ((uintptr_t) output + output_increment);
    if (output > last_output) {
      output = last_output;
    }
  }
}

void xnn_x64_transposec_ukernel__2x2_multi_mov_sse2(
    const uint64_t* input,
    uint64_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint64_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint64_t));

  const size_t tile_height = 2;
  const size_t tile_width = 2;
  const size_t tile_hbytes = tile_height * sizeof(uint64_t);
  const size_t tile_wbytes = tile_width * sizeof(uint64_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint64_t) - tile_hbytes;

  const uint64_t* i0 = input;
  const uint64_t* i1 = (const uint64_t*) ((uintptr_t) i0 + input_stride);
  uint64_t* o = (uint64_t*) ((uintptr_t) output - tile_hbytes);
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 1);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      const __m128i v1_0 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint64_t*) ((uintptr_t) i0 + input_offset);
      const __m128i v1_1 = _mm_loadu_si128((const __m128i*) i1);
      i1 = (uint64_t*) ((uintptr_t) i1 + input_offset);

      const __m128i v0_0 = _mm_unpacklo_epi64(v1_0, v1_1);
      const __m128i v0_1 = _mm_unpackhi_epi64(v1_0, v1_1);




      o = (uint64_t*) ((uintptr_t) o + oN_offset);
      _mm_storeu_si128((__m128i*) o, v0_1);
      uint64_t *oN = (uint64_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_0);
    }
    o = (uint64_t*) ((uintptr_t) o + tile_hbytes);
    if (bh != 0) {
      const __m128i v1_0 = _mm_loadu_si128((const __m128i*) i0);
      const __m128i v1_1 = _mm_undefined_si128();

      __m128i v0_0 = _mm_unpacklo_epi64(v1_0, v1_1);
      __m128i v0_1 = _mm_unpackhi_epi64(v1_0, v1_1);




      if (bh & 1) {
        o = (uint64_t*) ((uintptr_t) o + oN_stride);
        _mm_storel_epi64((__m128i*) o, v0_1);
        uint64_t *oN = (uint64_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_0);
      }

    }

    i0 = (const uint64_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint64_t*) ((uintptr_t) i0 + input_stride);
    o = (uint64_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x8_transposec_ukernel__16x16_reuse_mov_sse2(
    const uint8_t* input,
    uint8_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint8_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint8_t));

  const size_t tile_height = 16;
  const size_t tile_width = 16;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t) - tile_hbytes;

  const uint8_t* i0 = input;
  uint8_t* o = (uint8_t*) ((uintptr_t) output - tile_hbytes);
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 15);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;
    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const __m128i v4_0 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_1 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_2 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_3 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_4 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_5 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_6 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_7 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_8 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_9 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_10 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_11 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_12 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_13 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_14 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4_15 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);

      const __m128i v3_0 = _mm_unpacklo_epi8(v4_0, v4_1);
      const __m128i v3_1 = _mm_unpackhi_epi8(v4_0, v4_1);
      const __m128i v3_2 = _mm_unpacklo_epi8(v4_2, v4_3);
      const __m128i v3_3 = _mm_unpackhi_epi8(v4_2, v4_3);
      const __m128i v3_4 = _mm_unpacklo_epi8(v4_4, v4_5);
      const __m128i v3_5 = _mm_unpackhi_epi8(v4_4, v4_5);
      const __m128i v3_6 = _mm_unpacklo_epi8(v4_6, v4_7);
      const __m128i v3_7 = _mm_unpackhi_epi8(v4_6, v4_7);
      const __m128i v3_8 = _mm_unpacklo_epi8(v4_8, v4_9);
      const __m128i v3_9 = _mm_unpackhi_epi8(v4_8, v4_9);
      const __m128i v3_10 = _mm_unpacklo_epi8(v4_10, v4_11);
      const __m128i v3_11 = _mm_unpackhi_epi8(v4_10, v4_11);
      const __m128i v3_12 = _mm_unpacklo_epi8(v4_12, v4_13);
      const __m128i v3_13 = _mm_unpackhi_epi8(v4_12, v4_13);
      const __m128i v3_14 = _mm_unpacklo_epi8(v4_14, v4_15);
      const __m128i v3_15 = _mm_unpackhi_epi8(v4_14, v4_15);

      const __m128i v2_0 = _mm_unpacklo_epi16(v3_0, v3_2);
      const __m128i v2_1 = _mm_unpackhi_epi16(v3_0, v3_2);
      const __m128i v2_2 = _mm_unpacklo_epi16(v3_1, v3_3);
      const __m128i v2_3 = _mm_unpackhi_epi16(v3_1, v3_3);
      const __m128i v2_4 = _mm_unpacklo_epi16(v3_4, v3_6);
      const __m128i v2_5 = _mm_unpackhi_epi16(v3_4, v3_6);
      const __m128i v2_6 = _mm_unpacklo_epi16(v3_5, v3_7);
      const __m128i v2_7 = _mm_unpackhi_epi16(v3_5, v3_7);
      const __m128i v2_8 = _mm_unpacklo_epi16(v3_8, v3_10);
      const __m128i v2_9 = _mm_unpackhi_epi16(v3_8, v3_10);
      const __m128i v2_10 = _mm_unpacklo_epi16(v3_9, v3_11);
      const __m128i v2_11 = _mm_unpackhi_epi16(v3_9, v3_11);
      const __m128i v2_12 = _mm_unpacklo_epi16(v3_12, v3_14);
      const __m128i v2_13 = _mm_unpackhi_epi16(v3_12, v3_14);
      const __m128i v2_14 = _mm_unpacklo_epi16(v3_13, v3_15);
      const __m128i v2_15 = _mm_unpackhi_epi16(v3_13, v3_15);

      const __m128i v1_0 = _mm_unpacklo_epi32(v2_0, v2_4);
      const __m128i v1_1 = _mm_unpackhi_epi32(v2_0, v2_4);
      const __m128i v1_2 = _mm_unpacklo_epi32(v2_1, v2_5);
      const __m128i v1_3 = _mm_unpackhi_epi32(v2_1, v2_5);
      const __m128i v1_4 = _mm_unpacklo_epi32(v2_2, v2_6);
      const __m128i v1_5 = _mm_unpackhi_epi32(v2_2, v2_6);
      const __m128i v1_6 = _mm_unpacklo_epi32(v2_3, v2_7);
      const __m128i v1_7 = _mm_unpackhi_epi32(v2_3, v2_7);
      const __m128i v1_8 = _mm_unpacklo_epi32(v2_8, v2_12);
      const __m128i v1_9 = _mm_unpackhi_epi32(v2_8, v2_12);
      const __m128i v1_10 = _mm_unpacklo_epi32(v2_9, v2_13);
      const __m128i v1_11 = _mm_unpackhi_epi32(v2_9, v2_13);
      const __m128i v1_12 = _mm_unpacklo_epi32(v2_10, v2_14);
      const __m128i v1_13 = _mm_unpackhi_epi32(v2_10, v2_14);
      const __m128i v1_14 = _mm_unpacklo_epi32(v2_11, v2_15);
      const __m128i v1_15 = _mm_unpackhi_epi32(v2_11, v2_15);

      const __m128i v0_0 = _mm_unpacklo_epi64(v1_0, v1_8);
      const __m128i v0_1 = _mm_unpackhi_epi64(v1_0, v1_8);
      const __m128i v0_2 = _mm_unpacklo_epi64(v1_1, v1_9);
      const __m128i v0_3 = _mm_unpackhi_epi64(v1_1, v1_9);
      const __m128i v0_4 = _mm_unpacklo_epi64(v1_2, v1_10);
      const __m128i v0_5 = _mm_unpackhi_epi64(v1_2, v1_10);
      const __m128i v0_6 = _mm_unpacklo_epi64(v1_3, v1_11);
      const __m128i v0_7 = _mm_unpackhi_epi64(v1_3, v1_11);
      const __m128i v0_8 = _mm_unpacklo_epi64(v1_4, v1_12);
      const __m128i v0_9 = _mm_unpackhi_epi64(v1_4, v1_12);
      const __m128i v0_10 = _mm_unpacklo_epi64(v1_5, v1_13);
      const __m128i v0_11 = _mm_unpackhi_epi64(v1_5, v1_13);
      const __m128i v0_12 = _mm_unpacklo_epi64(v1_6, v1_14);
      const __m128i v0_13 = _mm_unpackhi_epi64(v1_6, v1_14);
      const __m128i v0_14 = _mm_unpacklo_epi64(v1_7, v1_15);
      const __m128i v0_15 = _mm_unpackhi_epi64(v1_7, v1_15);

      o = (uint8_t*) ((uintptr_t) o + oN_offset);
      _mm_storeu_si128((__m128i*) o, v0_15);
      uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 15) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_14);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 15) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_13);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 13) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_12);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 13) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_11);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 11) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_10);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 11) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_9);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 9) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_8);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 9) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_7);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 7) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_6);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 7) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_5);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 5) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_4);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 5) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_3);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 3) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_2);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 3) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_1);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = oN;
      }
      _mm_storeu_si128((__m128i*) o, v0_0);
    }
    o = (uint8_t*) ((uintptr_t) o + tile_hbytes);
    if (bh != 0) {
      const __m128i v4_0 = _mm_loadu_si128((const __m128i*) i0);
      const uint8_t *i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m128i v4_1 = _mm_loadu_si128((const __m128i*) i1);
      const uint8_t *i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m128i v4_2 = _mm_loadu_si128((const __m128i*) i2);
      const uint8_t *i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m128i v4_3 = _mm_loadu_si128((const __m128i*) i3);
      const uint8_t *i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m128i v4_4 = _mm_loadu_si128((const __m128i*) i4);
      const uint8_t *i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m128i v4_5 = _mm_loadu_si128((const __m128i*) i5);
      const uint8_t *i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m128i v4_6 = _mm_loadu_si128((const __m128i*) i6);
      const uint8_t *i7 = (const uint8_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const __m128i v4_7 = _mm_loadu_si128((const __m128i*) i7);
      const uint8_t *i8 = (const uint8_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const __m128i v4_8 = _mm_loadu_si128((const __m128i*) i8);
      const uint8_t *i9 = (const uint8_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const __m128i v4_9 = _mm_loadu_si128((const __m128i*) i9);
      const uint8_t *i10 = (const uint8_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const __m128i v4_10 = _mm_loadu_si128((const __m128i*) i10);
      const uint8_t *i11 = (const uint8_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const __m128i v4_11 = _mm_loadu_si128((const __m128i*) i11);
      const uint8_t *i12 = (const uint8_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const __m128i v4_12 = _mm_loadu_si128((const __m128i*) i12);
      const uint8_t *i13 = (const uint8_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const __m128i v4_13 = _mm_loadu_si128((const __m128i*) i13);
      const uint8_t *i14 = (const uint8_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const __m128i v4_14 = _mm_loadu_si128((const __m128i*) i14);
      const __m128i v4_15 = _mm_undefined_si128();

      const __m128i v3_0 = _mm_unpacklo_epi8(v4_0, v4_1);
      const __m128i v3_1 = _mm_unpackhi_epi8(v4_0, v4_1);
      const __m128i v3_2 = _mm_unpacklo_epi8(v4_2, v4_3);
      const __m128i v3_3 = _mm_unpackhi_epi8(v4_2, v4_3);
      const __m128i v3_4 = _mm_unpacklo_epi8(v4_4, v4_5);
      const __m128i v3_5 = _mm_unpackhi_epi8(v4_4, v4_5);
      const __m128i v3_6 = _mm_unpacklo_epi8(v4_6, v4_7);
      const __m128i v3_7 = _mm_unpackhi_epi8(v4_6, v4_7);
      const __m128i v3_8 = _mm_unpacklo_epi8(v4_8, v4_9);
      const __m128i v3_9 = _mm_unpackhi_epi8(v4_8, v4_9);
      const __m128i v3_10 = _mm_unpacklo_epi8(v4_10, v4_11);
      const __m128i v3_11 = _mm_unpackhi_epi8(v4_10, v4_11);
      const __m128i v3_12 = _mm_unpacklo_epi8(v4_12, v4_13);
      const __m128i v3_13 = _mm_unpackhi_epi8(v4_12, v4_13);
      const __m128i v3_14 = _mm_unpacklo_epi8(v4_14, v4_15);
      const __m128i v3_15 = _mm_unpackhi_epi8(v4_14, v4_15);

      const __m128i v2_0 = _mm_unpacklo_epi16(v3_0, v3_2);
      const __m128i v2_1 = _mm_unpackhi_epi16(v3_0, v3_2);
      const __m128i v2_2 = _mm_unpacklo_epi16(v3_1, v3_3);
      const __m128i v2_3 = _mm_unpackhi_epi16(v3_1, v3_3);
      const __m128i v2_4 = _mm_unpacklo_epi16(v3_4, v3_6);
      const __m128i v2_5 = _mm_unpackhi_epi16(v3_4, v3_6);
      const __m128i v2_6 = _mm_unpacklo_epi16(v3_5, v3_7);
      const __m128i v2_7 = _mm_unpackhi_epi16(v3_5, v3_7);
      const __m128i v2_8 = _mm_unpacklo_epi16(v3_8, v3_10);
      const __m128i v2_9 = _mm_unpackhi_epi16(v3_8, v3_10);
      const __m128i v2_10 = _mm_unpacklo_epi16(v3_9, v3_11);
      const __m128i v2_11 = _mm_unpackhi_epi16(v3_9, v3_11);
      const __m128i v2_12 = _mm_unpacklo_epi16(v3_12, v3_14);
      const __m128i v2_13 = _mm_unpackhi_epi16(v3_12, v3_14);
      const __m128i v2_14 = _mm_unpacklo_epi16(v3_13, v3_15);
      const __m128i v2_15 = _mm_unpackhi_epi16(v3_13, v3_15);

      const __m128i v1_0 = _mm_unpacklo_epi32(v2_0, v2_4);
      const __m128i v1_1 = _mm_unpackhi_epi32(v2_0, v2_4);
      const __m128i v1_2 = _mm_unpacklo_epi32(v2_1, v2_5);
      const __m128i v1_3 = _mm_unpackhi_epi32(v2_1, v2_5);
      const __m128i v1_4 = _mm_unpacklo_epi32(v2_2, v2_6);
      const __m128i v1_5 = _mm_unpackhi_epi32(v2_2, v2_6);
      const __m128i v1_6 = _mm_unpacklo_epi32(v2_3, v2_7);
      const __m128i v1_7 = _mm_unpackhi_epi32(v2_3, v2_7);
      const __m128i v1_8 = _mm_unpacklo_epi32(v2_8, v2_12);
      const __m128i v1_9 = _mm_unpackhi_epi32(v2_8, v2_12);
      const __m128i v1_10 = _mm_unpacklo_epi32(v2_9, v2_13);
      const __m128i v1_11 = _mm_unpackhi_epi32(v2_9, v2_13);
      const __m128i v1_12 = _mm_unpacklo_epi32(v2_10, v2_14);
      const __m128i v1_13 = _mm_unpackhi_epi32(v2_10, v2_14);
      const __m128i v1_14 = _mm_unpacklo_epi32(v2_11, v2_15);
      const __m128i v1_15 = _mm_unpackhi_epi32(v2_11, v2_15);

      __m128i v0_0 = _mm_unpacklo_epi64(v1_0, v1_8);
      __m128i v0_1 = _mm_unpackhi_epi64(v1_0, v1_8);
      __m128i v0_2 = _mm_unpacklo_epi64(v1_1, v1_9);
      __m128i v0_3 = _mm_unpackhi_epi64(v1_1, v1_9);
      __m128i v0_4 = _mm_unpacklo_epi64(v1_2, v1_10);
      __m128i v0_5 = _mm_unpackhi_epi64(v1_2, v1_10);
      __m128i v0_6 = _mm_unpacklo_epi64(v1_3, v1_11);
      __m128i v0_7 = _mm_unpackhi_epi64(v1_3, v1_11);
      __m128i v0_8 = _mm_unpacklo_epi64(v1_4, v1_12);
      __m128i v0_9 = _mm_unpackhi_epi64(v1_4, v1_12);
      __m128i v0_10 = _mm_unpacklo_epi64(v1_5, v1_13);
      __m128i v0_11 = _mm_unpackhi_epi64(v1_5, v1_13);
      __m128i v0_12 = _mm_unpacklo_epi64(v1_6, v1_14);
      __m128i v0_13 = _mm_unpackhi_epi64(v1_6, v1_14);
      __m128i v0_14 = _mm_unpacklo_epi64(v1_7, v1_15);
      __m128i v0_15 = _mm_unpackhi_epi64(v1_7, v1_15);

      if (bh & 8) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        _mm_storel_epi64((__m128i*) o, v0_15);
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_14);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_13);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_12);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_11);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_10);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_9);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_8);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_7);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_6);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_5);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_4);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_3);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_2);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_0);
        o += 8;
        v0_0 = _mm_unpackhi_epi64(v0_0, v0_0);
        v0_1 = _mm_unpackhi_epi64(v0_1, v0_1);
        v0_2 = _mm_unpackhi_epi64(v0_2, v0_2);
        v0_3 = _mm_unpackhi_epi64(v0_3, v0_3);
        v0_4 = _mm_unpackhi_epi64(v0_4, v0_4);
        v0_5 = _mm_unpackhi_epi64(v0_5, v0_5);
        v0_6 = _mm_unpackhi_epi64(v0_6, v0_6);
        v0_7 = _mm_unpackhi_epi64(v0_7, v0_7);
        v0_8 = _mm_unpackhi_epi64(v0_8, v0_8);
        v0_9 = _mm_unpackhi_epi64(v0_9, v0_9);
        v0_10 = _mm_unpackhi_epi64(v0_10, v0_10);
        v0_11 = _mm_unpackhi_epi64(v0_11, v0_11);
        v0_12 = _mm_unpackhi_epi64(v0_12, v0_12);
        v0_13 = _mm_unpackhi_epi64(v0_13, v0_13);
        v0_14 = _mm_unpackhi_epi64(v0_14, v0_14);
        v0_15 = _mm_unpackhi_epi64(v0_15, v0_15);
      }

      if (bh & 4) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_15));
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_14));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_13));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_12));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_11));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_10));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_9));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_8));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_7));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_6));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_5));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_4));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_3));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_2));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_1));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(v0_0));
        o += 4;
        v0_0 = _mm_srli_epi64(v0_0, 32);
        v0_1 = _mm_srli_epi64(v0_1, 32);
        v0_2 = _mm_srli_epi64(v0_2, 32);
        v0_3 = _mm_srli_epi64(v0_3, 32);
        v0_4 = _mm_srli_epi64(v0_4, 32);
        v0_5 = _mm_srli_epi64(v0_5, 32);
        v0_6 = _mm_srli_epi64(v0_6, 32);
        v0_7 = _mm_srli_epi64(v0_7, 32);
        v0_8 = _mm_srli_epi64(v0_8, 32);
        v0_9 = _mm_srli_epi64(v0_9, 32);
        v0_10 = _mm_srli_epi64(v0_10, 32);
        v0_11 = _mm_srli_epi64(v0_11, 32);
        v0_12 = _mm_srli_epi64(v0_12, 32);
        v0_13 = _mm_srli_epi64(v0_13, 32);
        v0_14 = _mm_srli_epi64(v0_14, 32);
        v0_15 = _mm_srli_epi64(v0_15, 32);
      }
      if (bh & 2) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_15));
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_14));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_13));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_12));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_11));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_10));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_9));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_8));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_7));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_6));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_5));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_4));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_3));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_2));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_1));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_0));
        o += 2;
        v0_0 = _mm_srli_epi32(v0_0, 16);
        v0_1 = _mm_srli_epi32(v0_1, 16);
        v0_2 = _mm_srli_epi32(v0_2, 16);
        v0_3 = _mm_srli_epi32(v0_3, 16);
        v0_4 = _mm_srli_epi32(v0_4, 16);
        v0_5 = _mm_srli_epi32(v0_5, 16);
        v0_6 = _mm_srli_epi32(v0_6, 16);
        v0_7 = _mm_srli_epi32(v0_7, 16);
        v0_8 = _mm_srli_epi32(v0_8, 16);
        v0_9 = _mm_srli_epi32(v0_9, 16);
        v0_10 = _mm_srli_epi32(v0_10, 16);
        v0_11 = _mm_srli_epi32(v0_11, 16);
        v0_12 = _mm_srli_epi32(v0_12, 16);
        v0_13 = _mm_srli_epi32(v0_13, 16);
        v0_14 = _mm_srli_epi32(v0_14, 16);
        v0_15 = _mm_srli_epi32(v0_15, 16);
      }
      if (bh & 1) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        *o = (uint8_t) _mm_cvtsi128_si32(v0_15);
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_14);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_13);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_12);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_11);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_10);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_9);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_8);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_7);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_6);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_5);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_4);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_3);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_2);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_0);
      }
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o = (uint8_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x8_zip_x2_ukernel__sse2(
    size_t n,
    const uint8_t* input,
    uint8_t* output)
{
  const uint8_t* x = input;
  const uint8_t* y = (const uint8_t*) ((uintptr_t) x + n);
  uint8_t* o = output;

  if (n >= 16) {
    do {
      const __m128i vx = _mm_loadu_si128((const __m128i*) x);
      x += 16;
      const __m128i vy = _mm_loadu_si128((const __m128i*) y);
      y += 16;
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      _mm_storeu_si128((__m128i*) o, vxy_lo);
      _mm_storeu_si128((__m128i*) (o + 16), vxy_hi);
      o = (void*) ((uintptr_t) o + 32);
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      const __m128i vx = _mm_loadu_si128((const __m128i*) ((uintptr_t) x + address_increment));
      const __m128i vy = _mm_loadu_si128((const __m128i*) ((uintptr_t) y + address_increment));
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      o = (void*) ((uintptr_t) o + address_increment * 2);
      _mm_storeu_si128((__m128i*) o, vxy_lo);
      _mm_storeu_si128((__m128i*) o + 1, vxy_hi);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      o[0] = vx;
      o[1] = vy;
      o += 2;
    } while (--n != 0);
  }
}

void xnn_x8_zip_x3_ukernel__sse2(
    size_t n,
    const uint8_t* input,
    uint8_t* output)
{
  const uint8_t* x = input;
  const uint8_t* y = (const uint8_t*) ((uintptr_t) x + n);
  const uint8_t* z = (const uint8_t*) ((uintptr_t) y + n);
  uint8_t* o = output;

  if (n >= 16) {
    const __m128i vmask0x00FF00FF = _mm_set1_epi16(0x00FF);
    const __m128i vmask0x0000FFFF = _mm_set1_epi32(0x0000FFFF);
    do {
      // vx  = ( x15, x14, x13, x12, x11, x10,  x9,  x8,  x7,  x6,  x5,  x4, x3, x2, x1, x0 )
      const __m128i vx = _mm_loadu_si128((const __m128i*) x);
      x += 16;
      // vy  = ( y15, y14, y13, y12, y11, y10,  y9,  y8,  y7,  y6,  y5,  y4, y3, y2, y1, y0 )
      const __m128i vy = _mm_loadu_si128((const __m128i*) y);
      y += 16;
      // vz  = ( z15, z14, z13, z12, z11, z10,  z9,  z8,  z7,  z6,  z5,  z4, z3, z2, z1, z0 )
      const __m128i vz = _mm_loadu_si128((const __m128i*) z);
      z += 16;

      // vxeye     = ( y14, x14, y12, x12, y10, x10,  y8,  x8,  y6,  x6,  y4,  x4,  y2,  x2,  y0,  x0 )
      const __m128i vxeye = _mm_or_si128(_mm_and_si128(vx, vmask0x00FF00FF), _mm_slli_epi16(vy, 8));
      // vyozo     = ( z15, y15, z13, y13, z11, y11,  z9,  y9,  z7,  y7,  z5,  y5,  z3,  y3,  z1,  y1 )
      const __m128i vyozo = _mm_or_si128(_mm_andnot_si128(vmask0x00FF00FF, vz), _mm_srli_epi16(vy, 8));
      // vzoxo     = ( x15, z14, x13, z12, x11, z10,  x9,  z8,  x7,  z6,  x5,  z4,  x3,  z2,  x1,  z0 )
      const __m128i vzexo = _mm_or_si128(_mm_and_si128(vz, vmask0x00FF00FF), _mm_andnot_si128(vmask0x00FF00FF, vx));

      // vxeyezexo = ( x13, z12, y12, x12,  x9,  z8,  y8,  x8,  x5,  z4,  y4,  x4,  x1,  z0,  y0,  x0 )
      const __m128i vxeyezexo = _mm_or_si128(_mm_and_si128(vxeye, vmask0x0000FFFF), _mm_slli_epi32(vzexo, 16));
      // vyozoxeye = ( y14, x14, z13, y13, y10, x10,  z9,  y9,  y6,  x6,  z5,  y5,  y2,  x2,  z1,  y1 )
      const __m128i vyozoxeye = _mm_or_si128(_mm_and_si128(vyozo, vmask0x0000FFFF), _mm_andnot_si128(vmask0x0000FFFF, vxeye));
      // vzexoyozo = ( z15, y15, x15, z14, z11, y11, x11, z10,  z7,  y7,  x7,  z6,  z3,  y3,  x3,  z2 )
      const __m128i vzexoyozo = _mm_or_si128(_mm_andnot_si128(vmask0x0000FFFF, vyozo), _mm_srli_epi32(vzexo, 16));

      // vtemp0    = ( x13, z12, y12, x12,  x5,  z4,  y4,  x4, z11, y11, x11, z10,  z3,  y3,  x3,  z2 )
      const __m128i vtemp0 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vzexoyozo), _mm_castsi128_ps(vxeyezexo), _MM_SHUFFLE(3, 1, 2, 0)));
      // vtemp1    = ( y10, x10,  z9,  y9,  y2,  x2,  z1,  y1,  x9,  z8,  y8,  x8,  x1,  z0,  y0,  x0 )
      const __m128i vtemp1 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vxeyezexo), _mm_castsi128_ps(vyozoxeye), _MM_SHUFFLE(2, 0, 2, 0)));
      // vtemp2    = ( z15, y15, x15, z14,  z7,  y7,  x7,  z6, y14, x14, z13, y13,  y6,  x6,  z5,  y5 )
      const __m128i vtemp2 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vyozoxeye), _mm_castsi128_ps(vzexoyozo), _MM_SHUFFLE(3, 1, 3, 1)));

      // vxyz0     = (  x5,  z4,  y4,  x4,  z3,  y3,  x3,  z2,  y2,  x2,  z1,  y1,  x1,  z0,  y0,  x0 )
      const __m128i vxyz0 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vtemp1), _mm_castsi128_ps(vtemp0), _MM_SHUFFLE(2, 0, 2, 0)));
      // vxyz1     = ( y10, x10,  z9,  y9,  x9,  z8,  y8,  x8,  z7,  y7,  x7,  z6,  y6,  x6,  z5,  y5 )
      const __m128i vxyz1 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vtemp2), _mm_castsi128_ps(vtemp1), _MM_SHUFFLE(3, 1, 2, 0)));
      // vxyz2     = ( z15, y15, x15, z14, y14, x14, z13, y13, x13, z12, y12, x12, z11, y11, x11, z10 )
      const __m128i vxyz2 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vtemp0), _mm_castsi128_ps(vtemp2), _MM_SHUFFLE(3, 1, 3, 1)));

      _mm_storeu_si128((__m128i*) o, vxyz0);
      _mm_storeu_si128((__m128i*) o + 1, vxyz1);
      _mm_storeu_si128((__m128i*) o + 2, vxyz2);
      o += 48;
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      // vx  = ( x15, x14, x13, x12, x11, x10,  x9,  x8,  x7,  x6,  x5,  x4, x3, x2, x1, x0 )
      const __m128i vx = _mm_loadu_si128((const __m128i*) ((uintptr_t) x + address_increment));
      // vy  = ( y15, y14, y13, y12, y11, y10,  y9,  y8,  y7,  y6,  y5,  y4, y3, y2, y1, y0 )
      const __m128i vy = _mm_loadu_si128((const __m128i*) ((uintptr_t) y + address_increment));
      // vz  = ( z15, z14, z13, z12, z11, z10,  z9,  z8,  z7,  z6,  z5,  z4, z3, z2, z1, z0 )
      const __m128i vz = _mm_loadu_si128((const __m128i*) ((uintptr_t) z + address_increment));

      // vxeye     = ( y14, x14, y12, x12, y10, x10,  y8,  x8,  y6,  x6,  y4,  x4,  y2,  x2,  y0,  x0 )
      const __m128i vxeye = _mm_or_si128(_mm_and_si128(vx, vmask0x00FF00FF), _mm_slli_epi16(vy, 8));
      // vyozo     = ( z15, y15, z13, y13, z11, y11,  z9,  y9,  z7,  y7,  z5,  y5,  z3,  y3,  z1,  y1 )
      const __m128i vyozo = _mm_or_si128(_mm_andnot_si128(vmask0x00FF00FF, vz), _mm_srli_epi16(vy, 8));
      // vzoxo     = ( x15, z14, x13, z12, x11, z10,  x9,  z8,  x7,  z6,  x5,  z4,  x3,  z2,  x1,  z0 )
      const __m128i vzexo = _mm_or_si128(_mm_and_si128(vz, vmask0x00FF00FF), _mm_andnot_si128(vmask0x00FF00FF, vx));

      // vxeyezexo = ( x13, z12, y12, x12,  x9,  z8,  y8,  x8,  x5,  z4,  y4,  x4,  x1,  z0,  y0,  x0 )
      const __m128i vxeyezexo = _mm_or_si128(_mm_and_si128(vxeye, vmask0x0000FFFF), _mm_slli_epi32(vzexo, 16));
      // vyozoxeye = ( y14, x14, z13, y13, y10, x10,  z9,  y9,  y6,  x6,  z5,  y5,  y2,  x2,  z1,  y1 )
      const __m128i vyozoxeye = _mm_or_si128(_mm_and_si128(vyozo, vmask0x0000FFFF), _mm_andnot_si128(vmask0x0000FFFF, vxeye));
      // vzexoyozo = ( z15, y15, x15, z14, z11, y11, x11, z10,  z7,  y7,  x7,  z6,  z3,  y3,  x3,  z2 )
      const __m128i vzexoyozo = _mm_or_si128(_mm_andnot_si128(vmask0x0000FFFF, vyozo), _mm_srli_epi32(vzexo, 16));

      // vtemp0    = ( x13, z12, y12, x12,  x5,  z4,  y4,  x4, z11, y11, x11, z10,  z3,  y3,  x3,  z2 )
      const __m128i vtemp0 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vzexoyozo), _mm_castsi128_ps(vxeyezexo), _MM_SHUFFLE(3, 1, 2, 0)));
      // vtemp1    = ( y10, x10,  z9,  y9,  y2,  x2,  z1,  y1,  x9,  z8,  y8,  x8,  x1,  z0,  y0,  x0 )
      const __m128i vtemp1 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vxeyezexo), _mm_castsi128_ps(vyozoxeye), _MM_SHUFFLE(2, 0, 2, 0)));
      // vtemp2    = ( z15, y15, x15, z14,  z7,  y7,  x7,  z6, y14, x14, z13, y13,  y6,  x6,  z5,  y5 )
      const __m128i vtemp2 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vyozoxeye), _mm_castsi128_ps(vzexoyozo), _MM_SHUFFLE(3, 1, 3, 1)));

      // vxyz0     = (  x5,  z4,  y4,  x4,  z3,  y3,  x3,  z2,  y2,  x2,  z1,  y1,  x1,  z0,  y0,  x0 )
      const __m128i vxyz0 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vtemp1), _mm_castsi128_ps(vtemp0), _MM_SHUFFLE(2, 0, 2, 0)));
      // vxyz1     = ( y10, x10,  z9,  y9,  x9,  z8,  y8,  x8,  z7,  y7,  x7,  z6,  y6,  x6,  z5,  y5 )
      const __m128i vxyz1 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vtemp2), _mm_castsi128_ps(vtemp1), _MM_SHUFFLE(3, 1, 2, 0)));
      // vxyz2     = ( z15, y15, x15, z14, y14, x14, z13, y13, x13, z12, y12, x12, z11, y11, x11, z10 )
      const __m128i vxyz2 = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(vtemp0), _mm_castsi128_ps(vtemp2), _MM_SHUFFLE(3, 1, 3, 1)));

      o = (uint8_t*) ((uintptr_t) o + address_increment * 3);
      _mm_storeu_si128((__m128i*) o, vxyz0);
      _mm_storeu_si128((__m128i*) o + 1, vxyz1);
      _mm_storeu_si128((__m128i*) o + 2, vxyz2);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      const uint8_t vz = *z++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o += 3;
    } while (--n != 0);
  }
}

void xnn_x8_zip_x4_ukernel__sse2(
    size_t n,
    const uint8_t* input,
    uint8_t* output)
{
  const uint8_t* x = input;
  const uint8_t* y = (const uint8_t*) ((uintptr_t) x + n);
  const uint8_t* z = (const uint8_t*) ((uintptr_t) y + n);
  const uint8_t* w = (const uint8_t*) ((uintptr_t) z + n);
  uint8_t* o = output;

  if (n >= 16) {
    do {
      const __m128i vx = _mm_loadu_si128((const __m128i*) x);
      x += 16;
      const __m128i vy = _mm_loadu_si128((const __m128i*) y);
      y += 16;
      const __m128i vz = _mm_loadu_si128((const __m128i*) z);
      z += 16;
      const __m128i vw = _mm_loadu_si128((const __m128i*) w);
      w += 16;
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);
      const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);
      const __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
      const __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
      const __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
      const __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);
      _mm_storeu_si128((__m128i*) o, vxyzw0);
      _mm_storeu_si128((__m128i*) o + 1, vxyzw1);
      _mm_storeu_si128((__m128i*) o + 2, vxyzw2);
      _mm_storeu_si128((__m128i*) o + 3, vxyzw3);
      o = (void*) ((uintptr_t) o + 64);
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      const __m128i vx = _mm_loadu_si128((const __m128i*) ((uintptr_t) x + address_increment));
      const __m128i vy = _mm_loadu_si128((const __m128i*) ((uintptr_t) y + address_increment));
      const __m128i vz = _mm_loadu_si128((const __m128i*) ((uintptr_t) z + address_increment));
      const __m128i vw = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + address_increment));
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);
      const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);
      const __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
      const __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
      const __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
      const __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);
      o = (void*) ((uintptr_t) o + address_increment * 4);
      _mm_storeu_si128((__m128i*) o, vxyzw0);
      _mm_storeu_si128((__m128i*) o + 1, vxyzw1);
      _mm_storeu_si128((__m128i*) o + 2, vxyzw2);
      _mm_storeu_si128((__m128i*) o + 3, vxyzw3);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      const uint8_t vz = *z++;
      const uint8_t vw = *w++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o[3] = vw;
      o += 4;
    } while (--n != 0);
  }
}

void xnn_x8_zip_xm_ukernel__sse2(
    size_t n,
    size_t m,
    const uint8_t* input,
    uint8_t* output)
{
  const uint8_t* w = input;
  const size_t input_increment = n * 3;
  const size_t output_increment = 4 - m * n;
  const uint8_t* last_input = w + n * (m - 1);
  uint8_t* last_output = (uint8_t*) ((uintptr_t) output + (m - 4));

  if (n >= 8) {
    for (size_t i = 0; i < m; i += 4) {
      size_t k = n;
      w = (const uint8_t*) ((uintptr_t) w + input_increment);
      if (w >= last_input) {
        w = last_input;
      }
      const uint8_t* z = (const uint8_t*) ((uintptr_t) w - n);
      const uint8_t* y = (const uint8_t*) ((uintptr_t) z - n);
      const uint8_t* x = (const uint8_t*) ((uintptr_t) y - n);
      while (k >= 16) {
        const __m128i vx = _mm_loadu_si128((const __m128i*) x);
        x += 16;
        const __m128i vy = _mm_loadu_si128((const __m128i*) y);
        y += 16;
        const __m128i vz = _mm_loadu_si128((const __m128i*) z);
        z += 16;
        const __m128i vw = _mm_loadu_si128((const __m128i*) w);
        w += 16;
        const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
        const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
        const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);
        const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);
        __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
        __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
        __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
        __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_unpackhi_epi64(vxyzw1, vxyzw1);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw2));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw2 = _mm_shufflelo_epi16(vxyzw2, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw2));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw2 = _mm_unpackhi_epi64(vxyzw2, vxyzw2);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw2));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw2 = _mm_shufflelo_epi16(vxyzw2, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw2));
        output = (uint8_t*) ((uintptr_t) output + m);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw3));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw3 = _mm_shufflelo_epi16(vxyzw3, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw3));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw3 = _mm_unpackhi_epi64(vxyzw3, vxyzw3);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw3));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw3 = _mm_shufflelo_epi16(vxyzw3, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw3));
        output = (uint8_t*) ((uintptr_t) output + m);
        k -= 16;
      };
      if (k >= 8) {
        const __m128i vx = _mm_loadl_epi64((const __m128i*) x);
        x += 8;
        const __m128i vy = _mm_loadl_epi64((const __m128i*) y);
        y += 8;
        const __m128i vz = _mm_loadl_epi64((const __m128i*) z);
        z += 8;
        const __m128i vw = _mm_loadl_epi64((const __m128i*) w);
        w += 8;
        const __m128i vxy = _mm_unpacklo_epi8(vx, vy);
        const __m128i vzw = _mm_unpacklo_epi8(vz, vw);
        __m128i vxyzw0 = _mm_unpacklo_epi16(vxy, vzw);
        __m128i vxyzw1 = _mm_unpackhi_epi16(vxy, vzw);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_unpackhi_epi64(vxyzw1, vxyzw1);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        k -= 8;
      }
      if (k != 0) {
        const size_t address_decrement = 8 - k;
        x -= address_decrement;
        y -= address_decrement;
        z -= address_decrement;
        w -= address_decrement;
        const __m128i vshift = _mm_cvtsi32_si128((int) address_decrement * 8);

        const __m128i vx = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) x), vshift);
        const __m128i vy = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) y), vshift);
        const __m128i vz = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) z), vshift);
        const __m128i vw = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) w), vshift);
        w += 8;
        const __m128i vxy = _mm_unpacklo_epi8(vx, vy);
        const __m128i vzw = _mm_unpacklo_epi8(vz, vw);
        __m128i vxyzw0 = _mm_unpacklo_epi16(vxy, vzw);
        __m128i vxyzw1 = _mm_unpackhi_epi16(vxy, vzw);

        if (k & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = vxyzw1;
        }

        if (k & 2) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
        }
        if (k & 1) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
        }
      }
      output = (uint8_t*) ((uintptr_t) output + output_increment);
      if (output > last_output) {
        output = last_output;
      }
    }
  } else {
    const uint8_t* i = input;
    uint8_t* o = output;
    size_t k = n;
    do {
      size_t l = m;
      const uint8_t* ii = i++;
      do {
        *o++ = *ii;
        ii += n;
      } while (--l != 0);
    } while (--k != 0);
  }
}

void xnn_xx_fill_ukernel__sse2_u64(
    size_t rows,
    size_t channels,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern)
{
  assert(rows != 0);
  assert(channels != 0);

  const size_t output_increment = output_stride - channels;

  const __m128i vfill = _mm_shuffle_epi32(_mm_cvtsi32_si128(fill_pattern), _MM_SHUFFLE(0, 0, 0, 0));
  do {
    size_t c = channels;
    for (; c >= 64 * sizeof(uint8_t); c -= 64 * sizeof(uint8_t)) {
      _mm_storeu_si128((__m128i*) output, vfill);
      _mm_storeu_si128((__m128i*) output + 1, vfill);
      _mm_storeu_si128((__m128i*) output + 2, vfill);
      _mm_storeu_si128((__m128i*) output + 3, vfill);
      output = ((uint8_t*) output + 64);
    }
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      _mm_storeu_si128((__m128i*) output, vfill);
      output = ((uint8_t*) output + 16);
    }
    if XNN_UNLIKELY(c != 0) {
      if XNN_LIKELY(c & (8 * sizeof(uint8_t))) {
        _mm_storel_epi64(output, vfill);
        output = ((uint8_t*) output + 8);
      }
      if XNN_LIKELY(c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, fill_pattern);
        output = ((uint8_t*) output + 4);
      }
      uint32_t vfill_subpattern = fill_pattern;
      if XNN_LIKELY(c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_subpattern);
        vfill_subpattern >>= 16;
        output = ((uint8_t*) output + 2);
      }
      if XNN_LIKELY(c & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = ((uint8_t*) output + 1);
      }
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}

void xnn_xx_pad_ukernel_p16__sse2_u16(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern) XNN_OOB_READS
{
  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const __m128i vfill_pattern = _mm_shuffle_epi32(_mm_cvtsi32_si128((int) fill_pattern), _MM_SHUFFLE(0, 0, 0, 0));
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      for (; l >= 16 * sizeof(uint8_t); l -= 16 * sizeof(uint8_t)) {
        _mm_storeu_si128((__m128i*) output, vfill_pattern);
        output = (uint8_t*) output + 16;
      }
      if (l & (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vfill_pattern);
        output = (uint8_t*) output + 8;
      }
      uint32_t vfill_subpattern = fill_pattern;
      if (l & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, vfill_subpattern);
        output = (uint8_t*) output + 4;
      }
      if (l & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, vfill_subpattern);
        vfill_subpattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if (l & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = (uint8_t*) output + 1;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      const __m128i vdata = _mm_loadu_si128((const __m128i*) input);
      input = (const uint8_t*) input + 16;

      _mm_storeu_si128((__m128i*) output, vdata);
      output = (uint8_t*) output + 16;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128i vdata = _mm_loadu_si128((const __m128i*) input);
      input = (const void*) ((uintptr_t) input + c);
      if (c & (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vdata);
        vdata = _mm_unpackhi_epi64(vdata, vdata);
        output = (uint8_t*) output + 8;
      }
      if (c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vdata));
        vdata = _mm_srli_epi64(vdata, 32);
        output = (uint8_t*) output + 4;
      }
      uint32_t vsubdata = (uint32_t) _mm_cvtsi128_si32(vdata);
      if (c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vsubdata);
        vsubdata >>= 16;
        output = (uint8_t*) output + 2;
      }
      if (c & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vsubdata;
        output = (uint8_t*) output + 1;
      }
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      for (; r >= 16 * sizeof(uint8_t); r -= 16 * sizeof(uint8_t)) {
        _mm_storeu_si128((__m128i*) output, vfill_pattern);
        output = (uint8_t*) output + 16;
      }
      if (r & (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vfill_pattern);
        output = (uint8_t*) output + 8;
      }
      uint32_t vfill_subpattern = fill_pattern;
      if (r & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, vfill_subpattern);
        output = (uint8_t*) output + 4;
      }
      if (r & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_subpattern);
        vfill_subpattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if (r & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = (uint8_t*) output + 1;
      }
    }

    input = (const void*) ((uintptr_t) input + input_increment);
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}

void xnn_f32_vabs_ukernel__sse2_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2 * xnn_simd_size_f32;

    const xnn_simd_f32_t vy0 = xnn_abs_f32(vx0);
    const xnn_simd_f32_t vy1 = xnn_abs_f32(vx1);

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    output += 2 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_abs_f32(vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_abs_f32(vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysign_ukernel__sse2_u8(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vsign_0 = xnn_loadu_f32(sign);
    xnn_simd_f32_t vsign_1 = xnn_loadu_f32(sign + 1 * xnn_simd_size_f32);
    sign += 8;

    vsign_0 = xnn_and_f32(vsign_0, vsign_mask);
    vsign_1 = xnn_and_f32(vsign_1, vsign_mask);

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    mag += 8;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign_0, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign_1, vmag_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vsign = xnn_load_tail_f32(sign, batch >> XNN_LOG2_SIZEOF_FLOAT);
    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysignc_ukernel__sse2_u8(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    mag += 8;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign, vmag_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u12(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA || (XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR)
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1638283730e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1158981323e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, 7.9788452387e-01f);
  XNN_SIMD_CONST_F32(valpha_3, 6.6972173750e-02f);
  XNN_SIMD_CONST_F32(valpha_5, 9.3065137044e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 3.2973114867e-04f);
  XNN_SIMD_CONST_F32(valpha_9, 1.2609783880e-05f);
  XNN_SIMD_CONST_F32(valpha_11, 4.5835321316e-08f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.5060352683e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.8431978077e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 1.8622842617e-03f);
  XNN_SIMD_CONST_F32(vbeta_8, 7.2267655923e-05f);
  XNN_SIMD_CONST_F32(vbeta_10, 1.1988805682e-06f);

  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const xnn_simd_f32_t vx_orig_0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx_orig_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx_orig_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    input += 12;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx_0 = xnn_min_f32(vmax_abs_x, vx_orig_0);
    xnn_simd_f32_t vx_1 = xnn_min_f32(vmax_abs_x, vx_orig_1);
    xnn_simd_f32_t vx_2 = xnn_min_f32(vmax_abs_x, vx_orig_2);
    vx_0 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_1);
    vx_2 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_2);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);
    const xnn_simd_f32_t vx2_2 = xnn_mul_f32(vx_2, vx_2);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_11, valpha_9);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_11, valpha_9);
    xnn_simd_f32_t vp_2 = xnn_fmadd_f32(vx2_2, valpha_11, valpha_9);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_7);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_7);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_1);
    vp_2 = xnn_fmadd_f32(vx2_2, vp_2, valpha_1);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);
    vp_2 = xnn_mul_f32(vx_2, vp_2);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_2 = xnn_fmadd_f32(vx2_2, vbeta_10, vbeta_8);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_6);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_6);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_6);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_4);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_4);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);
    vq_2 = xnn_fmadd_f32(vx2_2, vq_2, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t verf_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t verf_1 = xnn_div_f32(vp_1, vq_1);
    const xnn_simd_f32_t verf_2 = xnn_div_f32(vp_2, vq_2);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(xnn_mul_f32(vx_orig_0, vhalf),
                                        xnn_add_f32(verf_0, vone));
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(xnn_mul_f32(vx_orig_1, vhalf),
                                        xnn_add_f32(verf_1, vone));
    const xnn_simd_f32_t vy_2 = xnn_mul_f32(xnn_mul_f32(vx_orig_2, vhalf),
                                        xnn_add_f32(verf_2, vone));

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    output += 12;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx_orig = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx = xnn_min_f32(vmax_abs_x, vx_orig);
    vx = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_11, valpha_9);
    vp = xnn_fmadd_f32(vx2, vp, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_10, vbeta_8);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f32_t verf =  xnn_div_f32(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy = xnn_mul_f32(xnn_mul_f32(vx_orig, vhalf),
                                          xnn_add_f32(verf, vone));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx_orig = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

  // See above for comments.
  xnn_simd_f32_t vx = xnn_min_f32(vmax_abs_x, vx_orig);
  vx = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx);
  const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
  xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_11, valpha_9);
  vp = xnn_fmadd_f32(vx2, vp, valpha_7);
  vp = xnn_fmadd_f32(vx2, vp, valpha_5);
  vp = xnn_fmadd_f32(vx2, vp, valpha_3);
  vp = xnn_fmadd_f32(vx2, vp, valpha_1);
  vp = xnn_mul_f32(vx, vp);
  xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_10, vbeta_8);
  vq = xnn_fmadd_f32(vx2, vq, vbeta_6);
  vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
  vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
  vq = xnn_fmadd_f32(vx2, vq, vone);
  const xnn_simd_f32_t verf =  xnn_div_f32(vp, vq);
  const xnn_simd_f32_t vy = xnn_mul_f32(xnn_mul_f32(vx_orig, vhalf),
                                        xnn_add_f32(verf, vone));

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

static XNN_INLINE xnn_simd_f32_t xnn_signed_getexp_f32(xnn_simd_f32_t a) {
  // The bits of IEE754 single-precision floating-point format are:
  //
  //   s | e e e e e e e e | m m m m m m m m m m m m m m m m m m m m m m m
  //
  // We start by masking out the sign and exponent and shifting it 8 bits to the
  // right arithmetically, i.e. extending by the leftmost sign bit:
  //
  //   s | s s s s s s s s | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // These bits are then `or`-ed with `256.0f`, which has a biased exponent of
  // `135` and all mantissa bit set to zero. This is equivalent to adding the
  // biased integer exponent to `256.0`:
  //
  //   0 | 1 0 0 0 0 1 1 1 | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // We can re-extract the exponent as a `float` value by subtracting `256.0`
  // plus the exponent bias `127.0`, i.e. `383.0`.
  //
  // Note that if the sign bit is `1`, we end up with the floating point bits:
  //
  //   1 | 1 1 1 1 1 1 1 1 | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // Which is `-NaN` if the exponent is non-zero, and `-Inf` if the exponent is
  // zero (e.g. the input was `0.0f` or denormal).

  // Some useful constants.
  XNN_SIMD_CONST_F32(sign_mask, -0.0f);
  XNN_SIMD_CONST_U32(sign_and_exp_mask, 0xff800000);
  XNN_SIMD_CONST_F32(bias_256, 256.0f);
  XNN_SIMD_CONST_F32(bias_383, 383.0f);

  // If `a` is `0.0f`, flip its sign bit so that we return `-Inf`.
  a = xnn_or_f32(xnn_and_f32(xnn_cmpeq_f32(a, xnn_zero_f32()), sign_mask), a);

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const xnn_simd_f32_t exp =
      xnn_sra_f32(xnn_and_f32(a, sign_and_exp_mask), 8);

  // Add the shifted exponent to `256.0f` by copying its bits to the mantissa,
  // then subtract out `383.0f`, i.e. the original `256.0f` plus the `127`
  // exponent bias, resulting in the unbiased exponent.
  return xnn_sub_f32(xnn_or_f32(bias_256, exp), bias_383);
}

void xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vln2, M_LN2);
  XNN_SIMD_CONST_U32(vmantissa_bits_mask, 0x007FFFFFUL);

  // Note that these two values are not _exactly_ `(float)M_SQRT2` and
  // `(float)M_SQRT1_2`, but are instead chosen such that their product is
  // exactly `1.0f` when evaluated in `float` precision.
  XNN_SIMD_CONST_F32(vsqrt2, 1.4142134190e+00);
  XNN_SIMD_CONST_F32(vsqrt1_2, 7.0710688829e-01);

  // The monomial coefficients of the numerator polynomial.
  // XNN_SIMD_CONST_F32(valpha_0, 0.0f);
  // XNN_SIMD_CONST_F32(valpha_1, 1.0f);
  // XNN_SIMD_CONST_F32(valpha_2, 1.0f);
  XNN_SIMD_CONST_F32(valpha_3, 1.824996918440e-01);

  // The monomial coefficients of the denominator polynomial.
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_1, 1.5f);
  XNN_SIMD_CONST_F32(vbeta_2, 0.599170029163);
  XNN_SIMD_CONST_F32(vbeta_3, 0.049584995955);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 8;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx_0 = xnn_mul_f32(vx_0, vsqrt2);
    vx_1 = xnn_mul_f32(vx_1, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp_0 = xnn_signed_getexp_f32(vx_0);
    const xnn_simd_f32_t vexp_1 = xnn_signed_getexp_f32(vx_1);

    // Normalize `x` to an exponent of zero.
    vx_0 = xnn_or_f32(xnn_and_f32(vx_0, vmantissa_bits_mask), vone);
    vx_1 = xnn_or_f32(xnn_and_f32(vx_1, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx_0 = xnn_sub_f32(xnn_mul_f32(vx_0, vsqrt1_2), vone);
    vx_1 = xnn_sub_f32(xnn_mul_f32(vx_1, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx_0, valpha_3, vone);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx_1, valpha_3, vone);
    vp_0 = xnn_fmadd_f32(vx_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx_1, vp_1, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx_0, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx_1, vbeta_3, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vbeta_1);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vbeta_1);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy_0 = xnn_fmadd_f32(vexp_0, vln2, vy_0);
    vy_1 = xnn_fmadd_f32(vexp_1, vln2, vy_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = xnn_mul_f32(vx, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp = xnn_signed_getexp_f32(vx);

    // Normalize `x` to an exponent of zero.
    vx = xnn_or_f32(xnn_and_f32(vx, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx = xnn_sub_f32(xnn_mul_f32(vx, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx, valpha_3, vone);
    vp = xnn_fmadd_f32(vx, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f32(vx, vq, vbeta_1);
    vq = xnn_fmadd_f32(vx, vq, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = xnn_fmadd_f32(vexp, vln2, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // See the loop above for comments.
    vx = xnn_mul_f32(vx, vsqrt2);
    const xnn_simd_f32_t vexp = xnn_signed_getexp_f32(vx);
    vx = xnn_or_f32(xnn_and_f32(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f32(xnn_mul_f32(vx, vsqrt1_2), vone);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx, valpha_3, vone);
    vp = xnn_fmadd_f32(vx, vp, vone);
    vp = xnn_mul_f32(vx, vp);
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f32(vx, vq, vbeta_1);
    vq = xnn_fmadd_f32(vx, vq, vone);
    xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);
    vy = xnn_fmadd_f32(vexp, vln2, vy);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vneg_ukernel__sse2_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2 * xnn_simd_size_f32;

    const xnn_simd_f32_t vy0 = xnn_neg_f32(vx0);
    const xnn_simd_f32_t vy1 = xnn_neg_f32(vx1);

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    output += 2 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_neg_f32(vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_neg_f32(vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vrcopysignc_ukernel__sse2_u8(
    size_t batch,
    const float* sign,
    const float* mag,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(sign != NULL);
  assert(mag != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vmag = xnn_abs_f32(xnn_set1_f32(*mag));

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vsign_0 = xnn_loadu_f32(sign);
    xnn_simd_f32_t vsign_1 = xnn_loadu_f32(sign + 1 * xnn_simd_size_f32);
    sign += 8;

    vsign_0 = xnn_and_f32(vsign_0, vsign_mask);
    vsign_1 = xnn_and_f32(vsign_1, vsign_mask);

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign_0, vmag);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign_1, vmag);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vsign = xnn_load_tail_f32(sign, batch >> XNN_LOG2_SIZEOF_FLOAT);
    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsqr_ukernel__sse2_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2 * xnn_simd_size_f32;

    const xnn_simd_f32_t vy0 = xnn_mul_f32(vx0, vx0);
    const xnn_simd_f32_t vy1 = xnn_mul_f32(vx1, vx1);

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    output += 2 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vtanh_ukernel__sse2_rational_9_6_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.646893501282f);
    XNN_SIMD_CONST_F32(vmin_x, -7.646893501282f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.623543739319f);
    XNN_SIMD_CONST_F32(vmin_x, -7.623543739319f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, -9.022999554873e-03f);
  XNN_SIMD_CONST_F32(valpha_3, -1.146968104877e-03f);
  XNN_SIMD_CONST_F32(valpha_5, -2.432360815874e-05f);
  XNN_SIMD_CONST_F32(valpha_7, -6.458659385089e-08f);
  XNN_SIMD_CONST_F32(valpha_9, 5.535878699892e-11f);

  // The monomial coefficients of the denominator polynomial (even).
  XNN_SIMD_CONST_F32(vbeta_0, -9.023001417518e-03f);
  XNN_SIMD_CONST_F32(vbeta_2, -4.154618829489e-03f);
  XNN_SIMD_CONST_F32(vbeta_4, -2.061512641376e-04f);
  XNN_SIMD_CONST_F32(vbeta_6, -1.774490101525e-06f);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 8;

    // Clamp the inputs to the interpolation range.
    vx_0 = xnn_min_f32(vmax_x, vx_0);
    vx_1 = xnn_min_f32(vmax_x, vx_1);
    vx_0 = xnn_max_f32(vmin_x, vx_0);
    vx_1 = xnn_max_f32(vmin_x, vx_1);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_9, valpha_7);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_9, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_1);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_6, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_0);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_6, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_6, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
