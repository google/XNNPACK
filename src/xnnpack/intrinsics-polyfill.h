// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"

static XNN_INTRINSIC uint32_t broadcast2x_uint16(uint16_t x) {
  return (uint32_t) x | ((uint32_t) x) << 16;
}

#if defined(__SSE2__)
#include <emmintrin.h>

// GCC pre-11, Clang pre-8, Android NDK Clang pre-8.0.7, Apple Clang pre-11, and ICC pre-16
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && __GNUC__ < 11) || \
    (defined(__clang__) && !defined(__apple_build_version__) && (__clang_major__ < 8)) || \
    (defined(__clang__) && defined(__ANDROID__) && (__clang_major__ == 8) && (__clang_minor__ == 0) && (__clang_patchlevel__ < 7)) || \
    (defined(__clang__) && defined(__apple_build_version__) && (__apple_build_version__ < 11000000)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1600))

static XNN_INTRINSIC
void _mm_storeu_si32(void* address, __m128i v) {
  unaligned_store_u32(address, (uint32_t) _mm_cvtsi128_si32(v));
}

static XNN_INTRINSIC
void _mm_storeu_si16(void* address, __m128i v) {
  unaligned_store_u16(address, (uint16_t) _mm_extract_epi16(v, 0));
}
#endif  // GCC pre-11, Clang pre-8, Android NDK Clang pre-8.0.7, Apple Clang pre-11, and ICC pre-16
#endif  // SSE2

#ifdef __AVX__
#include <immintrin.h>

// GCC pre-10
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 10)
static XNN_INTRINSIC __m256 _mm256_zextps128_ps256(__m128 v) {
  return _mm256_insertf128_ps(_mm256_setzero_ps(), v, 0);
}
#endif  // GCC pre-10

#endif  // __AVX__

#ifdef __AVX512F__
#include <immintrin.h>

// GCC pre-7, Clang pre-8, Android NDK Clang pre-8.0.7, Apple Clang pre-11, ICC pre-18, and MSVC pre-2019
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 7)) || \
    (defined(__clang__) && !defined(__apple_build_version__) && (__clang_major__ < 8)) || \
    (defined(__clang__) && defined(__ANDROID__) && (__clang_major__ == 8) && (__clang_minor__ == 0) && (__clang_patchlevel__ < 7)) || \
    (defined(__clang__) && defined(__apple_build_version__) && (__apple_build_version__ < 11000000)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1800)) || \
    (defined(_MSC_VER) && !defined(__clang__) && !defined(__GNUC__) && (_MSC_VER <= 1916))

static XNN_INTRINSIC
__mmask16 _cvtu32_mask16(unsigned int mask) {
  return (__mmask16) mask;
}

static XNN_INTRINSIC
__mmask64 _cvtu64_mask64(unsigned long long mask) {
  return (__mmask64) mask;
}

static XNN_INTRINSIC
__mmask64 _kshiftli_mask64(__mmask64 a, unsigned int count) {
  return (__mmask64) ((unsigned long long) a << count);
}

static XNN_INTRINSIC
__mmask64 _kshiftri_mask64(__mmask64 a, unsigned int count) {
  return (__mmask64) ((unsigned long long) a >> count);
}

#endif  // GCC pre-7, Clang pre-8, Android NDK Clang pre-8.0.7, Apple Clang pre-11, and ICC pre-18

// GCC pre-7, Clang pre-4, and ICC pre-18
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 7)) || \
    (defined(__clang__) && (__clang_major__ < 4)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1800))

static XNN_INTRINSIC
float _mm512_reduce_add_ps(__m512 v) {
#if __AVX512DQ__
  const __m256 sum2 = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
#else  // __AVX512DQ__
  const __m256 sum2 = _mm256_add_ps(_mm512_castps512_ps256(v), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1)));
#endif  // __AVX512DQ__
  const __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum2), _mm256_extractf128_ps(sum2, 1));
  const __m128 sum8 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  const __m128 sum16 = _mm_add_ss(sum8, _mm_movehdup_ps(sum8));
  return _mm_cvtss_f32(sum16);
}

static XNN_INTRINSIC
float _mm512_reduce_max_ps(__m512 v) {
#if __AVX512DQ__
  const __m256 sum2 = _mm256_max_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
#else  // __AVX512DQ__
  const __m256 sum2 = _mm256_max_ps(_mm512_castps512_ps256(v), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1)));
#endif  // __AVX512DQ__
  const __m128 sum4 = _mm_max_ps(_mm256_castps256_ps128(sum2), _mm256_extractf128_ps(sum2, 1));
  const __m128 sum8 = _mm_max_ps(sum4, _mm_movehl_ps(sum4, sum4));
  const __m128 sum16 = _mm_max_ss(sum8, _mm_movehdup_ps(sum8));
  return _mm_cvtss_f32(sum16);
}

#endif  // GCC pre-7, Clang pre-4, and ICC pre-18

// GCC pre-9
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 9)
static XNN_INTRINSIC
__m512i _mm512_set_epi8(
  char e63, char e62, char e61, char e60,
  char e59, char e58, char e57, char e56,
  char e55, char e54, char e53, char e52,
  char e51, char e50, char e49, char e48,
  char e47, char e46, char e45, char e44,
  char e43, char e42, char e41, char e40,
  char e39, char e38, char e37, char e36,
  char e35, char e34, char e33, char e32,
  char e31, char e30, char e29, char e28,
  char e27, char e26, char e25, char e24,
  char e23, char e22, char e21, char e20,
  char e19, char e18, char e17, char e16,
  char e15, char e14, char e13, char e12,
  char e11, char e10, char e09, char e08,
  char e07, char e06, char e05, char e04,
  char e03, char e02, char e01, char e00)
{
  return (__m512i) (__v64qi) {
    e00, e01, e02, e03, e04, e05, e06, e07,
    e08, e09, e10, e11, e12, e13, e14, e15,
    e16, e17, e18, e19, e20, e21, e22, e23,
    e24, e25, e26, e27, e28, e29, e30, e31,
    e32, e33, e34, e35, e36, e37, e38, e39,
    e40, e41, e42, e43, e44, e45, e46, e47,
    e48, e49, e50, e51, e52, e53, e54, e55,
    e56, e57, e58, e59, e60, e61, e62, e63
  };
}
#endif  // GCC pre-9

// GCC pre-10
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 10)
static XNN_INTRINSIC __m512 _mm512_zextps128_ps512(__m128 v) {
  return _mm512_insertf32x4(_mm512_setzero_ps(), v, 0);
}

static XNN_INTRINSIC
__m512i _mm512_loadu_epi16 (void const* mem_addr) {
  return _mm512_set1_epi16((int) unaligned_load_s16(mem_addr));
}

static XNN_INTRINSIC
__m512i _mm512_loadu_epi32 (void const* mem_addr) {
  return _mm512_set1_epi32((int) unaligned_load_s32(mem_addr));
}

static XNN_INTRINSIC
void _mm512_storeu_epi16 (void* mem_addr, __m512i a) {
  _mm512_storeu_si512(mem_addr, a);
}

static XNN_INTRINSIC
void _mm512_storeu_epi32 (void* mem_addr, __m512i a) {
  _mm512_storeu_si512(mem_addr, a);
}
#endif  // GCC pre-10

#ifdef __AVX512BW__
// VNNI replacement that uses vpmaddubsw.
// u4 is uint4 in lower 4 bits.
// subtracting zero_point (8) converts 4 bit value to sign extended 8 bit value.
static XNN_INTRINSIC
__m512i _mm512_dpbusd_epi32_madd(__m512i i32, const __m512i u8, const __m512i u4) {
  const __m512i vzero_point = _mm512_set1_epi8(8);
  const __m512i vone = _mm512_set1_epi16(1);
  const __m512i i4 = _mm512_sub_epi8(u4, vzero_point);
  const __m512i i12 = _mm512_maddubs_epi16(u8, i4);  // u8 * i4 = i12
  const __m512i v = _mm512_madd_epi16(i12, vone);  // convert 16 bits to 32 bits
  return _mm512_add_epi32(i32, v);
}
#endif  // __AVX512BW__
#endif  // __AVX512F__

#ifdef __AVX2__

// AVXVNNI replacement that uses vpmaddubsw.
// u4 is uint4 in lower 4 bits.
static XNN_INTRINSIC
__m256i _mm256_dpbusd_epi32_madd(__m256i i32, const __m256i u8, const __m256i u4) {
  const __m256i vzero_point = _mm256_set1_epi8(8);
  const __m256i vone = _mm256_set1_epi16(1);
  const __m256i i4 = _mm256_sub_epi8(u4, vzero_point);  // convert uint4 to int4
  const __m256i i12 = _mm256_maddubs_epi16(u8, i4);  // u8 * i4 = i12
  const __m256i v = _mm256_madd_epi16(i12, vone);  // convert 16 bits to 32 bits
  return _mm256_add_epi32(i32, v);
}

#endif  // __AVX2__

#if defined(__SSSE3__) || defined(_M_X64) || (defined _M_IX86_FP && _M_IX86_FP >= 2)

#include <tmmintrin.h>

// SSE VNNI replacement that uses vpmaddubsw.
// u4 is uint4 in lower 4 bits.
static XNN_INTRINSIC
__m128i _mm_dpbusd_epi32_madd(__m128i i32, const __m128i u8, const __m128i u4) {
  const __m128i vzero_point = _mm_set1_epi8(8);
  const __m128i vone = _mm_set1_epi16(1);
  const __m128i i4 = _mm_sub_epi8(u4, vzero_point);  // convert uint4 to int4
  const __m128i i12 = _mm_maddubs_epi16(u8, i4);  // u8 * i4 = i12
  const __m128i v = _mm_madd_epi16(i12, vone);  // convert 16 bits to 32 bits
  return _mm_add_epi32(i32, v);
}

#endif  // defined(__SSSE3__) || defined _M_X64 || (defined _M_IX86_FP && _M_IX86_FP >= 2)

#if XNN_ARCH_ARM

// AArch32 GCC 10+ implements arm_acle.h header, but lacks __ror intrinsic
#if defined(__GNUC__) && !defined(__clang__)
static XNN_INTRINSIC uint32_t __ror(uint32_t x, uint32_t y) {
   return (x >> y) | (x << (32 - y));
}
#endif  // AArch32 GCC

#endif  // ARM

#if XNN_ARCH_ARM && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>

// AArch32 GCC targeting ARMv8 NEON, see
// - https://gcc.gnu.org/bugzilla/show_bug.cgi?id=71233
// - https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95399
#if defined(__GNUC__) && !defined(__clang__) && (__ARM_ARCH >= 8)
static XNN_INTRINSIC
int32x4_t vcvtnq_s32_f32(float32x4_t v) {
  return vcvtq_s32_f32(vrndnq_f32(v));
}
#endif  // AArch32 GCC targeting ARMv8 NEON

#endif  // ARM NEON

// AArch32 Clang targeting ARMv8.2-A with FP16 arithmetics
#if XNN_ARCH_ARM && (defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && defined(__clang__))
#include <arm_fp16.h>

static XNN_INTRINSIC
float16_t vaddh_f16(float16_t a, float16_t b) {
  return a + b;
}

static XNN_INTRINSIC
float16_t vdivh_f16(float16_t a, float16_t b) {
  return a / b;
}

static XNN_INTRINSIC
float16_t vmaxnmh_f16(float16_t a, float16_t b) {
  return XNN_UNPREDICTABLE(b < a) ? a : b;
}

static XNN_INTRINSIC
float16_t vminnmh_f16(float16_t a, float16_t b) {
  return XNN_UNPREDICTABLE(b < a) ? b : a;
}

static XNN_INTRINSIC
float16_t vmulh_f16(float16_t a, float16_t b) {
  return a * b;
}

static XNN_INTRINSIC
float16_t vsubh_f16(float16_t a, float16_t b) {
  return a - b;
}

static XNN_INTRINSIC
float16_t vsqrth_f16(float16_t v) {
  return __builtin_sqrtf(v);
}
#endif  // AArch32 Clang targeting ARMv8.2-A with FP16 arithmetics

// AArch32 targeting ARMv8.2-A with NEON+FP16 arithmetics
#if XNN_ARCH_ARM && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>

#if !defined(__GNUC__) || defined(__clang_major__) & (__clang_major__ < 10)
  // GNU-style assembly is not supported, or "x" constraint is not supported (Clang pre-10)
  #define vmlaq_lane_f16(acc, x, y, n) \
    vaddq_f16((acc), vmulq_lane_f16((x), (y), (n)))
  #define vmla_lane_f16(acc, x, y, n) \
    vadd_f16((acc), vmul_lane_f16((x), (y), (n)))
#else
  #define vmlaq_lane_f16(acc, x, y, n)       \
    ({                                       \
      float16x8_t result = acc;              \
      __asm__ ("vmla.f16 %q0, %q1, %P2[%c3]" \
          : "+w" (result)                    \
          : "w" (x), "x" (y), "i" (n));      \
      result;                                \
    })
  #define vmla_lane_f16(acc, x, y, n)        \
    ({                                       \
      float16x4_t result = acc;              \
      __asm__ ("vmla.f16 %P0, %P1, %P2[%c3]" \
          : "+w" (result)                    \
          : "w" (x), "x" (y), "i" (n));      \
      result;                                \
    })
#endif
#endif  // AArch32 targeting ARMv8.2-A with NEON+FP16 arithmetics

#if XNN_ARCH_ARM64
#include <arm_neon.h>

// AArch64 GCC pre-8, 8.1-8.4, 9.1-9.3
#if defined(__GNUC__) && !defined(__clang__) && \
  (__GNUC__ < 8 || __GNUC__ == 8 && __GNUC_MINOR__ < 5 || __GNUC__ == 9 && __GNUC_MINOR__ < 4)
static XNN_INTRINSIC
uint8x16x4_t vld1q_u8_x4(const uint8_t* address) {
  uint8x16x4_t result;
  result.val[0] = vld1q_u8(address);
  result.val[1] = vld1q_u8(address + 16);
  result.val[2] = vld1q_u8(address + 32);
  result.val[3] = vld1q_u8(address + 48);
  return result;
}
#endif  // AArch64 GCC pre-8, 8.1-8.4, 9.1-9.3

#endif  // ARM64 NEON

// Hexagon
#if XNN_ARCH_HEXAGON
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

// Conditional Store:
// - addr: destination
// - n: number of elements * sizeof(datatype) where n <= 128
// - vin: input
static XNN_INTRINSIC
void Q6_V_vstu_variable(void *addr, uint32_t n, HVX_Vector vin)
{
    // Rotate as needed.
    vin = Q6_V_vlalign_VVR(vin, vin, (size_t) addr);

    uint32_t left_off = (size_t) addr & 127;
    uint32_t right_off = left_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) addr);
    HVX_VectorPred qr = Q6_Q_vsetq2_R(right_off);

    if (right_off > 128)
    {
        Q6_vmem_QRIV(qr, (HVX_Vector*) addr + 1, vin);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(vin, vin);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector*) addr, vin);
}

static XNN_INTRINSIC
void vstu_variable_scalar(char *bytes, size_t num_bytes, HVX_Vector vin) {
  char temp[128]  __attribute__((aligned(128)));
  *((HVX_Vector *)temp) = vin;
  for (size_t idx = 0; idx < num_bytes; idx++){
     *bytes = temp[idx];
     bytes++;
  }
}

// 32x16 Integer Multiplication:
// - multiplier: 32-bit integer
// - vin: 16-bit signed integer
// - Return 'HVX_VectorPair' keeping the same order as vin
//   but with elements widened to 32-bit.
static XNN_INTRINSIC
HVX_VectorPair Q6_Vw_vmpyi_VwVh(HVX_Vector multiplier, HVX_Vector vin)
{
    vin = Q6_Vh_vshuffe_VhVh(vin, vin);
    HVX_Vector mul_e = Q6_Vw_vmpyio_VwVh(multiplier, vin);
    HVX_Vector mul_o = Q6_Vw_vmpyio_VwVh(multiplier, vin);
    
    return Q6_W_vshuff_VVR(mul_o, mul_e, -4);
}

// 32x16 Integer Multiplication of even elements in the 'vin':
// multiplier_hi: upper part of 32-bit integer multiplier
// multiplier_lo: lower part of 32-bit integer multiplier
// vin: 16-bit signed integer
// - Return 'vout' in the HVX_Vector format,
//   containing only the multiplication results of the even elements from 'vin' and
//   widened to 32-bit.
static XNN_INTRINSIC
HVX_Vector Q6_Vw_vmpyie_VwVh(HVX_Vector multiplier_lo, HVX_Vector multiplier_hi, HVX_Vector vin)
{
    multiplier_hi = Q6_Vh_vshuffe_VhVh(multiplier_hi, multiplier_hi);
    HVX_Vector vout = Q6_Vw_vmpyieo_VhVh(vin, multiplier_hi);
    vout = Q6_Vw_vmpyieacc_VwVwVh(vout, multiplier_lo, vin);

    return vout;
}

// Horizontal vector sum by pairwise addition.
// To calculate fewer elements than the full 128 bytes in 'vin', 
// use the following code first before calling the intrinsic:
//   vin = Q6_V_vand_QV(Q6_Q_vsetq_R(batch), vin);
// where 'batch' is equal to 'elements * sizeof(float)'
static XNN_INTRINSIC
float Q6_f32_vrsum_Vsf(HVX_Vector vin){
    HVX_VectorPair vsum_pair = Q6_W_vshuff_VVR(vin, vin, 64);
    vin = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

    vsum_pair = Q6_W_vshuff_VVR(vin, vin, 32);
    vin = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

    vsum_pair = Q6_W_vshuff_VVR(vin, vin, 16);
    vin = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

    vsum_pair = Q6_W_vshuff_VVR(vin, vin, 8);
    vin = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

    vsum_pair = Q6_W_vshuff_VVR(vin, vin, 4);
    vin = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

    return *((float *) &vin);
}

// DIV implementation using Newton-Raphson reciprocal approximation
// Implementation comes from Halide project
// a/b = a * fast_inverse__vsf(b)
static XNN_INTRINSIC
HVX_Vector fast_inverse__vsf(HVX_Vector vin) {
    const uint32_t fp_exp_norm = 0x7F000000;  // IEEE sf: sign=0, exp=254, mant=0
    const uint32_t fp_exp_mask = 0xFF800000;  // mask for IEEE sf exp
    const uint32_t nr_T1       = 0x5a5a5a7f;  // Newton Raphson T1=24.0/17.0 (qf32)
    const uint32_t nr_T2       = 0x8787877d;  // Newton Raphson T2=-8.0/17.0 (qf32)
    const uint32_t qf_one      = 0x4000007F;  // 1.0 (qf32)

    HVX_Vector vfp_exp_norm = Q6_V_vsplat_R(fp_exp_norm);
    HVX_Vector vfp_exp_mask = Q6_V_vsplat_R(fp_exp_mask);

    HVX_Vector vnr_T1 = Q6_V_vsplat_R(nr_T1);
    HVX_Vector vnr_T2 = Q6_V_vsplat_R(nr_T2);

    HVX_Vector vone  = Q6_V_vsplat_R(qf_one);
    HVX_Vector vzero = Q6_V_vzero();

    // IEEE sf: sign[i] = sign(den[i]), exp[i] = exp(den[i]), mant = 0
    HVX_Vector vfp_exp = Q6_V_vand_VV(vin, vfp_exp_mask);

    // normalization factor in IEEE sf:
    //   sign[i] = sign(den[i]), exp[i] = 254 - exp(den[i]), mant = 0
    HVX_Vector vfp_norm = Q6_Vw_vsub_VwVw(vfp_exp_norm, vfp_exp);
    HVX_Vector vnorm = Q6_Vqf32_vadd_VsfVsf(vfp_norm, vzero);  // qf32

    HVX_Vector vout = Q6_Vqf32_vmpy_VsfVsf(vin, vfp_norm);  // normalize den[i]

    // initial estimate X0[i] = T1 + (T2 * den[i])
    HVX_Vector vtmp = Q6_Vqf32_vmpy_Vqf32Vqf32(vnr_T2, vout);
    HVX_Vector vX0  = Q6_Vqf32_vadd_Vqf32Vqf32(vnr_T1, vtmp);

#pragma clang loop unroll(enable)
    for (int newtRaph = 0; newtRaph < 3; newtRaph++) {
        vtmp = Q6_Vqf32_vmpy_Vqf32Vqf32(vX0,  vout);  // X0[i] * den[i]
        vtmp = Q6_Vqf32_vsub_Vqf32Vqf32(vone, vtmp);  // (1.0 - X0[i] * den[i])
        vtmp = Q6_Vqf32_vmpy_Vqf32Vqf32(vX0,  vtmp);  // X0[i] * (1.0 - X0[i] * den[i])
        vX0  = Q6_Vqf32_vadd_Vqf32Vqf32(vX0,  vtmp);  // X0[i] = X0[i] + X0[i] * (1.0 - X0[i] * den[i])
    }

    // multiply result by same normalization factor applied to denominator earlier.
    vout = Q6_Vqf32_vmpy_Vqf32Vqf32(vX0, vnorm);

    vout = Q6_Vsf_equals_Vqf32(vout);  // convert output back to IEEE sf
    return vout;
}

static XNN_INTRINSIC
HVX_Vector Q6_Vsf_vdiv_VsfVsf(HVX_Vector vin1, HVX_Vector vin2){
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vin1, fast_inverse__vsf(vin2)));
}
#endif  // XNN_ARCH_HEXAGON
