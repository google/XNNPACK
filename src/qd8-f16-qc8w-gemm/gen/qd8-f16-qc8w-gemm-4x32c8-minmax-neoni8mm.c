// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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
  uint16_t* c0 = (uint16_t*) c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  // Loop over groups of 32 columns.
  do {
    // Initialize accumulators with bias. 32 bias values are loaded from the
    // weight matrix, at the start of the group of 32 columns.
    const int32x4_t vinput_zero_point01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumGHIJ = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xGHIJ = vmulq_lane_s32(vksumGHIJ, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xGHIJ = vmulq_lane_s32(vksumGHIJ, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumKLMN = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xKLMN = vmulq_lane_s32(vksumKLMN, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xKLMN = vmulq_lane_s32(vksumKLMN, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumOPQR = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xOPQR = vmulq_lane_s32(vksumOPQR, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xOPQR = vmulq_lane_s32(vksumOPQR, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumSTUV = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xSTUV = vmulq_lane_s32(vksumSTUV, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xSTUV = vmulq_lane_s32(vksumSTUV, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vinput_zero_point23 = vld1q_s32(&quantization_params[2].zero_point);
    const int32x4_t vksumzp2x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xGHIJ = vmulq_lane_s32(vksumGHIJ, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xGHIJ = vmulq_lane_s32(vksumGHIJ, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xKLMN = vmulq_lane_s32(vksumKLMN, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xKLMN = vmulq_lane_s32(vksumKLMN, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xOPQR = vmulq_lane_s32(vksumOPQR, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xOPQR = vmulq_lane_s32(vksumOPQR, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xSTUV = vmulq_lane_s32(vksumSTUV, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xSTUV = vmulq_lane_s32(vksumSTUV, vget_high_s32(vinput_zero_point23), 0);

    #if XNN_ARCH_ARM64
      int32x4_t vacc01x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp1x0123)));
      int32x4_t vacc01x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp1x0123)));
      int32x4_t vacc01x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x4567), vreinterpretq_u64_s32(vksumzp1x4567)));
      int32x4_t vacc01x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x4567), vreinterpretq_u64_s32(vksumzp1x4567)));
      int32x4_t vacc01x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x89AB), vreinterpretq_u64_s32(vksumzp1x89AB)));
      int32x4_t vacc01xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x89AB), vreinterpretq_u64_s32(vksumzp1x89AB)));
      int32x4_t vacc01xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xCDEF), vreinterpretq_u64_s32(vksumzp1xCDEF)));
      int32x4_t vacc01xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xCDEF), vreinterpretq_u64_s32(vksumzp1xCDEF)));
      int32x4_t vacc01xGH = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xGHIJ), vreinterpretq_u64_s32(vksumzp1xGHIJ)));
      int32x4_t vacc01xIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xGHIJ), vreinterpretq_u64_s32(vksumzp1xGHIJ)));
      int32x4_t vacc01xKL = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xKLMN), vreinterpretq_u64_s32(vksumzp1xKLMN)));
      int32x4_t vacc01xMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xKLMN), vreinterpretq_u64_s32(vksumzp1xKLMN)));
      int32x4_t vacc01xOP = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xOPQR), vreinterpretq_u64_s32(vksumzp1xOPQR)));
      int32x4_t vacc01xQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xOPQR), vreinterpretq_u64_s32(vksumzp1xOPQR)));
      int32x4_t vacc01xST = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xSTUV), vreinterpretq_u64_s32(vksumzp1xSTUV)));
      int32x4_t vacc01xUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xSTUV), vreinterpretq_u64_s32(vksumzp1xSTUV)));
      int32x4_t vacc23x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x0123), vreinterpretq_u64_s32(vksumzp3x0123)));
      int32x4_t vacc23x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x0123), vreinterpretq_u64_s32(vksumzp3x0123)));
      int32x4_t vacc23x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x4567), vreinterpretq_u64_s32(vksumzp3x4567)));
      int32x4_t vacc23x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x4567), vreinterpretq_u64_s32(vksumzp3x4567)));
      int32x4_t vacc23x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x89AB), vreinterpretq_u64_s32(vksumzp3x89AB)));
      int32x4_t vacc23xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x89AB), vreinterpretq_u64_s32(vksumzp3x89AB)));
      int32x4_t vacc23xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xCDEF), vreinterpretq_u64_s32(vksumzp3xCDEF)));
      int32x4_t vacc23xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xCDEF), vreinterpretq_u64_s32(vksumzp3xCDEF)));
      int32x4_t vacc23xGH = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xGHIJ), vreinterpretq_u64_s32(vksumzp3xGHIJ)));
      int32x4_t vacc23xIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xGHIJ), vreinterpretq_u64_s32(vksumzp3xGHIJ)));
      int32x4_t vacc23xKL = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xKLMN), vreinterpretq_u64_s32(vksumzp3xKLMN)));
      int32x4_t vacc23xMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xKLMN), vreinterpretq_u64_s32(vksumzp3xKLMN)));
      int32x4_t vacc23xOP = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xOPQR), vreinterpretq_u64_s32(vksumzp3xOPQR)));
      int32x4_t vacc23xQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xOPQR), vreinterpretq_u64_s32(vksumzp3xOPQR)));
      int32x4_t vacc23xST = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xSTUV), vreinterpretq_u64_s32(vksumzp3xSTUV)));
      int32x4_t vacc23xUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xSTUV), vreinterpretq_u64_s32(vksumzp3xSTUV)));
    #else
      int32x4_t vacc01x01 = vcombine_s32(vget_low_s32(vksumzp0x0123), vget_low_s32(vksumzp1x0123));
      int32x4_t vacc01x23 = vcombine_s32(vget_high_s32(vksumzp0x0123), vget_high_s32(vksumzp1x0123));
      int32x4_t vacc01x45 = vcombine_s32(vget_low_s32(vksumzp0x4567), vget_low_s32(vksumzp1x4567));
      int32x4_t vacc01x67 = vcombine_s32(vget_high_s32(vksumzp0x4567), vget_high_s32(vksumzp1x4567));
      int32x4_t vacc01x89 = vcombine_s32(vget_low_s32(vksumzp0x89AB), vget_low_s32(vksumzp1x89AB));
      int32x4_t vacc01xAB = vcombine_s32(vget_high_s32(vksumzp0x89AB), vget_high_s32(vksumzp1x89AB));
      int32x4_t vacc01xCD = vcombine_s32(vget_low_s32(vksumzp0xCDEF), vget_low_s32(vksumzp1xCDEF));
      int32x4_t vacc01xEF = vcombine_s32(vget_high_s32(vksumzp0xCDEF), vget_high_s32(vksumzp1xCDEF));
      int32x4_t vacc01xGH = vcombine_s32(vget_low_s32(vksumzp0xGHIJ), vget_low_s32(vksumzp1xGHIJ));
      int32x4_t vacc01xIJ = vcombine_s32(vget_high_s32(vksumzp0xGHIJ), vget_high_s32(vksumzp1xGHIJ));
      int32x4_t vacc01xKL = vcombine_s32(vget_low_s32(vksumzp0xKLMN), vget_low_s32(vksumzp1xKLMN));
      int32x4_t vacc01xMN = vcombine_s32(vget_high_s32(vksumzp0xKLMN), vget_high_s32(vksumzp1xKLMN));
      int32x4_t vacc01xOP = vcombine_s32(vget_low_s32(vksumzp0xOPQR), vget_low_s32(vksumzp1xOPQR));
      int32x4_t vacc01xQR = vcombine_s32(vget_high_s32(vksumzp0xOPQR), vget_high_s32(vksumzp1xOPQR));
      int32x4_t vacc01xST = vcombine_s32(vget_low_s32(vksumzp0xSTUV), vget_low_s32(vksumzp1xSTUV));
      int32x4_t vacc01xUV = vcombine_s32(vget_high_s32(vksumzp0xSTUV), vget_high_s32(vksumzp1xSTUV));
      int32x4_t vacc23x01 = vcombine_s32(vget_low_s32(vksumzp2x0123), vget_low_s32(vksumzp3x0123));
      int32x4_t vacc23x23 = vcombine_s32(vget_high_s32(vksumzp2x0123), vget_high_s32(vksumzp3x0123));
      int32x4_t vacc23x45 = vcombine_s32(vget_low_s32(vksumzp2x4567), vget_low_s32(vksumzp3x4567));
      int32x4_t vacc23x67 = vcombine_s32(vget_high_s32(vksumzp2x4567), vget_high_s32(vksumzp3x4567));
      int32x4_t vacc23x89 = vcombine_s32(vget_low_s32(vksumzp2x89AB), vget_low_s32(vksumzp3x89AB));
      int32x4_t vacc23xAB = vcombine_s32(vget_high_s32(vksumzp2x89AB), vget_high_s32(vksumzp3x89AB));
      int32x4_t vacc23xCD = vcombine_s32(vget_low_s32(vksumzp2xCDEF), vget_low_s32(vksumzp3xCDEF));
      int32x4_t vacc23xEF = vcombine_s32(vget_high_s32(vksumzp2xCDEF), vget_high_s32(vksumzp3xCDEF));
      int32x4_t vacc23xGH = vcombine_s32(vget_low_s32(vksumzp2xGHIJ), vget_low_s32(vksumzp3xGHIJ));
      int32x4_t vacc23xIJ = vcombine_s32(vget_high_s32(vksumzp2xGHIJ), vget_high_s32(vksumzp3xGHIJ));
      int32x4_t vacc23xKL = vcombine_s32(vget_low_s32(vksumzp2xKLMN), vget_low_s32(vksumzp3xKLMN));
      int32x4_t vacc23xMN = vcombine_s32(vget_high_s32(vksumzp2xKLMN), vget_high_s32(vksumzp3xKLMN));
      int32x4_t vacc23xOP = vcombine_s32(vget_low_s32(vksumzp2xOPQR), vget_low_s32(vksumzp3xOPQR));
      int32x4_t vacc23xQR = vcombine_s32(vget_high_s32(vksumzp2xOPQR), vget_high_s32(vksumzp3xOPQR));
      int32x4_t vacc23xST = vcombine_s32(vget_low_s32(vksumzp2xSTUV), vget_low_s32(vksumzp3xSTUV));
      int32x4_t vacc23xUV = vcombine_s32(vget_high_s32(vksumzp2xSTUV), vget_high_s32(vksumzp3xSTUV));
    #endif

    // Inner accumulation loop along the 32 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.

    uint64x2x2_t va01x0123456789ABCDEF;
    va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
    va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
    uint64x2x2_t va23x0123456789ABCDEF;
    va23x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
    va23x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

    while (k >= 16 * sizeof(int8_t)) {
      // Load a 4x16 block of activations.
      #if XNN_ARCH_ARM64
        va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
        va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a2, va23x0123456789ABCDEF, 0); a2 += 16;
        va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a1, va01x0123456789ABCDEF, 1); a1 += 16;
        va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a3, va23x0123456789ABCDEF, 1); a3 += 16;
      #else
        va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[0], 0); a0 += 8;
        va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a0, va01x0123456789ABCDEF.val[1], 0); a0 += 8;
        va23x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a2, va23x0123456789ABCDEF.val[0], 0); a2 += 8;
        va23x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a2, va23x0123456789ABCDEF.val[1], 0); a2 += 8;
        va01x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a1, va01x0123456789ABCDEF.val[0], 1); a1 += 8;
        va01x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a1, va01x0123456789ABCDEF.val[1], 1); a1 += 8;
        va23x0123456789ABCDEF.val[0] = vld1q_lane_u64((const void*) a3, va23x0123456789ABCDEF.val[0], 1); a3 += 8;
        va23x0123456789ABCDEF.val[1] = vld1q_lane_u64((const void*) a3, va23x0123456789ABCDEF.val[1], 1); a3 += 8;
      #endif

      const int8x16_t va01x01234567 = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]);
      const int8x16_t va01x89ABCDEF = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]);
      const int8x16_t va23x01234567 = vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]);
      const int8x16_t va23x89ABCDEF = vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]);

      // Load a 16x32 block of weights.
      const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbGHx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbIJx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbKLx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbMNx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbOPx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbQRx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbSTx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbUVx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbGHx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbIJx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbKLx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbMNx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbOPx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbQRx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbSTx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbUVx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x8 * 8x32 --> 4x32.
      vacc01x01 = vmmlaq_s32(vacc01x01, va01x01234567, vb01x01234567);
      vacc01x23 = vmmlaq_s32(vacc01x23, va01x01234567, vb23x01234567);
      vacc01x45 = vmmlaq_s32(vacc01x45, va01x01234567, vb45x01234567);
      vacc01x67 = vmmlaq_s32(vacc01x67, va01x01234567, vb67x01234567);
      vacc01x89 = vmmlaq_s32(vacc01x89, va01x01234567, vb89x01234567);
      vacc01xAB = vmmlaq_s32(vacc01xAB, va01x01234567, vbABx01234567);
      vacc01xCD = vmmlaq_s32(vacc01xCD, va01x01234567, vbCDx01234567);
      vacc01xEF = vmmlaq_s32(vacc01xEF, va01x01234567, vbEFx01234567);
      vacc01xGH = vmmlaq_s32(vacc01xGH, va01x01234567, vbGHx01234567);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, va01x01234567, vbIJx01234567);
      vacc01xKL = vmmlaq_s32(vacc01xKL, va01x01234567, vbKLx01234567);
      vacc01xMN = vmmlaq_s32(vacc01xMN, va01x01234567, vbMNx01234567);
      vacc01xOP = vmmlaq_s32(vacc01xOP, va01x01234567, vbOPx01234567);
      vacc01xQR = vmmlaq_s32(vacc01xQR, va01x01234567, vbQRx01234567);
      vacc01xST = vmmlaq_s32(vacc01xST, va01x01234567, vbSTx01234567);
      vacc01xUV = vmmlaq_s32(vacc01xUV, va01x01234567, vbUVx01234567);
      vacc23x01 = vmmlaq_s32(vacc23x01, va23x01234567, vb01x01234567);
      vacc23x23 = vmmlaq_s32(vacc23x23, va23x01234567, vb23x01234567);
      vacc23x45 = vmmlaq_s32(vacc23x45, va23x01234567, vb45x01234567);
      vacc23x67 = vmmlaq_s32(vacc23x67, va23x01234567, vb67x01234567);
      vacc23x89 = vmmlaq_s32(vacc23x89, va23x01234567, vb89x01234567);
      vacc23xAB = vmmlaq_s32(vacc23xAB, va23x01234567, vbABx01234567);
      vacc23xCD = vmmlaq_s32(vacc23xCD, va23x01234567, vbCDx01234567);
      vacc23xEF = vmmlaq_s32(vacc23xEF, va23x01234567, vbEFx01234567);
      vacc23xGH = vmmlaq_s32(vacc23xGH, va23x01234567, vbGHx01234567);
      vacc23xIJ = vmmlaq_s32(vacc23xIJ, va23x01234567, vbIJx01234567);
      vacc23xKL = vmmlaq_s32(vacc23xKL, va23x01234567, vbKLx01234567);
      vacc23xMN = vmmlaq_s32(vacc23xMN, va23x01234567, vbMNx01234567);
      vacc23xOP = vmmlaq_s32(vacc23xOP, va23x01234567, vbOPx01234567);
      vacc23xQR = vmmlaq_s32(vacc23xQR, va23x01234567, vbQRx01234567);
      vacc23xST = vmmlaq_s32(vacc23xST, va23x01234567, vbSTx01234567);
      vacc23xUV = vmmlaq_s32(vacc23xUV, va23x01234567, vbUVx01234567);
      vacc01x01 = vmmlaq_s32(vacc01x01, va01x89ABCDEF, vb01x89ABCDEF);
      vacc01x23 = vmmlaq_s32(vacc01x23, va01x89ABCDEF, vb23x89ABCDEF);
      vacc01x45 = vmmlaq_s32(vacc01x45, va01x89ABCDEF, vb45x89ABCDEF);
      vacc01x67 = vmmlaq_s32(vacc01x67, va01x89ABCDEF, vb67x89ABCDEF);
      vacc01x89 = vmmlaq_s32(vacc01x89, va01x89ABCDEF, vb89x89ABCDEF);
      vacc01xAB = vmmlaq_s32(vacc01xAB, va01x89ABCDEF, vbABx89ABCDEF);
      vacc01xCD = vmmlaq_s32(vacc01xCD, va01x89ABCDEF, vbCDx89ABCDEF);
      vacc01xEF = vmmlaq_s32(vacc01xEF, va01x89ABCDEF, vbEFx89ABCDEF);
      vacc01xGH = vmmlaq_s32(vacc01xGH, va01x89ABCDEF, vbGHx89ABCDEF);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, va01x89ABCDEF, vbIJx89ABCDEF);
      vacc01xKL = vmmlaq_s32(vacc01xKL, va01x89ABCDEF, vbKLx89ABCDEF);
      vacc01xMN = vmmlaq_s32(vacc01xMN, va01x89ABCDEF, vbMNx89ABCDEF);
      vacc01xOP = vmmlaq_s32(vacc01xOP, va01x89ABCDEF, vbOPx89ABCDEF);
      vacc01xQR = vmmlaq_s32(vacc01xQR, va01x89ABCDEF, vbQRx89ABCDEF);
      vacc01xST = vmmlaq_s32(vacc01xST, va01x89ABCDEF, vbSTx89ABCDEF);
      vacc01xUV = vmmlaq_s32(vacc01xUV, va01x89ABCDEF, vbUVx89ABCDEF);
      vacc23x01 = vmmlaq_s32(vacc23x01, va23x89ABCDEF, vb01x89ABCDEF);
      vacc23x23 = vmmlaq_s32(vacc23x23, va23x89ABCDEF, vb23x89ABCDEF);
      vacc23x45 = vmmlaq_s32(vacc23x45, va23x89ABCDEF, vb45x89ABCDEF);
      vacc23x67 = vmmlaq_s32(vacc23x67, va23x89ABCDEF, vb67x89ABCDEF);
      vacc23x89 = vmmlaq_s32(vacc23x89, va23x89ABCDEF, vb89x89ABCDEF);
      vacc23xAB = vmmlaq_s32(vacc23xAB, va23x89ABCDEF, vbABx89ABCDEF);
      vacc23xCD = vmmlaq_s32(vacc23xCD, va23x89ABCDEF, vbCDx89ABCDEF);
      vacc23xEF = vmmlaq_s32(vacc23xEF, va23x89ABCDEF, vbEFx89ABCDEF);
      vacc23xGH = vmmlaq_s32(vacc23xGH, va23x89ABCDEF, vbGHx89ABCDEF);
      vacc23xIJ = vmmlaq_s32(vacc23xIJ, va23x89ABCDEF, vbIJx89ABCDEF);
      vacc23xKL = vmmlaq_s32(vacc23xKL, va23x89ABCDEF, vbKLx89ABCDEF);
      vacc23xMN = vmmlaq_s32(vacc23xMN, va23x89ABCDEF, vbMNx89ABCDEF);
      vacc23xOP = vmmlaq_s32(vacc23xOP, va23x89ABCDEF, vbOPx89ABCDEF);
      vacc23xQR = vmmlaq_s32(vacc23xQR, va23x89ABCDEF, vbQRx89ABCDEF);
      vacc23xST = vmmlaq_s32(vacc23xST, va23x89ABCDEF, vbSTx89ABCDEF);
      vacc23xUV = vmmlaq_s32(vacc23xUV, va23x89ABCDEF, vbUVx89ABCDEF);

      k -= 16 * sizeof(int8_t);
    }
    // Handle up to 8 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 4x8 block of activations.
      uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;
      uint64x2_t va23x01234567 = vld1q_dup_u64((const void*) a2); a2 += 8;
      va01x01234567 = vld1q_lane_u64((const void*) a1, va01x01234567, 1); a1 += 8;
      va23x01234567 = vld1q_lane_u64((const void*) a3, va23x01234567, 1); a3 += 8;

      // Load a 16x32 block of weights.
      const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbGHx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbIJx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbKLx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbMNx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbOPx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbQRx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbSTx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbUVx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 4x4 * 4x32 --> 4x32.
      vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x01234567), vb01x01234567);
      vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x01234567), vb23x01234567);
      vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x01234567), vb45x01234567);
      vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x01234567), vb67x01234567);
      vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x01234567), vb89x01234567);
      vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x01234567), vbABx01234567);
      vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x01234567), vbCDx01234567);
      vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x01234567), vbEFx01234567);
      vacc01xGH = vmmlaq_s32(vacc01xGH, vreinterpretq_s8_u64(va01x01234567), vbGHx01234567);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, vreinterpretq_s8_u64(va01x01234567), vbIJx01234567);
      vacc01xKL = vmmlaq_s32(vacc01xKL, vreinterpretq_s8_u64(va01x01234567), vbKLx01234567);
      vacc01xMN = vmmlaq_s32(vacc01xMN, vreinterpretq_s8_u64(va01x01234567), vbMNx01234567);
      vacc01xOP = vmmlaq_s32(vacc01xOP, vreinterpretq_s8_u64(va01x01234567), vbOPx01234567);
      vacc01xQR = vmmlaq_s32(vacc01xQR, vreinterpretq_s8_u64(va01x01234567), vbQRx01234567);
      vacc01xST = vmmlaq_s32(vacc01xST, vreinterpretq_s8_u64(va01x01234567), vbSTx01234567);
      vacc01xUV = vmmlaq_s32(vacc01xUV, vreinterpretq_s8_u64(va01x01234567), vbUVx01234567);
      vacc23x01 = vmmlaq_s32(vacc23x01, vreinterpretq_s8_u64(va23x01234567), vb01x01234567);
      vacc23x23 = vmmlaq_s32(vacc23x23, vreinterpretq_s8_u64(va23x01234567), vb23x01234567);
      vacc23x45 = vmmlaq_s32(vacc23x45, vreinterpretq_s8_u64(va23x01234567), vb45x01234567);
      vacc23x67 = vmmlaq_s32(vacc23x67, vreinterpretq_s8_u64(va23x01234567), vb67x01234567);
      vacc23x89 = vmmlaq_s32(vacc23x89, vreinterpretq_s8_u64(va23x01234567), vb89x01234567);
      vacc23xAB = vmmlaq_s32(vacc23xAB, vreinterpretq_s8_u64(va23x01234567), vbABx01234567);
      vacc23xCD = vmmlaq_s32(vacc23xCD, vreinterpretq_s8_u64(va23x01234567), vbCDx01234567);
      vacc23xEF = vmmlaq_s32(vacc23xEF, vreinterpretq_s8_u64(va23x01234567), vbEFx01234567);
      vacc23xGH = vmmlaq_s32(vacc23xGH, vreinterpretq_s8_u64(va23x01234567), vbGHx01234567);
      vacc23xIJ = vmmlaq_s32(vacc23xIJ, vreinterpretq_s8_u64(va23x01234567), vbIJx01234567);
      vacc23xKL = vmmlaq_s32(vacc23xKL, vreinterpretq_s8_u64(va23x01234567), vbKLx01234567);
      vacc23xMN = vmmlaq_s32(vacc23xMN, vreinterpretq_s8_u64(va23x01234567), vbMNx01234567);
      vacc23xOP = vmmlaq_s32(vacc23xOP, vreinterpretq_s8_u64(va23x01234567), vbOPx01234567);
      vacc23xQR = vmmlaq_s32(vacc23xQR, vreinterpretq_s8_u64(va23x01234567), vbQRx01234567);
      vacc23xST = vmmlaq_s32(vacc23xST, vreinterpretq_s8_u64(va23x01234567), vbSTx01234567);
      vacc23xUV = vmmlaq_s32(vacc23xUV, vreinterpretq_s8_u64(va23x01234567), vbUVx01234567);
    }

    #if XNN_ARCH_ARM64
      int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
      int32x4_t vacc1x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
      int32x4_t vacc0x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
      int32x4_t vacc1x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
      int32x4_t vacc0x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
      int32x4_t vacc1x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
      int32x4_t vacc0xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
      int32x4_t vacc1xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
      int32x4_t vacc0xGHIJ = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xGH), vreinterpretq_u64_s32(vacc01xIJ)));
      int32x4_t vacc1xGHIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xGH), vreinterpretq_u64_s32(vacc01xIJ)));
      int32x4_t vacc0xKLMN = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xKL), vreinterpretq_u64_s32(vacc01xMN)));
      int32x4_t vacc1xKLMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xKL), vreinterpretq_u64_s32(vacc01xMN)));
      int32x4_t vacc0xOPQR = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xOP), vreinterpretq_u64_s32(vacc01xQR)));
      int32x4_t vacc1xOPQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xOP), vreinterpretq_u64_s32(vacc01xQR)));
      int32x4_t vacc0xSTUV = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xST), vreinterpretq_u64_s32(vacc01xUV)));
      int32x4_t vacc1xSTUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xST), vreinterpretq_u64_s32(vacc01xUV)));
      int32x4_t vacc2x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x01), vreinterpretq_u64_s32(vacc23x23)));
      int32x4_t vacc3x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x01), vreinterpretq_u64_s32(vacc23x23)));
      int32x4_t vacc2x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x45), vreinterpretq_u64_s32(vacc23x67)));
      int32x4_t vacc3x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x45), vreinterpretq_u64_s32(vacc23x67)));
      int32x4_t vacc2x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x89), vreinterpretq_u64_s32(vacc23xAB)));
      int32x4_t vacc3x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x89), vreinterpretq_u64_s32(vacc23xAB)));
      int32x4_t vacc2xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xCD), vreinterpretq_u64_s32(vacc23xEF)));
      int32x4_t vacc3xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xCD), vreinterpretq_u64_s32(vacc23xEF)));
      int32x4_t vacc2xGHIJ = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xGH), vreinterpretq_u64_s32(vacc23xIJ)));
      int32x4_t vacc3xGHIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xGH), vreinterpretq_u64_s32(vacc23xIJ)));
      int32x4_t vacc2xKLMN = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xKL), vreinterpretq_u64_s32(vacc23xMN)));
      int32x4_t vacc3xKLMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xKL), vreinterpretq_u64_s32(vacc23xMN)));
      int32x4_t vacc2xOPQR = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xOP), vreinterpretq_u64_s32(vacc23xQR)));
      int32x4_t vacc3xOPQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xOP), vreinterpretq_u64_s32(vacc23xQR)));
      int32x4_t vacc2xSTUV = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xST), vreinterpretq_u64_s32(vacc23xUV)));
      int32x4_t vacc3xSTUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xST), vreinterpretq_u64_s32(vacc23xUV)));
    #else
      int32x4_t vacc0x0123 = vcombine_s32(vget_low_s32(vacc01x01), vget_low_s32(vacc01x23));
      int32x4_t vacc1x0123 = vcombine_s32(vget_high_s32(vacc01x01), vget_high_s32(vacc01x23));
      int32x4_t vacc0x4567 = vcombine_s32(vget_low_s32(vacc01x45), vget_low_s32(vacc01x67));
      int32x4_t vacc1x4567 = vcombine_s32(vget_high_s32(vacc01x45), vget_high_s32(vacc01x67));
      int32x4_t vacc0x89AB = vcombine_s32(vget_low_s32(vacc01x89), vget_low_s32(vacc01xAB));
      int32x4_t vacc1x89AB = vcombine_s32(vget_high_s32(vacc01x89), vget_high_s32(vacc01xAB));
      int32x4_t vacc0xCDEF = vcombine_s32(vget_low_s32(vacc01xCD), vget_low_s32(vacc01xEF));
      int32x4_t vacc1xCDEF = vcombine_s32(vget_high_s32(vacc01xCD), vget_high_s32(vacc01xEF));
      int32x4_t vacc0xGHIJ = vcombine_s32(vget_low_s32(vacc01xGH), vget_low_s32(vacc01xIJ));
      int32x4_t vacc1xGHIJ = vcombine_s32(vget_high_s32(vacc01xGH), vget_high_s32(vacc01xIJ));
      int32x4_t vacc0xKLMN = vcombine_s32(vget_low_s32(vacc01xKL), vget_low_s32(vacc01xMN));
      int32x4_t vacc1xKLMN = vcombine_s32(vget_high_s32(vacc01xKL), vget_high_s32(vacc01xMN));
      int32x4_t vacc0xOPQR = vcombine_s32(vget_low_s32(vacc01xOP), vget_low_s32(vacc01xQR));
      int32x4_t vacc1xOPQR = vcombine_s32(vget_high_s32(vacc01xOP), vget_high_s32(vacc01xQR));
      int32x4_t vacc0xSTUV = vcombine_s32(vget_low_s32(vacc01xST), vget_low_s32(vacc01xUV));
      int32x4_t vacc1xSTUV = vcombine_s32(vget_high_s32(vacc01xST), vget_high_s32(vacc01xUV));
      int32x4_t vacc2x0123 = vcombine_s32(vget_low_s32(vacc23x01), vget_low_s32(vacc23x23));
      int32x4_t vacc3x0123 = vcombine_s32(vget_high_s32(vacc23x01), vget_high_s32(vacc23x23));
      int32x4_t vacc2x4567 = vcombine_s32(vget_low_s32(vacc23x45), vget_low_s32(vacc23x67));
      int32x4_t vacc3x4567 = vcombine_s32(vget_high_s32(vacc23x45), vget_high_s32(vacc23x67));
      int32x4_t vacc2x89AB = vcombine_s32(vget_low_s32(vacc23x89), vget_low_s32(vacc23xAB));
      int32x4_t vacc3x89AB = vcombine_s32(vget_high_s32(vacc23x89), vget_high_s32(vacc23xAB));
      int32x4_t vacc2xCDEF = vcombine_s32(vget_low_s32(vacc23xCD), vget_low_s32(vacc23xEF));
      int32x4_t vacc3xCDEF = vcombine_s32(vget_high_s32(vacc23xCD), vget_high_s32(vacc23xEF));
      int32x4_t vacc2xGHIJ = vcombine_s32(vget_low_s32(vacc23xGH), vget_low_s32(vacc23xIJ));
      int32x4_t vacc3xGHIJ = vcombine_s32(vget_high_s32(vacc23xGH), vget_high_s32(vacc23xIJ));
      int32x4_t vacc2xKLMN = vcombine_s32(vget_low_s32(vacc23xKL), vget_low_s32(vacc23xMN));
      int32x4_t vacc3xKLMN = vcombine_s32(vget_high_s32(vacc23xKL), vget_high_s32(vacc23xMN));
      int32x4_t vacc2xOPQR = vcombine_s32(vget_low_s32(vacc23xOP), vget_low_s32(vacc23xQR));
      int32x4_t vacc3xOPQR = vcombine_s32(vget_high_s32(vacc23xOP), vget_high_s32(vacc23xQR));
      int32x4_t vacc2xSTUV = vcombine_s32(vget_low_s32(vacc23xST), vget_low_s32(vacc23xUV));
      int32x4_t vacc3xSTUV = vcombine_s32(vget_high_s32(vacc23xST), vget_high_s32(vacc23xUV));
    #endif
    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout0xGHIJ = vcvtq_f32_s32(vacc0xGHIJ);
    float32x4_t vout0xKLMN = vcvtq_f32_s32(vacc0xKLMN);
    float32x4_t vout0xOPQR = vcvtq_f32_s32(vacc0xOPQR);
    float32x4_t vout0xSTUV = vcvtq_f32_s32(vacc0xSTUV);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vout1xGHIJ = vcvtq_f32_s32(vacc1xGHIJ);
    float32x4_t vout1xKLMN = vcvtq_f32_s32(vacc1xKLMN);
    float32x4_t vout1xOPQR = vcvtq_f32_s32(vacc1xOPQR);
    float32x4_t vout1xSTUV = vcvtq_f32_s32(vacc1xSTUV);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vout2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vout2xGHIJ = vcvtq_f32_s32(vacc2xGHIJ);
    float32x4_t vout2xKLMN = vcvtq_f32_s32(vacc2xKLMN);
    float32x4_t vout2xOPQR = vcvtq_f32_s32(vacc2xOPQR);
    float32x4_t vout2xSTUV = vcvtq_f32_s32(vacc2xSTUV);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vout3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
    float32x4_t vout3xGHIJ = vcvtq_f32_s32(vacc3xGHIJ);
    float32x4_t vout3xKLMN = vcvtq_f32_s32(vacc3xKLMN);
    float32x4_t vout3xOPQR = vcvtq_f32_s32(vacc3xOPQR);
    float32x4_t vout3xSTUV = vcvtq_f32_s32(vacc3xSTUV);

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    vout0x89AB = vmulq_lane_f32(vout0x89AB, vget_low_f32(vinput_scale01), 1);
    vout1x89AB = vmulq_lane_f32(vout1x89AB, vget_high_f32(vinput_scale01), 1);
    vout0xCDEF = vmulq_lane_f32(vout0xCDEF, vget_low_f32(vinput_scale01), 1);
    vout1xCDEF = vmulq_lane_f32(vout1xCDEF, vget_high_f32(vinput_scale01), 1);
    vout0xGHIJ = vmulq_lane_f32(vout0xGHIJ, vget_low_f32(vinput_scale01), 1);
    vout1xGHIJ = vmulq_lane_f32(vout1xGHIJ, vget_high_f32(vinput_scale01), 1);
    vout0xKLMN = vmulq_lane_f32(vout0xKLMN, vget_low_f32(vinput_scale01), 1);
    vout1xKLMN = vmulq_lane_f32(vout1xKLMN, vget_high_f32(vinput_scale01), 1);
    vout0xOPQR = vmulq_lane_f32(vout0xOPQR, vget_low_f32(vinput_scale01), 1);
    vout1xOPQR = vmulq_lane_f32(vout1xOPQR, vget_high_f32(vinput_scale01), 1);
    vout0xSTUV = vmulq_lane_f32(vout0xSTUV, vget_low_f32(vinput_scale01), 1);
    vout1xSTUV = vmulq_lane_f32(vout1xSTUV, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    vout2x4567 = vmulq_lane_f32(vout2x4567, vget_low_f32(vinput_scale23), 1);
    vout3x4567 = vmulq_lane_f32(vout3x4567, vget_high_f32(vinput_scale23), 1);
    vout2x89AB = vmulq_lane_f32(vout2x89AB, vget_low_f32(vinput_scale23), 1);
    vout3x89AB = vmulq_lane_f32(vout3x89AB, vget_high_f32(vinput_scale23), 1);
    vout2xCDEF = vmulq_lane_f32(vout2xCDEF, vget_low_f32(vinput_scale23), 1);
    vout3xCDEF = vmulq_lane_f32(vout3xCDEF, vget_high_f32(vinput_scale23), 1);
    vout2xGHIJ = vmulq_lane_f32(vout2xGHIJ, vget_low_f32(vinput_scale23), 1);
    vout3xGHIJ = vmulq_lane_f32(vout3xGHIJ, vget_high_f32(vinput_scale23), 1);
    vout2xKLMN = vmulq_lane_f32(vout2xKLMN, vget_low_f32(vinput_scale23), 1);
    vout3xKLMN = vmulq_lane_f32(vout3xKLMN, vget_high_f32(vinput_scale23), 1);
    vout2xOPQR = vmulq_lane_f32(vout2xOPQR, vget_low_f32(vinput_scale23), 1);
    vout3xOPQR = vmulq_lane_f32(vout3xOPQR, vget_high_f32(vinput_scale23), 1);
    vout2xSTUV = vmulq_lane_f32(vout2xSTUV, vget_low_f32(vinput_scale23), 1);
    vout3xSTUV = vmulq_lane_f32(vout3xSTUV, vget_high_f32(vinput_scale23), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleKLMN = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleOPQR = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleSTUV = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
      const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
      const float32x4_t vbiasGHIJ = vld1q_f32(w); w = (const float*) w + 4;
      vout0xGHIJ = vfmaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
      vout1xGHIJ = vfmaq_f32(vbiasGHIJ, vout1xGHIJ, vfilter_output_scaleGHIJ);
      vout2xGHIJ = vfmaq_f32(vbiasGHIJ, vout2xGHIJ, vfilter_output_scaleGHIJ);
      vout3xGHIJ = vfmaq_f32(vbiasGHIJ, vout3xGHIJ, vfilter_output_scaleGHIJ);
      const float32x4_t vbiasKLMN = vld1q_f32(w); w = (const float*) w + 4;
      vout0xKLMN = vfmaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
      vout1xKLMN = vfmaq_f32(vbiasKLMN, vout1xKLMN, vfilter_output_scaleKLMN);
      vout2xKLMN = vfmaq_f32(vbiasKLMN, vout2xKLMN, vfilter_output_scaleKLMN);
      vout3xKLMN = vfmaq_f32(vbiasKLMN, vout3xKLMN, vfilter_output_scaleKLMN);
      const float32x4_t vbiasOPQR = vld1q_f32(w); w = (const float*) w + 4;
      vout0xOPQR = vfmaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
      vout1xOPQR = vfmaq_f32(vbiasOPQR, vout1xOPQR, vfilter_output_scaleOPQR);
      vout2xOPQR = vfmaq_f32(vbiasOPQR, vout2xOPQR, vfilter_output_scaleOPQR);
      vout3xOPQR = vfmaq_f32(vbiasOPQR, vout3xOPQR, vfilter_output_scaleOPQR);
      const float32x4_t vbiasSTUV = vld1q_f32(w); w = (const float*) w + 4;
      vout0xSTUV = vfmaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
      vout1xSTUV = vfmaq_f32(vbiasSTUV, vout1xSTUV, vfilter_output_scaleSTUV);
      vout2xSTUV = vfmaq_f32(vbiasSTUV, vout2xSTUV, vfilter_output_scaleSTUV);
      vout3xSTUV = vfmaq_f32(vbiasSTUV, vout3xSTUV, vfilter_output_scaleSTUV);
    #else
      const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
      const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
      const float32x4_t vbiasGHIJ = vld1q_f32(w); w = (const float*) w + 4;
      vout0xGHIJ = vmlaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
      vout1xGHIJ = vmlaq_f32(vbiasGHIJ, vout1xGHIJ, vfilter_output_scaleGHIJ);
      vout2xGHIJ = vmlaq_f32(vbiasGHIJ, vout2xGHIJ, vfilter_output_scaleGHIJ);
      vout3xGHIJ = vmlaq_f32(vbiasGHIJ, vout3xGHIJ, vfilter_output_scaleGHIJ);
      const float32x4_t vbiasKLMN = vld1q_f32(w); w = (const float*) w + 4;
      vout0xKLMN = vmlaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
      vout1xKLMN = vmlaq_f32(vbiasKLMN, vout1xKLMN, vfilter_output_scaleKLMN);
      vout2xKLMN = vmlaq_f32(vbiasKLMN, vout2xKLMN, vfilter_output_scaleKLMN);
      vout3xKLMN = vmlaq_f32(vbiasKLMN, vout3xKLMN, vfilter_output_scaleKLMN);
      const float32x4_t vbiasOPQR = vld1q_f32(w); w = (const float*) w + 4;
      vout0xOPQR = vmlaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
      vout1xOPQR = vmlaq_f32(vbiasOPQR, vout1xOPQR, vfilter_output_scaleOPQR);
      vout2xOPQR = vmlaq_f32(vbiasOPQR, vout2xOPQR, vfilter_output_scaleOPQR);
      vout3xOPQR = vmlaq_f32(vbiasOPQR, vout3xOPQR, vfilter_output_scaleOPQR);
      const float32x4_t vbiasSTUV = vld1q_f32(w); w = (const float*) w + 4;
      vout0xSTUV = vmlaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
      vout1xSTUV = vmlaq_f32(vbiasSTUV, vout1xSTUV, vfilter_output_scaleSTUV);
      vout2xSTUV = vmlaq_f32(vbiasSTUV, vout2xSTUV, vfilter_output_scaleSTUV);
      vout3xSTUV = vmlaq_f32(vbiasSTUV, vout3xSTUV, vfilter_output_scaleSTUV);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    float16x8_t vfp16out0xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout0xGHIJ), vcvt_f16_f32(vout0xKLMN));
    float16x8_t vfp16out0xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout0xOPQR), vcvt_f16_f32(vout0xSTUV));
    float16x8_t vfp16out1x01234567 = vcombine_f16(vcvt_f16_f32(vout1x0123), vcvt_f16_f32(vout1x4567));
    float16x8_t vfp16out1x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout1x89AB), vcvt_f16_f32(vout1xCDEF));
    float16x8_t vfp16out1xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout1xGHIJ), vcvt_f16_f32(vout1xKLMN));
    float16x8_t vfp16out1xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout1xOPQR), vcvt_f16_f32(vout1xSTUV));
    float16x8_t vfp16out2x01234567 = vcombine_f16(vcvt_f16_f32(vout2x0123), vcvt_f16_f32(vout2x4567));
    float16x8_t vfp16out2x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout2x89AB), vcvt_f16_f32(vout2xCDEF));
    float16x8_t vfp16out2xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout2xGHIJ), vcvt_f16_f32(vout2xKLMN));
    float16x8_t vfp16out2xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout2xOPQR), vcvt_f16_f32(vout2xSTUV));
    float16x8_t vfp16out3x01234567 = vcombine_f16(vcvt_f16_f32(vout3x0123), vcvt_f16_f32(vout3x4567));
    float16x8_t vfp16out3x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout3x89AB), vcvt_f16_f32(vout3xCDEF));
    float16x8_t vfp16out3xGHIJKLMN = vcombine_f16(vcvt_f16_f32(vout3xGHIJ), vcvt_f16_f32(vout3xKLMN));
    float16x8_t vfp16out3xOPQRSTUV = vcombine_f16(vcvt_f16_f32(vout3xOPQR), vcvt_f16_f32(vout3xSTUV));

    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    vfp16out0xGHIJKLMN = vmaxq_f16(vfp16out0xGHIJKLMN, voutput_min);
    vfp16out0xOPQRSTUV = vmaxq_f16(vfp16out0xOPQRSTUV, voutput_min);
    vfp16out1x01234567 = vmaxq_f16(vfp16out1x01234567, voutput_min);
    vfp16out1x89ABCDEF = vmaxq_f16(vfp16out1x89ABCDEF, voutput_min);
    vfp16out1xGHIJKLMN = vmaxq_f16(vfp16out1xGHIJKLMN, voutput_min);
    vfp16out1xOPQRSTUV = vmaxq_f16(vfp16out1xOPQRSTUV, voutput_min);
    vfp16out2x01234567 = vmaxq_f16(vfp16out2x01234567, voutput_min);
    vfp16out2x89ABCDEF = vmaxq_f16(vfp16out2x89ABCDEF, voutput_min);
    vfp16out2xGHIJKLMN = vmaxq_f16(vfp16out2xGHIJKLMN, voutput_min);
    vfp16out2xOPQRSTUV = vmaxq_f16(vfp16out2xOPQRSTUV, voutput_min);
    vfp16out3x01234567 = vmaxq_f16(vfp16out3x01234567, voutput_min);
    vfp16out3x89ABCDEF = vmaxq_f16(vfp16out3x89ABCDEF, voutput_min);
    vfp16out3xGHIJKLMN = vmaxq_f16(vfp16out3xGHIJKLMN, voutput_min);
    vfp16out3xOPQRSTUV = vmaxq_f16(vfp16out3xOPQRSTUV, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);
    vfp16out0xGHIJKLMN = vminq_f16(vfp16out0xGHIJKLMN, voutput_max);
    vfp16out0xOPQRSTUV = vminq_f16(vfp16out0xOPQRSTUV, voutput_max);
    vfp16out1x01234567 = vminq_f16(vfp16out1x01234567, voutput_max);
    vfp16out1x89ABCDEF = vminq_f16(vfp16out1x89ABCDEF, voutput_max);
    vfp16out1xGHIJKLMN = vminq_f16(vfp16out1xGHIJKLMN, voutput_max);
    vfp16out1xOPQRSTUV = vminq_f16(vfp16out1xOPQRSTUV, voutput_max);
    vfp16out2x01234567 = vminq_f16(vfp16out2x01234567, voutput_max);
    vfp16out2x89ABCDEF = vminq_f16(vfp16out2x89ABCDEF, voutput_max);
    vfp16out2xGHIJKLMN = vminq_f16(vfp16out2xGHIJKLMN, voutput_max);
    vfp16out2xOPQRSTUV = vminq_f16(vfp16out2xOPQRSTUV, voutput_max);
    vfp16out3x01234567 = vminq_f16(vfp16out3x01234567, voutput_max);
    vfp16out3x89ABCDEF = vminq_f16(vfp16out3x89ABCDEF, voutput_max);
    vfp16out3xGHIJKLMN = vminq_f16(vfp16out3xGHIJKLMN, voutput_max);
    vfp16out3xOPQRSTUV = vminq_f16(vfp16out3xOPQRSTUV, voutput_max);
    if XNN_LIKELY(nc >= 32) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));
      vst1q_u16(c0 + 16, vreinterpretq_u16_f16(vfp16out0xGHIJKLMN));
      vst1q_u16(c0 + 24, vreinterpretq_u16_f16(vfp16out0xOPQRSTUV));
      vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vfp16out1x89ABCDEF));
      vst1q_u16(c1 + 16, vreinterpretq_u16_f16(vfp16out1xGHIJKLMN));
      vst1q_u16(c1 + 24, vreinterpretq_u16_f16(vfp16out1xOPQRSTUV));
      vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vfp16out2x89ABCDEF));
      vst1q_u16(c2 + 16, vreinterpretq_u16_f16(vfp16out2xGHIJKLMN));
      vst1q_u16(c2 + 24, vreinterpretq_u16_f16(vfp16out2xOPQRSTUV));
      vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vfp16out3x89ABCDEF));
      vst1q_u16(c3 + 16, vreinterpretq_u16_f16(vfp16out3xGHIJKLMN));
      vst1q_u16(c3 + 24, vreinterpretq_u16_f16(vfp16out3xOPQRSTUV));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 32;
    } else {
     if (nc & 16) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0xGHIJKLMN;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567)); c1 += 8;
       vfp16out1x01234567 = vfp16out1xGHIJKLMN;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567)); c2 += 8;
       vfp16out2x01234567 = vfp16out2xGHIJKLMN;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567)); c3 += 8;
       vfp16out3x01234567 = vfp16out3xGHIJKLMN;
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x89ABCDEF)); c0 += 8;
       vfp16out0x89ABCDEF = vfp16out0xOPQRSTUV;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x89ABCDEF)); c1 += 8;
       vfp16out1x89ABCDEF = vfp16out1xOPQRSTUV;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x89ABCDEF)); c2 += 8;
       vfp16out2x89ABCDEF = vfp16out2xOPQRSTUV;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x89ABCDEF)); c3 += 8;
       vfp16out3x89ABCDEF = vfp16out3xOPQRSTUV;
     }
     if (nc & 8) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567)); c1 += 8;
       vfp16out1x01234567 = vfp16out1x89ABCDEF;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567)); c2 += 8;
       vfp16out2x01234567 = vfp16out2x89ABCDEF;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567)); c3 += 8;
       vfp16out3x01234567 = vfp16out3x89ABCDEF;
     }
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     float16x4_t vfp16out1x0123 = vget_low_f16(vfp16out1x01234567);
     float16x4_t vfp16out2x0123 = vget_low_f16(vfp16out2x01234567);
     float16x4_t vfp16out3x0123 = vget_low_f16(vfp16out3x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vst1_u16(c1, vreinterpret_u16_f16(vfp16out1x0123)); c1 += 4;
       vst1_u16(c2, vreinterpret_u16_f16(vfp16out2x0123)); c2 += 4;
       vst1_u16(c3, vreinterpret_u16_f16(vfp16out3x0123)); c3 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
       vfp16out1x0123 = vget_high_f16(vfp16out1x01234567);
       vfp16out2x0123 = vget_high_f16(vfp16out2x01234567);
       vfp16out3x0123 = vget_high_f16(vfp16out3x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vfp16out1x0123), 0); c1 += 2;
       vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vfp16out2x0123), 0); c2 += 2;
       vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vfp16out3x0123), 0); c3 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
       vfp16out1x0123 = vext_f16(vfp16out1x0123, vfp16out1x0123, 2);
       vfp16out2x0123 = vext_f16(vfp16out2x0123, vfp16out2x0123, 2);
       vfp16out3x0123 = vext_f16(vfp16out3x0123, vfp16out3x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
       vst1_lane_u16(c1, vreinterpret_u16_f16(vfp16out1x0123), 0);
       vst1_lane_u16(c2, vreinterpret_u16_f16(vfp16out2x0123), 0);
       vst1_lane_u16(c3, vreinterpret_u16_f16(vfp16out3x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
