// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/fft.h>

#include <arm_neon.h>

void xnn_cs16_fftr_ukernel__neon_x4(
    size_t samples,
    int16_t* data,
    const int16_t* twiddle)
{
  assert(samples != 0);
  assert(samples % 8 == 0);
  assert(data != NULL);
  assert(twiddle != NULL);

  int16_t* dl = data;
  int16_t* dr = data + samples * 2;
  int32_t vdcr = (int32_t) dl[0];
  int32_t vdci = (int32_t) dl[1];

  vdcr = math_asr_s32(vdcr * 16383 + 16384, 15);
  vdci = math_asr_s32(vdci * 16383 + 16384, 15);

  dl[0] = vdcr + vdci;
  dl[1] = 0;
  dl += 2;
  dr[0] = vdcr - vdci;
  dr[1] = 0;

  const int16x4_t vdiv2 = vdup_n_s16(16383);

  do {
    dr -= 8;
    const int16x4x2_t vil = vld2_s16(dl);
    const int16x4x2_t vir = vld2_s16(dr);
    const int16x4x2_t vtw = vld2_s16(twiddle);  twiddle += 8;

    int16x4_t virr = vrev64_s16(vir.val[0]);
    int16x4_t viri = vrev64_s16(vir.val[1]);

    const int16x4_t vilr = vqrdmulh_s16(vil.val[0], vdiv2);
    const int16x4_t vili = vqrdmulh_s16(vil.val[1], vdiv2);
    virr = vqrdmulh_s16(virr, vdiv2);
    viri = vqrdmulh_s16(viri, vdiv2);

    const int32x4_t vacc1r = vaddl_s16(vilr, virr);
    const int32x4_t vacc1i = vsubl_s16(vili, viri);
    const int16x4_t vacc2r = vsub_s16(vilr, virr);
    const int16x4_t vacc2i = vadd_s16(vili, viri);

    int32x4_t vaccr = vmull_s16(vacc2r, vtw.val[0]);
    int32x4_t vacci = vmull_s16(vacc2r, vtw.val[1]);
    vaccr = vmlsl_s16(vaccr, vacc2i, vtw.val[1]);
    vacci = vmlal_s16(vacci, vacc2i, vtw.val[0]);
    vaccr = vrshrq_n_s32(vaccr, 15);
    vacci = vrshrq_n_s32(vacci, 15);

    const int32x4_t vacclr = vhaddq_s32(vacc1r, vaccr);
    const int32x4_t vaccli = vhaddq_s32(vacc1i, vacci);
    const int32x4_t vaccrr = vhsubq_s32(vacc1r, vaccr);
    const int32x4_t vaccri = vhsubq_s32(vacci, vacc1i);

    int16x4x2_t voutl;
    int16x4x2_t voutr;
    voutl.val[0] = vmovn_s32(vacclr);
    voutl.val[1] = vmovn_s32(vaccli);
    voutr.val[0] = vrev64_s16(vmovn_s32(vaccrr));
    voutr.val[1] = vrev64_s16(vmovn_s32(vaccri));

    vst2_s16(dl, voutl);
    vst2_s16(dr, voutr);
    dl += 8;

    samples -= 8;
  } while(samples != 0);
}
