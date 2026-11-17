// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-avgpool/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/avgpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


#if __HVX_ARCH__ >= 73
static XNN_INLINE HVX_Vector rescale_fp32(HVX_Vector vacc, HVX_Vector vscale)
{
  const HVX_Vector vaccf = Q6_Vsf_equals_Vw(vacc);
  const HVX_Vector vscaledqf = Q6_Vqf32_vmpy_VsfVsf(vaccf, vscale);

  // Create a vector of `0.5f` with the same sign as the entries of `a`.
  const HVX_Vector vhalf = Q6_V_vsplat_R(float_as_uint32(0.5f));
  const HVX_Vector vsign_mask = Q6_V_vsplat_R(0x80000000);
  const HVX_Vector vsigned_half = Q6_V_vor_VV(Q6_V_vand_VV(vaccf, vsign_mask), vhalf);
  const HVX_Vector vresult = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(vscaledqf, vsigned_half)));
  return vresult;
}
#else
static HVX_Vector rescale_fp32(HVX_Vector vacc, HVX_Vector vscale)
{
  XNN_ALIGN(128) int32_t vacc_buffer[32];
  XNN_ALIGN(128) float vscale_buffer[32];

  *((HVX_Vector*) &vacc_buffer) = vacc;
  *((HVX_Vector*) &vscale_buffer) = vscale;

  for (int i = 0; i < 32; ++i) {
    vacc_buffer[i] = (int32_t) lrintf((float) vacc_buffer[i] * vscale_buffer[i]);
  }
  return *(HVX_Vector*) &vacc_buffer;
}
#endif  // __HVX_ARCH__ >= 73

void xnn_qs8_avgpool_minmax_fp32_ukernel_9p__hvx_u32(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    size_t input_pixel_stride,
    const int8_t* zero,
    const float* multiplier,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_qs8_avgpool_minmax_params* restrict params) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const HVX_Vector vinit_bias = Q6_V_vsplat_R(params->scalar.init_bias);
  HVX_Vector vscale = Q6_V_vsplat_R(float_as_uint32(params->scalar.scale));
  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(params->scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->scalar.output_min);
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(params->scalar.output_max);

  do {
    if (multiplier != NULL) {
      vscale = Q6_V_vsplat_R(float_as_uint32(*multiplier++));
    }

    int8_t* o = output;
    size_t c_offset = 0;
    size_t c = channels;
    for (; c >= 32; c -= 32) {
      const int8_t** i = input;
      size_t k = kernel_elements;
      HVX_Vector vacc = Q6_V_vzero();
      do {
        vacc = Q6_Vw_vadd_VwVw(vacc, vinit_bias);

        const int8_t* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
        }
        i0 += c_offset;
        const HVX_Vector vi0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i0)));
        const int8_t* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
        }
        i1 += c_offset;
        const HVX_Vector vi1 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i1)));
        const int8_t* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
        }
        i2 += c_offset;
        const HVX_Vector vi2 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i2)));
        const int8_t* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
        }
        i3 += c_offset;
        const HVX_Vector vi3 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i3)));
        const int8_t* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
        }
        i4 += c_offset;
        const HVX_Vector vi4 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i4)));
        const int8_t* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
        }
        i5 += c_offset;
        const HVX_Vector vi5 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i5)));
        const int8_t* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
        }
        i6 += c_offset;
        const HVX_Vector vi6 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i6)));
        const int8_t* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
        }
        i7 += c_offset;
        const HVX_Vector vi7 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i7)));
        const int8_t* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
        }
        i8 += c_offset;
        const HVX_Vector vi8 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i8)));

        const HVX_Vector vsum018 = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(vi0, vi1), vi8);
        const HVX_Vector vsum23 = Q6_Vh_vadd_VhVh(vi2, vi3);
        const HVX_Vector vsum45 = Q6_Vh_vadd_VhVh(vi4, vi5);
        const HVX_Vector vsum67 = Q6_Vh_vadd_VhVh(vi6, vi7);
        const HVX_Vector vsum = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(vsum23, vsum45), Q6_Vh_vadd_VhVh(vsum018, vsum67));
        vacc = Q6_Vw_vadd_VwVw(vacc, Q6_V_lo_W(Q6_Ww_vunpack_Vh(vsum)));

        k = doz(k, 9);
      } while (k != 0);

      vacc = rescale_fp32(vacc, vscale);

      HVX_Vector vout = Q6_Vh_vpack_VwVw_sat(vacc, vacc);
      vout = Q6_Vh_vadd_VhVh_sat(vout, voutput_zero_point);
      vout = Q6_Vb_vpack_VhVh_sat(vout, vout);
      vout = Q6_Vb_vmax_VbVb(vout, voutput_min);
      vout = Q6_Vb_vmin_VbVb(vout, voutput_max);

      Q6_V_vstu_variable(o, 32, vout);
      o += 32;
      c_offset += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t** i = input;
      size_t k = kernel_elements;
      HVX_Vector vacc = Q6_V_vzero();
      do {
        vacc = Q6_Vw_vadd_VwVw(vacc, vinit_bias);

        const int8_t* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
        }
        i0 += c_offset;
        const HVX_Vector vi0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i0)));
        const int8_t* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
        }
        i1 += c_offset;
        const HVX_Vector vi1 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i1)));
        const int8_t* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
        }
        i2 += c_offset;
        const HVX_Vector vi2 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i2)));
        const int8_t* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
        }
        i3 += c_offset;
        const HVX_Vector vi3 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i3)));
        const int8_t* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
        }
        i4 += c_offset;
        const HVX_Vector vi4 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i4)));
        const int8_t* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
        }
        i5 += c_offset;
        const HVX_Vector vi5 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i5)));
        const int8_t* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
        }
        i6 += c_offset;
        const HVX_Vector vi6 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i6)));
        const int8_t* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
        }
        i7 += c_offset;
        const HVX_Vector vi7 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i7)));
        const int8_t* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
        }
        i8 += c_offset;
        const HVX_Vector vi8 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i8)));

        const HVX_Vector vsum018 = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(vi0, vi1), vi8);
        const HVX_Vector vsum23 = Q6_Vh_vadd_VhVh(vi2, vi3);
        const HVX_Vector vsum45 = Q6_Vh_vadd_VhVh(vi4, vi5);
        const HVX_Vector vsum67 = Q6_Vh_vadd_VhVh(vi6, vi7);
        const HVX_Vector vsum = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(vsum23, vsum45), Q6_Vh_vadd_VhVh(vsum018, vsum67));
        vacc = Q6_Vw_vadd_VwVw(vacc, Q6_V_lo_W(Q6_Ww_vunpack_Vh(vsum)));

        k = doz(k, 9);
      } while (k != 0);

      vacc = rescale_fp32(vacc, vscale);

      HVX_Vector vout = Q6_Vh_vpack_VwVw_sat(vacc, vacc);
      vout = Q6_Vh_vadd_VhVh_sat(vout, voutput_zero_point);
      vout = Q6_Vb_vpack_VhVh_sat(vout, vout);
      vout = Q6_Vb_vmax_VbVb(vout, voutput_min);
      vout = Q6_Vb_vmin_VbVb(vout, voutput_max);

      Q6_V_vstu_variable(o, c, vout);
    }

    input = (const int8_t**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
