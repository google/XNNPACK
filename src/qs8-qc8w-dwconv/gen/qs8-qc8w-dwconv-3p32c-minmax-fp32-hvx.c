// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
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

  *((HVX_Vector *)&vacc_buffer) = vacc;
  *((HVX_Vector *)&vscale_buffer) = vscale;

  for (int i = 0; i < 32; ++i) {
    vacc_buffer[i] = (int32_t)lrintf((float)vacc_buffer[i] * vscale_buffer[i]);
  }
  return *(HVX_Vector *)&vacc_buffer;
}
#endif  // __HVX_ARCH__ >= 73

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p32c__hvx(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(params->fp32_scalar.output_max);

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
      HVX_Vector vacc = *((const HVX_UVector*) w);

      const HVX_Vector vi0 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i0)))));
      const HVX_Vector vk0 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t)))))));
      i0 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi0, vk0);
      const HVX_Vector vi1 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i1)))));
      const HVX_Vector vk1 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t)))))));
      i1 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi1, vk1);
      const HVX_Vector vi2 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i2)))));
      const HVX_Vector vk2 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t)))))));
      i2 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi2, vk2);

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t));

      const HVX_Vector vscale = *((const HVX_UVector*) w);
      w = (const void*) ((const float*) w + 32);
      vacc = rescale_fp32(vacc, vscale);

      HVX_Vector vout = Q6_Vh_vpack_VwVw_sat(vacc, vacc);
      vout = Q6_Vh_vadd_VhVh_sat(vout, voutput_zero_point);
      vout = Q6_Vb_vpack_VhVh_sat(vout, vout);
      vout = Q6_Vb_vmax_VbVb(vout, voutput_min);
      vout = Q6_Vb_vmin_VbVb(vout, voutput_max);

      Q6_V_vstu_variable(output, 32, vout);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      HVX_Vector vacc = *((const HVX_UVector*) w);

      const HVX_Vector vi0 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i0)))));
      const HVX_Vector vk0 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi0, vk0);
      const HVX_Vector vi1 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i1)))));
      const HVX_Vector vk1 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi1, vk1);
      const HVX_Vector vi2 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i2)))));
      const HVX_Vector vk2 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi2, vk2);

      const HVX_Vector vscale = *((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      vacc = rescale_fp32(vacc, vscale);

      HVX_Vector vout = Q6_Vh_vpack_VwVw_sat(vacc, vacc);
      vout = Q6_Vh_vadd_VhVh_sat(vout, voutput_zero_point);
      vout = Q6_Vb_vpack_VhVh_sat(vout, vout);
      vout = Q6_Vb_vmax_VbVb(vout, voutput_min);
      vout = Q6_Vb_vmin_VbVb(vout, voutput_max);

      Q6_V_vstu_variable(output, c, vout);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
