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

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__hvx(
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
      const HVX_Vector vi3 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i3)))));
      const HVX_Vector vk3 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t)))))));
      i3 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi3, vk3);
      const HVX_Vector vi4 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i4)))));
      const HVX_Vector vk4 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t)))))));
      i4 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi4, vk4);
      const HVX_Vector vi5 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i5)))));
      const HVX_Vector vk5 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t)))))));
      i5 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi5, vk5);
      const HVX_Vector vi6 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i6)))));
      const HVX_Vector vk6 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t)))))));
      i6 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi6, vk6);
      const HVX_Vector vi7 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i7)))));
      const HVX_Vector vk7 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t)))))));
      i7 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi7, vk7);
      const HVX_Vector vi8 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i8)))));
      const HVX_Vector vk8 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t)))))));
      i8 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi8, vk8);

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t));

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
      const HVX_Vector vi3 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i3)))));
      const HVX_Vector vk3 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi3, vk3);
      const HVX_Vector vi4 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i4)))));
      const HVX_Vector vk4 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi4, vk4);
      const HVX_Vector vi5 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i5)))));
      const HVX_Vector vk5 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi5, vk5);
      const HVX_Vector vi6 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i6)))));
      const HVX_Vector vk6 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi6, vk6);
      const HVX_Vector vi7 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i7)))));
      const HVX_Vector vk7 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi7, vk7);
      const HVX_Vector vi8 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i8)))));
      const HVX_Vector vk8 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi8, vk8);

      const HVX_Vector vscale = *((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t)));
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
