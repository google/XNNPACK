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

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__hvx(
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
      const HVX_Vector vi9 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i9)))));
      const HVX_Vector vk9 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t)))))));
      i9 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi9, vk9);
      const HVX_Vector vi10 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i10)))));
      const HVX_Vector vk10 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t)))))));
      i10 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi10, vk10);
      const HVX_Vector vi11 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i11)))));
      const HVX_Vector vk11 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t)))))));
      i11 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi11, vk11);
      const HVX_Vector vi12 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i12)))));
      const HVX_Vector vk12 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t)))))));
      i12 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi12, vk12);
      const HVX_Vector vi13 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i13)))));
      const HVX_Vector vk13 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t)))))));
      i13 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi13, vk13);
      const HVX_Vector vi14 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i14)))));
      const HVX_Vector vk14 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t)))))));
      i14 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi14, vk14);
      const HVX_Vector vi15 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i15)))));
      const HVX_Vector vk15 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t)))))));
      i15 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi15, vk15);
      const HVX_Vector vi16 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i16)))));
      const HVX_Vector vk16 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t)))))));
      i16 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi16, vk16);
      const HVX_Vector vi17 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i17)))));
      const HVX_Vector vk17 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t)))))));
      i17 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi17, vk17);
      const HVX_Vector vi18 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i18)))));
      const HVX_Vector vk18 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t)))))));
      i18 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi18, vk18);
      const HVX_Vector vi19 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i19)))));
      const HVX_Vector vk19 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t)))))));
      i19 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi19, vk19);
      const HVX_Vector vi20 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i20)))));
      const HVX_Vector vk20 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t)))))));
      i20 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi20, vk20);
      const HVX_Vector vi21 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i21)))));
      const HVX_Vector vk21 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t)))))));
      i21 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi21, vk21);
      const HVX_Vector vi22 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i22)))));
      const HVX_Vector vk22 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t)))))));
      i22 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi22, vk22);
      const HVX_Vector vi23 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i23)))));
      const HVX_Vector vk23 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t)))))));
      i23 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi23, vk23);
      const HVX_Vector vi24 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i24)))));
      const HVX_Vector vk24 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t)))))));
      i24 += 32;
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi24, vk24);

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t));

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
      const HVX_Vector vi9 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i9)))));
      const HVX_Vector vk9 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi9, vk9);
      const HVX_Vector vi10 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i10)))));
      const HVX_Vector vk10 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi10, vk10);
      const HVX_Vector vi11 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i11)))));
      const HVX_Vector vk11 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi11, vk11);
      const HVX_Vector vi12 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i12)))));
      const HVX_Vector vk12 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi12, vk12);
      const HVX_Vector vi13 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i13)))));
      const HVX_Vector vk13 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi13, vk13);
      const HVX_Vector vi14 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i14)))));
      const HVX_Vector vk14 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi14, vk14);
      const HVX_Vector vi15 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i15)))));
      const HVX_Vector vk15 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi15, vk15);
      const HVX_Vector vi16 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i16)))));
      const HVX_Vector vk16 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi16, vk16);
      const HVX_Vector vi17 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i17)))));
      const HVX_Vector vk17 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi17, vk17);
      const HVX_Vector vi18 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i18)))));
      const HVX_Vector vk18 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi18, vk18);
      const HVX_Vector vi19 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i19)))));
      const HVX_Vector vk19 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi19, vk19);
      const HVX_Vector vi20 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i20)))));
      const HVX_Vector vk20 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi20, vk20);
      const HVX_Vector vi21 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i21)))));
      const HVX_Vector vk21 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi21, vk21);
      const HVX_Vector vi22 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i22)))));
      const HVX_Vector vk22 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi22, vk22);
      const HVX_Vector vi23 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i23)))));
      const HVX_Vector vk23 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi23, vk23);
      const HVX_Vector vi24 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) i24)))));
      const HVX_Vector vk24 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t)))))));
      vacc = Q6_Vw_vmpyieacc_VwVwVh(vacc, vi24, vk24);

      const HVX_Vector vscale = *((const HVX_UVector*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t)));
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
