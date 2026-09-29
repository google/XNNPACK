// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-qs8-vcvt/rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/vcvt.h"


void xnn_f16_qu8_vcvt_ukernel__rvvfp16arith_u4v(
    size_t batch,
    const xnn_float16* input,
    uint8_t* output,
    const struct xnn_f16_qu8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;

  // A reciprocal output scale that underflows fp16 to zero would turn an
  // infinite input into NaN via inf * 0, which the clamp below resolves to the
  // wrong bound. Clamp the scale to the smallest positive fp16 so infinities
  // saturate by sign, mirroring the scalar kernel's max(FLT_MIN, scale) guard.
  // Nothing representable lies between 0 and 0x1.0p-24, so non-zero scales are
  // unaffected.
  xnn_float16 vscale = params->scalar.scale;
  if (!(vscale > (xnn_float16) 0x1.0p-24f)) {
    vscale = (xnn_float16) 0x1.0p-24f;
  }
  const int16_t voutput_zero_point = params->scalar.output_zero_point;
  // vfcvt and vncvt do not saturate, so clamp before converting. The bounds
  // span at most [-255, 255], which fp16 represents exactly, and the scaled
  // value is in [0, 255] once the zero point is added back.
  const xnn_float16 voutput_min_less_zero_point =
      (xnn_float16) (0 - (int32_t) voutput_zero_point);
  const xnn_float16 voutput_max_less_zero_point =
      (xnn_float16) (255 - (int32_t) voutput_zero_point);

  do {
    const size_t n = __riscv_vsetvl_e16m4(batch);

    vfloat16m4_t vx = __riscv_vle16_v_f16m4(input, n);
    input += n;

    vx = __riscv_vfmul_vf_f16m4(vx, vscale, n);
    vx = __riscv_vfmax_vf_f16m4(vx, voutput_min_less_zero_point, n);
    vx = __riscv_vfmin_vf_f16m4(vx, voutput_max_less_zero_point, n);

    vint16m4_t vacc = __riscv_vfcvt_x_f_v_i16m4(vx, n);
    vacc = __riscv_vadd_vx_i16m4(vacc, voutput_zero_point, n);

    __riscv_vse8_v_u8m2(output, __riscv_vncvt_x_x_w_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(vacc), n), n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
