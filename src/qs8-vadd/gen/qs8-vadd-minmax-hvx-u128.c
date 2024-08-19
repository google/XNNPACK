// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <hvx_hexagon_protos.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"

void xnn_qs8_vadd_minmax_ukernel__hvx_u128(
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

  const HVX_Vector vbias = Q6_V_vsplat_R(*((int32_t *) &params->scalar.bias));
  const HVX_Vector va_multiplier = Q6_V_vsplat_R(*((int32_t *) &params->scalar.a_multiplier));
  const HVX_Vector vb_multiplier = Q6_V_vsplat_R(*((int32_t *) &params->scalar.b_multiplier));
  const int32_t first_shift = params->scalar.first_shift;
  const int32_t rest_shift = params->scalar.rest_shift;
  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(*((int16_t *) &params->scalar.output_zero_point));
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(*((int8_t *) &params->scalar.output_min));
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(*((int8_t *) &params->scalar.output_max));

  for (; batch >= 128 * sizeof(int8_t); batch -= 128 * sizeof(int8_t)) {
    HVX_Vector va0 = *((HVX_UVector*)input_a);
    HVX_Vector vb0 = *((HVX_UVector*)input_b);
    input_a += 128;
    input_b += 128;

    // widen 8-bit to 16-bit
    HVX_VectorPair va0_i16 = Q6_Wh_vunpack_Vb(va0);
    HVX_VectorPair vb0_i16 = Q6_Wh_vunpack_Vb(vb0);
    HVX_Vector va0_lo = Q6_V_lo_W(va0_i16);
    HVX_Vector va0_hi = Q6_V_hi_W(va0_i16);
    HVX_Vector vb0_lo = Q6_V_lo_W(vb0_i16);
    HVX_Vector vb0_hi = Q6_V_hi_W(vb0_i16);

    // vacc = vbias + va * va_multiplier + vb * vb_multiplier with widening 16-bit to 32-bit
    HVX_Vector vacc0_lo_even = vbias;
    vacc0_lo_even = Q6_Vw_vmpyieacc_VwVwVh(vacc0_lo_even, va_multiplier, va0_lo);
    HVX_Vector vacc0_lo_odd = Q6_Vw_vadd_VwVw(vbias, Q6_Vw_vmpyio_VwVh(va_multiplier, va0_lo));

    vacc0_lo_even = Q6_Vw_vmpyieacc_VwVwVh(vacc0_lo_even, vb_multiplier, vb0_lo);
    vacc0_lo_odd = Q6_Vw_vadd_VwVw(vacc0_lo_odd, Q6_Vw_vmpyio_VwVh(vb_multiplier, vb0_lo));

    HVX_Vector vacc0_hi_even = vbias;
    vacc0_hi_even = Q6_Vw_vmpyieacc_VwVwVh(vacc0_hi_even, va_multiplier, va0_hi);
    HVX_Vector vacc0_hi_odd = Q6_Vw_vadd_VwVw(vbias, Q6_Vw_vmpyio_VwVh(va_multiplier, va0_hi));

    vacc0_hi_even = Q6_Vw_vmpyieacc_VwVwVh(vacc0_hi_even, vb_multiplier, vb0_hi);
    vacc0_hi_odd = Q6_Vw_vadd_VwVw(vacc0_hi_odd, Q6_Vw_vmpyio_VwVh(vb_multiplier, vb0_hi));

    // narrow shift to 16-bit
    // vacc = vacc + voutput_zero_point
    HVX_Vector vacc0_lo = Q6_Vh_vasr_VwVwR_sat(vacc0_lo_odd, vacc0_lo_even, first_shift);
    vacc0_lo = Q6_Vh_vadd_VhVh(voutput_zero_point, Q6_Vh_vasr_VhR(vacc0_lo, rest_shift));
    HVX_Vector vacc0_hi = Q6_Vh_vasr_VwVwR_sat(vacc0_hi_odd, vacc0_hi_even, first_shift);
    vacc0_hi = Q6_Vh_vadd_VhVh(voutput_zero_point, Q6_Vh_vasr_VhR(vacc0_hi, rest_shift));

    // narrow 16-bit to 8-bit
    HVX_Vector vout0 = Q6_Vb_vpack_VhVh_sat(vacc0_hi, vacc0_lo);

    // minmax
    vout0 = Q6_Vb_vmax_VbVb(voutput_min, vout0);
    vout0 = Q6_Vb_vmin_VbVb(voutput_max, vout0);

    // store output
    *((HVX_UVector *) output) = vout0;
    output += 128;
  }
  if XNN_UNLIKELY(batch != 0){
    do {
      HVX_Vector va = *((HVX_UVector*)input_a);
      HVX_Vector vb = *((HVX_UVector*)input_b);
      if XNN_LIKELY(batch > (32 * sizeof(int8_t))) {
        input_a += 32;
        input_b += 32;
      }

      HVX_VectorPair va_i16 = Q6_Wh_vunpack_Vb(va);
      HVX_Vector va_lo = Q6_V_lo_W(va_i16);

      HVX_VectorPair vb_i16 = Q6_Wh_vunpack_Vb(vb);
      HVX_Vector vb_lo = Q6_V_lo_W(vb_i16);

      HVX_Vector vacc_even = vbias;
      vacc_even = Q6_Vw_vmpyieacc_VwVwVh(vacc_even, va_multiplier, va_lo);
      HVX_Vector vacc_odd = Q6_Vw_vadd_VwVw(vbias, Q6_Vw_vmpyio_VwVh(va_multiplier, va_lo));

      vacc_even = Q6_Vw_vmpyieacc_VwVwVh(vacc_even, vb_multiplier, vb_lo);
      vacc_odd = Q6_Vw_vadd_VwVw(vacc_odd, Q6_Vw_vmpyio_VwVh(vb_multiplier, vb_lo));

      HVX_Vector vacc = Q6_Vh_vasr_VwVwR_sat(vacc_odd, vacc_even, first_shift);
      vacc = Q6_Vh_vadd_VhVh(voutput_zero_point, Q6_Vh_vasr_VhR(vacc, rest_shift));

      HVX_Vector vout = Q6_Vb_vpack_VhVh_sat(vacc, vacc);

      vout = Q6_Vb_vmax_VbVb(voutput_min, vout);
      vout = Q6_Vb_vmin_VbVb(voutput_max, vout);

      if XNN_LIKELY(batch > (32 * sizeof(int8_t))) {
        Q6_V_vstu_variable(output, 32, vout);
        output += 32;
        batch -=32;
      }
      else{
        Q6_V_vstu_variable(output, batch, vout);
        batch = 0;
      }
    } while (batch != 0);
  }
}
