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

#include <xnnpack/vbinary.h>
#include <xnnpack/intrinsics-polyfill.h>

void xnn_qs8_vadd_minmax_ukernel__hvx_u32(
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

  const HVX_Vector vbias = Q6_V_vsplat_R(*((int32_t *) &params->hvx.bias));
  const HVX_Vector va_multiplier = Q6_V_vsplat_R(*((int32_t *) &params->hvx.a_multiplier));
  const HVX_Vector vb_multiplier = Q6_V_vsplat_R(*((int32_t *) &params->hvx.b_multiplier));
  const int32_t shift = params->hvx.shift;
  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(*((int16_t *) &params->hvx.output_zero_point));
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(*((int8_t *) &params->hvx.output_min));
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(*((int8_t *) &params->hvx.output_max));
  int8_t* ptr_o = output;

  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    HVX_Vector va0 = *((HVX_UVector*)input_a);
    HVX_Vector vb0 = *((HVX_UVector*)input_b);
    input_a += 32;
    input_b += 32;

    // unpack: 8 bit to 16 bit
    HVX_VectorPair va0_i16 = Q6_Wh_vsxt_Vb(va0);
    HVX_Vector va0_i16_even = Q6_V_lo_W(va0_i16);
    HVX_Vector va0_i16_odd = Q6_V_hi_W(va0_i16);

    HVX_VectorPair vb0_i16 = Q6_Wh_vsxt_Vb(vb0);
    HVX_Vector vb0_i16_even = Q6_V_lo_W(vb0_i16);
    HVX_Vector vb0_i16_odd = Q6_V_hi_W(vb0_i16);

    // vacc = va * va_multiplier + vb * vb_multiplier with expanding to 32 bit
    HVX_VectorPair va0_mul_even = Q6_Vw_vmpyi_VwVh(va_multiplier, va0_i16_even);
    HVX_VectorPair va0_mul_odd = Q6_Vw_vmpyi_VwVh(va_multiplier, va0_i16_odd);
    HVX_VectorPair vb0_mul_even = Q6_Vw_vmpyi_VwVh(vb_multiplier, vb0_i16_even);
    HVX_VectorPair vb0_mul_odd = Q6_Vw_vmpyi_VwVh(vb_multiplier, vb0_i16_odd);

    HVX_Vector vacc0_even = Q6_Vw_vadd_VwVw(Q6_V_lo_W(va0_mul_even), Q6_V_lo_W(vb0_mul_even));
    HVX_Vector vacc0_odd = Q6_Vw_vadd_VwVw(Q6_V_lo_W(va0_mul_odd), Q6_V_lo_W(vb0_mul_odd));

    // vacc = vbias + vacc
    vacc0_even = Q6_Vw_vadd_VwVw(vbias, vacc0_even);
    vacc0_odd = Q6_Vw_vadd_VwVw(vbias, vacc0_odd);

    // right shift, lower to 16 bit, add output_zero_point
    HVX_Vector vacc0 = Q6_Vh_vasr_VwVwR_sat(Q6_Vw_vasr_VwR(vacc0_odd, shift), Q6_Vw_vasr_VwR(vacc0_even, shift), 0);
    vacc0 = Q6_Vh_vadd_VhVh(voutput_zero_point, vacc0);

    // pack: 16 bit to 8 bit
    HVX_Vector vout0 = Q6_Vb_vpack_VhVh_sat(vacc0, vacc0);

    // minmax
    vout0 = Q6_Vb_vmax_VbVb(voutput_min, vout0);
    vout0 = Q6_Vb_vmin_VbVb(voutput_max, vout0);

    // store output
    Q6_V_vstu_variable(ptr_o, 32, vout0);
    ptr_o += 32;
  }
  if XNN_UNLIKELY(batch != 0){
    do {
      HVX_Vector va = *((HVX_UVector*)input_a);
      HVX_Vector vb = *((HVX_UVector*)input_b);
      if XNN_LIKELY(batch > (32 * sizeof(int8_t))) {
        input_a += 32;
        input_b += 32;
      }

      // unpack: 8 bit to 16 bit
      HVX_Vector va_i16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(va));
      HVX_Vector vb_i16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vb));

      // vacc = vbias + va * va_multiplier + vb * vb_multiplier
      HVX_Vector vmul_a = Q6_V_lo_W(Q6_Vw_vmpyi_VwVh(va_multiplier, va_i16));
      HVX_Vector vmul_b = Q6_V_lo_W(Q6_Vw_vmpyi_VwVh(vb_multiplier, vb_i16));
      HVX_Vector vacc = Q6_Vw_vadd_VwVw(vmul_a, vmul_b);
      vacc = Q6_Vw_vadd_VwVw(vacc, vbias);

      // right shift
      vacc = Q6_Vw_vasr_VwR(vacc, shift);

      // pack: 32 bit to 16 bit
      HVX_Vector vout = Q6_Vh_vadd_VhVh(voutput_zero_point, Q6_Vh_vpack_VwVw_sat(vacc, vacc));

      // pack: 16 bit to 8 bit
      vout = Q6_Vb_vpack_VhVh_sat(vout, vout);

      // minmax
      vout = Q6_Vb_vmax_VbVb(voutput_min, vout);
      vout = Q6_Vb_vmin_VbVb(voutput_max, vout);

      // store output
      if XNN_LIKELY(batch > (32 * sizeof(int8_t))) {
        Q6_V_vstu_variable(ptr_o, 32, vout);
        ptr_o += 32;
        batch -=32;
      }
      else{
        Q6_V_vstu_variable(ptr_o, batch, vout);
        batch = 0;
      }
    } while (batch != 0);
  }
}
