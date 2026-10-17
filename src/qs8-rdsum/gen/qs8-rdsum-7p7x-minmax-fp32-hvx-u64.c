// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/hvx.c.in
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
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/s32-hvx.h"


void xnn_qs8_rdsum_ukernel_7p7x__hvx_u64(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 64; channels -= 64) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

    HVX_Vector vacc0 = Q6_V_vzero();
    HVX_Vector vacc32 = Q6_V_vzero();

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int num_batches = floor((rows + 251) / 252);
    int r = rows;
    for (; num_batches > 0; --num_batches) {
      HVX_Vector v16acc_0 = Q6_V_vzero();
      for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
        if XNN_UNPREDICTABLE(current_batch < 2) {
          i1 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 2) {
          i2 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 4) {
          i3 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 4) {
          i4 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 6) {
          i5 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 6) {
          i6 = zero;
        }
        const HVX_Vector vin0_0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i0[0])));
        v16acc_0 = Q6_Vh_vadd_VhVh(v16acc_0, vin0_0);
        const HVX_Vector vin1_0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i1[0])));
        v16acc_0 = Q6_Vh_vadd_VhVh(v16acc_0, vin1_0);
        const HVX_Vector vin2_0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i2[0])));
        v16acc_0 = Q6_Vh_vadd_VhVh(v16acc_0, vin2_0);
        const HVX_Vector vin3_0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i3[0])));
        v16acc_0 = Q6_Vh_vadd_VhVh(v16acc_0, vin3_0);
        const HVX_Vector vin4_0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i4[0])));
        v16acc_0 = Q6_Vh_vadd_VhVh(v16acc_0, vin4_0);
        const HVX_Vector vin5_0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i5[0])));
        v16acc_0 = Q6_Vh_vadd_VhVh(v16acc_0, vin5_0);
        const HVX_Vector vin6_0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i6[0])));
        v16acc_0 = Q6_Vh_vadd_VhVh(v16acc_0, vin6_0);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }
      const HVX_VectorPair vacc32_0 = Q6_Ww_vunpack_Vh(v16acc_0);
      vacc0 = Q6_Vw_vadd_VwVw(vacc0, Q6_V_lo_W(vacc32_0));
      vacc32 = Q6_Vw_vadd_VwVw(vacc32, Q6_V_hi_W(vacc32_0));
      r = doz(r, 252);
    }

    const int32_t* o = output;
    HVX_Vector vo0 = *((const HVX_UVector*) o); o += 32;
    HVX_Vector vo32 = *((const HVX_UVector*) o); o += 32;
    vo0 = Q6_Vw_vadd_VwVw(vacc0, vo0);
    vo32 = Q6_Vw_vadd_VwVw(vacc32, vo32);
    *((HVX_UVector*) output) = vo0; output += 32;
    *((HVX_UVector*) output) = vo32; output += 32;

    input = (const int8_t*) ((uintptr_t) input + 64 * sizeof(int8_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    // 256 int8s may be summed into an int16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const int8_t* i0 = input;
      const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
      const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
      const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
      const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
      const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
      const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

      HVX_Vector vacc0 = Q6_V_vzero();
      HVX_Vector vacc1 = Q6_V_vzero();

      for (; num_batches > 0; --num_batches) {
        HVX_Vector v16acc = Q6_V_vzero();
        for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
          if XNN_UNPREDICTABLE(current_batch < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 6) {
            i6 = zero;
          }

          const HVX_Vector vin0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i0[0])));
          v16acc = Q6_Vh_vadd_VhVh(v16acc, vin0);
          const HVX_Vector vin1 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i1[0])));
          v16acc = Q6_Vh_vadd_VhVh(v16acc, vin1);
          const HVX_Vector vin2 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i2[0])));
          v16acc = Q6_Vh_vadd_VhVh(v16acc, vin2);
          const HVX_Vector vin3 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i3[0])));
          v16acc = Q6_Vh_vadd_VhVh(v16acc, vin3);
          const HVX_Vector vin4 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i4[0])));
          v16acc = Q6_Vh_vadd_VhVh(v16acc, vin4);
          const HVX_Vector vin5 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i5[0])));
          v16acc = Q6_Vh_vadd_VhVh(v16acc, vin5);
          const HVX_Vector vin6 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*((const HVX_UVector*) &i6[0])));
          v16acc = Q6_Vh_vadd_VhVh(v16acc, vin6);
          i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
        }
        const HVX_VectorPair vacc32 = Q6_Ww_vunpack_Vh(v16acc);
        vacc0 = Q6_Vw_vadd_VwVw(vacc0, Q6_V_lo_W(vacc32));
        vacc1 = Q6_Vw_vadd_VwVw(vacc1, Q6_V_hi_W(vacc32));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 64) {
        HVX_Vector vo0 = *((const HVX_UVector*) output);
        HVX_Vector vo1 = *((const HVX_UVector*) (output + 32));
        vo0 = Q6_Vw_vadd_VwVw(vo0, vacc0);
        vo1 = Q6_Vw_vadd_VwVw(vo1, vacc1);
        *((HVX_UVector*) output) = vo0; output += 32;
        *((HVX_UVector*) output) = vo1; output += 32;
        channels -= 64;
        input = (const int8_t*) ((uintptr_t) input + 64 * sizeof(int8_t));
      } else {
        if (channels >= 32) {
          HVX_Vector vo0 = *((const HVX_UVector*) output);
          vo0 = Q6_Vw_vadd_VwVw(vo0, vacc0);
          *((HVX_UVector*) output) = vo0; output += 32;
          channels -= 32;
          vacc0 = vacc1;
        }
        if (channels != 0) {
          HVX_Vector vo0 = xnn_load_tail_safe_s32(output, channels);
          vo0 = Q6_Vw_vadd_VwVw(vo0, vacc0);
          Q6_V_vstu_variable(output, channels * sizeof(int32_t), vo0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
