// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum2/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/f32-scalar.h"


void xnn_f32_rdsum2_ukernel_7p7x__scalar_u4(
  size_t channels, size_t k1, size_t k2, size_t k3, const float* input,
  size_t input_stride1, size_t input_stride2, size_t input_stride3,
  const float* zero, float* output,
  const struct xnn_f32_scale_params* restrict params) {
  assert(k1 != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_simd_f32_t vscale = xnn_set1_f32(params->scalar.scale);
  float* original_output = output;
  size_t original_channels = channels;

  for (size_t k = 0; k < k3; ++k) {
    for (size_t j = 0; j < k2; ++j) {
      const float* input_row =
          (const float*)((uintptr_t)input + j * input_stride2
                         + k * input_stride3);
      output = original_output;
      channels = original_channels;

      assert(input_row != NULL);

      size_t input_increment = 7 * input_stride1;
      for (; channels >= 4; channels -= 4) {
        const float* i0 = input_row;
        const float* i1 = 
            (const float*) ((uintptr_t) input_row + 1 * input_stride1);
        const float* i2 = 
            (const float*) ((uintptr_t) input_row + 2 * input_stride1);
        const float* i3 = 
            (const float*) ((uintptr_t) input_row + 3 * input_stride1);
        const float* i4 = 
            (const float*) ((uintptr_t) input_row + 4 * input_stride1);
        const float* i5 = 
            (const float*) ((uintptr_t) input_row + 5 * input_stride1);
        const float* i6 = 
            (const float*) ((uintptr_t) input_row + 6 * input_stride1);

        xnn_simd_f32_t vacc0 = xnn_zero_f32();
        xnn_simd_f32_t vacc1 = xnn_zero_f32();
        xnn_simd_f32_t vacc2 = xnn_zero_f32();
        xnn_simd_f32_t vacc3 = xnn_zero_f32();

        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = zero;
          }
          xnn_simd_f32_t vin0;
          xnn_simd_f32_t vin1;
          xnn_simd_f32_t vin2;
          xnn_simd_f32_t vin3;
          vin0 = xnn_loadu_f32(&i0[0]);
          vin1 = xnn_loadu_f32(&i0[1]);
          vin2 = xnn_loadu_f32(&i0[2]);
          vin3 = xnn_loadu_f32(&i0[3]);
          vin0 = xnn_mul_f32(vin0, vin0);
          vin1 = xnn_mul_f32(vin1, vin1);
          vin2 = xnn_mul_f32(vin2, vin2);
          vin3 = xnn_mul_f32(vin3, vin3);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vin0 = xnn_loadu_f32(&i1[0]);
          vin1 = xnn_loadu_f32(&i1[1]);
          vin2 = xnn_loadu_f32(&i1[2]);
          vin3 = xnn_loadu_f32(&i1[3]);
          vin0 = xnn_mul_f32(vin0, vin0);
          vin1 = xnn_mul_f32(vin1, vin1);
          vin2 = xnn_mul_f32(vin2, vin2);
          vin3 = xnn_mul_f32(vin3, vin3);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vin0 = xnn_loadu_f32(&i2[0]);
          vin1 = xnn_loadu_f32(&i2[1]);
          vin2 = xnn_loadu_f32(&i2[2]);
          vin3 = xnn_loadu_f32(&i2[3]);
          vin0 = xnn_mul_f32(vin0, vin0);
          vin1 = xnn_mul_f32(vin1, vin1);
          vin2 = xnn_mul_f32(vin2, vin2);
          vin3 = xnn_mul_f32(vin3, vin3);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vin0 = xnn_loadu_f32(&i3[0]);
          vin1 = xnn_loadu_f32(&i3[1]);
          vin2 = xnn_loadu_f32(&i3[2]);
          vin3 = xnn_loadu_f32(&i3[3]);
          vin0 = xnn_mul_f32(vin0, vin0);
          vin1 = xnn_mul_f32(vin1, vin1);
          vin2 = xnn_mul_f32(vin2, vin2);
          vin3 = xnn_mul_f32(vin3, vin3);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vin0 = xnn_loadu_f32(&i4[0]);
          vin1 = xnn_loadu_f32(&i4[1]);
          vin2 = xnn_loadu_f32(&i4[2]);
          vin3 = xnn_loadu_f32(&i4[3]);
          vin0 = xnn_mul_f32(vin0, vin0);
          vin1 = xnn_mul_f32(vin1, vin1);
          vin2 = xnn_mul_f32(vin2, vin2);
          vin3 = xnn_mul_f32(vin3, vin3);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vin0 = xnn_loadu_f32(&i5[0]);
          vin1 = xnn_loadu_f32(&i5[1]);
          vin2 = xnn_loadu_f32(&i5[2]);
          vin3 = xnn_loadu_f32(&i5[3]);
          vin0 = xnn_mul_f32(vin0, vin0);
          vin1 = xnn_mul_f32(vin1, vin1);
          vin2 = xnn_mul_f32(vin2, vin2);
          vin3 = xnn_mul_f32(vin3, vin3);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vin0 = xnn_loadu_f32(&i6[0]);
          vin1 = xnn_loadu_f32(&i6[1]);
          vin2 = xnn_loadu_f32(&i6[2]);
          vin3 = xnn_loadu_f32(&i6[3]);
          vin0 = xnn_mul_f32(vin0, vin0);
          vin1 = xnn_mul_f32(vin1, vin1);
          vin2 = xnn_mul_f32(vin2, vin2);
          vin3 = xnn_mul_f32(vin3, vin3);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          i0 = (const float*) ((uintptr_t) i0 + input_increment);
          i1 = (const float*) ((uintptr_t) i1 + input_increment);
          i2 = (const float*) ((uintptr_t) i2 + input_increment);
          i3 = (const float*) ((uintptr_t) i3 + input_increment);
          i4 = (const float*) ((uintptr_t) i4 + input_increment);
          i5 = (const float*) ((uintptr_t) i5 + input_increment);
          i6 = (const float*) ((uintptr_t) i6 + input_increment);
        }
        vacc0 = xnn_mul_f32(vacc0, vscale);
        vacc1 = xnn_mul_f32(vacc1, vscale);
        vacc2 = xnn_mul_f32(vacc2, vscale);
        vacc3 = xnn_mul_f32(vacc3, vscale);

        const float* o = output;
        xnn_simd_f32_t vo0 = xnn_loadu_f32(o); o += 1;
        xnn_simd_f32_t vo1 = xnn_loadu_f32(o); o += 1;
        xnn_simd_f32_t vo2 = xnn_loadu_f32(o); o += 1;
        xnn_simd_f32_t vo3 = xnn_loadu_f32(o); o += 1;
        vacc0 = xnn_add_f32(vo0, vacc0);
        vacc1 = xnn_add_f32(vo1, vacc1);
        vacc2 = xnn_add_f32(vo2, vacc2);
        vacc3 = xnn_add_f32(vo3, vacc3);
        xnn_storeu_f32(output, vacc0); output += 1;
        xnn_storeu_f32(output, vacc1); output += 1;
        xnn_storeu_f32(output, vacc2); output += 1;
        xnn_storeu_f32(output, vacc3); output += 1;

        input_row = (const float*) ((uintptr_t) input_row + 4 * sizeof(float));
      }
      if (channels != 0) {
        input_increment = 7 * input_stride1;
        const float* i0 = input_row;
        const float* i1 =
            (const float*) ((uintptr_t) input_row + 1 * input_stride1);
        const float* i2 =
            (const float*) ((uintptr_t) input_row + 2 * input_stride1);
        const float* i3 =
            (const float*) ((uintptr_t) input_row + 3 * input_stride1);
        const float* i4 =
            (const float*) ((uintptr_t) input_row + 4 * input_stride1);
        const float* i5 =
            (const float*) ((uintptr_t) input_row + 5 * input_stride1);
        const float* i6 =
            (const float*) ((uintptr_t) input_row + 6 * input_stride1);
        xnn_simd_f32_t vacc[4];
        vacc[0] = xnn_zero_f32();
        vacc[1] = xnn_zero_f32();
        vacc[2] = xnn_zero_f32();
        vacc[3] = xnn_zero_f32();

        const size_t num_full_chunks = channels >> 0;
        const size_t num_chunks = round_up_po2(channels, 1) >> 0;
        const size_t remainder = channels & 0;
        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = zero;
          }
          for (int i = 0; i < num_full_chunks; ++i) {
            xnn_simd_f32_t vin0 = xnn_loadu_f32(&i0[i*1]);
            xnn_simd_f32_t vin1 = xnn_loadu_f32(&i1[i*1]);
            xnn_simd_f32_t vin2 = xnn_loadu_f32(&i2[i*1]);
            xnn_simd_f32_t vin3 = xnn_loadu_f32(&i3[i*1]);
            xnn_simd_f32_t vin4 = xnn_loadu_f32(&i4[i*1]);
            xnn_simd_f32_t vin5 = xnn_loadu_f32(&i5[i*1]);
            xnn_simd_f32_t vin6 = xnn_loadu_f32(&i6[i*1]);
            vin0 = xnn_mul_f32(vin0, vin0);
            vin1 = xnn_mul_f32(vin1, vin1);
            vin2 = xnn_mul_f32(vin2, vin2);
            vin3 = xnn_mul_f32(vin3, vin3);
            vin4 = xnn_mul_f32(vin4, vin4);
            vin5 = xnn_mul_f32(vin5, vin5);
            vin6 = xnn_mul_f32(vin6, vin6);
            vacc[i] = xnn_add_f32(vin0, vacc[i]);
            vacc[i] = xnn_add_f32(vin1, vacc[i]);
            vacc[i] = xnn_add_f32(vin2, vacc[i]);
            vacc[i] = xnn_add_f32(vin3, vacc[i]);
            vacc[i] = xnn_add_f32(vin4, vacc[i]);
            vacc[i] = xnn_add_f32(vin5, vacc[i]);
            vacc[i] = xnn_add_f32(vin6, vacc[i]);
          }

          if (remainder) {
            xnn_simd_f32_t vin0 = xnn_load_tail_f32(&i0[num_full_chunks*1], remainder);
            xnn_simd_f32_t vin1 = xnn_load_tail_f32(&i1[num_full_chunks*1], remainder);
            xnn_simd_f32_t vin2 = xnn_load_tail_f32(&i2[num_full_chunks*1], remainder);
            xnn_simd_f32_t vin3 = xnn_load_tail_f32(&i3[num_full_chunks*1], remainder);
            xnn_simd_f32_t vin4 = xnn_load_tail_f32(&i4[num_full_chunks*1], remainder);
            xnn_simd_f32_t vin5 = xnn_load_tail_f32(&i5[num_full_chunks*1], remainder);
            xnn_simd_f32_t vin6 = xnn_load_tail_f32(&i6[num_full_chunks*1], remainder);
            vin0 = xnn_mul_f32(vin0, vin0);
            vin1 = xnn_mul_f32(vin1, vin1);
            vin2 = xnn_mul_f32(vin2, vin2);
            vin3 = xnn_mul_f32(vin3, vin3);
            vin4 = xnn_mul_f32(vin4, vin4);
            vin5 = xnn_mul_f32(vin5, vin5);
            vin6 = xnn_mul_f32(vin6, vin6);
            vacc[num_full_chunks] = xnn_add_f32(vin0, vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(vin1, vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(vin2, vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(vin3, vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(vin4, vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(vin5, vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(vin6, vacc[num_full_chunks]);
          }
          i0 = (const float*) ((uintptr_t) i0 + input_increment);
          i1 = (const float*) ((uintptr_t) i1 + input_increment);
          i2 = (const float*) ((uintptr_t) i2 + input_increment);
          i3 = (const float*) ((uintptr_t) i3 + input_increment);
          i4 = (const float*) ((uintptr_t) i4 + input_increment);
          i5 = (const float*) ((uintptr_t) i5 + input_increment);
          i6 = (const float*) ((uintptr_t) i6 + input_increment);
        }
        for (size_t i = 0; i < num_chunks; ++i) {
          vacc[i] = xnn_mul_f32(vacc[i], vscale);
        }

        xnn_simd_f32_t vo[4];
        const float* o = output;
        for (int i = 0; i < channels >> 0; ++i) {
          vo[i] = xnn_loadu_f32(o); o += 1;
        }
        for (int i = 0; i < channels >> 0; ++i) {
          vacc[i] = xnn_add_f32(vo[i], vacc[i]);
        }
        for (int i = 0; i < channels >> 0; ++i) {
          xnn_storeu_f32(output, vacc[i]); output += 1;
        }
        if (remainder) {
          const size_t pos = num_full_chunks;
          xnn_simd_f32_t vout = vacc[pos];
          const xnn_simd_f32_t vdata = xnn_load_tail_safe_f32(output, remainder);
          vout = xnn_add_f32(vout, vdata);
          xnn_store_tail_f32(output, vout, remainder);
        }
      }
    }
  }
}
