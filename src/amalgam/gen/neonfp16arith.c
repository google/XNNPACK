// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/conv.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/gemm.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pavgpool.h"
#include "xnnpack/prefetch.h"
#include "xnnpack/prelu.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "xnnpack/reduce.h"
#include "xnnpack/spmm.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vmulcaddc.h"
#include "xnnpack/vunary.h"


void xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  do {
    {
      const uint16_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint16_t* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

        vst1q_u16(b, vreinterpretq_u16_f16(vsum)); b += 8;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b));

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        vst1q_u16(b, vreinterpretq_u16_f16(vsum)); b += 8;
      }
    }

    assert(k >= 1);
    {
      const uint16_t* i0 = (const uint16_t*) input[0];
      assert(i0 != NULL);
      const uint16_t* i1 = (const uint16_t*) input[1];
      const uint16_t* i2 = (const uint16_t*) input[2];
      const uint16_t* i3 = (const uint16_t*) input[3];
      const uint16_t* i4 = (const uint16_t*) input[4];
      const uint16_t* i5 = (const uint16_t*) input[5];
      const uint16_t* i6 = (const uint16_t*) input[6];
      const uint16_t* i7 = (const uint16_t*) input[7];
      input = (const void**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = (const uint16_t*) zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = (const uint16_t*) zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = (const uint16_t*) zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = (const uint16_t*) zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = (const uint16_t*) zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = (const uint16_t*) zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = (const uint16_t*) zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      uint16_t* b = (uint16_t*) buffer;
      while (c >= 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        float16x8_t vout = vmulq_f16(vsum, vscale);
        vout = vmaxq_f16(vout, vmin);
        vout = vminq_f16(vout, vmax);

        vst1q_u16(output, vreinterpretq_u16_f16(vout)); output = (uint16_t*) output + 8;

        c -= 8;
      }
      if (c != 0) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0));
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1));
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2));
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3));
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4));
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5));
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6));
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7));
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b));

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        float16x8_t vout = vmulq_f16(vsum, vscale);
        vout = vmaxq_f16(vout, vmin);
        vout = vminq_f16(vout, vmax);

        float16x4_t vout_lo = vget_low_f16(vout);
        if (c & 4) {
          vst1_u16(output, vreinterpret_u16_f16(vout_lo)); output = (uint16_t*) output + 4;
          vout_lo = vget_high_f16(vout);
        }
        if (c & 2) {
          vst1_lane_u32(output, vreinterpret_u32_f16(vout_lo), 0); output = (uint16_t*) output + 2;
          vout_lo = vext_f16(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_u16(output, vreinterpret_u16_f16(vout_lo), 0); output = (uint16_t*) output + 1;
        }
      }
    }
    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    const uint16_t* i1 = (const uint16_t*) input[1];
    const uint16_t* i2 = (const uint16_t*) input[2];
    const uint16_t* i3 = (const uint16_t*) input[3];
    const uint16_t* i4 = (const uint16_t*) input[4];
    const uint16_t* i5 = (const uint16_t*) input[5];
    const uint16_t* i6 = (const uint16_t*) input[6];
    const uint16_t* i7 = (const uint16_t*) input[7];
    const uint16_t* i8 = (const uint16_t*) input[8];
    input = (const void**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = (const uint16_t*) zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = (const uint16_t*) zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = (const uint16_t*) zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = (const uint16_t*) zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = (const uint16_t*) zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = (const uint16_t*) zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = (const uint16_t*) zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = (const uint16_t*) zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    while (c >= 8) {
      const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;

      const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
      const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
      const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
      const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
      const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
      const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
      const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
      const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

      float16x8_t vout = vmulq_f16(vsum, vscale);
      vout = vmaxq_f16(vout, vmin);
      vout = vminq_f16(vout, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vout)); output = (uint16_t*) output + 8;

      c -= 8;
    }
    if (c != 0) {
      const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3));
      const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4));
      const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5));
      const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6));
      const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7));
      const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8));

      const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
      const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
      const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
      const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
      const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
      const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
      const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
      const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

      float16x8_t vout = vmulq_f16(vsum, vscale);
      vout = vmaxq_f16(vout, vmin);
      vout = vminq_f16(vout, vmax);

      float16x4_t vout_lo = vget_low_f16(vout);
      if (c & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vout_lo)); output = (uint16_t*) output + 4;
        vout_lo = vget_high_f16(vout);
      }
      if (c & 2) {
        vst1_lane_u32(output, vreinterpret_u32_f16(vout_lo), 0); output = (uint16_t*) output + 2;
        vout_lo = vext_f16(vout_lo, vout_lo, 2);
      }
      if (c & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vout_lo), 0); output = (uint16_t*) output + 1;
      }
    }
    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfp16arith_2x2(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const void* input,
    const void* zero,
    const void* weights,
    void* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(uint16_t);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(uint16_t);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(uint16_t);

  // Adjustment for padding processed below
  const uint16_t* i0 = (const uint16_t*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_height_stride);
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_height_stride);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_height_stride);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_height_stride);
  uint16_t* output0 = (uint16_t*) ((uintptr_t) output + output_height_stride * output_y_start);
  uint16_t* output1 = (uint16_t*) ((uintptr_t) output0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float16x4_t vmax = vreinterpret_f16_u16(vld1_dup_u16(&params->fp16arith.max));
  const float16x4_t vmin = vreinterpret_f16_u16(vld1_dup_u16(&params->fp16arith.min));

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      output1 = output0;
    }

    const uint16_t* w = weights;
    size_t c = output_channels;
    uint16_t* o0c0 = output0;
    uint16_t* o1c0 = output1;
    uint16_t* o0c1 = (uint16_t*) ((uintptr_t) o0c0 + output_channel_stride);
    uint16_t* o1c1 = (uint16_t*) ((uintptr_t) o1c0 + output_channel_stride);
    uint16_t* o0c2 = (uint16_t*) ((uintptr_t) o0c1 + output_channel_stride);
    uint16_t* o1c2 = (uint16_t*) ((uintptr_t) o1c1 + output_channel_stride);
    uint16_t* o0c3 = (uint16_t*) ((uintptr_t) o0c2 + output_channel_stride);
    uint16_t* o1c3 = (uint16_t*) ((uintptr_t) o1c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
        o1c1 = o1c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
        o1c2 = o1c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
        o1c3 = o1c2;
      }

      // viMx0 = ( iM0c2, iM0c1, iM0c0, --- )
      float16x4_t vi0x0 = vreinterpret_f16_u16(vmov_n_u16(0));
      float16x4_t vi1x0 = vreinterpret_f16_u16(vmov_n_u16(0));
      float16x4_t vi2x0 = vreinterpret_f16_u16(vmov_n_u16(0));
      float16x4_t vi3x0 = vreinterpret_f16_u16(vmov_n_u16(0));
      float16x4_t vi4x0 = vreinterpret_f16_u16(vmov_n_u16(0));

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        float16x4_t vo0x0 = vreinterpret_f16_u16(vld1_u16(w));
        float16x4_t vo1x0 = vo0x0;
        float16x4_t vo0x1 = vo0x0;
        float16x4_t vo1x1 = vo0x0;

        const float16x4_t vk00c0 = vreinterpret_f16_u16(vld1_u16(w + 4));

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const float16x4_t vi0x1 = vreinterpret_f16_u16(vld1_u16(i0)); i0 += 4;
        const float16x4_t vi1x1 = vreinterpret_f16_u16(vld1_u16(i1)); i1 += 4;
        const float16x4_t vi2x1 = vreinterpret_f16_u16(vld1_u16(i2)); i2 += 4;
        const float16x4_t vi3x1 = vreinterpret_f16_u16(vld1_u16(i3)); i3 += 4;
        const float16x4_t vi4x1 = vreinterpret_f16_u16(vld1_u16(i4)); i4 += 4;

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c0, vi2x0, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c0, vi0x1, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c0, vi2x1, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c0, vi2x0, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c0, vi0x1, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c0, vi2x1, 3);
#endif
        const float16x4_t vk10c0 = vreinterpret_f16_u16(vld1_u16(w + 8));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c0, vi3x0, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c0, vi1x1, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c0, vi3x1, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c0, vi3x0, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c0, vi1x1, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c0, vi3x1, 3);
#endif
        const float16x4_t vk20c0 = vreinterpret_f16_u16(vld1_u16(w + 12));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c0, vi4x0, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c0, vi2x1, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c0, vi4x1, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c0, vi4x0, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c0, vi2x1, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c0, vi4x1, 3);
#endif
        const float16x4_t vk00c1 = vreinterpret_f16_u16(vld1_u16(w + 16));

        // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
        const float16x4_t vi0x2 = vreinterpret_f16_u16(vld1_u16(i0)); i0 += 4;
        const float16x4_t vi1x2 = vreinterpret_f16_u16(vld1_u16(i1)); i1 += 4;
        const float16x4_t vi2x2 = vreinterpret_f16_u16(vld1_u16(i2)); i2 += 4;
        const float16x4_t vi3x2 = vreinterpret_f16_u16(vld1_u16(i3)); i3 += 4;
        const float16x4_t vi4x2 = vreinterpret_f16_u16(vld1_u16(i4)); i4 += 4;

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c1, vi2x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c1, vi2x2, 0);
#endif
        const float16x4_t vk10c1 = vreinterpret_f16_u16(vld1_u16(w + 20));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c1, vi3x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c1, vi3x2, 0);
#endif
        const float16x4_t vk20c1 = vreinterpret_f16_u16(vld1_u16(w + 24));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c1, vi4x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c1, vi4x2, 0);
#endif
        const float16x4_t vk00c2 = vreinterpret_f16_u16(vld1_u16(w + 28));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c2, vi2x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c2, vi2x2, 1);
#endif
        const float16x4_t vk10c2 = vreinterpret_f16_u16(vld1_u16(w + 32));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c2, vi3x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c2, vi3x2, 1);
#endif
        const float16x4_t vk20c2 = vreinterpret_f16_u16(vld1_u16(w + 36));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c2, vi4x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c2, vi4x2, 1);
#endif
        const float16x4_t vk01c0 = vreinterpret_f16_u16(vld1_u16(w + 40));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c0, vi2x1, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk01c0, vi0x2, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk01c0, vi2x2, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c0, vi2x1, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk01c0, vi0x2, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk01c0, vi2x2, 2);
#endif
        const float16x4_t vk11c0 = vreinterpret_f16_u16(vld1_u16(w + 44));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c0, vi3x1, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk11c0, vi1x2, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk11c0, vi3x2, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c0, vi3x1, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk11c0, vi1x2, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk11c0, vi3x2, 2);
#endif
        const float16x4_t vk21c0 = vreinterpret_f16_u16(vld1_u16(w + 48));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c0, vi4x1, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk21c0, vi2x2, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk21c0, vi4x2, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c0, vi4x1, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk21c0, vi2x2, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk21c0, vi4x2, 2);
#endif
        const float16x4_t vk01c1 = vreinterpret_f16_u16(vld1_u16(w + 52));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c1, vi2x1, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk01c1, vi0x2, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk01c1, vi2x2, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c1, vi2x1, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk01c1, vi0x2, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk01c1, vi2x2, 3);
#endif
        const float16x4_t vk11c1 = vreinterpret_f16_u16(vld1_u16(w + 56));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c1, vi3x1, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk11c1, vi1x2, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk11c1, vi3x2, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c1, vi3x1, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk11c1, vi1x2, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk11c1, vi3x2, 3);
#endif
        const float16x4_t vk21c1 = vreinterpret_f16_u16(vld1_u16(w + 60));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c1, vi4x1, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk21c1, vi2x2, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk21c1, vi4x2, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c1, vi4x1, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk21c1, vi2x2, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk21c1, vi4x2, 3);
#endif
        const float16x4_t vk01c2 = vreinterpret_f16_u16(vld1_u16(w + 64));

        // viMx3 = ( iM4c2, iM4c1, iM4c0, iM3c2 )
        const float16x4_t vi0x3 = vreinterpret_f16_u16(vld1_u16(i0)); i0 += 4;
        const float16x4_t vi1x3 = vreinterpret_f16_u16(vld1_u16(i1)); i1 += 4;
        const float16x4_t vi2x3 = vreinterpret_f16_u16(vld1_u16(i2)); i2 += 4;
        const float16x4_t vi3x3 = vreinterpret_f16_u16(vld1_u16(i3)); i3 += 4;
        const float16x4_t vi4x3 = vreinterpret_f16_u16(vld1_u16(i4)); i4 += 4;

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk01c2, vi2x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk01c2, vi2x3, 0);
#endif
        const float16x4_t vk11c2 = vreinterpret_f16_u16(vld1_u16(w + 68));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk11c2, vi3x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk11c2, vi3x3, 0);
#endif
        const float16x4_t vk21c2 = vreinterpret_f16_u16(vld1_u16(w + 72));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk21c2, vi4x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk21c2, vi4x3, 0);
#endif
        const float16x4_t vk02c0 = vreinterpret_f16_u16(vld1_u16(w + 76));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk02c0, vi0x1, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk02c0, vi2x1, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk02c0, vi0x3, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk02c0, vi2x3, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk02c0, vi0x1, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk02c0, vi2x1, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk02c0, vi0x3, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk02c0, vi2x3, 1);
#endif
        const float16x4_t vk12c0 = vreinterpret_f16_u16(vld1_u16(w + 80));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk12c0, vi1x1, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk12c0, vi3x1, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk12c0, vi1x3, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk12c0, vi3x3, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk12c0, vi1x1, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk12c0, vi3x1, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk12c0, vi1x3, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk12c0, vi3x3, 1);
#endif
        const float16x4_t vk22c0 = vreinterpret_f16_u16(vld1_u16(w + 84));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk22c0, vi2x1, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk22c0, vi4x1, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk22c0, vi2x3, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk22c0, vi4x3, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk22c0, vi2x1, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk22c0, vi4x1, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk22c0, vi2x3, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk22c0, vi4x3, 1);
#endif
        const float16x4_t vk02c1 = vreinterpret_f16_u16(vld1_u16(w + 88));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk02c1, vi0x2, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk02c1, vi2x2, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk02c1, vi0x3, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk02c1, vi2x3, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk02c1, vi0x2, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk02c1, vi2x2, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk02c1, vi0x3, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk02c1, vi2x3, 2);
#endif
        const float16x4_t vk12c1 = vreinterpret_f16_u16(vld1_u16(w + 92));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk12c1, vi1x2, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk12c1, vi3x2, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk12c1, vi1x3, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk12c1, vi3x3, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk12c1, vi1x2, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk12c1, vi3x2, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk12c1, vi1x3, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk12c1, vi3x3, 2);
#endif
        const float16x4_t vk22c1 = vreinterpret_f16_u16(vld1_u16(w + 96));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk22c1, vi2x2, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk22c1, vi4x2, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk22c1, vi2x3, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk22c1, vi4x3, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk22c1, vi2x2, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk22c1, vi4x2, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk22c1, vi2x3, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk22c1, vi4x3, 2);
#endif
        const float16x4_t vk02c2 = vreinterpret_f16_u16(vld1_u16(w + 100));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk02c2, vi0x2, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk02c2, vi2x2, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk02c2, vi0x3, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk02c2, vi2x3, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk02c2, vi0x2, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk02c2, vi2x2, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk02c2, vi0x3, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk02c2, vi2x3, 3);
#endif
        const float16x4_t vk12c2 = vreinterpret_f16_u16(vld1_u16(w + 104));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk12c2, vi1x2, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk12c2, vi3x2, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk12c2, vi1x3, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk12c2, vi3x3, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk12c2, vi1x2, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk12c2, vi3x2, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk12c2, vi1x3, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk12c2, vi3x3, 3);
#endif
        const float16x4_t vk22c2 = vreinterpret_f16_u16(vld1_u16(w + 108));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk22c2, vi2x2, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk22c2, vi4x2, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk22c2, vi2x3, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk22c2, vi4x3, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk22c2, vi2x2, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk22c2, vi4x2, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk22c2, vi2x3, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk22c2, vi4x3, 3);
#endif
        vi0x0 = vi0x3;
        vi1x0 = vi1x3;
        vi2x0 = vi2x3;
        vi3x0 = vi3x3;
        vi4x0 = vi4x3;

        vo0x0 = vmax_f16(vo0x0, vmin);
        vo1x0 = vmax_f16(vo1x0, vmin);
        vo0x1 = vmax_f16(vo0x1, vmin);
        vo1x1 = vmax_f16(vo1x1, vmin);

        vo0x0 = vmin_f16(vo0x0, vmax);
        vo1x0 = vmin_f16(vo1x0, vmax);
        vo0x1 = vmin_f16(vo0x1, vmax);
        vo1x1 = vmin_f16(vo1x1, vmax);

        const float16x4x2_t vo0c0123 = vzip_f16(vo0x0, vo0x1);
        const float16x4x2_t vo1c0123 = vzip_f16(vo1x0, vo1x1);

        // Always 2+ output width elements remaining
        vst1_lane_u32((void*) o1c0, vreinterpret_u32_f16(vo1c0123.val[0]), 0); o1c0 += 2;
        vst1_lane_u32((void*) o1c1, vreinterpret_u32_f16(vo1c0123.val[0]), 1); o1c1 += 2;
        vst1_lane_u32((void*) o1c2, vreinterpret_u32_f16(vo1c0123.val[1]), 0); o1c2 += 2;
        vst1_lane_u32((void*) o1c3, vreinterpret_u32_f16(vo1c0123.val[1]), 1); o1c3 += 2;

        vst1_lane_u32((void*) o0c0, vreinterpret_u32_f16(vo0c0123.val[0]), 0); o0c0 += 2;
        vst1_lane_u32((void*) o0c1, vreinterpret_u32_f16(vo0c0123.val[0]), 1); o0c1 += 2;
        vst1_lane_u32((void*) o0c2, vreinterpret_u32_f16(vo0c0123.val[1]), 0); o0c2 += 2;
        vst1_lane_u32((void*) o0c3, vreinterpret_u32_f16(vo0c0123.val[1]), 1); o0c3 += 2;
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw != 0) {
        float16x4_t vo0x0 = vreinterpret_f16_u16(vld1_u16(w));
        float16x4_t vo1x0 = vo0x0;
        float16x4_t vo0x1 = vo0x0;
        float16x4_t vo1x1 = vo0x0;

        const float16x4_t vk00c0 = vreinterpret_f16_u16(vld1_u16(w + 4));

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        float16x4_t vi0x1 = vreinterpret_f16_u16(vld1_u16(i0));
        float16x4_t vi1x1 = vreinterpret_f16_u16(vld1_u16(i1));
        float16x4_t vi2x1 = vreinterpret_f16_u16(vld1_u16(i2));
        float16x4_t vi3x1 = vreinterpret_f16_u16(vld1_u16(i3));
        float16x4_t vi4x1 = vreinterpret_f16_u16(vld1_u16(i4));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c0, vi2x0, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk00c0, vi0x1, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk00c0, vi2x1, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c0, vi2x0, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk00c0, vi0x1, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk00c0, vi2x1, 3);
        }
#endif
        const float16x4_t vk10c0 = vreinterpret_f16_u16(vld1_u16(w + 8));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c0, vi3x0, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk10c0, vi1x1, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk10c0, vi3x1, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c0, vi3x0, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk10c0, vi1x1, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk10c0, vi3x1, 3);
        }
#endif
        const float16x4_t vk20c0 = vreinterpret_f16_u16(vld1_u16(w + 12));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c0, vi4x0, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk20c0, vi2x1, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk20c0, vi4x1, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c0, vi4x0, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk20c0, vi2x1, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk20c0, vi4x1, 3);
        }
#endif
        const float16x4_t vk00c1 = vreinterpret_f16_u16(vld1_u16(w + 16));

        float16x4_t vi0x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi1x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi2x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi3x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi4x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        if (iw >= 2) {
          // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
          vi0x2 = vreinterpret_f16_u16(vld1_u16(i0 + 4));
          vi1x2 = vreinterpret_f16_u16(vld1_u16(i1 + 4));
          vi2x2 = vreinterpret_f16_u16(vld1_u16(i2 + 4));
          vi3x2 = vreinterpret_f16_u16(vld1_u16(i3 + 4));
          vi4x2 = vreinterpret_f16_u16(vld1_u16(i4 + 4));
        }

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c1, vi2x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c1, vi2x2, 0);
#endif
        const float16x4_t vk10c1 = vreinterpret_f16_u16(vld1_u16(w + 20));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c1, vi3x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c1, vi3x2, 0);
#endif
        const float16x4_t vk20c1 = vreinterpret_f16_u16(vld1_u16(w + 24));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c1, vi4x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c1, vi4x2, 0);
#endif
        const float16x4_t vk00c2 = vreinterpret_f16_u16(vld1_u16(w + 28));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c2, vi2x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c2, vi2x2, 1);
#endif
        const float16x4_t vk10c2 = vreinterpret_f16_u16(vld1_u16(w + 32));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c2, vi3x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c2, vi3x2, 1);
#endif
        const float16x4_t vk20c2 = vreinterpret_f16_u16(vld1_u16(w + 36));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c2, vi4x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c2, vi4x2, 1);
#endif
        const float16x4_t vk01c0 = vreinterpret_f16_u16(vld1_u16(w + 40));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c0, vi2x1, 0);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk01c0, vi0x2, 2);
          vo1x1 = vfma_lane_f16(vo1x1, vk01c0, vi2x2, 2);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c0, vi2x1, 0);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk01c0, vi0x2, 2);
          vo1x1 = vmla_lane_f16(vo1x1, vk01c0, vi2x2, 2);
        }
#endif
        const float16x4_t vk11c0 = vreinterpret_f16_u16(vld1_u16(w + 44));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c0, vi3x1, 0);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk11c0, vi1x2, 2);
          vo1x1 = vfma_lane_f16(vo1x1, vk11c0, vi3x2, 2);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c0, vi3x1, 0);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk11c0, vi1x2, 2);
          vo1x1 = vmla_lane_f16(vo1x1, vk11c0, vi3x2, 2);
        }
#endif
        const float16x4_t vk21c0 = vreinterpret_f16_u16(vld1_u16(w + 48));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c0, vi4x1, 0);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk21c0, vi2x2, 2);
          vo1x1 = vfma_lane_f16(vo1x1, vk21c0, vi4x2, 2);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c0, vi4x1, 0);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk21c0, vi2x2, 2);
          vo1x1 = vmla_lane_f16(vo1x1, vk21c0, vi4x2, 2);
        }
#endif
        const float16x4_t vk01c1 = vreinterpret_f16_u16(vld1_u16(w + 52));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c1, vi2x1, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk01c1, vi0x2, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk01c1, vi2x2, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c1, vi2x1, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk01c1, vi0x2, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk01c1, vi2x2, 3);
        }
#endif
        const float16x4_t vk11c1 = vreinterpret_f16_u16(vld1_u16(w + 56));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c1, vi3x1, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk11c1, vi1x2, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk11c1, vi3x2, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c1, vi3x1, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk11c1, vi1x2, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk11c1, vi3x2, 3);
        }
#endif
        const float16x4_t vk21c1 = vreinterpret_f16_u16(vld1_u16(w + 60));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c1, vi4x1, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk21c1, vi2x2, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk21c1, vi4x2, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c1, vi4x1, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk21c1, vi2x2, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk21c1, vi4x2, 3);
        }
#endif
        const float16x4_t vk01c2 = vreinterpret_f16_u16(vld1_u16(w + 64));

        float16x4_t vi0x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi1x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi2x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi3x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi4x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        if (iw > 2) {
          // viMx3 = ( 0.0, 0.0, 0.0, iM3c2 )
          vi0x3 = vreinterpret_f16_u16(vld1_lane_u16(i0 + 8, vreinterpret_u16_f16(vi0x3), 0));
          vi1x3 = vreinterpret_f16_u16(vld1_lane_u16(i1 + 8, vreinterpret_u16_f16(vi1x3), 0));
          vi2x3 = vreinterpret_f16_u16(vld1_lane_u16(i2 + 8, vreinterpret_u16_f16(vi2x3), 0));
          vi3x3 = vreinterpret_f16_u16(vld1_lane_u16(i3 + 8, vreinterpret_u16_f16(vi3x3), 0));
          vi4x3 = vreinterpret_f16_u16(vld1_lane_u16(i4 + 8, vreinterpret_u16_f16(vi4x3), 0));
        }

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk01c2, vi2x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk01c2, vi2x3, 0);
#endif
        const float16x4_t vk11c2 = vreinterpret_f16_u16(vld1_u16(w + 68));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk11c2, vi3x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk11c2, vi3x3, 0);
#endif
        const float16x4_t vk21c2 = vreinterpret_f16_u16(vld1_u16(w + 72));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk21c2, vi4x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk21c2, vi4x3, 0);
#endif
        if (iw >= 2) {
          const float16x4_t vk02c0 = vreinterpret_f16_u16(vld1_u16(w + 76));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk02c0, vi0x1, 3);
          vo1x0 = vfma_lane_f16(vo1x0, vk02c0, vi2x1, 3);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk02c0, vi0x1, 3);
          vo1x0 = vmla_lane_f16(vo1x0, vk02c0, vi2x1, 3);
#endif
          const float16x4_t vk12c0 = vreinterpret_f16_u16(vld1_u16(w + 80));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk12c0, vi1x1, 3);
          vo1x0 = vfma_lane_f16(vo1x0, vk12c0, vi3x1, 3);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk12c0, vi1x1, 3);
          vo1x0 = vmla_lane_f16(vo1x0, vk12c0, vi3x1, 3);
#endif
          const float16x4_t vk22c0 = vreinterpret_f16_u16(vld1_u16(w + 84));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk22c0, vi2x1, 3);
          vo1x0 = vfma_lane_f16(vo1x0, vk22c0, vi4x1, 3);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk22c0, vi2x1, 3);
          vo1x0 = vmla_lane_f16(vo1x0, vk22c0, vi4x1, 3);
#endif
          const float16x4_t vk02c1 = vreinterpret_f16_u16(vld1_u16(w + 88));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk02c1, vi0x2, 0);
          vo1x0 = vfma_lane_f16(vo1x0, vk02c1, vi2x2, 0);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk02c1, vi0x2, 0);
          vo1x0 = vmla_lane_f16(vo1x0, vk02c1, vi2x2, 0);
#endif
          const float16x4_t vk12c1 = vreinterpret_f16_u16(vld1_u16(w + 92));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk12c1, vi1x2, 0);
          vo1x0 = vfma_lane_f16(vo1x0, vk12c1, vi3x2, 0);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk12c1, vi1x2, 0);
          vo1x0 = vmla_lane_f16(vo1x0, vk12c1, vi3x2, 0);
#endif
          const float16x4_t vk22c1 = vreinterpret_f16_u16(vld1_u16(w + 96));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk22c1, vi2x2, 0);
          vo1x0 = vfma_lane_f16(vo1x0, vk22c1, vi4x2, 0);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk22c1, vi2x2, 0);
          vo1x0 = vmla_lane_f16(vo1x0, vk22c1, vi4x2, 0);
#endif
          const float16x4_t vk02c2 = vreinterpret_f16_u16(vld1_u16(w + 100));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk02c2, vi0x2, 1);
          vo1x0 = vfma_lane_f16(vo1x0, vk02c2, vi2x2, 1);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk02c2, vi0x2, 1);
          vo1x0 = vmla_lane_f16(vo1x0, vk02c2, vi2x2, 1);
#endif
          const float16x4_t vk12c2 = vreinterpret_f16_u16(vld1_u16(w + 104));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk12c2, vi1x2, 1);
          vo1x0 = vfma_lane_f16(vo1x0, vk12c2, vi3x2, 1);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk12c2, vi1x2, 1);
          vo1x0 = vmla_lane_f16(vo1x0, vk12c2, vi3x2, 1);
#endif
          const float16x4_t vk22c2 = vreinterpret_f16_u16(vld1_u16(w + 108));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk22c2, vi2x2, 1);
          vo1x0 = vfma_lane_f16(vo1x0, vk22c2, vi4x2, 1);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk22c2, vi2x2, 1);
          vo1x0 = vmla_lane_f16(vo1x0, vk22c2, vi4x2, 1);
#endif
        }

        vo0x0 = vmax_f16(vo0x0, vmin);
        vo1x0 = vmax_f16(vo1x0, vmin);
        vo0x1 = vmax_f16(vo0x1, vmin);
        vo1x1 = vmax_f16(vo1x1, vmin);

        vo0x0 = vmin_f16(vo0x0, vmax);
        vo1x0 = vmin_f16(vo1x0, vmax);
        vo0x1 = vmin_f16(vo0x1, vmax);
        vo1x1 = vmin_f16(vo1x1, vmax);

        if (iw == 3) {
          // Exactly 2 output width elements remaining
          const float16x4x2_t vo0c0123 = vzip_f16(vo0x0, vo0x1);
          const float16x4x2_t vo1c0123 = vzip_f16(vo1x0, vo1x1);

          // Always 2+ output width elements remaining
          vst1_lane_u32((void*) o1c0, vreinterpret_u32_f16(vo1c0123.val[0]), 0); o1c0 += 2;
          vst1_lane_u32((void*) o1c1, vreinterpret_u32_f16(vo1c0123.val[0]), 1); o1c1 += 2;
          vst1_lane_u32((void*) o1c2, vreinterpret_u32_f16(vo1c0123.val[1]), 0); o1c2 += 2;
          vst1_lane_u32((void*) o1c3, vreinterpret_u32_f16(vo1c0123.val[1]), 1); o1c3 += 2;

          vst1_lane_u32((void*) o0c0, vreinterpret_u32_f16(vo0c0123.val[0]), 0); o0c0 += 2;
          vst1_lane_u32((void*) o0c1, vreinterpret_u32_f16(vo0c0123.val[0]), 1); o0c1 += 2;
          vst1_lane_u32((void*) o0c2, vreinterpret_u32_f16(vo0c0123.val[1]), 0); o0c2 += 2;
          vst1_lane_u32((void*) o0c3, vreinterpret_u32_f16(vo0c0123.val[1]), 1); o0c3 += 2;
        } else {
          // Exactly 1 output width element remaining

          vst1_lane_u16(o1c0, vreinterpret_u16_f16(vo1x0), 0); o1c0 += 1;
          vst1_lane_u16(o1c1, vreinterpret_u16_f16(vo1x0), 1); o1c1 += 1;
          vst1_lane_u16(o1c2, vreinterpret_u16_f16(vo1x0), 2); o1c2 += 1;
          vst1_lane_u16(o1c3, vreinterpret_u16_f16(vo1x0), 3); o1c3 += 1;

          vst1_lane_u16(o0c0, vreinterpret_u16_f16(vo0x0), 0); o0c0 += 1;
          vst1_lane_u16(o0c1, vreinterpret_u16_f16(vo0x0), 1); o0c1 += 1;
          vst1_lane_u16(o0c2, vreinterpret_u16_f16(vo0x0), 2); o0c2 += 1;
          vst1_lane_u16(o0c3, vreinterpret_u16_f16(vo0x0), 3); o0c3 += 1;
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (uint16_t*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (uint16_t*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (uint16_t*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (uint16_t*) ((uintptr_t) o0c3 + output_channel_increment);
      o1c0 = (uint16_t*) ((uintptr_t) o1c0 + output_channel_increment);
      o1c1 = (uint16_t*) ((uintptr_t) o1c1 + output_channel_increment);
      o1c2 = (uint16_t*) ((uintptr_t) o1c2 + output_channel_increment);
      o1c3 = (uint16_t*) ((uintptr_t) o1c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const uint16_t*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 - input_width_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 - input_width_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 - input_width_increment);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (uint16_t*) ((uintptr_t) output1 + output_height_stride);
    output1 = (uint16_t*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const uint16_t*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_height_stride);
  }
}

void xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith_acc2(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output_ptr,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  uint16_t* output = (uint16_t*) output_ptr;
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = (const uint16_t*) input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = (const uint16_t*) input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = (const uint16_t*) input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = (const uint16_t*) input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = (const uint16_t*) input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint16_t* i9 = (const uint16_t*) input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != (const uint16_t*) zero) {
      i9 = (const uint16_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint16_t* i10 = (const uint16_t*) input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != (const uint16_t*) zero) {
      i10 = (const uint16_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint16_t* i11 = (const uint16_t*) input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != (const uint16_t*) zero) {
      i11 = (const uint16_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint16_t* i12 = (const uint16_t*) input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != (const uint16_t*) zero) {
      i12 = (const uint16_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint16_t* i13 = (const uint16_t*) input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != (const uint16_t*) zero) {
      i13 = (const uint16_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint16_t* i14 = (const uint16_t*) input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != (const uint16_t*) zero) {
      i14 = (const uint16_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint16_t* i15 = (const uint16_t*) input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != (const uint16_t*) zero) {
      i15 = (const uint16_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint16_t* i16 = (const uint16_t*) input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != (const uint16_t*) zero) {
      i16 = (const uint16_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint16_t* i17 = (const uint16_t*) input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != (const uint16_t*) zero) {
      i17 = (const uint16_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint16_t* i18 = (const uint16_t*) input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != (const uint16_t*) zero) {
      i18 = (const uint16_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint16_t* i19 = (const uint16_t*) input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != (const uint16_t*) zero) {
      i19 = (const uint16_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint16_t* i20 = (const uint16_t*) input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != (const uint16_t*) zero) {
      i20 = (const uint16_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint16_t* i21 = (const uint16_t*) input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != (const uint16_t*) zero) {
      i21 = (const uint16_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint16_t* i22 = (const uint16_t*) input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != (const uint16_t*) zero) {
      i22 = (const uint16_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint16_t* i23 = (const uint16_t*) input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != (const uint16_t*) zero) {
      i23 = (const uint16_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint16_t* i24 = (const uint16_t*) input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != (const uint16_t*) zero) {
      i24 = (const uint16_t*) ((uintptr_t) i24 + input_offset);
    }

    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*) weights;
    for (; c >= 8; c -= 8) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      float16x8_t vacc01234567p1 = vmulq_f16(vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);

      const float16x8_t vi9x01234567 = vreinterpretq_f16_u16(vld1q_u16(i9)); i9 += 8;
      const float16x8_t vk9x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi9x01234567, vk9x01234567);

      const float16x8_t vi10x01234567 = vreinterpretq_f16_u16(vld1q_u16(i10)); i10 += 8;
      const float16x8_t vk10x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi10x01234567, vk10x01234567);

      const float16x8_t vi11x01234567 = vreinterpretq_f16_u16(vld1q_u16(i11)); i11 += 8;
      const float16x8_t vk11x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi11x01234567, vk11x01234567);

      const float16x8_t vi12x01234567 = vreinterpretq_f16_u16(vld1q_u16(i12)); i12 += 8;
      const float16x8_t vk12x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi12x01234567, vk12x01234567);

      const float16x8_t vi13x01234567 = vreinterpretq_f16_u16(vld1q_u16(i13)); i13 += 8;
      const float16x8_t vk13x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi13x01234567, vk13x01234567);

      const float16x8_t vi14x01234567 = vreinterpretq_f16_u16(vld1q_u16(i14)); i14 += 8;
      const float16x8_t vk14x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi14x01234567, vk14x01234567);

      const float16x8_t vi15x01234567 = vreinterpretq_f16_u16(vld1q_u16(i15)); i15 += 8;
      const float16x8_t vk15x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi15x01234567, vk15x01234567);

      const float16x8_t vi16x01234567 = vreinterpretq_f16_u16(vld1q_u16(i16)); i16 += 8;
      const float16x8_t vk16x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi16x01234567, vk16x01234567);

      const float16x8_t vi17x01234567 = vreinterpretq_f16_u16(vld1q_u16(i17)); i17 += 8;
      const float16x8_t vk17x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi17x01234567, vk17x01234567);

      const float16x8_t vi18x01234567 = vreinterpretq_f16_u16(vld1q_u16(i18)); i18 += 8;
      const float16x8_t vk18x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi18x01234567, vk18x01234567);

      const float16x8_t vi19x01234567 = vreinterpretq_f16_u16(vld1q_u16(i19)); i19 += 8;
      const float16x8_t vk19x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi19x01234567, vk19x01234567);

      const float16x8_t vi20x01234567 = vreinterpretq_f16_u16(vld1q_u16(i20)); i20 += 8;
      const float16x8_t vk20x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi20x01234567, vk20x01234567);

      const float16x8_t vi21x01234567 = vreinterpretq_f16_u16(vld1q_u16(i21)); i21 += 8;
      const float16x8_t vk21x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi21x01234567, vk21x01234567);

      const float16x8_t vi22x01234567 = vreinterpretq_f16_u16(vld1q_u16(i22)); i22 += 8;
      const float16x8_t vk22x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi22x01234567, vk22x01234567);

      const float16x8_t vi23x01234567 = vreinterpretq_f16_u16(vld1q_u16(i23)); i23 += 8;
      const float16x8_t vk23x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi23x01234567, vk23x01234567);

      const float16x8_t vi24x01234567 = vreinterpretq_f16_u16(vld1q_u16(i24)); i24 += 8;
      const float16x8_t vk24x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi24x01234567, vk24x01234567);

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = vaddq_f16(vacc01234567p0, vacc01234567p1);

      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      float16x8_t vacc01234567p1 = vmulq_f16(vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3));
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4));
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5));
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6));
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7));
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8));
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);

      const float16x8_t vi9x01234567 = vreinterpretq_f16_u16(vld1q_u16(i9));
      const float16x8_t vk9x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi9x01234567, vk9x01234567);

      const float16x8_t vi10x01234567 = vreinterpretq_f16_u16(vld1q_u16(i10));
      const float16x8_t vk10x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi10x01234567, vk10x01234567);

      const float16x8_t vi11x01234567 = vreinterpretq_f16_u16(vld1q_u16(i11));
      const float16x8_t vk11x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi11x01234567, vk11x01234567);

      const float16x8_t vi12x01234567 = vreinterpretq_f16_u16(vld1q_u16(i12));
      const float16x8_t vk12x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi12x01234567, vk12x01234567);

      const float16x8_t vi13x01234567 = vreinterpretq_f16_u16(vld1q_u16(i13));
      const float16x8_t vk13x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi13x01234567, vk13x01234567);

      const float16x8_t vi14x01234567 = vreinterpretq_f16_u16(vld1q_u16(i14));
      const float16x8_t vk14x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi14x01234567, vk14x01234567);

      const float16x8_t vi15x01234567 = vreinterpretq_f16_u16(vld1q_u16(i15));
      const float16x8_t vk15x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi15x01234567, vk15x01234567);

      const float16x8_t vi16x01234567 = vreinterpretq_f16_u16(vld1q_u16(i16));
      const float16x8_t vk16x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi16x01234567, vk16x01234567);

      const float16x8_t vi17x01234567 = vreinterpretq_f16_u16(vld1q_u16(i17));
      const float16x8_t vk17x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi17x01234567, vk17x01234567);

      const float16x8_t vi18x01234567 = vreinterpretq_f16_u16(vld1q_u16(i18));
      const float16x8_t vk18x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi18x01234567, vk18x01234567);

      const float16x8_t vi19x01234567 = vreinterpretq_f16_u16(vld1q_u16(i19));
      const float16x8_t vk19x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi19x01234567, vk19x01234567);

      const float16x8_t vi20x01234567 = vreinterpretq_f16_u16(vld1q_u16(i20));
      const float16x8_t vk20x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi20x01234567, vk20x01234567);

      const float16x8_t vi21x01234567 = vreinterpretq_f16_u16(vld1q_u16(i21));
      const float16x8_t vk21x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi21x01234567, vk21x01234567);

      const float16x8_t vi22x01234567 = vreinterpretq_f16_u16(vld1q_u16(i22));
      const float16x8_t vk22x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi22x01234567, vk22x01234567);

      const float16x8_t vi23x01234567 = vreinterpretq_f16_u16(vld1q_u16(i23));
      const float16x8_t vk23x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi23x01234567, vk23x01234567);

      const float16x8_t vi24x01234567 = vreinterpretq_f16_u16(vld1q_u16(i24));
      const float16x8_t vk24x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi24x01234567, vk24x01234567);

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = vaddq_f16(vacc01234567p0, vacc01234567p1);

      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (c & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output += 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (c & 2) {
        vst1_lane_u32((void*) output, vreinterpret_u32_f16(vacc0123), 0); output += 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (c & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output += 1;
      }
    }

    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_3p16c__neonfp16arith(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output_ptr,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  uint16_t* output = (uint16_t*) output_ptr;
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }

    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*) weights;
    for (; c >= 16; c -= 16) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      float16x8_t vacc89ABCDEFp0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi0x89ABCDEF, vk0x89ABCDEF);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi1x89ABCDEF, vk1x89ABCDEF);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi2x89ABCDEF, vk2x89ABCDEF);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      float16x8_t vacc89ABCDEF = vmaxq_f16(vacc89ABCDEFp0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);
      vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
      vst1q_u16(output, vreinterpretq_u16_f16(vacc89ABCDEF)); output += 8;
    }
    for (; c >= 8; c -= 8) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 8));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 24));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 40));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w));


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 16));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 32));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 48));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (c & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output += 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (c & 2) {
        vst1_lane_u32((void*) output, vreinterpret_u32_f16(vacc0123), 0); output += 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (c & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output += 1;
      }
    }

    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output_ptr,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  uint16_t* output = (uint16_t*) output_ptr;
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }

    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*) weights;
    for (; c >= 16; c -= 16) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      float16x8_t vacc89ABCDEFp0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi0x89ABCDEF, vk0x89ABCDEF);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi1x89ABCDEF, vk1x89ABCDEF);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi2x89ABCDEF, vk2x89ABCDEF);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi3x89ABCDEF, vk3x89ABCDEF);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      float16x8_t vacc89ABCDEF = vmaxq_f16(vacc89ABCDEFp0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);
      vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
      vst1q_u16(output, vreinterpretq_u16_f16(vacc89ABCDEF)); output += 8;
    }
    for (; c >= 8; c -= 8) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 8));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 24));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 40));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 56));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w));


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 16));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 32));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 48));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3));
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 64));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (c & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output += 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (c & 2) {
        vst1_lane_u32((void*) output, vreinterpret_u32_f16(vacc0123), 0); output += 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (c & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output += 1;
      }
    }

    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_9p16c__neonfp16arith(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output_ptr,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  uint16_t* output = (uint16_t*) output_ptr;
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = (const uint16_t*) input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = (const uint16_t*) input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = (const uint16_t*) input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = (const uint16_t*) input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = (const uint16_t*) input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }

    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*) weights;
    for (; c >= 16; c -= 16) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      float16x8_t vacc89ABCDEFp0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi0x89ABCDEF, vk0x89ABCDEF);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi1x89ABCDEF, vk1x89ABCDEF);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi2x89ABCDEF, vk2x89ABCDEF);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi3x89ABCDEF, vk3x89ABCDEF);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vi4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi4x89ABCDEF, vk4x89ABCDEF);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vi5x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk5x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi5x01234567, vk5x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi5x89ABCDEF, vk5x89ABCDEF);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vi6x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk6x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi6x89ABCDEF, vk6x89ABCDEF);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vi7x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk7x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi7x01234567, vk7x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi7x89ABCDEF, vk7x89ABCDEF);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;
      const float16x8_t vi8x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk8x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi8x89ABCDEF, vk8x89ABCDEF);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      float16x8_t vacc89ABCDEF = vmaxq_f16(vacc89ABCDEFp0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);
      vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
      vst1q_u16(output, vreinterpretq_u16_f16(vacc89ABCDEF)); output += 8;
    }
    for (; c >= 8; c -= 8) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 8));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 24));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 40));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 56));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 72));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 88));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 104));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 120));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 136));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w));


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 16));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 32));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 48));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3));
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 64));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4));
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 80));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5));
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 96));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6));
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 112));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7));
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 128));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8));
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 144));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (c & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output += 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (c & 2) {
        vst1_lane_u32((void*) output, vreinterpret_u32_f16(vacc0123), 0); output += 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (c & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output += 1;
      }
    }

    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_9p8c__neonfp16arith(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output_ptr,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  uint16_t* output = (uint16_t*) output_ptr;
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = (const uint16_t*) input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = (const uint16_t*) input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = (const uint16_t*) input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = (const uint16_t*) input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = (const uint16_t*) input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }

    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*) weights;
    for (; c >= 8; c -= 8) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3));
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4));
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5));
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6));
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7));
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8));
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (c & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output += 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (c & 2) {
        vst1_lane_u32((void*) output, vreinterpret_u32_f16(vacc0123), 0); output += 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (c & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output += 1;
      }
    }

    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_2x8(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(uint16_t) == 0);
  assert(padding_top == 1);

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->neonfp16arith_stride1.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->neonfp16arith_stride1.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif
  const uint16x8_t vmask = vld1q_u16(params->neonfp16arith_stride1.mask);

  const uint16_t* w = (const uint16_t*) weights;
  const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w));
  const float16x4_t vw89 = vreinterpret_f16_u32(vld1_dup_u32((const void*) (w + 8)));

  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(uint16_t));

  const uint16_t* i0 = zero;
  const uint16_t* i1 = input;
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);

  uint16_t* o0 = output;
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
    }

    float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));

    float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
    float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;

    size_t w = input_width;
    for (; w > 8 * sizeof(uint16_t); w -= 8 * sizeof(uint16_t)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x89ABCDEF, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x89ABCDEF, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x89ABCDEF, vw01234567, 2);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x89ABCDEF, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x89ABCDEF, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x89ABCDEF, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x89ABCDEF, vw01234567, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x89ABCDEF, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi2x89ABCDEF, vw89, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x89ABCDEF, vw89, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi3x89ABCDEF, vw89, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x89ABCDEF, vw89, 0);
      #endif
      // Left column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x789ABCDE, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x789ABCDE, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x789ABCDE, vw01234567, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x789ABCDE, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x789ABCDE, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x789ABCDE, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x789ABCDE, vw01234567, 4);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x789ABCDE, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x789ABCDE, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3x789ABCDE, vw01234567, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      vi0x01234567 = vi0x89ABCDEF;
      vi1x01234567 = vi1x89ABCDEF;
      vi2x01234567 = vi2x89ABCDEF;
      vi3x01234567 = vi3x89ABCDEF;

      // Right column
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x9ABCDEFG, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x9ABCDEFG, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x9ABCDEFG, vw01234567, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x9ABCDEFG, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x9ABCDEFG, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x9ABCDEFG, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x9ABCDEFG, vw01234567, 6);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x9ABCDEFG, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi2x9ABCDEFG, vw89, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x9ABCDEFG, vw89, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi3x9ABCDEFG, vw89, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x9ABCDEFG, vw89, 1);
      #endif
      vi0x89ABCDEF = vi0xGHIJKLMN;
      vi1x89ABCDEF = vi1xGHIJKLMN;
      vi2x89ABCDEF = vi2xGHIJKLMN;
      vi3x89ABCDEF = vi3xGHIJKLMN;


      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);

      vst1q_u16(o1, vreinterpretq_u16_f16(vo1)); o1 += 8;
      vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
    }

    // Always process the last block of 1..8 pixels.
    assert(w >= 1 * sizeof(uint16_t));
    assert(w <= 8 * sizeof(uint16_t));
    {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      vi0x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0x89ABCDEF)));
      vi1x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1x89ABCDEF)));
      vi2x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2x89ABCDEF)));
      vi3x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi3x89ABCDEF)));

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x89ABCDEF, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x89ABCDEF, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x89ABCDEF, vw01234567, 2);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x89ABCDEF, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x89ABCDEF, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x89ABCDEF, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x89ABCDEF, vw01234567, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x89ABCDEF, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi2x89ABCDEF, vw89, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x89ABCDEF, vw89, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi3x89ABCDEF, vw89, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x89ABCDEF, vw89, 0);
      #endif
      // Left column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x789ABCDE, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x789ABCDE, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x789ABCDE, vw01234567, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x789ABCDE, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x789ABCDE, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x789ABCDE, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x789ABCDE, vw01234567, 4);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x789ABCDE, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x789ABCDE, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3x789ABCDE, vw01234567, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      // Right column
      const float16x8_t vzero = vreinterpretq_f16_u16(vmovq_n_u16(0));
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vzero, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vzero, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vzero, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vzero, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x9ABCDEFG, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x9ABCDEFG, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x9ABCDEFG, vw01234567, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x9ABCDEFG, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x9ABCDEFG, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x9ABCDEFG, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x9ABCDEFG, vw01234567, 6);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x9ABCDEFG, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi2x9ABCDEFG, vw89, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x9ABCDEFG, vw89, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi3x9ABCDEFG, vw89, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x9ABCDEFG, vw89, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);

      if XNN_LIKELY(w == 8 * sizeof(uint16_t)) {
        vst1q_u16(o1, vreinterpretq_u16_f16(vo1)); o1 += 8;
        vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
      } else {
        float16x4_t vo1_lo = vget_low_f16(vo1);
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w & (4 * sizeof(uint16_t))) {
         vst1_u16(o1, vreinterpret_u16_f16(vo1_lo)); o1 += 4;
         vst1_u16(o0, vreinterpret_u16_f16(vo0_lo)); o0 += 4;

          vo1_lo = vget_high_f16(vo1);
          vo0_lo = vget_high_f16(vo0);
        }
        if (w & (2 * sizeof(uint16_t))) {
          vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vo1_lo), 0); o1 += 2;
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
          vo1_lo = vext_f16(vo1_lo, vo1_lo, 2);
        }
        if (w & (1 * sizeof(uint16_t))) {
          vst1_lane_u16(o1, vreinterpret_u16_f16(vo1_lo), 0); o1 += 1;
          vst1_lane_u16(o0, vreinterpret_u16_f16(vo0_lo), 0); o0 += 1;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i2 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i3 - input_decrement);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);

    o0 = o1;
    o1 = (uint16_t*) ((uintptr_t) o0 + input_width);

    output_height = doz(output_height, 2);
  } while (output_height != 0);
}

void xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x8(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(uint16_t) == 0);
  assert(padding_top <= 1);

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->neonfp16arith_stride2.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->neonfp16arith_stride2.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif
  const uint16x8_t vmask_even = vld1q_u16(params->neonfp16arith_stride2.mask_even);
  const uint16x8_t vmask_odd  = vld1q_u16(params->neonfp16arith_stride2.mask_odd);

  const uint16_t* w = (const uint16_t*) weights;
  const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w));
  const float16x4_t vw89 = vreinterpret_f16_u32(vld1_dup_u32((const void*) (w + 8)));

  const size_t input_decrement = round_down_po2(input_width, 8 /* SIMD output width */ * 2 /* subsampling */ * sizeof(uint16_t));

  const uint16_t* i0 = (const uint16_t*) ((uintptr_t) input - ((-padding_top) & input_width));
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);

  uint16_t* o0 = output;

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }

    float16x8_t vi0x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi1x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi2x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));

    size_t w = input_width;
    for (; w >= 16 * sizeof(uint16_t); w -= 16 * sizeof(uint16_t)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const uint16x8x2_t vi0xGIKMOQSUHJLNPRTV = vld2q_u16(i0); i0 += 16;
      const uint16x8x2_t vi1xGIKMOQSUHJLNPRTV = vld2q_u16(i1); i1 += 16;
      const uint16x8x2_t vi2xGIKMOQSUHJLNPRTV = vld2q_u16(i2); i2 += 16;

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vw89, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vw89, 0);
      #endif
      // Left column
      const float16x8_t vi0xFHJLNPRT = vextq_f16(vi0x13579BDF, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi0x13579BDF = vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi1xFHJLNPRT = vextq_f16(vi1x13579BDF, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi1x13579BDF = vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi2xFHJLNPRT = vextq_f16(vi2x13579BDF, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi2x13579BDF = vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xFHJLNPRT, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xFHJLNPRT, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xFHJLNPRT, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xFHJLNPRT, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xFHJLNPRT, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      // Right column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), vw89, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), vw89, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
    }

    // Last block has 0-15 pixels to process.
    assert(w < 16 * sizeof(uint16_t));
    if XNN_LIKELY(w != 0) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const uint16x8x2_t vi0xGIKMOQSUHJLNPRTV = vld2q_u16(i0);
      const uint16x8x2_t vi1xGIKMOQSUHJLNPRTV = vld2q_u16(i1);
      const uint16x8x2_t vi2xGIKMOQSUHJLNPRTV = vld2q_u16(i2);

      const float16x8_t vi0xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi0xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi0xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd,  vi0xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi1xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi1xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi1xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd,  vi1xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi2xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi2xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi2xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd,  vi2xGIKMOQSUHJLNPRTV.val[1]));

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xGIKMOQSU, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xGIKMOQSU, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xGIKMOQSU, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xGIKMOQSU, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi2xGIKMOQSU, vw89, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xGIKMOQSU, vw89, 0);
      #endif
      // Left column
      const float16x8_t vi0xFHJLNPRT = vextq_f16(vi0x13579BDF, vi0xHJLNPRTV, 7);
      const float16x8_t vi1xFHJLNPRT = vextq_f16(vi1x13579BDF, vi1xHJLNPRTV, 7);
      const float16x8_t vi2xFHJLNPRT = vextq_f16(vi2x13579BDF, vi2xHJLNPRTV, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xFHJLNPRT, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xFHJLNPRT, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xFHJLNPRT, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xFHJLNPRT, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xFHJLNPRT, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      // Right column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xHJLNPRTV, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xHJLNPRTV, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xHJLNPRTV, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xHJLNPRTV, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi2xHJLNPRTV, vw89, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xHJLNPRTV, vw89, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      w += 1 * sizeof(uint16_t);

      if XNN_LIKELY(w == 16 * sizeof(uint16_t)) {
        vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
      } else {
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w & (8 * sizeof(uint16_t))) {
         vst1_u16(o0, vreinterpret_u16_f16(vo0_lo)); o0 += 4;

          vo0_lo = vget_high_f16(vo0);
        }
        if (w & (4 * sizeof(uint16_t))) {
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
        }
        if (w & (2 * sizeof(uint16_t))) {
          vst1_lane_u16(o0, vreinterpret_u16_f16(vo0_lo), 0); o0 += 1;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i2 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i0 + input_width);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}

void xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x8(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(uint16_t) == 0);
  assert(padding_top == 2);

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->neonfp16arith_stride1.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->neonfp16arith_stride1.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif
  const uint16x8_t vmask = vld1q_u16(params->neonfp16arith_stride1.mask);

  const uint16_t* w = (const uint16_t*) weights;
  const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w));
  const float16x8_t vw89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w + 8));
  const float16x8_t vwGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w + 16));
  const float16x4_t vwOP = vreinterpret_f16_u32(vld1_dup_u32((const void*) (w + 24)));

  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(uint16_t));

  const uint16_t* i0 = zero;
  const uint16_t* i1 = zero;
  const uint16_t* i2 = input;
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);

  uint16_t* o0 = output;

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
    }

    float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));

    float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
    float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    float16x8_t vi4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;

    size_t w = input_width;
    for (; w > 16 * sizeof(uint16_t); w -= 8 * sizeof(uint16_t)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vi4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x89ABCDEF, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x789ABCDE, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x789ABCDE, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x789ABCDE, vw89ABCDEF, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Left by 2 column
      const float16x8_t vi0x6789ABCD = vextq_f16(vi0x01234567, vi0x89ABCDEF, 6);
      vi0x01234567 = vi0x89ABCDEF;
      const float16x8_t vi1x6789ABCD = vextq_f16(vi1x01234567, vi1x89ABCDEF, 6);
      vi1x01234567 = vi1x89ABCDEF;
      const float16x8_t vi2x6789ABCD = vextq_f16(vi2x01234567, vi2x89ABCDEF, 6);
      vi2x01234567 = vi2x89ABCDEF;
      const float16x8_t vi3x6789ABCD = vextq_f16(vi3x01234567, vi3x89ABCDEF, 6);
      vi3x01234567 = vi3x89ABCDEF;
      const float16x8_t vi4x6789ABCD = vextq_f16(vi4x01234567, vi4x89ABCDEF, 6);
      vi4x01234567 = vi4x89ABCDEF;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x6789ABCD, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x6789ABCD, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x6789ABCD, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x9ABCDEFG, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4x9ABCDEFG, vwOP, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x9ABCDEFG, vwOP, 0);
      #endif
      // Right by 2 column
      const float16x8_t vi0xABCDEFGH = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 2);
      vi0x89ABCDEF = vi0xGHIJKLMN;
      const float16x8_t vi1xABCDEFGH = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 2);
      vi1x89ABCDEF = vi1xGHIJKLMN;
      const float16x8_t vi2xABCDEFGH = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 2);
      vi2x89ABCDEF = vi2xGHIJKLMN;
      const float16x8_t vi3xABCDEFGH = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 2);
      vi3x89ABCDEF = vi3xGHIJKLMN;
      const float16x8_t vi4xABCDEFGH = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 2);
      vi4x89ABCDEF = vi4xGHIJKLMN;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xABCDEFGH, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
    }

    // Always process the last block of 5..16 pixels.
    assert(w <= 16 * sizeof(uint16_t));
    if XNN_LIKELY(w > 8 * sizeof(uint16_t)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      float16x8_t vi4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;

      vi0xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0xGHIJKLMN)));
      vi1xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1xGHIJKLMN)));
      vi2xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2xGHIJKLMN)));
      vi3xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi3xGHIJKLMN)));
      vi4xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi4xGHIJKLMN)));

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x89ABCDEF, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x789ABCDE, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x789ABCDE, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x789ABCDE, vw89ABCDEF, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Left by 2 column
      const float16x8_t vi0x6789ABCD = vextq_f16(vi0x01234567, vi0x89ABCDEF, 6);
      vi0x01234567 = vi0x89ABCDEF;
      const float16x8_t vi1x6789ABCD = vextq_f16(vi1x01234567, vi1x89ABCDEF, 6);
      vi1x01234567 = vi1x89ABCDEF;
      const float16x8_t vi2x6789ABCD = vextq_f16(vi2x01234567, vi2x89ABCDEF, 6);
      vi2x01234567 = vi2x89ABCDEF;
      const float16x8_t vi3x6789ABCD = vextq_f16(vi3x01234567, vi3x89ABCDEF, 6);
      vi3x01234567 = vi3x89ABCDEF;
      const float16x8_t vi4x6789ABCD = vextq_f16(vi4x01234567, vi4x89ABCDEF, 6);
      vi4x01234567 = vi4x89ABCDEF;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x6789ABCD, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x6789ABCD, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x6789ABCD, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x9ABCDEFG, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4x9ABCDEFG, vwOP, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x9ABCDEFG, vwOP, 0);
      #endif
      // Right by 2 column
      const float16x8_t vi0xABCDEFGH = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 2);
      vi0x89ABCDEF = vi0xGHIJKLMN;
      const float16x8_t vi1xABCDEFGH = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 2);
      vi1x89ABCDEF = vi1xGHIJKLMN;
      const float16x8_t vi2xABCDEFGH = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 2);
      vi2x89ABCDEF = vi2xGHIJKLMN;
      const float16x8_t vi3xABCDEFGH = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 2);
      vi3x89ABCDEF = vi3xGHIJKLMN;
      const float16x8_t vi4xABCDEFGH = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 2);
      vi4x89ABCDEF = vi4xGHIJKLMN;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xABCDEFGH, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;

      w -= 8 * sizeof(uint16_t);
    }

    assert(w >= 1 * sizeof(uint16_t));
    assert(w <= 8 * sizeof(uint16_t));
    {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      vi0x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0x89ABCDEF)));
      vi1x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1x89ABCDEF)));
      vi2x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2x89ABCDEF)));
      vi3x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi3x89ABCDEF)));
      vi4x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi4x89ABCDEF)));

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x89ABCDEF, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x789ABCDE, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x789ABCDE, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x789ABCDE, vw89ABCDEF, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Left by 2 column
      const float16x8_t vi0x6789ABCD = vextq_f16(vi0x01234567, vi0x89ABCDEF, 6);
      const float16x8_t vi1x6789ABCD = vextq_f16(vi1x01234567, vi1x89ABCDEF, 6);
      const float16x8_t vi2x6789ABCD = vextq_f16(vi2x01234567, vi2x89ABCDEF, 6);
      const float16x8_t vi3x6789ABCD = vextq_f16(vi3x01234567, vi3x89ABCDEF, 6);
      const float16x8_t vi4x6789ABCD = vextq_f16(vi4x01234567, vi4x89ABCDEF, 6);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x6789ABCD, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x6789ABCD, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x6789ABCD, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vzero = vreinterpretq_f16_u16(vmovq_n_u16(0));
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vzero, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vzero, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vzero, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vzero, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vzero, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x9ABCDEFG, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4x9ABCDEFG, vwOP, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x9ABCDEFG, vwOP, 0);
      #endif
      // Right by 2 column
      const float16x8_t vi0xABCDEFGH = vextq_f16(vi0x9ABCDEFG, vzero, 1);
      const float16x8_t vi1xABCDEFGH = vextq_f16(vi1x9ABCDEFG, vzero, 1);
      const float16x8_t vi2xABCDEFGH = vextq_f16(vi2x9ABCDEFG, vzero, 1);
      const float16x8_t vi3xABCDEFGH = vextq_f16(vi3x9ABCDEFG, vzero, 1);
      const float16x8_t vi4xABCDEFGH = vextq_f16(vi4x9ABCDEFG, vzero, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xABCDEFGH, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      if XNN_LIKELY(w == 8 * sizeof(uint16_t)) {
        vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
      } else {
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w & (4 * sizeof(uint16_t))) {
         vst1_u16(o0, vreinterpret_u16_f16(vo0_lo)); o0 += 4;

          vo0_lo = vget_high_f16(vo0);
        }
        if (w & (2 * sizeof(uint16_t))) {
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
        }
        if (w & (1 * sizeof(uint16_t))) {
          vst1_lane_u16(o0, vreinterpret_u16_f16(vo0_lo), 0); o0 += 1;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i1 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i2 - input_decrement);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);


  } while (--output_height != 0);
}

void xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(uint16_t) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->neonfp16arith_stride2.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->neonfp16arith_stride2.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif
  const uint16x8_t vmask_even = vld1q_u16(params->neonfp16arith_stride2.mask_even);
  const uint16x8_t vmask_odd = vld1q_u16(params->neonfp16arith_stride2.mask_odd);

  const uint16_t* w = (const uint16_t*) weights;
  const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w));
  const float16x8_t vw89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w + 8));
  const float16x8_t vwGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w + 16));
  const float16x4_t vwOP = vreinterpret_f16_u32(vld1_dup_u32((const void*) (w + 24)));

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 16 * sizeof(uint16_t));

  const uint16_t* i0 = zero;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);


  uint16_t* o0 = output;

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
    }

    float16x8_t vi0x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi1x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi2x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi3x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi4x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));

    float16x8_t vi0x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi1x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi2x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi3x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi4x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));

    uint16x8x2_t vi0xGIKMOQSUHJLNPRTV = vld2q_u16(i0); i0 += 16;
    uint16x8x2_t vi1xGIKMOQSUHJLNPRTV = vld2q_u16(i1); i1 += 16;
    uint16x8x2_t vi2xGIKMOQSUHJLNPRTV = vld2q_u16(i2); i2 += 16;
    uint16x8x2_t vi3xGIKMOQSUHJLNPRTV = vld2q_u16(i3); i3 += 16;
    uint16x8x2_t vi4xGIKMOQSUHJLNPRTV = vld2q_u16(i4); i4 += 16;

    size_t w = input_width;
    for (; w > 16 * sizeof(uint16_t); w -= 16 * sizeof(uint16_t)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), vwGHIJKLMN, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Right by 2 column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), vw89ABCDEF, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]), vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), vwOP, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), vwOP, 0);
      #endif
      // Left by 2 column
      const float16x8_t vi0xEGIKMOQS = vextq_f16(vi0x02468ACE, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi0x02468ACE = vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi1xEGIKMOQS = vextq_f16(vi1x02468ACE, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi1x02468ACE = vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi2xEGIKMOQS = vextq_f16(vi2x02468ACE, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi2x02468ACE = vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi3xEGIKMOQS = vextq_f16(vi3x02468ACE, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi3x02468ACE = vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi4xEGIKMOQS = vextq_f16(vi4x02468ACE, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi4x02468ACE = vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xEGIKMOQS, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xEGIKMOQS, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xEGIKMOQS, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xEGIKMOQS, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xEGIKMOQS, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xEGIKMOQS, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xEGIKMOQS, vwGHIJKLMN, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xEGIKMOQS, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xEGIKMOQS, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xEGIKMOQS, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Left by 1 column, s1
      const float16x8_t vi0xFHJLNPRT = vextq_f16(vi0x13579BDF, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi0x13579BDF = vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi1xFHJLNPRT = vextq_f16(vi1x13579BDF, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi1x13579BDF = vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi2xFHJLNPRT = vextq_f16(vi2x13579BDF, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi2x13579BDF = vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi3xFHJLNPRT = vextq_f16(vi3x13579BDF, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi3x13579BDF = vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi4xFHJLNPRT = vextq_f16(vi4x13579BDF, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi4x13579BDF = vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]);

      const uint16x8x2_t vi0xWYacegikXZbdfhjl = vld2q_u16(i0); i0 += 16;
      const uint16x8x2_t vi1xWYacegikXZbdfhjl = vld2q_u16(i1); i1 += 16;
      const uint16x8x2_t vi2xWYacegikXZbdfhjl = vld2q_u16(i2); i2 += 16;
      const uint16x8x2_t vi3xWYacegikXZbdfhjl = vld2q_u16(i3); i3 += 16;
      const uint16x8x2_t vi4xWYacegikXZbdfhjl = vld2q_u16(i4); i4 += 16;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xFHJLNPRT, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xFHJLNPRT, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xFHJLNPRT, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xFHJLNPRT, vw89ABCDEF, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xFHJLNPRT, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xFHJLNPRT, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xFHJLNPRT, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xFHJLNPRT, vwGHIJKLMN, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xFHJLNPRT, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Right by 1 column
      const float16x8_t vi0xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi0xWYacegikXZbdfhjl.val[0]), 1);
      vi0xGIKMOQSUHJLNPRTV = vi0xWYacegikXZbdfhjl;
      const float16x8_t vi1xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi1xWYacegikXZbdfhjl.val[0]), 1);
      vi1xGIKMOQSUHJLNPRTV = vi1xWYacegikXZbdfhjl;
      const float16x8_t vi2xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi2xWYacegikXZbdfhjl.val[0]), 1);
      vi2xGIKMOQSUHJLNPRTV = vi2xWYacegikXZbdfhjl;
      const float16x8_t vi3xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi3xWYacegikXZbdfhjl.val[0]), 1);
      vi3xGIKMOQSUHJLNPRTV = vi3xWYacegikXZbdfhjl;
      const float16x8_t vi4xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi4xWYacegikXZbdfhjl.val[0]), 1);
      vi4xGIKMOQSUHJLNPRTV = vi4xWYacegikXZbdfhjl;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xIKMOQSUW, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xIKMOQSUW, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xIKMOQSUW, vw89ABCDEF, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xIKMOQSUW, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xIKMOQSUW, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xIKMOQSUW, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xIKMOQSUW, vwGHIJKLMN, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xIKMOQSUW, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xIKMOQSUW, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xIKMOQSUW, vwOP, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
    }

    // Last block has 1-16 pixels to process.
    assert(w <= 16 * sizeof(uint16_t));
    assert(w >= 1 * sizeof(uint16_t));
    {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8_t vi0xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi0xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi1xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi1xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi2xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi2xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi3xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi3xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi4xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi4xGIKMOQSUHJLNPRTV.val[0]));

      const float16x8_t vi0xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi0xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi1xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi1xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi2xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi2xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi3xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi3xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi4xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi4xGIKMOQSUHJLNPRTV.val[1]));

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xGIKMOQSU, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xGIKMOQSU, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xGIKMOQSU, vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xGIKMOQSU, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xGIKMOQSU, vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xGIKMOQSU, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xGIKMOQSU, vwGHIJKLMN, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xGIKMOQSU, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xGIKMOQSU, vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xGIKMOQSU, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Right by 1 column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xHJLNPRTV, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xHJLNPRTV, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xHJLNPRTV, vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xHJLNPRTV, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xHJLNPRTV, vw89ABCDEF, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xHJLNPRTV, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xHJLNPRTV, vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xHJLNPRTV, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xHJLNPRTV, vwOP, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xHJLNPRTV, vwOP, 0);
      #endif
      // Left by 2 columns
      const float16x8_t vi0xEGIKMOQS = vextq_f16(vi0x02468ACE, vi0xGIKMOQSU, 7);
      const float16x8_t vi1xEGIKMOQS = vextq_f16(vi1x02468ACE, vi1xGIKMOQSU, 7);
      const float16x8_t vi2xEGIKMOQS = vextq_f16(vi2x02468ACE, vi2xGIKMOQSU, 7);
      const float16x8_t vi3xEGIKMOQS = vextq_f16(vi3x02468ACE, vi3xGIKMOQSU, 7);
      const float16x8_t vi4xEGIKMOQS = vextq_f16(vi4x02468ACE, vi4xGIKMOQSU, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xEGIKMOQS, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xEGIKMOQS, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xEGIKMOQS, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xEGIKMOQS, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xEGIKMOQS, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xEGIKMOQS, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xEGIKMOQS, vwGHIJKLMN, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xEGIKMOQS, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xEGIKMOQS, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xEGIKMOQS, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Left by 1 column
      const float16x8_t vi0xFHJLNPRT = vextq_f16(vi0x13579BDF, vi0xHJLNPRTV, 7);
      const float16x8_t vi1xFHJLNPRT = vextq_f16(vi1x13579BDF, vi1xHJLNPRTV, 7);
      const float16x8_t vi2xFHJLNPRT = vextq_f16(vi2x13579BDF, vi2xHJLNPRTV, 7);
      const float16x8_t vi3xFHJLNPRT = vextq_f16(vi3x13579BDF, vi3xHJLNPRTV, 7);
      const float16x8_t vi4xFHJLNPRT = vextq_f16(vi4x13579BDF, vi4xHJLNPRTV, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xFHJLNPRT, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xFHJLNPRT, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xFHJLNPRT, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xFHJLNPRT, vw89ABCDEF, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xFHJLNPRT, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xFHJLNPRT, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xFHJLNPRT, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xFHJLNPRT, vwGHIJKLMN, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xFHJLNPRT, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Right by 2 columns
      const float16x8_t vzero = vreinterpretq_f16_u16(vmovq_n_u16(0));
      const float16x8_t vi0xIKMOQSUW = vextq_f16(vi0xGIKMOQSU, vzero, 1);
      const float16x8_t vi1xIKMOQSUW = vextq_f16(vi1xGIKMOQSU, vzero, 1);
      const float16x8_t vi2xIKMOQSUW = vextq_f16(vi2xGIKMOQSU, vzero, 1);
      const float16x8_t vi3xIKMOQSUW = vextq_f16(vi3xGIKMOQSU, vzero, 1);
      const float16x8_t vi4xIKMOQSUW = vextq_f16(vi4xGIKMOQSU, vzero, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xIKMOQSUW, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xIKMOQSUW, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xIKMOQSUW, vw89ABCDEF, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xIKMOQSUW, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xIKMOQSUW, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xIKMOQSUW, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xIKMOQSUW, vwGHIJKLMN, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xIKMOQSUW, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xIKMOQSUW, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xIKMOQSUW, vwOP, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      const size_t w_tmp = (w + 1 * sizeof(uint16_t)) / (2 * sizeof(uint16_t));

      if XNN_LIKELY(w_tmp == 8) {
        vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
      } else {
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w_tmp & 4) {
         vst1_u16(o0, vreinterpret_u16_f16(vo0_lo)); o0 += 4;

          vo0_lo = vget_high_f16(vo0);
        }
        if (w_tmp & 2) {
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
        }
        if (w_tmp & 1) {
          vst1_lane_u16(o0, vreinterpret_u16_f16(vo0_lo), 0); o0 += 1;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i2 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i3 - input_decrement);
    i2 = (const uint16_t*) ((uintptr_t) i4 - input_decrement);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}

void xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    float* output,
    const union xnn_f16_f32acc_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vscale = vld1q_dup_f32(&params->scale);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 16; channels -= 16) {
    const uint16_t* i0 = input;
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input + 1 * input_stride);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input + 2 * input_stride);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input + 3 * input_stride);
    const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input + 4 * input_stride);
    const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input + 5 * input_stride);
    const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input + 6 * input_stride);

    float32x4_t vacc0 = vdupq_n_f32(0.f);
    float32x4_t vacc1 = vdupq_n_f32(0.f);
    float32x4_t vacc2 = vdupq_n_f32(0.f);
    float32x4_t vacc3 = vdupq_n_f32(0.f);

    for (int r = rows; r > 0; r -= 7) {
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
      float32x4_t vin0;
      float32x4_t vin1;
      float32x4_t vin2;
      float32x4_t vin3;
      vin0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i0[0])));
      vin1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i0[4])));
      vin2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i0[8])));
      vin3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i0[12])));
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vin0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i1[0])));
      vin1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i1[4])));
      vin2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i1[8])));
      vin3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i1[12])));
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vin0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i2[0])));
      vin1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i2[4])));
      vin2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i2[8])));
      vin3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i2[12])));
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vin0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i3[0])));
      vin1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i3[4])));
      vin2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i3[8])));
      vin3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i3[12])));
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vin0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i4[0])));
      vin1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i4[4])));
      vin2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i4[8])));
      vin3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i4[12])));
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vin0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i5[0])));
      vin1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i5[4])));
      vin2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i5[8])));
      vin3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i5[12])));
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vin0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i6[0])));
      vin1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i6[4])));
      vin2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i6[8])));
      vin3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i6[12])));
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = vmulq_f32(vacc0, vscale);
    vacc1 = vmulq_f32(vacc1, vscale);
    vacc2 = vmulq_f32(vacc2, vscale);
    vacc3 = vmulq_f32(vacc3, vscale);

    const float* o = (const float*) output;
    float32x4_t vo0 = vld1q_f32(o); o += 4;
    float32x4_t vo1 = vld1q_f32(o); o += 4;
    float32x4_t vo2 = vld1q_f32(o); o += 4;
    float32x4_t vo3 = vld1q_f32(o); o += 4;
    float32x4_t v_out0 = vaddq_f32(vo0, vacc0);
    float32x4_t v_out1 = vaddq_f32(vo1, vacc1);
    float32x4_t v_out2 = vaddq_f32(vo2, vacc2);
    float32x4_t v_out3 = vaddq_f32(vo3, vacc3);
    vst1q_f32(output, v_out0); output = (void*) ((uintptr_t) output + 4 * sizeof(float));
    vst1q_f32(output, v_out1); output = (void*) ((uintptr_t) output + 4 * sizeof(float));
    vst1q_f32(output, v_out2); output = (void*) ((uintptr_t) output + 4 * sizeof(float));
    vst1q_f32(output, v_out3); output = (void*) ((uintptr_t) output + 4 * sizeof(float));

    input = (const uint16_t*) ((uintptr_t) input + 16 * sizeof(uint16_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const uint16_t* i0 = input;
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input + 1 * input_stride);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input + 2 * input_stride);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input + 3 * input_stride);
    const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input + 4 * input_stride);
    const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input + 5 * input_stride);
    const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input + 6 * input_stride);
    float32x4_t vacc[4];
    vacc[0] = vdupq_n_f32(0.f);
    vacc[1] = vdupq_n_f32(0.f);
    vacc[2] = vdupq_n_f32(0.f);
    vacc[3] = vdupq_n_f32(0.f);

    const size_t num_chunks = round_up_po2(channels, 4) >> 2;
    const size_t num_full_chunks = channels >> 2;
    for (int r = rows; r > 0; r -= 7) {
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
      for (int i = 0; i < num_chunks; ++i) {
        vacc[i] = vaddq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i0[i*4]))), vacc[i]);
        vacc[i] = vaddq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i1[i*4]))), vacc[i]);
        vacc[i] = vaddq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i2[i*4]))), vacc[i]);
        vacc[i] = vaddq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i3[i*4]))), vacc[i]);
        vacc[i] = vaddq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i4[i*4]))), vacc[i]);
        vacc[i] = vaddq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i5[i*4]))), vacc[i]);
        vacc[i] = vaddq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&i6[i*4]))), vacc[i]);
      }
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
    }
    for (int i = 0; i < (channels + 4) >> 2; ++i) {
      vacc[i] = vmulq_f32(vacc[i], vscale);
    }

    float32x4_t vo[4];
    const float* o = (const float*) output;
    for (int i = 0; i < num_full_chunks; ++i) {
      vo[i] = vld1q_f32(o); o += 4;
    }
    float32x4_t v_out[4];
    for (int i = 0; i < num_full_chunks; ++i) {
      v_out[i] = vaddq_f32(vo[i], vacc[i]);
    }
    for (int i = 0; i < num_full_chunks; ++i) {
      vst1q_f32(output, v_out[i]); output = (void*) ((uintptr_t) output + 4 * sizeof(float));
    }

    const size_t pos = channels >> 2;
    channels &= 0x3;
    float32x2_t vacc_low = vget_low_f32(vacc[pos]);
    if (channels & 2) {
      vst1_f32(output, vadd_f32(vacc_low, vld1_f32(output))); output = (void*) ((uintptr_t) output + 2 * sizeof(float));
      vacc_low = vget_high_f32(vacc[pos]);
    }
    if (channels & 1) {
      vst1_lane_f32(output, vadd_f32(vacc_low, vld1_dup_f32(output)), 0);
    }
  }
}

void xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u32_acc4(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32acc_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  float32x4_t vacc3 = vmovq_n_f32(0.0f);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh67 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    const float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    const float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    const float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    const float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    const float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));
    const float32x4_t vt6 = vcvt_f32_f16(vget_low_f16(vh67));
    const float32x4_t vt7 = vcvt_f32_f16(vget_high_f16(vh67));

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc2 = vaddq_f32(vacc2, vt2);
    vacc3 = vaddq_f32(vacc3, vt3);
    vacc0 = vaddq_f32(vacc0, vt4);
    vacc1 = vaddq_f32(vacc1, vt5);
    vacc2 = vaddq_f32(vacc2, vt6);
    vacc3 = vaddq_f32(vacc3, vt7);
  }
  vacc0 = vaddq_f32(vacc0, vacc1);
  vacc2 = vaddq_f32(vacc2, vacc3);
  vacc0 = vaddq_f32(vacc0, vacc2);
  for (; batch >= 4 * sizeof(uint16_t); batch -= 4 * sizeof(uint16_t)) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16(i)); i += 4;
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  const float32x2_t vscale = vld1_dup_f32(&params->scale);
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i)); i += 2;
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8(
    size_t elements,
    size_t channels,
    const void* input,
    void* output,
    const union xnn_f16_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(elements != 0);
  assert(elements % sizeof(uint16_t) == 0);
  assert(channels != 0);

  const uint16x8_t vmask = vld1q_u16(params->neonfp16arith.mask);
  const float16x4_t vmultiplier = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.multiplier));
  const float16x4_t voutput_min = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.output_min));
  const float16x4_t voutput_max = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.output_max));

  uint16_t* o = (uint16_t*) output;
  const uint16_t* i = input;
  do {
    float16x8_t vsum0 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vsum1 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    size_t n = elements;
    if (n >= 32 * sizeof(uint16_t)) {
      do {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i));
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i + 8));
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i + 16));
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i + 24));
        i += 32;
        const float16x8_t acc0 = vaddq_f16(vi0, vi1);
        const float16x8_t acc1 = vaddq_f16(vi2, vi3);
        vsum0 = vaddq_f16(vsum0, acc0);
        vsum1 = vaddq_f16(vsum1, acc1);
        n -= 32 * sizeof(uint16_t);
      } while (n >= 32 * sizeof(uint16_t));
    }
    vsum0 = vaddq_f16(vsum0, vsum1);

    while (n >= 8 * sizeof(uint16_t)) {
      const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i));
      i += 8;
      vsum0 = vaddq_f16(vsum0, vi0);
      n -= 8 * sizeof(uint16_t);
    }

    if XNN_UNLIKELY(n != 0) {
      float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i)); i = (const uint16_t*) ((uintptr_t) i + n);

      vi0 = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0)));

      vsum0 = vaddq_f16(vsum0, vi0);
    }

    const float16x4_t vout4 = vpadd_f16(vget_low_f16(vsum0), vget_high_f16(vsum0));
    const float16x4_t vout2 = vpadd_f16(vout4, vout4);
    const float16x4_t vout1 = vpadd_f16(vout2, vout2);

    float16x4_t vout = vmul_f16(vout1, vmultiplier);

    vout = vmax_f16(vout, voutput_min);
    vout = vmin_f16(vout, voutput_max);

    vst1_lane_u16(o, vreinterpret_u16_f16(vout), 0); o += 1;
  } while (--channels != 0);
}

void xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* buffer,
    void* output,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(uint16_t);

  uint16_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {
    const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;

    const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vacc01234567 = vaddq_f16(vi0x01234567, vi1x01234567);

    const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
    const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
    const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
    const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
    vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

    vst1q_u16(b, vreinterpretq_u16_f16(vacc01234567)); b += 8;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);

    uint16_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {
      float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi0x01234567);
      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi1x01234567);
      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
      vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

      vst1q_u16(b, vreinterpretq_u16_f16(vacc01234567)); b += 8;
    }
  }

  i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const uint16_t*) zero;
  }
  i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const uint16_t*) zero;
  }
  i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const uint16_t*) zero;
  }
  i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const uint16_t*) zero;
  }
  i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const uint16_t*) zero;
  }
  i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const uint16_t*) zero;
  }

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  for (; channels >= 8; channels -= 8) {
    float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_u16(buffer)); buffer = (uint16_t*) buffer + 8;

    const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;

    const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi0x01234567);
    const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi1x01234567);
    const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
    const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
    const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
    const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
    vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

    vacc01234567 = vmulq_f16(vacc01234567, vscale);

    vacc01234567 = vmaxq_f16(vacc01234567, vmin);

    vacc01234567 = vminq_f16(vacc01234567, vmax);

    vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output = (uint16_t*) output + 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_u16(buffer)); buffer = (uint16_t*) buffer + 8;

      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi0x01234567);
      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi1x01234567);
      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
      vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

      vacc01234567 = vmulq_f16(vacc01234567, vscale);
      vacc01234567 = vmaxq_f16(vacc01234567, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (channels & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output = (uint16_t*) output + 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (channels & 2) {
        vst1_lane_u32(output, vreinterpret_u32_f16(vacc0123), 0); output = (uint16_t*) output + 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (channels & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output = (uint16_t*) output + 1;
      }
    }
  }
}

void xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const uint16_t*) zero;
  }
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const uint16_t*) zero;
  }
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const uint16_t*) zero;
  }
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const uint16_t*) zero;
  }
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const uint16_t*) zero;
  }
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const uint16_t*) zero;
  }

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  for (; channels >= 8; channels -= 8) {
    const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;

    const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vacc01234567 = vaddq_f16(vi0x01234567, vi1x01234567);

    const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
    const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
    const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
    const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
    vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

    vacc01234567 = vmulq_f16(vacc01234567, vscale);

    vacc01234567 = vmaxq_f16(vacc01234567, vmin);

    vacc01234567 = vminq_f16(vacc01234567, vmax);

    vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output = (uint16_t*) output + 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      float16x8_t vacc01234567 = vaddq_f16(vi0x01234567, vi1x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
      vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

      vacc01234567 = vmulq_f16(vacc01234567, vscale);
      vacc01234567 = vmaxq_f16(vacc01234567, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (channels & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output = (uint16_t*) output + 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (channels & 2) {
        vst1_lane_u32(output, vreinterpret_u32_f16(vacc0123), 0); output = (uint16_t*) output + 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (channels & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output = (uint16_t*) output + 1;
      }
    }
  }
}

void xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc0x1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

    size_t k = kc;
    while (k >= 4 * sizeof(uint16_t)) {
      const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;

      const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
      const float16x8_t vb1c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c0, va0, 0);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c0, va0, 0);
      #endif
      const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
      const float16x8_t vb1c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c1, va0, 1);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c1, va0, 1);
      #endif
      const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
      const float16x8_t vb1c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c2, va0, 2);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c2, va0, 2);
      #endif
      const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
      const float16x8_t vb1c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c3, va0, 3);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c3, va0, 3);
      #endif

      k -= 4 * sizeof(uint16_t);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;

        const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);
        vacc0x1 = vfmaq_f16(vacc0x1, va0, vb1);

        k -= sizeof(uint16_t);
      } while (k != 0);
    }

    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);
    vacc0x1 = vmaxq_f16(vacc0x1, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);
    vacc0x1 = vminq_f16(vacc0x1, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vacc0x1));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0)); c0 += 8;

        vacc0x0 = vacc0x1;
      }
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;

        vacc0 = vget_high_f16(vacc0x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;

        vacc0 = vext_f16(vacc0, vacc0, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

    size_t k = kc;
    while (k >= 4 * sizeof(uint16_t)) {
      const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;

      const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
      #endif
      const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
      #endif
      const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
      #endif
      const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
      #endif

      k -= 4 * sizeof(uint16_t);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;

        const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);

        k -= sizeof(uint16_t);
      } while (k != 0);
    }

    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;

        vacc0 = vget_high_f16(vacc0x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;

        vacc0 = vext_f16(vacc0, vacc0, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint16_t* a5 = (const uint16_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc0x1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc1x0 = vacc0x0;
    float16x8_t vacc1x1 = vacc0x1;
    float16x8_t vacc2x0 = vacc0x0;
    float16x8_t vacc2x1 = vacc0x1;
    float16x8_t vacc3x0 = vacc0x0;
    float16x8_t vacc3x1 = vacc0x1;
    float16x8_t vacc4x0 = vacc0x0;
    float16x8_t vacc4x1 = vacc0x1;
    float16x8_t vacc5x0 = vacc0x0;
    float16x8_t vacc5x1 = vacc0x1;

    size_t k = kc;
    while (k >= 4 * sizeof(uint16_t)) {
      const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;
      const float16x4_t va1 = vreinterpret_f16_u16(vld1_u16(a1)); a1 += 4;
      const float16x4_t va2 = vreinterpret_f16_u16(vld1_u16(a2)); a2 += 4;
      const float16x4_t va3 = vreinterpret_f16_u16(vld1_u16(a3)); a3 += 4;
      const float16x4_t va4 = vreinterpret_f16_u16(vld1_u16(a4)); a4 += 4;
      const float16x4_t va5 = vreinterpret_f16_u16(vld1_u16(a5)); a5 += 4;

      const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
      const float16x8_t vb1c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c0, va1, 0);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c0, va2, 0);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c0, va3, 0);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c0, va4, 0);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c0, va5, 0);
        vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c0, va0, 0);
        vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c0, va1, 0);
        vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c0, va2, 0);
        vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c0, va3, 0);
        vacc4x1 = vfmaq_lane_f16(vacc4x1, vb1c0, va4, 0);
        vacc5x1 = vfmaq_lane_f16(vacc5x1, vb1c0, va5, 0);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c0, va1, 0);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c0, va2, 0);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c0, va3, 0);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c0, va4, 0);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c0, va5, 0);
        vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c0, va0, 0);
        vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c0, va1, 0);
        vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c0, va2, 0);
        vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c0, va3, 0);
        vacc4x1 = vmlaq_lane_f16(vacc4x1, vb1c0, va4, 0);
        vacc5x1 = vmlaq_lane_f16(vacc5x1, vb1c0, va5, 0);
      #endif
      const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
      const float16x8_t vb1c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c1, va1, 1);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c1, va2, 1);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c1, va3, 1);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c1, va4, 1);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c1, va5, 1);
        vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c1, va0, 1);
        vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c1, va1, 1);
        vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c1, va2, 1);
        vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c1, va3, 1);
        vacc4x1 = vfmaq_lane_f16(vacc4x1, vb1c1, va4, 1);
        vacc5x1 = vfmaq_lane_f16(vacc5x1, vb1c1, va5, 1);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c1, va1, 1);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c1, va2, 1);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c1, va3, 1);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c1, va4, 1);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c1, va5, 1);
        vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c1, va0, 1);
        vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c1, va1, 1);
        vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c1, va2, 1);
        vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c1, va3, 1);
        vacc4x1 = vmlaq_lane_f16(vacc4x1, vb1c1, va4, 1);
        vacc5x1 = vmlaq_lane_f16(vacc5x1, vb1c1, va5, 1);
      #endif
      const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
      const float16x8_t vb1c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c2, va1, 2);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c2, va2, 2);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c2, va3, 2);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c2, va4, 2);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c2, va5, 2);
        vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c2, va0, 2);
        vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c2, va1, 2);
        vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c2, va2, 2);
        vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c2, va3, 2);
        vacc4x1 = vfmaq_lane_f16(vacc4x1, vb1c2, va4, 2);
        vacc5x1 = vfmaq_lane_f16(vacc5x1, vb1c2, va5, 2);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c2, va1, 2);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c2, va2, 2);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c2, va3, 2);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c2, va4, 2);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c2, va5, 2);
        vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c2, va0, 2);
        vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c2, va1, 2);
        vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c2, va2, 2);
        vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c2, va3, 2);
        vacc4x1 = vmlaq_lane_f16(vacc4x1, vb1c2, va4, 2);
        vacc5x1 = vmlaq_lane_f16(vacc5x1, vb1c2, va5, 2);
      #endif
      const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
      const float16x8_t vb1c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c3, va1, 3);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c3, va2, 3);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c3, va3, 3);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c3, va4, 3);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c3, va5, 3);
        vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c3, va0, 3);
        vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c3, va1, 3);
        vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c3, va2, 3);
        vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c3, va3, 3);
        vacc4x1 = vfmaq_lane_f16(vacc4x1, vb1c3, va4, 3);
        vacc5x1 = vfmaq_lane_f16(vacc5x1, vb1c3, va5, 3);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c3, va1, 3);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c3, va2, 3);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c3, va3, 3);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c3, va4, 3);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c3, va5, 3);
        vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c3, va0, 3);
        vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c3, va1, 3);
        vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c3, va2, 3);
        vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c3, va3, 3);
        vacc4x1 = vmlaq_lane_f16(vacc4x1, vb1c3, va4, 3);
        vacc5x1 = vmlaq_lane_f16(vacc5x1, vb1c3, va5, 3);
      #endif

      k -= 4 * sizeof(uint16_t);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;
        const float16x8_t va1 = vreinterpretq_f16_u16(vld1q_dup_u16(a1)); a1 += 1;
        const float16x8_t va2 = vreinterpretq_f16_u16(vld1q_dup_u16(a2)); a2 += 1;
        const float16x8_t va3 = vreinterpretq_f16_u16(vld1q_dup_u16(a3)); a3 += 1;
        const float16x8_t va4 = vreinterpretq_f16_u16(vld1q_dup_u16(a4)); a4 += 1;
        const float16x8_t va5 = vreinterpretq_f16_u16(vld1q_dup_u16(a5)); a5 += 1;

        const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);
        vacc1x0 = vfmaq_f16(vacc1x0, va1, vb0);
        vacc2x0 = vfmaq_f16(vacc2x0, va2, vb0);
        vacc3x0 = vfmaq_f16(vacc3x0, va3, vb0);
        vacc4x0 = vfmaq_f16(vacc4x0, va4, vb0);
        vacc5x0 = vfmaq_f16(vacc5x0, va5, vb0);
        vacc0x1 = vfmaq_f16(vacc0x1, va0, vb1);
        vacc1x1 = vfmaq_f16(vacc1x1, va1, vb1);
        vacc2x1 = vfmaq_f16(vacc2x1, va2, vb1);
        vacc3x1 = vfmaq_f16(vacc3x1, va3, vb1);
        vacc4x1 = vfmaq_f16(vacc4x1, va4, vb1);
        vacc5x1 = vfmaq_f16(vacc5x1, va5, vb1);

        k -= sizeof(uint16_t);
      } while (k != 0);
    }

    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);
    vacc1x0 = vmaxq_f16(vacc1x0, vmin);
    vacc2x0 = vmaxq_f16(vacc2x0, vmin);
    vacc3x0 = vmaxq_f16(vacc3x0, vmin);
    vacc4x0 = vmaxq_f16(vacc4x0, vmin);
    vacc5x0 = vmaxq_f16(vacc5x0, vmin);
    vacc0x1 = vmaxq_f16(vacc0x1, vmin);
    vacc1x1 = vmaxq_f16(vacc1x1, vmin);
    vacc2x1 = vmaxq_f16(vacc2x1, vmin);
    vacc3x1 = vmaxq_f16(vacc3x1, vmin);
    vacc4x1 = vmaxq_f16(vacc4x1, vmin);
    vacc5x1 = vmaxq_f16(vacc5x1, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);
    vacc1x0 = vminq_f16(vacc1x0, vmax);
    vacc2x0 = vminq_f16(vacc2x0, vmax);
    vacc3x0 = vminq_f16(vacc3x0, vmax);
    vacc4x0 = vminq_f16(vacc4x0, vmax);
    vacc5x0 = vminq_f16(vacc5x0, vmax);
    vacc0x1 = vminq_f16(vacc0x1, vmax);
    vacc1x1 = vminq_f16(vacc1x1, vmax);
    vacc2x1 = vminq_f16(vacc2x1, vmax);
    vacc3x1 = vminq_f16(vacc3x1, vmax);
    vacc4x1 = vminq_f16(vacc4x1, vmax);
    vacc5x1 = vminq_f16(vacc5x1, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vacc0x1));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vacc1x1));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vacc2x1));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vacc3x1));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x0));
      vst1q_u16(c4 + 8, vreinterpretq_u16_f16(vacc4x1));
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x0));
      vst1q_u16(c5 + 8, vreinterpretq_u16_f16(vacc5x1));
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
      a4 = (const uint16_t*) ((uintptr_t) a4 - kc);
      a5 = (const uint16_t*) ((uintptr_t) a5 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0)); c0 += 8;
        vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0)); c1 += 8;
        vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0)); c2 += 8;
        vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0)); c3 += 8;
        vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x0)); c4 += 8;
        vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x0)); c5 += 8;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
        vacc4x0 = vacc4x1;
        vacc5x0 = vacc5x1;
      }
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      float16x4_t vacc1 = vget_low_f16(vacc1x0);
      float16x4_t vacc2 = vget_low_f16(vacc2x0);
      float16x4_t vacc3 = vget_low_f16(vacc3x0);
      float16x4_t vacc4 = vget_low_f16(vacc4x0);
      float16x4_t vacc5 = vget_low_f16(vacc5x0);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;
        vst1_u16(c1, vreinterpret_u16_f16(vacc1)); c1 += 4;
        vst1_u16(c2, vreinterpret_u16_f16(vacc2)); c2 += 4;
        vst1_u16(c3, vreinterpret_u16_f16(vacc3)); c3 += 4;
        vst1_u16(c4, vreinterpret_u16_f16(vacc4)); c4 += 4;
        vst1_u16(c5, vreinterpret_u16_f16(vacc5)); c5 += 4;

        vacc0 = vget_high_f16(vacc0x0);
        vacc1 = vget_high_f16(vacc1x0);
        vacc2 = vget_high_f16(vacc2x0);
        vacc3 = vget_high_f16(vacc3x0);
        vacc4 = vget_high_f16(vacc4x0);
        vacc5 = vget_high_f16(vacc5x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vacc1), 0); c1 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vacc2), 0); c2 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vacc3), 0); c3 += 2;
        vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vacc4), 0); c4 += 2;
        vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vacc5), 0); c5 += 2;

        vacc0 = vext_f16(vacc0, vacc0, 2);
        vacc1 = vext_f16(vacc1, vacc1, 2);
        vacc2 = vext_f16(vacc2, vacc2, 2);
        vacc3 = vext_f16(vacc3, vacc3, 2);
        vacc4 = vext_f16(vacc4, vacc4, 2);
        vacc5 = vext_f16(vacc5, vacc5, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
        vst1_lane_u16(c1, vreinterpret_u16_f16(vacc1), 0);
        vst1_lane_u16(c2, vreinterpret_u16_f16(vacc2), 0);
        vst1_lane_u16(c3, vreinterpret_u16_f16(vacc3), 0);
        vst1_lane_u16(c4, vreinterpret_u16_f16(vacc4), 0);
        vst1_lane_u16(c5, vreinterpret_u16_f16(vacc5), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint16_t* a5 = (const uint16_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc1x0 = vacc0x0;
    float16x8_t vacc2x0 = vacc0x0;
    float16x8_t vacc3x0 = vacc0x0;
    float16x8_t vacc4x0 = vacc0x0;
    float16x8_t vacc5x0 = vacc0x0;

    size_t k = kc;
    while (k >= 4 * sizeof(uint16_t)) {
      const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;
      const float16x4_t va1 = vreinterpret_f16_u16(vld1_u16(a1)); a1 += 4;
      const float16x4_t va2 = vreinterpret_f16_u16(vld1_u16(a2)); a2 += 4;
      const float16x4_t va3 = vreinterpret_f16_u16(vld1_u16(a3)); a3 += 4;
      const float16x4_t va4 = vreinterpret_f16_u16(vld1_u16(a4)); a4 += 4;
      const float16x4_t va5 = vreinterpret_f16_u16(vld1_u16(a5)); a5 += 4;

      const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c0, va1, 0);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c0, va2, 0);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c0, va3, 0);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c0, va4, 0);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c0, va5, 0);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c0, va1, 0);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c0, va2, 0);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c0, va3, 0);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c0, va4, 0);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c0, va5, 0);
      #endif
      const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c1, va1, 1);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c1, va2, 1);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c1, va3, 1);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c1, va4, 1);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c1, va5, 1);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c1, va1, 1);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c1, va2, 1);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c1, va3, 1);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c1, va4, 1);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c1, va5, 1);
      #endif
      const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c2, va1, 2);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c2, va2, 2);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c2, va3, 2);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c2, va4, 2);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c2, va5, 2);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c2, va1, 2);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c2, va2, 2);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c2, va3, 2);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c2, va4, 2);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c2, va5, 2);
      #endif
      const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c3, va1, 3);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c3, va2, 3);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c3, va3, 3);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c3, va4, 3);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c3, va5, 3);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c3, va1, 3);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c3, va2, 3);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c3, va3, 3);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c3, va4, 3);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c3, va5, 3);
      #endif

      k -= 4 * sizeof(uint16_t);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;
        const float16x8_t va1 = vreinterpretq_f16_u16(vld1q_dup_u16(a1)); a1 += 1;
        const float16x8_t va2 = vreinterpretq_f16_u16(vld1q_dup_u16(a2)); a2 += 1;
        const float16x8_t va3 = vreinterpretq_f16_u16(vld1q_dup_u16(a3)); a3 += 1;
        const float16x8_t va4 = vreinterpretq_f16_u16(vld1q_dup_u16(a4)); a4 += 1;
        const float16x8_t va5 = vreinterpretq_f16_u16(vld1q_dup_u16(a5)); a5 += 1;

        const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);
        vacc1x0 = vfmaq_f16(vacc1x0, va1, vb0);
        vacc2x0 = vfmaq_f16(vacc2x0, va2, vb0);
        vacc3x0 = vfmaq_f16(vacc3x0, va3, vb0);
        vacc4x0 = vfmaq_f16(vacc4x0, va4, vb0);
        vacc5x0 = vfmaq_f16(vacc5x0, va5, vb0);

        k -= sizeof(uint16_t);
      } while (k != 0);
    }

    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);
    vacc1x0 = vmaxq_f16(vacc1x0, vmin);
    vacc2x0 = vmaxq_f16(vacc2x0, vmin);
    vacc3x0 = vmaxq_f16(vacc3x0, vmin);
    vacc4x0 = vmaxq_f16(vacc4x0, vmin);
    vacc5x0 = vmaxq_f16(vacc5x0, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);
    vacc1x0 = vminq_f16(vacc1x0, vmax);
    vacc2x0 = vminq_f16(vacc2x0, vmax);
    vacc3x0 = vminq_f16(vacc3x0, vmax);
    vacc4x0 = vminq_f16(vacc4x0, vmax);
    vacc5x0 = vminq_f16(vacc5x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x0));
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x0));
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
      a4 = (const uint16_t*) ((uintptr_t) a4 - kc);
      a5 = (const uint16_t*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      float16x4_t vacc1 = vget_low_f16(vacc1x0);
      float16x4_t vacc2 = vget_low_f16(vacc2x0);
      float16x4_t vacc3 = vget_low_f16(vacc3x0);
      float16x4_t vacc4 = vget_low_f16(vacc4x0);
      float16x4_t vacc5 = vget_low_f16(vacc5x0);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;
        vst1_u16(c1, vreinterpret_u16_f16(vacc1)); c1 += 4;
        vst1_u16(c2, vreinterpret_u16_f16(vacc2)); c2 += 4;
        vst1_u16(c3, vreinterpret_u16_f16(vacc3)); c3 += 4;
        vst1_u16(c4, vreinterpret_u16_f16(vacc4)); c4 += 4;
        vst1_u16(c5, vreinterpret_u16_f16(vacc5)); c5 += 4;

        vacc0 = vget_high_f16(vacc0x0);
        vacc1 = vget_high_f16(vacc1x0);
        vacc2 = vget_high_f16(vacc2x0);
        vacc3 = vget_high_f16(vacc3x0);
        vacc4 = vget_high_f16(vacc4x0);
        vacc5 = vget_high_f16(vacc5x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vacc1), 0); c1 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vacc2), 0); c2 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vacc3), 0); c3 += 2;
        vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vacc4), 0); c4 += 2;
        vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vacc5), 0); c5 += 2;

        vacc0 = vext_f16(vacc0, vacc0, 2);
        vacc1 = vext_f16(vacc1, vacc1, 2);
        vacc2 = vext_f16(vacc2, vacc2, 2);
        vacc3 = vext_f16(vacc3, vacc3, 2);
        vacc4 = vext_f16(vacc4, vacc4, 2);
        vacc5 = vext_f16(vacc5, vacc5, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
        vst1_lane_u16(c1, vreinterpret_u16_f16(vacc1), 0);
        vst1_lane_u16(c2, vreinterpret_u16_f16(vacc2), 0);
        vst1_lane_u16(c3, vreinterpret_u16_f16(vacc3), 0);
        vst1_lane_u16(c4, vreinterpret_u16_f16(vacc4), 0);
        vst1_lane_u16(c5, vreinterpret_u16_f16(vacc5), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8(
    size_t output_pixels,
    size_t channels,
    const void** restrict input,
    size_t input_offset,
    const void* restrict weights,
    void* restrict output,
    size_t input_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(uint16_t) == 0);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t** i = (const uint16_t**)input;
    const uint16_t* w = weights;
    size_t p = output_pixels;

    for (; p >= 8; p -= 8) {
      const uint16_t* itl0 = (const uint16_t*) ((uintptr_t) i[0] + input_offset);
      const uint16_t* ibl0 = (const uint16_t*) ((uintptr_t) i[1] + input_offset);
      const uint16_t* itl1 = (const uint16_t*) ((uintptr_t) i[2] + input_offset);
      const uint16_t* ibl1 = (const uint16_t*) ((uintptr_t) i[3] + input_offset);
      const uint16_t* itl2 = (const uint16_t*) ((uintptr_t) i[4] + input_offset);
      const uint16_t* ibl2 = (const uint16_t*) ((uintptr_t) i[5] + input_offset);
      const uint16_t* itl3 = (const uint16_t*) ((uintptr_t) i[6] + input_offset);
      const uint16_t* ibl3 = (const uint16_t*) ((uintptr_t) i[7] + input_offset);
      const uint16_t* itl4 = (const uint16_t*) ((uintptr_t) i[8] + input_offset);
      const uint16_t* ibl4 = (const uint16_t*) ((uintptr_t) i[9] + input_offset);
      const uint16_t* itl5 = (const uint16_t*) ((uintptr_t) i[10] + input_offset);
      const uint16_t* ibl5 = (const uint16_t*) ((uintptr_t) i[11] + input_offset);
      const uint16_t* itl6 = (const uint16_t*) ((uintptr_t) i[12] + input_offset);
      const uint16_t* ibl6 = (const uint16_t*) ((uintptr_t) i[13] + input_offset);
      const uint16_t* itl7 = (const uint16_t*) ((uintptr_t) i[14] + input_offset);
      const uint16_t* ibl7 = (const uint16_t*) ((uintptr_t) i[15] + input_offset);
      i += 2 * 8;

      const uint16x4x2_t vw0123 = vld2_u16(w); w += 8;
      const uint16x4x2_t vw4567 = vld2_u16(w); w += 8;

      float16x8_t vtltr0123 = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) itl0));
      float16x8_t vblbr0123 = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) ibl0));
      float16x8_t vtltr4567 = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) itl4));
      float16x8_t vblbr4567 = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) ibl4));

      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl1, vreinterpretq_u32_f16(vtltr0123), 1));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl1, vreinterpretq_u32_f16(vblbr0123), 1));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl5, vreinterpretq_u32_f16(vtltr4567), 1));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl5, vreinterpretq_u32_f16(vblbr4567), 1));
      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl2, vreinterpretq_u32_f16(vtltr0123), 2));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl2, vreinterpretq_u32_f16(vblbr0123), 2));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl6, vreinterpretq_u32_f16(vtltr4567), 2));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl6, vreinterpretq_u32_f16(vblbr4567), 2));
      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl3, vreinterpretq_u32_f16(vtltr0123), 3));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl3, vreinterpretq_u32_f16(vblbr0123), 3));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl7, vreinterpretq_u32_f16(vtltr4567), 3));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl7, vreinterpretq_u32_f16(vblbr4567), 3));

      const float16x8_t valphah01234567 = vreinterpretq_f16_u16(vcombine_u16(vw0123.val[0], vw4567.val[0]));
      const float16x8_t valphav01234567 = vreinterpretq_f16_u16(vcombine_u16(vw0123.val[1], vw4567.val[1]));

      const float16x8_t vldrd0123 = vsubq_f16(vblbr0123, vtltr0123);
      const float16x8_t vldrd4567 = vsubq_f16(vblbr4567, vtltr4567);

      const float16x8x2_t vld_t01234567 = vuzpq_f16(vldrd0123, vldrd4567);
      const float16x8_t vld01234567 = vld_t01234567.val[0];
      const float16x8_t vrd01234567 = vld_t01234567.val[1];

      const float16x8x2_t vtl_t01234567 = vuzpq_f16(vtltr0123, vtltr4567);
      const float16x8_t vtl01234567 = vtl_t01234567.val[0];
      const float16x8_t vtr01234567 = vtl_t01234567.val[1];

      const float16x8_t vl01234567 = vfmaq_f16(vtl01234567, vld01234567, valphav01234567);
      const float16x8_t vr01234567 = vfmaq_f16(vtr01234567, vrd01234567, valphav01234567);

      const float16x8_t vd01234567 = vsubq_f16(vr01234567, vl01234567);
      const float16x8_t vo01234567 = vfmaq_f16(vl01234567, vd01234567, valphah01234567);

      vst1q_u16(o, vreinterpretq_u16_f16(vo01234567)); o += 8;
    }
    for (; p >= 4; p -= 4) {
      const uint16_t* itl0 = (const uint16_t*) ((uintptr_t) i[0] + input_offset);
      const uint16_t* ibl0 = (const uint16_t*) ((uintptr_t) i[1] + input_offset);
      const uint16_t* itl1 = (const uint16_t*) ((uintptr_t) i[2] + input_offset);
      const uint16_t* ibl1 = (const uint16_t*) ((uintptr_t) i[3] + input_offset);
      const uint16_t* itl2 = (const uint16_t*) ((uintptr_t) i[4] + input_offset);
      const uint16_t* ibl2 = (const uint16_t*) ((uintptr_t) i[5] + input_offset);
      const uint16_t* itl3 = (const uint16_t*) ((uintptr_t) i[6] + input_offset);
      const uint16_t* ibl3 = (const uint16_t*) ((uintptr_t) i[7] + input_offset);
      i += 8;

      const uint16x4x2_t vw = vld2_u16(w); w += 8;

      float16x8_t vtltr = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) itl0));
      float16x8_t vblbr = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) ibl0));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl1, vreinterpretq_u32_f16(vtltr), 1));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl1, vreinterpretq_u32_f16(vblbr), 1));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl2, vreinterpretq_u32_f16(vtltr), 2));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl2, vreinterpretq_u32_f16(vblbr), 2));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl3, vreinterpretq_u32_f16(vtltr), 3));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl3, vreinterpretq_u32_f16(vblbr), 3));

      const float16x4_t valphah = vreinterpret_f16_u16(vw.val[0]);
      const float16x4_t valphav = vreinterpret_f16_u16(vw.val[1]);

      const float16x8_t vldrd = vsubq_f16(vblbr, vtltr);

      const float16x4x2_t vld_t = vuzp_f16(vget_low_f16(vldrd), vget_high_f16(vldrd));
      const float16x4_t vld = vld_t.val[0];
      const float16x4_t vrd = vld_t.val[1];

      const float16x4x2_t vtl_t = vuzp_f16(vget_low_f16(vtltr), vget_high_f16(vtltr));
      const float16x4_t vtl = vtl_t.val[0];
      const float16x4_t vtr = vtl_t.val[1];

      const float16x4_t vl = vfma_f16(vtl, vld, valphav);
      const float16x4_t vr = vfma_f16(vtr, vrd, valphav);

      const float16x4_t vd = vsub_f16(vr, vl);
      const float16x4_t vo = vfma_f16(vl, vd, valphah);

      vst1_u16(o, vreinterpret_u16_f16(vo)); o += 4;
    }
    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const uint16_t* itl0 = (const uint16_t*) ((uintptr_t) i[0] + input_offset);
        const uint16_t* ibl0 = (const uint16_t*) ((uintptr_t) i[1] + input_offset);
        const uint16_t* itl1 = (const uint16_t*) ((uintptr_t) i[2] + input_offset);
        const uint16_t* ibl1 = (const uint16_t*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const float16x4_t vw = vreinterpret_f16_u16(vld1_u16(w)); w += 4;

        const float16x4x2_t vwhv = vuzp_f16(vw, vw);
        const float16x4_t valphah = vwhv.val[0];
        const float16x4_t valphav = vwhv.val[1];

        float16x4_t vtltr = vreinterpret_f16_u32(vld1_dup_u32((const void*) itl0));
        float16x4_t vblbr = vreinterpret_f16_u32(vld1_dup_u32((const void*) ibl0));

        vtltr = vreinterpret_f16_u32(vld1_lane_u32((const void*) itl1, vreinterpret_u32_f16(vtltr), 1));
        vblbr = vreinterpret_f16_u32(vld1_lane_u32((const void*) ibl1, vreinterpret_u32_f16(vblbr), 1));

        const float16x4_t vldrd = vsub_f16(vblbr, vtltr);

        const float16x4x2_t vld_t = vuzp_f16(vldrd, vldrd);
        const float16x4_t vld = vld_t.val[0];
        const float16x4_t vrd = vld_t.val[1];

        const float16x4x2_t vtl_t = vuzp_f16(vtltr, vtltr);
        const float16x4_t vtl = vtl_t.val[0];
        const float16x4_t vtr = vtl_t.val[1];

        const float16x4_t vl = vfma_f16(vtl, vld, valphav);
        const float16x4_t vr = vfma_f16(vtr, vrd, valphav);

        const float16x4_t vd = vsub_f16(vr, vl);
        const float16x4_t vo = vfma_f16(vl, vd, valphah);

        vst1_lane_u32((void*) o, vreinterpret_u32_f16(vo), 0); o += 2;
      }
      if (p & 1) {
        // We are computing the following formula:
        //   result = (1 - alpha_h) * (1 - alpha_v) * top_left +
        //                 alpha_h  * (1 - alpha_v) * top_right +
        //            (1 - alpha_h) *      alpha_v  * bottom_left +
        //                 alpha_h  *      alpha_v  * bottom_right.
        //
        // Rearranging gives
        //   result =    left + alpha_h * (right        - left),
        // where
        //   left =  top_left + alpha_v * (bottom_left  - top_left),
        //  right = top_right + alpha_v * (bottom_right - top_right).

        const uint16_t* itl = (const uint16_t*) ((uintptr_t) i[0] + input_offset);
        const uint16_t* ibl = (const uint16_t*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        const float16x4_t vw = vreinterpret_f16_u32(vld1_dup_u32((const void*) w)); w += 2;

        const float16x4x2_t vwhv = vuzp_f16(vw, vw);
        const float16x4_t valphah = vwhv.val[0];
        const float16x4_t valphav = vwhv.val[1];

        const float16x4_t vtltr = vreinterpret_f16_u32(vld1_dup_u32((const void*) itl));
        const float16x4_t vblbr = vreinterpret_f16_u32(vld1_dup_u32((const void*) ibl));

        const float16x4_t vldrd = vsub_f16(vblbr, vtltr);

        const float16x4x2_t vld_t = vuzp_f16(vldrd, vldrd);
        const float16x4_t vld = vld_t.val[0];
        const float16x4_t vrd = vld_t.val[1];

        const float16x4x2_t vtl_t = vuzp_f16(vtltr, vtltr);
        const float16x4_t vtl = vtl_t.val[0];
        const float16x4_t vtr = vtl_t.val[1];

        const float16x4_t vl = vfma_f16(vtl, vld, valphav);
        const float16x4_t vr = vfma_f16(vtr, vrd, valphav);

        const float16x4_t vd = vsub_f16(vr, vl);
        const float16x4_t vo = vfma_f16(vl, vd, valphah);

        vst1_lane_u16(o, vreinterpret_u16_f16(vo), 0); o += 1;
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}

void xnn_f16_ibilinear_ukernel__neonfp16arith_c8(
    size_t output_pixels,
    size_t channels,
    const void** restrict input,
    size_t input_offset,
    const void* restrict weights,
    void* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = (const uint16_t*) ((uintptr_t) input[0] + input_offset);
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input[1] + input_offset);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input[2] + input_offset);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const float16x8_t valphah = vreinterpretq_f16_u16(vld1q_dup_u16(weights)); weights = (const uint16_t*) weights + 1;
    const float16x8_t valphav = vreinterpretq_f16_u16(vld1q_dup_u16(weights)); weights = (const uint16_t*) weights + 1;

    size_t c = channels;
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const float16x8_t vtl = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vtr = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vbl = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vbr = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;

      const float16x8_t vtd = vsubq_f16(vtr, vtl);
      const float16x8_t vbd = vsubq_f16(vbr, vbl);

      const float16x8_t vt = vfmaq_f16(vtl, vtd, valphah);
      const float16x8_t vb = vfmaq_f16(vbl, vbd, valphah);

      const float16x8_t vd = vsubq_f16(vb, vt);

      const float16x8_t vo = vfmaq_f16(vt, vd, valphav);

      vst1q_u16(o, vreinterpretq_u16_f16(vo)); o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const float16x8_t vtl = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vtr = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vbl = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vbr = vreinterpretq_f16_u16(vld1q_u16(i3));

      const float16x8_t vtd = vsubq_f16(vtr, vtl);
      const float16x8_t vbd = vsubq_f16(vbr, vbl);

      const float16x8_t vt = vfmaq_f16(vtl, vtd, valphah);
      const float16x8_t vb = vfmaq_f16(vbl, vbd, valphah);

      const float16x8_t vd = vsubq_f16(vb, vt);

      float16x8_t vo = vfmaq_f16(vt, vd, valphav);

      float16x4_t vo_lo = vget_low_f16(vo);
      if (c & (4 * sizeof(uint16_t))) {
        vst1_u16(o, vreinterpret_u16_f16(vo_lo)); o += 4;
        vo_lo = vget_high_f16(vo);
      }
      if (c & (2 * sizeof(uint16_t))) {
        vst1_lane_u32((void*) o, vreinterpret_u32_f16(vo_lo), 0); o += 2;
        vo_lo = vext_f16(vo_lo, vo_lo, 2);
      }
      if (c & (1 * sizeof(uint16_t))) {
        vst1_lane_u16(o, vreinterpret_u16_f16(vo_lo), 0); o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint16_t* c0 = (uint16_t*) c;

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc0x1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 4 * sizeof(uint16_t); k -= 4 * sizeof(uint16_t)) {
        const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;

        const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c0, va0, 0);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c0, va0, 0);
        #endif
        const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c1, va0, 1);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c1, va0, 1);
        #endif
        const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c2, va0, 2);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c2, va0, 2);
        #endif
        const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c3, va0, 3);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c3, va0, 3);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;

          const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
          const float16x8_t vb1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

          vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);
          vacc0x1 = vfmaq_f16(vacc0x1, va0, vb1);

          k -= sizeof(uint16_t);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);


    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);
    vacc0x1 = vmaxq_f16(vacc0x1, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);
    vacc0x1 = vminq_f16(vacc0x1, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vacc0x1));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0)); c0 += 8;

        vacc0x0 = vacc0x1;
      }
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;

        vacc0 = vget_high_f16(vacc0x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;

        vacc0 = vext_f16(vacc0, vacc0, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint16_t* c0 = (uint16_t*) c;

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 4 * sizeof(uint16_t); k -= 4 * sizeof(uint16_t)) {
        const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;

        const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        #endif
        const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        #endif
        const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        #endif
        const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;

          const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

          vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);

          k -= sizeof(uint16_t);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);


    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;

        vacc0 = vget_high_f16(vacc0x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;

        vacc0 = vext_f16(vacc0, vacc0, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint16_t* c0 = (uint16_t*) c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc0x1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc1x0 = vacc0x0;
    float16x8_t vacc1x1 = vacc0x1;
    float16x8_t vacc2x0 = vacc0x0;
    float16x8_t vacc2x1 = vacc0x1;
    float16x8_t vacc3x0 = vacc0x0;
    float16x8_t vacc3x1 = vacc0x1;
    float16x8_t vacc4x0 = vacc0x0;
    float16x8_t vacc4x1 = vacc0x1;
    float16x8_t vacc5x0 = vacc0x0;
    float16x8_t vacc5x1 = vacc0x1;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint16_t* restrict a1 = (const uint16_t*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint16_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint16_t* restrict a2 = (const uint16_t*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint16_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint16_t* restrict a3 = (const uint16_t*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint16_t*) ((uintptr_t) a3 + a_offset);
      }
      const uint16_t* restrict a4 = (const uint16_t*) a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const uint16_t*) ((uintptr_t) a4 + a_offset);
      }
      const uint16_t* restrict a5 = (const uint16_t*) a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const uint16_t*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      for (; k >= 4 * sizeof(uint16_t); k -= 4 * sizeof(uint16_t)) {
        const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;
        const float16x4_t va1 = vreinterpret_f16_u16(vld1_u16(a1)); a1 += 4;
        const float16x4_t va2 = vreinterpret_f16_u16(vld1_u16(a2)); a2 += 4;
        const float16x4_t va3 = vreinterpret_f16_u16(vld1_u16(a3)); a3 += 4;
        const float16x4_t va4 = vreinterpret_f16_u16(vld1_u16(a4)); a4 += 4;
        const float16x4_t va5 = vreinterpret_f16_u16(vld1_u16(a5)); a5 += 4;

        const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c0, va1, 0);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c0, va2, 0);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c0, va3, 0);
          vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c0, va4, 0);
          vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c0, va5, 0);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c0, va0, 0);
          vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c0, va1, 0);
          vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c0, va2, 0);
          vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c0, va3, 0);
          vacc4x1 = vfmaq_lane_f16(vacc4x1, vb1c0, va4, 0);
          vacc5x1 = vfmaq_lane_f16(vacc5x1, vb1c0, va5, 0);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c0, va1, 0);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c0, va2, 0);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c0, va3, 0);
          vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c0, va4, 0);
          vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c0, va5, 0);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c0, va0, 0);
          vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c0, va1, 0);
          vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c0, va2, 0);
          vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c0, va3, 0);
          vacc4x1 = vmlaq_lane_f16(vacc4x1, vb1c0, va4, 0);
          vacc5x1 = vmlaq_lane_f16(vacc5x1, vb1c0, va5, 0);
        #endif
        const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c1, va1, 1);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c1, va2, 1);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c1, va3, 1);
          vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c1, va4, 1);
          vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c1, va5, 1);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c1, va0, 1);
          vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c1, va1, 1);
          vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c1, va2, 1);
          vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c1, va3, 1);
          vacc4x1 = vfmaq_lane_f16(vacc4x1, vb1c1, va4, 1);
          vacc5x1 = vfmaq_lane_f16(vacc5x1, vb1c1, va5, 1);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c1, va1, 1);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c1, va2, 1);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c1, va3, 1);
          vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c1, va4, 1);
          vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c1, va5, 1);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c1, va0, 1);
          vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c1, va1, 1);
          vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c1, va2, 1);
          vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c1, va3, 1);
          vacc4x1 = vmlaq_lane_f16(vacc4x1, vb1c1, va4, 1);
          vacc5x1 = vmlaq_lane_f16(vacc5x1, vb1c1, va5, 1);
        #endif
        const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c2, va1, 2);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c2, va2, 2);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c2, va3, 2);
          vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c2, va4, 2);
          vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c2, va5, 2);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c2, va0, 2);
          vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c2, va1, 2);
          vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c2, va2, 2);
          vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c2, va3, 2);
          vacc4x1 = vfmaq_lane_f16(vacc4x1, vb1c2, va4, 2);
          vacc5x1 = vfmaq_lane_f16(vacc5x1, vb1c2, va5, 2);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c2, va1, 2);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c2, va2, 2);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c2, va3, 2);
          vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c2, va4, 2);
          vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c2, va5, 2);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c2, va0, 2);
          vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c2, va1, 2);
          vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c2, va2, 2);
          vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c2, va3, 2);
          vacc4x1 = vmlaq_lane_f16(vacc4x1, vb1c2, va4, 2);
          vacc5x1 = vmlaq_lane_f16(vacc5x1, vb1c2, va5, 2);
        #endif
        const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
        const float16x8_t vb1c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c3, va1, 3);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c3, va2, 3);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c3, va3, 3);
          vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c3, va4, 3);
          vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c3, va5, 3);
          vacc0x1 = vfmaq_lane_f16(vacc0x1, vb1c3, va0, 3);
          vacc1x1 = vfmaq_lane_f16(vacc1x1, vb1c3, va1, 3);
          vacc2x1 = vfmaq_lane_f16(vacc2x1, vb1c3, va2, 3);
          vacc3x1 = vfmaq_lane_f16(vacc3x1, vb1c3, va3, 3);
          vacc4x1 = vfmaq_lane_f16(vacc4x1, vb1c3, va4, 3);
          vacc5x1 = vfmaq_lane_f16(vacc5x1, vb1c3, va5, 3);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c3, va1, 3);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c3, va2, 3);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c3, va3, 3);
          vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c3, va4, 3);
          vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c3, va5, 3);
          vacc0x1 = vmlaq_lane_f16(vacc0x1, vb1c3, va0, 3);
          vacc1x1 = vmlaq_lane_f16(vacc1x1, vb1c3, va1, 3);
          vacc2x1 = vmlaq_lane_f16(vacc2x1, vb1c3, va2, 3);
          vacc3x1 = vmlaq_lane_f16(vacc3x1, vb1c3, va3, 3);
          vacc4x1 = vmlaq_lane_f16(vacc4x1, vb1c3, va4, 3);
          vacc5x1 = vmlaq_lane_f16(vacc5x1, vb1c3, va5, 3);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;
          const float16x8_t va1 = vreinterpretq_f16_u16(vld1q_dup_u16(a1)); a1 += 1;
          const float16x8_t va2 = vreinterpretq_f16_u16(vld1q_dup_u16(a2)); a2 += 1;
          const float16x8_t va3 = vreinterpretq_f16_u16(vld1q_dup_u16(a3)); a3 += 1;
          const float16x8_t va4 = vreinterpretq_f16_u16(vld1q_dup_u16(a4)); a4 += 1;
          const float16x8_t va5 = vreinterpretq_f16_u16(vld1q_dup_u16(a5)); a5 += 1;

          const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
          const float16x8_t vb1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

          vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);
          vacc1x0 = vfmaq_f16(vacc1x0, va1, vb0);
          vacc2x0 = vfmaq_f16(vacc2x0, va2, vb0);
          vacc3x0 = vfmaq_f16(vacc3x0, va3, vb0);
          vacc4x0 = vfmaq_f16(vacc4x0, va4, vb0);
          vacc5x0 = vfmaq_f16(vacc5x0, va5, vb0);
          vacc0x1 = vfmaq_f16(vacc0x1, va0, vb1);
          vacc1x1 = vfmaq_f16(vacc1x1, va1, vb1);
          vacc2x1 = vfmaq_f16(vacc2x1, va2, vb1);
          vacc3x1 = vfmaq_f16(vacc3x1, va3, vb1);
          vacc4x1 = vfmaq_f16(vacc4x1, va4, vb1);
          vacc5x1 = vfmaq_f16(vacc5x1, va5, vb1);

          k -= sizeof(uint16_t);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);


    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);
    vacc1x0 = vmaxq_f16(vacc1x0, vmin);
    vacc2x0 = vmaxq_f16(vacc2x0, vmin);
    vacc3x0 = vmaxq_f16(vacc3x0, vmin);
    vacc4x0 = vmaxq_f16(vacc4x0, vmin);
    vacc5x0 = vmaxq_f16(vacc5x0, vmin);
    vacc0x1 = vmaxq_f16(vacc0x1, vmin);
    vacc1x1 = vmaxq_f16(vacc1x1, vmin);
    vacc2x1 = vmaxq_f16(vacc2x1, vmin);
    vacc3x1 = vmaxq_f16(vacc3x1, vmin);
    vacc4x1 = vmaxq_f16(vacc4x1, vmin);
    vacc5x1 = vmaxq_f16(vacc5x1, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);
    vacc1x0 = vminq_f16(vacc1x0, vmax);
    vacc2x0 = vminq_f16(vacc2x0, vmax);
    vacc3x0 = vminq_f16(vacc3x0, vmax);
    vacc4x0 = vminq_f16(vacc4x0, vmax);
    vacc5x0 = vminq_f16(vacc5x0, vmax);
    vacc0x1 = vminq_f16(vacc0x1, vmax);
    vacc1x1 = vminq_f16(vacc1x1, vmax);
    vacc2x1 = vminq_f16(vacc2x1, vmax);
    vacc3x1 = vminq_f16(vacc3x1, vmax);
    vacc4x1 = vminq_f16(vacc4x1, vmax);
    vacc5x1 = vminq_f16(vacc5x1, vmax);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x0));
      vst1q_u16(c5 + 8, vreinterpretq_u16_f16(vacc5x1));
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x0));
      vst1q_u16(c4 + 8, vreinterpretq_u16_f16(vacc4x1));
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vacc3x1));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vacc2x1));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vacc1x1));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vacc0x1));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x0)); c5 += 8;
        vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x0)); c4 += 8;
        vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0)); c3 += 8;
        vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0)); c2 += 8;
        vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0)); c1 += 8;
        vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0)); c0 += 8;

        vacc5x0 = vacc5x1;
        vacc4x0 = vacc4x1;
        vacc3x0 = vacc3x1;
        vacc2x0 = vacc2x1;
        vacc1x0 = vacc1x1;
        vacc0x0 = vacc0x1;
      }
      float16x4_t vacc5 = vget_low_f16(vacc5x0);
      float16x4_t vacc4 = vget_low_f16(vacc4x0);
      float16x4_t vacc3 = vget_low_f16(vacc3x0);
      float16x4_t vacc2 = vget_low_f16(vacc2x0);
      float16x4_t vacc1 = vget_low_f16(vacc1x0);
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      if (nc & 4) {
        vst1_u16(c5, vreinterpret_u16_f16(vacc5)); c5 += 4;
        vst1_u16(c4, vreinterpret_u16_f16(vacc4)); c4 += 4;
        vst1_u16(c3, vreinterpret_u16_f16(vacc3)); c3 += 4;
        vst1_u16(c2, vreinterpret_u16_f16(vacc2)); c2 += 4;
        vst1_u16(c1, vreinterpret_u16_f16(vacc1)); c1 += 4;
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;

        vacc5 = vget_high_f16(vacc5x0);
        vacc4 = vget_high_f16(vacc4x0);
        vacc3 = vget_high_f16(vacc3x0);
        vacc2 = vget_high_f16(vacc2x0);
        vacc1 = vget_high_f16(vacc1x0);
        vacc0 = vget_high_f16(vacc0x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vacc5), 0); c5 += 2;
        vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vacc4), 0); c4 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vacc3), 0); c3 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vacc2), 0); c2 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vacc1), 0); c1 += 2;
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;

        vacc5 = vext_f16(vacc5, vacc5, 2);
        vacc4 = vext_f16(vacc4, vacc4, 2);
        vacc3 = vext_f16(vacc3, vacc3, 2);
        vacc2 = vext_f16(vacc2, vacc2, 2);
        vacc1 = vext_f16(vacc1, vacc1, 2);
        vacc0 = vext_f16(vacc0, vacc0, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c5, vreinterpret_u16_f16(vacc5), 0);
        vst1_lane_u16(c4, vreinterpret_u16_f16(vacc4), 0);
        vst1_lane_u16(c3, vreinterpret_u16_f16(vacc3), 0);
        vst1_lane_u16(c2, vreinterpret_u16_f16(vacc2), 0);
        vst1_lane_u16(c1, vreinterpret_u16_f16(vacc1), 0);
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint16_t* c0 = (uint16_t*) c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;
    float16x8_t vacc1x0 = vacc0x0;
    float16x8_t vacc2x0 = vacc0x0;
    float16x8_t vacc3x0 = vacc0x0;
    float16x8_t vacc4x0 = vacc0x0;
    float16x8_t vacc5x0 = vacc0x0;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint16_t* restrict a1 = (const uint16_t*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint16_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint16_t* restrict a2 = (const uint16_t*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint16_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint16_t* restrict a3 = (const uint16_t*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint16_t*) ((uintptr_t) a3 + a_offset);
      }
      const uint16_t* restrict a4 = (const uint16_t*) a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const uint16_t*) ((uintptr_t) a4 + a_offset);
      }
      const uint16_t* restrict a5 = (const uint16_t*) a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const uint16_t*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      for (; k >= 4 * sizeof(uint16_t); k -= 4 * sizeof(uint16_t)) {
        const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;
        const float16x4_t va1 = vreinterpret_f16_u16(vld1_u16(a1)); a1 += 4;
        const float16x4_t va2 = vreinterpret_f16_u16(vld1_u16(a2)); a2 += 4;
        const float16x4_t va3 = vreinterpret_f16_u16(vld1_u16(a3)); a3 += 4;
        const float16x4_t va4 = vreinterpret_f16_u16(vld1_u16(a4)); a4 += 4;
        const float16x4_t va5 = vreinterpret_f16_u16(vld1_u16(a5)); a5 += 4;

        const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c0, va1, 0);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c0, va2, 0);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c0, va3, 0);
          vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c0, va4, 0);
          vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c0, va5, 0);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c0, va1, 0);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c0, va2, 0);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c0, va3, 0);
          vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c0, va4, 0);
          vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c0, va5, 0);
        #endif
        const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c1, va1, 1);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c1, va2, 1);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c1, va3, 1);
          vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c1, va4, 1);
          vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c1, va5, 1);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c1, va1, 1);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c1, va2, 1);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c1, va3, 1);
          vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c1, va4, 1);
          vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c1, va5, 1);
        #endif
        const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c2, va1, 2);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c2, va2, 2);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c2, va3, 2);
          vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c2, va4, 2);
          vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c2, va5, 2);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c2, va1, 2);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c2, va2, 2);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c2, va3, 2);
          vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c2, va4, 2);
          vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c2, va5, 2);
        #endif
        const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        #if XNN_ARCH_ARM64
          vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
          vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c3, va1, 3);
          vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c3, va2, 3);
          vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c3, va3, 3);
          vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c3, va4, 3);
          vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c3, va5, 3);
        #else
          vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
          vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c3, va1, 3);
          vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c3, va2, 3);
          vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c3, va3, 3);
          vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c3, va4, 3);
          vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c3, va5, 3);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;
          const float16x8_t va1 = vreinterpretq_f16_u16(vld1q_dup_u16(a1)); a1 += 1;
          const float16x8_t va2 = vreinterpretq_f16_u16(vld1q_dup_u16(a2)); a2 += 1;
          const float16x8_t va3 = vreinterpretq_f16_u16(vld1q_dup_u16(a3)); a3 += 1;
          const float16x8_t va4 = vreinterpretq_f16_u16(vld1q_dup_u16(a4)); a4 += 1;
          const float16x8_t va5 = vreinterpretq_f16_u16(vld1q_dup_u16(a5)); a5 += 1;

          const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

          vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);
          vacc1x0 = vfmaq_f16(vacc1x0, va1, vb0);
          vacc2x0 = vfmaq_f16(vacc2x0, va2, vb0);
          vacc3x0 = vfmaq_f16(vacc3x0, va3, vb0);
          vacc4x0 = vfmaq_f16(vacc4x0, va4, vb0);
          vacc5x0 = vfmaq_f16(vacc5x0, va5, vb0);

          k -= sizeof(uint16_t);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);


    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);
    vacc1x0 = vmaxq_f16(vacc1x0, vmin);
    vacc2x0 = vmaxq_f16(vacc2x0, vmin);
    vacc3x0 = vmaxq_f16(vacc3x0, vmin);
    vacc4x0 = vmaxq_f16(vacc4x0, vmin);
    vacc5x0 = vmaxq_f16(vacc5x0, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);
    vacc1x0 = vminq_f16(vacc1x0, vmax);
    vacc2x0 = vminq_f16(vacc2x0, vmax);
    vacc3x0 = vminq_f16(vacc3x0, vmax);
    vacc4x0 = vminq_f16(vacc4x0, vmax);
    vacc5x0 = vminq_f16(vacc5x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x0));
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x0));
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      float16x4_t vacc5 = vget_low_f16(vacc5x0);
      float16x4_t vacc4 = vget_low_f16(vacc4x0);
      float16x4_t vacc3 = vget_low_f16(vacc3x0);
      float16x4_t vacc2 = vget_low_f16(vacc2x0);
      float16x4_t vacc1 = vget_low_f16(vacc1x0);
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      if (nc & 4) {
        vst1_u16(c5, vreinterpret_u16_f16(vacc5)); c5 += 4;
        vst1_u16(c4, vreinterpret_u16_f16(vacc4)); c4 += 4;
        vst1_u16(c3, vreinterpret_u16_f16(vacc3)); c3 += 4;
        vst1_u16(c2, vreinterpret_u16_f16(vacc2)); c2 += 4;
        vst1_u16(c1, vreinterpret_u16_f16(vacc1)); c1 += 4;
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;

        vacc5 = vget_high_f16(vacc5x0);
        vacc4 = vget_high_f16(vacc4x0);
        vacc3 = vget_high_f16(vacc3x0);
        vacc2 = vget_high_f16(vacc2x0);
        vacc1 = vget_high_f16(vacc1x0);
        vacc0 = vget_high_f16(vacc0x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vacc5), 0); c5 += 2;
        vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vacc4), 0); c4 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vacc3), 0); c3 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vacc2), 0); c2 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vacc1), 0); c1 += 2;
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;

        vacc5 = vext_f16(vacc5, vacc5, 2);
        vacc4 = vext_f16(vacc4, vacc4, 2);
        vacc3 = vext_f16(vacc3, vacc3, 2);
        vacc2 = vext_f16(vacc2, vacc2, 2);
        vacc1 = vext_f16(vacc1, vacc1, 2);
        vacc0 = vext_f16(vacc0, vacc0, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c5, vreinterpret_u16_f16(vacc5), 0);
        vst1_lane_u16(c4, vreinterpret_u16_f16(vacc4), 0);
        vst1_lane_u16(c3, vreinterpret_u16_f16(vacc3), 0);
        vst1_lane_u16(c2, vreinterpret_u16_f16(vacc2), 0);
        vst1_lane_u16(c1, vreinterpret_u16_f16(vacc1), 0);
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    uint16_t* o = output;
    {
      const uint16_t* i0 = *input++;
      const uint16_t* i1 = *input++;
      const uint16_t* i2 = *input++;
      const uint16_t* i3 = *input++;
      const uint16_t* i4 = *input++;
      const uint16_t* i5 = *input++;
      const uint16_t* i6 = *input++;
      const uint16_t* i7 = *input++;
      const uint16_t* i8 = *input++;
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;

        const float16x8_t vmax018 = vmaxq_f16(vmaxq_f16(vi0, vi1), vi8);
        const float16x8_t vmax23 = vmaxq_f16(vi2, vi3);
        const float16x8_t vmax45 = vmaxq_f16(vi4, vi5);
        const float16x8_t vmax67 = vmaxq_f16(vi6, vi7);

        const float16x8_t vmax2345 = vmaxq_f16(vmax23, vmax45);
        const float16x8_t vmax01678 = vmaxq_f16(vmax018, vmax67);
        const float16x8_t vmax = vmaxq_f16(vmax2345, vmax01678);
        const float16x8_t vout = vmaxq_f16(vminq_f16(vmax, voutput_max), voutput_min);

        vst1q_u16(o, vreinterpretq_u16_f16(vout)); o += 8;
      }
      if (c != 0) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;

        const float16x8_t vmax018 = vmaxq_f16(vmaxq_f16(vi0, vi1), vi8);
        const float16x8_t vmax23 = vmaxq_f16(vi2, vi3);
        const float16x8_t vmax45 = vmaxq_f16(vi4, vi5);
        const float16x8_t vmax67 = vmaxq_f16(vi6, vi7);

        const float16x8_t vmax2345 = vmaxq_f16(vmax23, vmax45);
        const float16x8_t vmax01678 = vmaxq_f16(vmax018, vmax67);
        const float16x8_t vmax = vmaxq_f16(vmax2345, vmax01678);
        float16x8_t vout = vmaxq_f16(vminq_f16(vmax, voutput_max), voutput_min);

        float16x4_t vout_lo = vget_low_f16(vout);
        if (c & 4) {
          vst1_u16(o, vreinterpret_u16_f16(vout_lo)); o += 4;
          vout_lo = vget_high_f16(vout);
        }
        if (c & 2) {
          vst1_lane_u32((void*) o, vreinterpret_u32_f16(vout_lo), 0); o += 2;
          vout_lo = vext_f16(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_u16(o, vreinterpret_u16_f16(vout_lo), 0); o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const uint16_t* i0 = *input++;
      const uint16_t* i1 = *input++;
      const uint16_t* i2 = *input++;
      const uint16_t* i3 = *input++;
      const uint16_t* i4 = *input++;
      const uint16_t* i5 = *input++;
      const uint16_t* i6 = *input++;
      const uint16_t* i7 = *input++;
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vo = vreinterpretq_f16_u16(vld1q_u16(o));

        const float16x8_t vmax01 = vmaxq_f16(vmaxq_f16(vi0, vi1), vo);
        const float16x8_t vmax23 = vmaxq_f16(vi2, vi3);
        const float16x8_t vmax45 = vmaxq_f16(vi4, vi5);
        const float16x8_t vmax67 = vmaxq_f16(vi6, vi7);

        const float16x8_t vmax2345 = vmaxq_f16(vmax23, vmax45);
        const float16x8_t vmax0167 = vmaxq_f16(vmax01, vmax67);
        const float16x8_t vmax = vmaxq_f16(vmax2345, vmax0167);
        const float16x8_t vout = vmaxq_f16(vminq_f16(vmax, voutput_max), voutput_min);

        vst1q_u16(o, vreinterpretq_u16_f16(vout)); o += 8;
      }
      if (c != 0) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0));
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1));
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2));
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3));
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4));
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5));
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6));
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7));
        const float16x8_t vo = vreinterpretq_f16_u16(vld1q_u16(o));

        const float16x8_t vmax01 = vmaxq_f16(vmaxq_f16(vi0, vi1), vo);
        const float16x8_t vmax23 = vmaxq_f16(vi2, vi3);
        const float16x8_t vmax45 = vmaxq_f16(vi4, vi5);
        const float16x8_t vmax67 = vmaxq_f16(vi6, vi7);

        const float16x8_t vmax2345 = vmaxq_f16(vmax23, vmax45);
        const float16x8_t vmax0167 = vmaxq_f16(vmax01, vmax67);
        const float16x8_t vmax = vmaxq_f16(vmax2345, vmax0167);
        float16x8_t vout = vmaxq_f16(vminq_f16(vmax, voutput_max), voutput_min);

        float16x4_t vout_lo = vget_low_f16(vout);
        if (c & 4) {
          vst1_u16(o, vreinterpret_u16_f16(vout_lo)); o += 4;
          vout_lo = vget_high_f16(vout);
        }
        if (c & 2) {
          vst1_lane_u32((void*) o, vreinterpret_u32_f16(vout_lo), 0); o += 2;
          vout_lo = vext_f16(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_u16(o, vreinterpret_u16_f16(vout_lo), 0); o += 1;
        }
      }
    }
    input = (const void**) ((uintptr_t) input + input_increment);
    output = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_pavgpool_minmax_ukernel_9p8x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  do {
    {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint16_t* i8 = (const uint16_t*) *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

        vst1q_u16(b, vreinterpretq_u16_f16(vsum)); b += 8;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b));

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        vst1q_u16(b, vreinterpretq_u16_f16(vsum)); b += 8;
      }
    }

    {
      const uint16_t* i0 = (const uint16_t*) input[0];
      assert(i0 != NULL);
      const uint16_t* i1 = (const uint16_t*) input[1];
      const uint16_t* i2 = (const uint16_t*) input[2];
      const uint16_t* i3 = (const uint16_t*) input[3];
      const uint16_t* i4 = (const uint16_t*) input[4];
      const uint16_t* i5 = (const uint16_t*) input[5];
      const uint16_t* i6 = (const uint16_t*) input[6];
      const uint16_t* i7 = (const uint16_t*) input[7];
      input = (const void**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = (const uint16_t*) zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = (const uint16_t*) zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = (const uint16_t*) zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = (const uint16_t*) zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = (const uint16_t*) zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = (const uint16_t*) zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = (const uint16_t*) zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      const float16x8_t vmultiplier = vreinterpretq_f16_u16(vld1q_dup_u16(multiplier)); multiplier = (const uint16_t*) multiplier + 1;

      size_t c = channels;
      const uint16_t* b = (const uint16_t*) buffer;
      while (c >= 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        float16x8_t vout = vmulq_f16(vsum, vmultiplier);
        vout = vmaxq_f16(vout, voutput_min);
        vout = vminq_f16(vout, voutput_max);

        vst1q_u16(output, vreinterpretq_u16_f16(vout)); output = (uint16_t*) output + 8;

        c -= 8;
      }
      if (c != 0) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0));
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1));
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2));
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3));
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4));
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5));
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6));
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7));
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b));

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        float16x8_t vout = vmulq_f16(vsum, vmultiplier);
        vout = vmaxq_f16(vout, voutput_min);
        vout = vminq_f16(vout, voutput_max);

        float16x4_t vout_lo = vget_low_f16(vout);
        if (c & 4) {
          vst1_u16(output, vreinterpret_u16_f16(vout_lo)); output = (uint16_t*) output + 4;
          vout_lo = vget_high_f16(vout);
        }
        if (c & 2) {
          vst1_lane_u32(output, vreinterpret_u32_f16(vout_lo), 0); output = (uint16_t*) output + 2;
          vout_lo = vext_f16(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_u16(output, vreinterpret_u16_f16(vout_lo), 0); output = (uint16_t*) output + 1;
        }
      }
    }
    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_pavgpool_minmax_ukernel_9x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    const uint16_t* i1 = (const uint16_t*) input[1];
    const uint16_t* i2 = (const uint16_t*) input[2];
    const uint16_t* i3 = (const uint16_t*) input[3];
    const uint16_t* i4 = (const uint16_t*) input[4];
    const uint16_t* i5 = (const uint16_t*) input[5];
    const uint16_t* i6 = (const uint16_t*) input[6];
    const uint16_t* i7 = (const uint16_t*) input[7];
    const uint16_t* i8 = (const uint16_t*) input[8];
    input = (const void**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = (const uint16_t*) zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = (const uint16_t*) zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = (const uint16_t*) zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = (const uint16_t*) zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = (const uint16_t*) zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = (const uint16_t*) zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = (const uint16_t*) zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = (const uint16_t*) zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }

    const float16x8_t vmultiplier = vreinterpretq_f16_u16(vld1q_dup_u16(multiplier)); multiplier = (const uint16_t*) multiplier + 1;

    size_t c = channels;
    while (c >= 8) {
      const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;

      const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
      const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
      const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
      const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
      const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
      const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
      const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
      const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

      float16x8_t vout = vmulq_f16(vsum, vmultiplier);
      vout = vmaxq_f16(vout, voutput_min);
      vout = vminq_f16(vout, voutput_max);

      vst1q_u16(output, vreinterpretq_u16_f16(vout)); output = (uint16_t*) output + 8;

      c -= 8;
    }
    if (c != 0) {
      const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3));
      const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4));
      const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5));
      const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6));
      const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7));
      const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8));

      const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
      const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
      const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
      const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
      const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
      const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
      const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
      const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

      float16x8_t vout = vmulq_f16(vsum, vmultiplier);
      vout = vmaxq_f16(vout, voutput_min);
      vout = vminq_f16(vout, voutput_max);

      float16x4_t vout_lo = vget_low_f16(vout);
      if (c & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vout_lo)); output = (uint16_t*) output + 4;
        vout_lo = vget_high_f16(vout);
      }
      if (c & 2) {
        vst1_lane_u32(output, vreinterpret_u32_f16(vout_lo), 0); output = (uint16_t*) output + 2;
        vout_lo = vext_f16(vout_lo, vout_lo, 2);
      }
      if (c & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vout_lo), 0); output = (uint16_t*) output + 1;
      }
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_prelu_ukernel__neonfp16arith_2x16(
    size_t rows,
    size_t channels,
    const void* restrict input,
    size_t input_stride,
    const void* restrict weights,
    void* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  const uint16_t* i0 = (const uint16_t*) input;
  uint16_t* o0 = (uint16_t*) output;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const uint16_t* w = (const uint16_t*) weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(uint16_t); c -= 16 * sizeof(uint16_t)) {
      const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vw89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;

      const float16x8_t vi0x001234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi0x089ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1x001234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi1x089ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;

      float16x8_t vacc0x001234567 = vmulq_f16(vi0x001234567, vw01234567);
      const uint16x8_t vm0x001234567 = vcltq_s16(vreinterpretq_s16_f16(vi0x001234567), vmovq_n_s16(0));
      float16x8_t vacc0x089ABCDEF = vmulq_f16(vi0x089ABCDEF, vw89ABCDEF);
      const uint16x8_t vm0x089ABCDEF = vcltq_s16(vreinterpretq_s16_f16(vi0x089ABCDEF), vmovq_n_s16(0));
      float16x8_t vacc1x001234567 = vmulq_f16(vi1x001234567, vw01234567);
      const uint16x8_t vm1x001234567 = vcltq_s16(vreinterpretq_s16_f16(vi1x001234567), vmovq_n_s16(0));
      float16x8_t vacc1x089ABCDEF = vmulq_f16(vi1x089ABCDEF, vw89ABCDEF);
      const uint16x8_t vm1x089ABCDEF = vcltq_s16(vreinterpretq_s16_f16(vi1x089ABCDEF), vmovq_n_s16(0));

      vacc0x001234567 = vbslq_f16(vm0x001234567, vacc0x001234567, vi0x001234567);
      vacc0x089ABCDEF = vbslq_f16(vm0x089ABCDEF, vacc0x089ABCDEF, vi0x089ABCDEF);
      vacc1x001234567 = vbslq_f16(vm1x001234567, vacc1x001234567, vi1x001234567);
      vacc1x089ABCDEF = vbslq_f16(vm1x089ABCDEF, vacc1x089ABCDEF, vi1x089ABCDEF);

      vst1q_u16(o0, vreinterpretq_u16_f16(vacc0x001234567)); o0 += 8;
      vst1q_u16(o0, vreinterpretq_u16_f16(vacc0x089ABCDEF)); o0 += 8;
      vst1q_u16(o1, vreinterpretq_u16_f16(vacc1x001234567)); o1 += 8;
      vst1q_u16(o1, vreinterpretq_u16_f16(vacc1x089ABCDEF)); o1 += 8;
    }
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;

      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      i0 += 8;
      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      i1 += 8;

      float16x8_t vacc0x01234567 = vmulq_f16(vi0x01234567, vw01234567);
      const uint16x8_t vm0x01234567 = vcltq_s16(vreinterpretq_s16_f16(vi0x01234567), vmovq_n_s16(0));
      float16x8_t vacc1x01234567 = vmulq_f16(vi1x01234567, vw01234567);
      const uint16x8_t vm1x01234567 = vcltq_s16(vreinterpretq_s16_f16(vi1x01234567), vmovq_n_s16(0));

      vacc0x01234567 = vbslq_f16(vm0x01234567, vacc0x01234567, vi0x01234567);
      vacc1x01234567 = vbslq_f16(vm1x01234567, vacc1x01234567, vi1x01234567);

      vst1q_u16(o0, vreinterpretq_u16_f16(vacc0x01234567)); o0 += 8;
      vst1q_u16(o1, vreinterpretq_u16_f16(vacc1x01234567)); o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w));

      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      i0 = (const uint16_t*) ((uintptr_t) i0 + c);
      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      i1 = (const uint16_t*) ((uintptr_t) i1 + c);

      float16x8_t vacc0x01234567 = vmulq_f16(vi0x01234567, vw01234567);
      const uint16x8_t vm0x01234567 = vcltq_s16(vreinterpretq_s16_f16(vi0x01234567), vmovq_n_s16(0));
      float16x8_t vacc1x01234567 = vmulq_f16(vi1x01234567, vw01234567);
      const uint16x8_t vm1x01234567 = vcltq_s16(vreinterpretq_s16_f16(vi1x01234567), vmovq_n_s16(0));

      vacc0x01234567 = vbslq_f16(vm0x01234567, vacc0x01234567, vi0x01234567);
      vacc1x01234567 = vbslq_f16(vm1x01234567, vacc1x01234567, vi1x01234567);

      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      float16x4_t vacc1x0123 = vget_low_f16(vacc1x01234567);
      if (c & (4 * sizeof(uint16_t))) {
        vst1_u16(o0, vreinterpret_u16_f16(vacc0x0123)); o0 += 4;
        vst1_u16(o1, vreinterpret_u16_f16(vacc1x0123)); o1 += 4;

        vacc0x0123 = vget_high_f16(vacc0x01234567);
        vacc1x0123 = vget_high_f16(vacc1x01234567);
      }
      if (c & (2 * sizeof(uint16_t))) {
        vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vacc0x0123), 0); o0 += 2;
        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
        vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vacc1x0123), 0); o1 += 2;
        vacc1x0123 = vext_f16(vacc1x0123, vacc1x0123, 2);
      }
      if (c & (1 * sizeof(uint16_t))) {
        vst1_lane_u16(o0, vreinterpret_u16_f16(vacc0x0123), 0); o0 += 1;
        vst1_lane_u16(o1, vreinterpret_u16_f16(vacc1x0123), 0); o1 += 1;
      }
    }
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32(
    size_t batch,
    const void* input,
    int8_t* output,
    const union xnn_f16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith.scale));
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neonfp16arith.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->neonfp16arith.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->neonfp16arith.output_max);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx8 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx16 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx24 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx0 = vmulq_f16(vx0, vscale);
    vx8 = vmulq_f16(vx8, vscale);
    vx16 = vmulq_f16(vx16, vscale);
    vx24 = vmulq_f16(vx24, vscale);

    int16x8_t vacc0 = vcvtnq_s16_f16(vx0);
    int16x8_t vacc8 = vcvtnq_s16_f16(vx8);
    int16x8_t vacc16 = vcvtnq_s16_f16(vx16);
    int16x8_t vacc24 = vcvtnq_s16_f16(vx24);

    vacc0 = vqaddq_s16(vacc0, voutput_zero_point);
    vacc8 = vqaddq_s16(vacc8, voutput_zero_point);
    vacc16 = vqaddq_s16(vacc16, voutput_zero_point);
    vacc24 = vqaddq_s16(vacc24, voutput_zero_point);

    int8x16_t vy0 = vcombine_s8(vqmovn_s16(vacc0), vqmovn_s16(vacc8));
    int8x16_t vy16 = vcombine_s8(vqmovn_s16(vacc16), vqmovn_s16(vacc24));

    vy0 = vmaxq_s8(vy0, voutput_min);
    vy16 = vmaxq_s8(vy16, voutput_min);

    vy0 = vminq_s8(vy0, voutput_max);
    vy16 = vminq_s8(vy16, voutput_max);

    vst1q_s8(output, vy0); output += 16;
    vst1q_s8(output, vy16); output += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);

    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, vget_low_s8(voutput_min));
    vy = vmin_s8(vy, vget_low_s8(voutput_max));
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, vget_low_s8(voutput_min));
    vy = vmin_s8(vy, vget_low_s8(voutput_max));

    if (batch & (4 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}

void xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32(
    size_t batch,
    const void* input,
    const void* max,
    void* output,
    void* sum,
    const union xnn_f16_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float16x8_t vlog2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3DC5)));  // 0x1.714p+0h
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vminus_ln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xB98C)));  // -0x1.630p-1h
  const float16x8_t vminus_ln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x0AF4)));  // 0x1.BD0p-13h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x37F9)));  // 0x1.FE4p-2h
  const float16x8_t vc1 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C0E)));  // 0x1.038p+0h
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float16x8_t vi_max = vreinterpretq_f16_u16(vld1q_dup_u16(max));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float16x8_t vacc0 = vreinterpretq_f16_u16(vmovq_n_u16(0));
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx0 = vsubq_f16(vi0, vi_max);
    const float16x8_t vx1 = vsubq_f16(vi1, vi_max);
    const float16x8_t vx2 = vsubq_f16(vi2, vi_max);
    const float16x8_t vx3 = vsubq_f16(vi3, vi_max);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vx0, vlog2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vx1, vlog2e);
    float16x8_t vn2 = vfmaq_f16(vmagic_bias, vx2, vlog2e);
    float16x8_t vn3 = vfmaq_f16(vmagic_bias, vx3, vlog2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    const float16x8_t vs2 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn2), 10));
    const float16x8_t vs3 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn3), 10));

    vn0 = vsubq_f16(vn0, vmagic_bias);
    vn1 = vsubq_f16(vn1, vmagic_bias);
    vn2 = vsubq_f16(vn2, vmagic_bias);
    vn3 = vsubq_f16(vn3, vmagic_bias);

    float16x8_t vt0 = vfmaq_f16(vx0, vn0, vminus_ln2_hi);
    float16x8_t vt1 = vfmaq_f16(vx1, vn1, vminus_ln2_hi);
    float16x8_t vt2 = vfmaq_f16(vx2, vn2, vminus_ln2_hi);
    float16x8_t vt3 = vfmaq_f16(vx3, vn3, vminus_ln2_hi);

    vt0 = vfmaq_f16(vt0, vn0, vminus_ln2_lo);
    vt1 = vfmaq_f16(vt1, vn1, vminus_ln2_lo);
    vt2 = vfmaq_f16(vt2, vn2, vminus_ln2_lo);
    vt3 = vfmaq_f16(vt3, vn3, vminus_ln2_lo);

    const float16x8_t vp0 = vfmaq_f16(vc1, vc2, vt0);
    const float16x8_t vp1 = vfmaq_f16(vc1, vc2, vt1);
    const float16x8_t vp2 = vfmaq_f16(vc1, vc2, vt2);
    const float16x8_t vp3 = vfmaq_f16(vc1, vc2, vt3);

    vt0 = vmulq_f16(vt0, vs0);
    vt1 = vmulq_f16(vt1, vs1);
    vt2 = vmulq_f16(vt2, vs2);
    vt3 = vmulq_f16(vt3, vs3);

    float16x8_t vf0 = vfmaq_f16(vs0, vp0, vt0);
    const uint16x8_t vm0 = vcltq_f16(vx0, vdenorm_cutoff);
    float16x8_t vf1 = vfmaq_f16(vs1, vp1, vt1);
    const uint16x8_t vm1 = vcltq_f16(vx1, vdenorm_cutoff);
    float16x8_t vf2 = vfmaq_f16(vs2, vp2, vt2);
    const uint16x8_t vm2 = vcltq_f16(vx2, vdenorm_cutoff);
    float16x8_t vf3 = vfmaq_f16(vs3, vp3, vt3);
    const uint16x8_t vm3 = vcltq_f16(vx3, vdenorm_cutoff);

    vf0 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf0), vm0));
    vf1 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf1), vm1));
    vf2 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf2), vm2));
    vf3 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf3), vm3));

    vst1q_u16(o, vreinterpretq_u16_f16(vf0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf1)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf2)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf3)); o += 8;

    vacc0 = vaddq_f16(vacc0, vf0);
    vacc0 = vaddq_f16(vacc0, vf1);
    vacc0 = vaddq_f16(vacc0, vf2);
    vacc0 = vaddq_f16(vacc0, vf3);
  }

  float16x8_t vacc = vacc0;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx = vsubq_f16(vi, vi_max);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f16(vt, vn, vminus_ln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);

    float16x8_t vf = vfmaq_f16(vs, vp, vt);
    const uint16x8_t vm = vcltq_f16(vx, vdenorm_cutoff);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vm));

    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;

    vacc = vaddq_f16(vacc, vf);
  }
  float16x4_t vacc_lo = vadd_f16(vget_low_f16(vacc), vget_high_f16(vacc));
  if (batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i));

    const float16x8_t vx = vsubq_f16(vi, vi_max);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f16(vt, vn, vminus_ln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);

    float16x8_t vf = vfmaq_f16(vs, vp, vt);
    const uint16x8_t vm = vcltq_f16(vx, vdenorm_cutoff);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vm));

    float16x4_t vf_lo = vget_low_f16(vf);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vf_lo)); o += 4;
      vacc_lo = vadd_f16(vacc_lo, vf_lo);
      vf_lo = vget_high_f16(vf);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vf_lo), 0); o += 2;
      vacc_lo = vadd_f16(vacc_lo, vreinterpret_f16_u64(vshl_n_u64(vreinterpret_u64_f16(vf_lo), 32)));
      vf_lo = vext_f16(vf_lo, vf_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vf_lo), 0);
      vacc_lo = vadd_f16(vacc_lo, vreinterpret_f16_u64(vshl_n_u64(vreinterpret_u64_f16(vf_lo), 48)));
    }
  }
  vacc_lo = vpadd_f16(vacc_lo, vacc_lo);
  vacc_lo = vpadd_f16(vacc_lo, vacc_lo);
  vst1_lane_u16(sum, vreinterpret_u16_f16(vacc_lo), 0);
}

void xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40(
    size_t batch,
    const void* input,
    const void* max,
    void* output,
    void* sum,
    const union xnn_f16_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float16x8_t vlog2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3DC5)));  // 0x1.714p+0h
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vminus_ln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xB98C)));  // -0x1.630p-1h
  const float16x8_t vminus_ln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x0AF4)));  // 0x1.BD0p-13h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x37F9)));  // 0x1.FE4p-2h
  const float16x8_t vc1 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C0E)));  // 0x1.038p+0h
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float16x8_t vi_max = vreinterpretq_f16_u16(vld1q_dup_u16(max));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float16x8_t vacc0 = vreinterpretq_f16_u16(vmovq_n_u16(0));
  for (; batch >= 40 * sizeof(uint16_t); batch -= 40 * sizeof(uint16_t)) {
    const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx0 = vsubq_f16(vi0, vi_max);
    const float16x8_t vx1 = vsubq_f16(vi1, vi_max);
    const float16x8_t vx2 = vsubq_f16(vi2, vi_max);
    const float16x8_t vx3 = vsubq_f16(vi3, vi_max);
    const float16x8_t vx4 = vsubq_f16(vi4, vi_max);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vx0, vlog2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vx1, vlog2e);
    float16x8_t vn2 = vfmaq_f16(vmagic_bias, vx2, vlog2e);
    float16x8_t vn3 = vfmaq_f16(vmagic_bias, vx3, vlog2e);
    float16x8_t vn4 = vfmaq_f16(vmagic_bias, vx4, vlog2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    const float16x8_t vs2 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn2), 10));
    const float16x8_t vs3 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn3), 10));
    const float16x8_t vs4 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn4), 10));

    vn0 = vsubq_f16(vn0, vmagic_bias);
    vn1 = vsubq_f16(vn1, vmagic_bias);
    vn2 = vsubq_f16(vn2, vmagic_bias);
    vn3 = vsubq_f16(vn3, vmagic_bias);
    vn4 = vsubq_f16(vn4, vmagic_bias);

    float16x8_t vt0 = vfmaq_f16(vx0, vn0, vminus_ln2_hi);
    float16x8_t vt1 = vfmaq_f16(vx1, vn1, vminus_ln2_hi);
    float16x8_t vt2 = vfmaq_f16(vx2, vn2, vminus_ln2_hi);
    float16x8_t vt3 = vfmaq_f16(vx3, vn3, vminus_ln2_hi);
    float16x8_t vt4 = vfmaq_f16(vx4, vn4, vminus_ln2_hi);

    vt0 = vfmaq_f16(vt0, vn0, vminus_ln2_lo);
    vt1 = vfmaq_f16(vt1, vn1, vminus_ln2_lo);
    vt2 = vfmaq_f16(vt2, vn2, vminus_ln2_lo);
    vt3 = vfmaq_f16(vt3, vn3, vminus_ln2_lo);
    vt4 = vfmaq_f16(vt4, vn4, vminus_ln2_lo);

    const float16x8_t vp0 = vfmaq_f16(vc1, vc2, vt0);
    const float16x8_t vp1 = vfmaq_f16(vc1, vc2, vt1);
    const float16x8_t vp2 = vfmaq_f16(vc1, vc2, vt2);
    const float16x8_t vp3 = vfmaq_f16(vc1, vc2, vt3);
    const float16x8_t vp4 = vfmaq_f16(vc1, vc2, vt4);

    vt0 = vmulq_f16(vt0, vs0);
    vt1 = vmulq_f16(vt1, vs1);
    vt2 = vmulq_f16(vt2, vs2);
    vt3 = vmulq_f16(vt3, vs3);
    vt4 = vmulq_f16(vt4, vs4);

    float16x8_t vf0 = vfmaq_f16(vs0, vp0, vt0);
    const uint16x8_t vm0 = vcltq_f16(vx0, vdenorm_cutoff);
    float16x8_t vf1 = vfmaq_f16(vs1, vp1, vt1);
    const uint16x8_t vm1 = vcltq_f16(vx1, vdenorm_cutoff);
    float16x8_t vf2 = vfmaq_f16(vs2, vp2, vt2);
    const uint16x8_t vm2 = vcltq_f16(vx2, vdenorm_cutoff);
    float16x8_t vf3 = vfmaq_f16(vs3, vp3, vt3);
    const uint16x8_t vm3 = vcltq_f16(vx3, vdenorm_cutoff);
    float16x8_t vf4 = vfmaq_f16(vs4, vp4, vt4);
    const uint16x8_t vm4 = vcltq_f16(vx4, vdenorm_cutoff);

    vf0 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf0), vm0));
    vf1 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf1), vm1));
    vf2 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf2), vm2));
    vf3 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf3), vm3));
    vf4 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf4), vm4));

    vst1q_u16(o, vreinterpretq_u16_f16(vf0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf1)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf2)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf3)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf4)); o += 8;

    vacc0 = vaddq_f16(vacc0, vf0);
    vacc0 = vaddq_f16(vacc0, vf1);
    vacc0 = vaddq_f16(vacc0, vf2);
    vacc0 = vaddq_f16(vacc0, vf3);
    vacc0 = vaddq_f16(vacc0, vf4);
  }

  float16x8_t vacc = vacc0;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx = vsubq_f16(vi, vi_max);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f16(vt, vn, vminus_ln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);

    float16x8_t vf = vfmaq_f16(vs, vp, vt);
    const uint16x8_t vm = vcltq_f16(vx, vdenorm_cutoff);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vm));

    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;

    vacc = vaddq_f16(vacc, vf);
  }
  float16x4_t vacc_lo = vadd_f16(vget_low_f16(vacc), vget_high_f16(vacc));
  if (batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i));

    const float16x8_t vx = vsubq_f16(vi, vi_max);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f16(vt, vn, vminus_ln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);

    float16x8_t vf = vfmaq_f16(vs, vp, vt);
    const uint16x8_t vm = vcltq_f16(vx, vdenorm_cutoff);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vm));

    float16x4_t vf_lo = vget_low_f16(vf);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vf_lo)); o += 4;
      vacc_lo = vadd_f16(vacc_lo, vf_lo);
      vf_lo = vget_high_f16(vf);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vf_lo), 0); o += 2;
      vacc_lo = vadd_f16(vacc_lo, vreinterpret_f16_u64(vshl_n_u64(vreinterpret_u64_f16(vf_lo), 32)));
      vf_lo = vext_f16(vf_lo, vf_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vf_lo), 0);
      vacc_lo = vadd_f16(vacc_lo, vreinterpret_f16_u64(vshl_n_u64(vreinterpret_u64_f16(vf_lo), 48)));
    }
  }
  vacc_lo = vpadd_f16(vacc_lo, vacc_lo);
  vacc_lo = vpadd_f16(vacc_lo, vacc_lo);
  vst1_lane_u16(sum, vreinterpret_u16_f16(vacc_lo), 0);
}

void xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float16x8_t vmax0 = vreinterpretq_f16_u16(vld1q_dup_u16(i));
  float16x8_t vmax1 = vmax0;
  float16x8_t vmax2 = vmax0;
  float16x8_t vmax3 = vmax0;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vt0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vmax0 = vmaxq_f16(vmax0, vt0);
    vmax1 = vmaxq_f16(vmax1, vt1);
    vmax2 = vmaxq_f16(vmax2, vt2);
    vmax3 = vmaxq_f16(vmax3, vt3);
  }
  vmax0 = vmaxq_f16(vmax0, vmax1);
  vmax2 = vmaxq_f16(vmax2, vmax3);
  vmax0 = vmaxq_f16(vmax0, vmax2);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vt = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vmax0 = vmaxq_f16(vmax0, vt);
  }
  float16x4_t vmax_lo = vmax_f16(vget_low_f16(vmax0), vget_high_f16(vmax0));

  if (XNN_UNLIKELY(batch != 0)) {
    const float16x8_t vt = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x4_t vt_lo = vget_low_f16(vt);
    if (batch & (4 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vt_lo);
      vt_lo = vget_high_f16(vt);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vext_f16(vmax_lo, vt_lo, 2));
      vt_lo = vext_f16(vt_lo, vt_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vext_f16(vmax_lo, vt_lo, 1));
    }
  }
  #if XNN_ARCH_ARM64 && defined(__GNUC__)
    *((__fp16*) o) = vmaxv_f16(vmax_lo);
  #else
    vmax_lo = vpmax_f16(vmax_lo, vmax_lo);
    vmax_lo = vpmax_f16(vmax_lo, vmax_lo);
    vst1_lane_u16(o, vreinterpret_u16_f16(vmax_lo), 0);
  #endif
}

void xnn_f16_rminmax_ukernel__neonfp16arith_u32_acc4(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float16x8_t vmin0 = vreinterpretq_f16_u16(vld1q_dup_u16(i));
  float16x8_t vmax0 = vmin0;
  float16x8_t vmin1 = vmin0;
  float16x8_t vmax1 = vmax0;
  float16x8_t vmin2 = vmin0;
  float16x8_t vmax2 = vmax0;
  float16x8_t vmin3 = vmin0;
  float16x8_t vmax3 = vmax0;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vt0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vmin0 = vminq_f16(vmin0, vt0);
    vmax0 = vmaxq_f16(vmax0, vt0);
    vmin1 = vminq_f16(vmin1, vt1);
    vmax1 = vmaxq_f16(vmax1, vt1);
    vmin2 = vminq_f16(vmin2, vt2);
    vmax2 = vmaxq_f16(vmax2, vt2);
    vmin3 = vminq_f16(vmin3, vt3);
    vmax3 = vmaxq_f16(vmax3, vt3);
  }
  vmin0 = vminq_f16(vmin0, vmin1);
  vmax0 = vmaxq_f16(vmax0, vmax1);
  vmin2 = vminq_f16(vmin2, vmin3);
  vmax2 = vmaxq_f16(vmax2, vmax3);
  vmin0 = vminq_f16(vmin0, vmin2);
  vmax0 = vmaxq_f16(vmax0, vmax2);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vt = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vmin0 = vminq_f16(vmin0, vt);
    vmax0 = vmaxq_f16(vmax0, vt);
  }
  float16x4_t vmin_lo = vmin_f16(vget_low_f16(vmin0), vget_high_f16(vmin0));
  float16x4_t vmax_lo = vmax_f16(vget_low_f16(vmax0), vget_high_f16(vmax0));

  if (XNN_UNLIKELY(batch != 0)) {
    const float16x8_t vt = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x4_t vt_lo = vget_low_f16(vt);
    if (batch & (4 * sizeof(uint16_t))) {
      vmin_lo = vmin_f16(vmin_lo, vt_lo);
      vmax_lo = vmax_f16(vmax_lo, vt_lo);
      vt_lo = vget_high_f16(vt);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmin_lo = vmin_f16(vmin_lo, vext_f16(vmin_lo, vt_lo, 2));
      vmax_lo = vmax_f16(vmax_lo, vext_f16(vmax_lo, vt_lo, 2));
      vt_lo = vext_f16(vt_lo, vt_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmin_lo = vmin_f16(vmin_lo, vext_f16(vmin_lo, vt_lo, 1));
      vmax_lo = vmax_f16(vmax_lo, vext_f16(vmax_lo, vt_lo, 1));
    }
  }
  #if XNN_ARCH_ARM64 && defined(__GNUC__)
    *((__fp16*) o) = vminv_f16(vmin_lo);
  #else
    vmin_lo = vpmin_f16(vmin_lo, vmin_lo);
    vmin_lo = vpmin_f16(vmin_lo, vmin_lo);
    vst1_lane_u16(o, vreinterpret_u16_f16(vmin_lo), 0);
  #endif
  #if XNN_ARCH_ARM64 && defined(__GNUC__)
    *((__fp16*) o + 1) = vmaxv_f16(vmax_lo);
  #else
    vmax_lo = vpmax_f16(vmax_lo, vmax_lo);
    vmax_lo = vpmax_f16(vmax_lo, vmax_lo);
    vst1_lane_u16(o + 1, vreinterpret_u16_f16(vmax_lo), 0);
  #endif
}

void xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined(
    size_t mc,
    size_t nc,
    const void* input,
    const void* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    void* output,
    size_t output_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(uint16_t) == 0);
  assert(nc != 0);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->fp16arith.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->fp16arith.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif

  size_t output_decrement = output_stride * nc - 32 * sizeof(uint16_t);
  while XNN_LIKELY(mc >= 32 * sizeof(uint16_t)) {
    const uint16_t* w = (const uint16_t*) weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    float16x8_t vw = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
    intptr_t diff = *dmap++;
    float16x8_t vi01234567 = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x8_t vi89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i + 8));
    float16x8_t viGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i + 16));
    float16x8_t viOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i + 24));
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      float16x8_t vacc01234567 = vw;
      float16x8_t vacc89ABCDEF = vw;
      float16x8_t vaccGHIJKLMN = vw;
      float16x8_t vaccOPQRSTUV = vw;
      vw = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
      if XNN_LIKELY(nnz != 0) {
        do {
          vacc01234567 = vfmaq_f16(vacc01234567, vi01234567, vw);
          vacc89ABCDEF = vfmaq_f16(vacc89ABCDEF, vi89ABCDEF, vw);
          vaccGHIJKLMN = vfmaq_f16(vaccGHIJKLMN, viGHIJKLMN, vw);
          vaccOPQRSTUV = vfmaq_f16(vaccOPQRSTUV, viOPQRSTUV, vw);
          i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
          xnn_prefetch_to_l1(i + 32);
          diff = *dmap++;
          vw = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
          xnn_prefetch_to_l1(w + 64);
          vi01234567 = vreinterpretq_f16_u16(vld1q_u16(i));
          vi89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i + 8));
          viGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i + 16));
          viOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i + 24));
        } while (--nnz != 0);
      }
      float16x8_t vout01234567 = vminq_f16(vacc01234567, vmax);
      float16x8_t vout89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);
      float16x8_t voutGHIJKLMN = vminq_f16(vaccGHIJKLMN, vmax);
      float16x8_t voutOPQRSTUV = vminq_f16(vaccOPQRSTUV, vmax);
      vout01234567 = vmaxq_f16(vout01234567, vmin);
      vout89ABCDEF = vmaxq_f16(vout89ABCDEF, vmin);
      voutGHIJKLMN = vmaxq_f16(voutGHIJKLMN, vmin);
      voutOPQRSTUV = vmaxq_f16(voutOPQRSTUV, vmin);
      vst1q_u16(o, vreinterpretq_u16_f16(vout01234567));
      vst1q_u16(o + 8, vreinterpretq_u16_f16(vout89ABCDEF));
      vst1q_u16(o + 16, vreinterpretq_u16_f16(voutGHIJKLMN));
      vst1q_u16(o + 24, vreinterpretq_u16_f16(voutOPQRSTUV));
      o = (uint16_t*) ((uintptr_t) o + output_stride);
    } while (--n != 0);
    o = (uint16_t*) ((uintptr_t) o - output_decrement);
    i += 32;
    mc -= 32 * sizeof(uint16_t);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(uint16_t);
    if (mc & (16 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
        float16x8_t vacc89ABCDEF = vacc01234567;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(i));
            const float16x8_t va89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i + 8));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x8_t vw = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
            vacc01234567 = vfmaq_f16(vacc01234567, va01234567, vw);
            vacc89ABCDEF = vfmaq_f16(vacc89ABCDEF, va89ABCDEF, vw);
          } while (--nnz != 0);
        }
        float16x8_t vout01234567 = vminq_f16(vacc01234567, vmax);
        float16x8_t vout89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);
        vout01234567 = vmaxq_f16(vout01234567, vmin);
        vout89ABCDEF = vmaxq_f16(vout89ABCDEF, vmin);
        vst1q_u16(o, vreinterpretq_u16_f16(vout01234567));
        vst1q_u16(o + 8, vreinterpretq_u16_f16(vout89ABCDEF));
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 16;
    }
    output_decrement += 8 * sizeof(uint16_t);
    if (mc & (8 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(i));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x8_t vw = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
            vacc01234567 = vfmaq_f16(vacc01234567, va01234567, vw);
          } while (--nnz != 0);
        }
        float16x8_t vout01234567 = vminq_f16(vacc01234567, vmax);
        vout01234567 = vmaxq_f16(vout01234567, vmin);
        vst1q_u16(o, vreinterpretq_u16_f16(vout01234567));
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 8;
    }
    output_decrement += 4 * sizeof(uint16_t);
    if (mc & (4 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x4_t vacc0123 = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x4_t va0123 = vreinterpret_f16_u16(vld1_u16(i));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x4_t vw = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
            vacc0123 = vfma_f16(vacc0123, va0123, vw);
          } while (--nnz != 0);
        }
        float16x4_t vout0123 = vmin_f16(vacc0123, vget_low_f16(vmax));
        vout0123 = vmax_f16(vout0123, vget_low_f16(vmin));
        vst1_u16(o, vreinterpret_u16_f16(vout0123));
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 4;
    }
    output_decrement += 2 * sizeof(uint16_t);
    if (mc & (2 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x4_t vacc01 = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x4_t va01 = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x4_t vw = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
            vacc01 = vfma_f16(vacc01, va01, vw);
          } while (--nnz != 0);
        }
        float16x4_t vout01 = vmin_f16(vacc01, vget_low_f16(vmax));
        vout01 = vmax_f16(vout01, vget_low_f16(vmin));
        vst1_lane_u32((void*) o, vreinterpret_u32_f16(vout01), 0);
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 2;
    }
    output_decrement += 1 * sizeof(uint16_t);
    if (mc & (1 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x4_t vacc0 = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x4_t va0 = vreinterpret_f16_u16(vld1_dup_u16(i));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x4_t vw = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
            vacc0 = vfma_f16(vacc0, va0, vw);
          } while (--nnz != 0);
        }
        float16x4_t vout0 = vmin_f16(vacc0, vget_low_f16(vmax));
        vout0 = vmax_f16(vout0, vget_low_f16(vmin));
        vst1_lane_u16(o, vreinterpret_u16_f16(vout0), 0);
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 1;
    }
  }
}

void xnn_f16_vadd_minmax_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb456789AB = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vaddq_f16(va01234567, vb01234567);
    float16x8_t vy456789AB = vaddq_f16(va456789AB, vb456789AB);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vaddq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

    float16x8_t vy01234567 = vaddq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vaddc_minmax_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  const float16x8_t vb = vreinterpretq_f16_u16(vld1q_dup_u16(b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vaddq_f16(va01234567, vb);
    float16x8_t vy456789AB = vaddq_f16(va456789AB, vb);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vaddq_f16(va01234567, vb);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));

    float16x8_t vy01234567 = vaddq_f16(va01234567, vb);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vmax_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb456789AB = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vmaxq_f16(va01234567, vb01234567);
    float16x8_t vy456789AB = vmaxq_f16(va456789AB, vb456789AB);



    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vmaxq_f16(va01234567, vb01234567);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

    float16x8_t vy01234567 = vmaxq_f16(va01234567, vb01234567);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vmaxc_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  const float16x8_t vb = vreinterpretq_f16_u16(vld1q_dup_u16(b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vmaxq_f16(va01234567, vb);
    float16x8_t vy456789AB = vmaxq_f16(va456789AB, vb);



    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vmaxq_f16(va01234567, vb);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));

    float16x8_t vy01234567 = vmaxq_f16(va01234567, vb);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vmin_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb456789AB = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vminq_f16(va01234567, vb01234567);
    float16x8_t vy456789AB = vminq_f16(va456789AB, vb456789AB);



    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vminq_f16(va01234567, vb01234567);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

    float16x8_t vy01234567 = vminq_f16(va01234567, vb01234567);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vminc_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  const float16x8_t vb = vreinterpretq_f16_u16(vld1q_dup_u16(b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vminq_f16(va01234567, vb);
    float16x8_t vy456789AB = vminq_f16(va456789AB, vb);



    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vminq_f16(va01234567, vb);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));

    float16x8_t vy01234567 = vminq_f16(va01234567, vb);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vmul_minmax_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb456789AB = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb01234567);
    float16x8_t vy456789AB = vmulq_f16(va456789AB, vb456789AB);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vmulc_minmax_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  const float16x8_t vb = vreinterpretq_f16_u16(vld1q_dup_u16(b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb);
    float16x8_t vy456789AB = vmulq_f16(va456789AB, vb);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vrsubc_minmax_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  const float16x8_t vb = vreinterpretq_f16_u16(vld1q_dup_u16(b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vsubq_f16(vb, va01234567);
    float16x8_t vy456789AB = vsubq_f16(vb, va456789AB);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vsubq_f16(vb, va01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));

    float16x8_t vy01234567 = vsubq_f16(vb, va01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vsqrdiff_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb456789AB = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb01234567);
    float16x8_t vy456789AB = vsubq_f16(va456789AB, vb456789AB);

    vy01234567 = vmulq_f16(vy01234567, vy01234567);
    vy456789AB = vmulq_f16(vy456789AB, vy456789AB);


    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb01234567);
    vy01234567 = vmulq_f16(vy01234567, vy01234567);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb01234567);
    vy01234567 = vmulq_f16(vy01234567, vy01234567);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vsqrdiffc_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  const float16x8_t vb = vreinterpretq_f16_u16(vld1q_dup_u16(b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb);
    float16x8_t vy456789AB = vsubq_f16(va456789AB, vb);

    vy01234567 = vmulq_f16(vy01234567, vy01234567);
    vy456789AB = vmulq_f16(vy456789AB, vy456789AB);


    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb);
    vy01234567 = vmulq_f16(vy01234567, vy01234567);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb);
    vy01234567 = vmulq_f16(vy01234567, vy01234567);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vsub_minmax_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb456789AB = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb01234567);
    float16x8_t vy456789AB = vsubq_f16(va456789AB, vb456789AB);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vsubc_minmax_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  const float16x8_t vb = vreinterpretq_f16_u16(vld1q_dup_u16(b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb);
    float16x8_t vy456789AB = vsubq_f16(va456789AB, vb);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}

void xnn_f16_vclamp_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->fp16arith.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->fp16arith.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc01234567 = vmaxq_f16(vacc01234567, vmin);
    vacc89ABCDEF = vmaxq_f16(vacc89ABCDEF, vmin);

    vacc01234567 = vminq_f16(vacc01234567, vmax);
    vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc89ABCDEF)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vmaxq_f16(vacc, vmin);
    vacc = vminq_f16(vacc, vmax);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    if (batch & (4 * sizeof(uint16_t))) {
      float16x4_t vacc = vreinterpret_f16_u16(vld1_u16(i)); i += 4;
      vacc = vmax_f16(vacc, vget_low_f16(vmin));
      vacc = vmin_f16(vacc, vget_low_f16(vmax));
      vst1_u16(o, vreinterpret_u16_f16(vacc)); o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      float16x4_t vacc = vreinterpret_f16_u32(vld1_dup_u32((const void*) i)); i += 2;
      vacc = vmax_f16(vacc, vget_low_f16(vmin));
      vacc = vmin_f16(vacc, vget_low_f16(vmax));
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc), 0); o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      float16x4_t vacc = vreinterpret_f16_u16(vld1_dup_u16(i));
      vacc = vmax_f16(vacc, vget_low_f16(vmin));
      vacc = vmin_f16(vacc, vget_low_f16(vmax));
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc), 0);
    }
  }
}

void xnn_f16_vcmul_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input_a,
    const void* input_b,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* ar = (const uint16_t*) input_a;
  const uint16_t* ai = (const uint16_t*) ((uintptr_t) input_a + batch);
  const uint16_t* br = (const uint16_t*) input_b;
  const uint16_t* bi = (const uint16_t*) ((uintptr_t) input_b + batch);
  uint16_t* or = (uint16_t*) output;
  uint16_t* oi = (uint16_t*) ((uintptr_t) output + batch);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va0r = vreinterpretq_f16_u16(vld1q_u16(ar)); ar += 8;
    const float16x8_t va0i = vreinterpretq_f16_u16(vld1q_u16(ai)); ai += 8;
    const float16x8_t vb0r = vreinterpretq_f16_u16(vld1q_u16(br)); br += 8;
    const float16x8_t vb0i = vreinterpretq_f16_u16(vld1q_u16(bi)); bi += 8;
    const float16x8_t va1r = vreinterpretq_f16_u16(vld1q_u16(ar)); ar += 8;
    const float16x8_t va1i = vreinterpretq_f16_u16(vld1q_u16(ai)); ai += 8;
    const float16x8_t vb1r = vreinterpretq_f16_u16(vld1q_u16(br)); br += 8;
    const float16x8_t vb1i = vreinterpretq_f16_u16(vld1q_u16(bi)); bi += 8;

    float16x8_t vacc0r = vmulq_f16(va0r, vb0r);
    float16x8_t vacc0i = vmulq_f16(va0r, vb0i);
    float16x8_t vacc1r = vmulq_f16(va1r, vb1r);
    float16x8_t vacc1i = vmulq_f16(va1r, vb1i);

    vacc0r = vfmsq_f16(vacc0r, va0i, vb0i);
    vacc0i = vfmaq_f16(vacc0i, va0i, vb0r);
    vacc1r = vfmsq_f16(vacc1r, va1i, vb1i);
    vacc1i = vfmaq_f16(vacc1i, va1i, vb1r);

    vst1q_u16(or, vreinterpretq_u16_f16(vacc0r)); or += 8;
    vst1q_u16(oi, vreinterpretq_u16_f16(vacc0i)); oi += 8;
    vst1q_u16(or, vreinterpretq_u16_f16(vacc1r)); or += 8;
    vst1q_u16(oi, vreinterpretq_u16_f16(vacc1i)); oi += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t var = vreinterpretq_f16_u16(vld1q_u16(ar)); ar += 8;
    const float16x8_t vai = vreinterpretq_f16_u16(vld1q_u16(ai)); ai += 8;
    const float16x8_t vbr = vreinterpretq_f16_u16(vld1q_u16(br)); br += 8;
    const float16x8_t vbi = vreinterpretq_f16_u16(vld1q_u16(bi)); bi += 8;

    float16x8_t vaccr = vmulq_f16(var, vbr);
    float16x8_t vacci = vmulq_f16(var, vbi);

    vaccr = vfmsq_f16(vaccr, vai, vbi);
    vacci = vfmaq_f16(vacci, vai, vbr);

    vst1q_u16(or, vreinterpretq_u16_f16(vaccr)); or += 8;
    vst1q_u16(oi, vreinterpretq_u16_f16(vacci)); oi += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t var = vreinterpretq_f16_u16(vld1q_u16(ar)); ar += 8;
    const float16x8_t vai = vreinterpretq_f16_u16(vld1q_u16(ai)); ai += 8;
    const float16x8_t vbr = vreinterpretq_f16_u16(vld1q_u16(br)); br += 8;
    const float16x8_t vbi = vreinterpretq_f16_u16(vld1q_u16(bi)); bi += 8;

    float16x8_t vaccr = vmulq_f16(var, vbr);
    float16x8_t vacci = vmulq_f16(var, vbi);

    vaccr = vfmsq_f16(vaccr, vai, vbi);
    vacci = vfmaq_f16(vacci, vai, vbr);

    float16x4_t vaccr_lo = vget_low_f16(vaccr);
    float16x4_t vacci_lo = vget_low_f16(vacci);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(or, vreinterpret_u16_f16(vaccr_lo)); or += 4;
      vst1_u16(oi, vreinterpret_u16_f16(vacci_lo)); oi += 4;
      vaccr_lo = vget_high_f16(vaccr);
      vacci_lo = vget_high_f16(vacci);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) or, vreinterpret_u32_f16(vaccr_lo), 0); or += 2;
      vst1_lane_u32((void*) oi, vreinterpret_u32_f16(vacci_lo), 0); oi += 2;
      vaccr_lo = vext_f16(vaccr_lo, vaccr_lo, 2);
      vacci_lo = vext_f16(vacci_lo, vacci_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(or, vreinterpret_u16_f16(vaccr_lo), 0);
      vst1_lane_u16(oi, vreinterpret_u16_f16(vacci_lo), 0);
    }
  }
}

void xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vsat_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC829)));  // -0x1.0A4p+3h
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vlog2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3DC5)));  // 0x1.714p+0h
  const float16x8_t vminus_ln2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xB98C)));  // -0x1.62E430p-1h
  const float16x8_t vc3 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x315B)));  // 0x1.56Cp-3h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3808)));  // 0x1.020p-1h

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vminus_ln2);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);

  const float16x8_t vprescale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->scalar.prescale));
  const float16x8_t vminus_alpha = vreinterpretq_f16_u16(vld1q_dup_u16(&params->scalar.minus_alpha));
  const float16x8_t vbeta = vreinterpretq_f16_u16(vld1q_dup_u16(&params->scalar.beta));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float16x8_t vz0 = vmulq_f16(vx0, vprescale);
    float16x8_t vz1 = vmulq_f16(vx1, vprescale);

    vz0 = vmaxq_f16(vz0, vsat_cutoff);
    vz1 = vmaxq_f16(vz1, vsat_cutoff);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vz0, vlog2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vz1, vlog2e);

    float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    vn0 = vsubq_f16(vn0, vmagic_bias);
    float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    vn1 = vsubq_f16(vn1, vmagic_bias);

    float16x8_t vt0 = vfmaq_f16(vz0, vn0, vminus_ln2);
    float16x8_t vt1 = vfmaq_f16(vz1, vn1, vminus_ln2);

    float16x8_t vp0 = vfmaq_f16(vc2, vc3, vt0);
    vp0 = vmulq_f16(vp0, vt0);
    float16x8_t vp1 = vfmaq_f16(vc2, vc3, vt1);
    vp1 = vmulq_f16(vp1, vt1);

    vt0 = vmulq_f16(vt0, vs0);
    vs0 = vfmsq_f16(vminus_alpha, vs0, vminus_alpha);
    vt1 = vmulq_f16(vt1, vs1);
    vs1 = vfmsq_f16(vminus_alpha, vs1, vminus_alpha);

    vp0 = vfmaq_f16(vt0, vp0, vt0);
    vp1 = vfmaq_f16(vt1, vp1, vt1);

    float16x8_t ve0 = vfmsq_f16(vs0, vp0, vminus_alpha);
    const uint16x8_t vm0 = vcltq_s16(vreinterpretq_s16_f16(vx0), vmovq_n_s16(0));
    float16x8_t ve1 = vfmsq_f16(vs1, vp1, vminus_alpha);
    const uint16x8_t vm1 = vcltq_s16(vreinterpretq_s16_f16(vx1), vmovq_n_s16(0));

    vx0 = vmulq_f16(vx0, vbeta);
    vx1 = vmulq_f16(vx1, vbeta);

    const float16x8_t vy0 = vbslq_f16(vm0, ve0, vx0);
    const float16x8_t vy1 = vbslq_f16(vm1, ve1, vx1);

    vst1q_u16(o, vreinterpretq_u16_f16(vy0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vz = vmulq_f16(vx, vprescale);
    vz = vmaxq_f16(vz, vsat_cutoff);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vlog2e);
    float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);
    float16x8_t vt = vfmaq_f16(vz, vn, vminus_ln2);

    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vmulq_f16(vp, vt);
    vt = vmulq_f16(vt, vs);
    vs = vfmsq_f16(vminus_alpha, vs, vminus_alpha);
    vp = vfmaq_f16(vt, vp, vt);
    float16x8_t ve = vfmsq_f16(vs, vp, vminus_alpha);

    const uint16x8_t vm = vcltq_s16(vreinterpretq_s16_f16(vx), vmovq_n_s16(0));
    vx = vmulq_f16(vx, vbeta);
    const float16x8_t vy = vbslq_f16(vm, ve, vx);
    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vz = vmulq_f16(vx, vprescale);
    vz = vmaxq_f16(vz, vsat_cutoff);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vlog2e);
    float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);
    float16x8_t vt = vfmaq_f16(vz, vn, vminus_ln2);

    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vmulq_f16(vp, vt);
    vt = vmulq_f16(vt, vs);
    vs = vfmsq_f16(vminus_alpha, vs, vminus_alpha);
    vp = vfmaq_f16(vt, vp, vt);
    float16x8_t ve = vfmsq_f16(vs, vp, vminus_alpha);

    const uint16x8_t vm = vcltq_s16(vreinterpretq_s16_f16(vx), vmovq_n_s16(0));
    vx = vmulq_f16(vx, vbeta);
    float16x8_t vy = vbslq_f16(vm, ve, vx);
    float16x4_t vy_lo = vget_low_f16(vy);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;
      vy_lo = vget_high_f16(vy);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o += 2;
      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}

void xnn_f16_vhswish_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* restrict input,
    void* restrict output,
    const union xnn_f16_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vsixth = vreinterpretq_f16_u16(vdupq_n_u16(UINT16_C(0x3155)));
  const float16x8_t vthree = vreinterpretq_f16_u16(vdupq_n_u16(UINT16_C(0x4200)));
  const int16x8_t vsix = vreinterpretq_s16_u16(vdupq_n_u16(UINT16_C(0x4600)));
  const int16x8_t vzero = vdupq_n_s16(0);

  XNN_FORCE_REALIZATION(vsixth);
  XNN_FORCE_REALIZATION(vthree);
  XNN_FORCE_REALIZATION(vsix);
  // XNN_FORCE_REALIZATION(vzero);

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vx01234567 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float16x8_t vacc01234567 = vaddq_f16(vx01234567, vthree);
    vx01234567 = vmulq_f16(vx01234567, vsixth);
    float16x8_t vacc89ABCDEF = vaddq_f16(vx89ABCDEF, vthree);
    vx89ABCDEF = vmulq_f16(vx89ABCDEF, vsixth);

    vacc01234567 = vreinterpretq_f16_s16(vmaxq_s16(vreinterpretq_s16_f16(vacc01234567), vzero));
    vacc89ABCDEF = vreinterpretq_f16_s16(vmaxq_s16(vreinterpretq_s16_f16(vacc89ABCDEF), vzero));

    vacc01234567 = vreinterpretq_f16_s16(vminq_s16(vreinterpretq_s16_f16(vacc01234567), vsix));
    vacc89ABCDEF = vreinterpretq_f16_s16(vminq_s16(vreinterpretq_s16_f16(vacc89ABCDEF), vsix));

    vacc01234567 = vmulq_f16(vacc01234567, vx01234567);
    vacc89ABCDEF = vmulq_f16(vacc89ABCDEF, vx89ABCDEF);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc89ABCDEF)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc = vaddq_f16(vx, vthree);
    vx = vmulq_f16(vx, vsixth);
    vacc = vreinterpretq_f16_s16(vmaxq_s16(vreinterpretq_s16_f16(vacc), vzero));
    vacc = vreinterpretq_f16_s16(vminq_s16(vreinterpretq_s16_f16(vacc), vsix));
    vacc = vmulq_f16(vacc, vx);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x8_t vacc = vaddq_f16(vx, vthree);
    vx = vmulq_f16(vx, vsixth);
    vacc = vreinterpretq_f16_s16(vmaxq_s16(vreinterpretq_s16_f16(vacc), vzero));
    vacc = vreinterpretq_f16_s16(vminq_s16(vreinterpretq_s16_f16(vacc), vsix));
    vacc = vmulq_f16(vacc, vx);

    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_f16_vlrelu_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vslope = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.slope));
  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t vx01234567 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float16x8_t vacc01234567 = vmulq_f16(vx01234567, vslope);
    const uint16x8_t vmask01234567 = vcltq_s16(vreinterpretq_s16_f16(vx01234567), vmovq_n_s16(0));
    float16x8_t vacc89ABCDEF = vmulq_f16(vx89ABCDEF, vslope);
    const uint16x8_t vmask89ABCDEF = vcltq_s16(vreinterpretq_s16_f16(vx89ABCDEF), vmovq_n_s16(0));

    vacc01234567 = vbslq_f16(vmask01234567, vacc01234567, vx01234567);
    vacc89ABCDEF = vbslq_f16(vmask89ABCDEF, vacc89ABCDEF, vx89ABCDEF);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc89ABCDEF)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc = vmulq_f16(vx, vslope);
    const uint16x8_t vmask = vcltq_s16(vreinterpretq_s16_f16(vx), vmovq_n_s16(0));
    vacc = vbslq_f16(vmask, vacc, vx);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x8_t vacc = vmulq_f16(vx, vslope);
    const uint16x8_t vmask = vcltq_s16(vreinterpretq_s16_f16(vx), vmovq_n_s16(0));
    vacc = vbslq_f16(vmask, vacc, vx);

    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x(
    size_t rows,
    size_t channels,
    const void* restrict input,
    size_t input_stride,
    const void* restrict weights,
    void* restrict output,
    size_t output_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  const uint16_t* i0 = (const uint16_t*) input;
  uint16_t* o0 = (uint16_t*) output;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const uint16_t* w = (const uint16_t*) weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const float16x8_t vscale01234567 = vreinterpretq_f16_u16(vld1q_u16(w));

      float16x8_t vacc0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      float16x8_t vacc1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;

      const float16x8_t vbias01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 8));
      w += 16;

      vacc0x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc0x01234567);
      vacc1x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc1x01234567);

      vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
      vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);

      vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
      vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);

      vst1q_u16(o0, vreinterpretq_u16_f16(vacc0x01234567)); o0 += 8;
      vst1q_u16(o1, vreinterpretq_u16_f16(vacc1x01234567)); o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const float16x8_t vscale01234567 = vreinterpretq_f16_u16(vld1q_u16(w));

      float16x8_t vacc0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 = (const uint16_t*) ((uintptr_t) i0 + c);
      float16x8_t vacc1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 = (const uint16_t*) ((uintptr_t) i1 + c);

      const float16x8_t vbias01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 8));

      vacc0x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc0x01234567);
      vacc1x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc1x01234567);

      vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
      vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);

      vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
      vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);

      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      float16x4_t vacc1x0123 = vget_low_f16(vacc1x01234567);
      if (c & (4 * sizeof(uint16_t))) {
        vst1_u16(o0, vreinterpret_u16_f16(vacc0x0123)); o0 += 4;
        vst1_u16(o1, vreinterpret_u16_f16(vacc1x0123)); o1 += 4;

        vacc0x0123 = vget_high_f16(vacc0x01234567);
        vacc1x0123 = vget_high_f16(vacc1x01234567);
      }
      if (c & (2 * sizeof(uint16_t))) {
        vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vacc0x0123), 0); o0 += 2;
        vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vacc1x0123), 0); o1 += 2;

        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
        vacc1x0123 = vext_f16(vacc1x0123, vacc1x0123, 2);
      }
      if (c & (1 * sizeof(uint16_t))) {
        vst1_lane_u16(o0, vreinterpret_u16_f16(vacc0x0123), 0); o0 += 1;
        vst1_lane_u16(o1, vreinterpret_u16_f16(vacc1x0123), 0); o1 += 1;
      }
    }
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f16_vrndd_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc0 = vrndmq_f16(vacc0);
    vacc1 = vrndmq_f16(vacc1);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vrndmq_f16(vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vrndmq_f16(vacc);
    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_f16_vrndne_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc0 = vrndnq_f16(vacc0);
    vacc1 = vrndnq_f16(vacc1);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vrndnq_f16(vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vrndnq_f16(vacc);
    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_f16_vrndu_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc0 = vrndpq_f16(vacc0);
    vacc1 = vrndpq_f16(vacc1);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vrndpq_f16(vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vrndpq_f16(vacc);
    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_f16_vrndz_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc0 = vrndq_f16(vacc0);
    vacc1 = vrndq_f16(vacc1);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vrndq_f16(vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vrndq_f16(vacc);
    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_f16_vrsqrt_ukernel__neonfp16arith_rsqrt_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vt0_0 = vrsqrteq_f16(vx0);
    const float16x8_t vt0_1 = vrsqrteq_f16(vx1);
    const float16x8_t vt1_0 = vmulq_f16(vt0_0, vt0_0);
    const float16x8_t vt1_1 = vmulq_f16(vt0_1, vt0_1);
    const float16x8_t vt2_0 = vrsqrtsq_f16(vx0, vt1_0);
    const float16x8_t vt2_1 = vrsqrtsq_f16(vx1, vt1_1);
    const float16x8_t vy0 = vmulq_f16(vt0_0, vt2_0);
    const float16x8_t vy1 = vmulq_f16(vt0_1, vt2_1);

    vst1q_u16(o, vreinterpretq_u16_f16(vy0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vt0 = vrsqrteq_f16(vx);
    const float16x8_t vt1 = vmulq_f16(vt0, vt0);
    const float16x8_t vt2 = vrsqrtsq_f16(vx, vt1);
    const float16x8_t vy = vmulq_f16(vt0, vt2);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));
    const float16x4_t vx_lo = vget_low_f16(vx);
    const float16x4_t vx_hi = vget_high_f16(vx);

    const float16x4_t vt0 = vrsqrte_f16(vx_lo);
    const float16x4_t vt1 = vmul_f16(vt0, vt0);
    const float16x4_t vt2 = vrsqrts_f16(vx_lo, vt1);
    float16x4_t vy_lo = vmul_f16(vt0, vt2);

    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;

      const float16x4_t vt0 = vrsqrte_f16(vx_hi);
      const float16x4_t vt1 = vmul_f16(vt0, vt0);
      const float16x4_t vt2 = vrsqrts_f16(vx_hi, vt1);
      vy_lo = vmul_f16(vt0, vt2);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o += 2;
      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}

void xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
  const float16x8_t vln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
  const float16x8_t vln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x8AF4)));  // -0x1.BD0p-13h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x37F9)));  // 0x1.FE4p-2h
  const float16x8_t vc1 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC0E)));  // -0x1.038p+0h
  const float16x8_t vone = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C00)));  // 1.0h
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_log2e);
  XNN_FORCE_REALIZATION(vln2_hi);
  XNN_FORCE_REALIZATION(vln2_lo);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 40 * sizeof(uint16_t); batch -= 40 * sizeof(uint16_t)) {
    const float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx4 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vz0 = vabsq_f16(vx0);
    const float16x8_t vz1 = vabsq_f16(vx1);
    const float16x8_t vz2 = vabsq_f16(vx2);
    const float16x8_t vz3 = vabsq_f16(vx3);
    const float16x8_t vz4 = vabsq_f16(vx4);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vz0, vminus_log2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vz1, vminus_log2e);
    float16x8_t vn2 = vfmaq_f16(vmagic_bias, vz2, vminus_log2e);
    float16x8_t vn3 = vfmaq_f16(vmagic_bias, vz3, vminus_log2e);
    float16x8_t vn4 = vfmaq_f16(vmagic_bias, vz4, vminus_log2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    const float16x8_t vs2 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn2), 10));
    const float16x8_t vs3 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn3), 10));
    const float16x8_t vs4 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn4), 10));

    vn0 = vsubq_f16(vn0, vmagic_bias);
    vn1 = vsubq_f16(vn1, vmagic_bias);
    vn2 = vsubq_f16(vn2, vmagic_bias);
    vn3 = vsubq_f16(vn3, vmagic_bias);
    vn4 = vsubq_f16(vn4, vmagic_bias);

    float16x8_t vt0 = vfmaq_f16(vz0, vn0, vln2_hi);
    float16x8_t vt1 = vfmaq_f16(vz1, vn1, vln2_hi);
    float16x8_t vt2 = vfmaq_f16(vz2, vn2, vln2_hi);
    float16x8_t vt3 = vfmaq_f16(vz3, vn3, vln2_hi);
    float16x8_t vt4 = vfmaq_f16(vz4, vn4, vln2_hi);

    vt0 = vfmaq_f16(vt0, vn0, vln2_lo);
    vt1 = vfmaq_f16(vt1, vn1, vln2_lo);
    vt2 = vfmaq_f16(vt2, vn2, vln2_lo);
    vt3 = vfmaq_f16(vt3, vn3, vln2_lo);
    vt4 = vfmaq_f16(vt4, vn4, vln2_lo);

    const float16x8_t vp0 = vfmaq_f16(vc1, vc2, vt0);
    const float16x8_t vp1 = vfmaq_f16(vc1, vc2, vt1);
    const float16x8_t vp2 = vfmaq_f16(vc1, vc2, vt2);
    const float16x8_t vp3 = vfmaq_f16(vc1, vc2, vt3);
    const float16x8_t vp4 = vfmaq_f16(vc1, vc2, vt4);

    vt0 = vmulq_f16(vt0, vs0);
    vt1 = vmulq_f16(vt1, vs1);
    vt2 = vmulq_f16(vt2, vs2);
    vt3 = vmulq_f16(vt3, vs3);
    vt4 = vmulq_f16(vt4, vs4);

    const float16x8_t ve0 = vfmaq_f16(vs0, vp0, vt0);
    const float16x8_t ve1 = vfmaq_f16(vs1, vp1, vt1);
    const float16x8_t ve2 = vfmaq_f16(vs2, vp2, vt2);
    const float16x8_t ve3 = vfmaq_f16(vs3, vp3, vt3);
    const float16x8_t ve4 = vfmaq_f16(vs4, vp4, vt4);

    const float16x8_t vd0 = vaddq_f16(ve0, vone);
    const float16x8_t vd1 = vaddq_f16(ve1, vone);
    const float16x8_t vd2 = vaddq_f16(ve2, vone);
    const float16x8_t vd3 = vaddq_f16(ve3, vone);
    const float16x8_t vd4 = vaddq_f16(ve4, vone);

    float16x8_t vr0 = vrecpeq_f16(vd0);
    float16x8_t vr1 = vrecpeq_f16(vd1);
    float16x8_t vr2 = vrecpeq_f16(vd2);
    float16x8_t vr3 = vrecpeq_f16(vd3);
    float16x8_t vr4 = vrecpeq_f16(vd4);

    const float16x8_t vadj0 = vfmsq_f16(vone, vr0, vd0);
    const float16x8_t vadj1 = vfmsq_f16(vone, vr1, vd1);
    const float16x8_t vadj2 = vfmsq_f16(vone, vr2, vd2);
    const float16x8_t vadj3 = vfmsq_f16(vone, vr3, vd3);
    const float16x8_t vadj4 = vfmsq_f16(vone, vr4, vd4);

    vr0 = vfmaq_f16(vr0, vr0, vadj0);
    vr1 = vfmaq_f16(vr1, vr1, vadj1);
    vr2 = vfmaq_f16(vr2, vr2, vadj2);
    vr3 = vfmaq_f16(vr3, vr3, vadj3);
    vr4 = vfmaq_f16(vr4, vr4, vadj4);

    float16x8_t vf0 = vmulq_f16(ve0, vr0);
    float16x8_t vf1 = vmulq_f16(ve1, vr1);
    float16x8_t vf2 = vmulq_f16(ve2, vr2);
    float16x8_t vf3 = vmulq_f16(ve3, vr3);
    float16x8_t vf4 = vmulq_f16(ve4, vr4);

    vf0 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf0), vcagtq_f16(vx0, vdenorm_cutoff)));
    vf1 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf1), vcagtq_f16(vx1, vdenorm_cutoff)));
    vf2 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf2), vcagtq_f16(vx2, vdenorm_cutoff)));
    vf3 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf3), vcagtq_f16(vx3, vdenorm_cutoff)));
    vf4 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf4), vcagtq_f16(vx4, vdenorm_cutoff)));

    const uint16x8_t vm0 = vcltq_f16(vx0, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    const uint16x8_t vm1 = vcltq_f16(vx1, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    const uint16x8_t vm2 = vcltq_f16(vx2, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    const uint16x8_t vm3 = vcltq_f16(vx3, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    const uint16x8_t vm4 = vcltq_f16(vx4, vreinterpretq_f16_u16(vmovq_n_u16(0)));

    vf0 = vbslq_f16(vm0, vf0, vsubq_f16(vone, vf0));
    vf1 = vbslq_f16(vm1, vf1, vsubq_f16(vone, vf1));
    vf2 = vbslq_f16(vm2, vf2, vsubq_f16(vone, vf2));
    vf3 = vbslq_f16(vm3, vf3, vsubq_f16(vone, vf3));
    vf4 = vbslq_f16(vm4, vf4, vsubq_f16(vone, vf4));

    vst1q_u16(o, vreinterpretq_u16_f16(vf0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf1)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf2)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf3)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf4)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vz = vabsq_f16(vx);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vz, vn, vln2_hi);
    vt = vfmaq_f16(vt, vn, vln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);
    const float16x8_t ve = vfmaq_f16(vs, vp, vt);
    const float16x8_t vd = vaddq_f16(ve, vone);

    float16x8_t vr = vrecpeq_f16(vd);
    const float16x8_t vadj = vfmsq_f16(vone, vr, vd);
    vr = vfmaq_f16(vr, vr, vadj);

    float16x8_t vf = vmulq_f16(ve, vr);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));

    const float16x8_t vz = vabsq_f16(vx);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vz, vn, vln2_hi);
    vt = vfmaq_f16(vt, vn, vln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);
    const float16x8_t ve = vfmaq_f16(vs, vp, vt);
    const float16x8_t vd = vaddq_f16(ve, vone);

    float16x8_t vr = vrecpeq_f16(vd);
    const float16x8_t vadj = vfmsq_f16(vone, vr, vd);
    vr = vfmaq_f16(vr, vr, vadj);

    float16x8_t vf = vmulq_f16(ve, vr);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    float16x4_t vf_lo = vget_low_f16(vf);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vf_lo)); o += 4;
      vf_lo = vget_high_f16(vf);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vf_lo), 0); o += 2;
      vf_lo = vext_f16(vf_lo, vf_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vf_lo), 0);
    }
  }
}

void xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
  const float16x8_t vln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
  const float16x8_t vln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x8AF4)));  // -0x1.BD0p-13h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x37F9)));  // 0x1.FE4p-2h
  const float16x8_t vc1 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC0E)));  // -0x1.038p+0h
  const float16x8_t vone = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C00)));  // 1.0h
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_log2e);
  XNN_FORCE_REALIZATION(vln2_hi);
  XNN_FORCE_REALIZATION(vln2_lo);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vz0 = vabsq_f16(vx0);
    const float16x8_t vz1 = vabsq_f16(vx1);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vz0, vminus_log2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vz1, vminus_log2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));

    vn0 = vsubq_f16(vn0, vmagic_bias);
    vn1 = vsubq_f16(vn1, vmagic_bias);

    float16x8_t vt0 = vfmaq_f16(vz0, vn0, vln2_hi);
    float16x8_t vt1 = vfmaq_f16(vz1, vn1, vln2_hi);

    vt0 = vfmaq_f16(vt0, vn0, vln2_lo);
    vt1 = vfmaq_f16(vt1, vn1, vln2_lo);

    const float16x8_t vp0 = vfmaq_f16(vc1, vc2, vt0);
    const float16x8_t vp1 = vfmaq_f16(vc1, vc2, vt1);

    vt0 = vmulq_f16(vt0, vs0);
    vt1 = vmulq_f16(vt1, vs1);

    const float16x8_t ve0 = vfmaq_f16(vs0, vp0, vt0);
    const float16x8_t ve1 = vfmaq_f16(vs1, vp1, vt1);

    const float16x8_t vd0 = vaddq_f16(ve0, vone);
    const float16x8_t vd1 = vaddq_f16(ve1, vone);

    float16x8_t vr0 = vrecpeq_f16(vd0);
    float16x8_t vr1 = vrecpeq_f16(vd1);

    const float16x8_t vadj0 = vrecpsq_f16(vr0, vd0);
    const float16x8_t vadj1 = vrecpsq_f16(vr1, vd1);

    vr0 = vmulq_f16(vr0, vadj0);
    vr1 = vmulq_f16(vr1, vadj1);

    float16x8_t vf0 = vmulq_f16(ve0, vr0);
    float16x8_t vf1 = vmulq_f16(ve1, vr1);

    vf0 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf0), vcagtq_f16(vx0, vdenorm_cutoff)));
    vf1 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf1), vcagtq_f16(vx1, vdenorm_cutoff)));

    const uint16x8_t vm0 = vcltq_f16(vx0, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    const uint16x8_t vm1 = vcltq_f16(vx1, vreinterpretq_f16_u16(vmovq_n_u16(0)));

    vf0 = vbslq_f16(vm0, vf0, vsubq_f16(vone, vf0));
    vf1 = vbslq_f16(vm1, vf1, vsubq_f16(vone, vf1));

    vst1q_u16(o, vreinterpretq_u16_f16(vf0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vz = vabsq_f16(vx);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vz, vn, vln2_hi);
    vt = vfmaq_f16(vt, vn, vln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);
    const float16x8_t ve = vfmaq_f16(vs, vp, vt);
    const float16x8_t vd = vaddq_f16(ve, vone);

    float16x8_t vr = vrecpeq_f16(vd);
    const float16x8_t vadj = vrecpsq_f16(vr, vd);
    vr = vmulq_f16(vr, vadj);

    float16x8_t vf = vmulq_f16(ve, vr);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));

    const float16x8_t vz = vabsq_f16(vx);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vz, vn, vln2_hi);
    vt = vfmaq_f16(vt, vn, vln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);
    const float16x8_t ve = vfmaq_f16(vs, vp, vt);
    const float16x8_t vd = vaddq_f16(ve, vone);

    float16x8_t vr = vrecpeq_f16(vd);
    const float16x8_t vadj = vrecpsq_f16(vr, vd);
    vr = vmulq_f16(vr, vadj);

    float16x8_t vf = vmulq_f16(ve, vr);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    float16x4_t vf_lo = vget_low_f16(vf);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vf_lo)); o += 4;
      vf_lo = vget_high_f16(vf);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vf_lo), 0); o += 2;
      vf_lo = vext_f16(vf_lo, vf_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vf_lo), 0);
    }
  }
}

void xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16x8_t vpositive_infinity = vmovq_n_u16(UINT16_C(0x7C00));
  const float16x8_t vhalf = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3800)));  // 0.5h
  const uint16x8_t vexp4_mask = vmovq_n_u16(UINT16_C(0x7800));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx = vbslq_f16(vexp4_mask, vhalf, vi);
    const int16x8_t vexp4i = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi), vexp4_mask));

    const float16x8_t vrsqrtx = vrsqrteq_f16(vx);
    const int16x8_t vpostscale = vhsubq_s16(vexp4i, vreinterpretq_s16_f16(vhalf));

    float16x8_t vsqrtx = vmulq_f16(vrsqrtx, vx);
    const float16x8_t vhalfrsqrtx = vmulq_f16(vrsqrtx, vhalf);
    uint16x8_t vspecial_mask = vcgeq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vresidual = vfmsq_f16(vhalf, vsqrtx, vhalfrsqrtx);
    const uint16x8_t vzero_mask = vceqq_f16(vi, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value = vmovq_n_u16(UINT16_C(0x7E00));

    vsqrtx = vfmaq_f16(vsqrtx, vresidual, vsqrtx);
    vspecial_mask = vorrq_u16(vspecial_mask, vzero_mask);
    const uint16x8_t vinfinity_mask = vceqq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vadjustment = vfmsq_f16(vx, vsqrtx, vsqrtx);
    const uint16x8_t vinput_mask = vorrq_u16(vinfinity_mask, vzero_mask);

    vsqrtx = vfmaq_f16(vsqrtx, vhalfrsqrtx, vadjustment);
    vspecial_value = vbslq_u16(vinput_mask, vreinterpretq_u16_f16(vi), vspecial_value);

    float16x8_t vy = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx), vpostscale));

    vy = vbslq_f16(vspecial_mask, vreinterpretq_f16_u16(vspecial_value), vy);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i));

    const float16x8_t vx = vbslq_f16(vexp4_mask, vhalf, vi);
    const int16x8_t vexp4i = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi), vexp4_mask));

    const float16x8_t vrsqrtx = vrsqrteq_f16(vx);
    const int16x8_t vpostscale = vhsubq_s16(vexp4i, vreinterpretq_s16_f16(vhalf));

    float16x8_t vsqrtx = vmulq_f16(vrsqrtx, vx);
    const float16x8_t vhalfrsqrtx = vmulq_f16(vrsqrtx, vhalf);
    uint16x8_t vspecial_mask = vcgeq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vresidual = vfmsq_f16(vhalf, vsqrtx, vhalfrsqrtx);
    const uint16x8_t vzero_mask = vceqq_f16(vi, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value = vmovq_n_u16(UINT16_C(0x7E00));

    vsqrtx = vfmaq_f16(vsqrtx, vresidual, vsqrtx);
    vspecial_mask = vorrq_u16(vspecial_mask, vzero_mask);
    const uint16x8_t vinfinity_mask = vceqq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vadjustment = vfmsq_f16(vx, vsqrtx, vsqrtx);
    const uint16x8_t vinput_mask = vorrq_u16(vinfinity_mask, vzero_mask);

    vsqrtx = vfmaq_f16(vsqrtx, vhalfrsqrtx, vadjustment);
    vspecial_value = vbslq_u16(vinput_mask, vreinterpretq_u16_f16(vi), vspecial_value);

    float16x8_t vy = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx), vpostscale));

    vy = vbslq_f16(vspecial_mask, vreinterpretq_f16_u16(vspecial_value), vy);

    float16x4_t vy_lo = vget_low_f16(vy);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;
      vy_lo = vget_high_f16(vy);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o += 2;
      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}

void xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32(
    size_t n,
    const void* input,
    void* output,
    const union xnn_f16_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vsat_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4482)));
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x620F)));
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));
  const float16x8_t vln2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));
  const float16x8_t vc3 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBD5B)));
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4008)));
  const float16x8_t vtwo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4000)));
  const float16x8_t vminus_one = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC00)));
  const uint16x8_t vsign_mask = vmovq_n_u16(UINT16_C(0x8000));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n >= 4 * sizeof(float16x8_t); n -= 4 * sizeof(float16x8_t)) {
    const float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float16x8_t vz0 = vabsq_f16(vx0);
    float16x8_t vz1 = vabsq_f16(vx1);
    float16x8_t vz2 = vabsq_f16(vx2);
    float16x8_t vz3 = vabsq_f16(vx3);

    vz0 = vminq_f16(vz0, vsat_cutoff);
    vz1 = vminq_f16(vz1, vsat_cutoff);
    vz2 = vminq_f16(vz2, vsat_cutoff);
    vz3 = vminq_f16(vz3, vsat_cutoff);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vz0, vminus_log2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vz1, vminus_log2e);
    float16x8_t vn2 = vfmaq_f16(vmagic_bias, vz2, vminus_log2e);
    float16x8_t vn3 = vfmaq_f16(vmagic_bias, vz3, vminus_log2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    vn0 = vsubq_f16(vn0, vmagic_bias);
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    vn1 = vsubq_f16(vn1, vmagic_bias);
    const float16x8_t vs2 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn2), 10));
    vn2 = vsubq_f16(vn2, vmagic_bias);
    const float16x8_t vs3 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn3), 10));
    vn3 = vsubq_f16(vn3, vmagic_bias);
    const float16x8_t vt0 = vfmaq_f16(vz0, vn0, vln2);
    const float16x8_t vt1 = vfmaq_f16(vz1, vn1, vln2);
    const float16x8_t vt2 = vfmaq_f16(vz2, vn2, vln2);
    const float16x8_t vt3 = vfmaq_f16(vz3, vn3, vln2);
    float16x8_t vp0 = vfmaq_f16(vc2, vc3, vt0);
    float16x8_t vp1 = vfmaq_f16(vc2, vc3, vt1);
    float16x8_t vp2 = vfmaq_f16(vc2, vc3, vt2);
    float16x8_t vp3 = vfmaq_f16(vc2, vc3, vt3);
    vp0 = vfmsq_f16(vtwo, vp0, vt0);
    vp1 = vfmsq_f16(vtwo, vp1, vt1);
    vp2 = vfmsq_f16(vtwo, vp2, vt2);
    vp3 = vfmsq_f16(vtwo, vp3, vt3);

    const float16x8_t vts0 = vmulq_f16(vt0, vs0);
    const float16x8_t vsmo0 = vaddq_f16(vs0, vminus_one);
    const float16x8_t vts1 = vmulq_f16(vt1, vs1);
    const float16x8_t vsmo1 = vaddq_f16(vs1, vminus_one);
    const float16x8_t vts2 = vmulq_f16(vt2, vs2);
    const float16x8_t vsmo2 = vaddq_f16(vs2, vminus_one);
    const float16x8_t vts3 = vmulq_f16(vt3, vs3);
    const float16x8_t vsmo3 = vaddq_f16(vs3, vminus_one);
    const float16x8_t vemo0 = vfmsq_f16(vsmo0, vp0, vts0);
    const float16x8_t vemo1 = vfmsq_f16(vsmo1, vp1, vts1);
    const float16x8_t vemo2 = vfmsq_f16(vsmo2, vp2, vts2);
    const float16x8_t vemo3 = vfmsq_f16(vsmo3, vp3, vts3);
    const float16x8_t vepo0 = vaddq_f16(vemo0, vtwo);
    const float16x8_t vepo1 = vaddq_f16(vemo1, vtwo);
    const float16x8_t vepo2 = vaddq_f16(vemo2, vtwo);
    const float16x8_t vepo3 = vaddq_f16(vemo3, vtwo);

    float16x8_t vrepo0 = vrecpeq_f16(vepo0);
    float16x8_t vrepo1 = vrecpeq_f16(vepo1);
    float16x8_t vrepo2 = vrecpeq_f16(vepo2);
    float16x8_t vrepo3 = vrecpeq_f16(vepo3);
    const float16x8_t verepo0 = vfmaq_f16(vminus_one, vrepo0, vepo0);
    const float16x8_t verepo1 = vfmaq_f16(vminus_one, vrepo1, vepo1);
    const float16x8_t verepo2 = vfmaq_f16(vminus_one, vrepo2, vepo2);
    const float16x8_t verepo3 = vfmaq_f16(vminus_one, vrepo3, vepo3);
    vrepo0 = vfmsq_f16(vrepo0, vrepo0, verepo0);
    vrepo1 = vfmsq_f16(vrepo1, vrepo1, verepo1);
    vrepo2 = vfmsq_f16(vrepo2, vrepo2, verepo2);
    vrepo3 = vfmsq_f16(vrepo3, vrepo3, verepo3);

    float16x8_t vy0 = vmulq_f16(vemo0, vrepo0);
    float16x8_t vy1 = vmulq_f16(vemo1, vrepo1);
    float16x8_t vy2 = vmulq_f16(vemo2, vrepo2);
    float16x8_t vy3 = vmulq_f16(vemo3, vrepo3);


    vy0 = vbslq_f16(vsign_mask, vx0, vy0);
    vy1 = vbslq_f16(vsign_mask, vx1, vy1);
    vy2 = vbslq_f16(vsign_mask, vx2, vy2);
    vy3 = vbslq_f16(vsign_mask, vx3, vy3);

    vst1q_u16(o, vreinterpretq_u16_f16(vy0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy1)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy2)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy3)); o += 8;
  }
  for (; n >= 1 * sizeof(float16x8_t); n -= 1 * sizeof(float16x8_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float16x8_t vz = vabsq_f16(vx);

    vz = vminq_f16(vz, vsat_cutoff);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);

    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);
    const float16x8_t vt = vfmaq_f16(vz, vn, vln2);
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmsq_f16(vtwo, vp, vt);

    const float16x8_t vts = vmulq_f16(vt, vs);
    const float16x8_t vsmo = vaddq_f16(vs, vminus_one);
    const float16x8_t vemo = vfmsq_f16(vsmo, vp, vts);
    const float16x8_t vepo = vaddq_f16(vemo, vtwo);

    float16x8_t vrepo = vrecpeq_f16(vepo);
    const float16x8_t verepo = vfmaq_f16(vminus_one, vrepo, vepo);
    vrepo = vfmsq_f16(vrepo, vrepo, verepo);

    float16x8_t vy = vmulq_f16(vemo, vrepo);


    vy = vbslq_f16(vsign_mask, vx, vy);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }

  if (n != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vz = vabsq_f16(vx);
    vz = vminq_f16(vz, vsat_cutoff);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);
    const float16x8_t vt = vfmaq_f16(vz, vn, vln2);
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmsq_f16(vtwo, vp, vt);

    const float16x8_t vts = vmulq_f16(vt, vs);
    const float16x8_t vsmo = vaddq_f16(vs, vminus_one);
    const float16x8_t vemo = vfmsq_f16(vsmo, vp, vts);
    const float16x8_t vepo = vaddq_f16(vemo, vtwo);

    float16x8_t vrepo = vrecpeq_f16(vepo);
    const float16x8_t verepo = vfmaq_f16(vminus_one, vrepo, vepo);
    vrepo = vfmsq_f16(vrepo, vrepo, verepo);

    float16x8_t vy = vmulq_f16(vemo, vrepo);


    vy = vbslq_f16(vsign_mask, vx, vy);

    float16x4_t vy_lo = vget_low_f16(vy);
    if (n & 4 * sizeof(uint16_t)) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;
      vy_lo = vget_high_f16(vy);
    }
    if (n & 2 * sizeof(uint16_t)) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o+= 2;
      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }
    if (n & 1 * sizeof(uint16_t)) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}

void xnn_f16_vabs_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc0 = vabsq_f16(vacc0);
    vacc1 = vabsq_f16(vacc1);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vabsq_f16(vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vabsq_f16(vacc);
    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_f16_vneg_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc0 = vnegq_f16(vacc0);
    vacc1 = vnegq_f16(vacc1);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vnegq_f16(vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vnegq_f16(vacc);
    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_f16_vsqr_ukernel__neonfp16arith_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc0 = vmulq_f16(vacc0, vacc0);
    vacc1 = vmulq_f16(vacc1, vacc1);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vmulq_f16(vacc, vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vmulq_f16(vacc, vacc);
    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}

void xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint16_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->fp16arith.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;

  const int8x8_t vmask = vmov_n_s8(INT8_C(0xF0));
  kc = round_up_po2(kc, 2);
  do {
    float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksum89AB = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksumCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vzp0 = vcvtq_f32_s32(vld1q_dup_s32(&quantization_params[0].zero_point));
    float32x4_t vout0x0123 = vmulq_f32(vksum0123, vzp0);
    float32x4_t vout0x4567 = vmulq_f32(vksum4567, vzp0);
    float32x4_t vout0x89AB = vmulq_f32(vksum89AB, vzp0);
    float32x4_t vout0xCDEF = vmulq_f32(vksumCDEF, vzp0);

    for (size_t kb=0; kb < kc; kb += bl) {
      int32x4_t vacc0x0123 = vdupq_n_s32(0);
      int32x4_t vacc0x4567 = vdupq_n_s32(0);
      int32x4_t vacc0x89AB = vdupq_n_s32(0);
      int32x4_t vacc0xCDEF = vdupq_n_s32(0);

      size_t k = bl;
      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int16x8_t vxa0 = vmovl_s8(va0);

        const int8x8_t vb01234567c01 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c23 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c45 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c67 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc67 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
        const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
        const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
        const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);
        const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
        const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
        const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
        const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
        const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
        const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
        const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
        const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
        const int8x8_t vb01234567c6 = vshl_n_s8(vb01234567c67, 4);
        const int8x8_t vb01234567c7 = vand_s8(vb01234567c67, vmask);
        const int8x8_t vb89ABCDEFc6 = vshl_n_s8(vb89ABCDEFc67, 4);
        const int8x8_t vb89ABCDEFc7 = vand_s8(vb89ABCDEFc67, vmask);

        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);
        const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
        const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);
        const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
        const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);
        const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
        const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);
        const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
        const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);
        const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
        const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);
        const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);
        const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);


        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);

        k -= 8 * sizeof(int8_t);
      }
      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int16x8_t vxa0 = vmovl_s8(va0);

        const int8x8_t vb01234567c01 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
        const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
        const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
        const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);

        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);

      if (k >= 2 * sizeof(int8_t)) {
          const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
          const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);

          if (k > 2 * sizeof(int8_t)) {
            const int8x8_t vb01234567c23 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
            const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
            const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
            const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
            const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
            const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);

            if (k >= 4 * sizeof(int8_t)) {
              const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
              const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

              if (k > 4 * sizeof(int8_t)) {
                const int8x8_t vb01234567c45 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
                const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
                const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
                const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
                const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
                const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);

                if (k >= 6 * sizeof(int8_t)) {
                  const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                  const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);

                }
              }
            }
          }
        }
      }

    const float32x4_t vfilter_output_scale0123 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;

    const float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
    const float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
    const float32x4_t vf0x89AB = vcvtq_f32_s32(vacc0x89AB);
    const float32x4_t vf0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      vout0x89AB = vfmaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      vout0xCDEF = vfmaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0x0123 = vmlaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      vout0x4567 = vmlaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      vout0x89AB = vmlaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      vout0xCDEF = vmlaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
    #endif
    }
    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vinput_scale0);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vinput_scale0);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vinput_scale0);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vinput_scale0);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vinput_scale0);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vinput_scale0);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vinput_scale0);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vinput_scale0);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
     }
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint16_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->fp16arith.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const int8x8_t vmask = vmov_n_s8(INT8_C(0xF0));
  kc = round_up_po2(kc, 2);
  do {
    float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksum89AB = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksumCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vzp01 = vcvtq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    float32x4_t vout0x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vzp01), 0);
    float32x4_t vout1x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vzp01), 0);
    float32x4_t vout0x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vzp01), 0);
    float32x4_t vout1x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vzp01), 0);
    float32x4_t vout0x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vzp01), 0);
    float32x4_t vout1x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vzp01), 0);
    float32x4_t vout0xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vzp01), 0);
    float32x4_t vout1xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vzp01), 0);
    const float32x4_t vzp23 = vcvtq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    float32x4_t vout2x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vzp23), 0);
    float32x4_t vout3x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vzp23), 0);
    float32x4_t vout2x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vzp23), 0);
    float32x4_t vout3x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vzp23), 0);
    float32x4_t vout2x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vzp23), 0);
    float32x4_t vout3x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vzp23), 0);
    float32x4_t vout2xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vzp23), 0);
    float32x4_t vout3xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vzp23), 0);
    const float32x4_t vzp45 = vcvtq_f32_s32(vld1q_s32(&quantization_params[4].zero_point));
    float32x4_t vout4x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vzp45), 0);
    float32x4_t vout5x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vzp45), 0);
    float32x4_t vout4x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vzp45), 0);
    float32x4_t vout5x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vzp45), 0);
    float32x4_t vout4x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vzp45), 0);
    float32x4_t vout5x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vzp45), 0);
    float32x4_t vout4xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vzp45), 0);
    float32x4_t vout5xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vzp45), 0);

    for (size_t kb=0; kb < kc; kb += bl) {
      int32x4_t vacc0x0123 = vdupq_n_s32(0);
      int32x4_t vacc0x4567 = vdupq_n_s32(0);
      int32x4_t vacc0x89AB = vdupq_n_s32(0);
      int32x4_t vacc0xCDEF = vdupq_n_s32(0);
      int32x4_t vacc1x0123 = vdupq_n_s32(0);
      int32x4_t vacc1x4567 = vdupq_n_s32(0);
      int32x4_t vacc1x89AB = vdupq_n_s32(0);
      int32x4_t vacc1xCDEF = vdupq_n_s32(0);
      int32x4_t vacc2x0123 = vdupq_n_s32(0);
      int32x4_t vacc2x4567 = vdupq_n_s32(0);
      int32x4_t vacc2x89AB = vdupq_n_s32(0);
      int32x4_t vacc2xCDEF = vdupq_n_s32(0);
      int32x4_t vacc3x0123 = vdupq_n_s32(0);
      int32x4_t vacc3x4567 = vdupq_n_s32(0);
      int32x4_t vacc3x89AB = vdupq_n_s32(0);
      int32x4_t vacc3xCDEF = vdupq_n_s32(0);
      int32x4_t vacc4x0123 = vdupq_n_s32(0);
      int32x4_t vacc4x4567 = vdupq_n_s32(0);
      int32x4_t vacc4x89AB = vdupq_n_s32(0);
      int32x4_t vacc4xCDEF = vdupq_n_s32(0);
      int32x4_t vacc5x0123 = vdupq_n_s32(0);
      int32x4_t vacc5x4567 = vdupq_n_s32(0);
      int32x4_t vacc5x89AB = vdupq_n_s32(0);
      int32x4_t vacc5xCDEF = vdupq_n_s32(0);

      size_t k = bl;
      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int16x8_t vxa0 = vmovl_s8(va0);
        const int8x8_t va1 = vld1_s8(a1); a1 += 8;
        const int16x8_t vxa1 = vmovl_s8(va1);
        const int8x8_t va2 = vld1_s8(a2); a2 += 8;
        const int16x8_t vxa2 = vmovl_s8(va2);
        const int8x8_t va3 = vld1_s8(a3); a3 += 8;
        const int16x8_t vxa3 = vmovl_s8(va3);
        const int8x8_t va4 = vld1_s8(a4); a4 += 8;
        const int16x8_t vxa4 = vmovl_s8(va4);
        const int8x8_t va5 = vld1_s8(a5); a5 += 8;
        const int16x8_t vxa5 = vmovl_s8(va5);

        const int8x8_t vb01234567c01 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c23 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c45 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c67 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc67 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
        const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
        const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
        const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);
        const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
        const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
        const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
        const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
        const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
        const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
        const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
        const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
        const int8x8_t vb01234567c6 = vshl_n_s8(vb01234567c67, 4);
        const int8x8_t vb01234567c7 = vand_s8(vb01234567c67, vmask);
        const int8x8_t vb89ABCDEFc6 = vshl_n_s8(vb89ABCDEFc67, 4);
        const int8x8_t vb89ABCDEFc7 = vand_s8(vb89ABCDEFc67, vmask);

        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);
        const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
        const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);
        const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
        const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);
        const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
        const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);
        const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
        const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);
        const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
        const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);
        const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);
        const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);


        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa4), 2);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa4), 2);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa5), 2);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa5), 2);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa4), 3);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa4), 3);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa5), 3);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa5), 3);

        k -= 8 * sizeof(int8_t);
      }
      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int16x8_t vxa0 = vmovl_s8(va0);
        const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
        const int16x8_t vxa1 = vmovl_s8(va1);
        const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
        const int16x8_t vxa2 = vmovl_s8(va2);
        const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);
        const int16x8_t vxa3 = vmovl_s8(va3);
        const int8x8_t va4 = vld1_s8(a4); a4 = (const int8_t*) ((uintptr_t) a4 + k);
        const int16x8_t vxa4 = vmovl_s8(va4);
        const int8x8_t va5 = vld1_s8(a5); a5 = (const int8_t*) ((uintptr_t) a5 + k);
        const int16x8_t vxa5 = vmovl_s8(va5);

        const int8x8_t vb01234567c01 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
        const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
        const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
        const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);

        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);

      if (k >= 2 * sizeof(int8_t)) {
          const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
          const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
          vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
          vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
          vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
          vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
          vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
          vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
          vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
          vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
          vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
          vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
          vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
          vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
          vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
          vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
          vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
          vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
          vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
          vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
          vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);

          if (k > 2 * sizeof(int8_t)) {
            const int8x8_t vb01234567c23 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
            const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
            const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
            const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
            const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
            const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
            vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
            vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
            vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
            vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
            vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
            vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
            vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
            vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
            vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
            vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);

            if (k >= 4 * sizeof(int8_t)) {
              const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
              const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
              vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
              vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
              vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
              vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
              vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
              vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
              vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
              vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
              vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
              vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
              vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
              vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
              vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
              vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
              vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
              vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
              vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
              vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);
              vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);

              if (k > 4 * sizeof(int8_t)) {
                const int8x8_t vb01234567c45 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
                const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
                const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
                const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
                const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
                const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
                vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
                vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
                vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
                vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
                vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
                vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
                vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
                vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
                vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
                vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
                vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
                vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
                vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
                vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
                vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
                vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
                vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);

                if (k >= 6 * sizeof(int8_t)) {
                  const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                  const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                  vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                  vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                  vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                  vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                  vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                  vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                  vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                  vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                  vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                  vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
                  vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
                  vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
                  vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
                  vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
                  vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
                  vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
                  vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
                  vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
                  vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);

                }
              }
            }
          }
        }
      }

    const float32x4_t vfilter_output_scale0123 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;

    const float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
    const float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
    const float32x4_t vf0x89AB = vcvtq_f32_s32(vacc0x89AB);
    const float32x4_t vf0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      vout0x89AB = vfmaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      vout0xCDEF = vfmaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0x0123 = vmlaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      vout0x4567 = vmlaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      vout0x89AB = vmlaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      vout0xCDEF = vmlaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
    #endif
    const float32x4_t vf1x0123 = vcvtq_f32_s32(vacc1x0123);
    const float32x4_t vf1x4567 = vcvtq_f32_s32(vacc1x4567);
    const float32x4_t vf1x89AB = vcvtq_f32_s32(vacc1x89AB);
    const float32x4_t vf1xCDEF = vcvtq_f32_s32(vacc1xCDEF);

    #if XNN_ARCH_ARM64
      vout1x0123 = vfmaq_f32(vout1x0123, vf1x0123, vfilter_output_scale0123);
      vout1x4567 = vfmaq_f32(vout1x4567, vf1x4567, vfilter_output_scale4567);
      vout1x89AB = vfmaq_f32(vout1x89AB, vf1x89AB, vfilter_output_scale89AB);
      vout1xCDEF = vfmaq_f32(vout1xCDEF, vf1xCDEF, vfilter_output_scaleCDEF);
    #else
      vout1x0123 = vmlaq_f32(vout1x0123, vf1x0123, vfilter_output_scale0123);
      vout1x4567 = vmlaq_f32(vout1x4567, vf1x4567, vfilter_output_scale4567);
      vout1x89AB = vmlaq_f32(vout1x89AB, vf1x89AB, vfilter_output_scale89AB);
      vout1xCDEF = vmlaq_f32(vout1xCDEF, vf1xCDEF, vfilter_output_scaleCDEF);
    #endif
    const float32x4_t vf2x0123 = vcvtq_f32_s32(vacc2x0123);
    const float32x4_t vf2x4567 = vcvtq_f32_s32(vacc2x4567);
    const float32x4_t vf2x89AB = vcvtq_f32_s32(vacc2x89AB);
    const float32x4_t vf2xCDEF = vcvtq_f32_s32(vacc2xCDEF);

    #if XNN_ARCH_ARM64
      vout2x0123 = vfmaq_f32(vout2x0123, vf2x0123, vfilter_output_scale0123);
      vout2x4567 = vfmaq_f32(vout2x4567, vf2x4567, vfilter_output_scale4567);
      vout2x89AB = vfmaq_f32(vout2x89AB, vf2x89AB, vfilter_output_scale89AB);
      vout2xCDEF = vfmaq_f32(vout2xCDEF, vf2xCDEF, vfilter_output_scaleCDEF);
    #else
      vout2x0123 = vmlaq_f32(vout2x0123, vf2x0123, vfilter_output_scale0123);
      vout2x4567 = vmlaq_f32(vout2x4567, vf2x4567, vfilter_output_scale4567);
      vout2x89AB = vmlaq_f32(vout2x89AB, vf2x89AB, vfilter_output_scale89AB);
      vout2xCDEF = vmlaq_f32(vout2xCDEF, vf2xCDEF, vfilter_output_scaleCDEF);
    #endif
    const float32x4_t vf3x0123 = vcvtq_f32_s32(vacc3x0123);
    const float32x4_t vf3x4567 = vcvtq_f32_s32(vacc3x4567);
    const float32x4_t vf3x89AB = vcvtq_f32_s32(vacc3x89AB);
    const float32x4_t vf3xCDEF = vcvtq_f32_s32(vacc3xCDEF);

    #if XNN_ARCH_ARM64
      vout3x0123 = vfmaq_f32(vout3x0123, vf3x0123, vfilter_output_scale0123);
      vout3x4567 = vfmaq_f32(vout3x4567, vf3x4567, vfilter_output_scale4567);
      vout3x89AB = vfmaq_f32(vout3x89AB, vf3x89AB, vfilter_output_scale89AB);
      vout3xCDEF = vfmaq_f32(vout3xCDEF, vf3xCDEF, vfilter_output_scaleCDEF);
    #else
      vout3x0123 = vmlaq_f32(vout3x0123, vf3x0123, vfilter_output_scale0123);
      vout3x4567 = vmlaq_f32(vout3x4567, vf3x4567, vfilter_output_scale4567);
      vout3x89AB = vmlaq_f32(vout3x89AB, vf3x89AB, vfilter_output_scale89AB);
      vout3xCDEF = vmlaq_f32(vout3xCDEF, vf3xCDEF, vfilter_output_scaleCDEF);
    #endif
    const float32x4_t vf4x0123 = vcvtq_f32_s32(vacc4x0123);
    const float32x4_t vf4x4567 = vcvtq_f32_s32(vacc4x4567);
    const float32x4_t vf4x89AB = vcvtq_f32_s32(vacc4x89AB);
    const float32x4_t vf4xCDEF = vcvtq_f32_s32(vacc4xCDEF);

    #if XNN_ARCH_ARM64
      vout4x0123 = vfmaq_f32(vout4x0123, vf4x0123, vfilter_output_scale0123);
      vout4x4567 = vfmaq_f32(vout4x4567, vf4x4567, vfilter_output_scale4567);
      vout4x89AB = vfmaq_f32(vout4x89AB, vf4x89AB, vfilter_output_scale89AB);
      vout4xCDEF = vfmaq_f32(vout4xCDEF, vf4xCDEF, vfilter_output_scaleCDEF);
    #else
      vout4x0123 = vmlaq_f32(vout4x0123, vf4x0123, vfilter_output_scale0123);
      vout4x4567 = vmlaq_f32(vout4x4567, vf4x4567, vfilter_output_scale4567);
      vout4x89AB = vmlaq_f32(vout4x89AB, vf4x89AB, vfilter_output_scale89AB);
      vout4xCDEF = vmlaq_f32(vout4xCDEF, vf4xCDEF, vfilter_output_scaleCDEF);
    #endif
    const float32x4_t vf5x0123 = vcvtq_f32_s32(vacc5x0123);
    const float32x4_t vf5x4567 = vcvtq_f32_s32(vacc5x4567);
    const float32x4_t vf5x89AB = vcvtq_f32_s32(vacc5x89AB);
    const float32x4_t vf5xCDEF = vcvtq_f32_s32(vacc5xCDEF);

    #if XNN_ARCH_ARM64
      vout5x0123 = vfmaq_f32(vout5x0123, vf5x0123, vfilter_output_scale0123);
      vout5x4567 = vfmaq_f32(vout5x4567, vf5x4567, vfilter_output_scale4567);
      vout5x89AB = vfmaq_f32(vout5x89AB, vf5x89AB, vfilter_output_scale89AB);
      vout5xCDEF = vfmaq_f32(vout5xCDEF, vf5xCDEF, vfilter_output_scaleCDEF);
    #else
      vout5x0123 = vmlaq_f32(vout5x0123, vf5x0123, vfilter_output_scale0123);
      vout5x4567 = vmlaq_f32(vout5x4567, vf5x4567, vfilter_output_scale4567);
      vout5x89AB = vmlaq_f32(vout5x89AB, vf5x89AB, vfilter_output_scale89AB);
      vout5xCDEF = vmlaq_f32(vout5xCDEF, vf5xCDEF, vfilter_output_scaleCDEF);
    #endif
    }
    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    const float32x4_t vinput_scale45 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[4].zero_point));

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_lane_f32(vbias0123, vout0x0123, vget_low_f32(vinput_scale01), 1);
      vout1x0123 = vfmaq_lane_f32(vbias0123, vout1x0123, vget_high_f32(vinput_scale01), 1);
      vout2x0123 = vfmaq_lane_f32(vbias0123, vout2x0123, vget_low_f32(vinput_scale23), 1);
      vout3x0123 = vfmaq_lane_f32(vbias0123, vout3x0123, vget_high_f32(vinput_scale23), 1);
      vout4x0123 = vfmaq_lane_f32(vbias0123, vout4x0123, vget_low_f32(vinput_scale45), 1);
      vout5x0123 = vfmaq_lane_f32(vbias0123, vout5x0123, vget_high_f32(vinput_scale45), 1);
    #else
      vout0x0123 = vmlaq_lane_f32(vbias0123, vout0x0123, vget_low_f32(vinput_scale01), 1);
      vout1x0123 = vmlaq_lane_f32(vbias0123, vout1x0123, vget_high_f32(vinput_scale01), 1);
      vout2x0123 = vmlaq_lane_f32(vbias0123, vout2x0123, vget_low_f32(vinput_scale23), 1);
      vout3x0123 = vmlaq_lane_f32(vbias0123, vout3x0123, vget_high_f32(vinput_scale23), 1);
      vout4x0123 = vmlaq_lane_f32(vbias0123, vout4x0123, vget_low_f32(vinput_scale45), 1);
      vout5x0123 = vmlaq_lane_f32(vbias0123, vout5x0123, vget_high_f32(vinput_scale45), 1);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_lane_f32(vbias4567, vout0x4567, vget_low_f32(vinput_scale01), 1);
      vout1x4567 = vfmaq_lane_f32(vbias4567, vout1x4567, vget_high_f32(vinput_scale01), 1);
      vout2x4567 = vfmaq_lane_f32(vbias4567, vout2x4567, vget_low_f32(vinput_scale23), 1);
      vout3x4567 = vfmaq_lane_f32(vbias4567, vout3x4567, vget_high_f32(vinput_scale23), 1);
      vout4x4567 = vfmaq_lane_f32(vbias4567, vout4x4567, vget_low_f32(vinput_scale45), 1);
      vout5x4567 = vfmaq_lane_f32(vbias4567, vout5x4567, vget_high_f32(vinput_scale45), 1);
    #else
      vout0x4567 = vmlaq_lane_f32(vbias4567, vout0x4567, vget_low_f32(vinput_scale01), 1);
      vout1x4567 = vmlaq_lane_f32(vbias4567, vout1x4567, vget_high_f32(vinput_scale01), 1);
      vout2x4567 = vmlaq_lane_f32(vbias4567, vout2x4567, vget_low_f32(vinput_scale23), 1);
      vout3x4567 = vmlaq_lane_f32(vbias4567, vout3x4567, vget_high_f32(vinput_scale23), 1);
      vout4x4567 = vmlaq_lane_f32(vbias4567, vout4x4567, vget_low_f32(vinput_scale45), 1);
      vout5x4567 = vmlaq_lane_f32(vbias4567, vout5x4567, vget_high_f32(vinput_scale45), 1);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_lane_f32(vbias89AB, vout0x89AB, vget_low_f32(vinput_scale01), 1);
      vout1x89AB = vfmaq_lane_f32(vbias89AB, vout1x89AB, vget_high_f32(vinput_scale01), 1);
      vout2x89AB = vfmaq_lane_f32(vbias89AB, vout2x89AB, vget_low_f32(vinput_scale23), 1);
      vout3x89AB = vfmaq_lane_f32(vbias89AB, vout3x89AB, vget_high_f32(vinput_scale23), 1);
      vout4x89AB = vfmaq_lane_f32(vbias89AB, vout4x89AB, vget_low_f32(vinput_scale45), 1);
      vout5x89AB = vfmaq_lane_f32(vbias89AB, vout5x89AB, vget_high_f32(vinput_scale45), 1);
    #else
      vout0x89AB = vmlaq_lane_f32(vbias89AB, vout0x89AB, vget_low_f32(vinput_scale01), 1);
      vout1x89AB = vmlaq_lane_f32(vbias89AB, vout1x89AB, vget_high_f32(vinput_scale01), 1);
      vout2x89AB = vmlaq_lane_f32(vbias89AB, vout2x89AB, vget_low_f32(vinput_scale23), 1);
      vout3x89AB = vmlaq_lane_f32(vbias89AB, vout3x89AB, vget_high_f32(vinput_scale23), 1);
      vout4x89AB = vmlaq_lane_f32(vbias89AB, vout4x89AB, vget_low_f32(vinput_scale45), 1);
      vout5x89AB = vmlaq_lane_f32(vbias89AB, vout5x89AB, vget_high_f32(vinput_scale45), 1);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_lane_f32(vbiasCDEF, vout0xCDEF, vget_low_f32(vinput_scale01), 1);
      vout1xCDEF = vfmaq_lane_f32(vbiasCDEF, vout1xCDEF, vget_high_f32(vinput_scale01), 1);
      vout2xCDEF = vfmaq_lane_f32(vbiasCDEF, vout2xCDEF, vget_low_f32(vinput_scale23), 1);
      vout3xCDEF = vfmaq_lane_f32(vbiasCDEF, vout3xCDEF, vget_high_f32(vinput_scale23), 1);
      vout4xCDEF = vfmaq_lane_f32(vbiasCDEF, vout4xCDEF, vget_low_f32(vinput_scale45), 1);
      vout5xCDEF = vfmaq_lane_f32(vbiasCDEF, vout5xCDEF, vget_high_f32(vinput_scale45), 1);
    #else
      vout0xCDEF = vmlaq_lane_f32(vbiasCDEF, vout0xCDEF, vget_low_f32(vinput_scale01), 1);
      vout1xCDEF = vmlaq_lane_f32(vbiasCDEF, vout1xCDEF, vget_high_f32(vinput_scale01), 1);
      vout2xCDEF = vmlaq_lane_f32(vbiasCDEF, vout2xCDEF, vget_low_f32(vinput_scale23), 1);
      vout3xCDEF = vmlaq_lane_f32(vbiasCDEF, vout3xCDEF, vget_high_f32(vinput_scale23), 1);
      vout4xCDEF = vmlaq_lane_f32(vbiasCDEF, vout4xCDEF, vget_low_f32(vinput_scale45), 1);
      vout5xCDEF = vmlaq_lane_f32(vbiasCDEF, vout5xCDEF, vget_high_f32(vinput_scale45), 1);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    float16x8_t vfp16out1x01234567 = vcombine_f16(vcvt_f16_f32(vout1x0123), vcvt_f16_f32(vout1x4567));
    float16x8_t vfp16out1x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout1x89AB), vcvt_f16_f32(vout1xCDEF));
    float16x8_t vfp16out2x01234567 = vcombine_f16(vcvt_f16_f32(vout2x0123), vcvt_f16_f32(vout2x4567));
    float16x8_t vfp16out2x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout2x89AB), vcvt_f16_f32(vout2xCDEF));
    float16x8_t vfp16out3x01234567 = vcombine_f16(vcvt_f16_f32(vout3x0123), vcvt_f16_f32(vout3x4567));
    float16x8_t vfp16out3x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout3x89AB), vcvt_f16_f32(vout3xCDEF));
    float16x8_t vfp16out4x01234567 = vcombine_f16(vcvt_f16_f32(vout4x0123), vcvt_f16_f32(vout4x4567));
    float16x8_t vfp16out4x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout4x89AB), vcvt_f16_f32(vout4xCDEF));
    float16x8_t vfp16out5x01234567 = vcombine_f16(vcvt_f16_f32(vout5x0123), vcvt_f16_f32(vout5x4567));
    float16x8_t vfp16out5x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout5x89AB), vcvt_f16_f32(vout5xCDEF));
    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    vfp16out1x01234567 = vmaxq_f16(vfp16out1x01234567, voutput_min);
    vfp16out1x89ABCDEF = vmaxq_f16(vfp16out1x89ABCDEF, voutput_min);
    vfp16out2x01234567 = vmaxq_f16(vfp16out2x01234567, voutput_min);
    vfp16out2x89ABCDEF = vmaxq_f16(vfp16out2x89ABCDEF, voutput_min);
    vfp16out3x01234567 = vmaxq_f16(vfp16out3x01234567, voutput_min);
    vfp16out3x89ABCDEF = vmaxq_f16(vfp16out3x89ABCDEF, voutput_min);
    vfp16out4x01234567 = vmaxq_f16(vfp16out4x01234567, voutput_min);
    vfp16out4x89ABCDEF = vmaxq_f16(vfp16out4x89ABCDEF, voutput_min);
    vfp16out5x01234567 = vmaxq_f16(vfp16out5x01234567, voutput_min);
    vfp16out5x89ABCDEF = vmaxq_f16(vfp16out5x89ABCDEF, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);
    vfp16out1x01234567 = vminq_f16(vfp16out1x01234567, voutput_max);
    vfp16out1x89ABCDEF = vminq_f16(vfp16out1x89ABCDEF, voutput_max);
    vfp16out2x01234567 = vminq_f16(vfp16out2x01234567, voutput_max);
    vfp16out2x89ABCDEF = vminq_f16(vfp16out2x89ABCDEF, voutput_max);
    vfp16out3x01234567 = vminq_f16(vfp16out3x01234567, voutput_max);
    vfp16out3x89ABCDEF = vminq_f16(vfp16out3x89ABCDEF, voutput_max);
    vfp16out4x01234567 = vminq_f16(vfp16out4x01234567, voutput_max);
    vfp16out4x89ABCDEF = vminq_f16(vfp16out4x89ABCDEF, voutput_max);
    vfp16out5x01234567 = vminq_f16(vfp16out5x01234567, voutput_max);
    vfp16out5x89ABCDEF = vminq_f16(vfp16out5x89ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));
      vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vfp16out1x89ABCDEF));
      vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vfp16out2x89ABCDEF));
      vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vfp16out3x89ABCDEF));
      vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567));
      vst1q_u16(c4 + 8, vreinterpretq_u16_f16(vfp16out4x89ABCDEF));
      vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567));
      vst1q_u16(c5 + 8, vreinterpretq_u16_f16(vfp16out5x89ABCDEF));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567)); c1 += 8;
       vfp16out1x01234567 = vfp16out1x89ABCDEF;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567)); c2 += 8;
       vfp16out2x01234567 = vfp16out2x89ABCDEF;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567)); c3 += 8;
       vfp16out3x01234567 = vfp16out3x89ABCDEF;
       vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567)); c4 += 8;
       vfp16out4x01234567 = vfp16out4x89ABCDEF;
       vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567)); c5 += 8;
       vfp16out5x01234567 = vfp16out5x89ABCDEF;
     }
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     float16x4_t vfp16out1x0123 = vget_low_f16(vfp16out1x01234567);
     float16x4_t vfp16out2x0123 = vget_low_f16(vfp16out2x01234567);
     float16x4_t vfp16out3x0123 = vget_low_f16(vfp16out3x01234567);
     float16x4_t vfp16out4x0123 = vget_low_f16(vfp16out4x01234567);
     float16x4_t vfp16out5x0123 = vget_low_f16(vfp16out5x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vst1_u16(c1, vreinterpret_u16_f16(vfp16out1x0123)); c1 += 4;
       vst1_u16(c2, vreinterpret_u16_f16(vfp16out2x0123)); c2 += 4;
       vst1_u16(c3, vreinterpret_u16_f16(vfp16out3x0123)); c3 += 4;
       vst1_u16(c4, vreinterpret_u16_f16(vfp16out4x0123)); c4 += 4;
       vst1_u16(c5, vreinterpret_u16_f16(vfp16out5x0123)); c5 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
       vfp16out1x0123 = vget_high_f16(vfp16out1x01234567);
       vfp16out2x0123 = vget_high_f16(vfp16out2x01234567);
       vfp16out3x0123 = vget_high_f16(vfp16out3x01234567);
       vfp16out4x0123 = vget_high_f16(vfp16out4x01234567);
       vfp16out5x0123 = vget_high_f16(vfp16out5x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vfp16out1x0123), 0); c1 += 2;
       vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vfp16out2x0123), 0); c2 += 2;
       vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vfp16out3x0123), 0); c3 += 2;
       vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vfp16out4x0123), 0); c4 += 2;
       vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vfp16out5x0123), 0); c5 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
       vfp16out1x0123 = vext_f16(vfp16out1x0123, vfp16out1x0123, 2);
       vfp16out2x0123 = vext_f16(vfp16out2x0123, vfp16out2x0123, 2);
       vfp16out3x0123 = vext_f16(vfp16out3x0123, vfp16out3x0123, 2);
       vfp16out4x0123 = vext_f16(vfp16out4x0123, vfp16out4x0123, 2);
       vfp16out5x0123 = vext_f16(vfp16out5x0123, vfp16out5x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
       vst1_lane_u16(c1, vreinterpret_u16_f16(vfp16out1x0123), 0);
       vst1_lane_u16(c2, vreinterpret_u16_f16(vfp16out2x0123), 0);
       vst1_lane_u16(c3, vreinterpret_u16_f16(vfp16out3x0123), 0);
       vst1_lane_u16(c4, vreinterpret_u16_f16(vfp16out4x0123), 0);
       vst1_lane_u16(c5, vreinterpret_u16_f16(vfp16out5x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;

  const int8x8_t vmask = vmov_n_s8(INT8_C(0xF0));
  kc = round_up_po2(kc, 2);
  do {
    int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vzp0 = vld1q_dup_s32(&quantization_params[0].zero_point);
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vzp0);
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vzp0);
    int32x4_t vacc0x89AB = vmulq_s32(vksum89AB, vzp0);
    int32x4_t vacc0xCDEF = vmulq_s32(vksumCDEF, vzp0);

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int16x8_t vxa0 = vmovl_s8(va0);

      const int8x8_t vb01234567c01 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c23 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c45 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c67 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc67 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
      const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
      const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
      const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);
      const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
      const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
      const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
      const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
      const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
      const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
      const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
      const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
      const int8x8_t vb01234567c6 = vshl_n_s8(vb01234567c67, 4);
      const int8x8_t vb01234567c7 = vand_s8(vb01234567c67, vmask);
      const int8x8_t vb89ABCDEFc6 = vshl_n_s8(vb89ABCDEFc67, 4);
      const int8x8_t vb89ABCDEFc7 = vand_s8(vb89ABCDEFc67, vmask);

      const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
      const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);
      const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
      const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);
      const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
      const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);
      const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
      const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);
      const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
      const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);
      const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
      const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);
      const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
      const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);
      const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);
      const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);


      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);

      k -= 8 * sizeof(int8_t);
    }
    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int16x8_t vxa0 = vmovl_s8(va0);

      const int8x8_t vb01234567c01 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
      const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
      const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
      const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);

      const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
      const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);

      if (k >= 2 * sizeof(int8_t)) {
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);

        if (k > 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c23 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
          const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
          const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
          const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
          const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
          const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);

          if (k >= 4 * sizeof(int8_t)) {
            const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
            const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

            if (k > 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c45 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
              const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
              const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
              const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
              const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
              const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);

              if (k >= 6 * sizeof(int8_t)) {
                const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);

              }
            }
          }
        }
      }
    }

    float32x4_t vout0x0123 = vcvtq_n_f32_s32(vacc0x0123, 4);
    float32x4_t vout0x4567 = vcvtq_n_f32_s32(vacc0x4567, 4);
    float32x4_t vout0x89AB = vcvtq_n_f32_s32(vacc0x89AB, 4);
    float32x4_t vout0xCDEF = vcvtq_n_f32_s32(vacc0xCDEF, 4);

    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale0);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale0);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
     }
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const int8x8_t vmask = vmov_n_s8(INT8_C(0xF0));
  kc = round_up_po2(kc, 2);
  do {
    int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vzp01 = vld1q_s32(&quantization_params[0].zero_point);
    int32x4_t vacc0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vzp01), 0);
    int32x4_t vacc1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vzp01), 0);
    int32x4_t vacc0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vzp01), 0);
    int32x4_t vacc1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vzp01), 0);
    int32x4_t vacc0x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vzp01), 0);
    int32x4_t vacc1x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vzp01), 0);
    int32x4_t vacc0xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vzp01), 0);
    int32x4_t vacc1xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vzp01), 0);
    const int32x4_t vzp23 = vld1q_s32(&quantization_params[2].zero_point);
    int32x4_t vacc2x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vzp23), 0);
    int32x4_t vacc3x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vzp23), 0);
    int32x4_t vacc2x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vzp23), 0);
    int32x4_t vacc3x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vzp23), 0);
    int32x4_t vacc2x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vzp23), 0);
    int32x4_t vacc3x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vzp23), 0);
    int32x4_t vacc2xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vzp23), 0);
    int32x4_t vacc3xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vzp23), 0);
    const int32x4_t vzp45 = vld1q_s32(&quantization_params[4].zero_point);
    int32x4_t vacc4x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vzp45), 0);
    int32x4_t vacc5x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vzp45), 0);
    int32x4_t vacc4x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vzp45), 0);
    int32x4_t vacc5x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vzp45), 0);
    int32x4_t vacc4x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vzp45), 0);
    int32x4_t vacc5x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vzp45), 0);
    int32x4_t vacc4xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vzp45), 0);
    int32x4_t vacc5xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vzp45), 0);

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int16x8_t vxa0 = vmovl_s8(va0);
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;
      const int16x8_t vxa1 = vmovl_s8(va1);
      const int8x8_t va2 = vld1_s8(a2); a2 += 8;
      const int16x8_t vxa2 = vmovl_s8(va2);
      const int8x8_t va3 = vld1_s8(a3); a3 += 8;
      const int16x8_t vxa3 = vmovl_s8(va3);
      const int8x8_t va4 = vld1_s8(a4); a4 += 8;
      const int16x8_t vxa4 = vmovl_s8(va4);
      const int8x8_t va5 = vld1_s8(a5); a5 += 8;
      const int16x8_t vxa5 = vmovl_s8(va5);

      const int8x8_t vb01234567c01 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c23 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c45 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c67 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc67 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
      const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
      const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
      const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);
      const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
      const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
      const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
      const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
      const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
      const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
      const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
      const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
      const int8x8_t vb01234567c6 = vshl_n_s8(vb01234567c67, 4);
      const int8x8_t vb01234567c7 = vand_s8(vb01234567c67, vmask);
      const int8x8_t vb89ABCDEFc6 = vshl_n_s8(vb89ABCDEFc67, 4);
      const int8x8_t vb89ABCDEFc7 = vand_s8(vb89ABCDEFc67, vmask);

      const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
      const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);
      const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
      const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);
      const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
      const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);
      const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
      const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);
      const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
      const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);
      const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
      const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);
      const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
      const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);
      const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);
      const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);


      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa4), 2);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa4), 2);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa5), 2);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa5), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa4), 3);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa4), 3);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa5), 3);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa5), 3);

      k -= 8 * sizeof(int8_t);
    }
    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int16x8_t vxa0 = vmovl_s8(va0);
      const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
      const int16x8_t vxa1 = vmovl_s8(va1);
      const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
      const int16x8_t vxa2 = vmovl_s8(va2);
      const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);
      const int16x8_t vxa3 = vmovl_s8(va3);
      const int8x8_t va4 = vld1_s8(a4); a4 = (const int8_t*) ((uintptr_t) a4 + k);
      const int16x8_t vxa4 = vmovl_s8(va4);
      const int8x8_t va5 = vld1_s8(a5); a5 = (const int8_t*) ((uintptr_t) a5 + k);
      const int16x8_t vxa5 = vmovl_s8(va5);

      const int8x8_t vb01234567c01 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
      const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
      const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
      const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);

      const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
      const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);

      if (k >= 2 * sizeof(int8_t)) {
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);

        if (k > 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c23 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
          const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
          const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
          const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
          const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
          const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
          vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
          vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
          vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
          vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
          vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
          vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
          vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
          vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
          vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
          vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
          vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
          vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
          vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
          vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
          vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
          vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
          vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
          vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
          vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
          vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);

          if (k >= 4 * sizeof(int8_t)) {
            const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
            const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
            vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
            vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
            vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
            vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
            vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
            vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
            vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
            vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
            vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);
            vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);

            if (k > 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c45 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
              const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
              const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
              const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
              const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
              const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
              vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
              vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
              vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
              vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
              vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
              vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
              vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
              vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
              vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
              vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
              vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
              vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
              vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
              vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
              vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
              vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
              vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
              vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
              vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
              vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);

              if (k >= 6 * sizeof(int8_t)) {
                const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
                vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
                vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
                vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
                vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
                vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
                vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
                vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
                vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
                vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);

              }
            }
          }
        }
      }
    }

    float32x4_t vout0x0123 = vcvtq_n_f32_s32(vacc0x0123, 4);
    float32x4_t vout0x4567 = vcvtq_n_f32_s32(vacc0x4567, 4);
    float32x4_t vout0x89AB = vcvtq_n_f32_s32(vacc0x89AB, 4);
    float32x4_t vout0xCDEF = vcvtq_n_f32_s32(vacc0xCDEF, 4);
    float32x4_t vout1x0123 = vcvtq_n_f32_s32(vacc1x0123, 4);
    float32x4_t vout1x4567 = vcvtq_n_f32_s32(vacc1x4567, 4);
    float32x4_t vout1x89AB = vcvtq_n_f32_s32(vacc1x89AB, 4);
    float32x4_t vout1xCDEF = vcvtq_n_f32_s32(vacc1xCDEF, 4);
    float32x4_t vout2x0123 = vcvtq_n_f32_s32(vacc2x0123, 4);
    float32x4_t vout2x4567 = vcvtq_n_f32_s32(vacc2x4567, 4);
    float32x4_t vout2x89AB = vcvtq_n_f32_s32(vacc2x89AB, 4);
    float32x4_t vout2xCDEF = vcvtq_n_f32_s32(vacc2xCDEF, 4);
    float32x4_t vout3x0123 = vcvtq_n_f32_s32(vacc3x0123, 4);
    float32x4_t vout3x4567 = vcvtq_n_f32_s32(vacc3x4567, 4);
    float32x4_t vout3x89AB = vcvtq_n_f32_s32(vacc3x89AB, 4);
    float32x4_t vout3xCDEF = vcvtq_n_f32_s32(vacc3xCDEF, 4);
    float32x4_t vout4x0123 = vcvtq_n_f32_s32(vacc4x0123, 4);
    float32x4_t vout4x4567 = vcvtq_n_f32_s32(vacc4x4567, 4);
    float32x4_t vout4x89AB = vcvtq_n_f32_s32(vacc4x89AB, 4);
    float32x4_t vout4xCDEF = vcvtq_n_f32_s32(vacc4xCDEF, 4);
    float32x4_t vout5x0123 = vcvtq_n_f32_s32(vacc5x0123, 4);
    float32x4_t vout5x4567 = vcvtq_n_f32_s32(vacc5x4567, 4);
    float32x4_t vout5x89AB = vcvtq_n_f32_s32(vacc5x89AB, 4);
    float32x4_t vout5xCDEF = vcvtq_n_f32_s32(vacc5xCDEF, 4);

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    vout0x89AB = vmulq_lane_f32(vout0x89AB, vget_low_f32(vinput_scale01), 1);
    vout1x89AB = vmulq_lane_f32(vout1x89AB, vget_high_f32(vinput_scale01), 1);
    vout0xCDEF = vmulq_lane_f32(vout0xCDEF, vget_low_f32(vinput_scale01), 1);
    vout1xCDEF = vmulq_lane_f32(vout1xCDEF, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    vout2x4567 = vmulq_lane_f32(vout2x4567, vget_low_f32(vinput_scale23), 1);
    vout3x4567 = vmulq_lane_f32(vout3x4567, vget_high_f32(vinput_scale23), 1);
    vout2x89AB = vmulq_lane_f32(vout2x89AB, vget_low_f32(vinput_scale23), 1);
    vout3x89AB = vmulq_lane_f32(vout3x89AB, vget_high_f32(vinput_scale23), 1);
    vout2xCDEF = vmulq_lane_f32(vout2xCDEF, vget_low_f32(vinput_scale23), 1);
    vout3xCDEF = vmulq_lane_f32(vout3xCDEF, vget_high_f32(vinput_scale23), 1);
    const float32x4_t vinput_scale45 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[4].zero_point));
    vout4x0123 = vmulq_lane_f32(vout4x0123, vget_low_f32(vinput_scale45), 1);
    vout5x0123 = vmulq_lane_f32(vout5x0123, vget_high_f32(vinput_scale45), 1);
    vout4x4567 = vmulq_lane_f32(vout4x4567, vget_low_f32(vinput_scale45), 1);
    vout5x4567 = vmulq_lane_f32(vout5x4567, vget_high_f32(vinput_scale45), 1);
    vout4x89AB = vmulq_lane_f32(vout4x89AB, vget_low_f32(vinput_scale45), 1);
    vout5x89AB = vmulq_lane_f32(vout5x89AB, vget_high_f32(vinput_scale45), 1);
    vout4xCDEF = vmulq_lane_f32(vout4xCDEF, vget_low_f32(vinput_scale45), 1);
    vout5xCDEF = vmulq_lane_f32(vout5xCDEF, vget_high_f32(vinput_scale45), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      vout4x0123 = vfmaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
      vout5x0123 = vfmaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
      vout4x0123 = vmlaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
      vout5x0123 = vmlaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      vout4x4567 = vfmaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
      vout5x4567 = vfmaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
      vout4x4567 = vmlaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
      vout5x4567 = vmlaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
      vout4x89AB = vfmaq_f32(vbias89AB, vout4x89AB, vfilter_output_scale89AB);
      vout5x89AB = vfmaq_f32(vbias89AB, vout5x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
      vout4x89AB = vmlaq_f32(vbias89AB, vout4x89AB, vfilter_output_scale89AB);
      vout5x89AB = vmlaq_f32(vbias89AB, vout5x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
      vout4xCDEF = vfmaq_f32(vbiasCDEF, vout4xCDEF, vfilter_output_scaleCDEF);
      vout5xCDEF = vfmaq_f32(vbiasCDEF, vout5xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
      vout4xCDEF = vmlaq_f32(vbiasCDEF, vout4xCDEF, vfilter_output_scaleCDEF);
      vout5xCDEF = vmlaq_f32(vbiasCDEF, vout5xCDEF, vfilter_output_scaleCDEF);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    float16x8_t vfp16out1x01234567 = vcombine_f16(vcvt_f16_f32(vout1x0123), vcvt_f16_f32(vout1x4567));
    float16x8_t vfp16out1x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout1x89AB), vcvt_f16_f32(vout1xCDEF));
    float16x8_t vfp16out2x01234567 = vcombine_f16(vcvt_f16_f32(vout2x0123), vcvt_f16_f32(vout2x4567));
    float16x8_t vfp16out2x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout2x89AB), vcvt_f16_f32(vout2xCDEF));
    float16x8_t vfp16out3x01234567 = vcombine_f16(vcvt_f16_f32(vout3x0123), vcvt_f16_f32(vout3x4567));
    float16x8_t vfp16out3x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout3x89AB), vcvt_f16_f32(vout3xCDEF));
    float16x8_t vfp16out4x01234567 = vcombine_f16(vcvt_f16_f32(vout4x0123), vcvt_f16_f32(vout4x4567));
    float16x8_t vfp16out4x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout4x89AB), vcvt_f16_f32(vout4xCDEF));
    float16x8_t vfp16out5x01234567 = vcombine_f16(vcvt_f16_f32(vout5x0123), vcvt_f16_f32(vout5x4567));
    float16x8_t vfp16out5x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout5x89AB), vcvt_f16_f32(vout5xCDEF));
    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    vfp16out1x01234567 = vmaxq_f16(vfp16out1x01234567, voutput_min);
    vfp16out1x89ABCDEF = vmaxq_f16(vfp16out1x89ABCDEF, voutput_min);
    vfp16out2x01234567 = vmaxq_f16(vfp16out2x01234567, voutput_min);
    vfp16out2x89ABCDEF = vmaxq_f16(vfp16out2x89ABCDEF, voutput_min);
    vfp16out3x01234567 = vmaxq_f16(vfp16out3x01234567, voutput_min);
    vfp16out3x89ABCDEF = vmaxq_f16(vfp16out3x89ABCDEF, voutput_min);
    vfp16out4x01234567 = vmaxq_f16(vfp16out4x01234567, voutput_min);
    vfp16out4x89ABCDEF = vmaxq_f16(vfp16out4x89ABCDEF, voutput_min);
    vfp16out5x01234567 = vmaxq_f16(vfp16out5x01234567, voutput_min);
    vfp16out5x89ABCDEF = vmaxq_f16(vfp16out5x89ABCDEF, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);
    vfp16out1x01234567 = vminq_f16(vfp16out1x01234567, voutput_max);
    vfp16out1x89ABCDEF = vminq_f16(vfp16out1x89ABCDEF, voutput_max);
    vfp16out2x01234567 = vminq_f16(vfp16out2x01234567, voutput_max);
    vfp16out2x89ABCDEF = vminq_f16(vfp16out2x89ABCDEF, voutput_max);
    vfp16out3x01234567 = vminq_f16(vfp16out3x01234567, voutput_max);
    vfp16out3x89ABCDEF = vminq_f16(vfp16out3x89ABCDEF, voutput_max);
    vfp16out4x01234567 = vminq_f16(vfp16out4x01234567, voutput_max);
    vfp16out4x89ABCDEF = vminq_f16(vfp16out4x89ABCDEF, voutput_max);
    vfp16out5x01234567 = vminq_f16(vfp16out5x01234567, voutput_max);
    vfp16out5x89ABCDEF = vminq_f16(vfp16out5x89ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));
      vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vfp16out1x89ABCDEF));
      vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vfp16out2x89ABCDEF));
      vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vfp16out3x89ABCDEF));
      vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567));
      vst1q_u16(c4 + 8, vreinterpretq_u16_f16(vfp16out4x89ABCDEF));
      vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567));
      vst1q_u16(c5 + 8, vreinterpretq_u16_f16(vfp16out5x89ABCDEF));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567)); c1 += 8;
       vfp16out1x01234567 = vfp16out1x89ABCDEF;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567)); c2 += 8;
       vfp16out2x01234567 = vfp16out2x89ABCDEF;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567)); c3 += 8;
       vfp16out3x01234567 = vfp16out3x89ABCDEF;
       vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567)); c4 += 8;
       vfp16out4x01234567 = vfp16out4x89ABCDEF;
       vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567)); c5 += 8;
       vfp16out5x01234567 = vfp16out5x89ABCDEF;
     }
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     float16x4_t vfp16out1x0123 = vget_low_f16(vfp16out1x01234567);
     float16x4_t vfp16out2x0123 = vget_low_f16(vfp16out2x01234567);
     float16x4_t vfp16out3x0123 = vget_low_f16(vfp16out3x01234567);
     float16x4_t vfp16out4x0123 = vget_low_f16(vfp16out4x01234567);
     float16x4_t vfp16out5x0123 = vget_low_f16(vfp16out5x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vst1_u16(c1, vreinterpret_u16_f16(vfp16out1x0123)); c1 += 4;
       vst1_u16(c2, vreinterpret_u16_f16(vfp16out2x0123)); c2 += 4;
       vst1_u16(c3, vreinterpret_u16_f16(vfp16out3x0123)); c3 += 4;
       vst1_u16(c4, vreinterpret_u16_f16(vfp16out4x0123)); c4 += 4;
       vst1_u16(c5, vreinterpret_u16_f16(vfp16out5x0123)); c5 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
       vfp16out1x0123 = vget_high_f16(vfp16out1x01234567);
       vfp16out2x0123 = vget_high_f16(vfp16out2x01234567);
       vfp16out3x0123 = vget_high_f16(vfp16out3x01234567);
       vfp16out4x0123 = vget_high_f16(vfp16out4x01234567);
       vfp16out5x0123 = vget_high_f16(vfp16out5x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vfp16out1x0123), 0); c1 += 2;
       vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vfp16out2x0123), 0); c2 += 2;
       vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vfp16out3x0123), 0); c3 += 2;
       vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vfp16out4x0123), 0); c4 += 2;
       vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vfp16out5x0123), 0); c5 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
       vfp16out1x0123 = vext_f16(vfp16out1x0123, vfp16out1x0123, 2);
       vfp16out2x0123 = vext_f16(vfp16out2x0123, vfp16out2x0123, 2);
       vfp16out3x0123 = vext_f16(vfp16out3x0123, vfp16out3x0123, 2);
       vfp16out4x0123 = vext_f16(vfp16out4x0123, vfp16out4x0123, 2);
       vfp16out5x0123 = vext_f16(vfp16out5x0123, vfp16out5x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
       vst1_lane_u16(c1, vreinterpret_u16_f16(vfp16out1x0123), 0);
       vst1_lane_u16(c2, vreinterpret_u16_f16(vfp16out2x0123), 0);
       vst1_lane_u16(c3, vreinterpret_u16_f16(vfp16out3x0123), 0);
       vst1_lane_u16(c4, vreinterpret_u16_f16(vfp16out4x0123), 0);
       vst1_lane_u16(c5, vreinterpret_u16_f16(vfp16out5x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    const int32x4_t vizp0 = vld1q_dup_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vizp0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vizp0);

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va0x1 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

      k -= 16 * sizeof(int8_t);
    }
    if (k != 0) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);

    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));

    #if XNN_ARCH_ARM64
      const uint16x8x2_t voutput_minmax = vld2q_dup_u16(&params->fp16arith.min);
      const float16x8_t voutput_min = vreinterpretq_f16_u16(voutput_minmax.val[0]);
      const float16x8_t voutput_max = vreinterpretq_f16_u16(voutput_minmax.val[1]);
    #else
      const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
      const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    #endif
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    const int32x4_t vizp01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vizp01), 0);
    int32x4_t vacc1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vizp01), 0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vizp01), 0);
    int32x4_t vacc1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vizp01), 0);

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
      int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
      int8x8_t va1x1 = vld1_s8(a1); a1 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
      const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
      vprod1x0123c0 = vmlal_s8(vprod1x0123c0, vb0123c0x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
      const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
      vprod1x4567c0 = vmlal_s8(vprod1x4567c0, vb4567c0x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
      const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
      vprod1x0123c1 = vmlal_s8(vprod1x0123c1, vb0123c1x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
      const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
      vprod1x4567c1 = vmlal_s8(vprod1x4567c1, vb4567c1x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
      const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
      vprod1x0123c2 = vmlal_s8(vprod1x0123c2, vb0123c2x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
      const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
      vprod1x4567c2 = vmlal_s8(vprod1x4567c2, vb4567c2x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
      const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
      vprod1x0123c3 = vmlal_s8(vprod1x0123c3, vb0123c3x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
      const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
      vprod1x4567c3 = vmlal_s8(vprod1x4567c3, vb4567c3x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

      k -= 16 * sizeof(int8_t);
    }
    if (k != 0) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va1x0 = vld1_s8(a1); a1 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

    }

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out1x01234567 = vcombine_f16(vcvt_f16_f32(vout1x0123), vcvt_f16_f32(vout1x4567));

    #if XNN_ARCH_ARM64
      const uint16x8x2_t voutput_minmax = vld2q_dup_u16(&params->fp16arith.min);
      const float16x8_t voutput_min = vreinterpretq_f16_u16(voutput_minmax.val[0]);
      const float16x8_t voutput_max = vreinterpretq_f16_u16(voutput_minmax.val[1]);
    #else
      const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
      const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    #endif
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out1x01234567 = vmaxq_f16(vfp16out1x01234567, voutput_min);
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out1x01234567 = vminq_f16(vfp16out1x01234567, voutput_max);
    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567));

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);

      nc -= 8;
    } else {
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     float16x4_t vfp16out1x0123 = vget_low_f16(vfp16out1x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vst1_u16(c1, vreinterpret_u16_f16(vfp16out1x0123)); c1 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
       vfp16out1x0123 = vget_high_f16(vfp16out1x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vfp16out1x0123), 0); c1 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
       vfp16out1x0123 = vext_f16(vfp16out1x0123, vfp16out1x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
       vst1_lane_u16(c1, vreinterpret_u16_f16(vfp16out1x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint16_t* c0 = (uint16_t*) c;

  const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
  const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;
      while (k >= 16 * sizeof(int8_t)) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va0x1 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

        k -= 16 * sizeof(int8_t);
      }
      if (k != 0) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);

    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    #if XNN_ARCH_ARM64
      const uint16x8x2_t voutput_minmax = vld2q_dup_u16(&params->fp16arith.min);
      const float16x8_t voutput_min = vreinterpretq_f16_u16(voutput_minmax.val[0]);
      const float16x8_t voutput_max = vreinterpretq_f16_u16(voutput_minmax.val[1]);
    #else
      // vld2_dup is to work around aarch32 clang bug with vld1q_dup
      const uint16x4x2_t vminmax = vld2_dup_u16(&params->fp16arith.min);
      const float16x8_t voutput_min = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
      const float16x8_t voutput_max = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
    #endif
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     if (nc & 4) {
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint16_t* c0 = (uint16_t*) c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
  const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point);
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      a += 2;

      size_t k = kc;
      while (k >= 16 * sizeof(int8_t)) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
        int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
        int8x8_t va1x1 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
        const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
        vprod1x0123c0 = vmlal_s8(vprod1x0123c0, vb0123c0x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
        const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
        vprod1x4567c0 = vmlal_s8(vprod1x4567c0, vb4567c0x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
        const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
        vprod1x0123c1 = vmlal_s8(vprod1x0123c1, vb0123c1x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
        const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
        vprod1x4567c1 = vmlal_s8(vprod1x4567c1, vb4567c1x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
        const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
        vprod1x0123c2 = vmlal_s8(vprod1x0123c2, vb0123c2x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
        const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
        vprod1x4567c2 = vmlal_s8(vprod1x4567c2, vb4567c2x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
        const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
        vprod1x0123c3 = vmlal_s8(vprod1x0123c3, vb0123c3x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
        const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
        vprod1x4567c3 = vmlal_s8(vprod1x4567c3, vb4567c3x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

        k -= 16 * sizeof(int8_t);
      }
      if (k != 0) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va1x0 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

      }

      p -= 2 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);

    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout1x0123 = vmulq_f32(vout1x0123, vinput_scale);
    vout1x4567 = vmulq_f32(vout1x4567, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
    #endif

    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out1x01234567 = vcombine_f16(vcvt_f16_f32(vout1x0123), vcvt_f16_f32(vout1x4567));
    #if XNN_ARCH_ARM64
      const uint16x8x2_t voutput_minmax = vld2q_dup_u16(&params->fp16arith.min);
      const float16x8_t voutput_min = vreinterpretq_f16_u16(voutput_minmax.val[0]);
      const float16x8_t voutput_max = vreinterpretq_f16_u16(voutput_minmax.val[1]);
    #else
      // vld2_dup is to work around aarch32 clang bug with vld1q_dup
      const uint16x4x2_t vminmax = vld2_dup_u16(&params->fp16arith.min);
      const float16x8_t voutput_min = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
      const float16x8_t voutput_max = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
    #endif
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out1x01234567 = vmaxq_f16(vfp16out1x01234567, voutput_min);
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out1x01234567 = vminq_f16(vfp16out1x01234567, voutput_max);
    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567));
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);

      nc -= 8;
    } else {
     float16x4_t vfp16out1x0123 = vget_low_f16(vfp16out1x01234567);
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     if (nc & 4) {
       vst1_u16(c1, vreinterpret_u16_f16(vfp16out1x0123)); c1 += 4;
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vfp16out1x0123 = vget_high_f16(vfp16out1x01234567);
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vfp16out1x0123), 0); c1 += 2;
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vfp16out1x0123 = vext_f16(vfp16out1x0123, vfp16out1x0123, 2);
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c1, vreinterpret_u16_f16(vfp16out1x0123), 0);
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32(
    size_t batch,
    const int8_t* input,
    void* output,
    const union xnn_qs8_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  const int16x8_t vminus_zero_point = vdupq_n_s16(-params->neon.zero_point);
#ifdef XNN_COMPILER_MSVC
  const float16x8_t vscale = vreinterpretq_f16_u16(vdupq_n_u16(params->neon.scale));
#else
  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neon.scale));
#endif
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const int8x8_t vx01234567 = vld1_s8(input); input += 8;
    const int8x8_t vx89ABCDEF = vld1_s8(input); input += 8;
    const int8x8_t vxGHIJKLMN = vld1_s8(input); input += 8;
    const int8x8_t vxOPQRSTUV = vld1_s8(input); input += 8;

    const int16x8_t vhx01234567 = vaddw_s8(vminus_zero_point, vx01234567);
    const int16x8_t vhx89ABCDEF = vaddw_s8(vminus_zero_point, vx89ABCDEF);
    const int16x8_t vhxGHIJKLMN = vaddw_s8(vminus_zero_point, vxGHIJKLMN);
    const int16x8_t vhxOPQRSTUV = vaddw_s8(vminus_zero_point, vxOPQRSTUV);

    float16x8_t vy01234567 = vcvtq_f16_s16(vhx01234567);
    float16x8_t vy89ABCDEF = vcvtq_f16_s16(vhx89ABCDEF);
    float16x8_t vyGHIJKLMN = vcvtq_f16_s16(vhxGHIJKLMN);
    float16x8_t vyOPQRSTUV = vcvtq_f16_s16(vhxOPQRSTUV);

    vy01234567 = vmulq_f16(vy01234567, vscale);
    vy89ABCDEF = vmulq_f16(vy89ABCDEF, vscale);
    vyGHIJKLMN = vmulq_f16(vyGHIJKLMN, vscale);
    vyOPQRSTUV = vmulq_f16(vyOPQRSTUV, vscale);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy89ABCDEF)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vyGHIJKLMN)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vyOPQRSTUV)); o += 8;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int8x8_t vx = vld1_s8(input); input += 8;

    const int16x8_t vhx = vaddw_s8(vminus_zero_point, vx);

    float16x8_t vy = vcvtq_f16_s16(vhx);

    vy = vmulq_f16(vy, vscale);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const int8x8_t vx = vld1_s8(input);

    const int16x8_t vhx = vaddw_s8(vminus_zero_point, vx);

    float16x8_t vy = vcvtq_f16_s16(vhx);
    vy = vmulq_f16(vy, vscale);

    float16x4_t vy_lo = vget_low_f16(vy);
    if (batch & (4 * sizeof(int8_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;
      vy_lo = vget_high_f16(vy);
    }
    if (batch & (2 * sizeof(int8_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o += 2;

      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(int8_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}
