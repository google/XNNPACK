// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/argmaxpool.h>
#include <xnnpack/avgpool.h>
#include <xnnpack/common.h>
#include <xnnpack/conv.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/fill.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/gemm.h>
#include <xnnpack/ibilinear.h>
#include <xnnpack/igemm.h>
#include <xnnpack/math.h>
#include <xnnpack/maxpool.h>
#include <xnnpack/packx.h>
#include <xnnpack/pad.h>
#include <xnnpack/pavgpool.h>
#include <xnnpack/prelu.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/rmax.h>
#include <xnnpack/spmm.h>
#include <xnnpack/transpose.h>
#include <xnnpack/unaligned.h>
#include <xnnpack/unpool.h>
#include <xnnpack/vadd.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vcvt.h>
#include <xnnpack/vlrelu.h>
#include <xnnpack/vmul.h>
#include <xnnpack/vmulcaddc.h>
#include <xnnpack/vunary.h>
#include <xnnpack/zip.h>


void xnn_f16_f32_vcvt_ukernel__scalar_x4(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t vsign_mask = params->scalar.sign_mask;
  const uint32_t vexp_offset = params->scalar.exp_offset;
  const float vexp_scale = params->scalar.exp_scale;
  const uint32_t vmagic_mask = params->scalar.magic_mask;
  const float vmagic_bias = params->scalar.magic_bias;
  const uint32_t vdenorm_cutoff = params->scalar.denorm_cutoff;

  const uint16_t* i = (const uint16_t*) input;
  uint32_t* o = (uint32_t*) output;
  for (; batch >= 4 * sizeof(uint16_t); batch -= 4 * sizeof(uint16_t)) {
    const uint16_t vh0 = i[0];
    const uint16_t vh1 = i[1];
    const uint16_t vh2 = i[2];
    const uint16_t vh3 = i[3];
    i += 4;

    const uint32_t vw0 = (uint32_t) vh0 << 16;
    const uint32_t vw1 = (uint32_t) vh1 << 16;
    const uint32_t vw2 = (uint32_t) vh2 << 16;
    const uint32_t vw3 = (uint32_t) vh3 << 16;

    const uint32_t vsign0 = vw0 & vsign_mask;
    const uint32_t vsign1 = vw1 & vsign_mask;
    const uint32_t vsign2 = vw2 & vsign_mask;
    const uint32_t vsign3 = vw3 & vsign_mask;

    const uint32_t v2w0 = vw0 + vw0;
    const uint32_t v2w1 = vw1 + vw1;
    const uint32_t v2w2 = vw2 + vw2;
    const uint32_t v2w3 = vw3 + vw3;

    const uint32_t vnorm0 = float_as_uint32(uint32_as_float((v2w0 >> 4) + vexp_offset) * vexp_scale);
    const uint32_t vnorm1 = float_as_uint32(uint32_as_float((v2w1 >> 4) + vexp_offset) * vexp_scale);
    const uint32_t vnorm2 = float_as_uint32(uint32_as_float((v2w2 >> 4) + vexp_offset) * vexp_scale);
    const uint32_t vnorm3 = float_as_uint32(uint32_as_float((v2w3 >> 4) + vexp_offset) * vexp_scale);

    const uint32_t vdenorm0 = float_as_uint32(uint32_as_float((v2w0 >> 17) | vmagic_mask) - vmagic_bias);
    const uint32_t vdenorm1 = float_as_uint32(uint32_as_float((v2w1 >> 17) | vmagic_mask) - vmagic_bias);
    const uint32_t vdenorm2 = float_as_uint32(uint32_as_float((v2w2 >> 17) | vmagic_mask) - vmagic_bias);
    const uint32_t vdenorm3 = float_as_uint32(uint32_as_float((v2w3 >> 17) | vmagic_mask) - vmagic_bias);

    const uint32_t vf0 = vsign0 | (XNN_UNPREDICTABLE(v2w0 < vdenorm_cutoff) ? vdenorm0 : vnorm0);
    const uint32_t vf1 = vsign1 | (XNN_UNPREDICTABLE(v2w1 < vdenorm_cutoff) ? vdenorm1 : vnorm1);
    const uint32_t vf2 = vsign2 | (XNN_UNPREDICTABLE(v2w2 < vdenorm_cutoff) ? vdenorm2 : vnorm2);
    const uint32_t vf3 = vsign3 | (XNN_UNPREDICTABLE(v2w3 < vdenorm_cutoff) ? vdenorm3 : vnorm3);

    o[0] = vf0;
    o[1] = vf1;
    o[2] = vf2;
    o[3] = vf3;
    o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const uint16_t vh = *i++;

      const uint32_t vw = (uint32_t) vh << 16;
      const uint32_t vsign = vw & vsign_mask;
      const uint32_t v2w = vw + vw;
      const uint32_t vnorm = float_as_uint32(uint32_as_float((v2w >> 4) + vexp_offset) * vexp_scale);
      const uint32_t vdenorm = float_as_uint32(uint32_as_float((v2w >> 17) | vmagic_mask) - vmagic_bias);
      const uint32_t vf = vsign | (XNN_UNPREDICTABLE(v2w < vdenorm_cutoff) ? vdenorm : vnorm);

      *o++ = vf;

      batch -= sizeof(uint16_t);
    } while (batch != 0);
  }
}

void xnn_f32_argmaxpool_ukernel_4x__scalar_c1(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 4);
  assert(channels != 0);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    i0 = (const float*) ((uintptr_t) i0 + input_offset);
    i1 = (const float*) ((uintptr_t) i1 + input_offset);
    i2 = (const float*) ((uintptr_t) i2 + input_offset);
    i3 = (const float*) ((uintptr_t) i3 + input_offset);
    if (pooling_elements < 2) {
      i1 = i0;
    }
    if (pooling_elements <= 2) {
      i2 = i0;
    }
    if (pooling_elements != 4) {
      i3 = i0;
    }

    size_t c = channels;
    do {
      const float vi0 = *i0++;
      const float vi1 = *i1++;
      const float vi2 = *i2++;
      const float vi3 = *i3++;

      float vmax = vi0;
      uint32_t vidx = 0;

      if (vi1 > vmax) {
        vmax = vi1;
        vidx = 1;
      }

      if (vi2 > vmax) {
        vmax = vi2;
        vidx = 2;
      }

      if (vi3 > vmax) {
        vmax = vi3;
        vidx = 3;
      }

      *output++ = vmax;
      *index++ = vidx;
    } while (--c != 0);
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* accumulation_buffer,
    uint32_t* index_buffer,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements > 9);
  assert(channels != 0);

  do {
    {
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);

      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *i8++;

        float vmax = vi0;
        uint32_t vidx = 0;

        if (vi1 > vmax) {
          vmax = vi1;
          vidx = 1;
        }

        if (vi2 > vmax) {
          vmax = vi2;
          vidx = 2;
        }

        if (vi3 > vmax) {
          vmax = vi3;
          vidx = 3;
        }

        if (vi4 > vmax) {
          vmax = vi4;
          vidx = 4;
        }

        if (vi5 > vmax) {
          vmax = vi5;
          vidx = 5;
        }

        if (vi6 > vmax) {
          vmax = vi6;
          vidx = 6;
        }

        if (vi7 > vmax) {
          vmax = vi7;
          vidx = 7;
        }

        if (vi8 > vmax) {
          vmax = vi8;
          vidx = 8;
        }

        *ab++ = vmax;
        *ib++ = vidx;
      } while (--c != 0);
    }
    uint32_t vidx0 = 9;
    size_t k = pooling_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);

      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;

        float vmax = *ab;
        uint32_t vidx = *ib;

        if (vi0 > vmax) {
          vmax = vi0;
          vidx = vidx0;
        }

        if (vi1 > vmax) {
          vmax = vi1;
          vidx = vidx0 + 1;
        }

        if (vi2 > vmax) {
          vmax = vi2;
          vidx = vidx0 + 2;
        }

        if (vi3 > vmax) {
          vmax = vi3;
          vidx = vidx0 + 3;
        }

        if (vi4 > vmax) {
          vmax = vi4;
          vidx = vidx0 + 4;
        }

        if (vi5 > vmax) {
          vmax = vi5;
          vidx = vidx0 + 5;
        }

        if (vi6 > vmax) {
          vmax = vi6;
          vidx = vidx0 + 6;
        }

        if (vi7 > vmax) {
          vmax = vi7;
          vidx = vidx0 + 7;
        }

        *ab++ = vmax;
        *ib++ = vidx;
      } while (--c != 0);
      vidx0 += 8;
    }

    float* o = output;
    uint32_t* i = index;
    {
      const float* i0 = input[0];
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      input = (const float**) ((uintptr_t) input + input_increment);
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
      if (k != 8) {
        i7 = i0;
      }

      size_t c = channels;
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;

        float vmax = *ab++;
        uint32_t vidx = *ib++;

        if (vi0 > vmax) {
          vmax = vi0;
          vidx = vidx0;
        }

        if (vi1 > vmax) {
          vmax = vi1;
          vidx = vidx0 + 1;
        }

        if (vi2 > vmax) {
          vmax = vi2;
          vidx = vidx0 + 2;
        }

        if (vi3 > vmax) {
          vmax = vi3;
          vidx = vidx0 + 3;
        }

        if (vi4 > vmax) {
          vmax = vi4;
          vidx = vidx0 + 4;
        }

        if (vi5 > vmax) {
          vmax = vi5;
          vidx = vidx0 + 5;
        }

        if (vi6 > vmax) {
          vmax = vi6;
          vidx = vidx0 + 6;
        }

        if (vi7 > vmax) {
          vmax = vi7;
          vidx = vidx0 + 7;
        }

        *o++ = vmax;
        *i++ = vidx;
      } while (--c != 0);
    }

    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--output_pixels != 0);
}

void xnn_f32_argmaxpool_ukernel_9x__scalar_c1(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 9);
  assert(channels != 0);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    i0 = (const float*) ((uintptr_t) i0 + input_offset);
    i1 = (const float*) ((uintptr_t) i1 + input_offset);
    i2 = (const float*) ((uintptr_t) i2 + input_offset);
    i3 = (const float*) ((uintptr_t) i3 + input_offset);
    i4 = (const float*) ((uintptr_t) i4 + input_offset);
    i5 = (const float*) ((uintptr_t) i5 + input_offset);
    i6 = (const float*) ((uintptr_t) i6 + input_offset);
    i7 = (const float*) ((uintptr_t) i7 + input_offset);
    i8 = (const float*) ((uintptr_t) i8 + input_offset);
    if (pooling_elements < 2) {
      i1 = i0;
    }
    if (pooling_elements <= 2) {
      i2 = i0;
    }
    if (pooling_elements < 4) {
      i3 = i0;
    }
    if (pooling_elements <= 4) {
      i4 = i0;
    }
    if (pooling_elements < 6) {
      i5 = i0;
    }
    if (pooling_elements <= 6) {
      i6 = i0;
    }
    if (pooling_elements < 8) {
      i7 = i0;
    }
    if (pooling_elements <= 8) {
      i8 = i0;
    }

    size_t c = channels;
    do {
      const float vi0 = *i0++;
      const float vi1 = *i1++;
      const float vi2 = *i2++;
      const float vi3 = *i3++;
      const float vi4 = *i4++;
      const float vi5 = *i5++;
      const float vi6 = *i6++;
      const float vi7 = *i7++;
      const float vi8 = *i8++;

      float vmax = vi0;
      uint32_t vidx = 0;

      if (vi1 > vmax) {
        vmax = vi1;
        vidx = 1;
      }

      if (vi2 > vmax) {
        vmax = vi2;
        vidx = 2;
      }

      if (vi3 > vmax) {
        vmax = vi3;
        vidx = 3;
      }

      if (vi4 > vmax) {
        vmax = vi4;
        vidx = 4;
      }

      if (vi5 > vmax) {
        vmax = vi5;
        vidx = 5;
      }

      if (vi6 > vmax) {
        vmax = vi6;
        vidx = 6;
      }

      if (vi7 > vmax) {
        vmax = vi7;
        vidx = 7;
      }

      if (vi8 > vmax) {
        vmax = vi8;
        vidx = 8;
      }

      *output++ = vmax;
      *index++ = vidx;
    } while (--c != 0);
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_avgpool_minmax_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  do {
    {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      float* b = buffer;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *i8++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum018 = vsum01 + vi8;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum01678 = vsum018 + vsum67;
        const float vsum = vsum2345 + vsum01678;

        *b++ = vsum;
      } while (--c != 0);
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      float* b = buffer;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        *b++ = vsum;
      } while (--c != 0);
    }

    {
      const float* i0 = input[0];
      assert(i0 != NULL);
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      float* b = buffer;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        float vout = vsum * vscale;
        vout = math_max_f32(vout, vmin);
        vout = math_min_f32(vout, vmax);

        *output++ = vout;
      } while (--c != 0);
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_avgpool_minmax_ukernel_9x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    input = (const float**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    do {
      const float vi0 = *i0++;
      const float vi1 = *i1++;
      const float vi2 = *i2++;
      const float vi3 = *i3++;
      const float vi4 = *i4++;
      const float vi5 = *i5++;
      const float vi6 = *i6++;
      const float vi7 = *i7++;
      const float vi8 = *i8++;

      const float vsum01 = vi0 + vi1;
      const float vsum23 = vi2 + vi3;
      const float vsum45 = vi4 + vi5;
      const float vsum67 = vi6 + vi7;
      const float vsum018 = vsum01 + vi8;
      const float vsum2345 = vsum23 + vsum45;
      const float vsum01678 = vsum018 + vsum67;
      const float vsum = vsum2345 + vsum01678;

      float vout = vsum * vscale;
      vout = math_max_f32(vout, vmin);
      vout = math_min_f32(vout, vmax);

      *output++ = vout;
    } while (--c != 0);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__scalar_1x1(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_width_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_decrement = round_down_po2(input_width - 1, 2) * 3 /* channels */ * sizeof(float);
  const size_t output_width = input_width / 2;
  const size_t output_channel_decrement = output_width * output_width_stride - 4 * sizeof(float);
  const size_t output_height_increment = output_height_stride - round_up_po2(output_channels, 4) * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  float* o0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float voutput_max = params->scalar.max;
  const float voutput_min = params->scalar.min;

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 1) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }

    const float* w = weights;
    size_t c = output_channels;
    do {
      float vi00c0 = i0[0];
      float vi00c1 = i0[1];
      float vi00c2 = i0[2];
      float vi10c0 = i1[0];
      float vi10c1 = i1[1];
      float vi10c2 = i1[2];
      float vi20c0 = i2[0];
      float vi20c1 = i2[1];
      float vi20c2 = i2[2];

      size_t iw = input_width - 1;
      for (; iw >= 2; iw -= 2) {
        // start with biases
        float voc0 = w[0];
        float voc1 = w[1];
        float voc2 = w[2];
        float voc3 = w[3];

        const float vk00c0x0 = w[4];
        const float vk00c0x1 = w[5];
        const float vk00c0x2 = w[6];
        const float vk00c0x3 = w[7];

        voc0 += vk00c0x0 * vi00c0;
        voc1 += vk00c0x1 * vi00c0;
        voc2 += vk00c0x2 * vi00c0;
        voc3 += vk00c0x3 * vi00c0;

        const float vk10c0x0 = w[8];
        const float vk10c0x1 = w[9];
        const float vk10c0x2 = w[10];
        const float vk10c0x3 = w[11];

        voc0 += vk10c0x0 * vi10c0;
        voc1 += vk10c0x1 * vi10c0;
        voc2 += vk10c0x2 * vi10c0;
        voc3 += vk10c0x3 * vi10c0;

        const float vk20c0x0 = w[12];
        const float vk20c0x1 = w[13];
        const float vk20c0x2 = w[14];
        const float vk20c0x3 = w[15];

        voc0 += vk20c0x0 * vi20c0;
        voc1 += vk20c0x1 * vi20c0;
        voc2 += vk20c0x2 * vi20c0;
        voc3 += vk20c0x3 * vi20c0;

        const float vk00c1x0 = w[16];
        const float vk00c1x1 = w[17];
        const float vk00c1x2 = w[18];
        const float vk00c1x3 = w[19];

        voc0 += vk00c1x0 * vi00c1;
        voc1 += vk00c1x1 * vi00c1;
        voc2 += vk00c1x2 * vi00c1;
        voc3 += vk00c1x3 * vi00c1;

        const float vk10c1x0 = w[20];
        const float vk10c1x1 = w[21];
        const float vk10c1x2 = w[22];
        const float vk10c1x3 = w[23];

        voc0 += vk10c1x0 * vi10c1;
        voc1 += vk10c1x1 * vi10c1;
        voc2 += vk10c1x2 * vi10c1;
        voc3 += vk10c1x3 * vi10c1;

        const float vk20c1x0 = w[24];
        const float vk20c1x1 = w[25];
        const float vk20c1x2 = w[26];
        const float vk20c1x3 = w[27];

        voc0 += vk20c1x0 * vi20c1;
        voc1 += vk20c1x1 * vi20c1;
        voc2 += vk20c1x2 * vi20c1;
        voc3 += vk20c1x3 * vi20c1;

        const float vk00c2x0 = w[28];
        const float vk00c2x1 = w[29];
        const float vk00c2x2 = w[30];
        const float vk00c2x3 = w[31];

        voc0 += vk00c2x0 * vi00c2;
        voc1 += vk00c2x1 * vi00c2;
        voc2 += vk00c2x2 * vi00c2;
        voc3 += vk00c2x3 * vi00c2;

        const float vk10c2x0 = w[32];
        const float vk10c2x1 = w[33];
        const float vk10c2x2 = w[34];
        const float vk10c2x3 = w[35];

        voc0 += vk10c2x0 * vi10c2;
        voc1 += vk10c2x1 * vi10c2;
        voc2 += vk10c2x2 * vi10c2;
        voc3 += vk10c2x3 * vi10c2;

        const float vk20c2x0 = w[36];
        const float vk20c2x1 = w[37];
        const float vk20c2x2 = w[38];
        const float vk20c2x3 = w[39];

        voc0 += vk20c2x0 * vi20c2;
        voc1 += vk20c2x1 * vi20c2;
        voc2 += vk20c2x2 * vi20c2;
        voc3 += vk20c2x3 * vi20c2;

        const float vk01c0x0 = w[40];
        const float vk01c0x1 = w[41];
        const float vk01c0x2 = w[42];
        const float vk01c0x3 = w[43];

        const float vi01c0 = i0[3];

        voc0 += vk01c0x0 * vi01c0;
        voc1 += vk01c0x1 * vi01c0;
        voc2 += vk01c0x2 * vi01c0;
        voc3 += vk01c0x3 * vi01c0;

        const float vk11c0x0 = w[44];
        const float vk11c0x1 = w[45];
        const float vk11c0x2 = w[46];
        const float vk11c0x3 = w[47];

        const float vi11c0 = i1[3];

        voc0 += vk11c0x0 * vi11c0;
        voc1 += vk11c0x1 * vi11c0;
        voc2 += vk11c0x2 * vi11c0;
        voc3 += vk11c0x3 * vi11c0;

        const float vk21c0x0 = w[48];
        const float vk21c0x1 = w[49];
        const float vk21c0x2 = w[50];
        const float vk21c0x3 = w[51];

        const float vi21c0 = i2[3];

        voc0 += vk21c0x0 * vi21c0;
        voc1 += vk21c0x1 * vi21c0;
        voc2 += vk21c0x2 * vi21c0;
        voc3 += vk21c0x3 * vi21c0;

        const float vk01c1x0 = w[52];
        const float vk01c1x1 = w[53];
        const float vk01c1x2 = w[54];
        const float vk01c1x3 = w[55];

        const float vi01c1 = i0[4];

        voc0 += vk01c1x0 * vi01c1;
        voc1 += vk01c1x1 * vi01c1;
        voc2 += vk01c1x2 * vi01c1;
        voc3 += vk01c1x3 * vi01c1;

        const float vk11c1x0 = w[56];
        const float vk11c1x1 = w[57];
        const float vk11c1x2 = w[58];
        const float vk11c1x3 = w[59];

        const float vi11c1 = i1[4];

        voc0 += vk11c1x0 * vi11c1;
        voc1 += vk11c1x1 * vi11c1;
        voc2 += vk11c1x2 * vi11c1;
        voc3 += vk11c1x3 * vi11c1;

        const float vk21c1x0 = w[60];
        const float vk21c1x1 = w[61];
        const float vk21c1x2 = w[62];
        const float vk21c1x3 = w[63];

        const float vi21c1 = i2[4];

        voc0 += vk21c1x0 * vi21c1;
        voc1 += vk21c1x1 * vi21c1;
        voc2 += vk21c1x2 * vi21c1;
        voc3 += vk21c1x3 * vi21c1;

        const float vk01c2x0 = w[64];
        const float vk01c2x1 = w[65];
        const float vk01c2x2 = w[66];
        const float vk01c2x3 = w[67];

        const float vi01c2 = i0[5];

        voc0 += vk01c2x0 * vi01c2;
        voc1 += vk01c2x1 * vi01c2;
        voc2 += vk01c2x2 * vi01c2;
        voc3 += vk01c2x3 * vi01c2;

        const float vk11c2x0 = w[68];
        const float vk11c2x1 = w[69];
        const float vk11c2x2 = w[70];
        const float vk11c2x3 = w[71];

        const float vi11c2 = i1[5];

        voc0 += vk11c2x0 * vi11c2;
        voc1 += vk11c2x1 * vi11c2;
        voc2 += vk11c2x2 * vi11c2;
        voc3 += vk11c2x3 * vi11c2;

        const float vk21c2x0 = w[72];
        const float vk21c2x1 = w[73];
        const float vk21c2x2 = w[74];
        const float vk21c2x3 = w[75];

        const float vi21c2 = i2[5];

        voc0 += vk21c2x0 * vi21c2;
        voc1 += vk21c2x1 * vi21c2;
        voc2 += vk21c2x2 * vi21c2;
        voc3 += vk21c2x3 * vi21c2;

        const float vk02c0x0 = w[76];
        const float vk02c0x1 = w[77];
        const float vk02c0x2 = w[78];
        const float vk02c0x3 = w[79];

        const float vi02c0 = i0[6];

        voc0 += vk02c0x0 * vi02c0;
        voc1 += vk02c0x1 * vi02c0;
        voc2 += vk02c0x2 * vi02c0;
        voc3 += vk02c0x3 * vi02c0;

        const float vk12c0x0 = w[80];
        const float vk12c0x1 = w[81];
        const float vk12c0x2 = w[82];
        const float vk12c0x3 = w[83];

        const float vi12c0 = i1[6];

        voc0 += vk12c0x0 * vi12c0;
        voc1 += vk12c0x1 * vi12c0;
        voc2 += vk12c0x2 * vi12c0;
        voc3 += vk12c0x3 * vi12c0;

        const float vk22c0x0 = w[84];
        const float vk22c0x1 = w[85];
        const float vk22c0x2 = w[86];
        const float vk22c0x3 = w[87];

        const float vi22c0 = i2[6];

        voc0 += vk22c0x0 * vi22c0;
        voc1 += vk22c0x1 * vi22c0;
        voc2 += vk22c0x2 * vi22c0;
        voc3 += vk22c0x3 * vi22c0;

        vi00c0 = vi02c0;
        vi10c0 = vi12c0;
        vi20c0 = vi22c0;

        const float vk02c1x0 = w[88];
        const float vk02c1x1 = w[89];
        const float vk02c1x2 = w[90];
        const float vk02c1x3 = w[91];

        const float vi02c1 = i0[7];

        voc0 += vk02c1x0 * vi02c1;
        voc1 += vk02c1x1 * vi02c1;
        voc2 += vk02c1x2 * vi02c1;
        voc3 += vk02c1x3 * vi02c1;

        const float vk12c1x0 = w[92];
        const float vk12c1x1 = w[93];
        const float vk12c1x2 = w[94];
        const float vk12c1x3 = w[95];

        const float vi12c1 = i1[7];

        voc0 += vk12c1x0 * vi12c1;
        voc1 += vk12c1x1 * vi12c1;
        voc2 += vk12c1x2 * vi12c1;
        voc3 += vk12c1x3 * vi12c1;

        const float vk22c1x0 = w[96];
        const float vk22c1x1 = w[97];
        const float vk22c1x2 = w[98];
        const float vk22c1x3 = w[99];

        const float vi22c1 = i2[7];

        voc0 += vk22c1x0 * vi22c1;
        voc1 += vk22c1x1 * vi22c1;
        voc2 += vk22c1x2 * vi22c1;
        voc3 += vk22c1x3 * vi22c1;

        vi00c1 = vi02c1;
        vi10c1 = vi12c1;
        vi20c1 = vi22c1;

        const float vk02c2x0 = w[100];
        const float vk02c2x1 = w[101];
        const float vk02c2x2 = w[102];
        const float vk02c2x3 = w[103];

        const float vi02c2 = i0[8];

        voc0 += vk02c2x0 * vi02c2;
        voc1 += vk02c2x1 * vi02c2;
        voc2 += vk02c2x2 * vi02c2;
        voc3 += vk02c2x3 * vi02c2;

        const float vk12c2x0 = w[104];
        const float vk12c2x1 = w[105];
        const float vk12c2x2 = w[106];
        const float vk12c2x3 = w[107];

        const float vi12c2 = i1[8];

        voc0 += vk12c2x0 * vi12c2;
        voc1 += vk12c2x1 * vi12c2;
        voc2 += vk12c2x2 * vi12c2;
        voc3 += vk12c2x3 * vi12c2;

        const float vk22c2x0 = w[108];
        const float vk22c2x1 = w[109];
        const float vk22c2x2 = w[110];
        const float vk22c2x3 = w[111];

        const float vi22c2 = i2[8];

        voc0 += vk22c2x0 * vi22c2;
        voc1 += vk22c2x1 * vi22c2;
        voc2 += vk22c2x2 * vi22c2;
        voc3 += vk22c2x3 * vi22c2;

        vi00c2 = vi02c2;
        vi10c2 = vi12c2;
        vi20c2 = vi22c2;

        voc0 = math_min_f32(voc0, voutput_max);
        voc1 = math_min_f32(voc1, voutput_max);
        voc2 = math_min_f32(voc2, voutput_max);
        voc3 = math_min_f32(voc3, voutput_max);

        voc0 = math_max_f32(voc0, voutput_min);
        voc1 = math_max_f32(voc1, voutput_min);
        voc2 = math_max_f32(voc2, voutput_min);
        voc3 = math_max_f32(voc3, voutput_min);

        if XNN_LIKELY(c >= 4) {
          o0[0] = voc0;
          o0[1] = voc1;
          o0[2] = voc2;
          o0[3] = voc3;
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        } else {
          float* o0_tmp = o0;
          if (c & 2) {
            o0_tmp[0] = voc0;
            o0_tmp[1] = voc1;
            o0_tmp += 2;
            voc0 = voc2;
          }
          if (c & 1) {
            *o0_tmp++ = voc0;
          }
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        }

        i0 += 6;
        i1 += 6;
        i2 += 6;
      }
      assert(iw < 2);
      if XNN_UNLIKELY(iw != 0) {
        float voc0 = w[0];
        float voc1 = w[1];
        float voc2 = w[2];
        float voc3 = w[3];

        const float vk00c0x0 = w[4];
        const float vk00c0x1 = w[5];
        const float vk00c0x2 = w[6];
        const float vk00c0x3 = w[7];

        voc0 += vk00c0x0 * vi00c0;
        voc1 += vk00c0x1 * vi00c0;
        voc2 += vk00c0x2 * vi00c0;
        voc3 += vk00c0x3 * vi00c0;

        const float vk10c0x0 = w[8];
        const float vk10c0x1 = w[9];
        const float vk10c0x2 = w[10];
        const float vk10c0x3 = w[11];

        voc0 += vk10c0x0 * vi10c0;
        voc1 += vk10c0x1 * vi10c0;
        voc2 += vk10c0x2 * vi10c0;
        voc3 += vk10c0x3 * vi10c0;

        const float vk20c0x0 = w[12];
        const float vk20c0x1 = w[13];
        const float vk20c0x2 = w[14];
        const float vk20c0x3 = w[15];

        voc0 += vk20c0x0 * vi20c0;
        voc1 += vk20c0x1 * vi20c0;
        voc2 += vk20c0x2 * vi20c0;
        voc3 += vk20c0x3 * vi20c0;

        const float vk00c1x0 = w[16];
        const float vk00c1x1 = w[17];
        const float vk00c1x2 = w[18];
        const float vk00c1x3 = w[19];

        voc0 += vk00c1x0 * vi00c1;
        voc1 += vk00c1x1 * vi00c1;
        voc2 += vk00c1x2 * vi00c1;
        voc3 += vk00c1x3 * vi00c1;

        const float vk10c1x0 = w[20];
        const float vk10c1x1 = w[21];
        const float vk10c1x2 = w[22];
        const float vk10c1x3 = w[23];

        voc0 += vk10c1x0 * vi10c1;
        voc1 += vk10c1x1 * vi10c1;
        voc2 += vk10c1x2 * vi10c1;
        voc3 += vk10c1x3 * vi10c1;

        const float vk20c1x0 = w[24];
        const float vk20c1x1 = w[25];
        const float vk20c1x2 = w[26];
        const float vk20c1x3 = w[27];

        voc0 += vk20c1x0 * vi20c1;
        voc1 += vk20c1x1 * vi20c1;
        voc2 += vk20c1x2 * vi20c1;
        voc3 += vk20c1x3 * vi20c1;

        const float vk00c2x0 = w[28];
        const float vk00c2x1 = w[29];
        const float vk00c2x2 = w[30];
        const float vk00c2x3 = w[31];

        voc0 += vk00c2x0 * vi00c2;
        voc1 += vk00c2x1 * vi00c2;
        voc2 += vk00c2x2 * vi00c2;
        voc3 += vk00c2x3 * vi00c2;

        const float vk10c2x0 = w[32];
        const float vk10c2x1 = w[33];
        const float vk10c2x2 = w[34];
        const float vk10c2x3 = w[35];

        voc0 += vk10c2x0 * vi10c2;
        voc1 += vk10c2x1 * vi10c2;
        voc2 += vk10c2x2 * vi10c2;
        voc3 += vk10c2x3 * vi10c2;

        const float vk20c2x0 = w[36];
        const float vk20c2x1 = w[37];
        const float vk20c2x2 = w[38];
        const float vk20c2x3 = w[39];

        voc0 += vk20c2x0 * vi20c2;
        voc1 += vk20c2x1 * vi20c2;
        voc2 += vk20c2x2 * vi20c2;
        voc3 += vk20c2x3 * vi20c2;

        const float vk01c0x0 = w[40];
        const float vk01c0x1 = w[41];
        const float vk01c0x2 = w[42];
        const float vk01c0x3 = w[43];

        const float vi01c0 = i0[3];

        voc0 += vk01c0x0 * vi01c0;
        voc1 += vk01c0x1 * vi01c0;
        voc2 += vk01c0x2 * vi01c0;
        voc3 += vk01c0x3 * vi01c0;

        const float vk11c0x0 = w[44];
        const float vk11c0x1 = w[45];
        const float vk11c0x2 = w[46];
        const float vk11c0x3 = w[47];

        const float vi11c0 = i1[3];

        voc0 += vk11c0x0 * vi11c0;
        voc1 += vk11c0x1 * vi11c0;
        voc2 += vk11c0x2 * vi11c0;
        voc3 += vk11c0x3 * vi11c0;

        const float vk21c0x0 = w[48];
        const float vk21c0x1 = w[49];
        const float vk21c0x2 = w[50];
        const float vk21c0x3 = w[51];

        const float vi21c0 = i2[3];

        voc0 += vk21c0x0 * vi21c0;
        voc1 += vk21c0x1 * vi21c0;
        voc2 += vk21c0x2 * vi21c0;
        voc3 += vk21c0x3 * vi21c0;

        const float vk01c1x0 = w[52];
        const float vk01c1x1 = w[53];
        const float vk01c1x2 = w[54];
        const float vk01c1x3 = w[55];

        const float vi01c1 = i0[4];

        voc0 += vk01c1x0 * vi01c1;
        voc1 += vk01c1x1 * vi01c1;
        voc2 += vk01c1x2 * vi01c1;
        voc3 += vk01c1x3 * vi01c1;

        const float vk11c1x0 = w[56];
        const float vk11c1x1 = w[57];
        const float vk11c1x2 = w[58];
        const float vk11c1x3 = w[59];

        const float vi11c1 = i1[4];

        voc0 += vk11c1x0 * vi11c1;
        voc1 += vk11c1x1 * vi11c1;
        voc2 += vk11c1x2 * vi11c1;
        voc3 += vk11c1x3 * vi11c1;

        const float vk21c1x0 = w[60];
        const float vk21c1x1 = w[61];
        const float vk21c1x2 = w[62];
        const float vk21c1x3 = w[63];

        const float vi21c1 = i2[4];

        voc0 += vk21c1x0 * vi21c1;
        voc1 += vk21c1x1 * vi21c1;
        voc2 += vk21c1x2 * vi21c1;
        voc3 += vk21c1x3 * vi21c1;

        const float vk01c2x0 = w[64];
        const float vk01c2x1 = w[65];
        const float vk01c2x2 = w[66];
        const float vk01c2x3 = w[67];

        const float vi01c2 = i0[5];

        voc0 += vk01c2x0 * vi01c2;
        voc1 += vk01c2x1 * vi01c2;
        voc2 += vk01c2x2 * vi01c2;
        voc3 += vk01c2x3 * vi01c2;

        const float vk11c2x0 = w[68];
        const float vk11c2x1 = w[69];
        const float vk11c2x2 = w[70];
        const float vk11c2x3 = w[71];

        const float vi11c2 = i1[5];

        voc0 += vk11c2x0 * vi11c2;
        voc1 += vk11c2x1 * vi11c2;
        voc2 += vk11c2x2 * vi11c2;
        voc3 += vk11c2x3 * vi11c2;

        const float vk21c2x0 = w[72];
        const float vk21c2x1 = w[73];
        const float vk21c2x2 = w[74];
        const float vk21c2x3 = w[75];

        const float vi21c2 = i2[5];

        voc0 += vk21c2x0 * vi21c2;
        voc1 += vk21c2x1 * vi21c2;
        voc2 += vk21c2x2 * vi21c2;
        voc3 += vk21c2x3 * vi21c2;

        voc0 = math_min_f32(voc0, voutput_max);
        voc1 = math_min_f32(voc1, voutput_max);
        voc2 = math_min_f32(voc2, voutput_max);
        voc3 = math_min_f32(voc3, voutput_max);

        voc0 = math_max_f32(voc0, voutput_min);
        voc1 = math_max_f32(voc1, voutput_min);
        voc2 = math_max_f32(voc2, voutput_min);
        voc3 = math_max_f32(voc3, voutput_min);

        if XNN_LIKELY(c >= 4) {
          o0[0] = voc0;
          o0[1] = voc1;
          o0[2] = voc2;
          o0[3] = voc3;
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        } else {
          float* o0_tmp = o0;
          if (c & 2) {
            o0_tmp[0] = voc0;
            o0_tmp[1] = voc1;
            o0_tmp += 2;
            voc0 = voc2;
          }
          if (c & 1) {
            *o0_tmp++ = voc0;
          }
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels
      o0 = (float*) ((uintptr_t) o0 - output_channel_decrement);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_decrement);
      i1 = (const float*) ((uintptr_t) i1 - input_width_decrement);
      i2 = (const float*) ((uintptr_t) i2 - input_width_decrement);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers back to the position of the first channel, and forward to the next block of rows
    o0 = (float*) ((uintptr_t) o0 + output_height_increment);
    // Move input pointers forward to the next row
    i0 = i2;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  }
}

void xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__scalar_1x1(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_width_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_decrement = round_down_po2(input_width, 2) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_decrement = output_width * output_width_stride - 4 * sizeof(float);
  const size_t output_height_increment = output_height_stride - round_up_po2(output_channels, 4) * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  float* o0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float voutput_max = params->scalar.max;
  const float voutput_min = params->scalar.min;

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 1) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }

    const float* w = weights;
    size_t c = output_channels;
    do {
      float vi00c0 = 0.0f;
      float vi00c1 = 0.0f;
      float vi00c2 = 0.0f;
      float vi10c0 = 0.0f;
      float vi10c1 = 0.0f;
      float vi10c2 = 0.0f;
      float vi20c0 = 0.0f;
      float vi20c1 = 0.0f;
      float vi20c2 = 0.0f;

      size_t iw = input_width;
      for (; iw >= 2; iw -= 2) {
        // start with biases
        float voc0 = w[0];
        float voc1 = w[1];
        float voc2 = w[2];
        float voc3 = w[3];

        const float vk00c0x0 = w[4];
        const float vk00c0x1 = w[5];
        const float vk00c0x2 = w[6];
        const float vk00c0x3 = w[7];

        voc0 += vk00c0x0 * vi00c0;
        voc1 += vk00c0x1 * vi00c0;
        voc2 += vk00c0x2 * vi00c0;
        voc3 += vk00c0x3 * vi00c0;

        const float vk10c0x0 = w[8];
        const float vk10c0x1 = w[9];
        const float vk10c0x2 = w[10];
        const float vk10c0x3 = w[11];

        voc0 += vk10c0x0 * vi10c0;
        voc1 += vk10c0x1 * vi10c0;
        voc2 += vk10c0x2 * vi10c0;
        voc3 += vk10c0x3 * vi10c0;

        const float vk20c0x0 = w[12];
        const float vk20c0x1 = w[13];
        const float vk20c0x2 = w[14];
        const float vk20c0x3 = w[15];

        voc0 += vk20c0x0 * vi20c0;
        voc1 += vk20c0x1 * vi20c0;
        voc2 += vk20c0x2 * vi20c0;
        voc3 += vk20c0x3 * vi20c0;

        const float vk00c1x0 = w[16];
        const float vk00c1x1 = w[17];
        const float vk00c1x2 = w[18];
        const float vk00c1x3 = w[19];

        voc0 += vk00c1x0 * vi00c1;
        voc1 += vk00c1x1 * vi00c1;
        voc2 += vk00c1x2 * vi00c1;
        voc3 += vk00c1x3 * vi00c1;

        const float vk10c1x0 = w[20];
        const float vk10c1x1 = w[21];
        const float vk10c1x2 = w[22];
        const float vk10c1x3 = w[23];

        voc0 += vk10c1x0 * vi10c1;
        voc1 += vk10c1x1 * vi10c1;
        voc2 += vk10c1x2 * vi10c1;
        voc3 += vk10c1x3 * vi10c1;

        const float vk20c1x0 = w[24];
        const float vk20c1x1 = w[25];
        const float vk20c1x2 = w[26];
        const float vk20c1x3 = w[27];

        voc0 += vk20c1x0 * vi20c1;
        voc1 += vk20c1x1 * vi20c1;
        voc2 += vk20c1x2 * vi20c1;
        voc3 += vk20c1x3 * vi20c1;

        const float vk00c2x0 = w[28];
        const float vk00c2x1 = w[29];
        const float vk00c2x2 = w[30];
        const float vk00c2x3 = w[31];

        voc0 += vk00c2x0 * vi00c2;
        voc1 += vk00c2x1 * vi00c2;
        voc2 += vk00c2x2 * vi00c2;
        voc3 += vk00c2x3 * vi00c2;

        const float vk10c2x0 = w[32];
        const float vk10c2x1 = w[33];
        const float vk10c2x2 = w[34];
        const float vk10c2x3 = w[35];

        voc0 += vk10c2x0 * vi10c2;
        voc1 += vk10c2x1 * vi10c2;
        voc2 += vk10c2x2 * vi10c2;
        voc3 += vk10c2x3 * vi10c2;

        const float vk20c2x0 = w[36];
        const float vk20c2x1 = w[37];
        const float vk20c2x2 = w[38];
        const float vk20c2x3 = w[39];

        voc0 += vk20c2x0 * vi20c2;
        voc1 += vk20c2x1 * vi20c2;
        voc2 += vk20c2x2 * vi20c2;
        voc3 += vk20c2x3 * vi20c2;

        const float vk01c0x0 = w[40];
        const float vk01c0x1 = w[41];
        const float vk01c0x2 = w[42];
        const float vk01c0x3 = w[43];

        const float vi01c0 = i0[0];

        voc0 += vk01c0x0 * vi01c0;
        voc1 += vk01c0x1 * vi01c0;
        voc2 += vk01c0x2 * vi01c0;
        voc3 += vk01c0x3 * vi01c0;

        const float vk11c0x0 = w[44];
        const float vk11c0x1 = w[45];
        const float vk11c0x2 = w[46];
        const float vk11c0x3 = w[47];

        const float vi11c0 = i1[0];

        voc0 += vk11c0x0 * vi11c0;
        voc1 += vk11c0x1 * vi11c0;
        voc2 += vk11c0x2 * vi11c0;
        voc3 += vk11c0x3 * vi11c0;

        const float vk21c0x0 = w[48];
        const float vk21c0x1 = w[49];
        const float vk21c0x2 = w[50];
        const float vk21c0x3 = w[51];

        const float vi21c0 = i2[0];

        voc0 += vk21c0x0 * vi21c0;
        voc1 += vk21c0x1 * vi21c0;
        voc2 += vk21c0x2 * vi21c0;
        voc3 += vk21c0x3 * vi21c0;

        const float vk01c1x0 = w[52];
        const float vk01c1x1 = w[53];
        const float vk01c1x2 = w[54];
        const float vk01c1x3 = w[55];

        const float vi01c1 = i0[1];

        voc0 += vk01c1x0 * vi01c1;
        voc1 += vk01c1x1 * vi01c1;
        voc2 += vk01c1x2 * vi01c1;
        voc3 += vk01c1x3 * vi01c1;

        const float vk11c1x0 = w[56];
        const float vk11c1x1 = w[57];
        const float vk11c1x2 = w[58];
        const float vk11c1x3 = w[59];

        const float vi11c1 = i1[1];

        voc0 += vk11c1x0 * vi11c1;
        voc1 += vk11c1x1 * vi11c1;
        voc2 += vk11c1x2 * vi11c1;
        voc3 += vk11c1x3 * vi11c1;

        const float vk21c1x0 = w[60];
        const float vk21c1x1 = w[61];
        const float vk21c1x2 = w[62];
        const float vk21c1x3 = w[63];

        const float vi21c1 = i2[1];

        voc0 += vk21c1x0 * vi21c1;
        voc1 += vk21c1x1 * vi21c1;
        voc2 += vk21c1x2 * vi21c1;
        voc3 += vk21c1x3 * vi21c1;

        const float vk01c2x0 = w[64];
        const float vk01c2x1 = w[65];
        const float vk01c2x2 = w[66];
        const float vk01c2x3 = w[67];

        const float vi01c2 = i0[2];

        voc0 += vk01c2x0 * vi01c2;
        voc1 += vk01c2x1 * vi01c2;
        voc2 += vk01c2x2 * vi01c2;
        voc3 += vk01c2x3 * vi01c2;

        const float vk11c2x0 = w[68];
        const float vk11c2x1 = w[69];
        const float vk11c2x2 = w[70];
        const float vk11c2x3 = w[71];

        const float vi11c2 = i1[2];

        voc0 += vk11c2x0 * vi11c2;
        voc1 += vk11c2x1 * vi11c2;
        voc2 += vk11c2x2 * vi11c2;
        voc3 += vk11c2x3 * vi11c2;

        const float vk21c2x0 = w[72];
        const float vk21c2x1 = w[73];
        const float vk21c2x2 = w[74];
        const float vk21c2x3 = w[75];

        const float vi21c2 = i2[2];

        voc0 += vk21c2x0 * vi21c2;
        voc1 += vk21c2x1 * vi21c2;
        voc2 += vk21c2x2 * vi21c2;
        voc3 += vk21c2x3 * vi21c2;

        const float vk02c0x0 = w[76];
        const float vk02c0x1 = w[77];
        const float vk02c0x2 = w[78];
        const float vk02c0x3 = w[79];

        const float vi02c0 = i0[3];

        voc0 += vk02c0x0 * vi02c0;
        voc1 += vk02c0x1 * vi02c0;
        voc2 += vk02c0x2 * vi02c0;
        voc3 += vk02c0x3 * vi02c0;

        const float vk12c0x0 = w[80];
        const float vk12c0x1 = w[81];
        const float vk12c0x2 = w[82];
        const float vk12c0x3 = w[83];

        const float vi12c0 = i1[3];

        voc0 += vk12c0x0 * vi12c0;
        voc1 += vk12c0x1 * vi12c0;
        voc2 += vk12c0x2 * vi12c0;
        voc3 += vk12c0x3 * vi12c0;

        const float vk22c0x0 = w[84];
        const float vk22c0x1 = w[85];
        const float vk22c0x2 = w[86];
        const float vk22c0x3 = w[87];

        const float vi22c0 = i2[3];

        voc0 += vk22c0x0 * vi22c0;
        voc1 += vk22c0x1 * vi22c0;
        voc2 += vk22c0x2 * vi22c0;
        voc3 += vk22c0x3 * vi22c0;

        vi00c0 = vi02c0;
        vi10c0 = vi12c0;
        vi20c0 = vi22c0;

        const float vk02c1x0 = w[88];
        const float vk02c1x1 = w[89];
        const float vk02c1x2 = w[90];
        const float vk02c1x3 = w[91];

        const float vi02c1 = i0[4];

        voc0 += vk02c1x0 * vi02c1;
        voc1 += vk02c1x1 * vi02c1;
        voc2 += vk02c1x2 * vi02c1;
        voc3 += vk02c1x3 * vi02c1;

        const float vk12c1x0 = w[92];
        const float vk12c1x1 = w[93];
        const float vk12c1x2 = w[94];
        const float vk12c1x3 = w[95];

        const float vi12c1 = i1[4];

        voc0 += vk12c1x0 * vi12c1;
        voc1 += vk12c1x1 * vi12c1;
        voc2 += vk12c1x2 * vi12c1;
        voc3 += vk12c1x3 * vi12c1;

        const float vk22c1x0 = w[96];
        const float vk22c1x1 = w[97];
        const float vk22c1x2 = w[98];
        const float vk22c1x3 = w[99];

        const float vi22c1 = i2[4];

        voc0 += vk22c1x0 * vi22c1;
        voc1 += vk22c1x1 * vi22c1;
        voc2 += vk22c1x2 * vi22c1;
        voc3 += vk22c1x3 * vi22c1;

        vi00c1 = vi02c1;
        vi10c1 = vi12c1;
        vi20c1 = vi22c1;

        const float vk02c2x0 = w[100];
        const float vk02c2x1 = w[101];
        const float vk02c2x2 = w[102];
        const float vk02c2x3 = w[103];

        const float vi02c2 = i0[5];

        voc0 += vk02c2x0 * vi02c2;
        voc1 += vk02c2x1 * vi02c2;
        voc2 += vk02c2x2 * vi02c2;
        voc3 += vk02c2x3 * vi02c2;

        const float vk12c2x0 = w[104];
        const float vk12c2x1 = w[105];
        const float vk12c2x2 = w[106];
        const float vk12c2x3 = w[107];

        const float vi12c2 = i1[5];

        voc0 += vk12c2x0 * vi12c2;
        voc1 += vk12c2x1 * vi12c2;
        voc2 += vk12c2x2 * vi12c2;
        voc3 += vk12c2x3 * vi12c2;

        const float vk22c2x0 = w[108];
        const float vk22c2x1 = w[109];
        const float vk22c2x2 = w[110];
        const float vk22c2x3 = w[111];

        const float vi22c2 = i2[5];

        voc0 += vk22c2x0 * vi22c2;
        voc1 += vk22c2x1 * vi22c2;
        voc2 += vk22c2x2 * vi22c2;
        voc3 += vk22c2x3 * vi22c2;

        vi00c2 = vi02c2;
        vi10c2 = vi12c2;
        vi20c2 = vi22c2;

        voc0 = math_min_f32(voc0, voutput_max);
        voc1 = math_min_f32(voc1, voutput_max);
        voc2 = math_min_f32(voc2, voutput_max);
        voc3 = math_min_f32(voc3, voutput_max);

        voc0 = math_max_f32(voc0, voutput_min);
        voc1 = math_max_f32(voc1, voutput_min);
        voc2 = math_max_f32(voc2, voutput_min);
        voc3 = math_max_f32(voc3, voutput_min);

        if XNN_LIKELY(c >= 4) {
          o0[0] = voc0;
          o0[1] = voc1;
          o0[2] = voc2;
          o0[3] = voc3;
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        } else {
          float* o0_tmp = o0;
          if (c & 2) {
            o0_tmp[0] = voc0;
            o0_tmp[1] = voc1;
            o0_tmp += 2;
            voc0 = voc2;
          }
          if (c & 1) {
            *o0_tmp++ = voc0;
          }
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        }

        i0 += 6;
        i1 += 6;
        i2 += 6;
      }
      assert(iw < 2);
      if XNN_UNLIKELY(iw != 0) {
        float voc0 = w[0];
        float voc1 = w[1];
        float voc2 = w[2];
        float voc3 = w[3];

        const float vk00c0x0 = w[4];
        const float vk00c0x1 = w[5];
        const float vk00c0x2 = w[6];
        const float vk00c0x3 = w[7];

        voc0 += vk00c0x0 * vi00c0;
        voc1 += vk00c0x1 * vi00c0;
        voc2 += vk00c0x2 * vi00c0;
        voc3 += vk00c0x3 * vi00c0;

        const float vk10c0x0 = w[8];
        const float vk10c0x1 = w[9];
        const float vk10c0x2 = w[10];
        const float vk10c0x3 = w[11];

        voc0 += vk10c0x0 * vi10c0;
        voc1 += vk10c0x1 * vi10c0;
        voc2 += vk10c0x2 * vi10c0;
        voc3 += vk10c0x3 * vi10c0;

        const float vk20c0x0 = w[12];
        const float vk20c0x1 = w[13];
        const float vk20c0x2 = w[14];
        const float vk20c0x3 = w[15];

        voc0 += vk20c0x0 * vi20c0;
        voc1 += vk20c0x1 * vi20c0;
        voc2 += vk20c0x2 * vi20c0;
        voc3 += vk20c0x3 * vi20c0;

        const float vk00c1x0 = w[16];
        const float vk00c1x1 = w[17];
        const float vk00c1x2 = w[18];
        const float vk00c1x3 = w[19];

        voc0 += vk00c1x0 * vi00c1;
        voc1 += vk00c1x1 * vi00c1;
        voc2 += vk00c1x2 * vi00c1;
        voc3 += vk00c1x3 * vi00c1;

        const float vk10c1x0 = w[20];
        const float vk10c1x1 = w[21];
        const float vk10c1x2 = w[22];
        const float vk10c1x3 = w[23];

        voc0 += vk10c1x0 * vi10c1;
        voc1 += vk10c1x1 * vi10c1;
        voc2 += vk10c1x2 * vi10c1;
        voc3 += vk10c1x3 * vi10c1;

        const float vk20c1x0 = w[24];
        const float vk20c1x1 = w[25];
        const float vk20c1x2 = w[26];
        const float vk20c1x3 = w[27];

        voc0 += vk20c1x0 * vi20c1;
        voc1 += vk20c1x1 * vi20c1;
        voc2 += vk20c1x2 * vi20c1;
        voc3 += vk20c1x3 * vi20c1;

        const float vk00c2x0 = w[28];
        const float vk00c2x1 = w[29];
        const float vk00c2x2 = w[30];
        const float vk00c2x3 = w[31];

        voc0 += vk00c2x0 * vi00c2;
        voc1 += vk00c2x1 * vi00c2;
        voc2 += vk00c2x2 * vi00c2;
        voc3 += vk00c2x3 * vi00c2;

        const float vk10c2x0 = w[32];
        const float vk10c2x1 = w[33];
        const float vk10c2x2 = w[34];
        const float vk10c2x3 = w[35];

        voc0 += vk10c2x0 * vi10c2;
        voc1 += vk10c2x1 * vi10c2;
        voc2 += vk10c2x2 * vi10c2;
        voc3 += vk10c2x3 * vi10c2;

        const float vk20c2x0 = w[36];
        const float vk20c2x1 = w[37];
        const float vk20c2x2 = w[38];
        const float vk20c2x3 = w[39];

        voc0 += vk20c2x0 * vi20c2;
        voc1 += vk20c2x1 * vi20c2;
        voc2 += vk20c2x2 * vi20c2;
        voc3 += vk20c2x3 * vi20c2;

        const float vk01c0x0 = w[40];
        const float vk01c0x1 = w[41];
        const float vk01c0x2 = w[42];
        const float vk01c0x3 = w[43];

        const float vi01c0 = i0[0];

        voc0 += vk01c0x0 * vi01c0;
        voc1 += vk01c0x1 * vi01c0;
        voc2 += vk01c0x2 * vi01c0;
        voc3 += vk01c0x3 * vi01c0;

        const float vk11c0x0 = w[44];
        const float vk11c0x1 = w[45];
        const float vk11c0x2 = w[46];
        const float vk11c0x3 = w[47];

        const float vi11c0 = i1[0];

        voc0 += vk11c0x0 * vi11c0;
        voc1 += vk11c0x1 * vi11c0;
        voc2 += vk11c0x2 * vi11c0;
        voc3 += vk11c0x3 * vi11c0;

        const float vk21c0x0 = w[48];
        const float vk21c0x1 = w[49];
        const float vk21c0x2 = w[50];
        const float vk21c0x3 = w[51];

        const float vi21c0 = i2[0];

        voc0 += vk21c0x0 * vi21c0;
        voc1 += vk21c0x1 * vi21c0;
        voc2 += vk21c0x2 * vi21c0;
        voc3 += vk21c0x3 * vi21c0;

        const float vk01c1x0 = w[52];
        const float vk01c1x1 = w[53];
        const float vk01c1x2 = w[54];
        const float vk01c1x3 = w[55];

        const float vi01c1 = i0[1];

        voc0 += vk01c1x0 * vi01c1;
        voc1 += vk01c1x1 * vi01c1;
        voc2 += vk01c1x2 * vi01c1;
        voc3 += vk01c1x3 * vi01c1;

        const float vk11c1x0 = w[56];
        const float vk11c1x1 = w[57];
        const float vk11c1x2 = w[58];
        const float vk11c1x3 = w[59];

        const float vi11c1 = i1[1];

        voc0 += vk11c1x0 * vi11c1;
        voc1 += vk11c1x1 * vi11c1;
        voc2 += vk11c1x2 * vi11c1;
        voc3 += vk11c1x3 * vi11c1;

        const float vk21c1x0 = w[60];
        const float vk21c1x1 = w[61];
        const float vk21c1x2 = w[62];
        const float vk21c1x3 = w[63];

        const float vi21c1 = i2[1];

        voc0 += vk21c1x0 * vi21c1;
        voc1 += vk21c1x1 * vi21c1;
        voc2 += vk21c1x2 * vi21c1;
        voc3 += vk21c1x3 * vi21c1;

        const float vk01c2x0 = w[64];
        const float vk01c2x1 = w[65];
        const float vk01c2x2 = w[66];
        const float vk01c2x3 = w[67];

        const float vi01c2 = i0[2];

        voc0 += vk01c2x0 * vi01c2;
        voc1 += vk01c2x1 * vi01c2;
        voc2 += vk01c2x2 * vi01c2;
        voc3 += vk01c2x3 * vi01c2;

        const float vk11c2x0 = w[68];
        const float vk11c2x1 = w[69];
        const float vk11c2x2 = w[70];
        const float vk11c2x3 = w[71];

        const float vi11c2 = i1[2];

        voc0 += vk11c2x0 * vi11c2;
        voc1 += vk11c2x1 * vi11c2;
        voc2 += vk11c2x2 * vi11c2;
        voc3 += vk11c2x3 * vi11c2;

        const float vk21c2x0 = w[72];
        const float vk21c2x1 = w[73];
        const float vk21c2x2 = w[74];
        const float vk21c2x3 = w[75];

        const float vi21c2 = i2[2];

        voc0 += vk21c2x0 * vi21c2;
        voc1 += vk21c2x1 * vi21c2;
        voc2 += vk21c2x2 * vi21c2;
        voc3 += vk21c2x3 * vi21c2;

        voc0 = math_min_f32(voc0, voutput_max);
        voc1 = math_min_f32(voc1, voutput_max);
        voc2 = math_min_f32(voc2, voutput_max);
        voc3 = math_min_f32(voc3, voutput_max);

        voc0 = math_max_f32(voc0, voutput_min);
        voc1 = math_max_f32(voc1, voutput_min);
        voc2 = math_max_f32(voc2, voutput_min);
        voc3 = math_max_f32(voc3, voutput_min);

        if XNN_LIKELY(c >= 4) {
          o0[0] = voc0;
          o0[1] = voc1;
          o0[2] = voc2;
          o0[3] = voc3;
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        } else {
          float* o0_tmp = o0;
          if (c & 2) {
            o0_tmp[0] = voc0;
            o0_tmp[1] = voc1;
            o0_tmp += 2;
            voc0 = voc2;
          }
          if (c & 1) {
            *o0_tmp++ = voc0;
          }
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels
      o0 = (float*) ((uintptr_t) o0 - output_channel_decrement);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_decrement);
      i1 = (const float*) ((uintptr_t) i1 - input_width_decrement);
      i2 = (const float*) ((uintptr_t) i2 - input_width_decrement);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers back to the position of the first channel, and forward to the next block of rows
    o0 = (float*) ((uintptr_t) o0 + output_height_increment);
    // Move input pointers forward to the next row
    i0 = i2;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  }
}

void xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_decrement = round_down_po2(input_width, 2) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float voutput_max = params->scalar.max;
  const float voutput_min = params->scalar.min;

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 1) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0c0 = output0;
    float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
    float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
    float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
      }

      // Left edge padding
      float vi00c0 = 0.0f;
      float vi00c1 = 0.0f;
      float vi00c2 = 0.0f;
      float vi10c0 = 0.0f;
      float vi10c1 = 0.0f;
      float vi10c2 = 0.0f;
      float vi20c0 = 0.0f;
      float vi20c1 = 0.0f;
      float vi20c2 = 0.0f;

      size_t iw = input_width;
      for (; iw >= 2; iw -= 2) {
        float voc0 = w[0];
        float voc1 = w[1];
        float voc2 = w[2];
        float voc3 = w[3];

        const float vk00c0x0 = w[4];
        const float vk00c0x1 = w[5];
        const float vk00c0x2 = w[6];
        const float vk00c0x3 = w[7];

        voc0 += vk00c0x0 * vi00c0;
        voc1 += vk00c0x1 * vi00c0;
        voc2 += vk00c0x2 * vi00c0;
        voc3 += vk00c0x3 * vi00c0;

        const float vk10c0x0 = w[8];
        const float vk10c0x1 = w[9];
        const float vk10c0x2 = w[10];
        const float vk10c0x3 = w[11];

        voc0 += vk10c0x0 * vi10c0;
        voc1 += vk10c0x1 * vi10c0;
        voc2 += vk10c0x2 * vi10c0;
        voc3 += vk10c0x3 * vi10c0;

        const float vk20c0x0 = w[12];
        const float vk20c0x1 = w[13];
        const float vk20c0x2 = w[14];
        const float vk20c0x3 = w[15];

        voc0 += vk20c0x0 * vi20c0;
        voc1 += vk20c0x1 * vi20c0;
        voc2 += vk20c0x2 * vi20c0;
        voc3 += vk20c0x3 * vi20c0;

        const float vk00c1x0 = w[16];
        const float vk00c1x1 = w[17];
        const float vk00c1x2 = w[18];
        const float vk00c1x3 = w[19];

        voc0 += vk00c1x0 * vi00c1;
        voc1 += vk00c1x1 * vi00c1;
        voc2 += vk00c1x2 * vi00c1;
        voc3 += vk00c1x3 * vi00c1;

        const float vk10c1x0 = w[20];
        const float vk10c1x1 = w[21];
        const float vk10c1x2 = w[22];
        const float vk10c1x3 = w[23];

        voc0 += vk10c1x0 * vi10c1;
        voc1 += vk10c1x1 * vi10c1;
        voc2 += vk10c1x2 * vi10c1;
        voc3 += vk10c1x3 * vi10c1;

        const float vk20c1x0 = w[24];
        const float vk20c1x1 = w[25];
        const float vk20c1x2 = w[26];
        const float vk20c1x3 = w[27];

        voc0 += vk20c1x0 * vi20c1;
        voc1 += vk20c1x1 * vi20c1;
        voc2 += vk20c1x2 * vi20c1;
        voc3 += vk20c1x3 * vi20c1;

        const float vk00c2x0 = w[28];
        const float vk00c2x1 = w[29];
        const float vk00c2x2 = w[30];
        const float vk00c2x3 = w[31];

        voc0 += vk00c2x0 * vi00c2;
        voc1 += vk00c2x1 * vi00c2;
        voc2 += vk00c2x2 * vi00c2;
        voc3 += vk00c2x3 * vi00c2;

        const float vk10c2x0 = w[32];
        const float vk10c2x1 = w[33];
        const float vk10c2x2 = w[34];
        const float vk10c2x3 = w[35];

        voc0 += vk10c2x0 * vi10c2;
        voc1 += vk10c2x1 * vi10c2;
        voc2 += vk10c2x2 * vi10c2;
        voc3 += vk10c2x3 * vi10c2;

        const float vk20c2x0 = w[36];
        const float vk20c2x1 = w[37];
        const float vk20c2x2 = w[38];
        const float vk20c2x3 = w[39];

        voc0 += vk20c2x0 * vi20c2;
        voc1 += vk20c2x1 * vi20c2;
        voc2 += vk20c2x2 * vi20c2;
        voc3 += vk20c2x3 * vi20c2;

        const float vk01c0x0 = w[40];
        const float vk01c0x1 = w[41];
        const float vk01c0x2 = w[42];
        const float vk01c0x3 = w[43];

        const float vi01c0 = i0[0];

        voc0 += vk01c0x0 * vi01c0;
        voc1 += vk01c0x1 * vi01c0;
        voc2 += vk01c0x2 * vi01c0;
        voc3 += vk01c0x3 * vi01c0;

        const float vk11c0x0 = w[44];
        const float vk11c0x1 = w[45];
        const float vk11c0x2 = w[46];
        const float vk11c0x3 = w[47];

        const float vi11c0 = i1[0];

        voc0 += vk11c0x0 * vi11c0;
        voc1 += vk11c0x1 * vi11c0;
        voc2 += vk11c0x2 * vi11c0;
        voc3 += vk11c0x3 * vi11c0;

        const float vk21c0x0 = w[48];
        const float vk21c0x1 = w[49];
        const float vk21c0x2 = w[50];
        const float vk21c0x3 = w[51];

        const float vi21c0 = i2[0];

        voc0 += vk21c0x0 * vi21c0;
        voc1 += vk21c0x1 * vi21c0;
        voc2 += vk21c0x2 * vi21c0;
        voc3 += vk21c0x3 * vi21c0;

        const float vk01c1x0 = w[52];
        const float vk01c1x1 = w[53];
        const float vk01c1x2 = w[54];
        const float vk01c1x3 = w[55];

        const float vi01c1 = i0[1];

        voc0 += vk01c1x0 * vi01c1;
        voc1 += vk01c1x1 * vi01c1;
        voc2 += vk01c1x2 * vi01c1;
        voc3 += vk01c1x3 * vi01c1;

        const float vk11c1x0 = w[56];
        const float vk11c1x1 = w[57];
        const float vk11c1x2 = w[58];
        const float vk11c1x3 = w[59];

        const float vi11c1 = i1[1];

        voc0 += vk11c1x0 * vi11c1;
        voc1 += vk11c1x1 * vi11c1;
        voc2 += vk11c1x2 * vi11c1;
        voc3 += vk11c1x3 * vi11c1;

        const float vk21c1x0 = w[60];
        const float vk21c1x1 = w[61];
        const float vk21c1x2 = w[62];
        const float vk21c1x3 = w[63];

        const float vi21c1 = i2[1];

        voc0 += vk21c1x0 * vi21c1;
        voc1 += vk21c1x1 * vi21c1;
        voc2 += vk21c1x2 * vi21c1;
        voc3 += vk21c1x3 * vi21c1;

        const float vk01c2x0 = w[64];
        const float vk01c2x1 = w[65];
        const float vk01c2x2 = w[66];
        const float vk01c2x3 = w[67];

        const float vi01c2 = i0[2];

        voc0 += vk01c2x0 * vi01c2;
        voc1 += vk01c2x1 * vi01c2;
        voc2 += vk01c2x2 * vi01c2;
        voc3 += vk01c2x3 * vi01c2;

        const float vk11c2x0 = w[68];
        const float vk11c2x1 = w[69];
        const float vk11c2x2 = w[70];
        const float vk11c2x3 = w[71];

        const float vi11c2 = i1[2];

        voc0 += vk11c2x0 * vi11c2;
        voc1 += vk11c2x1 * vi11c2;
        voc2 += vk11c2x2 * vi11c2;
        voc3 += vk11c2x3 * vi11c2;

        const float vk21c2x0 = w[72];
        const float vk21c2x1 = w[73];
        const float vk21c2x2 = w[74];
        const float vk21c2x3 = w[75];

        const float vi21c2 = i2[2];

        voc0 += vk21c2x0 * vi21c2;
        voc1 += vk21c2x1 * vi21c2;
        voc2 += vk21c2x2 * vi21c2;
        voc3 += vk21c2x3 * vi21c2;

        const float vk02c0x0 = w[76];
        const float vk02c0x1 = w[77];
        const float vk02c0x2 = w[78];
        const float vk02c0x3 = w[79];

        const float vi02c0 = i0[3];

        voc0 += vk02c0x0 * vi02c0;
        voc1 += vk02c0x1 * vi02c0;
        voc2 += vk02c0x2 * vi02c0;
        voc3 += vk02c0x3 * vi02c0;

        const float vk12c0x0 = w[80];
        const float vk12c0x1 = w[81];
        const float vk12c0x2 = w[82];
        const float vk12c0x3 = w[83];

        const float vi12c0 = i1[3];

        voc0 += vk12c0x0 * vi12c0;
        voc1 += vk12c0x1 * vi12c0;
        voc2 += vk12c0x2 * vi12c0;
        voc3 += vk12c0x3 * vi12c0;

        const float vk22c0x0 = w[84];
        const float vk22c0x1 = w[85];
        const float vk22c0x2 = w[86];
        const float vk22c0x3 = w[87];

        const float vi22c0 = i2[3];

        voc0 += vk22c0x0 * vi22c0;
        voc1 += vk22c0x1 * vi22c0;
        voc2 += vk22c0x2 * vi22c0;
        voc3 += vk22c0x3 * vi22c0;

        vi00c0 = vi02c0;
        vi10c0 = vi12c0;
        vi20c0 = vi22c0;

        const float vk02c1x0 = w[88];
        const float vk02c1x1 = w[89];
        const float vk02c1x2 = w[90];
        const float vk02c1x3 = w[91];

        const float vi02c1 = i0[4];

        voc0 += vk02c1x0 * vi02c1;
        voc1 += vk02c1x1 * vi02c1;
        voc2 += vk02c1x2 * vi02c1;
        voc3 += vk02c1x3 * vi02c1;

        const float vk12c1x0 = w[92];
        const float vk12c1x1 = w[93];
        const float vk12c1x2 = w[94];
        const float vk12c1x3 = w[95];

        const float vi12c1 = i1[4];

        voc0 += vk12c1x0 * vi12c1;
        voc1 += vk12c1x1 * vi12c1;
        voc2 += vk12c1x2 * vi12c1;
        voc3 += vk12c1x3 * vi12c1;

        const float vk22c1x0 = w[96];
        const float vk22c1x1 = w[97];
        const float vk22c1x2 = w[98];
        const float vk22c1x3 = w[99];

        const float vi22c1 = i2[4];

        voc0 += vk22c1x0 * vi22c1;
        voc1 += vk22c1x1 * vi22c1;
        voc2 += vk22c1x2 * vi22c1;
        voc3 += vk22c1x3 * vi22c1;

        vi00c1 = vi02c1;
        vi10c1 = vi12c1;
        vi20c1 = vi22c1;

        const float vk02c2x0 = w[100];
        const float vk02c2x1 = w[101];
        const float vk02c2x2 = w[102];
        const float vk02c2x3 = w[103];

        const float vi02c2 = i0[5];

        voc0 += vk02c2x0 * vi02c2;
        voc1 += vk02c2x1 * vi02c2;
        voc2 += vk02c2x2 * vi02c2;
        voc3 += vk02c2x3 * vi02c2;

        const float vk12c2x0 = w[104];
        const float vk12c2x1 = w[105];
        const float vk12c2x2 = w[106];
        const float vk12c2x3 = w[107];

        const float vi12c2 = i1[5];

        voc0 += vk12c2x0 * vi12c2;
        voc1 += vk12c2x1 * vi12c2;
        voc2 += vk12c2x2 * vi12c2;
        voc3 += vk12c2x3 * vi12c2;

        const float vk22c2x0 = w[108];
        const float vk22c2x1 = w[109];
        const float vk22c2x2 = w[110];
        const float vk22c2x3 = w[111];

        const float vi22c2 = i2[5];

        voc0 += vk22c2x0 * vi22c2;
        voc1 += vk22c2x1 * vi22c2;
        voc2 += vk22c2x2 * vi22c2;
        voc3 += vk22c2x3 * vi22c2;

        vi00c2 = vi02c2;
        vi10c2 = vi12c2;
        vi20c2 = vi22c2;

        voc0 = math_min_f32(voc0, voutput_max);
        voc1 = math_min_f32(voc1, voutput_max);
        voc2 = math_min_f32(voc2, voutput_max);
        voc3 = math_min_f32(voc3, voutput_max);

        voc0 = math_max_f32(voc0, voutput_min);
        voc1 = math_max_f32(voc1, voutput_min);
        voc2 = math_max_f32(voc2, voutput_min);
        voc3 = math_max_f32(voc3, voutput_min);

        *o0c0++ = voc0;
        *o0c1++ = voc1;
        *o0c2++ = voc2;
        *o0c3++ = voc3;

        i0 += 6;
        i1 += 6;
        i2 += 6;
      }
      assert(iw < 2);
      if XNN_UNLIKELY(iw != 0) {
        float voc0 = w[0];
        float voc1 = w[1];
        float voc2 = w[2];
        float voc3 = w[3];

        const float vk00c0x0 = w[4];
        const float vk00c0x1 = w[5];
        const float vk00c0x2 = w[6];
        const float vk00c0x3 = w[7];

        voc0 += vk00c0x0 * vi00c0;
        voc1 += vk00c0x1 * vi00c0;
        voc2 += vk00c0x2 * vi00c0;
        voc3 += vk00c0x3 * vi00c0;

        const float vk10c0x0 = w[8];
        const float vk10c0x1 = w[9];
        const float vk10c0x2 = w[10];
        const float vk10c0x3 = w[11];

        voc0 += vk10c0x0 * vi10c0;
        voc1 += vk10c0x1 * vi10c0;
        voc2 += vk10c0x2 * vi10c0;
        voc3 += vk10c0x3 * vi10c0;

        const float vk20c0x0 = w[12];
        const float vk20c0x1 = w[13];
        const float vk20c0x2 = w[14];
        const float vk20c0x3 = w[15];

        voc0 += vk20c0x0 * vi20c0;
        voc1 += vk20c0x1 * vi20c0;
        voc2 += vk20c0x2 * vi20c0;
        voc3 += vk20c0x3 * vi20c0;

        const float vk00c1x0 = w[16];
        const float vk00c1x1 = w[17];
        const float vk00c1x2 = w[18];
        const float vk00c1x3 = w[19];

        voc0 += vk00c1x0 * vi00c1;
        voc1 += vk00c1x1 * vi00c1;
        voc2 += vk00c1x2 * vi00c1;
        voc3 += vk00c1x3 * vi00c1;

        const float vk10c1x0 = w[20];
        const float vk10c1x1 = w[21];
        const float vk10c1x2 = w[22];
        const float vk10c1x3 = w[23];

        voc0 += vk10c1x0 * vi10c1;
        voc1 += vk10c1x1 * vi10c1;
        voc2 += vk10c1x2 * vi10c1;
        voc3 += vk10c1x3 * vi10c1;

        const float vk20c1x0 = w[24];
        const float vk20c1x1 = w[25];
        const float vk20c1x2 = w[26];
        const float vk20c1x3 = w[27];

        voc0 += vk20c1x0 * vi20c1;
        voc1 += vk20c1x1 * vi20c1;
        voc2 += vk20c1x2 * vi20c1;
        voc3 += vk20c1x3 * vi20c1;

        const float vk00c2x0 = w[28];
        const float vk00c2x1 = w[29];
        const float vk00c2x2 = w[30];
        const float vk00c2x3 = w[31];

        voc0 += vk00c2x0 * vi00c2;
        voc1 += vk00c2x1 * vi00c2;
        voc2 += vk00c2x2 * vi00c2;
        voc3 += vk00c2x3 * vi00c2;

        const float vk10c2x0 = w[32];
        const float vk10c2x1 = w[33];
        const float vk10c2x2 = w[34];
        const float vk10c2x3 = w[35];

        voc0 += vk10c2x0 * vi10c2;
        voc1 += vk10c2x1 * vi10c2;
        voc2 += vk10c2x2 * vi10c2;
        voc3 += vk10c2x3 * vi10c2;

        const float vk20c2x0 = w[36];
        const float vk20c2x1 = w[37];
        const float vk20c2x2 = w[38];
        const float vk20c2x3 = w[39];

        voc0 += vk20c2x0 * vi20c2;
        voc1 += vk20c2x1 * vi20c2;
        voc2 += vk20c2x2 * vi20c2;
        voc3 += vk20c2x3 * vi20c2;

        const float vk01c0x0 = w[40];
        const float vk01c0x1 = w[41];
        const float vk01c0x2 = w[42];
        const float vk01c0x3 = w[43];

        const float vi01c0 = i0[0];

        voc0 += vk01c0x0 * vi01c0;
        voc1 += vk01c0x1 * vi01c0;
        voc2 += vk01c0x2 * vi01c0;
        voc3 += vk01c0x3 * vi01c0;

        const float vk11c0x0 = w[44];
        const float vk11c0x1 = w[45];
        const float vk11c0x2 = w[46];
        const float vk11c0x3 = w[47];

        const float vi11c0 = i1[0];

        voc0 += vk11c0x0 * vi11c0;
        voc1 += vk11c0x1 * vi11c0;
        voc2 += vk11c0x2 * vi11c0;
        voc3 += vk11c0x3 * vi11c0;

        const float vk21c0x0 = w[48];
        const float vk21c0x1 = w[49];
        const float vk21c0x2 = w[50];
        const float vk21c0x3 = w[51];

        const float vi21c0 = i2[0];

        voc0 += vk21c0x0 * vi21c0;
        voc1 += vk21c0x1 * vi21c0;
        voc2 += vk21c0x2 * vi21c0;
        voc3 += vk21c0x3 * vi21c0;

        const float vk01c1x0 = w[52];
        const float vk01c1x1 = w[53];
        const float vk01c1x2 = w[54];
        const float vk01c1x3 = w[55];

        const float vi01c1 = i0[1];

        voc0 += vk01c1x0 * vi01c1;
        voc1 += vk01c1x1 * vi01c1;
        voc2 += vk01c1x2 * vi01c1;
        voc3 += vk01c1x3 * vi01c1;

        const float vk11c1x0 = w[56];
        const float vk11c1x1 = w[57];
        const float vk11c1x2 = w[58];
        const float vk11c1x3 = w[59];

        const float vi11c1 = i1[1];

        voc0 += vk11c1x0 * vi11c1;
        voc1 += vk11c1x1 * vi11c1;
        voc2 += vk11c1x2 * vi11c1;
        voc3 += vk11c1x3 * vi11c1;

        const float vk21c1x0 = w[60];
        const float vk21c1x1 = w[61];
        const float vk21c1x2 = w[62];
        const float vk21c1x3 = w[63];

        const float vi21c1 = i2[1];

        voc0 += vk21c1x0 * vi21c1;
        voc1 += vk21c1x1 * vi21c1;
        voc2 += vk21c1x2 * vi21c1;
        voc3 += vk21c1x3 * vi21c1;

        const float vk01c2x0 = w[64];
        const float vk01c2x1 = w[65];
        const float vk01c2x2 = w[66];
        const float vk01c2x3 = w[67];

        const float vi01c2 = i0[2];

        voc0 += vk01c2x0 * vi01c2;
        voc1 += vk01c2x1 * vi01c2;
        voc2 += vk01c2x2 * vi01c2;
        voc3 += vk01c2x3 * vi01c2;

        const float vk11c2x0 = w[68];
        const float vk11c2x1 = w[69];
        const float vk11c2x2 = w[70];
        const float vk11c2x3 = w[71];

        const float vi11c2 = i1[2];

        voc0 += vk11c2x0 * vi11c2;
        voc1 += vk11c2x1 * vi11c2;
        voc2 += vk11c2x2 * vi11c2;
        voc3 += vk11c2x3 * vi11c2;

        const float vk21c2x0 = w[72];
        const float vk21c2x1 = w[73];
        const float vk21c2x2 = w[74];
        const float vk21c2x3 = w[75];

        const float vi21c2 = i2[2];

        voc0 += vk21c2x0 * vi21c2;
        voc1 += vk21c2x1 * vi21c2;
        voc2 += vk21c2x2 * vi21c2;
        voc3 += vk21c2x3 * vi21c2;

        voc0 = math_min_f32(voc0, voutput_max);
        voc1 = math_min_f32(voc1, voutput_max);
        voc2 = math_min_f32(voc2, voutput_max);
        voc3 = math_min_f32(voc3, voutput_max);

        voc0 = math_max_f32(voc0, voutput_min);
        voc1 = math_max_f32(voc1, voutput_min);
        voc2 = math_max_f32(voc2, voutput_min);
        voc3 = math_max_f32(voc3, voutput_min);

        *o0c0++ = voc0;
        *o0c1++ = voc1;
        *o0c2++ = voc2;
        *o0c3++ = voc3;
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_decrement);
      i1 = (const float*) ((uintptr_t) i1 - input_width_decrement);
      i2 = (const float*) ((uintptr_t) i2 - input_width_decrement);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next row
    output0 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next row
    i0 = i2;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  }
}

void xnn_f32_dwconv_minmax_ukernel_25p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      const float vi4 = *i4++;
      const float vk4 = w[5];
      vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

      const float vi5 = *i5++;
      const float vk5 = w[6];
      vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);

      const float vi6 = *i6++;
      const float vk6 = w[7];
      vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);

      const float vi7 = *i7++;
      const float vk7 = w[8];
      vacc0p1 = math_muladd_f32(vi7, vk7, vacc0p1);

      const float vi8 = *i8++;
      const float vk8 = w[9];
      vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

      const float vi9 = *i9++;
      const float vk9 = w[10];
      vacc0p1 = math_muladd_f32(vi9, vk9, vacc0p1);

      const float vi10 = *i10++;
      const float vk10 = w[11];
      vacc0p0 = math_muladd_f32(vi10, vk10, vacc0p0);

      const float vi11 = *i11++;
      const float vk11 = w[12];
      vacc0p1 = math_muladd_f32(vi11, vk11, vacc0p1);

      const float vi12 = *i12++;
      const float vk12 = w[13];
      vacc0p0 = math_muladd_f32(vi12, vk12, vacc0p0);

      const float vi13 = *i13++;
      const float vk13 = w[14];
      vacc0p1 = math_muladd_f32(vi13, vk13, vacc0p1);

      const float vi14 = *i14++;
      const float vk14 = w[15];
      vacc0p0 = math_muladd_f32(vi14, vk14, vacc0p0);

      const float vi15 = *i15++;
      const float vk15 = w[16];
      vacc0p1 = math_muladd_f32(vi15, vk15, vacc0p1);

      const float vi16 = *i16++;
      const float vk16 = w[17];
      vacc0p0 = math_muladd_f32(vi16, vk16, vacc0p0);

      const float vi17 = *i17++;
      const float vk17 = w[18];
      vacc0p1 = math_muladd_f32(vi17, vk17, vacc0p1);

      const float vi18 = *i18++;
      const float vk18 = w[19];
      vacc0p0 = math_muladd_f32(vi18, vk18, vacc0p0);

      const float vi19 = *i19++;
      const float vk19 = w[20];
      vacc0p1 = math_muladd_f32(vi19, vk19, vacc0p1);

      const float vi20 = *i20++;
      const float vk20 = w[21];
      vacc0p0 = math_muladd_f32(vi20, vk20, vacc0p0);

      const float vi21 = *i21++;
      const float vk21 = w[22];
      vacc0p1 = math_muladd_f32(vi21, vk21, vacc0p1);

      const float vi22 = *i22++;
      const float vk22 = w[23];
      vacc0p0 = math_muladd_f32(vi22, vk22, vacc0p0);

      const float vi23 = *i23++;
      const float vk23 = w[24];
      vacc0p1 = math_muladd_f32(vi23, vk23, vacc0p1);

      const float vi24 = *i24++;
      const float vk24 = w[25];
      vacc0p0 = math_muladd_f32(vi24, vk24, vacc0p0);

      w += 26;

      vacc0p0 += vacc0p1;

      float vacc0 = math_max_f32(vacc0p0, vmin);
      vacc0 = math_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_25p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      const float vi4 = *i4++;
      const float vk4 = w[5];
      vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

      const float vi5 = *i5++;
      const float vk5 = w[6];
      vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);

      const float vi6 = *i6++;
      const float vk6 = w[7];
      vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);

      const float vi7 = *i7++;
      const float vk7 = w[8];
      vacc0p1 = math_muladd_f32(vi7, vk7, vacc0p1);

      const float vi8 = *i8++;
      const float vk8 = w[9];
      vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

      const float vi9 = *i9++;
      const float vk9 = w[10];
      vacc0p1 = math_muladd_f32(vi9, vk9, vacc0p1);

      const float vi10 = *i10++;
      const float vk10 = w[11];
      vacc0p0 = math_muladd_f32(vi10, vk10, vacc0p0);

      const float vi11 = *i11++;
      const float vk11 = w[12];
      vacc0p1 = math_muladd_f32(vi11, vk11, vacc0p1);

      const float vi12 = *i12++;
      const float vk12 = w[13];
      vacc0p0 = math_muladd_f32(vi12, vk12, vacc0p0);

      const float vi13 = *i13++;
      const float vk13 = w[14];
      vacc0p1 = math_muladd_f32(vi13, vk13, vacc0p1);

      const float vi14 = *i14++;
      const float vk14 = w[15];
      vacc0p0 = math_muladd_f32(vi14, vk14, vacc0p0);

      const float vi15 = *i15++;
      const float vk15 = w[16];
      vacc0p1 = math_muladd_f32(vi15, vk15, vacc0p1);

      const float vi16 = *i16++;
      const float vk16 = w[17];
      vacc0p0 = math_muladd_f32(vi16, vk16, vacc0p0);

      const float vi17 = *i17++;
      const float vk17 = w[18];
      vacc0p1 = math_muladd_f32(vi17, vk17, vacc0p1);

      const float vi18 = *i18++;
      const float vk18 = w[19];
      vacc0p0 = math_muladd_f32(vi18, vk18, vacc0p0);

      const float vi19 = *i19++;
      const float vk19 = w[20];
      vacc0p1 = math_muladd_f32(vi19, vk19, vacc0p1);

      const float vi20 = *i20++;
      const float vk20 = w[21];
      vacc0p0 = math_muladd_f32(vi20, vk20, vacc0p0);

      const float vi21 = *i21++;
      const float vk21 = w[22];
      vacc0p1 = math_muladd_f32(vi21, vk21, vacc0p1);

      const float vi22 = *i22++;
      const float vk22 = w[23];
      vacc0p0 = math_muladd_f32(vi22, vk22, vacc0p0);

      const float vi23 = *i23++;
      const float vk23 = w[24];
      vacc0p1 = math_muladd_f32(vi23, vk23, vacc0p1);

      const float vi24 = *i24++;
      const float vk24 = w[25];
      vacc0p0 = math_muladd_f32(vi24, vk24, vacc0p0);

      w += 26;

      vacc0p0 += vacc0p1;

      *output++ = vacc0p0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      w += 4;

      vacc0p0 += vacc0p1;

      float vacc0 = math_max_f32(vacc0p0, vmin);
      vacc0 = math_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_3p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      w += 4;

      vacc0p0 += vacc0p1;

      *output++ = vacc0p0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      w += 5;

      vacc0p0 += vacc0p1;

      float vacc0 = math_max_f32(vacc0p0, vmin);
      vacc0 = math_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_4p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      w += 5;

      vacc0p0 += vacc0p1;

      *output++ = vacc0p0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      const float vi4 = *i4++;
      const float vk4 = w[5];
      vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

      const float vi5 = *i5++;
      const float vk5 = w[6];
      vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);

      const float vi6 = *i6++;
      const float vk6 = w[7];
      vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);

      const float vi7 = *i7++;
      const float vk7 = w[8];
      vacc0p1 = math_muladd_f32(vi7, vk7, vacc0p1);

      const float vi8 = *i8++;
      const float vk8 = w[9];
      vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

      w += 10;

      vacc0p0 += vacc0p1;

      float vacc0 = math_max_f32(vacc0p0, vmin);
      vacc0 = math_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_9p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      const float vi4 = *i4++;
      const float vk4 = w[5];
      vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

      const float vi5 = *i5++;
      const float vk5 = w[6];
      vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);

      const float vi6 = *i6++;
      const float vk6 = w[7];
      vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);

      const float vi7 = *i7++;
      const float vk7 = w[8];
      vacc0p1 = math_muladd_f32(vi7, vk7, vacc0p1);

      const float vi8 = *i8++;
      const float vk8 = w[9];
      vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

      w += 10;

      vacc0p0 += vacc0p1;

      *output++ = vacc0p0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const float vbias = weights[0];
  const float vk00 = weights[1];
  const float vk01 = weights[2];
  const float vk02 = weights[3];
  const float vk10 = weights[4];
  const float vk11 = weights[5];
  const float vk12 = weights[6];
  const float vk20 = weights[7];
  const float vk21 = weights[8];
  const float vk22 = weights[9];

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
    }

    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;
    float vi3x0 = 0.0f;

    float vi0x1 = *i0++;
    float vi1x1 = *i1++;
    float vi2x1 = *i2++;
    float vi3x1 = *i3++;

    size_t w = input_width;
    for (; w > 1 * sizeof(float); w -= 1 * sizeof(float)) {
      const float vi0x2 = *i0++;
      const float vi1x2 = *i1++;
      const float vi2x2 = *i2++;
      const float vi3x2 = *i3++;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi1x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo1p1 = vi2x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi3x0 * vk20;

      vi0x0 = vi0x1;
      vi1x0 = vi1x1;
      vi2x0 = vi2x1;
      vi3x0 = vi3x1;

      vo0p1 += vi0x1 * vk01;
      vo1p1 += vi1x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi2x1 * vk11;
      vo0p1 += vi2x1 * vk21;
      vo1p1 += vi3x1 * vk21;

      vi0x1 = vi0x2;
      vi1x1 = vi1x2;
      vi2x1 = vi2x2;
      vi3x1 = vi3x2;

      vo0p0 += vi0x2 * vk02;
      vo1p0 += vi1x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo1p1 += vi2x2 * vk12;
      vo0p0 += vi2x2 * vk22;
      vo1p0 += vi3x2 * vk22;

      vo0p0 += vo0p1;
      vo1p0 += vo1p1;

      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);

      *o1++ = vo1;
      *o0++ = vo0;
    }
    // Always process the last pixel separately to account for right edge.
    assert(w == 1 * sizeof(float));
    {
      float vo0p0 = vbias + vi0x0 * vk00;
      float vo1p0 = vbias + vi1x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo1p1 = vi2x0 * vk10;
      vo0p0 += vi2x0 * vk20;
      vo1p0 += vi3x0 * vk20;

      vo0p1 += vi0x1 * vk01;
      vo1p1 += vi1x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo1p0 += vi2x1 * vk11;
      vo0p1 += vi2x1 * vk21;
      vo1p1 += vi3x1 * vk21;

      vo0p0 += vo0p1;
      vo1p0 += vo1p1;

      float vo0 = math_max_f32(vo0p0, vmin);
      float vo1 = math_max_f32(vo1p0, vmin);

      vo0 = math_min_f32(vo0, vmax);
      vo1 = math_min_f32(vo1, vmax);

      *o1++ = vo1;
      *o0++ = vo0;
    }

    i0 = (const float*) ((uintptr_t) i2 - input_width);
    i1 = (const float*) ((uintptr_t) i3 - input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + input_width);

    output_height = doz(output_height, 2);
  } while (output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 0);
  assert(padding_top <= 1);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const float vbias = weights[0];
  const float vk00 = weights[1];
  const float vk01 = weights[2];
  const float vk02 = weights[3];
  const float vk10 = weights[4];
  const float vk11 = weights[5];
  const float vk12 = weights[6];
  const float vk20 = weights[7];
  const float vk21 = weights[8];
  const float vk22 = weights[9];


  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);

  float* o0 = output;

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }

    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;

    size_t w = input_width;
    for (; w >= 2 * sizeof(float); w -= 2 * sizeof(float)) {
      const float vi0x1 = i0[0];
      const float vi1x1 = i1[0];
      const float vi2x1 = i2[0];

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      vo0p0 += vi2x0 * vk20;

      const float vi0x2 = i0[1];
      i0 += 2;
      const float vi1x2 = i1[1];
      i1 += 2;
      const float vi2x2 = i2[1];
      i2 += 2;

      vo0p1 += vi0x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo0p1 += vi2x1 * vk21;

      vi0x0 = vi0x2;
      vi1x0 = vi1x2;
      vi2x0 = vi2x2;

      vo0p0 += vi0x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo0p0 += vi2x2 * vk22;

      vo0p0 += vo0p1;

      float vo0 = math_max_f32(vo0p0, vmin);

      vo0 = math_min_f32(vo0, vmax);

      *o0++ = vo0;
    }
    // Potentially process the last pixel.
    assert(w <= 1 * sizeof(float));
    if (w != 0) {
      const float vi0x1 = *i0++;
      const float vi1x1 = *i1++;
      const float vi2x1 = *i2++;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      vo0p0 += vi2x0 * vk20;

      vo0p1 += vi0x1 * vk01;
      vo0p0 += vi1x1 * vk11;
      vo0p1 += vi2x1 * vk21;

      vo0p0 += vo0p1;

      float vo0 = math_max_f32(vo0p0, vmin);

      vo0 = math_min_f32(vo0, vmax);

      *o0++ = vo0;
    }

    i0 = (const float*) ((uintptr_t) i1);
    i1 = (const float*) ((uintptr_t) i2);
    i2 = (const float*) ((uintptr_t) i1 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 2);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const float vbias = weights[0];
  const float vk00 = weights[1];
  const float vk01 = weights[2];
  const float vk02 = weights[3];
  const float vk03 = weights[4];
  const float vk04 = weights[5];
  const float vk10 = weights[6];
  const float vk11 = weights[7];
  const float vk12 = weights[8];
  const float vk13 = weights[9];
  const float vk14 = weights[10];
  const float vk20 = weights[11];
  const float vk21 = weights[12];
  const float vk22 = weights[13];
  const float vk23 = weights[14];
  const float vk24 = weights[15];
  const float vk30 = weights[16];
  const float vk31 = weights[17];
  const float vk32 = weights[18];
  const float vk33 = weights[19];
  const float vk34 = weights[20];
  const float vk40 = weights[21];
  const float vk41 = weights[22];
  const float vk42 = weights[23];
  const float vk43 = weights[24];
  const float vk44 = weights[25];

  const float* i0 = zero;
  const float* i1 = zero;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

  float* o0 = output;

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
    }

    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;
    float vi3x0 = 0.0f;
    float vi4x0 = 0.0f;

    float vi0x1 = 0.0f;
    float vi1x1 = 0.0f;
    float vi2x1 = 0.0f;
    float vi3x1 = 0.0f;
    float vi4x1 = 0.0f;

    float vi0x2 = *i0++;
    float vi1x2 = *i1++;
    float vi2x2 = *i2++;
    float vi3x2 = *i3++;
    float vi4x2 = *i4++;

    size_t w = input_width;
    if (w > 1 * sizeof(float)) {
      float vi0x3 = *i0++;
      float vi1x3 = *i1++;
      float vi2x3 = *i2++;
      float vi3x3 = *i3++;
      float vi4x3 = *i4++;

      for (; w > 2 * sizeof(float); w -= 1 * sizeof(float)) {
        const float vi0x4 = *i0++;
        const float vi1x4 = *i1++;
        const float vi2x4 = *i2++;
        const float vi3x4 = *i3++;
        const float vi4x4 = *i4++;

        float vo0p0 = vbias + vi0x0 * vk00;
        float vo0p1 = vi1x0 * vk10;
        float vo0p2 = vi2x0 * vk20;
        float vo0p3 = vi3x0 * vk30;
        float vo0p4 = vi4x0 * vk40;

        vi0x0 = vi0x1;
        vi1x0 = vi1x1;
        vi2x0 = vi2x1;
        vi3x0 = vi3x1;
        vi4x0 = vi4x1;

        vo0p0 += vi0x1 * vk01;
        vo0p1 += vi1x1 * vk11;
        vo0p2 += vi2x1 * vk21;
        vo0p3 += vi3x1 * vk31;
        vo0p4 += vi4x1 * vk41;

        vi0x1 = vi0x2;
        vi1x1 = vi1x2;
        vi2x1 = vi2x2;
        vi3x1 = vi3x2;
        vi4x1 = vi4x2;

        vo0p0 += vi0x2 * vk02;
        vo0p1 += vi1x2 * vk12;
        vo0p2 += vi2x2 * vk22;
        vo0p3 += vi3x2 * vk32;
        vo0p4 += vi4x2 * vk42;

        vi0x2 = vi0x3;
        vi1x2 = vi1x3;
        vi2x2 = vi2x3;
        vi3x2 = vi3x3;
        vi4x2 = vi4x3;

        vo0p0 += vi0x3 * vk03;
        vo0p1 += vi1x3 * vk13;
        vo0p2 += vi2x3 * vk23;
        vo0p3 += vi3x3 * vk33;
        vo0p4 += vi4x3 * vk43;

        vi0x3 = vi0x4;
        vi1x3 = vi1x4;
        vi2x3 = vi2x4;
        vi3x3 = vi3x4;
        vi4x3 = vi4x4;

        vo0p0 += vi0x4 * vk04;
        vo0p1 += vi1x4 * vk14;
        vo0p2 += vi2x4 * vk24;
        vo0p3 += vi3x4 * vk34;
        vo0p4 += vi4x4 * vk44;

        vo0p0 += vo0p1;
        vo0p2 += vo0p3;
        vo0p0 += vo0p2;
        vo0p0 += vo0p4;

        float vo0 = math_max_f32(vo0p0, vmin);

        vo0 = math_min_f32(vo0, vmax);

        *o0++ = vo0;
      }
      assert(w == 2 * sizeof(float));
      {
        float vo0p0 = vbias + vi0x0 * vk00;
        float vo0p1 = vi1x0 * vk10;
        float vo0p2 = vi2x0 * vk20;
        float vo0p3 = vi3x0 * vk30;
        float vo0p4 = vi4x0 * vk40;

        vi0x0 = vi0x1;
        vi1x0 = vi1x1;
        vi2x0 = vi2x1;
        vi3x0 = vi3x1;
        vi4x0 = vi4x1;

        vo0p0 += vi0x1 * vk01;
        vo0p1 += vi1x1 * vk11;
        vo0p2 += vi2x1 * vk21;
        vo0p3 += vi3x1 * vk31;
        vo0p4 += vi4x1 * vk41;

        vi0x1 = vi0x2;
        vi1x1 = vi1x2;
        vi2x1 = vi2x2;
        vi3x1 = vi3x2;
        vi4x1 = vi4x2;

        vo0p0 += vi0x2 * vk02;
        vo0p1 += vi1x2 * vk12;
        vo0p2 += vi2x2 * vk22;
        vo0p3 += vi3x2 * vk32;
        vo0p4 += vi4x2 * vk42;

        vi0x2 = vi0x3;
        vi1x2 = vi1x3;
        vi2x2 = vi2x3;
        vi3x2 = vi3x3;
        vi4x2 = vi4x3;

        vo0p0 += vi0x3 * vk03;
        vo0p1 += vi1x3 * vk13;
        vo0p2 += vi2x3 * vk23;
        vo0p3 += vi3x3 * vk33;
        vo0p4 += vi4x3 * vk43;

        vo0p0 += vo0p1;
        vo0p2 += vo0p3;
        vo0p0 += vo0p2;
        vo0p0 += vo0p4;

        float vo0 = math_max_f32(vo0p0, vmin);

        vo0 = math_min_f32(vo0, vmax);

        *o0++ = vo0;
      }
      w -= 1 * sizeof(float);
    }
    assert(w == 1 * sizeof(float));
    {
      float vo0p0 = vbias + vi0x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo0p2 = vi2x0 * vk20;
      float vo0p3 = vi3x0 * vk30;
      float vo0p4 = vi4x0 * vk40;

      vo0p0 += vi0x1 * vk01;
      vo0p1 += vi1x1 * vk11;
      vo0p2 += vi2x1 * vk21;
      vo0p3 += vi3x1 * vk31;
      vo0p4 += vi4x1 * vk41;

      vo0p0 += vi0x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo0p2 += vi2x2 * vk22;
      vo0p3 += vi3x2 * vk32;
      vo0p4 += vi4x2 * vk42;

      vo0p0 += vo0p1;
      vo0p2 += vo0p3;
      vo0p0 += vo0p2;
      vo0p0 += vo0p4;

      float vo0 = math_max_f32(vo0p0, vmin);

      vo0 = math_min_f32(vo0, vmax);

      *o0++ = vo0;
    }

    i0 = (const float*) ((uintptr_t) i1 - input_width);
    i1 = (const float*) ((uintptr_t) i2 - input_width);


  } while (--output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const float vmax = params->scalar.max;
  const float vmin = params->scalar.min;

  const float vbias = weights[0];
  const float vk00 = weights[1];
  const float vk01 = weights[2];
  const float vk02 = weights[3];
  const float vk03 = weights[4];
  const float vk04 = weights[5];
  const float vk10 = weights[6];
  const float vk11 = weights[7];
  const float vk12 = weights[8];
  const float vk13 = weights[9];
  const float vk14 = weights[10];
  const float vk20 = weights[11];
  const float vk21 = weights[12];
  const float vk22 = weights[13];
  const float vk23 = weights[14];
  const float vk24 = weights[15];
  const float vk30 = weights[16];
  const float vk31 = weights[17];
  const float vk32 = weights[18];
  const float vk33 = weights[19];
  const float vk34 = weights[20];
  const float vk40 = weights[21];
  const float vk41 = weights[22];
  const float vk42 = weights[23];
  const float vk43 = weights[24];
  const float vk44 = weights[25];

  const uint32_t padding_top_less_1 = padding_top - 1;

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);


  float* o0 = output;

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
    }

    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;
    float vi3x0 = 0.0f;
    float vi4x0 = 0.0f;

    float vi0x1 = 0.0f;
    float vi1x1 = 0.0f;
    float vi2x1 = 0.0f;
    float vi3x1 = 0.0f;
    float vi4x1 = 0.0f;

    float vi0x2 = *i0++;
    float vi1x2 = *i1++;
    float vi2x2 = *i2++;
    float vi3x2 = *i3++;
    float vi4x2 = *i4++;

    size_t w = input_width;
    for (; w > 2 * sizeof(float); w -= 2 * sizeof(float)) {
      const float vi0x3 = i0[0];
      const float vi1x3 = i1[0];
      const float vi2x3 = i2[0];
      const float vi3x3 = i3[0];
      const float vi4x3 = i4[0];

      const float vi0x4 = i0[1];
      i0 += 2;
      const float vi1x4 = i1[1];
      i1 += 2;
      const float vi2x4 = i2[1];
      i2 += 2;
      const float vi3x4 = i3[1];
      i3 += 2;
      const float vi4x4 = i4[1];
      i4 += 2;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo0p2 = vi2x0 * vk20;
      float vo0p3 = vi3x0 * vk30;
      float vo0p4 = vi4x0 * vk40;

      vi0x0 = vi0x2;
      vi1x0 = vi1x2;
      vi2x0 = vi2x2;
      vi3x0 = vi3x2;
      vi4x0 = vi4x2;

      vo0p0 += vi0x1 * vk01;
      vo0p1 += vi1x1 * vk11;
      vo0p2 += vi2x1 * vk21;
      vo0p3 += vi3x1 * vk31;
      vo0p4 += vi4x1 * vk41;

      vi0x1 = vi0x3;
      vi1x1 = vi1x3;
      vi2x1 = vi2x3;
      vi3x1 = vi3x3;
      vi4x1 = vi4x3;

      vo0p0 += vi0x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo0p2 += vi2x2 * vk22;
      vo0p3 += vi3x2 * vk32;
      vo0p4 += vi4x2 * vk42;

      vi0x2 = vi0x4;
      vi1x2 = vi1x4;
      vi2x2 = vi2x4;
      vi3x2 = vi3x4;
      vi4x2 = vi4x4;

      vo0p0 += vi0x3 * vk03;
      vo0p1 += vi1x3 * vk13;
      vo0p2 += vi2x3 * vk23;
      vo0p3 += vi3x3 * vk33;
      vo0p4 += vi4x3 * vk43;

      vo0p0 += vi0x4 * vk04;
      vo0p1 += vi1x4 * vk14;
      vo0p2 += vi2x4 * vk24;
      vo0p3 += vi3x4 * vk34;
      vo0p4 += vi4x4 * vk44;

      vo0p0 += vo0p1;
      vo0p2 += vo0p3;
      vo0p0 += vo0p2;
      vo0p0 += vo0p4;

      float vo0 = math_max_f32(vo0p0, vmin);

      vo0 = math_min_f32(vo0, vmax);

      *o0++ = vo0;
    }
    if XNN_LIKELY(w == 2 * sizeof(float)) {
      const float vi0x3 = *i0++;
      const float vi1x3 = *i1++;
      const float vi2x3 = *i2++;
      const float vi3x3 = *i3++;
      const float vi4x3 = *i4++;

      float vo0p0 = vbias + vi0x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo0p2 = vi2x0 * vk20;
      float vo0p3 = vi3x0 * vk30;
      float vo0p4 = vi4x0 * vk40;

      vo0p0 += vi0x1 * vk01;
      vo0p1 += vi1x1 * vk11;
      vo0p2 += vi2x1 * vk21;
      vo0p3 += vi3x1 * vk31;
      vo0p4 += vi4x1 * vk41;

      vo0p0 += vi0x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo0p2 += vi2x2 * vk22;
      vo0p3 += vi3x2 * vk32;
      vo0p4 += vi4x2 * vk42;

      vo0p0 += vi0x3 * vk03;
      vo0p1 += vi1x3 * vk13;
      vo0p2 += vi2x3 * vk23;
      vo0p3 += vi3x3 * vk33;
      vo0p4 += vi4x3 * vk43;

      vo0p0 += vo0p1;
      vo0p2 += vo0p3;
      vo0p0 += vo0p2;
      vo0p0 += vo0p4;

      float vo0 = math_max_f32(vo0p0, vmin);

      vo0 = math_min_f32(vo0, vmax);

      *o0++ = vo0;
    } else {
      float vo0p0 = vbias + vi0x0 * vk00;
      float vo0p1 = vi1x0 * vk10;
      float vo0p2 = vi2x0 * vk20;
      float vo0p3 = vi3x0 * vk30;
      float vo0p4 = vi4x0 * vk40;

      vo0p0 += vi0x1 * vk01;
      vo0p1 += vi1x1 * vk11;
      vo0p2 += vi2x1 * vk21;
      vo0p3 += vi3x1 * vk31;
      vo0p4 += vi4x1 * vk41;

      vo0p0 += vi0x2 * vk02;
      vo0p1 += vi1x2 * vk12;
      vo0p2 += vi2x2 * vk22;
      vo0p3 += vi3x2 * vk32;
      vo0p4 += vi4x2 * vk42;

      vo0p0 += vo0p1;
      vo0p2 += vo0p3;
      vo0p0 += vo0p2;
      vo0p0 += vo0p4;

      float vo0 = math_max_f32(vo0p0, vmin);

      vo0 = math_min_f32(vo0, vmax);

      *o0++ = vo0;
    }

    i0 = (const float*) ((uintptr_t) i2 - input_width);
    i1 = (const float*) ((uintptr_t) i2);
    i2 = (const float*) ((uintptr_t) i3);
    i3 = (const float*) ((uintptr_t) i4);
    i4 = (const float*) ((uintptr_t) i3 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}

void xnn_f32_f16_vcvt_ukernel__scalar_fabsf_x2(
    size_t batch,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vscale_to_inf = params->scalar_fabsf.scale_to_inf;
  const uint32_t vexp_bias = params->scalar_fabsf.exp_bias;
  const float vscale_to_zero = params->scalar_fabsf.scale_to_zero;
  const uint32_t vexpw_max = params->scalar_fabsf.expw_max;
  const uint32_t vbias_min = params->scalar_fabsf.bias_min;
  const uint16_t vexph_mask = params->scalar_fabsf.exph_mask;
  const uint16_t vmanth_mask = params->scalar_fabsf.manth_mask;
  const uint16_t vnanh = params->scalar_fabsf.nanh;

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    const float vabsx0 = fabsf(vx0);
    const float vabsx1 = fabsf(vx1);
    uint32_t vsignw0 = float_as_uint32(vx0);
    uint32_t vsignw1 = float_as_uint32(vx1);

    const uint32_t vnonsignw0 = float_as_uint32(vabsx0);
    const uint32_t vnonsignw1 = float_as_uint32(vabsx1);
    float vf0 = vabsx0 * vscale_to_inf;
    float vf1 = vabsx1 * vscale_to_inf;

    uint32_t vbias0 = vnonsignw0 + vexp_bias;
    uint32_t vbias1 = vnonsignw1 + vexp_bias;
    vsignw0 ^= vnonsignw0;
    vsignw1 ^= vnonsignw1;

    vf0 *= vscale_to_zero;
    vf1 *= vscale_to_zero;
    vbias0 &= vexpw_max;
    vbias1 &= vexpw_max;

    vbias0 = math_max_u32(vbias0, vbias_min);
    vbias1 = math_max_u32(vbias1, vbias_min);

    vf0 += uint32_as_float(vbias0);
    vf1 += uint32_as_float(vbias1);

    const uint32_t vbits0 = float_as_uint32(vf0);
    const uint32_t vbits1 = float_as_uint32(vf1);

    const uint16_t vexph0 = (uint16_t) (vbits0 >> 13) & vexph_mask;
    const uint16_t vexph1 = (uint16_t) (vbits1 >> 13) & vexph_mask;
    const uint16_t vmanth0 = (uint16_t) vbits0 & vmanth_mask;
    const uint16_t vmanth1 = (uint16_t) vbits1 & vmanth_mask;
    const uint16_t vsignh0 = (uint16_t) (vsignw0 >> 16);
    const uint16_t vsignh1 = (uint16_t) (vsignw1 >> 16);

    uint16_t vh0 = vexph0 + vmanth0;
    uint16_t vh1 = vexph1 + vmanth1;
    if XNN_UNPREDICTABLE(vnonsignw0 > vexpw_max) {
      vh0 = vnanh;
    }
    if XNN_UNPREDICTABLE(vnonsignw1 > vexpw_max) {
      vh1 = vnanh;
    }
    vh0 |= vsignh0;
    vh1 |= vsignh1;

    o[0] = vh0;
    o[1] = vh1;
    o += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;

    const float vabsx = fabsf(vx);
    uint32_t vsignw = float_as_uint32(vx);

    const uint32_t vnonsignw = float_as_uint32(vabsx);
    float vf = vabsx * vscale_to_inf;

    uint32_t vbias = vnonsignw + vexp_bias;
    vsignw ^= vnonsignw;

    vf *= vscale_to_zero;
    vbias &= vexpw_max;

    vbias = math_max_u32(vbias, vbias_min);

    vf += uint32_as_float(vbias);

    const uint32_t vbits = float_as_uint32(vf);

    const uint16_t vexph = (uint16_t) (vbits >> 13) & vexph_mask;
    const uint16_t vmanth = (uint16_t) vbits & vmanth_mask;
    const uint16_t vsignh = (uint16_t) (vsignw >> 16);

    uint16_t vh = vexph + vmanth;
    if XNN_UNPREDICTABLE(vnonsignw > vexpw_max) {
      vh = vnanh;
    }
    vh |= vsignh;

    *o = vh;
  }
}

void xnn_f32_gavgpool_cw_ukernel__scalar_x1(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const float* i0 = input;

  const float vmultiplier = params->scalar.multiplier;
  const float voutput_max = params->scalar.output_max;
  const float voutput_min = params->scalar.output_min;

  while (channels != 0) {
    float vsum0 = 0.f;
    float vsum1 = 0.f;
    float vsum2 = 0.f;
    float vsum3 = 0.f;
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      vsum0 += i0[0];
      vsum1 += i0[1];
      vsum2 += i0[2];
      vsum3 += i0[3];

      i0 += 4;
      n -= 4 * sizeof(float);
    }

    while (n != 0) {
      vsum0 += *i0++;
      n -= sizeof(float);
    }

    float vout = ( (vsum0 + vsum1) + (vsum2 + vsum3) ) * vmultiplier;

    vout = math_min_f32(vout, voutput_max);
    vout = math_max_f32(vout, voutput_min);

    *output++ = vout;
    channels -= 1;
  }
}

void xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - channels * sizeof(float);

  float* b = buffer;
  size_t c = channels;
  do {
    const float vi0 = *i0++;
    const float vi1 = *i1++;
    const float vi2 = *i2++;
    const float vi3 = *i3++;
    const float vi4 = *i4++;
    const float vi5 = *i5++;
    const float vi6 = *i6++;

    const float vsum01 = vi0 + vi1;
    const float vsum23 = vi2 + vi3;
    const float vsum45 = vi4 + vi5;

    const float vsum016 = vsum01 + vi6;
    const float vsum2345 = vsum23 + vsum45;

    const float vsum = vsum016 + vsum2345;

    *b++ = vsum;
  } while (--c != 0);
  for (rows -= 7; rows > 7; rows -= 7) {
    b = buffer;

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    i4 = (const float*) ((uintptr_t) i4 + input_increment);
    i5 = (const float*) ((uintptr_t) i5 + input_increment);
    i6 = (const float*) ((uintptr_t) i6 + input_increment);

    size_t c = channels;
    do {
      const float vi0 = *i0++;
      const float vi1 = *i1++;
      const float vi2 = *i2++;
      const float vi3 = *i3++;
      const float vi4 = *i4++;
      const float vi5 = *i5++;
      const float vi6 = *i6++;
      const float vacc = *b;

      const float vsum01 = vi0 + vi1;
      const float vsum23 = vi2 + vi3;
      const float vsum45 = vi4 + vi5;
      const float vsum6a = vi6 + vacc;

      const float vsum0123 = vsum01 + vsum23;
      const float vsum456a = vsum45 + vsum6a;

      const float vsum = vsum0123 + vsum456a;

      *b++ = vsum;
    } while (--c != 0);
  }

  i0 = (const float*) ((uintptr_t) i0 + input_increment);
  i1 = (const float*) ((uintptr_t) i1 + input_increment);
  if (rows < 2) {
    i1 = zero;
  }
  i2 = (const float*) ((uintptr_t) i2 + input_increment);
  if (rows <= 2) {
    i2 = zero;
  }
  i3 = (const float*) ((uintptr_t) i3 + input_increment);
  if (rows < 4) {
    i3 = zero;
  }
  i4 = (const float*) ((uintptr_t) i4 + input_increment);
  if (rows <= 4) {
    i4 = zero;
  }
  i5 = (const float*) ((uintptr_t) i5 + input_increment);
  if (rows < 6) {
    i5 = zero;
  }
  i6 = (const float*) ((uintptr_t) i6 + input_increment);
  if (rows <= 6) {
    i6 = zero;
  }
  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  b = buffer;
  do {
    const float vi0 = *i0++;
    const float vi1 = *i1++;
    const float vi2 = *i2++;
    const float vi3 = *i3++;
    const float vi4 = *i4++;
    const float vi5 = *i5++;
    const float vi6 = *i6++;
    const float vacc = *b++;

    const float vsum01 = vi0 + vi1;
    const float vsum23 = vi2 + vi3;
    const float vsum45 = vi4 + vi5;
    const float vsum6a = vi6 + vacc;

    const float vsum0123 = vsum01 + vsum23;
    const float vsum456a = vsum45 + vsum6a;

    const float vsum = vsum0123 + vsum456a;

    float vout = vsum * vscale;
    vout = math_max_f32(vout, vmin);
    vout = math_min_f32(vout, vmax);

    *output++ = vout;
  } while (--channels != 0);
}

void xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  if (rows < 2) {
    i1 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  if (rows <= 2) {
    i2 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  if (rows < 4) {
    i3 = zero;
  }
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  if (rows <= 4) {
    i4 = zero;
  }
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  if (rows < 6) {
    i5 = zero;
  }
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  if (rows <= 6) {
    i6 = zero;
  }

  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float vi0 = *i0++;
    const float vi1 = *i1++;
    const float vi2 = *i2++;
    const float vi3 = *i3++;
    const float vi4 = *i4++;
    const float vi5 = *i5++;
    const float vi6 = *i6++;

    const float vsum01 = vi0 + vi1;
    const float vsum23 = vi2 + vi3;
    const float vsum45 = vi4 + vi5;

    const float vsum016 = vsum01 + vi6;
    const float vsum2345 = vsum23 + vsum45;

    const float vsum = vsum016 + vsum2345;

    float vout = vsum * vscale;
    vout = math_max_f32(vout, vmin);
    vout = math_min_f32(vout, vmax);

    *output++ = vout;
  } while (--channels != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t k = kc;
    do {
      const float va0 = *a0++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = math_max_f32(vacc00, vmin);
    vacc01 = math_max_f32(vacc01, vmin);
    vacc02 = math_max_f32(vacc02, vmin);
    vacc03 = math_max_f32(vacc03, vmin);

    vacc00 = math_min_f32(vacc00, vmax);
    vacc01 = math_min_f32(vacc01, vmax);
    vacc02 = math_min_f32(vacc02, vmax);
    vacc03 = math_min_f32(vacc03, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t k = kc;
    do {
      const float va0 = *a0++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = math_max_f32(vacc00, 0.0f);
    vacc01 = math_max_f32(vacc01, 0.0f);
    vacc02 = math_max_f32(vacc02, 0.0f);
    vacc03 = math_max_f32(vacc03, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t k = kc;
    do {
      const float va0 = *a0++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);

      k -= sizeof(float);
    } while (k != 0);


    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    w += 2;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc30 = vacc00;
    float vacc31 = vacc01;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      w += 2;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = math_max_f32(vacc00, vmin);
    vacc01 = math_max_f32(vacc01, vmin);
    vacc10 = math_max_f32(vacc10, vmin);
    vacc11 = math_max_f32(vacc11, vmin);
    vacc20 = math_max_f32(vacc20, vmin);
    vacc21 = math_max_f32(vacc21, vmin);
    vacc30 = math_max_f32(vacc30, vmin);
    vacc31 = math_max_f32(vacc31, vmin);

    vacc00 = math_min_f32(vacc00, vmax);
    vacc01 = math_min_f32(vacc01, vmax);
    vacc10 = math_min_f32(vacc10, vmax);
    vacc11 = math_min_f32(vacc11, vmax);
    vacc20 = math_min_f32(vacc20, vmax);
    vacc21 = math_min_f32(vacc21, vmax);
    vacc30 = math_min_f32(vacc30, vmax);
    vacc31 = math_min_f32(vacc31, vmax);

    if XNN_LIKELY(nc >= 2) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const void*) ((uintptr_t) a3 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 2;
    } else {
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    w += 2;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc30 = vacc00;
    float vacc31 = vacc01;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      w += 2;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);

      k -= sizeof(float);
    } while (k != 0);


    if XNN_LIKELY(nc >= 2) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const void*) ((uintptr_t) a3 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 2;
    } else {
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc22 = math_muladd_f32(va2, vb2, vacc22);
      vacc23 = math_muladd_f32(va2, vb3, vacc23);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);
      vacc32 = math_muladd_f32(va3, vb2, vacc32);
      vacc33 = math_muladd_f32(va3, vb3, vacc33);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = math_max_f32(vacc00, vmin);
    vacc01 = math_max_f32(vacc01, vmin);
    vacc02 = math_max_f32(vacc02, vmin);
    vacc03 = math_max_f32(vacc03, vmin);
    vacc10 = math_max_f32(vacc10, vmin);
    vacc11 = math_max_f32(vacc11, vmin);
    vacc12 = math_max_f32(vacc12, vmin);
    vacc13 = math_max_f32(vacc13, vmin);
    vacc20 = math_max_f32(vacc20, vmin);
    vacc21 = math_max_f32(vacc21, vmin);
    vacc22 = math_max_f32(vacc22, vmin);
    vacc23 = math_max_f32(vacc23, vmin);
    vacc30 = math_max_f32(vacc30, vmin);
    vacc31 = math_max_f32(vacc31, vmin);
    vacc32 = math_max_f32(vacc32, vmin);
    vacc33 = math_max_f32(vacc33, vmin);

    vacc00 = math_min_f32(vacc00, vmax);
    vacc01 = math_min_f32(vacc01, vmax);
    vacc02 = math_min_f32(vacc02, vmax);
    vacc03 = math_min_f32(vacc03, vmax);
    vacc10 = math_min_f32(vacc10, vmax);
    vacc11 = math_min_f32(vacc11, vmax);
    vacc12 = math_min_f32(vacc12, vmax);
    vacc13 = math_min_f32(vacc13, vmax);
    vacc20 = math_min_f32(vacc20, vmax);
    vacc21 = math_min_f32(vacc21, vmax);
    vacc22 = math_min_f32(vacc22, vmax);
    vacc23 = math_min_f32(vacc23, vmax);
    vacc30 = math_min_f32(vacc30, vmax);
    vacc31 = math_min_f32(vacc31, vmax);
    vacc32 = math_min_f32(vacc32, vmax);
    vacc33 = math_min_f32(vacc33, vmax);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const void*) ((uintptr_t) a3 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc22 = math_muladd_f32(va2, vb2, vacc22);
      vacc23 = math_muladd_f32(va2, vb3, vacc23);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);
      vacc32 = math_muladd_f32(va3, vb2, vacc32);
      vacc33 = math_muladd_f32(va3, vb3, vacc33);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = math_max_f32(vacc00, 0.0f);
    vacc01 = math_max_f32(vacc01, 0.0f);
    vacc02 = math_max_f32(vacc02, 0.0f);
    vacc03 = math_max_f32(vacc03, 0.0f);
    vacc10 = math_max_f32(vacc10, 0.0f);
    vacc11 = math_max_f32(vacc11, 0.0f);
    vacc12 = math_max_f32(vacc12, 0.0f);
    vacc13 = math_max_f32(vacc13, 0.0f);
    vacc20 = math_max_f32(vacc20, 0.0f);
    vacc21 = math_max_f32(vacc21, 0.0f);
    vacc22 = math_max_f32(vacc22, 0.0f);
    vacc23 = math_max_f32(vacc23, 0.0f);
    vacc30 = math_max_f32(vacc30, 0.0f);
    vacc31 = math_max_f32(vacc31, 0.0f);
    vacc32 = math_max_f32(vacc32, 0.0f);
    vacc33 = math_max_f32(vacc33, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const void*) ((uintptr_t) a3 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc22 = math_muladd_f32(va2, vb2, vacc22);
      vacc23 = math_muladd_f32(va2, vb3, vacc23);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);
      vacc32 = math_muladd_f32(va3, vb2, vacc32);
      vacc33 = math_muladd_f32(va3, vb3, vacc33);

      k -= sizeof(float);
    } while (k != 0);


    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const void*) ((uintptr_t) a3 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_ibilinear_chw_ukernel__scalar_p4(
    size_t output_pixels,
    size_t channels,
    const float**restrict input,
    size_t input_offset,
    const float*restrict weights,
    float*restrict output,
    size_t input_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  size_t c = channels;
  do {
    const float** i = input;
    const float* w = weights;

    size_t p = output_pixels;
    for (; p >= 4; p -= 4) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      i += 4 * 2;

      const float valphah0 = w[0];
      const float valphav0 = w[1];
      const float valphah1 = w[2];
      const float valphav1 = w[3];
      const float valphah2 = w[4];
      const float valphav2 = w[5];
      const float valphah3 = w[6];
      const float valphav3 = w[7];
      w += 4 * 2;

      const float vtl0 = itl0[0];
      const float vtr0 = itl0[1];
      const float vbl0 = ibl0[0];
      const float vbr0 = ibl0[1];
      const float vtl1 = itl1[0];
      const float vtr1 = itl1[1];
      const float vbl1 = ibl1[0];
      const float vbr1 = ibl1[1];
      const float vtl2 = itl2[0];
      const float vtr2 = itl2[1];
      const float vbl2 = ibl2[0];
      const float vbr2 = ibl2[1];
      const float vtl3 = itl3[0];
      const float vtr3 = itl3[1];
      const float vbl3 = ibl3[0];
      const float vbr3 = ibl3[1];

      const float vtd0 = vtr0 - vtl0;
      const float vbd0 = vbr0 - vbl0;
      const float vtd1 = vtr1 - vtl1;
      const float vbd1 = vbr1 - vbl1;
      const float vtd2 = vtr2 - vtl2;
      const float vbd2 = vbr2 - vbl2;
      const float vtd3 = vtr3 - vtl3;
      const float vbd3 = vbr3 - vbl3;

      const float vt0 = vtl0 + vtd0 * valphah0;
      const float vb0 = vbl0 + vbd0 * valphah0;
      const float vt1 = vtl1 + vtd1 * valphah1;
      const float vb1 = vbl1 + vbd1 * valphah1;
      const float vt2 = vtl2 + vtd2 * valphah2;
      const float vb2 = vbl2 + vbd2 * valphah2;
      const float vt3 = vtl3 + vtd3 * valphah3;
      const float vb3 = vbl3 + vbd3 * valphah3;

      const float vd0 = vb0 - vt0;
      const float vd1 = vb1 - vt1;
      const float vd2 = vb2 - vt2;
      const float vd3 = vb3 - vt3;

      const float vo0 = vt0 + vd0 * valphav0;
      const float vo1 = vt1 + vd1 * valphav1;
      const float vo2 = vt2 + vd2 * valphav2;
      const float vo3 = vt3 + vd3 * valphav3;

      output[0] = vo0;
      output[1] = vo1;
      output[2] = vo2;
      output[3] = vo3;
      output += 4;
    }

    for (; p >= 1; p -= 1) {
      const float* itl = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl = (const float*) ((uintptr_t) i[1] + input_offset);
      i += 2;

      const float valphah = w[0];
      const float valphav = w[1];
      w += 2;

      const float vtl = itl[0];
      const float vtr = itl[1];
      const float vbl = ibl[0];
      const float vbr = ibl[1];

      const float vtd = vtr - vtl;
      const float vbd = vbr - vbl;

      const float vt = vtl + vtd * valphah;
      const float vb = vbl + vbd * valphah;

      const float vd = vb - vt;

      const float vo = vt + vd * valphav;

      *output++ = vo;
    }

    input_offset += input_increment;

    c--;
  } while (c != 0);
}

void xnn_f32_ibilinear_ukernel__scalar_c2(
    size_t output_pixels,
    size_t channels,
    const float**restrict input,
    size_t input_offset,
    const float*restrict weights,
    float*restrict output,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  do {
    const float* i0 = (const float*) ((uintptr_t) input[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) input[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) input[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const float valphah = weights[0];
    const float valphav = weights[1];
    weights += 2;

    size_t c = channels;
    for (; c >= 2 * sizeof(float); c -= 2 * sizeof(float)) {
      const float vtl0 = i0[0];
      const float vtr0 = i1[0];
      const float vbl0 = i2[0];
      const float vbr0 = i3[0];
      const float vtl1 = i0[1];
      const float vtr1 = i1[1];
      const float vbl1 = i2[1];
      const float vbr1 = i3[1];
      i0 += 2;
      i1 += 2;
      i2 += 2;
      i3 += 2;

      const float vtd0 = vtr0 - vtl0;
      const float vbd0 = vbr0 - vbl0;
      const float vtd1 = vtr1 - vtl1;
      const float vbd1 = vbr1 - vbl1;

      const float vt0 = vtl0 + vtd0 * valphah;
      const float vb0 = vbl0 + vbd0 * valphah;
      const float vt1 = vtl1 + vtd1 * valphah;
      const float vb1 = vbl1 + vbd1 * valphah;

      const float vd0 = vb0 - vt0;
      const float vd1 = vb1 - vt1;

      const float vo0 = vt0 + vd0 * valphav;
      const float vo1 = vt1 + vd1 * valphav;

      output[0] = vo0;
      output[1] = vo1;
      output += 2;
    }
    for (; c >= sizeof(float); c -= sizeof(float)) {
      const float vtl = *i0++;
      const float vtr = *i1++;
      const float vbl = *i2++;
      const float vbr = *i3++;

      const float vtd = vtr - vtl;
      const float vbd = vbr - vbl;

      const float vt = vtl + vtd * valphah;
      const float vb = vbl + vbd * valphah;

      const float vd = vb - vt;

      const float vo = vt + vd * valphav;

      *output++ = vo;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc00 = math_max_f32(vacc00, vmin);
    vacc01 = math_max_f32(vacc01, vmin);
    vacc02 = math_max_f32(vacc02, vmin);
    vacc03 = math_max_f32(vacc03, vmin);

    vacc00 = math_min_f32(vacc00, vmax);
    vacc01 = math_min_f32(vacc01, vmax);
    vacc02 = math_min_f32(vacc02, vmax);
    vacc03 = math_min_f32(vacc03, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc00 = math_max_f32(vacc00, 0.0f);
    vacc01 = math_max_f32(vacc01, 0.0f);
    vacc02 = math_max_f32(vacc02, 0.0f);
    vacc03 = math_max_f32(vacc03, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    w += 2;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        w += 2;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);

        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc00 = math_max_f32(vacc00, vmin);
    vacc01 = math_max_f32(vacc01, vmin);
    vacc10 = math_max_f32(vacc10, vmin);
    vacc11 = math_max_f32(vacc11, vmin);
    vacc20 = math_max_f32(vacc20, vmin);
    vacc21 = math_max_f32(vacc21, vmin);
    vacc30 = math_max_f32(vacc30, vmin);
    vacc31 = math_max_f32(vacc31, vmin);

    vacc00 = math_min_f32(vacc00, vmax);
    vacc01 = math_min_f32(vacc01, vmax);
    vacc10 = math_min_f32(vacc10, vmax);
    vacc11 = math_min_f32(vacc11, vmax);
    vacc20 = math_min_f32(vacc20, vmax);
    vacc21 = math_min_f32(vacc21, vmax);
    vacc30 = math_min_f32(vacc30, vmax);
    vacc31 = math_min_f32(vacc31, vmax);

    if XNN_LIKELY(nc >= 2) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_4x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    w += 2;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        w += 2;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);

        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 2) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc12 = math_muladd_f32(va1, vb2, vacc12);
        vacc13 = math_muladd_f32(va1, vb3, vacc13);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc22 = math_muladd_f32(va2, vb2, vacc22);
        vacc23 = math_muladd_f32(va2, vb3, vacc23);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);
        vacc32 = math_muladd_f32(va3, vb2, vacc32);
        vacc33 = math_muladd_f32(va3, vb3, vacc33);

        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc00 = math_max_f32(vacc00, vmin);
    vacc01 = math_max_f32(vacc01, vmin);
    vacc02 = math_max_f32(vacc02, vmin);
    vacc03 = math_max_f32(vacc03, vmin);
    vacc10 = math_max_f32(vacc10, vmin);
    vacc11 = math_max_f32(vacc11, vmin);
    vacc12 = math_max_f32(vacc12, vmin);
    vacc13 = math_max_f32(vacc13, vmin);
    vacc20 = math_max_f32(vacc20, vmin);
    vacc21 = math_max_f32(vacc21, vmin);
    vacc22 = math_max_f32(vacc22, vmin);
    vacc23 = math_max_f32(vacc23, vmin);
    vacc30 = math_max_f32(vacc30, vmin);
    vacc31 = math_max_f32(vacc31, vmin);
    vacc32 = math_max_f32(vacc32, vmin);
    vacc33 = math_max_f32(vacc33, vmin);

    vacc00 = math_min_f32(vacc00, vmax);
    vacc01 = math_min_f32(vacc01, vmax);
    vacc02 = math_min_f32(vacc02, vmax);
    vacc03 = math_min_f32(vacc03, vmax);
    vacc10 = math_min_f32(vacc10, vmax);
    vacc11 = math_min_f32(vacc11, vmax);
    vacc12 = math_min_f32(vacc12, vmax);
    vacc13 = math_min_f32(vacc13, vmax);
    vacc20 = math_min_f32(vacc20, vmax);
    vacc21 = math_min_f32(vacc21, vmax);
    vacc22 = math_min_f32(vacc22, vmax);
    vacc23 = math_min_f32(vacc23, vmax);
    vacc30 = math_min_f32(vacc30, vmax);
    vacc31 = math_min_f32(vacc31, vmax);
    vacc32 = math_min_f32(vacc32, vmax);
    vacc33 = math_min_f32(vacc33, vmax);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc12 = math_muladd_f32(va1, vb2, vacc12);
        vacc13 = math_muladd_f32(va1, vb3, vacc13);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc22 = math_muladd_f32(va2, vb2, vacc22);
        vacc23 = math_muladd_f32(va2, vb3, vacc23);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);
        vacc32 = math_muladd_f32(va3, vb2, vacc32);
        vacc33 = math_muladd_f32(va3, vb3, vacc33);

        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc00 = math_max_f32(vacc00, 0.0f);
    vacc01 = math_max_f32(vacc01, 0.0f);
    vacc02 = math_max_f32(vacc02, 0.0f);
    vacc03 = math_max_f32(vacc03, 0.0f);
    vacc10 = math_max_f32(vacc10, 0.0f);
    vacc11 = math_max_f32(vacc11, 0.0f);
    vacc12 = math_max_f32(vacc12, 0.0f);
    vacc13 = math_max_f32(vacc13, 0.0f);
    vacc20 = math_max_f32(vacc20, 0.0f);
    vacc21 = math_max_f32(vacc21, 0.0f);
    vacc22 = math_max_f32(vacc22, 0.0f);
    vacc23 = math_max_f32(vacc23, 0.0f);
    vacc30 = math_max_f32(vacc30, 0.0f);
    vacc31 = math_max_f32(vacc31, 0.0f);
    vacc32 = math_max_f32(vacc32, 0.0f);
    vacc33 = math_max_f32(vacc33, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc12 = math_muladd_f32(va1, vb2, vacc12);
        vacc13 = math_muladd_f32(va1, vb3, vacc13);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc22 = math_muladd_f32(va2, vb2, vacc22);
        vacc23 = math_muladd_f32(va2, vb3, vacc23);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);
        vacc32 = math_muladd_f32(va3, vb2, vacc32);
        vacc33 = math_muladd_f32(va3, vb3, vacc33);

        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  do {
    float* o = output;
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
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
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *i8++;

        const float vmax01 = math_max_f32(vi0, vi1);
        const float vmax23 = math_max_f32(vi2, vi3);
        const float vmax45 = math_max_f32(vi4, vi5);
        const float vmax67 = math_max_f32(vi6, vi7);
        const float vmax018 = math_max_f32(vmax01, vi8);

        const float vmax2345 = math_max_f32(vmax23, vmax45);
        const float vmax01678 = math_max_f32(vmax018, vmax67);
        float vout = math_max_f32(vmax2345, vmax01678);
        vout = math_max_f32(vout, voutput_min);
        vout = math_min_f32(vout, voutput_max);

        *o++ = vout;
      } while (--c != 0);
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
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
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *o;

        const float vmax01 = math_max_f32(vi0, vi1);
        const float vmax23 = math_max_f32(vi2, vi3);
        const float vmax45 = math_max_f32(vi4, vi5);
        const float vmax67 = math_max_f32(vi6, vi7);
        const float vmax018 = math_max_f32(vmax01, vi8);

        const float vmax2345 = math_max_f32(vmax23, vmax45);
        const float vmax01678 = math_max_f32(vmax018, vmax67);
        float vout = math_max_f32(vmax2345, vmax01678);
        vout = math_max_f32(vout, voutput_min);
        vout = math_min_f32(vout, voutput_max);

        *o++ = vout;
      } while (--c != 0);
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_pavgpool_minmax_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    const float* multiplier,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  do {
    {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      float* b = buffer;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *i8++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum018 = vsum01 + vi8;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum01678 = vsum018 + vsum67;
        const float vsum = vsum2345 + vsum01678;

        *b++ = vsum;
      } while (--c != 0);
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      float* b = buffer;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        *b++ = vsum;
      } while (--c != 0);
    }

    {
      const float* i0 = input[0];
      assert(i0 != NULL);
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      const float vmultiplier = *multiplier++;

      size_t c = channels;
      float* b = buffer;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        float vout = vsum * vmultiplier;
        vout = math_max_f32(vout, voutput_min);
        vout = math_min_f32(vout, voutput_max);

        *output++ = vout;
      } while (--c != 0);
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_pavgpool_minmax_ukernel_9x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    const float* multiplier,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    input = (const float**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }

    const float vmultiplier = *multiplier++;

    size_t c = channels;
    do {
      const float vi0 = *i0++;
      const float vi1 = *i1++;
      const float vi2 = *i2++;
      const float vi3 = *i3++;
      const float vi4 = *i4++;
      const float vi5 = *i5++;
      const float vi6 = *i6++;
      const float vi7 = *i7++;
      const float vi8 = *i8++;

      const float vsum01 = vi0 + vi1;
      const float vsum23 = vi2 + vi3;
      const float vsum45 = vi4 + vi5;
      const float vsum67 = vi6 + vi7;
      const float vsum018 = vsum01 + vi8;
      const float vsum2345 = vsum23 + vsum45;
      const float vsum01678 = vsum018 + vsum67;
      const float vsum = vsum2345 + vsum01678;

      float vout = vsum * vmultiplier;
      vout = math_max_f32(vout, voutput_min);
      vout = math_min_f32(vout, voutput_max);

      *output++ = vout;
    } while (--c != 0);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_prelu_ukernel__scalar_2x4(
    size_t rows,
    size_t channels,
    const float*restrict input,
    size_t input_stride,
    const float*restrict weights,
    float*restrict output,
    size_t output_stride)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float vw0 = w[0];
      const float vw1 = w[1];
      const float vw2 = w[2];
      const float vw3 = w[3];

      const float vi0x0 = i0[0];
      const float vi0x1 = i0[1];
      const float vi0x2 = i0[2];
      const float vi0x3 = i0[3];
      i0 += 4;
      const float vi1x0 = i1[0];
      const float vi1x1 = i1[1];
      const float vi1x2 = i1[2];
      const float vi1x3 = i1[3];
      i1 += 4;

      const float vacc0x0 = XNN_UNPREDICTABLE(vi0x0 < 0.0f) ? vi0x0 * vw0 : vi0x0;
      const float vacc0x1 = XNN_UNPREDICTABLE(vi0x1 < 0.0f) ? vi0x1 * vw1 : vi0x1;
      const float vacc0x2 = XNN_UNPREDICTABLE(vi0x2 < 0.0f) ? vi0x2 * vw2 : vi0x2;
      const float vacc0x3 = XNN_UNPREDICTABLE(vi0x3 < 0.0f) ? vi0x3 * vw3 : vi0x3;
      const float vacc1x0 = XNN_UNPREDICTABLE(vi1x0 < 0.0f) ? vi1x0 * vw0 : vi1x0;
      const float vacc1x1 = XNN_UNPREDICTABLE(vi1x1 < 0.0f) ? vi1x1 * vw1 : vi1x1;
      const float vacc1x2 = XNN_UNPREDICTABLE(vi1x2 < 0.0f) ? vi1x2 * vw2 : vi1x2;
      const float vacc1x3 = XNN_UNPREDICTABLE(vi1x3 < 0.0f) ? vi1x3 * vw3 : vi1x3;

      o0[0] = vacc0x0;
      o0[1] = vacc0x1;
      o0[2] = vacc0x2;
      o0[3] = vacc0x3;
      o0 += 4;
      o1[0] = vacc1x0;
      o1[1] = vacc1x1;
      o1[2] = vacc1x2;
      o1[3] = vacc1x3;
      o1 += 4;

      w += 4;
    }
    for (; c != 0; c -= sizeof(float)) {
      const float vw = *w++;

      const float vi0 = *i0++;
      const float vi1 = *i1++;

      const float vacc0 = XNN_UNPREDICTABLE(vi0 < 0.0f) ? vi0 * vw : vi0;
      const float vacc1 = XNN_UNPREDICTABLE(vi1 < 0.0f) ? vi1 * vw : vi1;

      *o0++ = vacc0;
      *o1++ = vacc1;
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_x4(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vscale = params->scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar_lrintf.output_zero_point;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    input += 4;

    vx0 *= vscale;
    vx1 *= vscale;
    vx2 *= vscale;
    vx3 *= vscale;

    vx0 = math_max_f32(vx0, voutput_min_less_zero_point);
    vx1 = math_max_f32(vx1, voutput_min_less_zero_point);
    vx2 = math_max_f32(vx2, voutput_min_less_zero_point);
    vx3 = math_max_f32(vx3, voutput_min_less_zero_point);

    vx0 = math_min_f32(vx0, voutput_max_less_zero_point);
    vx1 = math_min_f32(vx1, voutput_max_less_zero_point);
    vx2 = math_min_f32(vx2, voutput_max_less_zero_point);
    vx3 = math_min_f32(vx3, voutput_max_less_zero_point);

    int32_t vy0 = (int32_t) lrintf(vx0);
    int32_t vy1 = (int32_t) lrintf(vx1);
    int32_t vy2 = (int32_t) lrintf(vx2);
    int32_t vy3 = (int32_t) lrintf(vx3);

    vy0 += voutput_zero_point;
    vy1 += voutput_zero_point;
    vy2 += voutput_zero_point;
    vy3 += voutput_zero_point;

    output[0] = (int8_t) vy0;
    output[1] = (int8_t) vy1;
    output[2] = (int8_t) vy2;
    output[3] = (int8_t) vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;
      vx *= vscale;
      vx = math_max_f32(vx, voutput_min_less_zero_point);
      vx = math_min_f32(vx, voutput_max_less_zero_point);

      int32_t vy = (int32_t) lrintf(vx);
      vy += voutput_zero_point;

      *output++ = (int8_t) vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_x4(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vscale = params->scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar_lrintf.output_zero_point;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    input += 4;

    vx0 *= vscale;
    vx1 *= vscale;
    vx2 *= vscale;
    vx3 *= vscale;

    vx0 = math_max_f32(vx0, voutput_min_less_zero_point);
    vx1 = math_max_f32(vx1, voutput_min_less_zero_point);
    vx2 = math_max_f32(vx2, voutput_min_less_zero_point);
    vx3 = math_max_f32(vx3, voutput_min_less_zero_point);

    vx0 = math_min_f32(vx0, voutput_max_less_zero_point);
    vx1 = math_min_f32(vx1, voutput_max_less_zero_point);
    vx2 = math_min_f32(vx2, voutput_max_less_zero_point);
    vx3 = math_min_f32(vx3, voutput_max_less_zero_point);

    int32_t vy0 = (int32_t) lrintf(vx0);
    int32_t vy1 = (int32_t) lrintf(vx1);
    int32_t vy2 = (int32_t) lrintf(vx2);
    int32_t vy3 = (int32_t) lrintf(vx3);

    vy0 += voutput_zero_point;
    vy1 += voutput_zero_point;
    vy2 += voutput_zero_point;
    vy3 += voutput_zero_point;

    output[0] = (uint8_t) vy0;
    output[1] = (uint8_t) vy1;
    output[2] = (uint8_t) vy2;
    output[3] = (uint8_t) vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;
      vx *= vscale;
      vx = math_max_f32(vx, voutput_min_less_zero_point);
      vx = math_min_f32(vx, voutput_max_less_zero_point);

      int32_t vy = (int32_t) lrintf(vx);
      vy += voutput_zero_point;

      *output++ = (uint8_t) vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4_acc2(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float vi_max = *max;
  const float vlog2e = params->scalar_rr2_p5.log2e;
  const float vmagic_bias = params->scalar_rr2_p5.magic_bias;
  const float vminus_ln2_hi = params->scalar_rr2_p5.minus_ln2_hi;
  const float vminus_ln2_lo = params->scalar_rr2_p5.minus_ln2_lo;
  const float vc5 = params->scalar_rr2_p5.c5;
  const float vc4 = params->scalar_rr2_p5.c4;
  const float vc3 = params->scalar_rr2_p5.c3;
  const float vc2 = params->scalar_rr2_p5.c2;
  const float vc1 = params->scalar_rr2_p5.c1;
  const float vdenorm_cutoff = params->scalar_rr2_p5.denorm_cutoff;

  float vacc0 = 0.0f;
  float vacc1 = 0.0f;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    // Load 4 inputs at a time.
    const float vi0 = input[0];
    const float vi1 = input[1];
    const float vi2 = input[2];
    const float vi3 = input[3];
    input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float vx0 = vi0 - vi_max;
    const float vx1 = vi1 - vi_max;
    const float vx2 = vi2 - vi_max;
    const float vx3 = vi3 - vi_max;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but that's ok, because inputs outside of [-87.336540, 0.0] underflow expf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    float vn0 = vx0 * vlog2e + vmagic_bias;
    float vn1 = vx1 * vlog2e + vmagic_bias;
    float vn2 = vx2 * vlog2e + vmagic_bias;
    float vn3 = vx3 * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const float vs0 = uint32_as_float(float_as_uint32(vn0) << 23);
    const float vs1 = uint32_as_float(float_as_uint32(vn1) << 23);
    const float vs2 = uint32_as_float(float_as_uint32(vn2) << 23);
    const float vs3 = uint32_as_float(float_as_uint32(vn3) << 23);

    // Subtract the large number back to get final n := round(x / log(2)).
    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;
    vn2 -= vmagic_bias;
    vn3 -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt0 = vn0 * vminus_ln2_hi + vx0;
    float vt1 = vn1 * vminus_ln2_hi + vx1;
    float vt2 = vn2 * vminus_ln2_hi + vx2;
    float vt3 = vn3 * vminus_ln2_hi + vx3;

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;
    vt2 = vn2 * vminus_ln2_lo + vt2;
    vt3 = vn3 * vminus_ln2_lo + vt3;

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    float vp0 = vc5 * vt0 + vc4;
    float vp1 = vc5 * vt1 + vc4;
    float vp2 = vc5 * vt2 + vc4;
    float vp3 = vc5 * vt3 + vc4;

    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;
    vp2 = vp2 * vt2 + vc3;
    vp3 = vp3 * vt3 + vc3;

    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;
    vp2 = vp2 * vt2 + vc2;
    vp3 = vp3 * vt3 + vc2;

    vp0 = vp0 * vt0 + vc1;
    vp1 = vp1 * vt1 + vc1;
    vp2 = vp2 * vt2 + vc1;
    vp3 = vp3 * vt3 + vc1;

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0 *= vs0;
    vt1 *= vs1;
    vt2 *= vs2;
    vt3 *= vs3;

    float vf0 = vt0 * vp0 + vs0;
    float vf1 = vt1 * vp1 + vs1;
    float vf2 = vt2 * vp2 + vs2;
    float vf3 = vt3 * vp3 + vs3;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vx0 < vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx1 < vdenorm_cutoff) {
      vf1 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx2 < vdenorm_cutoff) {
      vf2 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx3 < vdenorm_cutoff) {
      vf3 = 0.0f;
    }

    // Store 4 outputs at a time.
    output[0] = vf0;
    output[1] = vf1;
    output[2] = vf2;
    output[3] = vf3;
    output += 4;

    // Accumulate computed exponents.
    vacc0 += vf0;
    vacc1 += vf1;
    vacc0 += vf2;
    vacc1 += vf3;
  }
  // Add up all accumulators to vacc0
  vacc0 += vacc1;

  float vacc = vacc0;
  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    // Load 1 input at a time.
    const float vi = *input++;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float vx = vi - vi_max;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but that's ok, because inputs outside of [-87.336540, 0.0] underflow expf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    float vn = vx * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const float vs = uint32_as_float(float_as_uint32(vn) << 23);

    // Subtract the large number back to get final n := round(x / log(2)).
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt = vn * vminus_ln2_hi + vx;
    vt = vn * vminus_ln2_lo + vt;

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    float vp = vc5 * vt + vc4;
    vp = vp * vt + vc3;
    vp = vp * vt + vc2;
    vp = vp * vt + vc1;

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt *= vs;
    float vf = vt * vp + vs;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vx < vdenorm_cutoff) {
      vf = 0.0f;
    }

    // Store 1 output at a time.
    *output++ = vf;

    // Accumulate computed exponents.
    vacc += vf;
  }
  *sum = vacc;
}

void xnn_f32_rmax_ukernel__scalar(
    size_t batch,
    const float* input,
    float* output)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float vmax0 = *input;
  float vmax1 = vmax0;
  float vmax2 = vmax0;
  float vmax3 = vmax0;
  for (; batch >= 16; batch -= 16) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    vmax0 = math_max_f32(vx0, vmax0);
    vmax1 = math_max_f32(vx1, vmax1);
    vmax2 = math_max_f32(vx2, vmax2);
    vmax3 = math_max_f32(vx3, vmax3);
  }
  const float vmax01 = math_max_f32(vmax0, vmax1);
  const float vmax23 = math_max_f32(vmax2, vmax3);
  float vmax = math_max_f32(vmax01, vmax23);
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      vmax = math_max_f32(vx, vmax);
      batch -= 4;
    } while (batch != 0);
  }
  *output = vmax;
}

void xnn_f32_spmm_minmax_ukernel_8x1__scalar(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  size_t output_decrement = output_stride * nc - 8 * sizeof(float);
  while (mc >= 8 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 1) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = *w++;
      float vacc1x0 = vacc0x0;
      float vacc2x0 = vacc0x0;
      float vacc3x0 = vacc0x0;
      float vacc4x0 = vacc0x0;
      float vacc5x0 = vacc0x0;
      float vacc6x0 = vacc0x0;
      float vacc7x0 = vacc0x0;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float vi0 = input[0];
          const float vi1 = input[1];
          const float vi2 = input[2];
          const float vi3 = input[3];
          const float vi4 = input[4];
          const float vi5 = input[5];
          const float vi6 = input[6];
          const float vi7 = input[7];
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const float vw0 = *w++;
          vacc0x0 += vi0 * vw0;
          vacc1x0 += vi1 * vw0;
          vacc2x0 += vi2 * vw0;
          vacc3x0 += vi3 * vw0;
          vacc4x0 += vi4 * vw0;
          vacc5x0 += vi5 * vw0;
          vacc6x0 += vi6 * vw0;
          vacc7x0 += vi7 * vw0;
        } while (--nnz != 0);
      }
      float vout0x0 = math_min_f32(vacc0x0, vmax);
      float vout1x0 = math_min_f32(vacc1x0, vmax);
      float vout2x0 = math_min_f32(vacc2x0, vmax);
      float vout3x0 = math_min_f32(vacc3x0, vmax);
      float vout4x0 = math_min_f32(vacc4x0, vmax);
      float vout5x0 = math_min_f32(vacc5x0, vmax);
      float vout6x0 = math_min_f32(vacc6x0, vmax);
      float vout7x0 = math_min_f32(vacc7x0, vmax);
      vout0x0 = math_max_f32(vout0x0, vmin);
      vout1x0 = math_max_f32(vout1x0, vmin);
      vout2x0 = math_max_f32(vout2x0, vmin);
      vout3x0 = math_max_f32(vout3x0, vmin);
      vout4x0 = math_max_f32(vout4x0, vmin);
      vout5x0 = math_max_f32(vout5x0, vmin);
      vout6x0 = math_max_f32(vout6x0, vmin);
      vout7x0 = math_max_f32(vout7x0, vmin);
      output[0] = vout0x0;
      output[1] = vout1x0;
      output[2] = vout2x0;
      output[3] = vout3x0;
      output[4] = vout4x0;
      output[5] = vout5x0;
      output[6] = vout6x0;
      output[7] = vout7x0;
      output[0] = vout0x0;
      output[1] = vout1x0;
      output[2] = vout2x0;
      output[3] = vout3x0;
      output[4] = vout4x0;
      output[5] = vout5x0;
      output[6] = vout6x0;
      output[7] = vout7x0;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      n -= 1;
    }
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = *w++;
        float vacc1 = vacc0;
        float vacc2 = vacc0;
        float vacc3 = vacc0;
        float vacc4 = vacc0;
        float vacc5 = vacc0;
        float vacc6 = vacc0;
        float vacc7 = vacc0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            const float vi4 = input[4];
            const float vi5 = input[5];
            const float vi6 = input[6];
            const float vi7 = input[7];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw = *w++;
            vacc0 += vi0 * vw;
            vacc1 += vi1 * vw;
            vacc2 += vi2 * vw;
            vacc3 += vi3 * vw;
            vacc4 += vi4 * vw;
            vacc5 += vi5 * vw;
            vacc6 += vi6 * vw;
            vacc7 += vi7 * vw;
          } while (--nnz != 0);
        }
        float vout0 = math_min_f32(vacc0, vmax);
        float vout1 = math_min_f32(vacc1, vmax);
        float vout2 = math_min_f32(vacc2, vmax);
        float vout3 = math_min_f32(vacc3, vmax);
        float vout4 = math_min_f32(vacc4, vmax);
        float vout5 = math_min_f32(vacc5, vmax);
        float vout6 = math_min_f32(vacc6, vmax);
        float vout7 = math_min_f32(vacc7, vmax);
        vout0 = math_max_f32(vout0, vmin);
        vout1 = math_max_f32(vout1, vmin);
        vout2 = math_max_f32(vout2, vmin);
        vout3 = math_max_f32(vout3, vmin);
        vout4 = math_max_f32(vout4, vmin);
        vout5 = math_max_f32(vout5, vmin);
        vout6 = math_max_f32(vout6, vmin);
        vout7 = math_max_f32(vout7, vmin);
        output[0] = vout0;
        output[1] = vout1;
        output[2] = vout2;
        output[3] = vout3;
        output[4] = vout4;
        output[5] = vout5;
        output[6] = vout6;
        output[7] = vout7;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 8;
    mc -= 8 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 1) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc2x0 = vacc0x0;
        float vacc3x0 = vacc0x0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc2x0 += vi2 * vw0;
            vacc3x0 += vi3 * vw0;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout2x0 = math_min_f32(vacc2x0, vmax);
        float vout3x0 = math_min_f32(vacc3x0, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout2x0 = math_max_f32(vout2x0, vmin);
        vout3x0 = math_max_f32(vout3x0, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output[2] = vout2x0;
        output[3] = vout3x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          float vacc2 = vacc0;
          float vacc3 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              const float vi2 = input[2];
              const float vi3 = input[3];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
              vacc2 += vi2 * vw;
              vacc3 += vi3 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          float vout2 = math_min_f32(vacc2, vmax);
          float vout3 = math_min_f32(vacc3, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          vout2 = math_max_f32(vout2, vmin);
          vout3 = math_max_f32(vout3, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output[2] = vout2;
          output[3] = vout3;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 1) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc1x0 = vacc0x0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 1) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            vacc0x0 += vi0 * vw0;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        output[0] = vout0x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          vout0 = math_max_f32(vout0, vmin);
          output[0] = vout0;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}

void xnn_f32_spmm_minmax_ukernel_8x2__scalar(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  size_t output_decrement = output_stride * nc - 8 * sizeof(float);
  while (mc >= 8 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 2) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = *w++;
      float vacc1x0 = vacc0x0;
      float vacc2x0 = vacc0x0;
      float vacc3x0 = vacc0x0;
      float vacc4x0 = vacc0x0;
      float vacc5x0 = vacc0x0;
      float vacc6x0 = vacc0x0;
      float vacc7x0 = vacc0x0;
      float vacc0x1 = *w++;
      float vacc1x1 = vacc0x1;
      float vacc2x1 = vacc0x1;
      float vacc3x1 = vacc0x1;
      float vacc4x1 = vacc0x1;
      float vacc5x1 = vacc0x1;
      float vacc6x1 = vacc0x1;
      float vacc7x1 = vacc0x1;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float vi0 = input[0];
          const float vi1 = input[1];
          const float vi2 = input[2];
          const float vi3 = input[3];
          const float vi4 = input[4];
          const float vi5 = input[5];
          const float vi6 = input[6];
          const float vi7 = input[7];
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const float vw0 = *w++;
          const float vw1 = *w++;
          vacc0x0 += vi0 * vw0;
          vacc1x0 += vi1 * vw0;
          vacc2x0 += vi2 * vw0;
          vacc3x0 += vi3 * vw0;
          vacc4x0 += vi4 * vw0;
          vacc5x0 += vi5 * vw0;
          vacc6x0 += vi6 * vw0;
          vacc7x0 += vi7 * vw0;
          vacc0x1 += vi0 * vw1;
          vacc1x1 += vi1 * vw1;
          vacc2x1 += vi2 * vw1;
          vacc3x1 += vi3 * vw1;
          vacc4x1 += vi4 * vw1;
          vacc5x1 += vi5 * vw1;
          vacc6x1 += vi6 * vw1;
          vacc7x1 += vi7 * vw1;
        } while (--nnz != 0);
      }
      float vout0x0 = math_min_f32(vacc0x0, vmax);
      float vout1x0 = math_min_f32(vacc1x0, vmax);
      float vout2x0 = math_min_f32(vacc2x0, vmax);
      float vout3x0 = math_min_f32(vacc3x0, vmax);
      float vout4x0 = math_min_f32(vacc4x0, vmax);
      float vout5x0 = math_min_f32(vacc5x0, vmax);
      float vout6x0 = math_min_f32(vacc6x0, vmax);
      float vout7x0 = math_min_f32(vacc7x0, vmax);
      float vout0x1 = math_min_f32(vacc0x1, vmax);
      float vout1x1 = math_min_f32(vacc1x1, vmax);
      float vout2x1 = math_min_f32(vacc2x1, vmax);
      float vout3x1 = math_min_f32(vacc3x1, vmax);
      float vout4x1 = math_min_f32(vacc4x1, vmax);
      float vout5x1 = math_min_f32(vacc5x1, vmax);
      float vout6x1 = math_min_f32(vacc6x1, vmax);
      float vout7x1 = math_min_f32(vacc7x1, vmax);
      vout0x0 = math_max_f32(vout0x0, vmin);
      vout1x0 = math_max_f32(vout1x0, vmin);
      vout2x0 = math_max_f32(vout2x0, vmin);
      vout3x0 = math_max_f32(vout3x0, vmin);
      vout4x0 = math_max_f32(vout4x0, vmin);
      vout5x0 = math_max_f32(vout5x0, vmin);
      vout6x0 = math_max_f32(vout6x0, vmin);
      vout7x0 = math_max_f32(vout7x0, vmin);
      vout0x1 = math_max_f32(vout0x1, vmin);
      vout1x1 = math_max_f32(vout1x1, vmin);
      vout2x1 = math_max_f32(vout2x1, vmin);
      vout3x1 = math_max_f32(vout3x1, vmin);
      vout4x1 = math_max_f32(vout4x1, vmin);
      vout5x1 = math_max_f32(vout5x1, vmin);
      vout6x1 = math_max_f32(vout6x1, vmin);
      vout7x1 = math_max_f32(vout7x1, vmin);
      output[0] = vout0x1;
      output[1] = vout1x1;
      output[2] = vout2x1;
      output[3] = vout3x1;
      output[4] = vout4x1;
      output[5] = vout5x1;
      output[6] = vout6x1;
      output[7] = vout7x1;
      output[0] = vout0x0;
      output[1] = vout1x0;
      output[2] = vout2x0;
      output[3] = vout3x0;
      output[4] = vout4x0;
      output[5] = vout5x0;
      output[6] = vout6x0;
      output[7] = vout7x0;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      output[0] = vout0x1;
      output[1] = vout1x1;
      output[2] = vout2x1;
      output[3] = vout3x1;
      output[4] = vout4x1;
      output[5] = vout5x1;
      output[6] = vout6x1;
      output[7] = vout7x1;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      n -= 2;
    }
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = *w++;
        float vacc1 = vacc0;
        float vacc2 = vacc0;
        float vacc3 = vacc0;
        float vacc4 = vacc0;
        float vacc5 = vacc0;
        float vacc6 = vacc0;
        float vacc7 = vacc0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            const float vi4 = input[4];
            const float vi5 = input[5];
            const float vi6 = input[6];
            const float vi7 = input[7];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw = *w++;
            vacc0 += vi0 * vw;
            vacc1 += vi1 * vw;
            vacc2 += vi2 * vw;
            vacc3 += vi3 * vw;
            vacc4 += vi4 * vw;
            vacc5 += vi5 * vw;
            vacc6 += vi6 * vw;
            vacc7 += vi7 * vw;
          } while (--nnz != 0);
        }
        float vout0 = math_min_f32(vacc0, vmax);
        float vout1 = math_min_f32(vacc1, vmax);
        float vout2 = math_min_f32(vacc2, vmax);
        float vout3 = math_min_f32(vacc3, vmax);
        float vout4 = math_min_f32(vacc4, vmax);
        float vout5 = math_min_f32(vacc5, vmax);
        float vout6 = math_min_f32(vacc6, vmax);
        float vout7 = math_min_f32(vacc7, vmax);
        vout0 = math_max_f32(vout0, vmin);
        vout1 = math_max_f32(vout1, vmin);
        vout2 = math_max_f32(vout2, vmin);
        vout3 = math_max_f32(vout3, vmin);
        vout4 = math_max_f32(vout4, vmin);
        vout5 = math_max_f32(vout5, vmin);
        vout6 = math_max_f32(vout6, vmin);
        vout7 = math_max_f32(vout7, vmin);
        output[0] = vout0;
        output[1] = vout1;
        output[2] = vout2;
        output[3] = vout3;
        output[4] = vout4;
        output[5] = vout5;
        output[6] = vout6;
        output[7] = vout7;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 8;
    mc -= 8 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc2x0 = vacc0x0;
        float vacc3x0 = vacc0x0;
        float vacc0x1 = *w++;
        float vacc1x1 = vacc0x1;
        float vacc2x1 = vacc0x1;
        float vacc3x1 = vacc0x1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc2x0 += vi2 * vw0;
            vacc3x0 += vi3 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc1x1 += vi1 * vw1;
            vacc2x1 += vi2 * vw1;
            vacc3x1 += vi3 * vw1;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout2x0 = math_min_f32(vacc2x0, vmax);
        float vout3x0 = math_min_f32(vacc3x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout1x1 = math_min_f32(vacc1x1, vmax);
        float vout2x1 = math_min_f32(vacc2x1, vmax);
        float vout3x1 = math_min_f32(vacc3x1, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout2x0 = math_max_f32(vout2x0, vmin);
        vout3x0 = math_max_f32(vout3x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout1x1 = math_max_f32(vout1x1, vmin);
        vout2x1 = math_max_f32(vout2x1, vmin);
        vout3x1 = math_max_f32(vout3x1, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output[2] = vout2x0;
        output[3] = vout3x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output[1] = vout1x1;
        output[2] = vout2x1;
        output[3] = vout3x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 2;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          float vacc2 = vacc0;
          float vacc3 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              const float vi2 = input[2];
              const float vi3 = input[3];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
              vacc2 += vi2 * vw;
              vacc3 += vi3 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          float vout2 = math_min_f32(vacc2, vmax);
          float vout3 = math_min_f32(vacc3, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          vout2 = math_max_f32(vout2, vmin);
          vout3 = math_max_f32(vout3, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output[2] = vout2;
          output[3] = vout3;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc0x1 = *w++;
        float vacc1x1 = vacc0x1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc1x1 += vi1 * vw1;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout1x1 = math_min_f32(vacc1x1, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout1x1 = math_max_f32(vout1x1, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output[1] = vout1x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 2;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc0x1 = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc0x1 += vi0 * vw1;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        output[0] = vout0x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 2;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          vout0 = math_max_f32(vout0, vmin);
          output[0] = vout0;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}

void xnn_f32_spmm_minmax_ukernel_8x4__scalar(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  size_t output_decrement = output_stride * nc - 8 * sizeof(float);
  while (mc >= 8 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 4) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = *w++;
      float vacc1x0 = vacc0x0;
      float vacc2x0 = vacc0x0;
      float vacc3x0 = vacc0x0;
      float vacc4x0 = vacc0x0;
      float vacc5x0 = vacc0x0;
      float vacc6x0 = vacc0x0;
      float vacc7x0 = vacc0x0;
      float vacc0x1 = *w++;
      float vacc1x1 = vacc0x1;
      float vacc2x1 = vacc0x1;
      float vacc3x1 = vacc0x1;
      float vacc4x1 = vacc0x1;
      float vacc5x1 = vacc0x1;
      float vacc6x1 = vacc0x1;
      float vacc7x1 = vacc0x1;
      float vacc0x2 = *w++;
      float vacc1x2 = vacc0x2;
      float vacc2x2 = vacc0x2;
      float vacc3x2 = vacc0x2;
      float vacc4x2 = vacc0x2;
      float vacc5x2 = vacc0x2;
      float vacc6x2 = vacc0x2;
      float vacc7x2 = vacc0x2;
      float vacc0x3 = *w++;
      float vacc1x3 = vacc0x3;
      float vacc2x3 = vacc0x3;
      float vacc3x3 = vacc0x3;
      float vacc4x3 = vacc0x3;
      float vacc5x3 = vacc0x3;
      float vacc6x3 = vacc0x3;
      float vacc7x3 = vacc0x3;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float vi0 = input[0];
          const float vi1 = input[1];
          const float vi2 = input[2];
          const float vi3 = input[3];
          const float vi4 = input[4];
          const float vi5 = input[5];
          const float vi6 = input[6];
          const float vi7 = input[7];
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const float vw0 = *w++;
          const float vw1 = *w++;
          const float vw2 = *w++;
          const float vw3 = *w++;
          vacc0x0 += vi0 * vw0;
          vacc1x0 += vi1 * vw0;
          vacc2x0 += vi2 * vw0;
          vacc3x0 += vi3 * vw0;
          vacc4x0 += vi4 * vw0;
          vacc5x0 += vi5 * vw0;
          vacc6x0 += vi6 * vw0;
          vacc7x0 += vi7 * vw0;
          vacc0x1 += vi0 * vw1;
          vacc1x1 += vi1 * vw1;
          vacc2x1 += vi2 * vw1;
          vacc3x1 += vi3 * vw1;
          vacc4x1 += vi4 * vw1;
          vacc5x1 += vi5 * vw1;
          vacc6x1 += vi6 * vw1;
          vacc7x1 += vi7 * vw1;
          vacc0x2 += vi0 * vw2;
          vacc1x2 += vi1 * vw2;
          vacc2x2 += vi2 * vw2;
          vacc3x2 += vi3 * vw2;
          vacc4x2 += vi4 * vw2;
          vacc5x2 += vi5 * vw2;
          vacc6x2 += vi6 * vw2;
          vacc7x2 += vi7 * vw2;
          vacc0x3 += vi0 * vw3;
          vacc1x3 += vi1 * vw3;
          vacc2x3 += vi2 * vw3;
          vacc3x3 += vi3 * vw3;
          vacc4x3 += vi4 * vw3;
          vacc5x3 += vi5 * vw3;
          vacc6x3 += vi6 * vw3;
          vacc7x3 += vi7 * vw3;
        } while (--nnz != 0);
      }
      float vout0x0 = math_min_f32(vacc0x0, vmax);
      float vout1x0 = math_min_f32(vacc1x0, vmax);
      float vout2x0 = math_min_f32(vacc2x0, vmax);
      float vout3x0 = math_min_f32(vacc3x0, vmax);
      float vout4x0 = math_min_f32(vacc4x0, vmax);
      float vout5x0 = math_min_f32(vacc5x0, vmax);
      float vout6x0 = math_min_f32(vacc6x0, vmax);
      float vout7x0 = math_min_f32(vacc7x0, vmax);
      float vout0x1 = math_min_f32(vacc0x1, vmax);
      float vout1x1 = math_min_f32(vacc1x1, vmax);
      float vout2x1 = math_min_f32(vacc2x1, vmax);
      float vout3x1 = math_min_f32(vacc3x1, vmax);
      float vout4x1 = math_min_f32(vacc4x1, vmax);
      float vout5x1 = math_min_f32(vacc5x1, vmax);
      float vout6x1 = math_min_f32(vacc6x1, vmax);
      float vout7x1 = math_min_f32(vacc7x1, vmax);
      float vout0x2 = math_min_f32(vacc0x2, vmax);
      float vout1x2 = math_min_f32(vacc1x2, vmax);
      float vout2x2 = math_min_f32(vacc2x2, vmax);
      float vout3x2 = math_min_f32(vacc3x2, vmax);
      float vout4x2 = math_min_f32(vacc4x2, vmax);
      float vout5x2 = math_min_f32(vacc5x2, vmax);
      float vout6x2 = math_min_f32(vacc6x2, vmax);
      float vout7x2 = math_min_f32(vacc7x2, vmax);
      float vout0x3 = math_min_f32(vacc0x3, vmax);
      float vout1x3 = math_min_f32(vacc1x3, vmax);
      float vout2x3 = math_min_f32(vacc2x3, vmax);
      float vout3x3 = math_min_f32(vacc3x3, vmax);
      float vout4x3 = math_min_f32(vacc4x3, vmax);
      float vout5x3 = math_min_f32(vacc5x3, vmax);
      float vout6x3 = math_min_f32(vacc6x3, vmax);
      float vout7x3 = math_min_f32(vacc7x3, vmax);
      vout0x0 = math_max_f32(vout0x0, vmin);
      vout1x0 = math_max_f32(vout1x0, vmin);
      vout2x0 = math_max_f32(vout2x0, vmin);
      vout3x0 = math_max_f32(vout3x0, vmin);
      vout4x0 = math_max_f32(vout4x0, vmin);
      vout5x0 = math_max_f32(vout5x0, vmin);
      vout6x0 = math_max_f32(vout6x0, vmin);
      vout7x0 = math_max_f32(vout7x0, vmin);
      vout0x1 = math_max_f32(vout0x1, vmin);
      vout1x1 = math_max_f32(vout1x1, vmin);
      vout2x1 = math_max_f32(vout2x1, vmin);
      vout3x1 = math_max_f32(vout3x1, vmin);
      vout4x1 = math_max_f32(vout4x1, vmin);
      vout5x1 = math_max_f32(vout5x1, vmin);
      vout6x1 = math_max_f32(vout6x1, vmin);
      vout7x1 = math_max_f32(vout7x1, vmin);
      vout0x2 = math_max_f32(vout0x2, vmin);
      vout1x2 = math_max_f32(vout1x2, vmin);
      vout2x2 = math_max_f32(vout2x2, vmin);
      vout3x2 = math_max_f32(vout3x2, vmin);
      vout4x2 = math_max_f32(vout4x2, vmin);
      vout5x2 = math_max_f32(vout5x2, vmin);
      vout6x2 = math_max_f32(vout6x2, vmin);
      vout7x2 = math_max_f32(vout7x2, vmin);
      vout0x3 = math_max_f32(vout0x3, vmin);
      vout1x3 = math_max_f32(vout1x3, vmin);
      vout2x3 = math_max_f32(vout2x3, vmin);
      vout3x3 = math_max_f32(vout3x3, vmin);
      vout4x3 = math_max_f32(vout4x3, vmin);
      vout5x3 = math_max_f32(vout5x3, vmin);
      vout6x3 = math_max_f32(vout6x3, vmin);
      vout7x3 = math_max_f32(vout7x3, vmin);
      output[0] = vout0x3;
      output[1] = vout1x3;
      output[2] = vout2x3;
      output[3] = vout3x3;
      output[4] = vout4x3;
      output[5] = vout5x3;
      output[6] = vout6x3;
      output[7] = vout7x3;
      output[0] = vout0x0;
      output[1] = vout1x0;
      output[2] = vout2x0;
      output[3] = vout3x0;
      output[4] = vout4x0;
      output[5] = vout5x0;
      output[6] = vout6x0;
      output[7] = vout7x0;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      output[0] = vout0x1;
      output[1] = vout1x1;
      output[2] = vout2x1;
      output[3] = vout3x1;
      output[4] = vout4x1;
      output[5] = vout5x1;
      output[6] = vout6x1;
      output[7] = vout7x1;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      output[0] = vout0x2;
      output[1] = vout1x2;
      output[2] = vout2x2;
      output[3] = vout3x2;
      output[4] = vout4x2;
      output[5] = vout5x2;
      output[6] = vout6x2;
      output[7] = vout7x2;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      output[0] = vout0x3;
      output[1] = vout1x3;
      output[2] = vout2x3;
      output[3] = vout3x3;
      output[4] = vout4x3;
      output[5] = vout5x3;
      output[6] = vout6x3;
      output[7] = vout7x3;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      n -= 4;
    }
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = *w++;
        float vacc1 = vacc0;
        float vacc2 = vacc0;
        float vacc3 = vacc0;
        float vacc4 = vacc0;
        float vacc5 = vacc0;
        float vacc6 = vacc0;
        float vacc7 = vacc0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            const float vi4 = input[4];
            const float vi5 = input[5];
            const float vi6 = input[6];
            const float vi7 = input[7];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw = *w++;
            vacc0 += vi0 * vw;
            vacc1 += vi1 * vw;
            vacc2 += vi2 * vw;
            vacc3 += vi3 * vw;
            vacc4 += vi4 * vw;
            vacc5 += vi5 * vw;
            vacc6 += vi6 * vw;
            vacc7 += vi7 * vw;
          } while (--nnz != 0);
        }
        float vout0 = math_min_f32(vacc0, vmax);
        float vout1 = math_min_f32(vacc1, vmax);
        float vout2 = math_min_f32(vacc2, vmax);
        float vout3 = math_min_f32(vacc3, vmax);
        float vout4 = math_min_f32(vacc4, vmax);
        float vout5 = math_min_f32(vacc5, vmax);
        float vout6 = math_min_f32(vacc6, vmax);
        float vout7 = math_min_f32(vacc7, vmax);
        vout0 = math_max_f32(vout0, vmin);
        vout1 = math_max_f32(vout1, vmin);
        vout2 = math_max_f32(vout2, vmin);
        vout3 = math_max_f32(vout3, vmin);
        vout4 = math_max_f32(vout4, vmin);
        vout5 = math_max_f32(vout5, vmin);
        vout6 = math_max_f32(vout6, vmin);
        vout7 = math_max_f32(vout7, vmin);
        output[0] = vout0;
        output[1] = vout1;
        output[2] = vout2;
        output[3] = vout3;
        output[4] = vout4;
        output[5] = vout5;
        output[6] = vout6;
        output[7] = vout7;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 8;
    mc -= 8 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc2x0 = vacc0x0;
        float vacc3x0 = vacc0x0;
        float vacc0x1 = *w++;
        float vacc1x1 = vacc0x1;
        float vacc2x1 = vacc0x1;
        float vacc3x1 = vacc0x1;
        float vacc0x2 = *w++;
        float vacc1x2 = vacc0x2;
        float vacc2x2 = vacc0x2;
        float vacc3x2 = vacc0x2;
        float vacc0x3 = *w++;
        float vacc1x3 = vacc0x3;
        float vacc2x3 = vacc0x3;
        float vacc3x3 = vacc0x3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            const float vw2 = *w++;
            const float vw3 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc2x0 += vi2 * vw0;
            vacc3x0 += vi3 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc1x1 += vi1 * vw1;
            vacc2x1 += vi2 * vw1;
            vacc3x1 += vi3 * vw1;
            vacc0x2 += vi0 * vw2;
            vacc1x2 += vi1 * vw2;
            vacc2x2 += vi2 * vw2;
            vacc3x2 += vi3 * vw2;
            vacc0x3 += vi0 * vw3;
            vacc1x3 += vi1 * vw3;
            vacc2x3 += vi2 * vw3;
            vacc3x3 += vi3 * vw3;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout2x0 = math_min_f32(vacc2x0, vmax);
        float vout3x0 = math_min_f32(vacc3x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout1x1 = math_min_f32(vacc1x1, vmax);
        float vout2x1 = math_min_f32(vacc2x1, vmax);
        float vout3x1 = math_min_f32(vacc3x1, vmax);
        float vout0x2 = math_min_f32(vacc0x2, vmax);
        float vout1x2 = math_min_f32(vacc1x2, vmax);
        float vout2x2 = math_min_f32(vacc2x2, vmax);
        float vout3x2 = math_min_f32(vacc3x2, vmax);
        float vout0x3 = math_min_f32(vacc0x3, vmax);
        float vout1x3 = math_min_f32(vacc1x3, vmax);
        float vout2x3 = math_min_f32(vacc2x3, vmax);
        float vout3x3 = math_min_f32(vacc3x3, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout2x0 = math_max_f32(vout2x0, vmin);
        vout3x0 = math_max_f32(vout3x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout1x1 = math_max_f32(vout1x1, vmin);
        vout2x1 = math_max_f32(vout2x1, vmin);
        vout3x1 = math_max_f32(vout3x1, vmin);
        vout0x2 = math_max_f32(vout0x2, vmin);
        vout1x2 = math_max_f32(vout1x2, vmin);
        vout2x2 = math_max_f32(vout2x2, vmin);
        vout3x2 = math_max_f32(vout3x2, vmin);
        vout0x3 = math_max_f32(vout0x3, vmin);
        vout1x3 = math_max_f32(vout1x3, vmin);
        vout2x3 = math_max_f32(vout2x3, vmin);
        vout3x3 = math_max_f32(vout3x3, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output[2] = vout2x0;
        output[3] = vout3x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output[1] = vout1x1;
        output[2] = vout2x1;
        output[3] = vout3x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x2;
        output[1] = vout1x2;
        output[2] = vout2x2;
        output[3] = vout3x2;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x3;
        output[1] = vout1x3;
        output[2] = vout2x3;
        output[3] = vout3x3;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          float vacc2 = vacc0;
          float vacc3 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              const float vi2 = input[2];
              const float vi3 = input[3];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
              vacc2 += vi2 * vw;
              vacc3 += vi3 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          float vout2 = math_min_f32(vacc2, vmax);
          float vout3 = math_min_f32(vacc3, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          vout2 = math_max_f32(vout2, vmin);
          vout3 = math_max_f32(vout3, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output[2] = vout2;
          output[3] = vout3;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc0x1 = *w++;
        float vacc1x1 = vacc0x1;
        float vacc0x2 = *w++;
        float vacc1x2 = vacc0x2;
        float vacc0x3 = *w++;
        float vacc1x3 = vacc0x3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            const float vw2 = *w++;
            const float vw3 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc1x1 += vi1 * vw1;
            vacc0x2 += vi0 * vw2;
            vacc1x2 += vi1 * vw2;
            vacc0x3 += vi0 * vw3;
            vacc1x3 += vi1 * vw3;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout1x1 = math_min_f32(vacc1x1, vmax);
        float vout0x2 = math_min_f32(vacc0x2, vmax);
        float vout1x2 = math_min_f32(vacc1x2, vmax);
        float vout0x3 = math_min_f32(vacc0x3, vmax);
        float vout1x3 = math_min_f32(vacc1x3, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout1x1 = math_max_f32(vout1x1, vmin);
        vout0x2 = math_max_f32(vout0x2, vmin);
        vout1x2 = math_max_f32(vout1x2, vmin);
        vout0x3 = math_max_f32(vout0x3, vmin);
        vout1x3 = math_max_f32(vout1x3, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output[1] = vout1x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x2;
        output[1] = vout1x2;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x3;
        output[1] = vout1x3;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc0x1 = *w++;
        float vacc0x2 = *w++;
        float vacc0x3 = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            const float vw2 = *w++;
            const float vw3 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc0x2 += vi0 * vw2;
            vacc0x3 += vi0 * vw3;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout0x2 = math_min_f32(vacc0x2, vmax);
        float vout0x3 = math_min_f32(vacc0x3, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout0x2 = math_max_f32(vout0x2, vmin);
        vout0x3 = math_max_f32(vout0x3, vmin);
        output[0] = vout0x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x2;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x3;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          vout0 = math_max_f32(vout0, vmin);
          output[0] = vout0;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}

void xnn_f32_vadd_minmax_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 + vb0;
    float vacc1 = va1 + vb1;
    float vacc2 = va2 + vb2;
    float vacc3 = va3 + vb3;
    float vacc4 = va4 + vb4;
    float vacc5 = va5 + vb5;
    float vacc6 = va6 + vb6;
    float vacc7 = va7 + vb7;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va + vb;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vaddc_minmax_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 + vb;
    float vacc1 = va1 + vb;
    float vacc2 = va2 + vb;
    float vacc3 = va3 + vb;
    float vacc4 = va4 + vb;
    float vacc5 = va5 + vb;
    float vacc6 = va6 + vb;
    float vacc7 = va7 + vb;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va + vb;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vdiv_minmax_ukernel__scalar_x2(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    input_a += 2;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    input_b += 2;

    float vacc0 = va0 / vb0;
    float vacc1 = va1 / vb1;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch == sizeof(float));
    const float va = *input_a;
    const float vb = *input_b;
    float vacc = va / vb;
    vacc = math_max_f32(vacc, voutput_min);
    vacc = math_min_f32(vacc, voutput_max);
    *output = vacc;
  }
}

void xnn_f32_vdivc_minmax_ukernel__scalar_x2(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    input_a += 2;

    float vacc0 = va0 / vb;
    float vacc1 = va1 / vb;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch == sizeof(float));
    const float va = *input_a;
    float vacc = va / vb;
    vacc = math_max_f32(vacc, voutput_min);
    vacc = math_min_f32(vacc, voutput_max);
    *output = vacc;
  }
}

void xnn_f32_vmax_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = math_max_f32(va0, vb0);
    float vacc1 = math_max_f32(va1, vb1);
    float vacc2 = math_max_f32(va2, vb2);
    float vacc3 = math_max_f32(va3, vb3);
    float vacc4 = math_max_f32(va4, vb4);
    float vacc5 = math_max_f32(va5, vb5);
    float vacc6 = math_max_f32(va6, vb6);
    float vacc7 = math_max_f32(va7, vb7);



    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = math_max_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmaxc_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = math_max_f32(va0, vb);
    float vacc1 = math_max_f32(va1, vb);
    float vacc2 = math_max_f32(va2, vb);
    float vacc3 = math_max_f32(va3, vb);
    float vacc4 = math_max_f32(va4, vb);
    float vacc5 = math_max_f32(va5, vb);
    float vacc6 = math_max_f32(va6, vb);
    float vacc7 = math_max_f32(va7, vb);



    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = math_max_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmin_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = math_min_f32(va0, vb0);
    float vacc1 = math_min_f32(va1, vb1);
    float vacc2 = math_min_f32(va2, vb2);
    float vacc3 = math_min_f32(va3, vb3);
    float vacc4 = math_min_f32(va4, vb4);
    float vacc5 = math_min_f32(va5, vb5);
    float vacc6 = math_min_f32(va6, vb6);
    float vacc7 = math_min_f32(va7, vb7);



    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = math_min_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vminc_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = math_min_f32(va0, vb);
    float vacc1 = math_min_f32(va1, vb);
    float vacc2 = math_min_f32(va2, vb);
    float vacc3 = math_min_f32(va3, vb);
    float vacc4 = math_min_f32(va4, vb);
    float vacc5 = math_min_f32(va5, vb);
    float vacc6 = math_min_f32(va6, vb);
    float vacc7 = math_min_f32(va7, vb);



    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = math_min_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmul_minmax_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 * vb0;
    float vacc1 = va1 * vb1;
    float vacc2 = va2 * vb2;
    float vacc3 = va3 * vb3;
    float vacc4 = va4 * vb4;
    float vacc5 = va5 * vb5;
    float vacc6 = va6 * vb6;
    float vacc7 = va7 * vb7;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va * vb;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmulc_minmax_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 * vb;
    float vacc1 = va1 * vb;
    float vacc2 = va2 * vb;
    float vacc3 = va3 * vb;
    float vacc4 = va4 * vb;
    float vacc5 = va5 * vb;
    float vacc6 = va6 * vb;
    float vacc7 = va7 * vb;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va * vb;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vrdivc_minmax_ukernel__scalar_x2(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    input_a += 2;

    float vacc0 = vb / va0;
    float vacc1 = vb / va1;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch == sizeof(float));
    const float va = *input_a;
    float vacc = vb / va;
    vacc = math_max_f32(vacc, voutput_min);
    vacc = math_min_f32(vacc, voutput_max);
    *output = vacc;
  }
}

void xnn_f32_vrsubc_minmax_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = vb - va0;
    float vacc1 = vb - va1;
    float vacc2 = vb - va2;
    float vacc3 = vb - va3;
    float vacc4 = vb - va4;
    float vacc5 = vb - va5;
    float vacc6 = vb - va6;
    float vacc7 = vb - va7;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = vb - va;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vsqrdiff_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 - vb0;
    float vacc1 = va1 - vb1;
    float vacc2 = va2 - vb2;
    float vacc3 = va3 - vb3;
    float vacc4 = va4 - vb4;
    float vacc5 = va5 - vb5;
    float vacc6 = va6 - vb6;
    float vacc7 = va7 - vb7;

    vacc0 = vacc0 * vacc0;
    vacc1 = vacc1 * vacc1;
    vacc2 = vacc2 * vacc2;
    vacc3 = vacc3 * vacc3;
    vacc4 = vacc4 * vacc4;
    vacc5 = vacc5 * vacc5;
    vacc6 = vacc6 * vacc6;
    vacc7 = vacc7 * vacc7;


    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va - vb;
      vacc = vacc * vacc;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vsqrdiffc_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 - vb;
    float vacc1 = va1 - vb;
    float vacc2 = va2 - vb;
    float vacc3 = va3 - vb;
    float vacc4 = va4 - vb;
    float vacc5 = va5 - vb;
    float vacc6 = va6 - vb;
    float vacc7 = va7 - vb;

    vacc0 = vacc0 * vacc0;
    vacc1 = vacc1 * vacc1;
    vacc2 = vacc2 * vacc2;
    vacc3 = vacc3 * vacc3;
    vacc4 = vacc4 * vacc4;
    vacc5 = vacc5 * vacc5;
    vacc6 = vacc6 * vacc6;
    vacc7 = vacc7 * vacc7;


    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va - vb;
      vacc = vacc * vacc;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vsub_minmax_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 - vb0;
    float vacc1 = va1 - vb1;
    float vacc2 = va2 - vb2;
    float vacc3 = va3 - vb3;
    float vacc4 = va4 - vb4;
    float vacc5 = va5 - vb5;
    float vacc6 = va6 - vb6;
    float vacc7 = va7 - vb7;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va - vb;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vsubc_minmax_ukernel__scalar_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 - vb;
    float vacc1 = va1 - vb;
    float vacc2 = va2 - vb;
    float vacc3 = va3 - vb;
    float vacc4 = va4 - vb;
    float vacc5 = va5 - vb;
    float vacc6 = va6 - vb;
    float vacc7 = va7 - vb;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va - vb;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vclamp_ukernel__scalar_x4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vacc0 = input[0];
    float vacc1 = input[1];
    float vacc2 = input[2];
    float vacc3 = input[3];
    input += 4;

    vacc0 = math_max_f32(vacc0, vy_min);
    vacc1 = math_max_f32(vacc1, vy_min);
    vacc2 = math_max_f32(vacc2, vy_min);
    vacc3 = math_max_f32(vacc3, vy_min);

    vacc0 = math_min_f32(vacc0, vy_max);
    vacc1 = math_min_f32(vacc1, vy_max);
    vacc2 = math_min_f32(vacc2, vy_max);
    vacc3 = math_min_f32(vacc3, vy_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vacc = *input++;
      vacc = math_max_f32(vacc, vy_min);
      vacc = math_min_f32(vacc, vy_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vprescale = params->scalar_rr2_lut16_p3.prescale;
  const float valpha = params->scalar_rr2_lut16_p3.alpha;
  const float vbeta = params->scalar_rr2_lut16_p3.beta;
  const float vmagic_bias = params->scalar_rr2_lut16_p3.magic_bias;
  const float vlog2e = params->scalar_rr2_lut16_p3.log2e;
  const uint32_t vindex_mask = UINT32_C(0xF);
  const float vsat_cutoff = params->scalar_rr2_lut16_p3.sat_cutoff;
  const float vminus_ln2_hi = params->scalar_rr2_lut16_p3.minus_ln2_hi;
  const float vminus_ln2_lo = params->scalar_rr2_lut16_p3.minus_ln2_lo;
  const float vc3 = params->scalar_rr2_lut16_p3.c3;
  const float vc2 = params->scalar_rr2_lut16_p3.c2;
  const float vone = params->scalar_rr2_lut16_p3.one;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    input += 4;

    const float vz0 = vx0 * vprescale;
    const float vz1 = vx1 * vprescale;
    const float vz2 = vx2 * vprescale;
    const float vz3 = vx3 * vprescale;

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;
    float vn2 = vz2 * vlog2e + vmagic_bias;
    float vn3 = vz3 * vlog2e + vmagic_bias;

    const uint32_t ven0 = float_as_uint32(vn0) << 19;
    const uint32_t vidx0 = float_as_uint32(vn0) & vindex_mask;
    vn0 -= vmagic_bias;
    const uint32_t ven1 = float_as_uint32(vn1) << 19;
    const uint32_t vidx1 = float_as_uint32(vn1) & vindex_mask;
    vn1 -= vmagic_bias;
    const uint32_t ven2 = float_as_uint32(vn2) << 19;
    const uint32_t vidx2 = float_as_uint32(vn2) & vindex_mask;
    vn2 -= vmagic_bias;
    const uint32_t ven3 = float_as_uint32(vn3) << 19;
    const uint32_t vidx3 = float_as_uint32(vn3) & vindex_mask;
    vn3 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vs0 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx0] + ven0);
    float vt1 = vn1 * vminus_ln2_hi + vz1;
    float vs1 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx1] + ven1);
    float vt2 = vn2 * vminus_ln2_hi + vz2;
    float vs2 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx2] + ven2);
    float vt3 = vn3 * vminus_ln2_hi + vz3;
    float vs3 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx3] + ven3);

    vt0 = vn0 * vminus_ln2_lo + vt0;
    if XNN_UNPREDICTABLE(vz0 <= vsat_cutoff) {
      vs0 = 0.0f;
      vt0 = 0.0f;
    }
    vt1 = vn1 * vminus_ln2_lo + vt1;
    if XNN_UNPREDICTABLE(vz1 <= vsat_cutoff) {
      vs1 = 0.0f;
      vt1 = 0.0f;
    }
    vt2 = vn2 * vminus_ln2_lo + vt2;
    if XNN_UNPREDICTABLE(vz2 <= vsat_cutoff) {
      vs2 = 0.0f;
      vt2 = 0.0f;
    }
    vt3 = vn3 * vminus_ln2_lo + vt3;
    if XNN_UNPREDICTABLE(vz3 <= vsat_cutoff) {
      vs3 = 0.0f;
      vt3 = 0.0f;
    }

    float vp0 = vc3 * vt0 + vc2;
    float vp1 = vc3 * vt1 + vc2;
    float vp2 = vc3 * vt2 + vc2;
    float vp3 = vc3 * vt3 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;
    vp2 *= vt2;
    vp3 *= vt3;

    vt0 *= vs0;
    vs0 -= vone;
    vt1 *= vs1;
    vs1 -= vone;
    vt2 *= vs2;
    vs2 -= vone;
    vt3 *= vs3;
    vs3 -= vone;

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;
    vp2 = vp2 * vt2 + vt2;
    vp3 = vp3 * vt3 + vt3;

    const float ve0 = (vp0 + vs0) * valpha;
    float vy0 = vx0 * vbeta;
    const float ve1 = (vp1 + vs1) * valpha;
    float vy1 = vx1 * vbeta;
    const float ve2 = (vp2 + vs2) * valpha;
    float vy2 = vx2 * vbeta;
    const float ve3 = (vp3 + vs3) * valpha;
    float vy3 = vx3 * vbeta;

    if XNN_UNPREDICTABLE(vx0 < 0.0f) {
      vy0 = ve0;
    }
    if XNN_UNPREDICTABLE(vx1 < 0.0f) {
      vy1 = ve1;
    }
    if XNN_UNPREDICTABLE(vx2 < 0.0f) {
      vy2 = ve2;
    }
    if XNN_UNPREDICTABLE(vx3 < 0.0f) {
      vy3 = ve3;
    }

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;

      const float vz = vx * vprescale;

      float vn = vz * vlog2e + vmagic_bias;
      const uint32_t ven = float_as_uint32(vn) << 19;
      const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
      vn -= vmagic_bias;

      float vt = vn * vminus_ln2_hi + vz;
      float vs = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx] + ven);

      vt = vn * vminus_ln2_lo + vt;
      if XNN_UNPREDICTABLE(vz <= vsat_cutoff) {
        vs = 0.0f;
        vt = 0.0f;
      }

      float vp = vc3 * vt + vc2;
      vp *= vt;

      vt *= vs;
      vs -= vone;
      vp = vp * vt + vt;
      const float ve = (vp + vs) * valpha;

      float vy = vx * vbeta;
      if XNN_UNPREDICTABLE(vx < 0.0f) {
        vy = ve;
      }

      *output++ = vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vhswish_ukernel__scalar_x4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsixth = params->scalar.sixth;
  const float vthree = params->scalar.three;
  const float vsix = params->scalar.six;
  const float vzero = 0.0f;
  assert(vthree == 3.0f);
  assert(vsix == 6.0f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    input += 4;

    float vacc0 = vx0 + vthree;
    vx0 *= vsixth;
    float vacc1 = vx1 + vthree;
    vx1 *= vsixth;
    float vacc2 = vx2 + vthree;
    vx2 *= vsixth;
    float vacc3 = vx3 + vthree;
    vx3 *= vsixth;

    vacc0 = math_max_f32(vacc0, vzero);
    vacc1 = math_max_f32(vacc1, vzero);
    vacc2 = math_max_f32(vacc2, vzero);
    vacc3 = math_max_f32(vacc3, vzero);

    vacc0 = math_min_f32(vacc0, vsix);
    vacc1 = math_min_f32(vacc1, vsix);
    vacc2 = math_min_f32(vacc2, vsix);
    vacc3 = math_min_f32(vacc3, vsix);

    vacc0 *= vx0;
    vacc1 *= vx1;
    vacc2 *= vx2;
    vacc3 *= vx3;

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;
      float vacc = vx + vthree;
      vx *= vsixth;
      vacc = math_max_f32(vacc, vzero);
      vacc = math_min_f32(vacc, vsix);
      vacc *= vx;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vlrelu_ukernel__scalar_x4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vslope = params->scalar.slope;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    float vacc0 = vx0 * vslope;
    float vacc1 = vx1 * vslope;
    float vacc2 = vx2 * vslope;
    float vacc3 = vx3 * vslope;

    vacc0 = XNN_UNPREDICTABLE(vx0 < 0.0f) ? vacc0 : vx0;
    vacc1 = XNN_UNPREDICTABLE(vx1 < 0.0f) ? vacc1 : vx1;
    vacc2 = XNN_UNPREDICTABLE(vx2 < 0.0f) ? vacc2 : vx2;
    vacc3 = XNN_UNPREDICTABLE(vx3 < 0.0f) ? vacc3 : vx3;

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      float vacc = vx * vslope;
      vacc = XNN_UNPREDICTABLE(vx < 0.0f) ? vacc : vx;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x(
    size_t rows,
    size_t channels,
    const float*restrict input,
    size_t input_stride,
    const float*restrict weights,
    float*restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    do {
      const float vscale = w[0];

      float vacc0 = *i0++;
      float vacc1 = *i1++;

      const float vbias = w[1];

      vacc0 = vacc0 * vscale + vbias;
      vacc1 = vacc1 * vscale + vbias;

      vacc0 = math_max_f32(vacc0, vmin);
      vacc1 = math_max_f32(vacc1, vmin);

      vacc0 = math_min_f32(vacc0, vmax);
      vacc1 = math_min_f32(vacc1, vmax);

      *o0++ = vacc0;
      *o1++ = vacc1;

      w += 2;
      c -= sizeof(float);
    } while (c != 0);
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_vrelu_ukernel__scalar_x8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t* i = (const uint32_t*) input;
  uint32_t* o = (uint32_t*) output;

  for (; batch >= 8 * sizeof(uint32_t); batch -= 8 * sizeof(uint32_t)) {
    uint32_t vacc0 = i[0];
    uint32_t vacc1 = i[1];
    uint32_t vacc2 = i[2];
    uint32_t vacc3 = i[3];
    uint32_t vacc4 = i[4];
    uint32_t vacc5 = i[5];
    uint32_t vacc6 = i[6];
    uint32_t vacc7 = i[7];
    i += 8;

    vacc0 = ((vacc0 >> 31) - 1) & vacc0;
    vacc1 = ((vacc1 >> 31) - 1) & vacc1;
    vacc2 = ((vacc2 >> 31) - 1) & vacc2;
    vacc3 = ((vacc3 >> 31) - 1) & vacc3;
    vacc4 = ((vacc4 >> 31) - 1) & vacc4;
    vacc5 = ((vacc5 >> 31) - 1) & vacc5;
    vacc6 = ((vacc6 >> 31) - 1) & vacc6;
    vacc7 = ((vacc7 >> 31) - 1) & vacc7;

    o[0] = vacc0;
    o[1] = vacc1;
    o[2] = vacc2;
    o[3] = vacc3;
    o[4] = vacc4;
    o[5] = vacc5;
    o[6] = vacc6;
    o[7] = vacc7;
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      uint32_t vacc = *i++;
      vacc =  ((vacc >> 31) - 1) & vacc;
      *o++ = vacc;
      batch -= sizeof(uint32_t);
    } while (batch != 0);
  }
}

void xnn_f32_vrndd_ukernel__scalar_libm_x1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  do {
    const float vx = *input++;
    const float vy = floorf(vx);
    *output++ = vy;
    batch -= sizeof(float);
  } while (batch != 0);
}

void xnn_f32_vrndne_ukernel__scalar_libm_x1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  do {
    const float vx = *input++;
    const float vy = nearbyintf(vx);
    *output++ = vy;
    batch -= sizeof(float);
  } while (batch != 0);
}

void xnn_f32_vrndu_ukernel__scalar_libm_x1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  do {
    const float vx = *input++;
    const float vy = ceilf(vx);
    *output++ = vy;
    batch -= sizeof(float);
  } while (batch != 0);
}

void xnn_f32_vrndz_ukernel__scalar_libm_x1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  do {
    const float vx = *input++;
    const float vy = truncf(vx);
    *output++ = vy;
    batch -= sizeof(float);
  } while (batch != 0);
}

extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_64[64];

void xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vmagic_bias = params->scalar_rr2_lut64_p2.magic_bias;
  const float vminus_log2e = params->scalar_rr2_lut64_p2.minus_log2e;
  const uint32_t vindex_mask = UINT32_C(0x3F);
  const float vln2_hi = params->scalar_rr2_lut64_p2.ln2_hi;
  const float vln2_lo = params->scalar_rr2_lut64_p2.ln2_lo;
  const float vc2 = params->scalar_rr2_lut64_p2.c2;
  const float vone = params->scalar_rr2_lut64_p2.one;
  const float vdenorm_cutoff = params->scalar_rr2_lut64_p2.denorm_cutoff;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    const float vz0 = fabsf(vx0);
    const float vz1 = fabsf(vx1);

    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;

    const uint32_t ve0 = float_as_uint32(vn0) << 17;
    const uint32_t ve1 = float_as_uint32(vn1) << 17;

    const uint32_t vidx0 = float_as_uint32(vn0) & vindex_mask;
    const float vs0 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx0] + ve0);
    const uint32_t vidx1 = float_as_uint32(vn1) & vindex_mask;
    const float vs1 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx1] + ve1);

    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;

    float vt0 = vn0 * vln2_hi + vz0;
    float vt1 = vn1 * vln2_hi + vz1;

    vt0 = vn0 * vln2_lo + vt0;
    vt1 = vn1 * vln2_lo + vt1;

    float vp0 = vt0 * vc2;
    float vp1 = vt1 * vc2;

    vp0 = vt0 - vp0 * vt0;
    vp1 = vt1 - vp1 * vt1;

    const float vy0 = vs0 - vs0 * vp0;
    const float vy1 = vs1 - vs1 * vp1;

    const float vd0 = vy0 + vone;
    const float vd1 = vy1 + vone;

    float vf0 = vy0 / vd0;
    float vf1 = vy1 / vd1;

    if XNN_UNPREDICTABLE(vz0 > vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz1 > vdenorm_cutoff) {
      vf1 = 0.0f;
    }

    if XNN_UNPREDICTABLE(vx0 > 0.0f) {
      vf0 = vone - vf0;
    }
    if XNN_UNPREDICTABLE(vx1 > 0.0f) {
      vf1 = vone - vf1;
    }

    output[0] = vf0;
    output[1] = vf1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;

    const float vz = fabsf(vx);

    float vn = vz * vminus_log2e + vmagic_bias;
    const uint32_t ve = float_as_uint32(vn) << 17;
    const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
    const float vs = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx] + ve);
    vn -= vmagic_bias;

    float vt = vn * vln2_hi + vz;
    vt = vn * vln2_lo + vt;

    float vp = vt * vc2;
    vp = vt - vp * vt;

    const float vy = vs - vs * vp;
    const float vd = vy + vone;

    float vf = vy / vd;
    if XNN_UNPREDICTABLE(vz > vdenorm_cutoff) {
      vf = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx > 0.0f) {
      vf = vone - vf;
    }

    *output = vf;
  }
}

void xnn_f32_vsqrt_ukernel__scalar_sqrt_x1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    const float vx = *input++;
    const float vy = sqrtf(vx);
    *output++ = vy;
  }
}

void xnn_f32_vabs_ukernel__scalar_x4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_abs_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    const float vy0 = fabsf(vx0);
    const float vy1 = fabsf(vx1);
    const float vy2 = fabsf(vx2);
    const float vy3 = fabsf(vx3);

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vy = fabsf(vx);
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vneg_ukernel__scalar_x4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_neg_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    const float vy0 = -vx0;
    const float vy1 = -vx1;
    const float vy2 = -vx2;
    const float vy3 = -vx3;

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vy = -vx;
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vsqr_ukernel__scalar_x4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    const float vy0 = vx0 * vx0;
    const float vy1 = vx1 * vx1;
    const float vy2 = vx2 * vx2;
    const float vy3 = vx3 * vx3;

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vy = vx * vx;
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_25p2c__scalar_lrintf(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
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
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      const int32_t vi9x0 = (int32_t) i9[0];
      const int32_t vi9x1 = (int32_t) i9[1];
      i9 += 2;

      const int32_t vk9x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      const int32_t vk9x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[19];

      vacc0 += vi9x0 * vk9x0;
      vacc1 += vi9x1 * vk9x1;

      const int32_t vi10x0 = (int32_t) i10[0];
      const int32_t vi10x1 = (int32_t) i10[1];
      i10 += 2;

      const int32_t vk10x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      const int32_t vk10x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[21];

      vacc0 += vi10x0 * vk10x0;
      vacc1 += vi10x1 * vk10x1;

      const int32_t vi11x0 = (int32_t) i11[0];
      const int32_t vi11x1 = (int32_t) i11[1];
      i11 += 2;

      const int32_t vk11x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      const int32_t vk11x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[23];

      vacc0 += vi11x0 * vk11x0;
      vacc1 += vi11x1 * vk11x1;

      const int32_t vi12x0 = (int32_t) i12[0];
      const int32_t vi12x1 = (int32_t) i12[1];
      i12 += 2;

      const int32_t vk12x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      const int32_t vk12x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[25];

      vacc0 += vi12x0 * vk12x0;
      vacc1 += vi12x1 * vk12x1;

      const int32_t vi13x0 = (int32_t) i13[0];
      const int32_t vi13x1 = (int32_t) i13[1];
      i13 += 2;

      const int32_t vk13x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      const int32_t vk13x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[27];

      vacc0 += vi13x0 * vk13x0;
      vacc1 += vi13x1 * vk13x1;

      const int32_t vi14x0 = (int32_t) i14[0];
      const int32_t vi14x1 = (int32_t) i14[1];
      i14 += 2;

      const int32_t vk14x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      const int32_t vk14x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[29];

      vacc0 += vi14x0 * vk14x0;
      vacc1 += vi14x1 * vk14x1;

      const int32_t vi15x0 = (int32_t) i15[0];
      const int32_t vi15x1 = (int32_t) i15[1];
      i15 += 2;

      const int32_t vk15x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      const int32_t vk15x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[31];

      vacc0 += vi15x0 * vk15x0;
      vacc1 += vi15x1 * vk15x1;

      const int32_t vi16x0 = (int32_t) i16[0];
      const int32_t vi16x1 = (int32_t) i16[1];
      i16 += 2;

      const int32_t vk16x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      const int32_t vk16x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[33];

      vacc0 += vi16x0 * vk16x0;
      vacc1 += vi16x1 * vk16x1;

      const int32_t vi17x0 = (int32_t) i17[0];
      const int32_t vi17x1 = (int32_t) i17[1];
      i17 += 2;

      const int32_t vk17x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      const int32_t vk17x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[35];

      vacc0 += vi17x0 * vk17x0;
      vacc1 += vi17x1 * vk17x1;

      const int32_t vi18x0 = (int32_t) i18[0];
      const int32_t vi18x1 = (int32_t) i18[1];
      i18 += 2;

      const int32_t vk18x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      const int32_t vk18x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[37];

      vacc0 += vi18x0 * vk18x0;
      vacc1 += vi18x1 * vk18x1;

      const int32_t vi19x0 = (int32_t) i19[0];
      const int32_t vi19x1 = (int32_t) i19[1];
      i19 += 2;

      const int32_t vk19x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      const int32_t vk19x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[39];

      vacc0 += vi19x0 * vk19x0;
      vacc1 += vi19x1 * vk19x1;

      const int32_t vi20x0 = (int32_t) i20[0];
      const int32_t vi20x1 = (int32_t) i20[1];
      i20 += 2;

      const int32_t vk20x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      const int32_t vk20x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[41];

      vacc0 += vi20x0 * vk20x0;
      vacc1 += vi20x1 * vk20x1;

      const int32_t vi21x0 = (int32_t) i21[0];
      const int32_t vi21x1 = (int32_t) i21[1];
      i21 += 2;

      const int32_t vk21x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      const int32_t vk21x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[43];

      vacc0 += vi21x0 * vk21x0;
      vacc1 += vi21x1 * vk21x1;

      const int32_t vi22x0 = (int32_t) i22[0];
      const int32_t vi22x1 = (int32_t) i22[1];
      i22 += 2;

      const int32_t vk22x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      const int32_t vk22x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[45];

      vacc0 += vi22x0 * vk22x0;
      vacc1 += vi22x1 * vk22x1;

      const int32_t vi23x0 = (int32_t) i23[0];
      const int32_t vi23x1 = (int32_t) i23[1];
      i23 += 2;

      const int32_t vk23x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      const int32_t vk23x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[47];

      vacc0 += vi23x0 * vk23x0;
      vacc1 += vi23x1 * vk23x1;

      const int32_t vi24x0 = (int32_t) i24[0];
      const int32_t vi24x1 = (int32_t) i24[1];
      i24 += 2;

      const int32_t vk24x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      const int32_t vk24x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[49];

      vacc0 += vi24x0 * vk24x0;
      vacc1 += vi24x1 * vk24x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      const float vscale1 = unaligned_indexed_load_f32(w, 1);
      w = (const void*) ((const float*) w + 2);

      vfpacc0 *= vscale0;
      vfpacc1 *= vscale1;

      vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

      const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
      const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

      int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
      int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;
      const int32_t vi9 = (int32_t) *i9;
      const int32_t vk9 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      vacc += vi9 * vk9;
      const int32_t vi10 = (int32_t) *i10;
      const int32_t vk10 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      vacc += vi10 * vk10;
      const int32_t vi11 = (int32_t) *i11;
      const int32_t vk11 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      vacc += vi11 * vk11;
      const int32_t vi12 = (int32_t) *i12;
      const int32_t vk12 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      vacc += vi12 * vk12;
      const int32_t vi13 = (int32_t) *i13;
      const int32_t vk13 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      vacc += vi13 * vk13;
      const int32_t vi14 = (int32_t) *i14;
      const int32_t vk14 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      vacc += vi14 * vk14;
      const int32_t vi15 = (int32_t) *i15;
      const int32_t vk15 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      vacc += vi15 * vk15;
      const int32_t vi16 = (int32_t) *i16;
      const int32_t vk16 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      vacc += vi16 * vk16;
      const int32_t vi17 = (int32_t) *i17;
      const int32_t vk17 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      vacc += vi17 * vk17;
      const int32_t vi18 = (int32_t) *i18;
      const int32_t vk18 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      vacc += vi18 * vk18;
      const int32_t vi19 = (int32_t) *i19;
      const int32_t vk19 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      vacc += vi19 * vk19;
      const int32_t vi20 = (int32_t) *i20;
      const int32_t vk20 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      vacc += vi20 * vk20;
      const int32_t vi21 = (int32_t) *i21;
      const int32_t vk21 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      vacc += vi21 * vk21;
      const int32_t vi22 = (int32_t) *i22;
      const int32_t vk22 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      vacc += vi22 * vk22;
      const int32_t vi23 = (int32_t) *i23;
      const int32_t vk23 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      vacc += vi23 * vk23;
      const int32_t vi24 = (int32_t) *i24;
      const int32_t vk24 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      vacc += vi24 * vk24;

      const float vscale = unaligned_load_f32((const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(int8_t)));
      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      const int32_t vrndacc = (int32_t) lrintf(vfpacc);
      int32_t vout = vrndacc + voutput_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_3p2c__scalar_lrintf(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
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
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 6 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      const float vscale1 = unaligned_indexed_load_f32(w, 1);
      w = (const void*) ((const float*) w + 2);

      vfpacc0 *= vscale0;
      vfpacc1 *= vscale1;

      vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

      const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
      const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

      int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
      int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;

      const float vscale = unaligned_load_f32((const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 6 * sizeof(int8_t)));
      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      const int32_t vrndacc = (int32_t) lrintf(vfpacc);
      int32_t vout = vrndacc + voutput_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
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
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 18 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      const float vscale1 = unaligned_indexed_load_f32(w, 1);
      w = (const void*) ((const float*) w + 2);

      vfpacc0 *= vscale0;
      vfpacc1 *= vscale1;

      vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

      const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
      const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

      int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
      int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;

      const float vscale = unaligned_load_f32((const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 18 * sizeof(int8_t)));
      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      const int32_t vrndacc = (int32_t) lrintf(vfpacc);
      int32_t vout = vrndacc + voutput_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const void*) ((const int8_t*) w + 4);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const void*) ((const int8_t*) w + 4);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    vfpacc1x0 *= vscale0;
    vfpacc2x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    vfpacc1x1 *= vscale1;
    vfpacc2x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    vfpacc1x2 *= vscale2;
    vfpacc2x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    vfpacc1x3 *= vscale3;
    vfpacc2x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = math_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = math_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = math_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = math_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = math_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = math_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = math_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = math_max_f32(vfpacc2x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = math_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = math_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = math_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = math_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = math_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = math_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = math_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = math_min_f32(vfpacc2x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);
    const int32_t vrndacc1x0 = (int32_t) lrintf(vfpacc1x0);
    const int32_t vrndacc1x1 = (int32_t) lrintf(vfpacc1x1);
    const int32_t vrndacc1x2 = (int32_t) lrintf(vfpacc1x2);
    const int32_t vrndacc1x3 = (int32_t) lrintf(vfpacc1x3);
    const int32_t vrndacc2x0 = (int32_t) lrintf(vfpacc2x0);
    const int32_t vrndacc2x1 = (int32_t) lrintf(vfpacc2x1);
    const int32_t vrndacc2x2 = (int32_t) lrintf(vfpacc2x2);
    const int32_t vrndacc2x3 = (int32_t) lrintf(vfpacc2x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;
    int32_t vout1x0 = vrndacc1x0 + voutput_zero_point;
    int32_t vout1x1 = vrndacc1x1 + voutput_zero_point;
    int32_t vout1x2 = vrndacc1x2 + voutput_zero_point;
    int32_t vout1x3 = vrndacc1x3 + voutput_zero_point;
    int32_t vout2x0 = vrndacc2x0 + voutput_zero_point;
    int32_t vout2x1 = vrndacc2x1 + voutput_zero_point;
    int32_t vout2x2 = vrndacc2x2 + voutput_zero_point;
    int32_t vout2x3 = vrndacc2x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
        c1[0] = (int8_t) vout1x0;
        c2[0] = (int8_t) vout2x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t**restrict a,
    const void*restrict w,
    int8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qc8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t**restrict a,
    const void*restrict w,
    int8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;
        const int32_t va1 = (int32_t) *a1++;
        const int32_t va2 = (int32_t) *a2++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc1x2 += va1 * vb2;
        vacc1x3 += va1 * vb3;
        vacc2x0 += va2 * vb0;
        vacc2x1 += va2 * vb1;
        vacc2x2 += va2 * vb2;
        vacc2x3 += va2 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 3 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    vfpacc1x0 *= vscale0;
    vfpacc2x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    vfpacc1x1 *= vscale1;
    vfpacc2x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    vfpacc1x2 *= vscale2;
    vfpacc2x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    vfpacc1x3 *= vscale3;
    vfpacc2x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = math_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = math_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = math_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = math_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = math_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = math_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = math_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = math_max_f32(vfpacc2x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = math_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = math_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = math_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = math_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = math_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = math_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = math_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = math_min_f32(vfpacc2x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);
    const int32_t vrndacc1x0 = (int32_t) lrintf(vfpacc1x0);
    const int32_t vrndacc1x1 = (int32_t) lrintf(vfpacc1x1);
    const int32_t vrndacc1x2 = (int32_t) lrintf(vfpacc1x2);
    const int32_t vrndacc1x3 = (int32_t) lrintf(vfpacc1x3);
    const int32_t vrndacc2x0 = (int32_t) lrintf(vfpacc2x0);
    const int32_t vrndacc2x1 = (int32_t) lrintf(vfpacc2x1);
    const int32_t vrndacc2x2 = (int32_t) lrintf(vfpacc2x2);
    const int32_t vrndacc2x3 = (int32_t) lrintf(vfpacc2x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;
    int32_t vout1x0 = vrndacc1x0 + voutput_zero_point;
    int32_t vout1x1 = vrndacc1x1 + voutput_zero_point;
    int32_t vout1x2 = vrndacc1x2 + voutput_zero_point;
    int32_t vout1x3 = vrndacc1x3 + voutput_zero_point;
    int32_t vout2x0 = vrndacc2x0 + voutput_zero_point;
    int32_t vout2x1 = vrndacc2x1 + voutput_zero_point;
    int32_t vout2x2 = vrndacc2x2 + voutput_zero_point;
    int32_t vout2x3 = vrndacc2x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c2[0] = (int8_t) vout2x0;
        c1[0] = (int8_t) vout1x0;
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_25p2c__scalar_lrintf(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
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
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      const int32_t vi9x0 = (int32_t) i9[0];
      const int32_t vi9x1 = (int32_t) i9[1];
      i9 += 2;

      const int32_t vk9x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      const int32_t vk9x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[19];

      vacc0 += vi9x0 * vk9x0;
      vacc1 += vi9x1 * vk9x1;

      const int32_t vi10x0 = (int32_t) i10[0];
      const int32_t vi10x1 = (int32_t) i10[1];
      i10 += 2;

      const int32_t vk10x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      const int32_t vk10x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[21];

      vacc0 += vi10x0 * vk10x0;
      vacc1 += vi10x1 * vk10x1;

      const int32_t vi11x0 = (int32_t) i11[0];
      const int32_t vi11x1 = (int32_t) i11[1];
      i11 += 2;

      const int32_t vk11x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      const int32_t vk11x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[23];

      vacc0 += vi11x0 * vk11x0;
      vacc1 += vi11x1 * vk11x1;

      const int32_t vi12x0 = (int32_t) i12[0];
      const int32_t vi12x1 = (int32_t) i12[1];
      i12 += 2;

      const int32_t vk12x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      const int32_t vk12x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[25];

      vacc0 += vi12x0 * vk12x0;
      vacc1 += vi12x1 * vk12x1;

      const int32_t vi13x0 = (int32_t) i13[0];
      const int32_t vi13x1 = (int32_t) i13[1];
      i13 += 2;

      const int32_t vk13x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      const int32_t vk13x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[27];

      vacc0 += vi13x0 * vk13x0;
      vacc1 += vi13x1 * vk13x1;

      const int32_t vi14x0 = (int32_t) i14[0];
      const int32_t vi14x1 = (int32_t) i14[1];
      i14 += 2;

      const int32_t vk14x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      const int32_t vk14x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[29];

      vacc0 += vi14x0 * vk14x0;
      vacc1 += vi14x1 * vk14x1;

      const int32_t vi15x0 = (int32_t) i15[0];
      const int32_t vi15x1 = (int32_t) i15[1];
      i15 += 2;

      const int32_t vk15x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      const int32_t vk15x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[31];

      vacc0 += vi15x0 * vk15x0;
      vacc1 += vi15x1 * vk15x1;

      const int32_t vi16x0 = (int32_t) i16[0];
      const int32_t vi16x1 = (int32_t) i16[1];
      i16 += 2;

      const int32_t vk16x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      const int32_t vk16x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[33];

      vacc0 += vi16x0 * vk16x0;
      vacc1 += vi16x1 * vk16x1;

      const int32_t vi17x0 = (int32_t) i17[0];
      const int32_t vi17x1 = (int32_t) i17[1];
      i17 += 2;

      const int32_t vk17x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      const int32_t vk17x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[35];

      vacc0 += vi17x0 * vk17x0;
      vacc1 += vi17x1 * vk17x1;

      const int32_t vi18x0 = (int32_t) i18[0];
      const int32_t vi18x1 = (int32_t) i18[1];
      i18 += 2;

      const int32_t vk18x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      const int32_t vk18x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[37];

      vacc0 += vi18x0 * vk18x0;
      vacc1 += vi18x1 * vk18x1;

      const int32_t vi19x0 = (int32_t) i19[0];
      const int32_t vi19x1 = (int32_t) i19[1];
      i19 += 2;

      const int32_t vk19x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      const int32_t vk19x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[39];

      vacc0 += vi19x0 * vk19x0;
      vacc1 += vi19x1 * vk19x1;

      const int32_t vi20x0 = (int32_t) i20[0];
      const int32_t vi20x1 = (int32_t) i20[1];
      i20 += 2;

      const int32_t vk20x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      const int32_t vk20x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[41];

      vacc0 += vi20x0 * vk20x0;
      vacc1 += vi20x1 * vk20x1;

      const int32_t vi21x0 = (int32_t) i21[0];
      const int32_t vi21x1 = (int32_t) i21[1];
      i21 += 2;

      const int32_t vk21x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      const int32_t vk21x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[43];

      vacc0 += vi21x0 * vk21x0;
      vacc1 += vi21x1 * vk21x1;

      const int32_t vi22x0 = (int32_t) i22[0];
      const int32_t vi22x1 = (int32_t) i22[1];
      i22 += 2;

      const int32_t vk22x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      const int32_t vk22x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[45];

      vacc0 += vi22x0 * vk22x0;
      vacc1 += vi22x1 * vk22x1;

      const int32_t vi23x0 = (int32_t) i23[0];
      const int32_t vi23x1 = (int32_t) i23[1];
      i23 += 2;

      const int32_t vk23x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      const int32_t vk23x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[47];

      vacc0 += vi23x0 * vk23x0;
      vacc1 += vi23x1 * vk23x1;

      const int32_t vi24x0 = (int32_t) i24[0];
      const int32_t vi24x1 = (int32_t) i24[1];
      i24 += 2;

      const int32_t vk24x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      const int32_t vk24x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[49];

      vacc0 += vi24x0 * vk24x0;
      vacc1 += vi24x1 * vk24x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      vfpacc0 *= vscale;
      vfpacc1 *= vscale;

      vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

      const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
      const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

      int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
      int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;
      const int32_t vi9 = (int32_t) *i9;
      const int32_t vk9 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      vacc += vi9 * vk9;
      const int32_t vi10 = (int32_t) *i10;
      const int32_t vk10 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      vacc += vi10 * vk10;
      const int32_t vi11 = (int32_t) *i11;
      const int32_t vk11 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      vacc += vi11 * vk11;
      const int32_t vi12 = (int32_t) *i12;
      const int32_t vk12 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      vacc += vi12 * vk12;
      const int32_t vi13 = (int32_t) *i13;
      const int32_t vk13 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      vacc += vi13 * vk13;
      const int32_t vi14 = (int32_t) *i14;
      const int32_t vk14 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      vacc += vi14 * vk14;
      const int32_t vi15 = (int32_t) *i15;
      const int32_t vk15 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      vacc += vi15 * vk15;
      const int32_t vi16 = (int32_t) *i16;
      const int32_t vk16 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      vacc += vi16 * vk16;
      const int32_t vi17 = (int32_t) *i17;
      const int32_t vk17 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      vacc += vi17 * vk17;
      const int32_t vi18 = (int32_t) *i18;
      const int32_t vk18 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      vacc += vi18 * vk18;
      const int32_t vi19 = (int32_t) *i19;
      const int32_t vk19 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      vacc += vi19 * vk19;
      const int32_t vi20 = (int32_t) *i20;
      const int32_t vk20 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      vacc += vi20 * vk20;
      const int32_t vi21 = (int32_t) *i21;
      const int32_t vk21 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      vacc += vi21 * vk21;
      const int32_t vi22 = (int32_t) *i22;
      const int32_t vk22 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      vacc += vi22 * vk22;
      const int32_t vi23 = (int32_t) *i23;
      const int32_t vk23 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      vacc += vi23 * vk23;
      const int32_t vi24 = (int32_t) *i24;
      const int32_t vk24 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      vacc += vi24 * vk24;

      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      const int32_t vrndacc = (int32_t) lrintf(vfpacc);
      int32_t vout = vrndacc + voutput_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
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
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 18 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      vfpacc0 *= vscale;
      vfpacc1 *= vscale;

      vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

      const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
      const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

      int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
      int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;

      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      const int32_t vrndacc = (int32_t) lrintf(vfpacc);
      int32_t vout = vrndacc + voutput_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_f32_vcvt_ukernel__scalar_x4(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vzero_point = params->scalar.zero_point;
  const float vscale = params->scalar.scale;

  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    int32_t vx0 = (int32_t) input[0];
    int32_t vx1 = (int32_t) input[1];
    int32_t vx2 = (int32_t) input[2];
    int32_t vx3 = (int32_t) input[3];
    input += 4;

    vx0 -= vzero_point;
    vx1 -= vzero_point;
    vx2 -= vzero_point;
    vx3 -= vzero_point;

    float vy0 = (float) vx0;
    float vy1 = (float) vx1;
    float vy2 = (float) vx2;
    float vy3 = (float) vx3;

    vy0 *= vscale;
    vy1 *= vscale;
    vy2 *= vscale;
    vy3 *= vscale;

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      int32_t vx = *input++;
      vx -= vzero_point;

      float vy = (float) vx;
      vy *= vscale;
      *output++ = vy;

      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}

void xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 1) * sizeof(int8_t);

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  int32_t* b = buffer;
  size_t c = channels;
  do {
    int32_t vacc = vinit_bias;
    const int32_t vi0 = (int32_t) *i0++;
    const int32_t vi1 = (int32_t) *i1++;

    vacc += vi0;
    const int32_t vi2 = (int32_t) *i2++;
    vacc += vi1;
    const int32_t vi3 = (int32_t) *i3++;
    vacc += vi2;
    const int32_t vi4 = (int32_t) *i4++;
    vacc += vi3;
    const int32_t vi5 = (int32_t) *i5++;
    vacc += vi4;
    const int32_t vi6 = (int32_t) *i6++;

    vacc += vi5;
    vacc += vi6;

    *b++ = vacc;
  } while (--c != 0);

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    do {
      int32_t vacc = *b;
      const int32_t vi0 = (int32_t) *i0++;
      const int32_t vi1 = (int32_t) *i1++;

      vacc += vi0;
      const int32_t vi2 = (int32_t) *i2++;
      vacc += vi1;
      const int32_t vi3 = (int32_t) *i3++;
      vacc += vi2;
      const int32_t vi4 = (int32_t) *i4++;
      vacc += vi3;
      const int32_t vi5 = (int32_t) *i5++;
      vacc += vi4;
      const int32_t vi6 = (int32_t) *i6++;

      vacc += vi5;
      vacc += vi6;

      *b++ = vacc;
    } while (--c != 0);
  }

  i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  do {
    int32_t vacc = *buffer++;
    const int32_t vi0 = (int32_t) *i0++;
    const int32_t vi1 = (int32_t) *i1++;

    vacc += vi0;
    const int32_t vi2 = (int32_t) *i2++;
    vacc += vi1;
    const int32_t vi3 = (int32_t) *i3++;
    vacc += vi2;
    const int32_t vi4 = (int32_t) *i4++;
    vacc += vi3;
    const int32_t vi5 = (int32_t) *i5++;
    vacc += vi4;
    const int32_t vi6 = (int32_t) *i6++;

    vacc += vi5;
    vacc += vi6;

    float vfpacc = (float) vacc * vscale;
    vfpacc += vmagic_bias;
    int32_t vout = (int32_t) float_as_uint32(vfpacc);
    vout = math_max_s32(vout, vmagic_min);
    vout = math_min_s32(vout, vmagic_max);
    vout -= vmagic_bias_less_zero_point;

    *output++ = (int8_t) vout;
  } while (--channels != 0);
}

void xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  do {
    int32_t vacc = vinit_bias;
    const int32_t vi0 = (int32_t) *i0++;
    const int32_t vi1 = (int32_t) *i1++;

    vacc += vi0;
    const int32_t vi2 = (int32_t) *i2++;
    vacc += vi1;
    const int32_t vi3 = (int32_t) *i3++;
    vacc += vi2;
    const int32_t vi4 = (int32_t) *i4++;
    vacc += vi3;
    const int32_t vi5 = (int32_t) *i5++;
    vacc += vi4;
    const int32_t vi6 = (int32_t) *i6++;

    vacc += vi5;
    vacc += vi6;

    float vfpacc = (float) vacc * vscale;
    vfpacc += vmagic_bias;
    int32_t vout = (int32_t) float_as_uint32(vfpacc);
    vout = math_max_s32(vout, vmagic_min);
    vout = math_min_s32(vout, vmagic_max);
    vout -= vmagic_bias_less_zero_point;

    *output++ = (int8_t) vout;
  } while (--channels != 0);
}

void xnn_qs8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const void*) ((const int8_t*) w + 4);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale = params->fp32_scalar_lrintf.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const void*) ((const int8_t*) w + 4);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;

    const float vscale = params->fp32_scalar_lrintf.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;
    vfpacc2x0 *= vscale;
    vfpacc2x1 *= vscale;
    vfpacc2x2 *= vscale;
    vfpacc2x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = math_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = math_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = math_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = math_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = math_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = math_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = math_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = math_max_f32(vfpacc2x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = math_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = math_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = math_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = math_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = math_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = math_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = math_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = math_min_f32(vfpacc2x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);
    const int32_t vrndacc1x0 = (int32_t) lrintf(vfpacc1x0);
    const int32_t vrndacc1x1 = (int32_t) lrintf(vfpacc1x1);
    const int32_t vrndacc1x2 = (int32_t) lrintf(vfpacc1x2);
    const int32_t vrndacc1x3 = (int32_t) lrintf(vfpacc1x3);
    const int32_t vrndacc2x0 = (int32_t) lrintf(vfpacc2x0);
    const int32_t vrndacc2x1 = (int32_t) lrintf(vfpacc2x1);
    const int32_t vrndacc2x2 = (int32_t) lrintf(vfpacc2x2);
    const int32_t vrndacc2x3 = (int32_t) lrintf(vfpacc2x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;
    int32_t vout1x0 = vrndacc1x0 + voutput_zero_point;
    int32_t vout1x1 = vrndacc1x1 + voutput_zero_point;
    int32_t vout1x2 = vrndacc1x2 + voutput_zero_point;
    int32_t vout1x3 = vrndacc1x3 + voutput_zero_point;
    int32_t vout2x0 = vrndacc2x0 + voutput_zero_point;
    int32_t vout2x1 = vrndacc2x1 + voutput_zero_point;
    int32_t vout2x2 = vrndacc2x2 + voutput_zero_point;
    int32_t vout2x3 = vrndacc2x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
        c1[0] = (int8_t) vout1x0;
        c2[0] = (int8_t) vout2x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t**restrict a,
    const void*restrict w,
    int8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale = params->fp32_scalar_lrintf.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t**restrict a,
    const void*restrict w,
    int8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;
        const int32_t va1 = (int32_t) *a1++;
        const int32_t va2 = (int32_t) *a2++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc1x2 += va1 * vb2;
        vacc1x3 += va1 * vb3;
        vacc2x0 += va2 * vb0;
        vacc2x1 += va2 * vb1;
        vacc2x2 += va2 * vb2;
        vacc2x3 += va2 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 3 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;

    const float vscale = params->fp32_scalar_lrintf.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;
    vfpacc2x0 *= vscale;
    vfpacc2x1 *= vscale;
    vfpacc2x2 *= vscale;
    vfpacc2x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = math_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = math_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = math_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = math_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = math_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = math_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = math_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = math_max_f32(vfpacc2x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = math_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = math_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = math_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = math_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = math_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = math_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = math_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = math_min_f32(vfpacc2x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);
    const int32_t vrndacc1x0 = (int32_t) lrintf(vfpacc1x0);
    const int32_t vrndacc1x1 = (int32_t) lrintf(vfpacc1x1);
    const int32_t vrndacc1x2 = (int32_t) lrintf(vfpacc1x2);
    const int32_t vrndacc1x3 = (int32_t) lrintf(vfpacc1x3);
    const int32_t vrndacc2x0 = (int32_t) lrintf(vfpacc2x0);
    const int32_t vrndacc2x1 = (int32_t) lrintf(vfpacc2x1);
    const int32_t vrndacc2x2 = (int32_t) lrintf(vfpacc2x2);
    const int32_t vrndacc2x3 = (int32_t) lrintf(vfpacc2x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;
    int32_t vout1x0 = vrndacc1x0 + voutput_zero_point;
    int32_t vout1x1 = vrndacc1x1 + voutput_zero_point;
    int32_t vout1x2 = vrndacc1x2 + voutput_zero_point;
    int32_t vout1x3 = vrndacc1x3 + voutput_zero_point;
    int32_t vout2x0 = vrndacc2x0 + voutput_zero_point;
    int32_t vout2x1 = vrndacc2x1 + voutput_zero_point;
    int32_t vout2x2 = vrndacc2x2 + voutput_zero_point;
    int32_t vout2x3 = vrndacc2x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c2[0] = (int8_t) vout2x0;
        c1[0] = (int8_t) vout1x0;
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vadd_minmax_ukernel__scalar_x4(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t vbias = params->scalar.bias;
  const int32_t va_multiplier = params->scalar.a_multiplier;
  const int32_t vb_multiplier = params->scalar.b_multiplier;
  const uint32_t vshift = params->scalar.shift;
  const int32_t voutput_min_less_zero_point = params->scalar.output_min_less_zero_point;
  const int32_t voutput_max_less_zero_point = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;

  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0];
    const int32_t va1 = input_a[1];
    const int32_t va2 = input_a[2];
    const int32_t va3 = input_a[3];
    input_a += 4;

    const int32_t vb0 = input_b[0];
    int32_t vacc0 = vbias + va0 * va_multiplier;
    const int32_t vb1 = input_b[1];
    int32_t vacc1 = vbias + va1 * va_multiplier;
    const int32_t vb2 = input_b[2];
    int32_t vacc2 = vbias + va2 * va_multiplier;
    const int32_t vb3 = input_b[3];
    int32_t vacc3 = vbias + va3 * va_multiplier;
    input_b += 4;

    vacc0 += vb0 * vb_multiplier;
    vacc1 += vb1 * vb_multiplier;
    vacc2 += vb2 * vb_multiplier;
    vacc3 += vb3 * vb_multiplier;

    int32_t vout0 = math_asr_s32(vacc0, vshift);
    int32_t vout1 = math_asr_s32(vacc1, vshift);
    int32_t vout2 = math_asr_s32(vacc2, vshift);
    int32_t vout3 = math_asr_s32(vacc3, vshift);

    vout0 = math_max_s32(vout0, voutput_min_less_zero_point);
    vout1 = math_max_s32(vout1, voutput_min_less_zero_point);
    vout2 = math_max_s32(vout2, voutput_min_less_zero_point);
    vout3 = math_max_s32(vout3, voutput_min_less_zero_point);

    vout0 = math_min_s32(vout0, voutput_max_less_zero_point);
    vout1 = math_min_s32(vout1, voutput_max_less_zero_point);
    vout2 = math_min_s32(vout2, voutput_max_less_zero_point);
    vout3 = math_min_s32(vout3, voutput_max_less_zero_point);

    vout0 += voutput_zero_point;
    vout1 += voutput_zero_point;
    vout2 += voutput_zero_point;
    vout3 += voutput_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = *input_a++;
      const int32_t vb = *input_b++;
      const int32_t vacc = vbias + va * va_multiplier + vb * vb_multiplier;

      int32_t vout = math_asr_s32(vacc, vshift);
      vout = math_max_s32(vout, voutput_min_less_zero_point);
      vout = math_min_s32(vout, voutput_max_less_zero_point);
      *output++ = (int8_t) (vout + voutput_zero_point);

      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}

void xnn_qs8_vaddc_minmax_ukernel__scalar_x4(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t vbias = params->scalar.bias + (int32_t) *input_b * params->scalar.b_multiplier;
  const int32_t va_multiplier = params->scalar.a_multiplier;
  const uint32_t vshift = params->scalar.shift;
  const int32_t voutput_min_less_zero_point = params->scalar.output_min_less_zero_point;
  const int32_t voutput_max_less_zero_point = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;

  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0];
    const int32_t va1 = input_a[1];
    const int32_t va2 = input_a[2];
    const int32_t va3 = input_a[3];
    input_a += 4;

    const int32_t vacc0 = vbias + va0 * va_multiplier;
    const int32_t vacc1 = vbias + va1 * va_multiplier;
    const int32_t vacc2 = vbias + va2 * va_multiplier;
    const int32_t vacc3 = vbias + va3 * va_multiplier;
    input_b += 4;

    int32_t vout0 = math_asr_s32(vacc0, vshift);
    int32_t vout1 = math_asr_s32(vacc1, vshift);
    int32_t vout2 = math_asr_s32(vacc2, vshift);
    int32_t vout3 = math_asr_s32(vacc3, vshift);

    vout0 = math_max_s32(vout0, voutput_min_less_zero_point);
    vout1 = math_max_s32(vout1, voutput_min_less_zero_point);
    vout2 = math_max_s32(vout2, voutput_min_less_zero_point);
    vout3 = math_max_s32(vout3, voutput_min_less_zero_point);

    vout0 = math_min_s32(vout0, voutput_max_less_zero_point);
    vout1 = math_min_s32(vout1, voutput_max_less_zero_point);
    vout2 = math_min_s32(vout2, voutput_max_less_zero_point);
    vout3 = math_min_s32(vout3, voutput_max_less_zero_point);

    vout0 += voutput_zero_point;
    vout1 += voutput_zero_point;
    vout2 += voutput_zero_point;
    vout3 += voutput_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = *input_a++;
      const int32_t vacc = vbias + va * va_multiplier;

      int32_t vout = math_asr_s32(vacc, vshift);
      vout = math_max_s32(vout, voutput_min_less_zero_point);
      vout = math_min_s32(vout, voutput_max_less_zero_point);
      *output++ = (int8_t) (vout + voutput_zero_point);

      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}

void xnn_qs8_vcvt_ukernel__scalar_x4(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vbias = params->scalar.bias;
  const int32_t vmultiplier = params->scalar.multiplier;
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    int32_t vacc0 = input[0];
    int32_t vacc1 = input[1];
    int32_t vacc2 = input[2];
    int32_t vacc3 = input[3];
    input += 4;

    vacc0 = vbias + vacc0 * vmultiplier;
    vacc1 = vbias + vacc1 * vmultiplier;
    vacc2 = vbias + vacc2 * vmultiplier;
    vacc3 = vbias + vacc3 * vmultiplier;

    int32_t vout0 = math_asr_s32(vacc0, 8);
    int32_t vout1 = math_asr_s32(vacc1, 8);
    int32_t vout2 = math_asr_s32(vacc2, 8);
    int32_t vout3 = math_asr_s32(vacc3, 8);

    vout0 = math_max_s32(vout0, -128);
    vout1 = math_max_s32(vout1, -128);
    vout2 = math_max_s32(vout2, -128);
    vout3 = math_max_s32(vout3, -128);

    vout0 = math_min_s32(vout0, 127);
    vout1 = math_min_s32(vout1, 127);
    vout2 = math_min_s32(vout2, 127);
    vout3 = math_min_s32(vout3, 127);

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      int32_t vacc = *input++;
      vacc = vbias + vacc * vmultiplier;

      int32_t vout = math_asr_s32(vacc, 8);
      vout = math_max_s32(vout, -128);
      vout = math_min_s32(vout, 127);
      *output++ = (int8_t) vout;

      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}

void xnn_qs8_vlrelu_ukernel__scalar_andxor_x4(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vinput_zero_point = params->scalar_andxor.input_zero_point;
  const int32_t vmultiplier_diff = params->scalar_andxor.multiplier_diff;
  const int32_t vmultiplier_base = params->scalar_andxor.multiplier_base;
  const int32_t vbias = params->scalar_andxor.bias;
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    int32_t vacc0 = (int32_t) input[0];
    int32_t vacc1 = (int32_t) input[1];
    int32_t vacc2 = (int32_t) input[2];
    int32_t vacc3 = (int32_t) input[3];
    input += 4;

    vacc0 -= vinput_zero_point;
    vacc1 -= vinput_zero_point;
    vacc2 -= vinput_zero_point;
    vacc3 -= vinput_zero_point;

    int32_t vmultiplier0 = math_asr_s32(vacc0, 31);
    int32_t vmultiplier1 = math_asr_s32(vacc1, 31);
    int32_t vmultiplier2 = math_asr_s32(vacc2, 31);
    int32_t vmultiplier3 = math_asr_s32(vacc3, 31);

    vmultiplier0 &= vmultiplier_diff;
    vmultiplier1 &= vmultiplier_diff;
    vmultiplier2 &= vmultiplier_diff;
    vmultiplier3 &= vmultiplier_diff;

    vmultiplier0 ^= vmultiplier_base;
    vmultiplier1 ^= vmultiplier_base;
    vmultiplier2 ^= vmultiplier_base;
    vmultiplier3 ^= vmultiplier_base;

    vacc0 = vbias + vacc0 * vmultiplier0;
    vacc1 = vbias + vacc1 * vmultiplier1;
    vacc2 = vbias + vacc2 * vmultiplier2;
    vacc3 = vbias + vacc3 * vmultiplier3;

    int32_t vout0 = math_asr_s32(vacc0, 8);
    int32_t vout1 = math_asr_s32(vacc1, 8);
    int32_t vout2 = math_asr_s32(vacc2, 8);
    int32_t vout3 = math_asr_s32(vacc3, 8);

    vout0 = math_max_s32(vout0, -128);
    vout1 = math_max_s32(vout1, -128);
    vout2 = math_max_s32(vout2, -128);
    vout3 = math_max_s32(vout3, -128);

    vout0 = math_min_s32(vout0, 127);
    vout1 = math_min_s32(vout1, 127);
    vout2 = math_min_s32(vout2, 127);
    vout3 = math_min_s32(vout3, 127);

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      int32_t vacc = (int32_t) *input++ - vinput_zero_point;
      const int32_t vmultiplier = vmultiplier_base ^ (vmultiplier_diff & math_asr_s32(vacc, 31));
      vacc = vbias + vacc * vmultiplier;

      int32_t vout = math_asr_s32(vacc, 8);
      vout = math_max_s32(vout, -128);
      vout = math_min_s32(vout, 127);
      *output++ = (int8_t) vout;

      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}

void xnn_qs8_vmul_minmax_fp32_ukernel__scalar_x4(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t va_zero_point = params->fp32_scalar.a_zero_point;
  const int32_t vb_zero_point = params->fp32_scalar.b_zero_point;
  const float vscale = params->fp32_scalar.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0] - va_zero_point;
    const int32_t va1 = input_a[1] - va_zero_point;
    const int32_t va2 = input_a[2] - va_zero_point;
    const int32_t va3 = input_a[3] - va_zero_point;
    input_a += 4;

    const int32_t vb0 = input_b[0] - vb_zero_point;
    const int32_t vb1 = input_b[1] - vb_zero_point;
    const int32_t vb2 = input_b[2] - vb_zero_point;
    const int32_t vb3 = input_b[3] - vb_zero_point;
    input_b += 4;

    const int32_t vacc0 = va0 * vb0;
    const int32_t vacc1 = va1 * vb1;
    const int32_t vacc2 = va2 * vb2;
    const int32_t vacc3 = va3 * vb3;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;
    float vfpacc2 = (float) vacc2 * vscale;
    float vfpacc3 = (float) vacc3 * vscale;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;
    vfpacc2 += vmagic_bias;
    vfpacc3 += vmagic_bias;

    const int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    const int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;
    const int32_t vout2 = (int32_t) float_as_uint32(vfpacc2) - vmagic_bias_less_output_zero_point;
    const int32_t vout3 = (int32_t) float_as_uint32(vfpacc3) - vmagic_bias_less_output_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = (int32_t) *input_a++ - va_zero_point;
      const int32_t vb = (int32_t) *input_b++ - vb_zero_point;
      const int32_t vacc = va * vb;

      float vfpacc = (float) vacc * vscale;
      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      const int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;
      *output++ = (int8_t) vout;

      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}

void xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_x4(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t va_zero_point = params->fp32_scalar.a_zero_point;
  const float vscale = params->fp32_scalar.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  const int32_t vb = (int32_t) *input_b - params->fp32_scalar.b_zero_point;
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0] - va_zero_point;
    const int32_t va1 = input_a[1] - va_zero_point;
    const int32_t va2 = input_a[2] - va_zero_point;
    const int32_t va3 = input_a[3] - va_zero_point;
    input_a += 4;

    const int32_t vacc0 = va0 * vb;
    const int32_t vacc1 = va1 * vb;
    const int32_t vacc2 = va2 * vb;
    const int32_t vacc3 = va3 * vb;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;
    float vfpacc2 = (float) vacc2 * vscale;
    float vfpacc3 = (float) vacc3 * vscale;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;
    vfpacc2 += vmagic_bias;
    vfpacc3 += vmagic_bias;

    const int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    const int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;
    const int32_t vout2 = (int32_t) float_as_uint32(vfpacc2) - vmagic_bias_less_output_zero_point;
    const int32_t vout3 = (int32_t) float_as_uint32(vfpacc3) - vmagic_bias_less_output_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = (int32_t) *input_a++ - va_zero_point;
      const int32_t vacc = va * vb;

      float vfpacc = (float) vacc * vscale;
      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      const int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;
      *output++ = (int8_t) vout;

      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}

void xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  do {
    // First pass.
    {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint8_t* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
      }

      int32_t* b = buffer;
      size_t c = channels;
      do {
        int32_t vacc = vinit_bias;

        const int32_t vi0 = (int32_t) (uint32_t) *i0++;
        vacc += vi0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1++;
        vacc += vi1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2++;
        vacc += vi2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3++;
        vacc += vi3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4++;
        vacc += vi4;
        const int32_t vi5 = (int32_t) (uint32_t) *i5++;
        vacc += vi5;
        const int32_t vi6 = (int32_t) (uint32_t) *i6++;
        vacc += vi6;
        const int32_t vi7 = (int32_t) (uint32_t) *i7++;
        vacc += vi7;
        const int32_t vi8 = (int32_t) (uint32_t) *i8++;
        vacc += vi8;

        *b++ = vacc;
      } while (--c != 0);
    }

    size_t k = kernel_elements;
    // Intermediate passes.
    for (k -= 9; k > 8; k -= 8) {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      int32_t* b = buffer;
      size_t c = channels;
      do {
        int32_t vacc = *b;

        const int32_t vi0 = (int32_t) (uint32_t) *i0++;
        vacc += vi0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1++;
        vacc += vi1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2++;
        vacc += vi2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3++;
        vacc += vi3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4++;
        vacc += vi4;
        const int32_t vi5 = (int32_t) (uint32_t) *i5++;
        vacc += vi5;
        const int32_t vi6 = (int32_t) (uint32_t) *i6++;
        vacc += vi6;
        const int32_t vi7 = (int32_t) (uint32_t) *i7++;
        vacc += vi7;

        *b++ = vacc;
      } while (--c != 0);
    }

    // Last pass.
    {
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      const uint8_t* i1 = input[1];
      const uint8_t* i2 = input[2];
      const uint8_t* i3 = input[3];
      const uint8_t* i4 = input[4];
      const uint8_t* i5 = input[5];
      const uint8_t* i6 = input[6];
      const uint8_t* i7 = input[7];
      input = (const uint8_t**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      int32_t* b = buffer;
      do {
        int32_t vacc = *b++;

        const int32_t vi0 = (int32_t) (uint32_t) *i0++;
        vacc += vi0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1++;
        vacc += vi1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2++;
        vacc += vi2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3++;
        vacc += vi3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4++;
        vacc += vi4;
        const int32_t vi5 = (int32_t) (uint32_t) *i5++;
        vacc += vi5;
        const int32_t vi6 = (int32_t) (uint32_t) *i6++;
        vacc += vi6;
        const int32_t vi7 = (int32_t) (uint32_t) *i7++;
        vacc += vi7;

        float vfpacc = (float) vacc * vscale;
        vfpacc += vmagic_bias;
        int32_t vout = (int32_t) float_as_uint32(vfpacc);
        vout = math_max_s32(vout, vmagic_min);
        vout = math_min_s32(vout, vmagic_max);
        vout -= vmagic_bias_less_zero_point;

        *output++ = (uint8_t) vout;
      } while (--c != 0);
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    do {
      int32_t vacc = vinit_bias;

      const int32_t vi0 = (int32_t) (uint32_t) *i0++;
      vacc += vi0;
      const int32_t vi1 = (int32_t) (uint32_t) *i1++;
      vacc += vi1;
      const int32_t vi2 = (int32_t) (uint32_t) *i2++;
      vacc += vi2;
      const int32_t vi3 = (int32_t) (uint32_t) *i3++;
      vacc += vi3;
      const int32_t vi4 = (int32_t) (uint32_t) *i4++;
      vacc += vi4;
      const int32_t vi5 = (int32_t) (uint32_t) *i5++;
      vacc += vi5;
      const int32_t vi6 = (int32_t) (uint32_t) *i6++;
      vacc += vi6;
      const int32_t vi7 = (int32_t) (uint32_t) *i7++;
      vacc += vi7;
      const int32_t vi8 = (int32_t) (uint32_t) *i8++;
      vacc += vi8;

      float vfpacc = (float) vacc * vscale;
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc);
      vout = math_max_s32(vout, vmagic_min);
      vout = math_min_s32(vout, vmagic_max);
      vout -= vmagic_bias_less_zero_point;

      *output++ = (uint8_t) vout;
    } while (--c != 0);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_25p2c__scalar_lrintf(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
  const int32_t vkernel_zero_point = params->fp32_scalar_lrintf.kernel_zero_point;
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint8_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint8_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint8_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint8_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint8_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint8_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint8_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint8_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint8_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint8_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint8_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint8_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint8_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint8_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint8_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
      const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
      const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1] - vkernel_zero_point;

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
      const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
      const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3] - vkernel_zero_point;

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
      const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
      const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5] - vkernel_zero_point;

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
      const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
      const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7] - vkernel_zero_point;

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
      const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
      const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9] - vkernel_zero_point;

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) (uint32_t) i5[0];
      const int32_t vi5x1 = (int32_t) (uint32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10] - vkernel_zero_point;
      const int32_t vk5x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11] - vkernel_zero_point;

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) (uint32_t) i6[0];
      const int32_t vi6x1 = (int32_t) (uint32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12] - vkernel_zero_point;
      const int32_t vk6x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13] - vkernel_zero_point;

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) (uint32_t) i7[0];
      const int32_t vi7x1 = (int32_t) (uint32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14] - vkernel_zero_point;
      const int32_t vk7x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15] - vkernel_zero_point;

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) (uint32_t) i8[0];
      const int32_t vi8x1 = (int32_t) (uint32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16] - vkernel_zero_point;
      const int32_t vk8x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17] - vkernel_zero_point;

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      const int32_t vi9x0 = (int32_t) (uint32_t) i9[0];
      const int32_t vi9x1 = (int32_t) (uint32_t) i9[1];
      i9 += 2;

      const int32_t vk9x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18] - vkernel_zero_point;
      const int32_t vk9x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[19] - vkernel_zero_point;

      vacc0 += vi9x0 * vk9x0;
      vacc1 += vi9x1 * vk9x1;

      const int32_t vi10x0 = (int32_t) (uint32_t) i10[0];
      const int32_t vi10x1 = (int32_t) (uint32_t) i10[1];
      i10 += 2;

      const int32_t vk10x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20] - vkernel_zero_point;
      const int32_t vk10x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[21] - vkernel_zero_point;

      vacc0 += vi10x0 * vk10x0;
      vacc1 += vi10x1 * vk10x1;

      const int32_t vi11x0 = (int32_t) (uint32_t) i11[0];
      const int32_t vi11x1 = (int32_t) (uint32_t) i11[1];
      i11 += 2;

      const int32_t vk11x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22] - vkernel_zero_point;
      const int32_t vk11x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[23] - vkernel_zero_point;

      vacc0 += vi11x0 * vk11x0;
      vacc1 += vi11x1 * vk11x1;

      const int32_t vi12x0 = (int32_t) (uint32_t) i12[0];
      const int32_t vi12x1 = (int32_t) (uint32_t) i12[1];
      i12 += 2;

      const int32_t vk12x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24] - vkernel_zero_point;
      const int32_t vk12x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[25] - vkernel_zero_point;

      vacc0 += vi12x0 * vk12x0;
      vacc1 += vi12x1 * vk12x1;

      const int32_t vi13x0 = (int32_t) (uint32_t) i13[0];
      const int32_t vi13x1 = (int32_t) (uint32_t) i13[1];
      i13 += 2;

      const int32_t vk13x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26] - vkernel_zero_point;
      const int32_t vk13x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[27] - vkernel_zero_point;

      vacc0 += vi13x0 * vk13x0;
      vacc1 += vi13x1 * vk13x1;

      const int32_t vi14x0 = (int32_t) (uint32_t) i14[0];
      const int32_t vi14x1 = (int32_t) (uint32_t) i14[1];
      i14 += 2;

      const int32_t vk14x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28] - vkernel_zero_point;
      const int32_t vk14x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[29] - vkernel_zero_point;

      vacc0 += vi14x0 * vk14x0;
      vacc1 += vi14x1 * vk14x1;

      const int32_t vi15x0 = (int32_t) (uint32_t) i15[0];
      const int32_t vi15x1 = (int32_t) (uint32_t) i15[1];
      i15 += 2;

      const int32_t vk15x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30] - vkernel_zero_point;
      const int32_t vk15x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[31] - vkernel_zero_point;

      vacc0 += vi15x0 * vk15x0;
      vacc1 += vi15x1 * vk15x1;

      const int32_t vi16x0 = (int32_t) (uint32_t) i16[0];
      const int32_t vi16x1 = (int32_t) (uint32_t) i16[1];
      i16 += 2;

      const int32_t vk16x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32] - vkernel_zero_point;
      const int32_t vk16x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[33] - vkernel_zero_point;

      vacc0 += vi16x0 * vk16x0;
      vacc1 += vi16x1 * vk16x1;

      const int32_t vi17x0 = (int32_t) (uint32_t) i17[0];
      const int32_t vi17x1 = (int32_t) (uint32_t) i17[1];
      i17 += 2;

      const int32_t vk17x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34] - vkernel_zero_point;
      const int32_t vk17x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[35] - vkernel_zero_point;

      vacc0 += vi17x0 * vk17x0;
      vacc1 += vi17x1 * vk17x1;

      const int32_t vi18x0 = (int32_t) (uint32_t) i18[0];
      const int32_t vi18x1 = (int32_t) (uint32_t) i18[1];
      i18 += 2;

      const int32_t vk18x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36] - vkernel_zero_point;
      const int32_t vk18x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[37] - vkernel_zero_point;

      vacc0 += vi18x0 * vk18x0;
      vacc1 += vi18x1 * vk18x1;

      const int32_t vi19x0 = (int32_t) (uint32_t) i19[0];
      const int32_t vi19x1 = (int32_t) (uint32_t) i19[1];
      i19 += 2;

      const int32_t vk19x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38] - vkernel_zero_point;
      const int32_t vk19x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[39] - vkernel_zero_point;

      vacc0 += vi19x0 * vk19x0;
      vacc1 += vi19x1 * vk19x1;

      const int32_t vi20x0 = (int32_t) (uint32_t) i20[0];
      const int32_t vi20x1 = (int32_t) (uint32_t) i20[1];
      i20 += 2;

      const int32_t vk20x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40] - vkernel_zero_point;
      const int32_t vk20x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[41] - vkernel_zero_point;

      vacc0 += vi20x0 * vk20x0;
      vacc1 += vi20x1 * vk20x1;

      const int32_t vi21x0 = (int32_t) (uint32_t) i21[0];
      const int32_t vi21x1 = (int32_t) (uint32_t) i21[1];
      i21 += 2;

      const int32_t vk21x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42] - vkernel_zero_point;
      const int32_t vk21x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[43] - vkernel_zero_point;

      vacc0 += vi21x0 * vk21x0;
      vacc1 += vi21x1 * vk21x1;

      const int32_t vi22x0 = (int32_t) (uint32_t) i22[0];
      const int32_t vi22x1 = (int32_t) (uint32_t) i22[1];
      i22 += 2;

      const int32_t vk22x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44] - vkernel_zero_point;
      const int32_t vk22x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[45] - vkernel_zero_point;

      vacc0 += vi22x0 * vk22x0;
      vacc1 += vi22x1 * vk22x1;

      const int32_t vi23x0 = (int32_t) (uint32_t) i23[0];
      const int32_t vi23x1 = (int32_t) (uint32_t) i23[1];
      i23 += 2;

      const int32_t vk23x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46] - vkernel_zero_point;
      const int32_t vk23x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[47] - vkernel_zero_point;

      vacc0 += vi23x0 * vk23x0;
      vacc1 += vi23x1 * vk23x1;

      const int32_t vi24x0 = (int32_t) (uint32_t) i24[0];
      const int32_t vi24x1 = (int32_t) (uint32_t) i24[1];
      i24 += 2;

      const int32_t vk24x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48] - vkernel_zero_point;
      const int32_t vk24x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[49] - vkernel_zero_point;

      vacc0 += vi24x0 * vk24x0;
      vacc1 += vi24x1 * vk24x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(uint8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      vfpacc0 *= vscale;
      vfpacc1 *= vscale;

      vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

      const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
      const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

      int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
      int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;

      output[0] = (uint8_t) vout0;
      output[1] = (uint8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) (uint32_t) *i0;
      const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) (uint32_t) *i1;
      const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) (uint32_t) *i2;
      const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) (uint32_t) *i3;
      const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) (uint32_t) *i4;
      const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) (uint32_t) *i5;
      const int32_t vk5 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10] - vkernel_zero_point;
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) (uint32_t) *i6;
      const int32_t vk6 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12] - vkernel_zero_point;
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) (uint32_t) *i7;
      const int32_t vk7 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14] - vkernel_zero_point;
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) (uint32_t) *i8;
      const int32_t vk8 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16] - vkernel_zero_point;
      vacc += vi8 * vk8;
      const int32_t vi9 = (int32_t) (uint32_t) *i9;
      const int32_t vk9 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18] - vkernel_zero_point;
      vacc += vi9 * vk9;
      const int32_t vi10 = (int32_t) (uint32_t) *i10;
      const int32_t vk10 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20] - vkernel_zero_point;
      vacc += vi10 * vk10;
      const int32_t vi11 = (int32_t) (uint32_t) *i11;
      const int32_t vk11 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22] - vkernel_zero_point;
      vacc += vi11 * vk11;
      const int32_t vi12 = (int32_t) (uint32_t) *i12;
      const int32_t vk12 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24] - vkernel_zero_point;
      vacc += vi12 * vk12;
      const int32_t vi13 = (int32_t) (uint32_t) *i13;
      const int32_t vk13 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26] - vkernel_zero_point;
      vacc += vi13 * vk13;
      const int32_t vi14 = (int32_t) (uint32_t) *i14;
      const int32_t vk14 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28] - vkernel_zero_point;
      vacc += vi14 * vk14;
      const int32_t vi15 = (int32_t) (uint32_t) *i15;
      const int32_t vk15 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30] - vkernel_zero_point;
      vacc += vi15 * vk15;
      const int32_t vi16 = (int32_t) (uint32_t) *i16;
      const int32_t vk16 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32] - vkernel_zero_point;
      vacc += vi16 * vk16;
      const int32_t vi17 = (int32_t) (uint32_t) *i17;
      const int32_t vk17 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34] - vkernel_zero_point;
      vacc += vi17 * vk17;
      const int32_t vi18 = (int32_t) (uint32_t) *i18;
      const int32_t vk18 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36] - vkernel_zero_point;
      vacc += vi18 * vk18;
      const int32_t vi19 = (int32_t) (uint32_t) *i19;
      const int32_t vk19 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38] - vkernel_zero_point;
      vacc += vi19 * vk19;
      const int32_t vi20 = (int32_t) (uint32_t) *i20;
      const int32_t vk20 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40] - vkernel_zero_point;
      vacc += vi20 * vk20;
      const int32_t vi21 = (int32_t) (uint32_t) *i21;
      const int32_t vk21 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42] - vkernel_zero_point;
      vacc += vi21 * vk21;
      const int32_t vi22 = (int32_t) (uint32_t) *i22;
      const int32_t vk22 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44] - vkernel_zero_point;
      vacc += vi22 * vk22;
      const int32_t vi23 = (int32_t) (uint32_t) *i23;
      const int32_t vk23 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46] - vkernel_zero_point;
      vacc += vi23 * vk23;
      const int32_t vi24 = (int32_t) (uint32_t) *i24;
      const int32_t vk24 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48] - vkernel_zero_point;
      vacc += vi24 * vk24;

      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      const int32_t vrndacc = (int32_t) lrintf(vfpacc);
      int32_t vout = vrndacc + voutput_zero_point;

      *output++ = (uint8_t) vout;
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
  const int32_t vkernel_zero_point = params->fp32_scalar_lrintf.kernel_zero_point;
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
      const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
      const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1] - vkernel_zero_point;

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
      const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
      const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3] - vkernel_zero_point;

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
      const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
      const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5] - vkernel_zero_point;

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
      const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
      const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7] - vkernel_zero_point;

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
      const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
      const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9] - vkernel_zero_point;

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) (uint32_t) i5[0];
      const int32_t vi5x1 = (int32_t) (uint32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10] - vkernel_zero_point;
      const int32_t vk5x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11] - vkernel_zero_point;

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) (uint32_t) i6[0];
      const int32_t vi6x1 = (int32_t) (uint32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12] - vkernel_zero_point;
      const int32_t vk6x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13] - vkernel_zero_point;

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) (uint32_t) i7[0];
      const int32_t vi7x1 = (int32_t) (uint32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14] - vkernel_zero_point;
      const int32_t vk7x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15] - vkernel_zero_point;

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) (uint32_t) i8[0];
      const int32_t vi8x1 = (int32_t) (uint32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16] - vkernel_zero_point;
      const int32_t vk8x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17] - vkernel_zero_point;

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 18 * sizeof(uint8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      vfpacc0 *= vscale;
      vfpacc1 *= vscale;

      vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

      const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
      const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

      int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
      int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;

      output[0] = (uint8_t) vout0;
      output[1] = (uint8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) (uint32_t) *i0;
      const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) (uint32_t) *i1;
      const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) (uint32_t) *i2;
      const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) (uint32_t) *i3;
      const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) (uint32_t) *i4;
      const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) (uint32_t) *i5;
      const int32_t vk5 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10] - vkernel_zero_point;
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) (uint32_t) *i6;
      const int32_t vk6 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12] - vkernel_zero_point;
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) (uint32_t) *i7;
      const int32_t vk7 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14] - vkernel_zero_point;
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) (uint32_t) *i8;
      const int32_t vk8 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16] - vkernel_zero_point;
      vacc += vi8 * vk8;

      float vfpacc = (float) vacc * vscale;

      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      const int32_t vrndacc = (int32_t) lrintf(vfpacc);
      int32_t vout = vrndacc + voutput_zero_point;

      *output++ = (uint8_t) vout;
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_f32_vcvt_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vzero_point = params->scalar.zero_point;
  const float vscale = params->scalar.scale;

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    int32_t vx0 = (int32_t) input[0];
    int32_t vx1 = (int32_t) input[1];
    int32_t vx2 = (int32_t) input[2];
    int32_t vx3 = (int32_t) input[3];
    input += 4;

    vx0 -= vzero_point;
    vx1 -= vzero_point;
    vx2 -= vzero_point;
    vx3 -= vzero_point;

    float vy0 = (float) vx0;
    float vy1 = (float) vx1;
    float vy2 = (float) vx2;
    float vy3 = (float) vx3;

    vy0 *= vscale;
    vy1 *= vscale;
    vy2 *= vscale;
    vy3 *= vscale;

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      int32_t vx = *input++;
      vx -= vzero_point;

      float vy = (float) vx;
      vy *= vscale;
      *output++ = vy;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 1) * sizeof(uint8_t);

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  int32_t* b = buffer;
  size_t c = channels;
  do {
    int32_t vacc = vinit_bias;
    const int32_t vi0 = (int32_t) *i0++;
    const int32_t vi1 = (int32_t) *i1++;

    vacc += vi0;
    const int32_t vi2 = (int32_t) *i2++;
    vacc += vi1;
    const int32_t vi3 = (int32_t) *i3++;
    vacc += vi2;
    const int32_t vi4 = (int32_t) *i4++;
    vacc += vi3;
    const int32_t vi5 = (int32_t) *i5++;
    vacc += vi4;
    const int32_t vi6 = (int32_t) *i6++;

    vacc += vi5;
    vacc += vi6;

    *b++ = vacc;
  } while (--c != 0);

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    do {
      int32_t vacc = *b;
      const int32_t vi0 = (int32_t) *i0++;
      const int32_t vi1 = (int32_t) *i1++;

      vacc += vi0;
      const int32_t vi2 = (int32_t) *i2++;
      vacc += vi1;
      const int32_t vi3 = (int32_t) *i3++;
      vacc += vi2;
      const int32_t vi4 = (int32_t) *i4++;
      vacc += vi3;
      const int32_t vi5 = (int32_t) *i5++;
      vacc += vi4;
      const int32_t vi6 = (int32_t) *i6++;

      vacc += vi5;
      vacc += vi6;

      *b++ = vacc;
    } while (--c != 0);
  }

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  do {
    int32_t vacc = *buffer++;
    const int32_t vi0 = (int32_t) *i0++;
    const int32_t vi1 = (int32_t) *i1++;

    vacc += vi0;
    const int32_t vi2 = (int32_t) *i2++;
    vacc += vi1;
    const int32_t vi3 = (int32_t) *i3++;
    vacc += vi2;
    const int32_t vi4 = (int32_t) *i4++;
    vacc += vi3;
    const int32_t vi5 = (int32_t) *i5++;
    vacc += vi4;
    const int32_t vi6 = (int32_t) *i6++;

    vacc += vi5;
    vacc += vi6;

    float vfpacc = (float) vacc * vscale;
    vfpacc += vmagic_bias;
    int32_t vout = (int32_t) float_as_uint32(vfpacc);
    vout = math_max_s32(vout, vmagic_min);
    vout = math_min_s32(vout, vmagic_max);
    vout -= vmagic_bias_less_zero_point;

    *output++ = (uint8_t) vout;
  } while (--channels != 0);
}

void xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  do {
    int32_t vacc = vinit_bias;
    const int32_t vi0 = (int32_t) *i0++;
    const int32_t vi1 = (int32_t) *i1++;

    vacc += vi0;
    const int32_t vi2 = (int32_t) *i2++;
    vacc += vi1;
    const int32_t vi3 = (int32_t) *i3++;
    vacc += vi2;
    const int32_t vi4 = (int32_t) *i4++;
    vacc += vi3;
    const int32_t vi5 = (int32_t) *i5++;
    vacc += vi4;
    const int32_t vi6 = (int32_t) *i6++;

    vacc += vi5;
    vacc += vi6;

    float vfpacc = (float) vacc * vscale;
    vfpacc += vmagic_bias;
    int32_t vout = (int32_t) float_as_uint32(vfpacc);
    vout = math_max_s32(vout, vmagic_min);
    vout = math_min_s32(vout, vmagic_max);
    vout -= vmagic_bias_less_zero_point;

    *output++ = (uint8_t) vout;
  } while (--channels != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  const int32_t vb_zero_point = params->fp32_scalar_lrintf.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) (uint32_t) *a0++;

      const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
      const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
      const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
      const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
      w = (const void*) ((const uint8_t*) w + 4);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;

      k -= sizeof(uint8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale = params->fp32_scalar_lrintf.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  const int32_t vb_zero_point = params->fp32_scalar_lrintf.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) (uint32_t) *a0++;
      const int32_t va1 = (int32_t) (uint32_t) *a1++;
      const int32_t va2 = (int32_t) (uint32_t) *a2++;

      const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
      const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
      const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
      const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
      w = (const void*) ((const uint8_t*) w + 4);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;

      k -= sizeof(uint8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;

    const float vscale = params->fp32_scalar_lrintf.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;
    vfpacc2x0 *= vscale;
    vfpacc2x1 *= vscale;
    vfpacc2x2 *= vscale;
    vfpacc2x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = math_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = math_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = math_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = math_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = math_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = math_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = math_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = math_max_f32(vfpacc2x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = math_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = math_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = math_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = math_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = math_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = math_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = math_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = math_min_f32(vfpacc2x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);
    const int32_t vrndacc1x0 = (int32_t) lrintf(vfpacc1x0);
    const int32_t vrndacc1x1 = (int32_t) lrintf(vfpacc1x1);
    const int32_t vrndacc1x2 = (int32_t) lrintf(vfpacc1x2);
    const int32_t vrndacc1x3 = (int32_t) lrintf(vfpacc1x3);
    const int32_t vrndacc2x0 = (int32_t) lrintf(vfpacc2x0);
    const int32_t vrndacc2x1 = (int32_t) lrintf(vfpacc2x1);
    const int32_t vrndacc2x2 = (int32_t) lrintf(vfpacc2x2);
    const int32_t vrndacc2x3 = (int32_t) lrintf(vfpacc2x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;
    int32_t vout1x0 = vrndacc1x0 + voutput_zero_point;
    int32_t vout1x1 = vrndacc1x1 + voutput_zero_point;
    int32_t vout1x2 = vrndacc1x2 + voutput_zero_point;
    int32_t vout1x3 = vrndacc1x3 + voutput_zero_point;
    int32_t vout2x0 = vrndacc2x0 + voutput_zero_point;
    int32_t vout2x1 = vrndacc2x1 + voutput_zero_point;
    int32_t vout2x2 = vrndacc2x2 + voutput_zero_point;
    int32_t vout2x3 = vrndacc2x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;
      c1[2] = (uint8_t) vout1x2;
      c1[3] = (uint8_t) vout1x3;
      c2[0] = (uint8_t) vout2x0;
      c2[1] = (uint8_t) vout2x1;
      c2[2] = (uint8_t) vout2x2;
      c2[3] = (uint8_t) vout2x3;

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (uint8_t) vout1x0;
        c1[1] = (uint8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (uint8_t) vout2x0;
        c2[1] = (uint8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
      }
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
        c1[0] = (uint8_t) vout1x0;
        c2[0] = (uint8_t) vout2x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t**restrict a,
    const void*restrict w,
    uint8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;

  const int32_t vb_zero_point = params->fp32_scalar_lrintf.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) (uint32_t) *a0++;

        const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
        const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
        const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
        const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
        w = (const void*) ((const uint8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;

        k -= sizeof(uint8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale = params->fp32_scalar_lrintf.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t**restrict a,
    const void*restrict w,
    uint8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  const int32_t vb_zero_point = params->fp32_scalar_lrintf.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) (uint32_t) *a0++;
        const int32_t va1 = (int32_t) (uint32_t) *a1++;
        const int32_t va2 = (int32_t) (uint32_t) *a2++;

        const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
        const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
        const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
        const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
        w = (const void*) ((const uint8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc1x2 += va1 * vb2;
        vacc1x3 += va1 * vb3;
        vacc2x0 += va2 * vb0;
        vacc2x1 += va2 * vb1;
        vacc2x2 += va2 * vb2;
        vacc2x3 += va2 * vb3;

        k -= sizeof(uint8_t);
      } while (k != 0);
      p -= 3 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;

    const float vscale = params->fp32_scalar_lrintf.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;
    vfpacc2x0 *= vscale;
    vfpacc2x1 *= vscale;
    vfpacc2x2 *= vscale;
    vfpacc2x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = math_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = math_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = math_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = math_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = math_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = math_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = math_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = math_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = math_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = math_max_f32(vfpacc2x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = math_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = math_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = math_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = math_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = math_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = math_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = math_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = math_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = math_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = math_min_f32(vfpacc2x3, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc0x2 = (int32_t) lrintf(vfpacc0x2);
    const int32_t vrndacc0x3 = (int32_t) lrintf(vfpacc0x3);
    const int32_t vrndacc1x0 = (int32_t) lrintf(vfpacc1x0);
    const int32_t vrndacc1x1 = (int32_t) lrintf(vfpacc1x1);
    const int32_t vrndacc1x2 = (int32_t) lrintf(vfpacc1x2);
    const int32_t vrndacc1x3 = (int32_t) lrintf(vfpacc1x3);
    const int32_t vrndacc2x0 = (int32_t) lrintf(vfpacc2x0);
    const int32_t vrndacc2x1 = (int32_t) lrintf(vfpacc2x1);
    const int32_t vrndacc2x2 = (int32_t) lrintf(vfpacc2x2);
    const int32_t vrndacc2x3 = (int32_t) lrintf(vfpacc2x3);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout0x2 = vrndacc0x2 + voutput_zero_point;
    int32_t vout0x3 = vrndacc0x3 + voutput_zero_point;
    int32_t vout1x0 = vrndacc1x0 + voutput_zero_point;
    int32_t vout1x1 = vrndacc1x1 + voutput_zero_point;
    int32_t vout1x2 = vrndacc1x2 + voutput_zero_point;
    int32_t vout1x3 = vrndacc1x3 + voutput_zero_point;
    int32_t vout2x0 = vrndacc2x0 + voutput_zero_point;
    int32_t vout2x1 = vrndacc2x1 + voutput_zero_point;
    int32_t vout2x2 = vrndacc2x2 + voutput_zero_point;
    int32_t vout2x3 = vrndacc2x3 + voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c2[0] = (uint8_t) vout2x0;
      c2[1] = (uint8_t) vout2x1;
      c2[2] = (uint8_t) vout2x2;
      c2[3] = (uint8_t) vout2x3;
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;
      c1[2] = (uint8_t) vout1x2;
      c1[3] = (uint8_t) vout1x3;
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;

      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c2[0] = (uint8_t) vout2x0;
        c2[1] = (uint8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c1[0] = (uint8_t) vout1x0;
        c1[1] = (uint8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c2[0] = (uint8_t) vout2x0;
        c1[0] = (uint8_t) vout1x0;
        c0[0] = (uint8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vadd_minmax_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t vbias = params->scalar.bias;
  const int32_t va_multiplier = params->scalar.a_multiplier;
  const int32_t vb_multiplier = params->scalar.b_multiplier;
  const uint32_t vshift = params->scalar.shift;
  const int32_t voutput_min_less_zero_point = params->scalar.output_min_less_zero_point;
  const int32_t voutput_max_less_zero_point = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const int32_t va0 = input_a[0];
    const int32_t va1 = input_a[1];
    const int32_t va2 = input_a[2];
    const int32_t va3 = input_a[3];
    input_a += 4;

    const int32_t vb0 = input_b[0];
    int32_t vacc0 = vbias + va0 * va_multiplier;
    const int32_t vb1 = input_b[1];
    int32_t vacc1 = vbias + va1 * va_multiplier;
    const int32_t vb2 = input_b[2];
    int32_t vacc2 = vbias + va2 * va_multiplier;
    const int32_t vb3 = input_b[3];
    int32_t vacc3 = vbias + va3 * va_multiplier;
    input_b += 4;

    vacc0 += vb0 * vb_multiplier;
    vacc1 += vb1 * vb_multiplier;
    vacc2 += vb2 * vb_multiplier;
    vacc3 += vb3 * vb_multiplier;

    int32_t vout0 = math_asr_s32(vacc0, vshift);
    int32_t vout1 = math_asr_s32(vacc1, vshift);
    int32_t vout2 = math_asr_s32(vacc2, vshift);
    int32_t vout3 = math_asr_s32(vacc3, vshift);

    vout0 = math_max_s32(vout0, voutput_min_less_zero_point);
    vout1 = math_max_s32(vout1, voutput_min_less_zero_point);
    vout2 = math_max_s32(vout2, voutput_min_less_zero_point);
    vout3 = math_max_s32(vout3, voutput_min_less_zero_point);

    vout0 = math_min_s32(vout0, voutput_max_less_zero_point);
    vout1 = math_min_s32(vout1, voutput_max_less_zero_point);
    vout2 = math_min_s32(vout2, voutput_max_less_zero_point);
    vout3 = math_min_s32(vout3, voutput_max_less_zero_point);

    vout0 += voutput_zero_point;
    vout1 += voutput_zero_point;
    vout2 += voutput_zero_point;
    vout3 += voutput_zero_point;

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = *input_a++;
      const int32_t vb = *input_b++;
      const int32_t vacc = vbias + va * va_multiplier + vb * vb_multiplier;

      int32_t vout = math_asr_s32(vacc, vshift);
      vout = math_max_s32(vout, voutput_min_less_zero_point);
      vout = math_min_s32(vout, voutput_max_less_zero_point);
      *output++ = (uint8_t) (vout + voutput_zero_point);

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_qu8_vaddc_minmax_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t vbias = params->scalar.bias + (int32_t) *input_b * params->scalar.b_multiplier;
  const int32_t va_multiplier = params->scalar.a_multiplier;
  const uint32_t vshift = params->scalar.shift;
  const int32_t voutput_min_less_zero_point = params->scalar.output_min_less_zero_point;
  const int32_t voutput_max_less_zero_point = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const int32_t va0 = input_a[0];
    const int32_t va1 = input_a[1];
    const int32_t va2 = input_a[2];
    const int32_t va3 = input_a[3];
    input_a += 4;

    const int32_t vacc0 = vbias + va0 * va_multiplier;
    const int32_t vacc1 = vbias + va1 * va_multiplier;
    const int32_t vacc2 = vbias + va2 * va_multiplier;
    const int32_t vacc3 = vbias + va3 * va_multiplier;
    input_b += 4;

    int32_t vout0 = math_asr_s32(vacc0, vshift);
    int32_t vout1 = math_asr_s32(vacc1, vshift);
    int32_t vout2 = math_asr_s32(vacc2, vshift);
    int32_t vout3 = math_asr_s32(vacc3, vshift);

    vout0 = math_max_s32(vout0, voutput_min_less_zero_point);
    vout1 = math_max_s32(vout1, voutput_min_less_zero_point);
    vout2 = math_max_s32(vout2, voutput_min_less_zero_point);
    vout3 = math_max_s32(vout3, voutput_min_less_zero_point);

    vout0 = math_min_s32(vout0, voutput_max_less_zero_point);
    vout1 = math_min_s32(vout1, voutput_max_less_zero_point);
    vout2 = math_min_s32(vout2, voutput_max_less_zero_point);
    vout3 = math_min_s32(vout3, voutput_max_less_zero_point);

    vout0 += voutput_zero_point;
    vout1 += voutput_zero_point;
    vout2 += voutput_zero_point;
    vout3 += voutput_zero_point;

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = *input_a++;
      const int32_t vacc = vbias + va * va_multiplier;

      int32_t vout = math_asr_s32(vacc, vshift);
      vout = math_max_s32(vout, voutput_min_less_zero_point);
      vout = math_min_s32(vout, voutput_max_less_zero_point);
      *output++ = (uint8_t) (vout + voutput_zero_point);

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_qu8_vcvt_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vbias = params->scalar.bias;
  const int32_t vmultiplier = params->scalar.multiplier;
  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    int32_t vacc0 = input[0];
    int32_t vacc1 = input[1];
    int32_t vacc2 = input[2];
    int32_t vacc3 = input[3];
    input += 4;

    vacc0 = vbias + vacc0 * vmultiplier;
    vacc1 = vbias + vacc1 * vmultiplier;
    vacc2 = vbias + vacc2 * vmultiplier;
    vacc3 = vbias + vacc3 * vmultiplier;

    int32_t vout0 = math_asr_s32(vacc0, 8);
    int32_t vout1 = math_asr_s32(vacc1, 8);
    int32_t vout2 = math_asr_s32(vacc2, 8);
    int32_t vout3 = math_asr_s32(vacc3, 8);

    vout0 = math_max_s32(vout0, 0);
    vout1 = math_max_s32(vout1, 0);
    vout2 = math_max_s32(vout2, 0);
    vout3 = math_max_s32(vout3, 0);

    vout0 = math_min_s32(vout0, 255);
    vout1 = math_min_s32(vout1, 255);
    vout2 = math_min_s32(vout2, 255);
    vout3 = math_min_s32(vout3, 255);

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      int32_t vacc = *input++;
      vacc = vbias + vacc * vmultiplier;

      int32_t vout = math_asr_s32(vacc, 8);
      vout = math_max_s32(vout, 0);
      vout = math_min_s32(vout, 255);
      *output++ = (uint8_t) vout;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_qu8_vlrelu_ukernel__scalar_andxor_x4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vinput_zero_point = params->scalar_andxor.input_zero_point;
  const int32_t vmultiplier_diff = params->scalar_andxor.multiplier_diff;
  const int32_t vmultiplier_base = params->scalar_andxor.multiplier_base;
  const int32_t vbias = params->scalar_andxor.bias;
  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    int32_t vacc0 = (int32_t) input[0];
    int32_t vacc1 = (int32_t) input[1];
    int32_t vacc2 = (int32_t) input[2];
    int32_t vacc3 = (int32_t) input[3];
    input += 4;

    vacc0 -= vinput_zero_point;
    vacc1 -= vinput_zero_point;
    vacc2 -= vinput_zero_point;
    vacc3 -= vinput_zero_point;

    int32_t vmultiplier0 = math_asr_s32(vacc0, 31);
    int32_t vmultiplier1 = math_asr_s32(vacc1, 31);
    int32_t vmultiplier2 = math_asr_s32(vacc2, 31);
    int32_t vmultiplier3 = math_asr_s32(vacc3, 31);

    vmultiplier0 &= vmultiplier_diff;
    vmultiplier1 &= vmultiplier_diff;
    vmultiplier2 &= vmultiplier_diff;
    vmultiplier3 &= vmultiplier_diff;

    vmultiplier0 ^= vmultiplier_base;
    vmultiplier1 ^= vmultiplier_base;
    vmultiplier2 ^= vmultiplier_base;
    vmultiplier3 ^= vmultiplier_base;

    vacc0 = vbias + vacc0 * vmultiplier0;
    vacc1 = vbias + vacc1 * vmultiplier1;
    vacc2 = vbias + vacc2 * vmultiplier2;
    vacc3 = vbias + vacc3 * vmultiplier3;

    int32_t vout0 = math_asr_s32(vacc0, 8);
    int32_t vout1 = math_asr_s32(vacc1, 8);
    int32_t vout2 = math_asr_s32(vacc2, 8);
    int32_t vout3 = math_asr_s32(vacc3, 8);

    vout0 = math_max_s32(vout0, 0);
    vout1 = math_max_s32(vout1, 0);
    vout2 = math_max_s32(vout2, 0);
    vout3 = math_max_s32(vout3, 0);

    vout0 = math_min_s32(vout0, 255);
    vout1 = math_min_s32(vout1, 255);
    vout2 = math_min_s32(vout2, 255);
    vout3 = math_min_s32(vout3, 255);

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      int32_t vacc = (int32_t) *input++ - vinput_zero_point;
      const int32_t vmultiplier = vmultiplier_base ^ (vmultiplier_diff & math_asr_s32(vacc, 31));
      vacc = vbias + vacc * vmultiplier;

      int32_t vout = math_asr_s32(vacc, 8);
      vout = math_max_s32(vout, 0);
      vout = math_min_s32(vout, 255);
      *output++ = (uint8_t) vout;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_qu8_vmul_minmax_fp32_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t va_zero_point = params->fp32_scalar.a_zero_point;
  const int32_t vb_zero_point = params->fp32_scalar.b_zero_point;
  const float vscale = params->fp32_scalar.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const int32_t va0 = input_a[0] - va_zero_point;
    const int32_t va1 = input_a[1] - va_zero_point;
    const int32_t va2 = input_a[2] - va_zero_point;
    const int32_t va3 = input_a[3] - va_zero_point;
    input_a += 4;

    const int32_t vb0 = input_b[0] - vb_zero_point;
    const int32_t vb1 = input_b[1] - vb_zero_point;
    const int32_t vb2 = input_b[2] - vb_zero_point;
    const int32_t vb3 = input_b[3] - vb_zero_point;
    input_b += 4;

    const int32_t vacc0 = va0 * vb0;
    const int32_t vacc1 = va1 * vb1;
    const int32_t vacc2 = va2 * vb2;
    const int32_t vacc3 = va3 * vb3;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;
    float vfpacc2 = (float) vacc2 * vscale;
    float vfpacc3 = (float) vacc3 * vscale;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;
    vfpacc2 += vmagic_bias;
    vfpacc3 += vmagic_bias;

    const int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    const int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;
    const int32_t vout2 = (int32_t) float_as_uint32(vfpacc2) - vmagic_bias_less_output_zero_point;
    const int32_t vout3 = (int32_t) float_as_uint32(vfpacc3) - vmagic_bias_less_output_zero_point;

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = (int32_t) *input_a++ - va_zero_point;
      const int32_t vb = (int32_t) *input_b++ - vb_zero_point;
      const int32_t vacc = va * vb;

      float vfpacc = (float) vacc * vscale;
      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      const int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;
      *output++ = (uint8_t) vout;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t va_zero_point = params->fp32_scalar.a_zero_point;
  const float vscale = params->fp32_scalar.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  const int32_t vb = (int32_t) *input_b - params->fp32_scalar.b_zero_point;
  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const int32_t va0 = input_a[0] - va_zero_point;
    const int32_t va1 = input_a[1] - va_zero_point;
    const int32_t va2 = input_a[2] - va_zero_point;
    const int32_t va3 = input_a[3] - va_zero_point;
    input_a += 4;

    const int32_t vacc0 = va0 * vb;
    const int32_t vacc1 = va1 * vb;
    const int32_t vacc2 = va2 * vb;
    const int32_t vacc3 = va3 * vb;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;
    float vfpacc2 = (float) vacc2 * vscale;
    float vfpacc3 = (float) vacc3 * vscale;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;
    vfpacc2 += vmagic_bias;
    vfpacc3 += vmagic_bias;

    const int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    const int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;
    const int32_t vout2 = (int32_t) float_as_uint32(vfpacc2) - vmagic_bias_less_output_zero_point;
    const int32_t vout3 = (int32_t) float_as_uint32(vfpacc3) - vmagic_bias_less_output_zero_point;

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = (int32_t) *input_a++ - va_zero_point;
      const int32_t vacc = va * vb;

      float vfpacc = (float) vacc * vscale;
      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      const int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;
      *output++ = (uint8_t) vout;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_s8_ibilinear_ukernel__scalar_c1(
    size_t output_pixels,
    size_t channels,
    const int8_t**restrict input,
    size_t input_offset,
    const int16_t*restrict weights,
    int8_t*restrict output,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const int8_t* i0 = (const int8_t*) ((uintptr_t) input[0] + input_offset);
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input[1] + input_offset);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input[2] + input_offset);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const int32_t valphah = (int32_t) (uint32_t) (uint16_t) weights[0];
    const int32_t valphav = (int32_t) (uint32_t) (uint16_t) weights[1];
    weights += 2;

    const int32_t vrounding = INT32_C(0x00200000);

    size_t c = channels;
    do {
      const int32_t vtl = (int32_t) *i0++;
      const int32_t vtr = (int32_t) *i1++;
      const int32_t vbl = (int32_t) *i2++;
      const int32_t vbr = (int32_t) *i3++;

      const int32_t vtd = vtr - vtl;
      const int32_t vbd = vbr - vbl;

      const int32_t vt = (int32_t) ((uint32_t) vtl << 11) + vtd * valphah;
      const int32_t vb = (int32_t) ((uint32_t) vbl << 11) + vbd * valphah;

      const int32_t vd = vb - vt;

      const int32_t vacc = (int32_t) ((uint32_t) vt << 11) + vd * valphav;

      const int32_t vo = math_asr_s32(vacc + vrounding, 22);

      *output++ = vo;

      c -= sizeof(int8_t);
    } while (c != 0);

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const int32_t voutput_max = params->scalar.max;
  const int32_t voutput_min = params->scalar.min;
  do {
    int8_t* o = output;
    {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      const int8_t* i8 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
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
      do {
        const int32_t vi0 = (int32_t) *i0++;
        const int32_t vi1 = (int32_t) *i1++;
        const int32_t vi2 = (int32_t) *i2++;
        const int32_t vi3 = (int32_t) *i3++;
        const int32_t vi4 = (int32_t) *i4++;
        const int32_t vi5 = (int32_t) *i5++;
        const int32_t vi6 = (int32_t) *i6++;
        const int32_t vi7 = (int32_t) *i7++;
        const int32_t vi8 = (int32_t) *i8++;

        const int32_t vmax01 = math_max_s32(vi0, vi1);
        const int32_t vmax23 = math_max_s32(vi2, vi3);
        const int32_t vmax45 = math_max_s32(vi4, vi5);
        const int32_t vmax67 = math_max_s32(vi6, vi7);
        const int32_t vmax018 = math_max_s32(vmax01, vi8);

        const int32_t vmax2345 = math_max_s32(vmax23, vmax45);
        const int32_t vmax01678 = math_max_s32(vmax018, vmax67);

        int32_t vout = math_max_s32(vmax2345, vmax01678);
        vout = math_min_s32(vout, voutput_max);
        vout = math_max_s32(vout, voutput_min);

        *o++ = (int8_t) vout;
      } while (--c != 0);
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
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
      do {
        const int32_t vi0 = (int32_t) *i0++;
        const int32_t vi1 = (int32_t) *i1++;
        const int32_t vi2 = (int32_t) *i2++;
        const int32_t vi3 = (int32_t) *i3++;
        const int32_t vi4 = (int32_t) *i4++;
        const int32_t vi5 = (int32_t) *i5++;
        const int32_t vi6 = (int32_t) *i6++;
        const int32_t vi7 = (int32_t) *i7++;
        const int32_t vi8 = (int32_t) *o;

        const int32_t vmax01 = math_max_s32(vi0, vi1);
        const int32_t vmax23 = math_max_s32(vi2, vi3);
        const int32_t vmax45 = math_max_s32(vi4, vi5);
        const int32_t vmax67 = math_max_s32(vi6, vi7);
        const int32_t vmax018 = math_max_s32(vmax01, vi8);

        const int32_t vmax2345 = math_max_s32(vmax23, vmax45);
        const int32_t vmax01678 = math_max_s32(vmax018, vmax67);

        int32_t vout = math_max_s32(vmax2345, vmax01678);
        vout = math_min_s32(vout, voutput_max);
        vout = math_max_s32(vout, voutput_min);

        *o++ = (int8_t) vout;
      } while (--c != 0);
    }
    input = (const int8_t**) ((uintptr_t) input + input_increment);
    output = (int8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_s8_vclamp_ukernel__scalar_x4(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t voutput_max = params->scalar.max;
  const int32_t voutput_min = params->scalar.min;

  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    int32_t vt0 = (int32_t) input[0];
    int32_t vt1 = (int32_t) input[1];
    int32_t vt2 = (int32_t) input[2];
    int32_t vt3 = (int32_t) input[3];
    input += 4;

    vt0 = math_max_s32(vt0, voutput_min);
    vt1 = math_max_s32(vt1, voutput_min);
    vt2 = math_max_s32(vt2, voutput_min);
    vt3 = math_max_s32(vt3, voutput_min);

    vt0 = math_min_s32(vt0, voutput_max);
    vt1 = math_min_s32(vt1, voutput_max);
    vt2 = math_min_s32(vt2, voutput_max);
    vt3 = math_min_s32(vt3, voutput_max);

    output[0] = (int8_t) vt0;
    output[1] = (int8_t) vt1;
    output[2] = (int8_t) vt2;
    output[3] = (int8_t) vt3;
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    do {
      int32_t vt = (int32_t) *input++;
      vt = math_max_s32(vt, voutput_min);
      vt = math_min_s32(vt, voutput_max);
      *output++ = (int8_t) vt;

      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}

void xnn_u8_ibilinear_ukernel__scalar_c1(
    size_t output_pixels,
    size_t channels,
    const uint8_t**restrict input,
    size_t input_offset,
    const int16_t*restrict weights,
    uint8_t*restrict output,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const uint8_t* i0 = (const uint8_t*) ((uintptr_t) input[0] + input_offset);
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input[1] + input_offset);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input[2] + input_offset);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const int32_t valphah = (int32_t) (uint32_t) (uint16_t) weights[0];
    const int32_t valphav = (int32_t) (uint32_t) (uint16_t) weights[1];
    weights += 2;

    const int32_t vrounding = INT32_C(0x00200000);

    size_t c = channels;
    do {
      const int32_t vtl = (int32_t) *i0++;
      const int32_t vtr = (int32_t) *i1++;
      const int32_t vbl = (int32_t) *i2++;
      const int32_t vbr = (int32_t) *i3++;

      const int32_t vtd = vtr - vtl;
      const int32_t vbd = vbr - vbl;

      const int32_t vt = (int32_t) ((uint32_t) vtl << 11) + vtd * valphah;
      const int32_t vb = (int32_t) ((uint32_t) vbl << 11) + vbd * valphah;

      const int32_t vd = vb - vt;

      const int32_t vacc = (int32_t) ((uint32_t) vt << 11) + vd * valphav;

      const int32_t vo = math_asr_s32(vacc + vrounding, 22);

      *output++ = vo;

      c -= sizeof(uint8_t);
    } while (c != 0);

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const uint32_t voutput_min = params->scalar.min;
  const uint32_t voutput_max = params->scalar.max;
  do {
    uint8_t* o = output;
    {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      const uint8_t* i8 = *input++;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
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
      do {
        const uint32_t vi0 = (uint32_t) *i0++;
        const uint32_t vi1 = (uint32_t) *i1++;
        const uint32_t vi2 = (uint32_t) *i2++;
        const uint32_t vi3 = (uint32_t) *i3++;
        const uint32_t vi4 = (uint32_t) *i4++;
        const uint32_t vi5 = (uint32_t) *i5++;
        const uint32_t vi6 = (uint32_t) *i6++;
        const uint32_t vi7 = (uint32_t) *i7++;
        const uint32_t vi8 = (uint32_t) *i8++;

        const uint32_t vmax01 = math_max_u32(vi0, vi1);
        const uint32_t vmax23 = math_max_u32(vi2, vi3);
        const uint32_t vmax45 = math_max_u32(vi4, vi5);
        const uint32_t vmax67 = math_max_u32(vi6, vi7);
        const uint32_t vmax018 = math_max_u32(vmax01, vi8);

        const uint8_t vmax2345 = math_max_u32(vmax23, vmax45);
        const uint8_t vmax01678 = math_max_u32(vmax018, vmax67);

        uint32_t vout = math_max_u32(vmax2345, vmax01678);
        vout = math_max_u32(vout, voutput_min);
        vout = math_min_u32(vout, voutput_max);

        *o++ = vout;
      } while (--c != 0);
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
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
      do {
        const uint32_t vi0 = (uint32_t) *i0++;
        const uint32_t vi1 = (uint32_t) *i1++;
        const uint32_t vi2 = (uint32_t) *i2++;
        const uint32_t vi3 = (uint32_t) *i3++;
        const uint32_t vi4 = (uint32_t) *i4++;
        const uint32_t vi5 = (uint32_t) *i5++;
        const uint32_t vi6 = (uint32_t) *i6++;
        const uint32_t vi7 = (uint32_t) *i7++;
        const uint32_t vi8 = (uint32_t) *o;

        const uint32_t vmax01 = math_max_u32(vi0, vi1);
        const uint32_t vmax23 = math_max_u32(vi2, vi3);
        const uint32_t vmax45 = math_max_u32(vi4, vi5);
        const uint32_t vmax67 = math_max_u32(vi6, vi7);
        const uint32_t vmax018 = math_max_u32(vmax01, vi8);

        const uint32_t vmax2345 = math_max_u32(vmax23, vmax45);
        const uint32_t vmax01678 = math_max_u32(vmax018, vmax67);

        uint32_t vout = math_max_u32(vmax2345, vmax01678);
        vout = math_max_u32(vout, voutput_min);
        vout = math_min_u32(vout, voutput_max);

        *o++ = vout;
      } while (--c != 0);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    output = (uint8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_u8_rmax_ukernel__scalar(
    size_t batch,
    const uint8_t* input,
    uint8_t* output)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint8_t vmax0 = 0;
  uint8_t vmax1 = 0;
  for (; batch >= 2 * sizeof(uint8_t); batch -= 2 * sizeof(uint8_t)) {
    const uint8_t vt0 = input[0];
    const uint8_t vt1 = input[1];
    input += 2;

    vmax0 = vt0 > vmax0 ? vt0 : vmax0;
    vmax1 = vt1 > vmax1 ? vt1 : vmax1;
  }
  uint8_t vmax = vmax0 > vmax1 ? vmax0 : vmax1;
  if (batch != 0) {
    const uint8_t vt = *input++;
    vmax = vt > vmax ? vt : vmax;
  }
  *output = vmax;
}

void xnn_u8_vclamp_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t voutput_max = params->scalar.max;
  const uint32_t voutput_min = params->scalar.min;

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    uint32_t vt0 = (uint32_t) input[0];
    uint32_t vt1 = (uint32_t) input[1];
    uint32_t vt2 = (uint32_t) input[2];
    uint32_t vt3 = (uint32_t) input[3];
    input += 4;

    vt0 = math_max_u32(vt0, voutput_min);
    vt1 = math_max_u32(vt1, voutput_min);
    vt2 = math_max_u32(vt2, voutput_min);
    vt3 = math_max_u32(vt3, voutput_min);

    vt0 = math_min_u32(vt0, voutput_max);
    vt1 = math_min_u32(vt1, voutput_max);
    vt2 = math_min_u32(vt2, voutput_max);
    vt3 = math_min_u32(vt3, voutput_max);

    output[0] = (uint8_t) vt0;
    output[1] = (uint8_t) vt1;
    output[2] = (uint8_t) vt2;
    output[3] = (uint8_t) vt3;
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    do {
      uint32_t vt = (uint32_t) *input++;
      vt = math_max_u32(vt, voutput_min);
      vt = math_min_u32(vt, voutput_max);
      *output++ = (uint8_t) vt;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_x16_transposec_ukernel__2x4_scalar_int(
    const uint16_t *input,
    uint16_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x16_transpose_params* params) XNN_OOB_READS
{
  assert(output_stride >= block_height * sizeof(int16_t));
  assert(input_stride >= block_width * sizeof(int16_t));

  const size_t tile_height = 2;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(int16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(int16_t);
  const size_t input_offset = tile_height * input_stride;

  const int16_t* i0 = (const int16_t*) input;
  const int16_t* i1 = (const int16_t*) ((uintptr_t) i0 + input_stride);

  int16_t* o0 = (int16_t*) output;
  int16_t* o1 = (int16_t*) ((uintptr_t) o0 + output_stride);
  int16_t* o2 = (int16_t*) ((uintptr_t) o1 + output_stride);
  int16_t* o3 = (int16_t*) ((uintptr_t) o2 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      *o3++ = i0[3];
      *o3++ = i1[3];
      *o2++ = i0[2];
      *o2++ = i1[2];
      *o1++ = i0[1];
      *o1++ = i1[1];
      *o0++ = i0[0];
      *o0++ = i1[0];
      i0 = (const int16_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int16_t*) ((uintptr_t) i1 + input_offset);
    }
    if (bh & 1) {
      o3[0] = i0[3];
      o2[0] = i0[2];
      o1[0] = i0[1];
      o0[0] = i0[0];
    }

    i0 = (const int16_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const int16_t*) ((uintptr_t) i0 + input_stride);
    o0 = (int16_t*) ((uintptr_t) o0 + output_reset);
    o1 = (int16_t*) ((uintptr_t) o1 + output_reset);
    o2 = (int16_t*) ((uintptr_t) o2 + output_reset);
    o3 = (int16_t*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x32_packx_ukernel_2x__scalar(
    size_t m,
    size_t k,
    const uint32_t* restrict x,
    size_t x_stride,
    uint32_t* restrict y)
{
  assert(m != 0);
  assert(k != 0);

  const float* x0 = (const float*) x;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (m != 2) {
    x1 = x0;
  }

  float*restrict y_f32 = (float*) y;

  do {
    const float vx0 = *x0++;
    const float vx1 = *x1++;

    y_f32[0] = vx0;
    y_f32[1] = vx1;
    y_f32 += 2;
  } while (--k != 0);
}

void xnn_x32_packx_ukernel_3x__scalar(
    size_t m,
    size_t k,
    const uint32_t* restrict x,
    size_t x_stride,
    uint32_t* restrict y)
{
  assert(m != 0);
  assert(k != 0);

  const float* x0 = (const float*) x;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (m < 2) {
    x1 = x0;
  }
  const float* x2 = (const float*) ((uintptr_t) x1 + x_stride);
  if (m <= 2) {
    x2 = x1;
  }

  float*restrict y_f32 = (float*) y;

  do {
    const float vx0 = *x0++;
    const float vx1 = *x1++;
    const float vx2 = *x2++;

    y_f32[0] = vx0;
    y_f32[1] = vx1;
    y_f32[2] = vx2;
    y_f32 += 3;
  } while (--k != 0);
}

void xnn_x32_packx_ukernel_4x__scalar(
    size_t m,
    size_t k,
    const uint32_t* restrict x,
    size_t x_stride,
    uint32_t* restrict y)
{
  assert(m != 0);
  assert(k != 0);

  const float* x0 = (const float*) x;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (m < 2) {
    x1 = x0;
  }
  const float* x2 = (const float*) ((uintptr_t) x1 + x_stride);
  if (m <= 2) {
    x2 = x1;
  }
  const float* x3 = (const float*) ((uintptr_t) x2 + x_stride);
  if (m != 4) {
    x3 = x2;
  }

  float*restrict y_f32 = (float*) y;

  do {
    const float vx0 = *x0++;
    const float vx1 = *x1++;
    const float vx2 = *x2++;
    const float vx3 = *x3++;

    y_f32[0] = vx0;
    y_f32[1] = vx1;
    y_f32[2] = vx2;
    y_f32[3] = vx3;
    y_f32 += 4;
  } while (--k != 0);
}

void xnn_x32_transposec_ukernel__2x4_scalar_int(
    const uint32_t *input,
    uint32_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x32_transpose_params* params) XNN_OOB_READS
{
  assert(output_stride >= block_height * sizeof(int));
  assert(input_stride >= block_width * sizeof(int));

  const size_t tile_height = 2;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(int);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(int);
  const size_t input_offset = tile_height * input_stride;

  const int* i0 = (const int*) input;
  const int* i1 = (const int*) ((uintptr_t) i0 + input_stride);

  int* o0 = (int*) output;
  int* o1 = (int*) ((uintptr_t) o0 + output_stride);
  int* o2 = (int*) ((uintptr_t) o1 + output_stride);
  int* o3 = (int*) ((uintptr_t) o2 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      *o3++ = i0[3];
      *o3++ = i1[3];
      *o2++ = i0[2];
      *o2++ = i1[2];
      *o1++ = i0[1];
      *o1++ = i1[1];
      *o0++ = i0[0];
      *o0++ = i1[0];
      i0 = (const int*) ((uintptr_t) i0 + input_offset);
      i1 = (const int*) ((uintptr_t) i1 + input_offset);
    }
    if (bh & 1) {
      o3[0] = i0[3];
      o2[0] = i0[2];
      o1[0] = i0[1];
      o0[0] = i0[0];
    }

    i0 = (const int*) ((uintptr_t) i0 + input_reset);
    i1 = (const int*) ((uintptr_t) i0 + input_stride);
    o0 = (int*) ((uintptr_t) o0 + output_reset);
    o1 = (int*) ((uintptr_t) o1 + output_reset);
    o2 = (int*) ((uintptr_t) o2 + output_reset);
    o3 = (int*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x32_unpool_ukernel__scalar(
    size_t kernel_elements,
    size_t channels,
    uint32_t fill,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output)
{
  // Pre-initialize outputs with constant.
  uint32_t** os = output;
  do {
    uint32_t* o = *os++;
    size_t c = channels;
    do {
      *o++ = fill;
    } while (--c != 0);
  } while (--kernel_elements != 0);

  // Copy indexed elements to output.
  size_t offset = 0;
  do {
    const uint32_t i = *index++;
    *((uint32_t*) ((uintptr_t) output[i] + offset)) = *input++;
    offset += sizeof(uint32_t);
  } while (--channels != 0);
}

void xnn_x32_zip_x2_ukernel__scalar(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);

  do {
    const uint32_t vx = *x++;
    const uint32_t vy = *y++;
    output[0] = vx;
    output[1] = vy;
    output += 2;

    n -= 4;
  } while (n != 0);
}

void xnn_x32_zip_x3_ukernel__scalar(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  const uint32_t* z = (const uint32_t*) ((uintptr_t) y + n);
  uint32_t* o = output;

  do {
    const uint32_t vx = *x++;
    const uint32_t vy = *y++;
    const uint32_t vz = *z++;
    o[0] = vx;
    o[1] = vy;
    o[2] = vz;
    o += 3;

    n -= 4;
  } while (n != 0);
}

void xnn_x32_zip_x4_ukernel__scalar(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  const uint32_t* z = (const uint32_t*) ((uintptr_t) y + n);
  const uint32_t* w = (const uint32_t*) ((uintptr_t) z + n);
  uint32_t* o = output;

  do {
    const uint32_t vx = *x++;
    const uint32_t vy = *y++;
    const uint32_t vz = *z++;
    const uint32_t vw = *w++;
    o[0] = vx;
    o[1] = vy;
    o[2] = vz;
    o[3] = vw;
    o += 4;

    n -= 4;
  } while (n != 0);
}

void xnn_x32_zip_xm_ukernel__scalar(
    size_t n,
    size_t m,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);
  assert(m >= 4);

  size_t k = n;
  do {
    size_t l = m;
    const uint32_t* input_column = input++;
    do {
      *output++ = *input_column;
      input_column = (uint32_t*) ((uintptr_t) input_column + n);
    } while (--l != 0);
    k -= 4;
  } while (k != 0);
}

void xnn_x8_transposec_ukernel__2x4_scalar_int(
    const uint8_t *input,
    uint8_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x8_transpose_params* params) XNN_OOB_READS
{
  assert(output_stride >= block_height * sizeof(int8_t));
  assert(input_stride >= block_width * sizeof(int8_t));

  const size_t tile_height = 2;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(int8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(int8_t);
  const size_t input_offset = tile_height * input_stride;

  const int8_t* i0 = (const int8_t*) input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);

  int8_t* o0 = (int8_t*) output;
  int8_t* o1 = (int8_t*) ((uintptr_t) o0 + output_stride);
  int8_t* o2 = (int8_t*) ((uintptr_t) o1 + output_stride);
  int8_t* o3 = (int8_t*) ((uintptr_t) o2 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      *o3++ = i0[3];
      *o3++ = i1[3];
      *o2++ = i0[2];
      *o2++ = i1[2];
      *o1++ = i0[1];
      *o1++ = i1[1];
      *o0++ = i0[0];
      *o0++ = i1[0];
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    if (bh & 1) {
      o3[0] = i0[3];
      o2[0] = i0[2];
      o1[0] = i0[1];
      o0[0] = i0[0];
    }

    i0 = (const int8_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
    o0 = (int8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (int8_t*) ((uintptr_t) o1 + output_reset);
    o2 = (int8_t*) ((uintptr_t) o2 + output_reset);
    o3 = (int8_t*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x8_zip_x2_ukernel__scalar(
    size_t n,
    const uint8_t* input,
    uint8_t* output)
{
  assert(n != 0);

  const uint8_t* x = input;
  const uint8_t* y = (const uint8_t*) ((uintptr_t) x + n);
  uint8_t* o = output;

  do {
    const uint8_t vx = *x++;
    const uint8_t vy = *y++;
    o[0] = vx;
    o[1] = vy;
    o += 2;

    n -= sizeof(uint8_t);
  } while (n != 0);
}

void xnn_x8_zip_x3_ukernel__scalar(
    size_t n,
    const uint8_t* input,
    uint8_t* output)
{
  const uint8_t* x = input;
  const uint8_t* y = (const uint8_t*) ((uintptr_t) x + n);
  const uint8_t* z = (const uint8_t*) ((uintptr_t) y + n);
  uint8_t* o = output;

  do {
    const uint8_t vx = *x++;
    const uint8_t vy = *y++;
    const uint8_t vz = *z++;
    o[0] = vx;
    o[1] = vy;
    o[2] = vz;
    o += 3;

    n -= sizeof(uint8_t);
  } while (n != 0);
}

void xnn_x8_zip_x4_ukernel__scalar(
    size_t n,
    const uint8_t* input,
    uint8_t* output)
{
  assert(n != 0);

  const uint8_t* x = input;
  const uint8_t* y = (const uint8_t*) ((uintptr_t) x + n);
  const uint8_t* z = (const uint8_t*) ((uintptr_t) y + n);
  const uint8_t* w = (const uint8_t*) ((uintptr_t) z + n);
  uint8_t* o = output;

  do {
    const uint8_t vx = *x++;
    const uint8_t vy = *y++;
    const uint8_t vz = *z++;
    const uint8_t vw = *w++;
    o[0] = vx;
    o[1] = vy;
    o[2] = vz;
    o[3] = vw;
    o += 4;

    n -= sizeof(uint8_t);
  } while (n != 0);
}

void xnn_x8_zip_xm_ukernel__scalar(
    size_t n,
    size_t m,
    const uint8_t* input,
    uint8_t* output)
{
  assert(n != 0);
  assert(m >= 4);

  size_t k = n;
  do {
    size_t l = m;
    const uint8_t* input_column = input++;
    do {
      *output++ = *input_column;
      input_column = (uint8_t*) ((uintptr_t) input_column + n);
    } while (--l != 0);
    k -= sizeof(uint8_t);
  } while (k != 0);
}

void xnn_xx_fill_ukernel__scalar_x16(
    size_t rows,
    size_t channels,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern)
{
  assert(rows != 0);
  assert(channels != 0);

  const size_t output_increment = output_stride - channels;

  do {
    uint32_t vfill_pattern = fill_pattern;
    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      unaligned_indexed_store_u32(output, 0, vfill_pattern);
      unaligned_indexed_store_u32(output, 1, vfill_pattern);
      unaligned_indexed_store_u32(output, 2, vfill_pattern);
      unaligned_indexed_store_u32(output, 3, vfill_pattern);
      output = ((uint8_t*) output + 16);
    }
    if XNN_UNLIKELY(c != 0) {
      if XNN_LIKELY(c & (8 * sizeof(uint8_t))) {
        unaligned_indexed_store_u32(output, 0, vfill_pattern);
        unaligned_indexed_store_u32(output, 1, vfill_pattern);
        output = ((uint8_t*) output + 8);
      }
      if XNN_LIKELY(c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, vfill_pattern);
        output = ((uint8_t*) output + 4);
      }
      if XNN_LIKELY(c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_pattern);
        vfill_pattern >>= 16;
        output = ((uint8_t*) output + 2);
      }
      if XNN_LIKELY(c & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_pattern;
        output = ((uint8_t*) output + 1);
      }
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}

void xnn_xx_pad_ukernel__scalar(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern) XNN_OOB_READS
{
  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      uint32_t vfill_pattern = fill_pattern;
      for (; l >= 4 * sizeof(uint8_t); l -= 4 * sizeof(uint8_t)) {
        unaligned_store_u32(output, vfill_pattern);
        output = (uint8_t*) output + 4;
      }
      if XNN_LIKELY(l & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_pattern);
        vfill_pattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if XNN_LIKELY(l & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_pattern;
        output = (uint8_t*) output + 1;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      const uint32_t vdata0 = unaligned_indexed_load_u32(input, 0);
      const uint32_t vdata1 = unaligned_indexed_load_u32(input, 1);
      const uint32_t vdata2 = unaligned_indexed_load_u32(input, 2);
      const uint32_t vdata3 = unaligned_indexed_load_u32(input, 3);
      input = (const uint8_t*) input + 16;

      unaligned_indexed_store_u32(output, 0, vdata0);
      unaligned_indexed_store_u32(output, 1, vdata1);
      unaligned_indexed_store_u32(output, 2, vdata2);
      unaligned_indexed_store_u32(output, 3, vdata3);
      output = (uint8_t*) output + 16;
    }
    if XNN_UNLIKELY(c != 0) {
      for (; c >= 4 * sizeof(uint8_t); c -= 4 * sizeof(uint8_t)) {
        unaligned_store_u32(output, unaligned_load_u32(input));
        input = (const uint8_t*) input + 4;
        output = (uint8_t*) output + 4;
      }
      if XNN_UNLIKELY(c != 0) {
        uint32_t vdata = unaligned_load_u32(input);
        input = (const void*) ((uintptr_t) input + c);

        if XNN_LIKELY(c & (2 * sizeof(uint8_t))) {
          unaligned_store_u16(output, vdata);
          vdata >>= 16;
          output = (uint8_t*) output + 2;
        }
        if XNN_LIKELY(c & (1 * sizeof(uint8_t))) {
          *((uint8_t*) output) = (uint8_t) vdata;
          output = (uint8_t*) output + 1;
        }
      }
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      uint32_t vfill_pattern = fill_pattern;
      for (; r >= 4 * sizeof(uint8_t); r -= 4 * sizeof(uint8_t)) {
        unaligned_store_u32(output, vfill_pattern);
        output = (uint8_t*) output + 4;
      }
      if XNN_LIKELY(r & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, vfill_pattern);
        vfill_pattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if XNN_LIKELY(r & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_pattern;
        output = (uint8_t*) output + 1;
      }
    }

    input = (const uint32_t*) ((uintptr_t) input + input_increment);
    output = (uint32_t*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
