// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <fxdiv.h>
#include <string.h>

#include <xnnpack/common.h>
#include <xnnpack/lut.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>
#include <xnnpack/vunary.h>


static inline uint32_t compute_sum(
    size_t n,
    const uint8_t* x,
    const uint32_t* t)
{
  assert(n != 0);

  uint32_t vsum = 0;
  do {
    const size_t vx = *x++;
    vsum += t[vx];
  } while (--n != 0);
  return vsum;
}

void xnn_u8_lut32norm_ukernel__scalar(
    size_t n,
    const uint8_t* x,
    const uint32_t* t,
    uint8_t* y)
{
  assert(n != 0);

  const uint32_t vsum = compute_sum(n, x, t);
  assert(vsum != 0);

  struct fxdiv_divisor_uint32_t vsum_divisor = fxdiv_init_uint32_t(vsum);
  const uint32_t vrounding = (vsum >> 1);
  do {
    const size_t vx = *x++;
    const uint32_t vt = t[vx];
    const uint32_t vq = fxdiv_quotient_uint32_t((vt << 8) + vrounding, vsum_divisor);
    const uint8_t vy = vq > 255 ? UINT8_C(255) : (uint8_t) vq;
    *y++ = vy;
  } while (--n != 0);
}

void xnn_x24_transposec_ukernel__1x2_scalar(
    const void *input,
    void * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x24_transpose_params* params) XNN_OOB_READS
{
  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  const size_t input_reset = 6 - block_height * input_stride;
  const size_t output_reset = 2 * output_stride - block_height * 3;
  const size_t input_offset = 1 * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 1; bh -= 1) {
      o1[0] = i0[3];
      o1[1] = i0[4];
      o1[2] = i0[5];
      o1 += 3;
      o0[0] = i0[0];
      o0[1] = i0[1];
      o0[2] = i0[2];
      o0 += 3;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_reset);
    block_width = doz(block_width, 2);
  } while (block_width != 0);
}

void xnn_x8_lut_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const size_t vx0 = (size_t) input[0];
    const size_t vx1 = (size_t) input[1];
    const size_t vx2 = (size_t) input[2];
    const size_t vx3 = (size_t) input[3];
    input += 4;

    const uint32_t vt0 = (uint32_t) table[vx0];
    const uint32_t vt1 = (uint32_t) table[vx1];
    const uint32_t vt2 = (uint32_t) table[vx2];
    const uint32_t vt3 = (uint32_t) table[vx3];

    output[0] = (uint8_t) vt0;
    output[1] = (uint8_t) vt1;
    output[2] = (uint8_t) vt2;
    output[3] = (uint8_t) vt3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const size_t vx = (size_t) *input++;
      const uint32_t vt = (uint32_t) table[vx];
      *output++ = (uint8_t) vt;
      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_xx_copy_ukernel__scalar_memcpy(size_t batch, const void* input, void* output, const void* params) {
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  memcpy(output, input, batch);
}

void xnn_xx_transposev_ukernel__1x1_scalar_memcpy(
    const void* input,
    void* output,
    size_t input_row_stride,
    size_t output_row_stride,
    size_t input_element_stride,
    size_t output_element_stride,
    size_t element_size,
    size_t block_width,
    size_t block_height)
{
  const size_t input_reset = input_element_stride - block_height * input_row_stride;
  const size_t output_reset = output_row_stride - block_height * output_element_stride;

  const void* i = (const void*) input;
  void* o = (void*) output;

  do {
    size_t bh = block_height;
    for (; bh >= 1; bh -= 1) {
      memcpy(o, i, element_size);
      i = (const void*) ((uintptr_t) i + input_row_stride);
      o = (void*) ((uintptr_t) o + output_element_stride);
    }

    i = (const void*) ((uintptr_t) i + input_reset);
    o = (void*) ((uintptr_t) o + output_reset);
    block_width -= 1;
  } while (block_width != 0);
}
