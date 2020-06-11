// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>

PSIMD_INTRINSIC psimd_f32 rotright_f32(psimd_f32 a) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, a, 3, 0, 1, 2);
  #else
    return __builtin_shuffle(a, (psimd_s32){3, 0, 1, 2});
  #endif  // defined(__clang__)
}

PSIMD_INTRINSIC psimd_f32 movess_f32(psimd_f32 a, psimd_f32 b) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, b, 4, 1, 2, 3);
  #else
    return __builtin_shuffle(a, b, (psimd_s32){4, 1, 2, 3});
  #endif  // defined(__clang__)
}


void xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_width_stride,
    size_t output_width_stride,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height!= 0);
  assert(input_width != 0);
  assert(padding_top >= 0 && padding_top <= 1);

  const size_t padded_input_height = input_height + padding_top + 1 /* padding_bottom */;
  const size_t output_height = (padded_input_height - 3) / 2 + 1;

  const psimd_s32 vmask_even = psimd_load_s32(params->scalar.mask_even);
  const psimd_s32 vmask_odd  = psimd_load_s32(params->scalar.mask_odd);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);

  const size_t input_width_decrement_single = input_width / 8  * input_tuple_stride * 2;
  const size_t input_width_increment = input_width_stride * 2 - input_width_decrement_single;
  const size_t output_width_increment = output_width_stride - input_width / 8 * output_tuple_stride;

  const float* i0;
  const float* i1;
  const float* i2;

  if (padding_top == 0) {
    i0 = input;
    i1 = (const float*) ((uintptr_t) i0 + input_width_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
    if (input_height <= 2) {
      i2 = zero;
    }
    if (input_height == 1) {
      i1 = zero;
    }
  } else {
    i0 = zero;
    i1 = input;
    i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
    if (input_height == 1) {
      i2 = zero;
    }
  }

  const psimd_f32 vbias = psimd_load_splat_f32(weights);
  const psimd_f32 vk00 = psimd_load_splat_f32(weights + 1);
  const psimd_f32 vk01 = psimd_load_splat_f32(weights + 2);
  const psimd_f32 vk02 = psimd_load_splat_f32(weights + 3);
  const psimd_f32 vk10 = psimd_load_splat_f32(weights + 4);
  const psimd_f32 vk11 = psimd_load_splat_f32(weights + 5);
  const psimd_f32 vk12 = psimd_load_splat_f32(weights + 6);
  const psimd_f32 vk20 = psimd_load_splat_f32(weights + 7);
  const psimd_f32 vk21 = psimd_load_splat_f32(weights + 8);
  const psimd_f32 vk22 = psimd_load_splat_f32(weights + 9);

  size_t m = output_height;
  do {
    psimd_f32 vi0x7531 = psimd_zero_f32();
    psimd_f32 vi1x7531 = psimd_zero_f32();
    psimd_f32 vi2x7531 = psimd_zero_f32();

    size_t k = input_width;
    for (; k >= 8; k -= 8) {
      psimd_f32 vo8ACEp0 = vbias;

      const psimd_f32 vi0x89AB = psimd_load_f32(i0);
      i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const psimd_f32 vi1x89AB = psimd_load_f32(i1);
      i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const psimd_f32 vi2x89AB = psimd_load_f32(i2);
      i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

      const psimd_f32 vi0xCDEF = psimd_load_f32(i0);
      i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const psimd_f32 vi1xCDEF = psimd_load_f32(i1);
      i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const psimd_f32 vi2xCDEF = psimd_load_f32(i2);
      i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

      const psimd_f32 vi0x8ACE = psimd_concat_even_f32(vi0x89AB, vi0xCDEF);
      const psimd_f32 vi0x9BDF = psimd_concat_odd_f32(vi0x89AB, vi0xCDEF);
      const psimd_f32 vi1x8ACE = psimd_concat_even_f32(vi1x89AB, vi1xCDEF);
      const psimd_f32 vi1x9BDF = psimd_concat_odd_f32(vi1x89AB, vi1xCDEF);
      const psimd_f32 vi2x8ACE = psimd_concat_even_f32(vi2x89AB, vi2xCDEF);
      const psimd_f32 vi2x9BDF = psimd_concat_odd_f32(vi2x89AB, vi2xCDEF);

      vo8ACEp0 = psimd_add_f32(vo8ACEp0, psimd_mul_f32(vi0x8ACE, vk01));
      psimd_f32 vo8ACEp1 = psimd_mul_f32(vi1x8ACE, vk11);
      psimd_f32 vo8ACEp2 = psimd_mul_f32(vi2x8ACE, vk21);

      const psimd_f32 vi0xF9BD = rotright_f32(vi0x9BDF);
      const psimd_f32 vi1xF9BD = rotright_f32(vi1x9BDF);
      const psimd_f32 vi2xF9BD = rotright_f32(vi2x9BDF);

      vo8ACEp0 = psimd_add_f32(vo8ACEp0, psimd_mul_f32(vi0x9BDF, vk02));
      vo8ACEp1 = psimd_add_f32(vo8ACEp1, psimd_mul_f32(vi1x9BDF, vk12));
      vo8ACEp2 = psimd_add_f32(vo8ACEp2, psimd_mul_f32(vi2x9BDF, vk22));

      const psimd_f32 vi0x7BDF = movess_f32(vi0xF9BD, vi0x7531);
      const psimd_f32 vi1x7BDF = movess_f32(vi1xF9BD, vi1x7531);
      const psimd_f32 vi2x7BDF = movess_f32(vi2xF9BD, vi2x7531);

      vi0x7531 = vi0xF9BD;
      vi1x7531 = vi1xF9BD;
      vi2x7531 = vi2xF9BD;

      vo8ACEp0 = psimd_add_f32(vo8ACEp0, psimd_mul_f32(vi0x7BDF, vk00));
      vo8ACEp1 = psimd_add_f32(vo8ACEp1, psimd_mul_f32(vi1x7BDF, vk10));
      vo8ACEp2 = psimd_add_f32(vo8ACEp2, psimd_mul_f32(vi2x7BDF, vk20));

      psimd_f32 vo = psimd_add_f32(vo8ACEp0, vo8ACEp1);
      vo = psimd_add_f32(vo, vo8ACEp2);

      vo = psimd_max_f32(vo, vmin);
      vo = psimd_min_f32(vo, vmax);

      psimd_store_f32(output, vo);
      output = (float*) ((uintptr_t) output + output_tuple_stride);
    }
    // Last block has 0-7 pixels to process.
    assert(k < 8);
    if XNN_LIKELY(k != 0) {
      psimd_f32 vo8ACEp0 = vbias;

      const psimd_f32 vi0x89AB = psimd_load_f32(i0);
      const psimd_f32 vi1x89AB = psimd_load_f32(i1);
      const psimd_f32 vi2x89AB = psimd_load_f32(i2);

      const psimd_f32 vi0xCDEF = psimd_load_f32((const float*) ((uintptr_t) i0 + input_tuple_stride));
      const psimd_f32 vi1xCDEF = psimd_load_f32((const float*) ((uintptr_t) i1 + input_tuple_stride));
      const psimd_f32 vi2xCDEF = psimd_load_f32((const float*) ((uintptr_t) i2 + input_tuple_stride));

      const psimd_f32 vi0x8ACE = psimd_andmask_f32(vmask_even, psimd_concat_even_f32(vi0x89AB, vi0xCDEF));
      const psimd_f32 vi0x9BDF = psimd_andmask_f32(vmask_odd,  psimd_concat_odd_f32(vi0x89AB, vi0xCDEF));
      const psimd_f32 vi1x8ACE = psimd_andmask_f32(vmask_even, psimd_concat_even_f32(vi1x89AB, vi1xCDEF));
      const psimd_f32 vi1x9BDF = psimd_andmask_f32(vmask_odd,  psimd_concat_odd_f32(vi1x89AB, vi1xCDEF));
      const psimd_f32 vi2x8ACE = psimd_andmask_f32(vmask_even, psimd_concat_even_f32(vi2x89AB, vi2xCDEF));
      const psimd_f32 vi2x9BDF = psimd_andmask_f32(vmask_odd,  psimd_concat_odd_f32(vi2x89AB, vi2xCDEF));

      vo8ACEp0 = psimd_add_f32(vo8ACEp0, psimd_mul_f32(vi0x8ACE, vk01));
      psimd_f32 vo8ACEp1 = psimd_mul_f32(vi1x8ACE, vk11);
      psimd_f32 vo8ACEp2 = psimd_mul_f32(vi2x8ACE, vk21);

      const psimd_f32 vi0xF9BD = rotright_f32(vi0x9BDF);
      const psimd_f32 vi1xF9BD = rotright_f32(vi1x9BDF);
      const psimd_f32 vi2xF9BD = rotright_f32(vi2x9BDF);

      vo8ACEp0 = psimd_add_f32(vo8ACEp0, psimd_mul_f32(vi0x9BDF, vk02));
      vo8ACEp1 = psimd_add_f32(vo8ACEp1, psimd_mul_f32(vi1x9BDF, vk12));
      vo8ACEp2 = psimd_add_f32(vo8ACEp2, psimd_mul_f32(vi2x9BDF, vk22));

      const psimd_f32 vi0x7BDF = movess_f32(vi0xF9BD, vi0x7531);
      const psimd_f32 vi1x7BDF = movess_f32(vi1xF9BD, vi1x7531);
      const psimd_f32 vi2x7BDF = movess_f32(vi2xF9BD, vi2x7531);

      vo8ACEp0 = psimd_add_f32(vo8ACEp0, psimd_mul_f32(vi0x7BDF, vk00));
      vo8ACEp1 = psimd_add_f32(vo8ACEp1, psimd_mul_f32(vi1x7BDF, vk10));
      vo8ACEp2 = psimd_add_f32(vo8ACEp2, psimd_mul_f32(vi2x7BDF, vk20));

      psimd_f32 vo = psimd_add_f32(vo8ACEp0, vo8ACEp1);
      vo = psimd_add_f32(vo, vo8ACEp2);

      vo = psimd_max_f32(vo, vmin);
      vo = psimd_min_f32(vo, vmax);

      if (k == 7) {
        psimd_store_f32(output, vo);
      } else {
        float* output_lo = output;
        k += 1;
        if (k & 4) {
          psimd_store2_f32(output_lo, vo);
          output_lo += 2;
          vo = psimd_concat_hi_f32(vo, vo);
        }
        if (k & 2) {
          psimd_store1_f32(output_lo, vo);
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_width_decrement_single);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment);
    output = (float*) ((uintptr_t) output + output_width_increment);
    m -= 1;
    if (m == 1 && padding_top == input_height % 2) {
      // to mimic the following code with only one if, we do some small
      // shenanigans...
      // if (padding_top == 0 && input_height % 2 == 0) {
      //   i2 = zero;
      // } else if (padding_top == 1 && input_height % 2 == 1) {
      //   i2 = zero;
      // }
      i2 = zero;
    }
  } while (m != 0);
}
