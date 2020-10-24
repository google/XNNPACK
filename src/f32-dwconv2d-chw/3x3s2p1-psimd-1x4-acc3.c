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


void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__psimd_1x4_acc3(
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

  const psimd_s32 vmask_even = psimd_load_s32(params->scalar.mask_even);
  const psimd_s32 vmask_odd  = psimd_load_s32(params->scalar.mask_odd);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);

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

  const size_t input_decrement = round_down_po2(input_width, 4 /* SIMD output width */ * 2 /* subsampling */ * sizeof(float));

  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height <= 3) {
      i2 = zero;
    }

    psimd_f32 vi0x7531 = psimd_zero_f32();
    psimd_f32 vi1x7531 = psimd_zero_f32();
    psimd_f32 vi2x7531 = psimd_zero_f32();

    size_t w = input_width;
    for (; w >= 8 * sizeof(float); w -= 8 * sizeof(float)) {
      psimd_f32 vo8ACEp0 = vbias;

      const psimd_f32 vi0x89AB = psimd_load_f32(i0);
      const psimd_f32 vi1x89AB = psimd_load_f32(i1);
      const psimd_f32 vi2x89AB = psimd_load_f32(i2);

      const psimd_f32 vi0xCDEF = psimd_load_f32(i0 + 4);
      i0 += 8;
      const psimd_f32 vi1xCDEF = psimd_load_f32(i1 + 4);
      i1 += 8;
      const psimd_f32 vi2xCDEF = psimd_load_f32(i2 + 4);
      i2 += 8;

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
      output += 4;
    }
    // Potentially process the last block of 0..7 pixels.
    assert(w < 8 * sizeof(float));
    if XNN_LIKELY(w != 0) {
      psimd_f32 vo8ACEp0 = vbias;

      const psimd_f32 vi0x89AB = psimd_load_f32(i0);
      const psimd_f32 vi1x89AB = psimd_load_f32(i1);
      const psimd_f32 vi2x89AB = psimd_load_f32(i2);

      const psimd_f32 vi0xCDEF = psimd_load_f32(i0 + 4);
      const psimd_f32 vi1xCDEF = psimd_load_f32(i1 + 4);
      const psimd_f32 vi2xCDEF = psimd_load_f32(i2 + 4);

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

      if (w == 7 * sizeof(float)) {
        psimd_store_f32(output, vo);
        output += 4;
      } else {
        w += 1 * sizeof(float);
        if (w & (4 * sizeof(float))) {
          psimd_store2_f32(output, vo);
          output += 2;
          vo = psimd_concat_hi_f32(vo, vo);
        }
        if (w & (2 * sizeof(float))) {
          psimd_store1_f32(output, vo);
          output += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i0 + input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);

    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
