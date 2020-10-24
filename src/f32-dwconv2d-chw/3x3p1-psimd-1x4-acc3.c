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

PSIMD_INTRINSIC psimd_f32 rotleft_f32(psimd_f32 a) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, a, 1, 2, 3, 0);
  #else
    return __builtin_shuffle(a, (psimd_s32){1, 2, 3, 0});
  #endif  // defined(__clang__)
}

PSIMD_INTRINSIC psimd_f32 movess_f32(psimd_f32 a, psimd_f32 b) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, b, 4, 1, 2, 3);
  #else
    return __builtin_shuffle(a, b, (psimd_s32){4, 1, 2, 3});
  #endif  // defined(__clang__)
}

void xnn_f32_dwconv2d_chw_ukernel_3x3p1__psimd_1x4_acc3(
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

  const psimd_s32 vmask = psimd_load_s32(params->scalar.mask);
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

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height == 1) {
      i2 = zero;
    }

    // vi0x3012 = ( vi02, vi01, vi00, vi03 )
    psimd_f32 vi0x3012 = psimd_zero_f32();
    // vi1x3012 = ( vi12, vi11, vi10, vi13 )
    psimd_f32 vi1x3012 = psimd_zero_f32();
    // vi2x3012 = ( vi22, vi21, vi20, vi13 )
    psimd_f32 vi2x3012 = psimd_zero_f32();
    // vi0x4567 = ( vi07, vi06, vi05, vi04 )
    psimd_f32 vi0x4567 = psimd_load_f32(i0);
    i0 += 4;
    // vi1x4567 = ( vi17, vi16, vi15, vi14 )
    psimd_f32 vi1x4567 = psimd_load_f32(i1);
    i1 += 4;
    // vi2x4567 = ( vi27, vi26, vi25, vi24 )
    psimd_f32 vi2x4567 = psimd_load_f32(i2);
    i2 += 4;

    size_t w = input_width;
    for (; w > 4 * sizeof(float); w -= 4 * sizeof(float)) {
      psimd_f32 vo4567p0 = vbias;

      // vi0x89AB = ( vi0B, vi0A, vi09, vi08 )
      const psimd_f32 vi0x89AB = psimd_load_f32(i0);
      i0 += 4;
      // vi1x89AB = ( vi1B, vi0A, vi09, vi08 )
      const psimd_f32 vi1x89AB = psimd_load_f32(i1);
      i1 += 4;
      // vi2x89AB = ( vi2B, vi0A, vi09, vi08 )
      const psimd_f32 vi2x89AB = psimd_load_f32(i2);
      i2 += 4;

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const psimd_f32 vi0x7456 = rotright_f32(vi0x4567);
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const psimd_f32 vi1x7456 = rotright_f32(vi1x4567);
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const psimd_f32 vi2x7456 = rotright_f32(vi2x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x4567, vk01);
      psimd_f32 vo4567p1 = psimd_mul_f32(vi1x4567, vk11);
      psimd_f32 vo4567p2 = psimd_mul_f32(vi2x4567, vk21);

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const psimd_f32 vi0x3456 = movess_f32(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const psimd_f32 vi1x3456 = movess_f32(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const psimd_f32 vi2x3456 = movess_f32(vi2x7456, vi2x3012);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x3456, vk00);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x3456, vk10);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x3456, vk20);

      vi0x3012 = vi0x7456;
      vi1x3012 = vi1x7456;
      vi2x3012 = vi2x7456;

      // vi0x8567 = ( vi07, vi06, vi05, vi08 )
      const psimd_f32 vi0x8567 = movess_f32(vi0x4567, vi0x89AB);
      // vi1x8567 = ( vi17, vi16, vi15, vi18 )
      const psimd_f32 vi1x8567 = movess_f32(vi1x4567, vi1x89AB);
      // vi2x8567 = ( vi27, vi26, vi25, vi28 )
      const psimd_f32 vi2x8567 = movess_f32(vi2x4567, vi2x89AB);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const psimd_f32 vi0x5678 = rotleft_f32(vi0x8567);
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const psimd_f32 vi1x5678 = rotleft_f32(vi1x8567);
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const psimd_f32 vi2x5678 = rotleft_f32(vi2x8567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x5678, vk02);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x5678, vk12);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x5678, vk22);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;

      psimd_f32 vo = psimd_add_f32(vo4567p0, vo4567p1);
      vo = psimd_add_f32(vo, vo4567p2);

      vo = psimd_max_f32(vo, vmin);
      vo = psimd_min_f32(vo, vmax);

      psimd_store_f32(output, vo);
      output += 4;
    }
    // Always process the last block of 1..4 pixels.
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      psimd_f32 vo4567p0 = vbias;

      vi0x4567 = psimd_andmask_f32(vmask, vi0x4567);
      vi1x4567 = psimd_andmask_f32(vmask, vi1x4567);
      vi2x4567 = psimd_andmask_f32(vmask, vi2x4567);

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const psimd_f32 vi0x7456 = rotright_f32(vi0x4567);
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const psimd_f32 vi1x7456 = rotright_f32(vi1x4567);
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const psimd_f32 vi2x7456 = rotright_f32(vi2x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x4567, vk01);
      psimd_f32 vo4567p1 = psimd_mul_f32(vi1x4567, vk11);
      psimd_f32 vo4567p2 = psimd_mul_f32(vi2x4567, vk21);

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const psimd_f32 vi0x3456 = movess_f32(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const psimd_f32 vi1x3456 = movess_f32(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const psimd_f32 vi2x3456 = movess_f32(vi2x7456, vi2x3012);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x3456, vk00);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x3456, vk10);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x3456, vk20);

      const psimd_f32 vzero = psimd_zero_f32();
      // vi0x8567 = ( vi07, vi06, vi05, 0.0 )
      const psimd_f32 vi0x8567 = movess_f32(vi0x4567, vzero);
      // vi1x8567 = ( vi17, vi16, vi15, 0.0 )
      const psimd_f32 vi1x8567 = movess_f32(vi1x4567, vzero);
      // vi2x8567 = ( vi27, vi26, vi25, 0.0 )
      const psimd_f32 vi2x8567 = movess_f32(vi2x4567, vzero);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const psimd_f32 vi0x5678 = rotleft_f32(vi0x8567);
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const psimd_f32 vi1x5678 = rotleft_f32(vi1x8567);
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const psimd_f32 vi2x5678 = rotleft_f32(vi2x8567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x5678, vk02);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x5678, vk12);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x5678, vk22);

      psimd_f32 vo = psimd_add_f32(vo4567p0, vo4567p1);
      vo = psimd_add_f32(vo, vo4567p2);

      vo = psimd_max_f32(vo, vmin);
      vo = psimd_min_f32(vo, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
        psimd_store_f32(output, vo);
        output += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          psimd_store2_f32(output, vo);
          output += 2;
          vo = psimd_concat_hi_f32(vo, vo);
        }
        if (w & (1 * sizeof(float))) {
          psimd_store1_f32(output, vo);
          output += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i1 - input_decrement);
    i1 = (const float*) ((uintptr_t) i2 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
  } while (--output_height != 0);
}
