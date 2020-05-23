// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/pad.h>


void xnn_x32_pad_ukernel__psimd(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const uint32_t* fill_value,
    const uint32_t* input,
    size_t input_stride,
    uint32_t* output,
    size_t output_stride)
{
  assert(channels % sizeof(uint32_t) == 0);
  assert(pre_padding % sizeof(uint32_t) == 0);
  assert(post_padding % sizeof(uint32_t) == 0);
  assert(fill_value != NULL);

  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const psimd_u32 vfill = psimd_load_splat_u32(fill_value);
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      for (; l >= 4 * sizeof(uint32_t); l -= 4 * sizeof(uint32_t)) {
        psimd_store_u32(output, vfill);
        output += 4;
      }
      if (l & (2 * sizeof(uint32_t))) {
        psimd_store2_u32(output, vfill);
        output += 2;
      }
      if (l & sizeof(uint32_t)) {
        psimd_store1_u32(output, vfill);
        output += 1;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      const psimd_u32 vtmp = psimd_load_u32(input);
      input += 4;

      psimd_store_u32(output, vtmp);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      psimd_u32 vtmp = psimd_load_u32(input);
      input = (const void*) ((uintptr_t) input + c);
      if (c & (2 * sizeof(uint32_t))) {
        psimd_store2_u32(output, vtmp);
        output += 2;

        vtmp = psimd_concat_hi_u32(vtmp, vtmp);
      }
      if (c & sizeof(uint32_t)) {
        psimd_store1_u32(output, vtmp);
        output += 1;
      }
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      for (; r >= 4 * sizeof(uint32_t); r -= 4 * sizeof(uint32_t)) {
        psimd_store_u32(output, vfill);
        output += 4;
      }
      if (r & (2 * sizeof(uint32_t))) {
        psimd_store2_u32(output, vfill);
        output += 2;
      }
      if (r & sizeof(uint32_t)) {
        psimd_store1_u32(output, vfill);
        output += 1;
      }
    }

    input = (const uint32_t*) ((uintptr_t) input + input_increment);
    output = (uint32_t*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
