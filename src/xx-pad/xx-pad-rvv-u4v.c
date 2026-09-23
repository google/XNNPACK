// Copyright 2026 Léandre Le Duc
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>

#include "src/xnnpack/pad.h"

void xnn_xx_pad_ukernel__rvv_u4v(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    uint32_t fill_pattern) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);

  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const vuint8m4_t vfill = __riscv_vreinterpret_v_u32m4_u8m4(
      __riscv_vmv_v_x_u32m4(fill_pattern, __riscv_vsetvlmax_e32m4()));

  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      for(size_t vl; l > 0; l -= vl, output += vl) {
        vl = __riscv_vsetvl_e8m4(l);
        __riscv_vse8_v_u8m4(output, vfill, vl);
      }

    }

    // Copy input channels.
    size_t c = channels;
    for (size_t vl; c > 0; c -= vl, output += vl, input += vl) {
      vl = __riscv_vsetvl_e8m4(c);
      const vuint8m4_t vdata = __riscv_vle8_v_u8m4(input, vl);
      __riscv_vse8_v_u8m4(output, vdata, vl);
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      for(size_t vl; r > 0; r -= vl, output += vl) {
        vl = __riscv_vsetvl_e8m4(r);
        __riscv_vse8_v_u8m4(output, vfill, vl);
      }

    }

    input = (void*)((uintptr_t)input + input_increment);
    output = (void*)((uintptr_t)output + output_increment);
  } while (--rows != 0);
}
