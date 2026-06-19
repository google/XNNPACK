// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-maxpool/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include "src/xnnpack/maxpool.h"
#include <riscv_vector.h>


void xnn_u8_maxpool_minmax_ukernel_9p__rvv_u2v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    size_t input_pixel_stride,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_u8_minmax_params* restrict params)
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const uint8_t output_min = params->scalar.min;
  const uint8_t output_max = params->scalar.max;
  do {
    const uint8_t** i = input;

    // First pass: load the inputs, store the max pool in the output.
    const uint8_t* i0 = *i++;
    const uint8_t* i1 = 1 < kernel_elements ? *i++ : i0;
    const uint8_t* i2 = 2 < kernel_elements ? *i++ : i0;
    const uint8_t* i3 = 3 < kernel_elements ? *i++ : i0;
    const uint8_t* i4 = 4 < kernel_elements ? *i++ : i0;
    const uint8_t* i5 = 5 < kernel_elements ? *i++ : i0;
    const uint8_t* i6 = 6 < kernel_elements ? *i++ : i0;
    const uint8_t* i7 = 7 < kernel_elements ? *i++ : i0;
    const uint8_t* i8 = 8 < kernel_elements ? *i++ : i0;
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);

    uint8_t* o = output;
    size_t c = channels;
    do {
      size_t vl = __riscv_vsetvl_e8m2(c);
      vuint8m2_t vi0 = __riscv_vle8_v_u8m2(i0, vl); i0 += vl;
      vuint8m2_t vi1 = __riscv_vle8_v_u8m2(i1, vl); i1 += vl;
      vuint8m2_t vi2 = __riscv_vle8_v_u8m2(i2, vl); i2 += vl;
      vuint8m2_t vi3 = __riscv_vle8_v_u8m2(i3, vl); i3 += vl;
      vuint8m2_t vi4 = __riscv_vle8_v_u8m2(i4, vl); i4 += vl;
      vuint8m2_t vi5 = __riscv_vle8_v_u8m2(i5, vl); i5 += vl;
      vuint8m2_t vi6 = __riscv_vle8_v_u8m2(i6, vl); i6 += vl;
      vuint8m2_t vi7 = __riscv_vle8_v_u8m2(i7, vl); i7 += vl;
      vuint8m2_t vi8 = __riscv_vle8_v_u8m2(i8, vl); i8 += vl;

      vuint8m2_t vmax01 = __riscv_vmaxu(vi0, vi1, vl);
      vuint8m2_t vmax23 = __riscv_vmaxu(vi2, vi3, vl);
      vuint8m2_t vmax45 = __riscv_vmaxu(vi4, vi5, vl);
      vuint8m2_t vmax67 = __riscv_vmaxu(vi6, vi7, vl);
      vuint8m2_t vmax018 = __riscv_vmaxu(vmax01, vi8, vl);

      vuint8m2_t vmax2345 = __riscv_vmaxu(vmax23, vmax45, vl);
      vuint8m2_t vmax01678 = __riscv_vmaxu(vmax67, vmax018, vl);
      vuint8m2_t vacc = __riscv_vmaxu(vmax2345, vmax01678, vl);

      vacc = __riscv_vmaxu(vacc, output_min, vl);
      vacc = __riscv_vminu(vacc, output_max, vl);
      __riscv_vse8_v_u8m2(o, vacc, vl); o += vl;

      c -= vl;
    } while (c != 0);

    // Passes 1 - n: Max more inputs to the output.
    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 9) {
      const uint8_t* i0 = *i++;
      const uint8_t* i1 = 1 < k ? *i++ : i0;
      const uint8_t* i2 = 2 < k ? *i++ : i0;
      const uint8_t* i3 = 3 < k ? *i++ : i0;
      const uint8_t* i4 = 4 < k ? *i++ : i0;
      const uint8_t* i5 = 5 < k ? *i++ : i0;
      const uint8_t* i6 = 6 < k ? *i++ : i0;
      const uint8_t* i7 = 7 < k ? *i++ : i0;
      const uint8_t* i8 = 8 < k ? *i++ : i0;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);

      o = output;
      size_t c = channels;
      do {
        size_t vl = __riscv_vsetvl_e8m2(c);

        vuint8m2_t vi0 = __riscv_vle8_v_u8m2(i0, vl); i0 += vl;
        vuint8m2_t vi1 = __riscv_vle8_v_u8m2(i1, vl); i1 += vl;
        vuint8m2_t vi2 = __riscv_vle8_v_u8m2(i2, vl); i2 += vl;
        vuint8m2_t vi3 = __riscv_vle8_v_u8m2(i3, vl); i3 += vl;
        vuint8m2_t vi4 = __riscv_vle8_v_u8m2(i4, vl); i4 += vl;
        vuint8m2_t vi5 = __riscv_vle8_v_u8m2(i5, vl); i5 += vl;
        vuint8m2_t vi6 = __riscv_vle8_v_u8m2(i6, vl); i6 += vl;
        vuint8m2_t vi7 = __riscv_vle8_v_u8m2(i7, vl); i7 += vl;
        vuint8m2_t vi8 = __riscv_vle8_v_u8m2(i8, vl); i8 += vl;

        vuint8m2_t vprev = __riscv_vle8_v_u8m2(o, vl);

        vuint8m2_t vmax01 = __riscv_vmaxu(vi0, vi1, vl);
        vuint8m2_t vmax23 = __riscv_vmaxu(vi2, vi3, vl);
        vuint8m2_t vmax45 = __riscv_vmaxu(vi4, vi5, vl);
        vuint8m2_t vmax67 = __riscv_vmaxu(vi6, vi7, vl);
        vuint8m2_t vmax018 = __riscv_vmaxu(vmax01, vi8, vl);

        vuint8m2_t vmax2345 = __riscv_vmaxu(vmax23, vmax45, vl);
        vuint8m2_t vmax01678 = __riscv_vmaxu(vmax67, vmax018, vl);
        vuint8m2_t vmax012345678 = __riscv_vmaxu(vmax2345, vmax01678, vl);

        vuint8m2_t vacc = __riscv_vmaxu(vprev, vmax012345678, vl);
        vacc = __riscv_vminu(vacc, output_max, vl);
        __riscv_vse8_v_u8m2(o, vacc, vl); o += vl;

        c -= vl;
      } while (c != 0);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
