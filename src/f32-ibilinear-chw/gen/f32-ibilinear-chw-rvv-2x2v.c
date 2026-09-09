// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear-chw/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Kalray
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/ibilinear.h"


void xnn_f32_ibilinear_chw_ukernel__rvv_2x2v(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t input_increment)
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  for (size_t cb = 0; cb < channels; cb += 2) {
    const size_t block = (channels - cb) < 2 ? (channels - cb) : 2;

    const float** i = input;
    const float* w = weights;
    float* out = output + cb * output_pixels;
    const size_t block_base = input_offset + cb * input_increment;

    for (size_t p = output_pixels, vl; p > 0;
          p -= vl, out += vl, i += 2 * vl, w += 2 * vl) {
      vl = __riscv_vsetvl_e32m2(p);

      const vuint64m4x2_t vptrs =
          __riscv_vlseg2e64_v_u64m4x2((const uint64_t*)i, vl);
      const vuint64m4_t vti_addr =
          __riscv_vget_v_u64m4x2_u64m4(vptrs, 0);
      const vuint64m4_t vbi_addr =
          __riscv_vget_v_u64m4x2_u64m4(vptrs, 1);

      const vuint64m4_t wpair =
          __riscv_vle64_v_u64m4((const uint64_t*)w, vl);
      const vfloat32m2_t vwh = __riscv_vreinterpret_v_u32m2_f32m2(
          __riscv_vnsrl(wpair, 0, vl));
      const vfloat32m2_t vwv = __riscv_vreinterpret_v_u32m2_f32m2(
          __riscv_vnsrl(wpair, 32, vl));

      float* o = out;
      size_t channel_offset = block_base;

      for (size_t c = block; c != 0; c--) {
        const vfloat32m2x2_t vti = __riscv_vluxseg2ei64_v_f32m2x2(
            (const float*)channel_offset, vti_addr, vl);
        const vfloat32m2x2_t vbi = __riscv_vluxseg2ei64_v_f32m2x2(
            (const float*)channel_offset, vbi_addr, vl);

        const vfloat32m2_t vtl =
            __riscv_vget_v_f32m2x2_f32m2(vti, 0);
        const vfloat32m2_t vtr =
            __riscv_vget_v_f32m2x2_f32m2(vti, 1);
        const vfloat32m2_t vbl =
            __riscv_vget_v_f32m2x2_f32m2(vbi, 0);
        const vfloat32m2_t vbr =
            __riscv_vget_v_f32m2x2_f32m2(vbi, 1);

        const vfloat32m2_t vt = __riscv_vfmacc_vv_f32m2(
            vtl, __riscv_vfsub_vv_f32m2(vtr, vtl, vl), vwh, vl);
        const vfloat32m2_t vb = __riscv_vfmacc_vv_f32m2(
            vbl, __riscv_vfsub_vv_f32m2(vbr, vbl, vl), vwh, vl);
        const vfloat32m2_t vo = __riscv_vfmacc_vv_f32m2(
            vt, __riscv_vfsub_vv_f32m2(vb, vt, vl), vwv, vl);

        __riscv_vse32_v_f32m2(o, vo, vl);

        channel_offset += input_increment;
        o += output_pixels;
      }
    }
  }
}
