// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/ibilinear.h"


void xnn_f32_ibilinear_ukernel__rvv_u2v(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
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

    size_t c = channels >> XNN_LOG2_SIZEOF_FLOAT;
    do {
      const size_t n = __riscv_vsetvl_e32m2(c);

      // Load top-left, top-right, bottom-left, bottom-right.
      vfloat32m2_t vtl = __riscv_vle32_v_f32m2(i0, n); i0 += n;
      vfloat32m2_t vtr = __riscv_vle32_v_f32m2(i1, n); i1 += n;
      vfloat32m2_t vbl = __riscv_vle32_v_f32m2(i2, n); i2 += n;
      vfloat32m2_t vbr = __riscv_vle32_v_f32m2(i3, n); i3 += n;

      // Horizontal interpolation differences.
      vfloat32m2_t vtd = __riscv_vfsub_vv_f32m2(vtr, vtl, n);
      vfloat32m2_t vbd = __riscv_vfsub_vv_f32m2(vbr, vbl, n);

      // Horizontal interpolation: top = tl + (tr - tl) * alphah.
      vfloat32m2_t vt = __riscv_vfmacc_vf_f32m2(vtl, valphah, vtd, n);
      vfloat32m2_t vb = __riscv_vfmacc_vf_f32m2(vbl, valphah, vbd, n);

      // Vertical interpolation: output = top + (bottom - top) * alphav.
      vfloat32m2_t vd = __riscv_vfsub_vv_f32m2(vb, vt, n);
      vfloat32m2_t vo = __riscv_vfmacc_vf_f32m2(vt, valphav, vd, n);

      __riscv_vse32_v_f32m2(output, vo, n);
      output += n;

      c -= n;
    } while (c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
