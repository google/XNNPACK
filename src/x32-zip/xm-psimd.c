// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/zip.h>


void xnn_x32_zip_xm_ukernel__psimd(
    size_t n,
    size_t m,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);
  assert(m >= 4);

  const uint32_t* w = input;
  const size_t group_increment = m * 4;
  const size_t input_increment = n * 3;
  const size_t output_increment = 16 - m * n;
  const uint32_t* last_input = (const uint32_t*) ((uintptr_t) input + n * (m - 1));
  uint32_t* last_output = (uint32_t*) ((uintptr_t) output + (m * 4 - 16));

  for (size_t i = 0; i < m; i += 4) {
    w = (const uint32_t*) ((uintptr_t) w + input_increment);
    if (w >= last_input) {
      w = last_input;
    }
    const uint32_t* z = (const uint32_t*) ((uintptr_t) w - n);
    const uint32_t* y = (const uint32_t*) ((uintptr_t) z - n);
    const uint32_t* x = (const uint32_t*) ((uintptr_t) y - n);

    size_t k = n;
    while (k >= 16) {
      const psimd_u32 vx = psimd_load_u32((const psimd_u32*) x);
      x += 4;
      const psimd_u32 vy = psimd_load_u32((const psimd_u32*) y);
      y += 4;
      const psimd_u32 vz = psimd_load_u32((const psimd_u32*) z);
      z += 4;
      const psimd_u32 vw = psimd_load_u32((const psimd_u32*) w);
      w += 4;

      const psimd_u32 vxy_lo = psimd_interleave_lo_u32(vx, vy);
      const psimd_u32 vxy_hi = psimd_interleave_hi_u32(vx, vy);
      const psimd_u32 vzw_lo = psimd_interleave_lo_u32(vz, vw);
      const psimd_u32 vzw_hi = psimd_interleave_hi_u32(vz, vw);

      const psimd_u32 vxyzw0 = psimd_concat_lo_u32(vxy_lo, vzw_lo);
      const psimd_u32 vxyzw1 = psimd_concat_hi_u32(vxy_lo, vzw_lo);
      const psimd_u32 vxyzw2 = psimd_concat_lo_u32(vxy_hi, vzw_hi);
      const psimd_u32 vxyzw3 = psimd_concat_hi_u32(vxy_hi, vzw_hi);

      psimd_store_u32(output, vxyzw0);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      psimd_store_u32(output, vxyzw1);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      psimd_store_u32(output, vxyzw2);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      psimd_store_u32(output, vxyzw3);
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      k -= 16;
    }
    if XNN_UNLIKELY(k != 0) {
      if (k & 8) {
        const psimd_u32 vx = psimd_load2_u32(x);
        x += 2;
        const psimd_u32 vy = psimd_load2_u32(y);
        y += 2;
        const psimd_u32 vz = psimd_load2_u32(z);
        z += 2;
        const psimd_u32 vw = psimd_load2_u32(w);
        w += 2;

        const psimd_u32 vxy = psimd_interleave_lo_u32(vx, vy);
        const psimd_u32 vzw = psimd_interleave_lo_u32(vz, vw);

        const psimd_u32 vxyzw_lo = psimd_concat_lo_u32(vxy, vzw);
        const psimd_u32 vxyzw_hi = psimd_concat_hi_u32(vxy, vzw);

        psimd_store_u32(output, vxyzw_lo);
        output = (uint32_t*) ((uintptr_t) output + group_increment);

        psimd_store_u32(output, vxyzw_hi);
        output = (uint32_t*) ((uintptr_t) output + group_increment);
      }
      if (k & 4) {
        const uint32_t vx = *x;
        const uint32_t vy = *y;
        const uint32_t vz = *z;
        const uint32_t vw = *w++;

        output[0] = vx;
        output[1] = vy;
        output[2] = vz;
        output[3] = vw;
        output = (uint32_t*) ((uintptr_t) output + group_increment);
      }
    }
    output = (uint32_t*) ((uintptr_t) output + output_increment);
    if (output > last_output) {
      output = last_output;
    }
  }
}
