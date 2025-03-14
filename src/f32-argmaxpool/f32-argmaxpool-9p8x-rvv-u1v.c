// Copyright 2024 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>

#include "src/xnnpack/argmaxpool.h"

void xnn_f32_argmaxpool_ukernel_9p8x__rvv_u1v(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements > 9);
  assert(channels != 0);

  vuint32m1_t fidx_f32v = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvl_e32m1(channels));
  do {
    // Accumulators start out null, after each pass the accumulator is set to
    // the output.
    const float* ab = NULL;
    const uint32_t* ib = NULL;
    const float** id = input;

    uint32_t idx0 = 0;
    ptrdiff_t k = pooling_elements;
    for (; k > 0; k -= 9) {
      const float* i0 = *id++;
      const float* i1 = 1 < k ? *id++ : i0;
      const float* i2 = 2 < k ? *id++ : i0;
      const float* i3 = 3 < k ? *id++ : i0;
      const float* i4 = 4 < k ? *id++ : i0;
      const float* i5 = 5 < k ? *id++ : i0;
      const float* i6 = 6 < k ? *id++ : i0;
      const float* i7 = 7 < k ? *id++ : i0;
      const float* i8 = 8 < k ? *id++ : i0;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);

      float* o = output;
      uint32_t* i = index;
      for (size_t c = channels; c != 0; ) {
        int32_t n = __riscv_vsetvl_e32m1(c); c -= n;

        vfloat32m1_t i0_f32v = __riscv_vle32_v_f32m1(i0, n); i0 += n;
        vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i1, n); i1 += n;
        vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i2, n); i2 += n;
        vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i3, n); i3 += n;
        vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i4, n); i4 += n;
        vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i5, n); i5 += n;
        vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i6, n); i6 += n;
        vfloat32m1_t i7_f32v = __riscv_vle32_v_f32m1(i7, n); i7 += n;
        vfloat32m1_t i8_f32v = __riscv_vle32_v_f32m1(i8, n); i8 += n;

        vbool32_t mask_b32v;
        vfloat32m1_t max_f32v;
        vuint32m1_t idx_f32v;
        if (ab) {
          max_f32v = __riscv_vle32_v_f32m1(ab, n); ab += n;
          idx_f32v = __riscv_vle32_v_u32m1(ib, n); ib += n;

          mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i0_f32v, max_f32v, n);
          max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i0_f32v, mask_b32v, n);
          idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0, mask_b32v, n);
        } else {
          max_f32v = i0_f32v;
          idx_f32v = fidx_f32v;
        }

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i1_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i1_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0+1, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i2_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i2_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0+2, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i3_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i3_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0+3, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i4_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i4_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0+4, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i5_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i5_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0+5, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i6_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i6_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0+6, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i7_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i7_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0+7, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i8_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i8_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0+8, mask_b32v, n);

        __riscv_vse32_v_f32m1(o, max_f32v, n); o += n;
        __riscv_vse32_v_u32m1(i, idx_f32v, n); i += n;
      }
      idx0 += 9;
      ab = output;
      ib = index;
    }

    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
    index += channels;
  } while (--output_pixels != 0);
}
