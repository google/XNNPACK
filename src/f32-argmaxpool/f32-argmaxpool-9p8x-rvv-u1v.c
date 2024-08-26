// Copyright 2024 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>

#include "xnnpack/argmaxpool.h"

void xnn_f32_argmaxpool_ukernel_9p8x__rvv_u1v(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* accumulation_buffer,
    uint32_t* index_buffer,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements > 9);
  assert(channels != 0);

  vuint32m1_t fidx_f32v = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvl_e32m1(channels));
  do {
    {
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);

      for (size_t c = channels; c != 0; ) {
        int32_t n = __riscv_vsetvl_e32m1(c); c -= n;

        vfloat32m1_t max_f32v = __riscv_vle32_v_f32m1(i0, n); i0 += n;
        vfloat32m1_t i1_f32v = __riscv_vle32_v_f32m1(i1, n); i1 += n;
        vfloat32m1_t i2_f32v = __riscv_vle32_v_f32m1(i2, n); i2 += n;
        vfloat32m1_t i3_f32v = __riscv_vle32_v_f32m1(i3, n); i3 += n;
        vfloat32m1_t i4_f32v = __riscv_vle32_v_f32m1(i4, n); i4 += n;
        vfloat32m1_t i5_f32v = __riscv_vle32_v_f32m1(i5, n); i5 += n;
        vfloat32m1_t i6_f32v = __riscv_vle32_v_f32m1(i6, n); i6 += n;
        vfloat32m1_t i7_f32v = __riscv_vle32_v_f32m1(i7, n); i7 += n;
        vfloat32m1_t i8_f32v = __riscv_vle32_v_f32m1(i8, n); i8 += n;

        vbool32_t mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i1_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i1_f32v, mask_b32v, n);
        vuint32m1_t idx_f32v = __riscv_vmerge_vxm_u32m1(fidx_f32v, 1, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i2_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i2_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, 2, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i3_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i3_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, 3, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i4_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i4_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, 4, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i5_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i5_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, 5, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i6_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i6_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, 6, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i7_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i7_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, 7, mask_b32v, n);

        mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i8_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i8_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, 8, mask_b32v, n);

        __riscv_vse32_v_f32m1(ab, max_f32v, n); ab += n;
        __riscv_vse32_v_u32m1(ib, idx_f32v, n); ib += n;
        
      }
    }

    uint32_t idx0 = 9;
    size_t k = pooling_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);

      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

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

        vfloat32m1_t max_f32v = __riscv_vle32_v_f32m1(ab, n);
        vuint32m1_t idx_f32v = __riscv_vle32_v_u32m1(ib, n);

        vbool32_t mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i0_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i0_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0, mask_b32v, n);

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

        __riscv_vse32_v_f32m1(ab, max_f32v, n); ab += n;
        __riscv_vse32_v_u32m1(ib, idx_f32v, n); ib += n;
      }
      idx0 += 8;
    }

    float* o = output;
    uint32_t* i = index;
    {
      const float* i0 = input[0];
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k != 8) {
        i7 = i0;
      }

      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;
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

        vfloat32m1_t max_f32v = __riscv_vle32_v_f32m1(ab, n); ab += n;
        vuint32m1_t idx_f32v = __riscv_vle32_v_u32m1(ib, n); ib += n;

        vbool32_t mask_b32v = __riscv_vmfgt_vv_f32m1_b32(i0_f32v, max_f32v, n);
        max_f32v = __riscv_vmerge_vvm_f32m1(max_f32v, i0_f32v, mask_b32v, n);
        idx_f32v = __riscv_vmerge_vxm_u32m1(idx_f32v, idx0, mask_b32v, n);

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

        __riscv_vse32_v_f32m1(o, max_f32v, n); o += n;
        __riscv_vse32_v_u32m1(i, idx_f32v, n); i += n;
      }
    }

    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--output_pixels != 0);
}
