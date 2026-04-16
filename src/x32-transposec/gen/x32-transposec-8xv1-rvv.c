// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/rvv-u.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Optimized by Autocomp (https://github.com/ucb-bar/autocomp)
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/transpose.h"


void xnn_x32_transposec_ukernel__8xv1_rvv(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t input_stride_u32 = input_stride / sizeof(uint32_t);
  const size_t output_stride_u32 = output_stride / sizeof(uint32_t);

  for (size_t bh = 0; bh < block_height; ) {
    const size_t vl = __riscv_vsetvl_e32m1(block_height - bh);

    const uint32_t* i_row = input + bh * input_stride_u32;
    uint32_t* o_col = output + bh;

    size_t bw = 0;
    for (; bw + 8 <= block_width; bw += 8) {
      const uint32_t* i_ptr = i_row + bw;
      uint32_t* o_ptr = o_col + bw * output_stride_u32;

      // Issue loads with the first half of each tuple's stores interleaved to
      // cover segmented-load latency (mirrors the hand-written m2 kernel).
      vuint32m1x8_t tuple0 = __riscv_vlsseg8e32_v_u32m1x8(i_ptr + 0, input_stride, vl);

      // Drain remaining stores.
      __riscv_vse32_v_u32m1(o_ptr + 0 * output_stride_u32, __riscv_vget_v_u32m1x8_u32m1(tuple0, 0), vl);
      __riscv_vse32_v_u32m1(o_ptr + 1 * output_stride_u32, __riscv_vget_v_u32m1x8_u32m1(tuple0, 1), vl);
      __riscv_vse32_v_u32m1(o_ptr + 2 * output_stride_u32, __riscv_vget_v_u32m1x8_u32m1(tuple0, 2), vl);
      __riscv_vse32_v_u32m1(o_ptr + 3 * output_stride_u32, __riscv_vget_v_u32m1x8_u32m1(tuple0, 3), vl);
      __riscv_vse32_v_u32m1(o_ptr + 4 * output_stride_u32, __riscv_vget_v_u32m1x8_u32m1(tuple0, 4), vl);
      __riscv_vse32_v_u32m1(o_ptr + 5 * output_stride_u32, __riscv_vget_v_u32m1x8_u32m1(tuple0, 5), vl);
      __riscv_vse32_v_u32m1(o_ptr + 6 * output_stride_u32, __riscv_vget_v_u32m1x8_u32m1(tuple0, 6), vl);
      __riscv_vse32_v_u32m1(o_ptr + 7 * output_stride_u32, __riscv_vget_v_u32m1x8_u32m1(tuple0, 7), vl);
    }

    // Column tail: 0 < (block_width - bw) < TILE_WIDTH.
    // Keep LMUL=1 so the same vl rows are processed, splitting the tail
    // into legal vlsseg calls where needed (EMUL * NFIELDS <= 8).
    if (bw < block_width) {
      const size_t bw_tail = block_width - bw;
      const uint32_t* i_ptr = i_row + bw;
      uint32_t* o_ptr = o_col + bw * output_stride_u32;
      switch (bw_tail) {
        case 7: {
          vuint32m1x7_t tuple0 = __riscv_vlsseg7e32_v_u32m1x7(i_ptr + 0, input_stride, vl);
          __riscv_vse32_v_u32m1(o_ptr + 0 * output_stride_u32, __riscv_vget_v_u32m1x7_u32m1(tuple0, 0), vl);
          __riscv_vse32_v_u32m1(o_ptr + 1 * output_stride_u32, __riscv_vget_v_u32m1x7_u32m1(tuple0, 1), vl);
          __riscv_vse32_v_u32m1(o_ptr + 2 * output_stride_u32, __riscv_vget_v_u32m1x7_u32m1(tuple0, 2), vl);
          __riscv_vse32_v_u32m1(o_ptr + 3 * output_stride_u32, __riscv_vget_v_u32m1x7_u32m1(tuple0, 3), vl);
          __riscv_vse32_v_u32m1(o_ptr + 4 * output_stride_u32, __riscv_vget_v_u32m1x7_u32m1(tuple0, 4), vl);
          __riscv_vse32_v_u32m1(o_ptr + 5 * output_stride_u32, __riscv_vget_v_u32m1x7_u32m1(tuple0, 5), vl);
          __riscv_vse32_v_u32m1(o_ptr + 6 * output_stride_u32, __riscv_vget_v_u32m1x7_u32m1(tuple0, 6), vl);
          break;
        }
        case 6: {
          vuint32m1x6_t tuple0 = __riscv_vlsseg6e32_v_u32m1x6(i_ptr + 0, input_stride, vl);
          __riscv_vse32_v_u32m1(o_ptr + 0 * output_stride_u32, __riscv_vget_v_u32m1x6_u32m1(tuple0, 0), vl);
          __riscv_vse32_v_u32m1(o_ptr + 1 * output_stride_u32, __riscv_vget_v_u32m1x6_u32m1(tuple0, 1), vl);
          __riscv_vse32_v_u32m1(o_ptr + 2 * output_stride_u32, __riscv_vget_v_u32m1x6_u32m1(tuple0, 2), vl);
          __riscv_vse32_v_u32m1(o_ptr + 3 * output_stride_u32, __riscv_vget_v_u32m1x6_u32m1(tuple0, 3), vl);
          __riscv_vse32_v_u32m1(o_ptr + 4 * output_stride_u32, __riscv_vget_v_u32m1x6_u32m1(tuple0, 4), vl);
          __riscv_vse32_v_u32m1(o_ptr + 5 * output_stride_u32, __riscv_vget_v_u32m1x6_u32m1(tuple0, 5), vl);
          break;
        }
        case 5: {
          vuint32m1x5_t tuple0 = __riscv_vlsseg5e32_v_u32m1x5(i_ptr + 0, input_stride, vl);
          __riscv_vse32_v_u32m1(o_ptr + 0 * output_stride_u32, __riscv_vget_v_u32m1x5_u32m1(tuple0, 0), vl);
          __riscv_vse32_v_u32m1(o_ptr + 1 * output_stride_u32, __riscv_vget_v_u32m1x5_u32m1(tuple0, 1), vl);
          __riscv_vse32_v_u32m1(o_ptr + 2 * output_stride_u32, __riscv_vget_v_u32m1x5_u32m1(tuple0, 2), vl);
          __riscv_vse32_v_u32m1(o_ptr + 3 * output_stride_u32, __riscv_vget_v_u32m1x5_u32m1(tuple0, 3), vl);
          __riscv_vse32_v_u32m1(o_ptr + 4 * output_stride_u32, __riscv_vget_v_u32m1x5_u32m1(tuple0, 4), vl);
          break;
        }
        case 4: {
          vuint32m1x4_t tuple0 = __riscv_vlsseg4e32_v_u32m1x4(i_ptr + 0, input_stride, vl);
          __riscv_vse32_v_u32m1(o_ptr + 0 * output_stride_u32, __riscv_vget_v_u32m1x4_u32m1(tuple0, 0), vl);
          __riscv_vse32_v_u32m1(o_ptr + 1 * output_stride_u32, __riscv_vget_v_u32m1x4_u32m1(tuple0, 1), vl);
          __riscv_vse32_v_u32m1(o_ptr + 2 * output_stride_u32, __riscv_vget_v_u32m1x4_u32m1(tuple0, 2), vl);
          __riscv_vse32_v_u32m1(o_ptr + 3 * output_stride_u32, __riscv_vget_v_u32m1x4_u32m1(tuple0, 3), vl);
          break;
        }
        case 3: {
          vuint32m1x3_t tuple0 = __riscv_vlsseg3e32_v_u32m1x3(i_ptr + 0, input_stride, vl);
          __riscv_vse32_v_u32m1(o_ptr + 0 * output_stride_u32, __riscv_vget_v_u32m1x3_u32m1(tuple0, 0), vl);
          __riscv_vse32_v_u32m1(o_ptr + 1 * output_stride_u32, __riscv_vget_v_u32m1x3_u32m1(tuple0, 1), vl);
          __riscv_vse32_v_u32m1(o_ptr + 2 * output_stride_u32, __riscv_vget_v_u32m1x3_u32m1(tuple0, 2), vl);
          break;
        }
        case 2: {
          vuint32m1x2_t tuple0 = __riscv_vlsseg2e32_v_u32m1x2(i_ptr + 0, input_stride, vl);
          __riscv_vse32_v_u32m1(o_ptr + 0 * output_stride_u32, __riscv_vget_v_u32m1x2_u32m1(tuple0, 0), vl);
          __riscv_vse32_v_u32m1(o_ptr + 1 * output_stride_u32, __riscv_vget_v_u32m1x2_u32m1(tuple0, 1), vl);
          break;
        }
        case 1: {
          vuint32m1_t v = __riscv_vlse32_v_u32m1(i_ptr, input_stride, vl);
          __riscv_vse32_v_u32m1(o_ptr, v, vl);
          break;
        }
        default:
          XNN_UNREACHABLE;
      }
    }

    bh += vl;
  }
}
