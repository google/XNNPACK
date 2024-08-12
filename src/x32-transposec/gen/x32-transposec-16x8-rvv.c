// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <riscv_vector.h>

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"

void xnn_x32_transposec_ukernel__16x8_rvv(
  const uint32_t* input,
  uint32_t* output,
  size_t input_stride,
  size_t output_stride,
  size_t block_width,
  size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 16;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;

  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t* o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t* o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);
  uint32_t* o4 = (uint32_t*) ((uintptr_t) o3 + output_stride);
  uint32_t* o5 = (uint32_t*) ((uintptr_t) o4 + output_stride);
  uint32_t* o6 = (uint32_t*) ((uintptr_t) o5 + output_stride);
  uint32_t* o7 = (uint32_t*) ((uintptr_t) o6 + output_stride);

  do {
    size_t bh = block_height;
    size_t vl = __riscv_vsetvl_e32m1(tile_height);
    for (; bh >= 16; bh -= 16) {
      if (block_width >= tile_width) {
        vuint32m1x8_t tuple = __riscv_vlsseg8e32_v_u32m1x8(i0, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x8_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x8_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x8_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x8_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
        vuint32m1_t v_d4 = __riscv_vget_v_u32m1x8_u32m1(tuple, 4);
        __riscv_vse32_v_u32m1(o4, v_d4, vl);
        vuint32m1_t v_d5 = __riscv_vget_v_u32m1x8_u32m1(tuple, 5);
        __riscv_vse32_v_u32m1(o5, v_d5, vl);
        vuint32m1_t v_d6 = __riscv_vget_v_u32m1x8_u32m1(tuple, 6);
        __riscv_vse32_v_u32m1(o6, v_d6, vl);
        vuint32m1_t v_d7 = __riscv_vget_v_u32m1x8_u32m1(tuple, 7);
        __riscv_vse32_v_u32m1(o7, v_d7, vl);

      } else {
        switch (block_width) {
          case 7: {
            vuint32m1x7_t tuple = __riscv_vlsseg7e32_v_u32m1x7(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x7_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x7_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x7_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x7_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x7_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x7_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            vuint32m1_t v_d6 = __riscv_vget_v_u32m1x7_u32m1(tuple, 6);
            __riscv_vse32_v_u32m1(o6, v_d6, vl);
            break;
          }

          case 6: {
            vuint32m1x6_t tuple = __riscv_vlsseg6e32_v_u32m1x6(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x6_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x6_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x6_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x6_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x6_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x6_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            break;
          }

          case 5: {
            vuint32m1x5_t tuple = __riscv_vlsseg5e32_v_u32m1x5(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x5_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x5_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x5_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x5_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x5_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            break;
          }

          case 4: {
            vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            break;
          }

          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }

          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i0, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i0, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      o7 = (uint32_t*) ((uintptr_t) o7 + tile_hbytes);
      o6 = (uint32_t*) ((uintptr_t) o6 + tile_hbytes);
      o5 = (uint32_t*) ((uintptr_t) o5 + tile_hbytes);
      o4 = (uint32_t*) ((uintptr_t) o4 + tile_hbytes);
      o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint32_t* i = i0;
      vl = __riscv_vsetvl_e32m1(bh);
      if (block_width >= tile_width) {
        vuint32m1x8_t tuple = __riscv_vlsseg8e32_v_u32m1x8(i, input_stride, vl);

        vuint32m1_t v_d0 = __riscv_vget_v_u32m1x8_u32m1(tuple, 0);
        __riscv_vse32_v_u32m1(o0, v_d0, vl);
        vuint32m1_t v_d1 = __riscv_vget_v_u32m1x8_u32m1(tuple, 1);
        __riscv_vse32_v_u32m1(o1, v_d1, vl);
        vuint32m1_t v_d2 = __riscv_vget_v_u32m1x8_u32m1(tuple, 2);
        __riscv_vse32_v_u32m1(o2, v_d2, vl);
        vuint32m1_t v_d3 = __riscv_vget_v_u32m1x8_u32m1(tuple, 3);
        __riscv_vse32_v_u32m1(o3, v_d3, vl);
        vuint32m1_t v_d4 = __riscv_vget_v_u32m1x8_u32m1(tuple, 4);
        __riscv_vse32_v_u32m1(o4, v_d4, vl);
        vuint32m1_t v_d5 = __riscv_vget_v_u32m1x8_u32m1(tuple, 5);
        __riscv_vse32_v_u32m1(o5, v_d5, vl);
        vuint32m1_t v_d6 = __riscv_vget_v_u32m1x8_u32m1(tuple, 6);
        __riscv_vse32_v_u32m1(o6, v_d6, vl);
        vuint32m1_t v_d7 = __riscv_vget_v_u32m1x8_u32m1(tuple, 7);
        __riscv_vse32_v_u32m1(o7, v_d7, vl);
      } else {
        switch(block_width) {
          case 7: {
            vuint32m1x7_t tuple = __riscv_vlsseg7e32_v_u32m1x7(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x7_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x7_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x7_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x7_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x7_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x7_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            vuint32m1_t v_d6 = __riscv_vget_v_u32m1x7_u32m1(tuple, 6);
            __riscv_vse32_v_u32m1(o6, v_d6, vl);
            break;
          }
          case 6: {
            vuint32m1x6_t tuple = __riscv_vlsseg6e32_v_u32m1x6(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x6_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x6_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x6_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x6_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x6_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            vuint32m1_t v_d5 = __riscv_vget_v_u32m1x6_u32m1(tuple, 5);
            __riscv_vse32_v_u32m1(o5, v_d5, vl);
            break;
          }
          case 5: {
            vuint32m1x5_t tuple = __riscv_vlsseg5e32_v_u32m1x5(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x5_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x5_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x5_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x5_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            vuint32m1_t v_d4 = __riscv_vget_v_u32m1x5_u32m1(tuple, 4);
            __riscv_vse32_v_u32m1(o4, v_d4, vl);
            break;
          }
          case 4: {
            vuint32m1x4_t tuple = __riscv_vlsseg4e32_v_u32m1x4(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x4_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x4_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x4_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            vuint32m1_t v_d3 = __riscv_vget_v_u32m1x4_u32m1(tuple, 3);
            __riscv_vse32_v_u32m1(o3, v_d3, vl);
            break;
          }
          case 3: {
            vuint32m1x3_t tuple = __riscv_vlsseg3e32_v_u32m1x3(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x3_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x3_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            vuint32m1_t v_d2 = __riscv_vget_v_u32m1x3_u32m1(tuple, 2);
            __riscv_vse32_v_u32m1(o2, v_d2, vl);
            break;
          }
          case 2: {
            vuint32m1x2_t tuple = __riscv_vlsseg2e32_v_u32m1x2(i, input_stride, vl);

            vuint32m1_t v_d0 = __riscv_vget_v_u32m1x2_u32m1(tuple, 0);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            vuint32m1_t v_d1 = __riscv_vget_v_u32m1x2_u32m1(tuple, 1);
            __riscv_vse32_v_u32m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint32m1_t v_d0 = __riscv_vlse32_v_u32m1(i, input_stride, vl);
            __riscv_vse32_v_u32m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      if (bh & 8) {
        o7 += 8;
        o6 += 8;
        o5 += 8;
        o4 += 8;
        o3 += 8;
        o2 += 8;
        o1 += 8;
        o0 += 8;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 8);
      }
      if (bh & 4) {
        o7 += 4;
        o6 += 4;
        o5 += 4;
        o4 += 4;
        o3 += 4;
        o2 += 4;
        o1 += 4;
        o0 += 4;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 4);
      }
      if (bh & 2) {
        o7 += 2;
        o6 += 2;
        o5 += 2;
        o4 += 2;
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        i = (uint32_t*) ((uintptr_t) i + input_stride * 2);
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);

    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint32_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint32_t*) ((uintptr_t) o3 + output_reset);
    o4 = (uint32_t*) ((uintptr_t) o4 + output_reset);
    o5 = (uint32_t*) ((uintptr_t) o5 + output_reset);
    o6 = (uint32_t*) ((uintptr_t) o6 + output_reset);
    o7 = (uint32_t*) ((uintptr_t) o7 + output_reset);

    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
