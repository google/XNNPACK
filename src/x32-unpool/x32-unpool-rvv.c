// Copyright 2026 Léandre Le Duc
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>

#include "src/xnnpack/unpool.h"

void xnn_x32_unpool_ukernel__rvv(size_t kernel_elements, size_t channels,
                                 uint32_t fill, const uint32_t* input,
                                 const uint32_t* index, uint32_t** output) {
  uint32_t** os = output;

  do {
    --kernel_elements;

    uint32_t* o = os[kernel_elements];
    const uint32_t* in = input;
    const uint32_t* idx = index;
    size_t c = channels;

    for (size_t vl; c > 0; c -= vl, in += vl, idx += vl, o += vl) {
      vl = __riscv_vsetvl_e32m1(c);
      vuint32m1_t vval = __riscv_vle32_v_u32m1(in, vl);
      vuint32m1_t vidx = __riscv_vle32_v_u32m1(idx, vl);

      vuint32m1_t vfill = __riscv_vmv_v_x_u32m1(fill, vl);

      // Construct the a fill mask based on the vl equality
      vbool32_t m = __riscv_vmseq_vx_u32m1_b32(vidx, kernel_elements, vl);

      // Final value by mask filling
      vuint32m1_t vfinal = __riscv_vmerge_vvm_u32m1(vfill, vval, m, vl);

      // Store the final value
      __riscv_vse32_v_u32m1(o, vfinal, vl);
    }
  } while (kernel_elements > 0);
}
