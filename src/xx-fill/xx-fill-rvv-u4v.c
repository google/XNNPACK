// Copyright 2026 Léandre Le Duc
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>

#include "src/xnnpack/fill.h"

void xnn_xx_fill_ukernel__rvv_u4v(size_t rows, size_t channels, void* output,
                                  size_t output_stride,
                                  const uint32_t fill_pattern) {
  assert(rows != 0);
  assert(channels != 0);

  const size_t output_increment = output_stride - channels;

  const vuint8m4_t vfill = __riscv_vreinterpret_v_u32m4_u8m4(
      __riscv_vmv_v_x_u32m4(fill_pattern, __riscv_vsetvlmax_e32m4()));

  do {
    size_t c = channels;
    for (size_t vl; c > 0; c -= vl, output += vl) {
      vl = __riscv_vsetvl_e8m4(c);
      __riscv_vse8_v_u8m4(output, vfill, vl);
    }
    output = (void*)((uintptr_t)output + output_increment);
  } while (--rows != 0);
}
