// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vmulcaddc/rvvfp16arith.c.in
//   Generator: tools/xngen
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vmulcaddc.h"


void xnn_f16_vmulcaddc_minmax_ukernel_c8v__rvvfp16arith_2x(
    size_t rows,
    size_t channels,
    const xnn_float16* restrict input,
    size_t input_stride,
    const xnn_float16* restrict weights,
    xnn_float16* restrict output,
    size_t output_stride,
    const struct xnn_f16_minmax_params* restrict params) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(xnn_float16) == 0);

  const xnn_float16* i0 = input;
  xnn_float16* o0 = output;
  const xnn_float16* i1 = (const xnn_float16*) ((uintptr_t) i0 + input_stride);
  xnn_float16* o1 = (xnn_float16*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const xnn_float16 vmin_val = params->scalar.min;
  const xnn_float16 vmax_val = params->scalar.max;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const xnn_float16* w = weights;
    size_t c = channels / sizeof(xnn_float16);
    const size_t vlmax = __riscv_vsetvlmax_e16m8();

    while (c > 0) {
      size_t vl = __riscv_vsetvl_e16m8(c);

      vfloat16m8_t vscale = __riscv_vle16_v_f16m8(w, vl);
      vfloat16m8_t vbias = __riscv_vle16_v_f16m8(w + vlmax, vl);
      w += 2 * vlmax;

      vfloat16m8_t vacc0 = __riscv_vle16_v_f16m8(i0, vl); i0 += vl;
      vfloat16m8_t vacc1 = __riscv_vle16_v_f16m8(i1, vl); i1 += vl;

      vacc0 = __riscv_vfmadd_vv_f16m8(vacc0, vscale, vbias, vl);
      vacc1 = __riscv_vfmadd_vv_f16m8(vacc1, vscale, vbias, vl);

      vacc0 = __riscv_vfmax_vf_f16m8(vacc0, vmin_val, vl);
      vacc1 = __riscv_vfmax_vf_f16m8(vacc1, vmin_val, vl);

      vacc0 = __riscv_vfmin_vf_f16m8(vacc0, vmax_val, vl);
      vacc1 = __riscv_vfmin_vf_f16m8(vacc1, vmax_val, vl);

      __riscv_vse16_v_f16m8(o0, vacc0, vl); o0 += vl;
      __riscv_vse16_v_f16m8(o1, vacc1, vl); o1 += vl;

      c -= vl;
    }

    i0 = (const xnn_float16*) ((uintptr_t) i0 + input_increment);
    o0 = (xnn_float16*) ((uintptr_t) o0 + output_increment);
    i1 = (const xnn_float16*) ((uintptr_t) i1 + input_increment);
    o1 = (xnn_float16*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
