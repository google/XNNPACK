// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vmulcaddc/rvv.c.in
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


void xnn_f32_vmulcaddc_minmax_ukernel_c4__rvv_2x(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float min_val = params->scalar.min;
  const float max_val = params->scalar.max;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels / sizeof(float);

    while (c > 0) {
      size_t vl = __riscv_vsetvl_e32m1(c > 4 ? 4 : c);

      vfloat32m1_t vscale = __riscv_vle32_v_f32m1(w, vl);
      vfloat32m1_t vbias = __riscv_vle32_v_f32m1(w + 4, vl);
      w += 8;

      vfloat32m1_t vacc0 = __riscv_vle32_v_f32m1(i0, vl); i0 += vl;
      vfloat32m1_t vacc1 = __riscv_vle32_v_f32m1(i1, vl); i1 += vl;

      vacc0 = __riscv_vfmadd_vv_f32m1(vacc0, vscale, vbias, vl);
      vacc1 = __riscv_vfmadd_vv_f32m1(vacc1, vscale, vbias, vl);

      vbool32_t nan0 = __riscv_vmfne_vv_f32m1_b32(vacc0, vacc0, vl);
      vbool32_t nan1 = __riscv_vmfne_vv_f32m1_b32(vacc1, vacc1, vl);

      vacc0 = __riscv_vfmax_vf_f32m1(vacc0, min_val, vl);
      vacc1 = __riscv_vfmax_vf_f32m1(vacc1, min_val, vl);

      vacc0 = __riscv_vfmin_vf_f32m1(vacc0, max_val, vl);
      vacc1 = __riscv_vfmin_vf_f32m1(vacc1, max_val, vl);

      vacc0 = __riscv_vfmerge_vfm_f32m1(vacc0, __builtin_nanf(""), nan0, vl);
      vacc1 = __riscv_vfmerge_vfm_f32m1(vacc1, __builtin_nanf(""), nan1, vl);

      __riscv_vse32_v_f32m1(o0, vacc0, vl); o0 += vl;
      __riscv_vse32_v_f32m1(o1, vacc1, vl); o1 += vl;

      c -= vl;
    }

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
