// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/scalar-pipelined.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_1x1__scalar_pipelined(
    uint32_t m,
    uint32_t n,
    const float*restrict a,
    const float*restrict weights,
    const int32_t*restrict widx_dmap,
    const uint32_t*restrict nidx_nnzmap,
    float*restrict c,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(m != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  size_t i = m;
  while XNN_LIKELY(i >= 1) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    float vw = *w++;
    intptr_t diff = *dmap++;
    float va0 = a[0];
    size_t j = n;
    do {
      uint32_t nnz = *nnzmap++;
      float vacc0 = vw;
      vw = *w++;
      if XNN_LIKELY(nnz != 0) {
        do {
          vacc0 += va0 * vw;
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);

          diff = *dmap++;
          vw = *w++;
          va0 = a[0];
        } while (--nnz != 0);
      }
      float vout0 = math_min_f32(vacc0, vmax);
      vout0 = math_max_f32(vout0, vmin);
      c[0] = vout0;
      c += m;
    } while (--j != 0);
    c -= m * n;
    c += 1;
    a += 1;
    i -= 1;
  }
  if XNN_UNLIKELY(i != 0) {
  }
}
