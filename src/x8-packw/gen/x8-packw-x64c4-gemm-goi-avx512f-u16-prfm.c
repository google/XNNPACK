// Auto-generated file. Do not edit!
//   Template: src/x8-packw/c4.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/packw.h>


void xnn_x8_packw_gemm_goi_ukernel_x64c4__avx512f_u16_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const uint32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 64);   // This kernel is for NR=64
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  assert((kc & 3) == 0);

  xnn_x32_packw_gemm_goi_ukernel_x64__avx512f_u4_prfm(g, nc, kc / 4, nr, 1, sr,
    (const uint32_t*) weights,
    (const uint32_t*) bias,
    scale,
    (uint32_t*)packed_weights,
    extra_bytes,
    params);
}
