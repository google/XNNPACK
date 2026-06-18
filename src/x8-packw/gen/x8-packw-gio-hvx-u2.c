// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/gio-simd.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/simd/s8-hvx.h"



// Pack pre-transposed weights (GIO) for use by int8 gemm
void xnn_x8_packw_gemm_gio_ukernel_x128__hvx_u2(
  size_t g,                  // Batch size (outer loop). usually 1
  size_t nc,                 // Number of columns and typically large
  size_t kc,                 // Number of rows and typically small
  size_t nr,                 // Matches gemm and is a multiple of vector sizes
  size_t kr,                 // unused - must be 1
  size_t sr,                 // unused - must be 1
  size_t k_stride,           // Elements per row (typically same as nc)
  const int8_t* weights,     // Weights to pack. unaligned, unpadded
  const uint32_t* bias,      // Bias to pack. unaligned, unpadded, can be NULL
  const void* scale,         // unused
  int8_t* packed_weights,    // packed weights output buffer - aligned, padded
  size_t extra_bytes,        // number of extra bytes between weights. aligned
  const void* params)        // unused
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 128);   // This kernel is for NR=128
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const xnn_simd_s8_t vzero = xnn_set1_s8(0);
  (void) vzero;
  const uint32_t* b = bias;
  int8_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 128
    const int8_t* w = weights;
    size_t n = nc;

    for (; n >= 128; n -= 128) {
      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, 128 * sizeof(uint32_t));
        packed_w += 128 * sizeof(uint32_t);
        b += 128;
      } else {
        memset(packed_w, 0, 128 * sizeof(uint32_t));
        packed_w += 128 * sizeof(uint32_t);
      }

      // KC main loop 2x128
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s8_t v0_0 = xnn_loadu_s8(w + 0 + 0 * k_stride);
        const xnn_simd_s8_t v0_1 = xnn_loadu_s8(w + 0 + 1 * k_stride);
        xnn_store_s8(packed_w + 0, v0_0);
        xnn_store_s8(packed_w + 128, v0_1);
        w += k_stride * 2;
        packed_w += 256;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s8_t v0 = xnn_loadu_s8(w + 0);
        xnn_store_s8(packed_w + 0, v0);
        w += k_stride;
        packed_w += 128;
      }
      packed_w += extra_bytes;
      w = w - kc * k_stride + 128;  // Advance to next column of 128 int8_t
    }

    // NC remainder (1..127)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid elements (depends on n).

      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, n * sizeof(uint32_t));
        memset(packed_w + n * sizeof(uint32_t), 0, (128 - n) * sizeof(uint32_t));
        packed_w += 128 * sizeof(uint32_t);
        b += n;
      } else {
        memset(packed_w, 0, 128 * sizeof(uint32_t));
        packed_w += 128 * sizeof(uint32_t);
      }

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s8_t v0 = xnn_load_tail_safe_s8(w, n);
        xnn_store_s8(packed_w + 0, v0);
        w += k_stride;
        packed_w += 128;
      }
      packed_w += extra_bytes;
    }
    weights += nc * kc;
  } while (--g != 0);
}

// Pack pre-transposed weights (GIO) for use by int8 gemm
void xnn_x8_packw_gemm_gio_ukernel_x256__hvx_u2(
  size_t g,                  // Batch size (outer loop). usually 1
  size_t nc,                 // Number of columns and typically large
  size_t kc,                 // Number of rows and typically small
  size_t nr,                 // Matches gemm and is a multiple of vector sizes
  size_t kr,                 // unused - must be 1
  size_t sr,                 // unused - must be 1
  size_t k_stride,           // Elements per row (typically same as nc)
  const int8_t* weights,     // Weights to pack. unaligned, unpadded
  const uint32_t* bias,      // Bias to pack. unaligned, unpadded, can be NULL
  const void* scale,         // unused
  int8_t* packed_weights,    // packed weights output buffer - aligned, padded
  size_t extra_bytes,        // number of extra bytes between weights. aligned
  const void* params)        // unused
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 256);   // This kernel is for NR=256
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const xnn_simd_s8_t vzero = xnn_set1_s8(0);
  (void) vzero;
  const uint32_t* b = bias;
  int8_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 256
    const int8_t* w = weights;
    size_t n = nc;

    for (; n >= 256; n -= 256) {
      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, 256 * sizeof(uint32_t));
        packed_w += 256 * sizeof(uint32_t);
        b += 256;
      } else {
        memset(packed_w, 0, 256 * sizeof(uint32_t));
        packed_w += 256 * sizeof(uint32_t);
      }

      // KC main loop 2x256
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s8_t v0_0 = xnn_loadu_s8(w + 0 + 0 * k_stride);
        const xnn_simd_s8_t v1_0 = xnn_loadu_s8(w + 128 + 0 * k_stride);
        const xnn_simd_s8_t v0_1 = xnn_loadu_s8(w + 0 + 1 * k_stride);
        const xnn_simd_s8_t v1_1 = xnn_loadu_s8(w + 128 + 1 * k_stride);
        xnn_store_s8(packed_w + 0, v0_0);
        xnn_store_s8(packed_w + 128, v1_0);
        xnn_store_s8(packed_w + 256, v0_1);
        xnn_store_s8(packed_w + 384, v1_1);
        w += k_stride * 2;
        packed_w += 512;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s8_t v0 = xnn_loadu_s8(w + 0);
        const xnn_simd_s8_t v1 = xnn_loadu_s8(w + 128);
        xnn_store_s8(packed_w + 0, v0);
        xnn_store_s8(packed_w + 128, v1);
        w += k_stride;
        packed_w += 256;
      }
      packed_w += extra_bytes;
      w = w - kc * k_stride + 256;  // Advance to next column of 256 int8_t
    }

    // NC remainder (1..255)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid elements (depends on n).
      const size_t vcount0 = (int) (n - 0) < 0 ? 0 : ((int) (n - 0) > 128 ? 128 : n - 0);
      const size_t vcount1 = (int) (n - 128) < 0 ? 0 : ((int) (n - 128) > 128 ? 128 : n - 128);

      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, n * sizeof(uint32_t));
        memset(packed_w + n * sizeof(uint32_t), 0, (256 - n) * sizeof(uint32_t));
        packed_w += 256 * sizeof(uint32_t);
        b += n;
      } else {
        memset(packed_w, 0, 256 * sizeof(uint32_t));
        packed_w += 256 * sizeof(uint32_t);
      }

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s8_t v0 = vcount0 == 128 ? xnn_loadu_s8(w + 0) : (vcount0 == 0 ? vzero : xnn_load_tail_safe_s8(w + 0, vcount0));
        const xnn_simd_s8_t v1 = vcount1 == 128 ? xnn_loadu_s8(w + 128) : (vcount1 == 0 ? vzero : xnn_load_tail_safe_s8(w + 128, vcount1));
        xnn_store_s8(packed_w + 0, v0);
        xnn_store_s8(packed_w + 128, v1);
        w += k_stride;
        packed_w += 256;
      }
      packed_w += extra_bytes;
    }
    weights += nc * kc;
  } while (--g != 0);
}
