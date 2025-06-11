// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/gio-simd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/simd/s32-neon.h"

#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"


// Pack pre-transposed weights (GIO) for use by f32-gemm
void xnn_x32_packw_gemm_gio_ukernel_x4__neon_u2(
  size_t g,                  // Batch size (outer loop).  usually 1
  size_t nc,                 // Number of columns and typically large
  size_t kc,                 // Number of rows and typically small
  size_t nr,                 // Matches gemm and is a multiple of vector sizes
  size_t kr,                 // unused - must be 1
  size_t sr,                 // unused - must be 1
  size_t k_stride,           // Elements per row (typically same as nc)
  const uint32_t* weights,   // Weights to pack. unaligned, unpadded
  const uint32_t* bias,      // Bias to pack. unaligned, unpadded, can be NULL
  const void* scale,         // unused
  uint32_t* packed_weights,  // packed weights output buffer - aligned, padded
  size_t extra_bytes,        // number of extra bytes between weights. aligned
  const void* params)        // unused
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 4);   // This kernel is for NR=4
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const xnn_simd_s32_t vzero = xnn_set1_s32(0);
  const int32_t* b = (const int32_t*) bias;
  int32_t* packed_w = (int32_t*) packed_weights;
  do {
    // NC main loop multiple of 4
    const int32_t* w = (const int32_t*) weights;
    size_t n = nc;

    for (; n >= 4; n -= 4) {
      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_loadu_s32(b + 0);
        xnn_store_s32(packed_w + 0, vb0);
        b += 4;
      } else {
        xnn_store_s32(packed_w + 0, vzero);
      }
      packed_w += 4;

      // KC main loop 2x4
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s32_t v0_0 = xnn_loadu_s32(w + 0 + 0 * k_stride);
        const xnn_simd_s32_t v0_1 = xnn_loadu_s32(w + 0 + 1 * k_stride);
        xnn_store_s32(packed_w + 0, v0_0);
        xnn_store_s32(packed_w + 4, v0_1);
        w += k_stride * 2;
        packed_w += 8;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s32_t v0 = xnn_loadu_s32(w + 0);
        xnn_store_s32(packed_w + 0, v0);
        w += k_stride;
        packed_w += 4;
      }
      w = w - kc * k_stride + 4;  // Advance to next column of 4 int32_t
    }

    // NC remainder (1..3)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid 32-bit elements (depends on n).
      const size_t vcount0 = n;

      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_load_tail_safe_s32(b + 0, vcount0);
        xnn_store_s32(packed_w + 0, vb0);
        b += n;
      } else {
        xnn_store_s32(packed_w + 0, vzero);
      }
      packed_w += 4;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s32_t v0 = xnn_load_tail_safe_s32(w + 0, vcount0);
        xnn_store_s32(packed_w + 0, v0);
        w += k_stride;
        packed_w += 4;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}

// Pack pre-transposed weights (GIO) for use by f32-gemm
void xnn_x32_packw_gemm_gio_ukernel_x8__neon_u2(
  size_t g,                  // Batch size (outer loop).  usually 1
  size_t nc,                 // Number of columns and typically large
  size_t kc,                 // Number of rows and typically small
  size_t nr,                 // Matches gemm and is a multiple of vector sizes
  size_t kr,                 // unused - must be 1
  size_t sr,                 // unused - must be 1
  size_t k_stride,           // Elements per row (typically same as nc)
  const uint32_t* weights,   // Weights to pack. unaligned, unpadded
  const uint32_t* bias,      // Bias to pack. unaligned, unpadded, can be NULL
  const void* scale,         // unused
  uint32_t* packed_weights,  // packed weights output buffer - aligned, padded
  size_t extra_bytes,        // number of extra bytes between weights. aligned
  const void* params)        // unused
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);   // This kernel is for NR=8
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const xnn_simd_s32_t vzero = xnn_set1_s32(0);
  const int32_t* b = (const int32_t*) bias;
  int32_t* packed_w = (int32_t*) packed_weights;
  do {
    // NC main loop multiple of 8
    const int32_t* w = (const int32_t*) weights;
    size_t n = nc;

    for (; n >= 8; n -= 8) {
      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_loadu_s32(b + 0);
        const xnn_simd_s32_t vb1 = xnn_loadu_s32(b + 4);
        xnn_store_s32(packed_w + 0, vb0);
        xnn_store_s32(packed_w + 4, vb1);
        b += 8;
      } else {
        xnn_store_s32(packed_w + 0, vzero);
        xnn_store_s32(packed_w + 4, vzero);
      }
      packed_w += 8;

      // KC main loop 2x8
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s32_t v0_0 = xnn_loadu_s32(w + 0 + 0 * k_stride);
        const xnn_simd_s32_t v1_0 = xnn_loadu_s32(w + 4 + 0 * k_stride);
        const xnn_simd_s32_t v0_1 = xnn_loadu_s32(w + 0 + 1 * k_stride);
        const xnn_simd_s32_t v1_1 = xnn_loadu_s32(w + 4 + 1 * k_stride);
        xnn_store_s32(packed_w + 0, v0_0);
        xnn_store_s32(packed_w + 4, v1_0);
        xnn_store_s32(packed_w + 8, v0_1);
        xnn_store_s32(packed_w + 12, v1_1);
        w += k_stride * 2;
        packed_w += 16;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s32_t v0 = xnn_loadu_s32(w + 0);
        const xnn_simd_s32_t v1 = xnn_loadu_s32(w + 4);
        xnn_store_s32(packed_w + 0, v0);
        xnn_store_s32(packed_w + 4, v1);
        w += k_stride;
        packed_w += 8;
      }
      w = w - kc * k_stride + 8;  // Advance to next column of 8 int32_t
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid 32-bit elements (depends on n).
      const size_t vcount0 = (int) (n - 0) < 0 ? 0 : ((int) (n - 0) > 4 ? 4 : n - 0);
      const size_t vcount1 = (int) (n - 4) < 0 ? 0 : ((int) (n - 4) > 4 ? 4 : n - 4);

      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_load_tail_safe_s32(b + 0, vcount0);
        const xnn_simd_s32_t vb1 = xnn_load_tail_safe_s32(b + 4, vcount1);
        xnn_store_s32(packed_w + 0, vb0);
        xnn_store_s32(packed_w + 4, vb1);
        b += n;
      } else {
        xnn_store_s32(packed_w + 0, vzero);
        xnn_store_s32(packed_w + 4, vzero);
      }
      packed_w += 8;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s32_t v0 = xnn_load_tail_safe_s32(w + 0, vcount0);
        const xnn_simd_s32_t v1 = xnn_load_tail_safe_s32(w + 4, vcount1);
        xnn_store_s32(packed_w + 0, v0);
        xnn_store_s32(packed_w + 4, v1);
        w += k_stride;
        packed_w += 8;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}

// Pack pre-transposed weights (GIO) for use by f32-gemm
void xnn_x32_packw_gemm_gio_ukernel_x12__neon_u2(
  size_t g,                  // Batch size (outer loop).  usually 1
  size_t nc,                 // Number of columns and typically large
  size_t kc,                 // Number of rows and typically small
  size_t nr,                 // Matches gemm and is a multiple of vector sizes
  size_t kr,                 // unused - must be 1
  size_t sr,                 // unused - must be 1
  size_t k_stride,           // Elements per row (typically same as nc)
  const uint32_t* weights,   // Weights to pack. unaligned, unpadded
  const uint32_t* bias,      // Bias to pack. unaligned, unpadded, can be NULL
  const void* scale,         // unused
  uint32_t* packed_weights,  // packed weights output buffer - aligned, padded
  size_t extra_bytes,        // number of extra bytes between weights. aligned
  const void* params)        // unused
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 12);   // This kernel is for NR=12
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const xnn_simd_s32_t vzero = xnn_set1_s32(0);
  const int32_t* b = (const int32_t*) bias;
  int32_t* packed_w = (int32_t*) packed_weights;
  do {
    // NC main loop multiple of 12
    const int32_t* w = (const int32_t*) weights;
    size_t n = nc;

    for (; n >= 12; n -= 12) {
      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_loadu_s32(b + 0);
        const xnn_simd_s32_t vb1 = xnn_loadu_s32(b + 4);
        const xnn_simd_s32_t vb2 = xnn_loadu_s32(b + 8);
        xnn_store_s32(packed_w + 0, vb0);
        xnn_store_s32(packed_w + 4, vb1);
        xnn_store_s32(packed_w + 8, vb2);
        b += 12;
      } else {
        xnn_store_s32(packed_w + 0, vzero);
        xnn_store_s32(packed_w + 4, vzero);
        xnn_store_s32(packed_w + 8, vzero);
      }
      packed_w += 12;

      // KC main loop 2x12
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s32_t v0_0 = xnn_loadu_s32(w + 0 + 0 * k_stride);
        const xnn_simd_s32_t v1_0 = xnn_loadu_s32(w + 4 + 0 * k_stride);
        const xnn_simd_s32_t v2_0 = xnn_loadu_s32(w + 8 + 0 * k_stride);
        const xnn_simd_s32_t v0_1 = xnn_loadu_s32(w + 0 + 1 * k_stride);
        const xnn_simd_s32_t v1_1 = xnn_loadu_s32(w + 4 + 1 * k_stride);
        const xnn_simd_s32_t v2_1 = xnn_loadu_s32(w + 8 + 1 * k_stride);
        xnn_store_s32(packed_w + 0, v0_0);
        xnn_store_s32(packed_w + 4, v1_0);
        xnn_store_s32(packed_w + 8, v2_0);
        xnn_store_s32(packed_w + 12, v0_1);
        xnn_store_s32(packed_w + 16, v1_1);
        xnn_store_s32(packed_w + 20, v2_1);
        w += k_stride * 2;
        packed_w += 24;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s32_t v0 = xnn_loadu_s32(w + 0);
        const xnn_simd_s32_t v1 = xnn_loadu_s32(w + 4);
        const xnn_simd_s32_t v2 = xnn_loadu_s32(w + 8);
        xnn_store_s32(packed_w + 0, v0);
        xnn_store_s32(packed_w + 4, v1);
        xnn_store_s32(packed_w + 8, v2);
        w += k_stride;
        packed_w += 12;
      }
      w = w - kc * k_stride + 12;  // Advance to next column of 12 int32_t
    }

    // NC remainder (1..11)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid 32-bit elements (depends on n).
      const size_t vcount0 = (int) (n - 0) < 0 ? 0 : ((int) (n - 0) > 4 ? 4 : n - 0);
      const size_t vcount1 = (int) (n - 4) < 0 ? 0 : ((int) (n - 4) > 4 ? 4 : n - 4);
      const size_t vcount2 = (int) (n - 8) < 0 ? 0 : ((int) (n - 8) > 4 ? 4 : n - 8);

      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_load_tail_safe_s32(b + 0, vcount0);
        const xnn_simd_s32_t vb1 = xnn_load_tail_safe_s32(b + 4, vcount1);
        const xnn_simd_s32_t vb2 = xnn_load_tail_safe_s32(b + 8, vcount2);
        xnn_store_s32(packed_w + 0, vb0);
        xnn_store_s32(packed_w + 4, vb1);
        xnn_store_s32(packed_w + 8, vb2);
        b += n;
      } else {
        xnn_store_s32(packed_w + 0, vzero);
        xnn_store_s32(packed_w + 4, vzero);
        xnn_store_s32(packed_w + 8, vzero);
      }
      packed_w += 12;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s32_t v0 = xnn_load_tail_safe_s32(w + 0, vcount0);
        const xnn_simd_s32_t v1 = xnn_load_tail_safe_s32(w + 4, vcount1);
        const xnn_simd_s32_t v2 = xnn_load_tail_safe_s32(w + 8, vcount2);
        xnn_store_s32(packed_w + 0, v0);
        xnn_store_s32(packed_w + 4, v1);
        xnn_store_s32(packed_w + 8, v2);
        w += k_stride;
        packed_w += 12;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}

// Pack pre-transposed weights (GIO) for use by f32-gemm
void xnn_x32_packw_gemm_gio_ukernel_x16__neon_u2(
  size_t g,                  // Batch size (outer loop).  usually 1
  size_t nc,                 // Number of columns and typically large
  size_t kc,                 // Number of rows and typically small
  size_t nr,                 // Matches gemm and is a multiple of vector sizes
  size_t kr,                 // unused - must be 1
  size_t sr,                 // unused - must be 1
  size_t k_stride,           // Elements per row (typically same as nc)
  const uint32_t* weights,   // Weights to pack. unaligned, unpadded
  const uint32_t* bias,      // Bias to pack. unaligned, unpadded, can be NULL
  const void* scale,         // unused
  uint32_t* packed_weights,  // packed weights output buffer - aligned, padded
  size_t extra_bytes,        // number of extra bytes between weights. aligned
  const void* params)        // unused
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const xnn_simd_s32_t vzero = xnn_set1_s32(0);
  const int32_t* b = (const int32_t*) bias;
  int32_t* packed_w = (int32_t*) packed_weights;
  do {
    // NC main loop multiple of 16
    const int32_t* w = (const int32_t*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_loadu_s32(b + 0);
        const xnn_simd_s32_t vb1 = xnn_loadu_s32(b + 4);
        const xnn_simd_s32_t vb2 = xnn_loadu_s32(b + 8);
        const xnn_simd_s32_t vb3 = xnn_loadu_s32(b + 12);
        xnn_store_s32(packed_w + 0, vb0);
        xnn_store_s32(packed_w + 4, vb1);
        xnn_store_s32(packed_w + 8, vb2);
        xnn_store_s32(packed_w + 12, vb3);
        b += 16;
      } else {
        xnn_store_s32(packed_w + 0, vzero);
        xnn_store_s32(packed_w + 4, vzero);
        xnn_store_s32(packed_w + 8, vzero);
        xnn_store_s32(packed_w + 12, vzero);
      }
      packed_w += 16;

      // KC main loop 2x16
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s32_t v0_0 = xnn_loadu_s32(w + 0 + 0 * k_stride);
        const xnn_simd_s32_t v1_0 = xnn_loadu_s32(w + 4 + 0 * k_stride);
        const xnn_simd_s32_t v2_0 = xnn_loadu_s32(w + 8 + 0 * k_stride);
        const xnn_simd_s32_t v3_0 = xnn_loadu_s32(w + 12 + 0 * k_stride);
        const xnn_simd_s32_t v0_1 = xnn_loadu_s32(w + 0 + 1 * k_stride);
        const xnn_simd_s32_t v1_1 = xnn_loadu_s32(w + 4 + 1 * k_stride);
        const xnn_simd_s32_t v2_1 = xnn_loadu_s32(w + 8 + 1 * k_stride);
        const xnn_simd_s32_t v3_1 = xnn_loadu_s32(w + 12 + 1 * k_stride);
        xnn_store_s32(packed_w + 0, v0_0);
        xnn_store_s32(packed_w + 4, v1_0);
        xnn_store_s32(packed_w + 8, v2_0);
        xnn_store_s32(packed_w + 12, v3_0);
        xnn_store_s32(packed_w + 16, v0_1);
        xnn_store_s32(packed_w + 20, v1_1);
        xnn_store_s32(packed_w + 24, v2_1);
        xnn_store_s32(packed_w + 28, v3_1);
        w += k_stride * 2;
        packed_w += 32;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s32_t v0 = xnn_loadu_s32(w + 0);
        const xnn_simd_s32_t v1 = xnn_loadu_s32(w + 4);
        const xnn_simd_s32_t v2 = xnn_loadu_s32(w + 8);
        const xnn_simd_s32_t v3 = xnn_loadu_s32(w + 12);
        xnn_store_s32(packed_w + 0, v0);
        xnn_store_s32(packed_w + 4, v1);
        xnn_store_s32(packed_w + 8, v2);
        xnn_store_s32(packed_w + 12, v3);
        w += k_stride;
        packed_w += 16;
      }
      w = w - kc * k_stride + 16;  // Advance to next column of 16 int32_t
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid 32-bit elements (depends on n).
      const size_t vcount0 = (int) (n - 0) < 0 ? 0 : ((int) (n - 0) > 4 ? 4 : n - 0);
      const size_t vcount1 = (int) (n - 4) < 0 ? 0 : ((int) (n - 4) > 4 ? 4 : n - 4);
      const size_t vcount2 = (int) (n - 8) < 0 ? 0 : ((int) (n - 8) > 4 ? 4 : n - 8);
      const size_t vcount3 = (int) (n - 12) < 0 ? 0 : ((int) (n - 12) > 4 ? 4 : n - 12);

      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_load_tail_safe_s32(b + 0, vcount0);
        const xnn_simd_s32_t vb1 = xnn_load_tail_safe_s32(b + 4, vcount1);
        const xnn_simd_s32_t vb2 = xnn_load_tail_safe_s32(b + 8, vcount2);
        const xnn_simd_s32_t vb3 = xnn_load_tail_safe_s32(b + 12, vcount3);
        xnn_store_s32(packed_w + 0, vb0);
        xnn_store_s32(packed_w + 4, vb1);
        xnn_store_s32(packed_w + 8, vb2);
        xnn_store_s32(packed_w + 12, vb3);
        b += n;
      } else {
        xnn_store_s32(packed_w + 0, vzero);
        xnn_store_s32(packed_w + 4, vzero);
        xnn_store_s32(packed_w + 8, vzero);
        xnn_store_s32(packed_w + 12, vzero);
      }
      packed_w += 16;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s32_t v0 = xnn_load_tail_safe_s32(w + 0, vcount0);
        const xnn_simd_s32_t v1 = xnn_load_tail_safe_s32(w + 4, vcount1);
        const xnn_simd_s32_t v2 = xnn_load_tail_safe_s32(w + 8, vcount2);
        const xnn_simd_s32_t v3 = xnn_load_tail_safe_s32(w + 12, vcount3);
        xnn_store_s32(packed_w + 0, v0);
        xnn_store_s32(packed_w + 4, v1);
        xnn_store_s32(packed_w + 8, v2);
        xnn_store_s32(packed_w + 12, v3);
        w += k_stride;
        packed_w += 16;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
