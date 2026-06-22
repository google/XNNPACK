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
#include "src/xnnpack/simd/s8-neon.h"



// Pack pre-transposed weights (GIO) for use by int8 gemm
void xnn_x8_packw_gemm_gio_ukernel_x16__neon_u2(
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
  assert(nr == 16);   // This kernel is for NR=16
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
    // NC main loop multiple of 16
    const int8_t* w = weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, 16 * sizeof(uint32_t));
        packed_w += 16 * sizeof(uint32_t);
        b += 16;
      } else {
        memset(packed_w, 0, 16 * sizeof(uint32_t));
        packed_w += 16 * sizeof(uint32_t);
      }

      // KC main loop 2x16
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s8_t v0_0 = xnn_loadu_s8(w + 0 + 0 * k_stride);
        const xnn_simd_s8_t v0_1 = xnn_loadu_s8(w + 0 + 1 * k_stride);
        xnn_store_s8(packed_w + 0, v0_0);
        xnn_store_s8(packed_w + 16, v0_1);
        w += k_stride * 2;
        packed_w += 32;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s8_t v0 = xnn_loadu_s8(w + 0);
        xnn_store_s8(packed_w + 0, v0);
        w += k_stride;
        packed_w += 16;
      }
      packed_w += extra_bytes;
      w = w - kc * k_stride + 16;  // Advance to next column of 16 int8_t
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid elements (depends on n).

      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, n * sizeof(uint32_t));
        memset(packed_w + n * sizeof(uint32_t), 0, (16 - n) * sizeof(uint32_t));
        packed_w += 16 * sizeof(uint32_t);
        b += n;
      } else {
        memset(packed_w, 0, 16 * sizeof(uint32_t));
        packed_w += 16 * sizeof(uint32_t);
      }

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s8_t v0 = xnn_load_tail_safe_s8(w, n);
        xnn_store_s8(packed_w + 0, v0);
        w += k_stride;
        packed_w += 16;
      }
      packed_w += extra_bytes;
    }
    weights += nc * kc;
  } while (--g != 0);
}

// Pack pre-transposed weights (GIO) for use by int8 gemm
void xnn_x8_packw_gemm_gio_ukernel_x32__neon_u2(
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
  assert(nr == 32);   // This kernel is for NR=32
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
    // NC main loop multiple of 32
    const int8_t* w = weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, 32 * sizeof(uint32_t));
        packed_w += 32 * sizeof(uint32_t);
        b += 32;
      } else {
        memset(packed_w, 0, 32 * sizeof(uint32_t));
        packed_w += 32 * sizeof(uint32_t);
      }

      // KC main loop 2x32
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s8_t v0_0 = xnn_loadu_s8(w + 0 + 0 * k_stride);
        const xnn_simd_s8_t v1_0 = xnn_loadu_s8(w + 16 + 0 * k_stride);
        const xnn_simd_s8_t v0_1 = xnn_loadu_s8(w + 0 + 1 * k_stride);
        const xnn_simd_s8_t v1_1 = xnn_loadu_s8(w + 16 + 1 * k_stride);
        xnn_store_s8(packed_w + 0, v0_0);
        xnn_store_s8(packed_w + 16, v1_0);
        xnn_store_s8(packed_w + 32, v0_1);
        xnn_store_s8(packed_w + 48, v1_1);
        w += k_stride * 2;
        packed_w += 64;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s8_t v0 = xnn_loadu_s8(w + 0);
        const xnn_simd_s8_t v1 = xnn_loadu_s8(w + 16);
        xnn_store_s8(packed_w + 0, v0);
        xnn_store_s8(packed_w + 16, v1);
        w += k_stride;
        packed_w += 32;
      }
      packed_w += extra_bytes;
      w = w - kc * k_stride + 32;  // Advance to next column of 32 int8_t
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid elements (depends on n).
      const size_t vcount0 = (int) (n - 0) < 0 ? 0 : ((int) (n - 0) > 16 ? 16 : n - 0);
      const size_t vcount1 = (int) (n - 16) < 0 ? 0 : ((int) (n - 16) > 16 ? 16 : n - 16);

      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, n * sizeof(uint32_t));
        memset(packed_w + n * sizeof(uint32_t), 0, (32 - n) * sizeof(uint32_t));
        packed_w += 32 * sizeof(uint32_t);
        b += n;
      } else {
        memset(packed_w, 0, 32 * sizeof(uint32_t));
        packed_w += 32 * sizeof(uint32_t);
      }

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s8_t v0 = vcount0 == 16 ? xnn_loadu_s8(w + 0) : (vcount0 == 0 ? vzero : xnn_load_tail_safe_s8(w + 0, vcount0));
        const xnn_simd_s8_t v1 = vcount1 == 16 ? xnn_loadu_s8(w + 16) : (vcount1 == 0 ? vzero : xnn_load_tail_safe_s8(w + 16, vcount1));
        xnn_store_s8(packed_w + 0, v0);
        xnn_store_s8(packed_w + 16, v1);
        w += k_stride;
        packed_w += 32;
      }
      packed_w += extra_bytes;
    }
    weights += nc * kc;
  } while (--g != 0);
}

// Pack pre-transposed weights (GIO) for use by int8 gemm
void xnn_x8_packw_gemm_gio_ukernel_x48__neon_u2(
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
  assert(nr == 48);   // This kernel is for NR=48
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
    // NC main loop multiple of 48
    const int8_t* w = weights;
    size_t n = nc;

    for (; n >= 48; n -= 48) {
      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, 48 * sizeof(uint32_t));
        packed_w += 48 * sizeof(uint32_t);
        b += 48;
      } else {
        memset(packed_w, 0, 48 * sizeof(uint32_t));
        packed_w += 48 * sizeof(uint32_t);
      }

      // KC main loop 2x48
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s8_t v0_0 = xnn_loadu_s8(w + 0 + 0 * k_stride);
        const xnn_simd_s8_t v1_0 = xnn_loadu_s8(w + 16 + 0 * k_stride);
        const xnn_simd_s8_t v2_0 = xnn_loadu_s8(w + 32 + 0 * k_stride);
        const xnn_simd_s8_t v0_1 = xnn_loadu_s8(w + 0 + 1 * k_stride);
        const xnn_simd_s8_t v1_1 = xnn_loadu_s8(w + 16 + 1 * k_stride);
        const xnn_simd_s8_t v2_1 = xnn_loadu_s8(w + 32 + 1 * k_stride);
        xnn_store_s8(packed_w + 0, v0_0);
        xnn_store_s8(packed_w + 16, v1_0);
        xnn_store_s8(packed_w + 32, v2_0);
        xnn_store_s8(packed_w + 48, v0_1);
        xnn_store_s8(packed_w + 64, v1_1);
        xnn_store_s8(packed_w + 80, v2_1);
        w += k_stride * 2;
        packed_w += 96;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s8_t v0 = xnn_loadu_s8(w + 0);
        const xnn_simd_s8_t v1 = xnn_loadu_s8(w + 16);
        const xnn_simd_s8_t v2 = xnn_loadu_s8(w + 32);
        xnn_store_s8(packed_w + 0, v0);
        xnn_store_s8(packed_w + 16, v1);
        xnn_store_s8(packed_w + 32, v2);
        w += k_stride;
        packed_w += 48;
      }
      packed_w += extra_bytes;
      w = w - kc * k_stride + 48;  // Advance to next column of 48 int8_t
    }

    // NC remainder (1..47)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid elements (depends on n).
      const size_t vcount0 = (int) (n - 0) < 0 ? 0 : ((int) (n - 0) > 16 ? 16 : n - 0);
      const size_t vcount1 = (int) (n - 16) < 0 ? 0 : ((int) (n - 16) > 16 ? 16 : n - 16);
      const size_t vcount2 = (int) (n - 32) < 0 ? 0 : ((int) (n - 32) > 16 ? 16 : n - 32);

      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, n * sizeof(uint32_t));
        memset(packed_w + n * sizeof(uint32_t), 0, (48 - n) * sizeof(uint32_t));
        packed_w += 48 * sizeof(uint32_t);
        b += n;
      } else {
        memset(packed_w, 0, 48 * sizeof(uint32_t));
        packed_w += 48 * sizeof(uint32_t);
      }

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s8_t v0 = vcount0 == 16 ? xnn_loadu_s8(w + 0) : (vcount0 == 0 ? vzero : xnn_load_tail_safe_s8(w + 0, vcount0));
        const xnn_simd_s8_t v1 = vcount1 == 16 ? xnn_loadu_s8(w + 16) : (vcount1 == 0 ? vzero : xnn_load_tail_safe_s8(w + 16, vcount1));
        const xnn_simd_s8_t v2 = vcount2 == 16 ? xnn_loadu_s8(w + 32) : (vcount2 == 0 ? vzero : xnn_load_tail_safe_s8(w + 32, vcount2));
        xnn_store_s8(packed_w + 0, v0);
        xnn_store_s8(packed_w + 16, v1);
        xnn_store_s8(packed_w + 32, v2);
        w += k_stride;
        packed_w += 48;
      }
      packed_w += extra_bytes;
    }
    weights += nc * kc;
  } while (--g != 0);
}

// Pack pre-transposed weights (GIO) for use by int8 gemm
void xnn_x8_packw_gemm_gio_ukernel_x64__neon_u2(
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
  assert(nr == 64);   // This kernel is for NR=64
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
    // NC main loop multiple of 64
    const int8_t* w = weights;
    size_t n = nc;

    for (; n >= 64; n -= 64) {
      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, 64 * sizeof(uint32_t));
        packed_w += 64 * sizeof(uint32_t);
        b += 64;
      } else {
        memset(packed_w, 0, 64 * sizeof(uint32_t));
        packed_w += 64 * sizeof(uint32_t);
      }

      // KC main loop 2x64
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const xnn_simd_s8_t v0_0 = xnn_loadu_s8(w + 0 + 0 * k_stride);
        const xnn_simd_s8_t v1_0 = xnn_loadu_s8(w + 16 + 0 * k_stride);
        const xnn_simd_s8_t v2_0 = xnn_loadu_s8(w + 32 + 0 * k_stride);
        const xnn_simd_s8_t v3_0 = xnn_loadu_s8(w + 48 + 0 * k_stride);
        const xnn_simd_s8_t v0_1 = xnn_loadu_s8(w + 0 + 1 * k_stride);
        const xnn_simd_s8_t v1_1 = xnn_loadu_s8(w + 16 + 1 * k_stride);
        const xnn_simd_s8_t v2_1 = xnn_loadu_s8(w + 32 + 1 * k_stride);
        const xnn_simd_s8_t v3_1 = xnn_loadu_s8(w + 48 + 1 * k_stride);
        xnn_store_s8(packed_w + 0, v0_0);
        xnn_store_s8(packed_w + 16, v1_0);
        xnn_store_s8(packed_w + 32, v2_0);
        xnn_store_s8(packed_w + 48, v3_0);
        xnn_store_s8(packed_w + 64, v0_1);
        xnn_store_s8(packed_w + 80, v1_1);
        xnn_store_s8(packed_w + 96, v2_1);
        xnn_store_s8(packed_w + 112, v3_1);
        w += k_stride * 2;
        packed_w += 128;
      }

      // KC remainder loop
      for (; k > 0; --k) {
        const xnn_simd_s8_t v0 = xnn_loadu_s8(w + 0);
        const xnn_simd_s8_t v1 = xnn_loadu_s8(w + 16);
        const xnn_simd_s8_t v2 = xnn_loadu_s8(w + 32);
        const xnn_simd_s8_t v3 = xnn_loadu_s8(w + 48);
        xnn_store_s8(packed_w + 0, v0);
        xnn_store_s8(packed_w + 16, v1);
        xnn_store_s8(packed_w + 32, v2);
        xnn_store_s8(packed_w + 48, v3);
        w += k_stride;
        packed_w += 64;
      }
      packed_w += extra_bytes;
      w = w - kc * k_stride + 64;  // Advance to next column of 64 int8_t
    }

    // NC remainder (1..63)
    if XNN_UNLIKELY(n != 0) {
      // Prepare count for valid elements (depends on n).
      const size_t vcount0 = (int) (n - 0) < 0 ? 0 : ((int) (n - 0) > 16 ? 16 : n - 0);
      const size_t vcount1 = (int) (n - 16) < 0 ? 0 : ((int) (n - 16) > 16 ? 16 : n - 16);
      const size_t vcount2 = (int) (n - 32) < 0 ? 0 : ((int) (n - 32) > 16 ? 16 : n - 32);
      const size_t vcount3 = (int) (n - 48) < 0 ? 0 : ((int) (n - 48) > 16 ? 16 : n - 48);

      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, n * sizeof(uint32_t));
        memset(packed_w + n * sizeof(uint32_t), 0, (64 - n) * sizeof(uint32_t));
        packed_w += 64 * sizeof(uint32_t);
        b += n;
      } else {
        memset(packed_w, 0, 64 * sizeof(uint32_t));
        packed_w += 64 * sizeof(uint32_t);
      }

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const xnn_simd_s8_t v0 = vcount0 == 16 ? xnn_loadu_s8(w + 0) : (vcount0 == 0 ? vzero : xnn_load_tail_safe_s8(w + 0, vcount0));
        const xnn_simd_s8_t v1 = vcount1 == 16 ? xnn_loadu_s8(w + 16) : (vcount1 == 0 ? vzero : xnn_load_tail_safe_s8(w + 16, vcount1));
        const xnn_simd_s8_t v2 = vcount2 == 16 ? xnn_loadu_s8(w + 32) : (vcount2 == 0 ? vzero : xnn_load_tail_safe_s8(w + 32, vcount2));
        const xnn_simd_s8_t v3 = vcount3 == 16 ? xnn_loadu_s8(w + 48) : (vcount3 == 0 ? vzero : xnn_load_tail_safe_s8(w + 48, vcount3));
        xnn_store_s8(packed_w + 0, v0);
        xnn_store_s8(packed_w + 16, v1);
        xnn_store_s8(packed_w + 32, v2);
        xnn_store_s8(packed_w + 48, v3);
        w += k_stride;
        packed_w += 64;
      }
      packed_w += extra_bytes;
    }
    weights += nc * kc;
  } while (--g != 0);
}
