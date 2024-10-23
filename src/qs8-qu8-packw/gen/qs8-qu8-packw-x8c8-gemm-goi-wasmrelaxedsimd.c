// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-wasmdot.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/packw.h"


void xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__wasmrelaxedsimd(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params) XNN_OOB_READS
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const v128_t vone = wasm_i8x16_splat(1);
  const v128_t vzero = wasm_i32x4_splat(0);
  XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vzero);
  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 128): 128);
  v128_t vzeropoint = wasm_i32x4_splat((int32_t) izp);

  do {
    // NC main loop multiple of 8
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 8; n -= 8) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const v128_t vb0 = wasm_v128_load(b + 0);
        wasm_v128_store(out + 0, vb0);
        const v128_t vb1 = wasm_v128_load(b + 4);
        wasm_v128_store(out + 16, vb1);
        b += 8;
      } else {
        wasm_v128_store(out + 0, vzero);
        wasm_v128_store(out + 16, vzero);
      }
      out += 8 * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;

      v128_t vacc01 = wasm_i32x4_splat(0);
      v128_t vacc23 = wasm_i32x4_splat(0);
      v128_t vacc45 = wasm_i32x4_splat(0);
      v128_t vacc67 = wasm_i32x4_splat(0);

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        v128_t v0_01 = wasm_v128_load(w0);
        v128_t v1_01 = wasm_v128_load(w1);
        v128_t v2_01 = wasm_v128_load(w2);
        v128_t v3_01 = wasm_v128_load(w3);
        v128_t v4_01 = wasm_v128_load(w4);
        v128_t v5_01 = wasm_v128_load(w5);
        v128_t v6_01 = wasm_v128_load(w6);
        v128_t v7_01 = wasm_v128_load(w7);

        v128_t v01_0 = wasm_i64x2_shuffle(v0_01, v1_01, 0, 2);
        v128_t v01_1 = wasm_i64x2_shuffle(v0_01, v1_01, 1, 3);
        v128_t v23_0 = wasm_i64x2_shuffle(v2_01, v3_01, 0, 2);
        v128_t v23_1 = wasm_i64x2_shuffle(v2_01, v3_01, 1, 3);
        v128_t v45_0 = wasm_i64x2_shuffle(v4_01, v5_01, 0, 2);
        v128_t v45_1 = wasm_i64x2_shuffle(v4_01, v5_01, 1, 3);
        v128_t v67_0 = wasm_i64x2_shuffle(v6_01, v7_01, 0, 2);
        v128_t v67_1 = wasm_i64x2_shuffle(v6_01, v7_01, 1, 3);

        vacc01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v01_0, vone, vacc01);
        vacc01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v01_1, vone, vacc01);
        vacc23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v23_0, vone, vacc23);
        vacc23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v23_1, vone, vacc23);
        vacc45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v45_0, vone, vacc45);
        vacc45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v45_1, vone, vacc45);
        vacc67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v67_0, vone, vacc67);
        vacc67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v67_1, vone, vacc67);

        wasm_v128_store(out + 0, v01_0);
        wasm_v128_store(out + 16, v23_0);
        wasm_v128_store(out + 32, v45_0);
        wasm_v128_store(out + 48, v67_0);

        wasm_v128_store(out + 64, v01_1);
        wasm_v128_store(out + 80, v23_1);
        wasm_v128_store(out + 96, v45_1);
        wasm_v128_store(out + 112, v67_1);

        w0 += 16;
        w1 += 16;
        w2 += 16;
        w3 += 16;
        w4 += 16;
        w5 += 16;
        w6 += 16;
        w7 += 16;
        out += 128;
      }

      for (; k >= 8; k -= 8) {
        const v128_t v0 = wasm_v128_load64_splat(w0);
        const v128_t v1 = wasm_v128_load64_splat(w1);
        const v128_t v01 = wasm_i64x2_shuffle(v0, v1, 0, 3);
        const v128_t v2 = wasm_v128_load64_splat(w2);
        const v128_t v3 = wasm_v128_load64_splat(w3);
        const v128_t v23 = wasm_i64x2_shuffle(v2, v3, 0, 3);
        const v128_t v4 = wasm_v128_load64_splat(w4);
        const v128_t v5 = wasm_v128_load64_splat(w5);
        const v128_t v45 = wasm_i64x2_shuffle(v4, v5, 0, 3);
        const v128_t v6 = wasm_v128_load64_splat(w6);
        const v128_t v7 = wasm_v128_load64_splat(w7);
        const v128_t v67 = wasm_i64x2_shuffle(v6, v7, 0, 3);

        vacc01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v01, vone, vacc01);
        vacc23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v23, vone, vacc23);
        vacc45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v45, vone, vacc45);
        vacc67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v67, vone, vacc67);

        wasm_v128_store(out + 0, v01);
        wasm_v128_store(out + 16, v23);
        wasm_v128_store(out + 32, v45);
        wasm_v128_store(out + 48, v67);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
      }

      // KC remainder 1..KR-1
      if (k != 0) {
        assert(k >= 1 && k <= 7);

        const v128_t vmask = wasm_u64x2_shr(wasm_i32x4_splat(-1), (8 - k) * sizeof(int8_t) * 8);

        const v128_t v0 = wasm_v128_load64_splat(w0);
        const v128_t v1 = wasm_v128_load64_splat(w1);
        v128_t v01 = wasm_i64x2_shuffle(v0, v1, 0, 3);
        v01 = wasm_v128_and(v01, vmask);
        const v128_t v2 = wasm_v128_load64_splat(w2);
        const v128_t v3 = wasm_v128_load64_splat(w3);
        v128_t v23 = wasm_i64x2_shuffle(v2, v3, 0, 3);
        v23 = wasm_v128_and(v23, vmask);
        const v128_t v4 = wasm_v128_load64_splat(w4);
        const v128_t v5 = wasm_v128_load64_splat(w5);
        v128_t v45 = wasm_i64x2_shuffle(v4, v5, 0, 3);
        v45 = wasm_v128_and(v45, vmask);
        const v128_t v6 = wasm_v128_load64_splat(w6);
        const v128_t v7 = wasm_v128_load64_splat(w7);
        v128_t v67 = wasm_i64x2_shuffle(v6, v7, 0, 3);
        v67 = wasm_v128_and(v67, vmask);

        vacc01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v01, vone, vacc01);
        vacc23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v23, vone, vacc23);
        vacc45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v45, vone, vacc45);
        vacc67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v67, vone, vacc67);

        wasm_v128_store(out + 0, v01);
        wasm_v128_store(out + 16, v23);
        wasm_v128_store(out + 32, v45);
        wasm_v128_store(out + 48, v67);

        w0 += k;
        w1 += k;
        w2 += k;
        w3 += k;
        w4 += k;
        w5 += k;
        w6 += k;
        w7 += k;
        out += 64;
      }

      v128_t vksum0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc01, vacc23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc01, vacc23, 1, 3, 5, 7));
      v128_t vksum4567 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc45, vacc67, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc45, vacc67, 1, 3, 5, 7));

      vksum0123 = wasm_i32x4_mul(vksum0123, vzeropoint);
      vksum4567 = wasm_i32x4_mul(vksum4567, vzeropoint);

      v128_t vpack0123 = wasm_v128_load(packed_b);
      v128_t vpack4567 = wasm_v128_load(packed_b + 4);

      wasm_v128_store(packed_b, wasm_i32x4_sub(vpack0123, vksum0123));
      wasm_v128_store(packed_b + 4, wasm_i32x4_sub(vpack4567, vksum4567));

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 7);

      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((uint32_t*) out) = *b++;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((uint32_t*) out) = 0;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      }
      out += (8 - n) * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }

      v128_t vacc01 = wasm_i32x4_splat(0);
      v128_t vacc23 = wasm_i32x4_splat(0);
      v128_t vacc45 = wasm_i32x4_splat(0);
      v128_t vacc67 = wasm_i32x4_splat(0);

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const v128_t v0 = wasm_v128_load64_splat(w0);
        const v128_t v1 = wasm_v128_load64_splat(w1);
        const v128_t v01 = wasm_i64x2_shuffle(v0, v1, 0, 3);
        const v128_t v2 = wasm_v128_load64_splat(w2);
        const v128_t v3 = wasm_v128_load64_splat(w3);
        const v128_t v23 = wasm_i64x2_shuffle(v2, v3, 0, 3);
        const v128_t v4 = wasm_v128_load64_splat(w4);
        const v128_t v5 = wasm_v128_load64_splat(w5);
        const v128_t v45 = wasm_i64x2_shuffle(v4, v5, 0, 3);
        const v128_t v6 = wasm_v128_load64_splat(w6);
        const v128_t v7 = wasm_v128_load64_splat(w7);
        const v128_t v67 = wasm_i64x2_shuffle(v6, v7, 0, 3);

        vacc01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v01, vone, vacc01);
        vacc23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v23, vone, vacc23);
        vacc45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v45, vone, vacc45);
        vacc67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v67, vone, vacc67);

        wasm_v128_store(out + 0, v01);
        wasm_v128_store(out + 16, v23);
        wasm_v128_store(out + 32, v45);
        wasm_v128_store(out + 48, v67);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);

        const v128_t vmask = wasm_u64x2_shr(wasm_i32x4_splat(-1), (8 - k) * sizeof(int8_t) * 8);

        const v128_t v0 = wasm_v128_load64_splat(w0);
        const v128_t v1 = wasm_v128_load64_splat(w1);
        v128_t v01 = wasm_i64x2_shuffle(v0, v1, 0, 3);
        v01 = wasm_v128_and(v01, vmask);
        const v128_t v2 = wasm_v128_load64_splat(w2);
        const v128_t v3 = wasm_v128_load64_splat(w3);
        v128_t v23 = wasm_i64x2_shuffle(v2, v3, 0, 3);
        v23 = wasm_v128_and(v23, vmask);
        const v128_t v4 = wasm_v128_load64_splat(w4);
        const v128_t v5 = wasm_v128_load64_splat(w5);
        v128_t v45 = wasm_i64x2_shuffle(v4, v5, 0, 3);
        v45 = wasm_v128_and(v45, vmask);
        const v128_t v6 = wasm_v128_load64_splat(w6);
        const v128_t v7 = wasm_v128_load64_splat(w7);
        v128_t v67 = wasm_i64x2_shuffle(v6, v7, 0, 3);
        v67 = wasm_v128_and(v67, vmask);

        vacc01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v01, vone, vacc01);
        vacc23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v23, vone, vacc23);
        vacc45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v45, vone, vacc45);
        vacc67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(v67, vone, vacc67);

        wasm_v128_store(out + 0, v01);
        wasm_v128_store(out + 16, v23);
        wasm_v128_store(out + 32, v45);
        wasm_v128_store(out + 48, v67);

        out += 64;
      }

      v128_t vksum0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc01, vacc23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc01, vacc23, 1, 3, 5, 7));
      v128_t vksum4567 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc45, vacc67, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc45, vacc67, 1, 3, 5, 7));

      vksum0123 = wasm_i32x4_mul(vksum0123, vzeropoint);
      vksum4567 = wasm_i32x4_mul(vksum4567, vzeropoint);

      v128_t vpack0123 = wasm_v128_load(packed_b);
      v128_t vpack4567 = wasm_v128_load(packed_b + 4);

      wasm_v128_store(packed_b, wasm_i32x4_sub(vpack0123, vksum0123));
      wasm_v128_store(packed_b + 4, wasm_i32x4_sub(vpack4567, vksum4567));

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
