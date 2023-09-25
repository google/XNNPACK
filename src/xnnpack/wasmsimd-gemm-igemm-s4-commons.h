// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>

#include <xnnpack/wasmsimd-gemm-igemm-commons.h>

#include <wasm_simd128.h>

namespace xnnpack {
namespace internal {

class GemmIGemmS4Commons : public GemmIGemmCommons {
 public:
  using GemmIGemmCommons::GemmIGemmCommons;
  void InnerLoop(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& kc, size_t max_mr,
                 size_t k_per_iteration, size_t k_const, bool full_unroll) {
    if (full_unroll) {
      InnerLoopFullUnroll(as, vacc0123, vacc4567, w, kc, max_mr, k_const);
    } else {
      InnerLoopPartialUnroll(as, vacc0123, vacc4567, w, kc, max_mr, k_per_iteration, k_const);
    }
  }

 private:
  static size_t IterationsInMainLoop(size_t k_const) { return (k_const / 4) * 4; }

  void InnerLoopFullUnroll(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& kc,
                           size_t max_mr, size_t k_const) {
    Local k = MakeLocal(kc);
    if (k_const >= 4) {
      InnerLoopFullUnrollNoRemainder(as, vacc0123, vacc4567, w, k, max_mr, IterationsInMainLoop(k_const));
    }
    Remainder(as, vacc0123, vacc4567, k, w, k_const, max_mr);
  }

  void InnerLoopPartialUnroll(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& kc,
                              size_t max_mr, size_t k_per_iteration, size_t k_const) {
    Local k = MakeLocal(kc);
    InnerLoopPartialUnrollNoRemainder(as, vacc0123, vacc4567, w, k, max_mr, k_per_iteration);
    InnerLoopFullUnrollNoRemainder(as, vacc0123, vacc4567, w, k, max_mr,
                                   IterationsInMainLoop(k_const) % k_per_iteration);

    Remainder(as, vacc0123, vacc4567, k, w, k_const, max_mr);
  }

  void InnerLoopPartialUnrollNoRemainder(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w,
                                         Local& k, size_t max_mr, size_t k_per_iteration) {
    While([&] { I32GeU(k, I32Const(k_per_iteration * sizeof(float))); },
          [&] {
            for (size_t iter = 0; iter < k_per_iteration / 4; iter++) {
              auto vas = MakeVas(as, max_mr, /*k=*/nullptr);
              LoopComputation(vas, vacc0123, vacc4567, w, max_mr, /*is_remainder=*/false, iter * 32 * sizeof(float));
            }

            w = I32Add(w, I32Const(k_per_iteration * 8 * sizeof(float)));
            k = I32Sub(k, I32Const(k_per_iteration * sizeof(float)));
          });
  }

  void InnerLoopFullUnrollNoRemainder(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& k,
                                      size_t max_mr, size_t k_const) {
    if (k_const / 4 == 0) return;
    for (size_t iter = 0; iter < k_const / 4; iter++) {
      auto vas = MakeVas(as, max_mr, /*k=*/nullptr);
      LoopComputation(vas, vacc0123, vacc4567, w, max_mr, /*is_remainder=*/false, iter * 32 * sizeof(float));
    }
    w = I32Add(w, I32Const(k_const * 8 * sizeof(float)));
    k = I32Sub(k, I32Const(k_const * sizeof(float)));
  }

  void Remainder(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& k, Local& w, size_t k_const,
                 size_t max_mr) {
    if (k_const % 4 != 0) {
      auto vas = MakeVas(as, max_mr, &k);
      LoopComputation(vas, vacc0123, vacc4567, w, max_mr, /*is_remainder=*/true, 0);
      w = I32Add(w, I32Const(32 * sizeof(float)));
    }
  }

  LocalsArray MakeVas(LocalsArray& as, size_t max_mr, const Local* k) {
    auto vas = MakeLocalsArray(max_mr, v128);
    for (size_t i = 0; i < max_mr; i++) {
      vas[i] = V128Load(as[i]);
      as[i] = I32Add(as[i], (k == nullptr) ? I32Const(4 * sizeof(float)) : *k);
    }
    return vas;
  }

  void MulAddCond(LocalsArray& vaccs, const LocalsArray& vas, const Local& vb, const Local& vzero, size_t max_mr) {
    for (size_t i = 0; i < max_mr; i++) {
      vaccs[i] = MultiplyAndAdd(V128Andnot(vas[i], F32x4Eq(vb, vzero)), vb, vaccs[i]);
    }
  }

  void ShuffleVas(LocalsArray& vas, size_t max_mr) {
    for (size_t i = 0; i < max_mr; i++) {
      vas[i] = I32x4Shuffle(vas[i], vas[i], {1, 2, 3, 0});
    }
  }

  void LoopComputation(LocalsArray& vas, LocalsArray& vacc0123, LocalsArray& vacc4567, const Local& w, size_t max_mr,
                       bool is_remainder, uint32_t w_offset) {
    auto vb0123 = MakeLocal(v128);
    auto vb4567 = MakeLocal(v128);
    Local vzero;
    if (is_remainder) {
      vzero = MakeLocal(F32x4Splat(F32Const(0)));
    }
    for (int c = 0; c < 4; c++) {
      LoadVbs(vb0123, vb4567, w, w_offset, c);
      if (is_remainder) {
        MulAddCond(vacc0123, vas, vb0123, vzero, max_mr);
        MulAddCond(vacc4567, vas, vb4567, vzero, max_mr);
      } else {
        MulAdd(vacc0123, vas, vb0123, max_mr);
        MulAdd(vacc4567, vas, vb4567, max_mr);
      }
      if (c < 3) {
        ShuffleVas(vas, max_mr);
      }
    }
  }
};
}  // namespace internal
}  // namespace xnnpack
