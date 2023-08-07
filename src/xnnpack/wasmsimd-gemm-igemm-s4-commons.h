// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/wasmsimd-gemm-igemm-commons.h>

#include <wasm_simd128.h>

namespace xnnpack {
namespace internal {

class GemmIGemmS4Commons : public GemmIGemmCommons {
 public:
  using GemmIGemmCommons::GemmIGemmCommons;
  void InnerLoop(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& kc, size_t max_mr,
                 size_t loop_unroll_iters, size_t iters, bool full_unroll) {
    Local k = MakeLocal(kc);
    While([&] { I32GeU(k, I32Const(4 * sizeof(float))); },
          [&] {
            auto vas = MakeVas(as, max_mr, /*k=*/nullptr);

            LoopComputation(vas, vacc0123, vacc4567, w, max_mr, /*is_remainder=*/false);

            w = I32Add(w, I32Const(32 * sizeof(float)));
            k = I32Sub(k, I32Const(4 * sizeof(float)));
          });
    if (iters % 4 != 0) {
      auto vas = MakeVas(as, max_mr, &k);

      LoopComputation(vas, vacc0123, vacc4567, w, max_mr, /*is_remainder=*/true);

      w = I32Add(w, I32Const(32 * sizeof(float)));
    }
  }

 private:
  LocalsArray MakeVas(LocalsArray& as, size_t max_mr, const Local* k) {
    auto vas = MakeLocalsArray(max_mr, v128);
    for (size_t i = 0; i < max_mr; i++) {
      vas[i] = V128Load(as[i]);
      as[i] = I32Add(as[i], (k == nullptr) ? I32Const(4 * sizeof(float)) : *k);
    }
    return vas;
  }

  void LoadVbs(Local& vb0123, Local& vb4567, const Local& w, size_t c) {
    vb0123 = V128Load(w, /*offset=*/(c * 8) * sizeof(float));
    vb4567 = V128Load(w, /*offset=*/(c * 8 + 4) * sizeof(float));
  }

  void MulAdd(LocalsArray& vaccs, const LocalsArray& vas, const Local& vb, size_t max_mr) {
    for (size_t i = 0; i < max_mr; i++) {
      vaccs[i] = F32x4Add(F32x4Mul(vas[i], vb), vaccs[i]);
    }
  }

  void MulAddCond(LocalsArray& vaccs, const LocalsArray& vas, const Local& vb, const Local& vzero, size_t max_mr) {
    for (size_t i = 0; i < max_mr; i++) {
      vaccs[i] = F32x4Add(F32x4Mul(V128Andnot(vas[i], F32x4Eq(vb, vzero)), vb), vaccs[i]);
    }
  }

  void ShuffleVas(LocalsArray& vas, size_t max_mr) {
    for (size_t i = 0; i < max_mr; i++) {
      vas[i] = I32x4Shuffle(vas[i], vas[i], {1, 2, 3, 0});
    }
  }

  void LoopComputation(LocalsArray& vas, LocalsArray& vacc0123, LocalsArray& vacc4567, const Local& w, size_t max_mr,
                       bool is_remainder) {
    auto vb0123 = MakeLocal(v128);
    auto vb4567 = MakeLocal(v128);
    Local vzero;
    if (is_remainder) {
      vzero = MakeLocal(F32x4Splat(F32Const(0)));
    }
    for (int c = 0; c < 4; c++) {
      LoadVbs(vb0123, vb4567, w, c);
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
