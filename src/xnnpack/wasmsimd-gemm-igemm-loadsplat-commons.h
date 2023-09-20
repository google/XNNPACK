// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <wasm_simd128.h>

#include <xnnpack/log.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/wasm-assembler.h>
#include <xnnpack/wasmsimd-gemm-igemm-commons.h>

namespace xnnpack {
namespace internal {

class GemmIGemmLoadsplatCommons : public GemmIGemmCommons {
 public:
  using GemmIGemmCommons::GemmIGemmCommons;

  void InnerLoopPartialUnroll(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& kc,
                              size_t max_mr, size_t loop_unroll_iters, bool use_fma) {
    Local k = MakeLocal(kc);
    InnerLoopMainPart(as, vacc0123, vacc4567, w, k, max_mr, loop_unroll_iters, use_fma);

    const size_t max_iters_left = loop_unroll_iters - 1;
    size_t mask = max_iters_left > 0 ? (1 << static_cast<size_t>(log2(max_iters_left))) : 0;

    if (max_iters_left > 0) {
      If([&] { I32NeZ(k); },
         [&] {
           while (mask > 0) {
             If([&] { I32GeU(k, I32Const(mask * sizeof(float))); },
                [&] { InnerLoopBody(as, vacc0123, vacc4567, w, k, max_mr, mask, use_fma); });
             mask >>= 1;
           }
         });
    }
  }

  void InnerLoopFullUnroll(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& kc,
                           size_t max_mr, size_t iters, bool use_fma) {
    Local k = MakeLocal(kc);
    InnerLoopMainPart(as, vacc0123, vacc4567, w, k, max_mr, iters, use_fma);
  }

  void InnerLoop(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& kc, size_t max_mr,
                 size_t loop_unroll_iters, size_t iters, bool full_unroll, bool use_fma) {
    if (full_unroll) {
      InnerLoopFullUnroll(as, vacc0123, vacc4567, w, kc, max_mr, iters, use_fma);
    } else {
      InnerLoopPartialUnroll(as, vacc0123, vacc4567, w, kc, max_mr, loop_unroll_iters, use_fma);
    }
  }

 private:
  void InnerLoopBody(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& k, size_t max_mr,
                     size_t loop_unroll_iters, bool use_fma) {
    for (size_t unrolled_iter = 0; unrolled_iter < loop_unroll_iters; unrolled_iter++) {
      auto vb0123 = MakeLocal(v128);
      auto vb4567 = MakeLocal(v128);
      LoadVbs(vb0123, vb4567, w, /*offset=*/(2 * unrolled_iter) * sizeof(v128_t));
      for (size_t i = 0; i < max_mr; i++) {
        const auto va = MakeLocal(V128Load32Splat(as[i]));
        vacc0123[i] = MultiplyAndAdd(va, vb0123, vacc0123[i], use_fma);
        vacc4567[i] = MultiplyAndAdd(va, vb4567, vacc4567[i], use_fma);
        as[i] = I32Add(as[i], I32Const(sizeof(float)));
      }
    }
    w = I32Add(w, I32Const(8 * loop_unroll_iters * sizeof(float)));
    k = I32Sub(k, I32Const(loop_unroll_iters * sizeof(float)));
  }

  ValueOnStack MultiplyAndAdd(const Local& a, const Local& b, const Local& c, bool use_fma) {
  #if XNN_ARCH_WASMRELAXEDSIMD
    if (use_fma) {
      return F32x4RelaxedMadd(a, b, c);
    } else
  #endif
    {
      return F32x4Add(c, F32x4Mul(a, b));
    }
  }

  void InnerLoopMainPart(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& k,
                         size_t max_mr, size_t loop_unroll_iters, bool use_fma) {
    const auto body = [&] { InnerLoopBody(as, vacc0123, vacc4567, w, k, max_mr, loop_unroll_iters, use_fma); };
    if (loop_unroll_iters == 1) {
      DoWhile(body, [&] { I32NeZ(k); });
    } else {
      While([&] { I32GeU(k, I32Const(loop_unroll_iters * sizeof(float))); }, body);
    }
  }
};
}  // namespace internal
}  // namespace xnnpack
