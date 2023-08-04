// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <cstddef>

#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/wasm-assembler.h>
#include <xnnpack/wasmsimd-gemm-igemm-s4-commons.h>

namespace xnnpack {
namespace {
class F32GemmS4Generator : public internal::GemmIGemmS4Commons {
 public:
  using GemmIGemmS4Commons::GemmIGemmS4Commons;

  void generate(const char* name, size_t max_mr, size_t iters, size_t loop_unroll_iters, size_t full_unroll,
                const jit_gemm_params* jit_gemm_params) {
    ValTypesToInt locals_declaration = {{i32, max_mr * 2 + 2}, {v128, max_mr * 3 + 8}};
    AddFunc<10>({}, name, locals_declaration,
                [&](auto mr, auto nc, auto kc, auto a, auto a_stride, auto w, auto c, auto cm_stride, auto cn_stride,
                    auto params) {
                  InitPostOps(jit_gemm_params, params);

                  LocalsArray as = MakeLocalsArray(max_mr, i32);
                  LocalsArray cs = MakeLocalsArray(max_mr, i32);
                  ClampAsAndCs(as, cs, mr, a, c, a_stride, cm_stride);

                  LocalsArray vacc0123 = MakeLocalsArray(max_mr, v128);
                  LocalsArray vacc4567 = MakeLocalsArray(max_mr, v128);

                  DoWhile(
                    [&] {
                      InitAccumulators(vacc0123, w, /*offset=*/0);
                      InitAccumulators(vacc4567, w, /*offset=*/sizeof(v128_t));

                      w = I32Add(w, I32Const(8 * sizeof(float)));

                      InnerLoop(as, vacc0123, vacc4567, w, kc, max_mr, 1, iters, false);

                      ApplyPostOps(vacc0123);
                      ApplyPostOps(vacc4567);

                      // TODO(b/294356273)
                      IfElse([&] { I32GeU(nc, I32Const(8)); },
                             [&] {
                               for (int i = max_mr - 1; i >= 0; i--) {
                                 V128Store(cs[i], vacc0123[i]);
                                 V128Store(cs[i], vacc4567[i], /*offset=*/sizeof(v128_t));
                                 cs[i] = I32Add(cs[i], cn_stride);
                               }
                               for (int i = max_mr - 1; i >= 0; i--) {
                                 as[i] = I32Sub(as[i], kc);
                               }

                               nc = I32Sub(nc, I32Const(8));
                             },
                             [&] {
                               If([&] { I32And(nc, I32Const(4)); },
                                  [&] {
                                    for (int i = max_mr - 1; i >= 0; i--) {
                                      V128Store(cs[i], vacc0123[i]);
                                      vacc0123[i] = vacc4567[i];
                                      cs[i] = I32Add(cs[i], I32Const(sizeof(v128_t)));
                                    }
                                  });
                               If([&] { I32And(nc, I32Const(2)); },
                                  [&] {
                                    for (int i = max_mr - 1; i >= 0; i--) {
                                      V128Store64Lane(cs[i], vacc0123[i], 0);
                                      vacc0123[i] = I64x2Shuffle(vacc0123[i], vacc0123[i], {1, 1});
                                      cs[i] = I32Add(cs[i], I32Const(2 * sizeof(float)));
                                    }
                                  });
                               If([&] { I32And(nc, I32Const(1)); },
                                  [&] {
                                    for (int i = max_mr - 1; i >= 0; i--) {
                                      V128Store32Lane(cs[i], vacc0123[i], 0);
                                    }
                                  });
                               Return();
                             });
                    },
                    [&] { I32Ne(nc, I32Const(0)); });
                });
  }
};

constexpr auto generate = internal::generate_gemm_or_igemm<F32GemmS4Generator>;

}  // namespace
}  // namespace xnnpack

extern "C" {
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd_x86_x1";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*loop_unroll_iters=*/1, /*full_unroll=*/false, params);
}
}
