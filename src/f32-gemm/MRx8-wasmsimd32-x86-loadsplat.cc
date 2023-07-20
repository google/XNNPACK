#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>

#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/wasm-assembler.h>
#include <xnnpack/wasmsimd-gemm-igemm-loadsplat-commons.h>


namespace xnnpack {
namespace {
class F32GemmLoadsplatGenerator : public internal::GemmIGemmLoadsplatCommons {
 public:
  using GemmIGemmLoadsplatCommons::GemmIGemmLoadsplatCommons;

  void generate(const char* name, size_t max_mr, size_t loop_unroll_iters, const jit_gemm_params* jit_gemm_params) {
    ValTypesToInt locals_declaration = {{i32, max_mr * 2 + 1}, {v128, max_mr * 3 + 8}};
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

                      InnerLoop(as, vacc0123, vacc4567, w, kc, max_mr, loop_unroll_iters);

                      ApplyPostOps(vacc0123);
                      ApplyPostOps(vacc4567);

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

 private:
  void InitAccumulators(LocalsArray& vaccs, const Local& w, size_t offset) {
    vaccs[0] = V128Load(w, offset);
    std::for_each(std::next(std::begin(vaccs)), std::end(vaccs), [&](auto& vacc) { vacc = vaccs[0]; });
  }

  void ClampAsAndCs(LocalsArray& as, LocalsArray& cs, const Local& mr, const Local& a, const Local& c,
                    const Local& a_stride, const Local& cm_stride) {
    as[0] = a;
    cs[0] = c;
    auto i_local = MakeLocal(I32Const(1));
    for (size_t i = 1; i < as.size(); i++) {
      as[i] = I32Add(as[i - 1], a_stride);
      cs[i] = I32Add(cs[i - 1], cm_stride);
      If([&] { I32GeU(i_local, mr); },
         [&] {
           as[i] = as[i - 1];
           cs[i] = cs[i - 1];
         });
      i_local = I32Add(i_local, I32Const(1));
    }
  }
};

xnn_status generate(xnn_code_buffer* b, const char* name, size_t max_mr, size_t loop_unroll_iters, const void* params) {
  xnnpack::F32GemmLoadsplatGenerator generator(b);

  generator.generate(name, max_mr, loop_unroll_iters, static_cast<const jit_gemm_params*>(params));
  generator.Emit();
  auto finalized = generator.finalize();
  if (finalized == nullptr || generator.error() != xnnpack::Error::kNoError) {
    return xnn_status_uninitialized;
  }
  return xnn_status_success;
}

}  // namespace
}  // namespace xnnpack

extern "C" {
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1(xnn_code_buffer* b, size_t max_mr,
                                                                          size_t nc_mod_nr, size_t kc,
                                                                          const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, 1, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2(xnn_code_buffer* b, size_t max_mr,
                                                                          size_t nc_mod_nr, size_t kc,
                                                                          const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, 2, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4(xnn_code_buffer* b, size_t max_mr,
                                                                          size_t nc_mod_nr, size_t kc,
                                                                          const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, 4, params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8(xnn_code_buffer* b, size_t max_mr,
                                                                          size_t nc_mod_nr, size_t kc,
                                                                          const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, 8, params);
}
}
