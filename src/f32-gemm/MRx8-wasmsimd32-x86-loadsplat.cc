#include <cstddef>

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

  void generate(const char* name, size_t max_mr, size_t iters, size_t loop_unroll_iters, size_t full_unroll,
                const jit_gemm_params* jit_gemm_params) {
    ValTypesToInt locals_declaration = {{i32, max_mr * 2 + 2}, {v128, max_mr * 3 + 8}};
    AddFunc<10>({}, name, locals_declaration,
                [&](Local mr, Local nc, Local kc, Ptr<float> a, Local a_stride, Ptr<v128_t> w, Ptr<float> c,
                    Local cm_stride, Local cn_stride, Local params) {
                  InitPostOps(jit_gemm_params, params);

                  PtrsArray<float> as = MakePtrsArray<float>(max_mr, i32);
                  PtrsArray<float> cs = MakePtrsArray<float>(max_mr, i32);
                  ClampAsAndCs(as, cs, mr, a, c, a_stride, cm_stride);

                  LocalsArray vacc0123 = MakeLocalsArray(max_mr, v128);
                  LocalsArray vacc4567 = MakeLocalsArray(max_mr, v128);

                  DoWhile(
                    [&] {
                      InitAccumulators(vacc0123, w, /*offset=*/0);
                      InitAccumulators(vacc4567, w, /*offset=*/sizeof(v128_t));

                      w.Advance(2);

                      InnerLoop(as, vacc0123, vacc4567, w, kc, max_mr, loop_unroll_iters, iters, full_unroll);

                      ApplyPostOps(vacc0123);
                      ApplyPostOps(vacc4567);

                      IfElse([&] { I32GeU(nc, I32Const(8)); },
                             [&] {
                               for (int i = max_mr - 1; i >= 0; i--) {
                                 V128Store(cs[i], vacc0123[i]);
                                 V128Store(cs[i], vacc4567[i], /*offset=*/sizeof(v128_t));
                                 cs[i].AdvanceBytes(cn_stride);
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
                                      cs[i].Advance(4);
                                    }
                                  });
                               If([&] { I32And(nc, I32Const(2)); },
                                  [&] {
                                    for (int i = max_mr - 1; i >= 0; i--) {
                                      V128Store64Lane(cs[i], vacc0123[i], 0);
                                      vacc0123[i] = I64x2Shuffle(vacc0123[i], vacc0123[i], {1, 1});
                                      cs[i].Advance(2);
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
  void ClampAsAndCs(PtrsArray<float>& as, PtrsArray<float>& cs, const Local& mr, const Ptr<float>& a,
                    const Ptr<float>& c, const Local& a_stride, const Local& cm_stride) {
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

constexpr auto generate = internal::generate_gemm_or_igemm<xnnpack::F32GemmLoadsplatGenerator>;

}  // namespace
}  // namespace xnnpack

extern "C" {
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1(xnn_code_buffer* b, size_t max_mr,
                                                                            size_t nc_mod_nr, size_t kc,
                                                                            const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 1, /*full_unroll=*/false, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2(xnn_code_buffer* b, size_t max_mr,
                                                                            size_t nc_mod_nr, size_t kc,
                                                                            const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 2, /*full_unroll=*/false, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4(xnn_code_buffer* b, size_t max_mr,
                                                                            size_t nc_mod_nr, size_t kc,
                                                                            const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 4, /*full_unroll=*/false, params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8(xnn_code_buffer* b, size_t max_mr,
                                                                            size_t nc_mod_nr, size_t kc,
                                                                            const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 8, /*full_unroll=*/false, params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                              size_t nc_mod_nr, size_t kc,
                                                                              const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, kc / sizeof(float), /*full_unroll=*/true, params);
}
}
