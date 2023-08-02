#include <cstddef>

#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/wasm-assembler.h>
#include <xnnpack/wasmsimd-gemm-igemm-loadsplat-commons.h>


namespace xnnpack {
namespace {
class F32IGemmLoadsplatGenerator : public internal::GemmIGemmLoadsplatCommons {
 public:
  using GemmIGemmLoadsplatCommons::GemmIGemmLoadsplatCommons;

  void generate(const char* name, size_t max_mr, size_t iters, size_t loop_unroll_iters, size_t full_unroll,
                const jit_gemm_params* jit_gemm_params) {
    ValTypesToInt locals_declaration = {{i32, max_mr * 2 + 5}, {v128, max_mr * 3 + 8}};
    AddFunc<12>({}, name, locals_declaration,
                [&](Local mr, Local nc, Local kc, const Local ks, Ptr<float**> a, Ptr<v128_t> w, Ptr<float> c,
                    Local cm_stride, Local cn_stride, Local a_offset, Ptr<float> zero, Local params) {
                  InitPostOps(jit_gemm_params, params);

                  PtrsArray<float> cs = MakePtrsArray<float>(max_mr, i32);
                  ClampCs(cs, mr, c, cm_stride);

                  LocalsArray vacc0123 = MakeLocalsArray(max_mr, v128);
                  LocalsArray vacc4567 = MakeLocalsArray(max_mr, v128);

                  DoWhile(
                    [&] {
                      InitAccumulators(vacc0123, w, /*offset=*/0);
                      InitAccumulators(vacc4567, w, /*offset=*/sizeof(v128_t));

                      w.Advance(2);

                      Local p = MakeLocal(ks);

                      DoWhile(
                        [&] {
                          PtrsArray<float> as = MakePtrsArray<float>(max_mr, i32);
                          for (size_t i = 0; i < max_mr; i++) {
                            as[i] = I32Load(a, /*offset=*/i * sizeof(void*));
                            as[i] = Select(I32Add(as[i], a_offset), as[i], I32Ne(as[i], zero));
                          }

                          InnerLoop(as, vacc0123, vacc4567, w, kc, max_mr, loop_unroll_iters, iters, full_unroll);

                          a.Advance(max_mr);
                          p = I32Sub(p, I32Const(max_mr * sizeof(void*)));
                        },
                        [&] { I32NeZ(p); });

                      ApplyPostOps(vacc0123);
                      ApplyPostOps(vacc4567);

                      IfElse([&] { I32GeU(nc, I32Const(8)); },
                             [&] {
                               for (int i = max_mr - 1; i >= 0; i--) {
                                 V128Store(cs[i], vacc0123[i]);
                                 V128Store(cs[i], vacc4567[i], /*offset=*/sizeof(v128_t));
                                 cs[i].AdvanceBytes(cn_stride);
                               }
                               a.BackBytes(ks);
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
                    [&] { I32NeZ(nc); });
                });
  }

 private:
  void ClampCs(PtrsArray<float>& cs, const Local& mr, const Ptr<float>& c, const Local& cm_stride) {
    cs[0] = c;
    for (size_t i = 1; i < cs.size(); i++) {
      cs[i] = Select(cs[i - 1], I32Add(cs[i - 1], cm_stride), I32GeU(I32Const(i), mr));
    }
  }
};

constexpr auto generate = internal::generate_gemm_or_igemm<xnnpack::F32IGemmLoadsplatGenerator>;

}  // namespace
}  // namespace xnnpack

extern "C" {
xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1(xnn_code_buffer* b, size_t max_mr,
                                                                             size_t nc_mod_nr, size_t kc, size_t ks,
                                                                             const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x1";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 1, /*full_unroll=*/false, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2(xnn_code_buffer* b, size_t max_mr,
                                                                             size_t nc_mod_nr, size_t kc, size_t ks,
                                                                             const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 2, /*full_unroll=*/false, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4(xnn_code_buffer* b, size_t max_mr,
                                                                             size_t nc_mod_nr, size_t kc, size_t ks,
                                                                             const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 4, /*full_unroll=*/false, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8(xnn_code_buffer* b, size_t max_mr,
                                                                             size_t nc_mod_nr, size_t kc, size_t ks,
                                                                             const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 8, /*full_unroll=*/false, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                               size_t nc_mod_nr, size_t kc, size_t ks,
                                                                               const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, kc / sizeof(float), /*full_unroll=*/true, params);
}
}
