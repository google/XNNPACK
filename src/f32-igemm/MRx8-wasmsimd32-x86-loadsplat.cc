#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>

#include <xnnpack/assembler.h>
#include <xnnpack/common.h>
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
                size_t nc_mod_nr, bool use_fma, const jit_gemm_params* jit_gemm_params) {
    assert(!use_fma || XNN_ARCH_WASMRELAXEDSIMD);
    use_fma_ = use_fma;
    ValTypesToInt locals_declaration = {{i32, max_mr * 2 + 5}, {v128, max_mr * 3 + 8}};
    AddFunc<12>({}, name, locals_declaration,
                [&](auto mr, auto nc, auto kc, const auto ks, auto a, auto w, auto c, auto cm_stride, auto cn_stride,
                    auto a_offset, auto zero, auto params) {
                  InitPostOps(jit_gemm_params, params);

                  LocalsArray cs = MakeLocalsArray(max_mr, i32);
                  ClampCs(cs, mr, c, cm_stride);

                  LocalsArray vacc0123 = MakeLocalsArray(max_mr, v128);
                  LocalsArray vacc4567 = MakeLocalsArray(max_mr, v128);

                  DoWhile(
                    [&] {
                      InitAccumulators(vacc0123, w, /*offset=*/0);
                      InitAccumulators(vacc4567, w, /*offset=*/sizeof(v128_t));

                      w = I32Add(w, I32Const(8 * sizeof(float)));

                      Local p = MakeLocal(ks);

                      DoWhile(
                        [&] {
                          LocalsArray as = MakeLocalsArray(max_mr, i32);
                          for (size_t i = 0; i < max_mr; i++) {
                            as[i] = I32Load(a, /*offset=*/i * sizeof(void*));
                            as[i] = Select(I32Add(as[i], a_offset), as[i], I32Ne(as[i], zero));
                          }

                          InnerLoop(as, vacc0123, vacc4567, w, kc, max_mr, loop_unroll_iters, iters, full_unroll);

                          a = I32Add(a, I32Const(max_mr * sizeof(void*)));
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
                                 cs[i] = I32Add(cs[i], cn_stride);
                               }
                               a = I32Sub(a, ks);
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
                    [&] { I32NeZ(nc); });
                });
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
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 1, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2(xnn_code_buffer* b, size_t max_mr,
                                                                             size_t nc_mod_nr, size_t kc, size_t ks,
                                                                             const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 2, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4(xnn_code_buffer* b, size_t max_mr,
                                                                             size_t nc_mod_nr, size_t kc, size_t ks,
                                                                             const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 4, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8(xnn_code_buffer* b, size_t max_mr,
                                                                             size_t nc_mod_nr, size_t kc, size_t ks,
                                                                             const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 8, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                               size_t nc_mod_nr, size_t kc, size_t ks,
                                                                               const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, kc / sizeof(float), /*full_unroll=*/true, nc_mod_nr,
                           /*use_fma=*/false, params);
}

#if XNN_ARCH_WASMRELAXEDSIMD
xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1(xnn_code_buffer* b,
                                                                                        size_t max_mr, size_t nc_mod_nr,
                                                                                        size_t kc, size_t ks,
                                                                                        const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x1";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 1, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2(xnn_code_buffer* b,
                                                                                        size_t max_mr, size_t nc_mod_nr,
                                                                                        size_t kc, size_t ks,
                                                                                        const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 2, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4(xnn_code_buffer* b,
                                                                                        size_t max_mr, size_t nc_mod_nr,
                                                                                        size_t kc, size_t ks,
                                                                                        const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 4, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8(xnn_code_buffer* b,
                                                                                        size_t max_mr, size_t nc_mod_nr,
                                                                                        size_t kc, size_t ks,
                                                                                        const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 8, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf(
  xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, kc / sizeof(float), /*full_unroll=*/true, nc_mod_nr,
                           /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1(xnn_code_buffer* b, size_t max_mr,
                                                                                    size_t nc_mod_nr, size_t kc,
                                                                                    size_t ks, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x1";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 1, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2(xnn_code_buffer* b, size_t max_mr,
                                                                                    size_t nc_mod_nr, size_t kc,
                                                                                    size_t ks, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 2, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4(xnn_code_buffer* b, size_t max_mr,
                                                                                    size_t nc_mod_nr, size_t kc,
                                                                                    size_t ks, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 4, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8(xnn_code_buffer* b, size_t max_mr,
                                                                                    size_t nc_mod_nr, size_t kc,
                                                                                    size_t ks, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 8, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                                      size_t nc_mod_nr, size_t kc,
                                                                                      size_t ks, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_igemm_ukernel_6x8__wasmsimd_x86_loadsplat_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, kc / sizeof(float), /*full_unroll=*/true, nc_mod_nr,
                           /*use_fma=*/false, params);
}

#endif
}
