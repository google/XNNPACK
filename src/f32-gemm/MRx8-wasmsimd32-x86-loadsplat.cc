#include <xnnpack/wasmsimd-gemm-igemm-loadsplat-commons.h>

#include <xnnpack/common.h>

namespace xnnpack {
namespace {
class F32GemmLoadsplatGenerator : public internal::GemmIGemmLoadsplatCommons {
 public:
  using GemmIGemmLoadsplatCommons::GemmIGemmLoadsplatCommons;

  void generate(const char* name, size_t max_mr, size_t iters, size_t loop_unroll_iters, size_t full_unroll,
                size_t nc_mod_nr, bool use_fma, const jit_gemm_params* jit_gemm_params) {
    assert(!use_fma || XNN_ARCH_WASMRELAXEDSIMD);
    use_fma_ = use_fma;
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

                      InnerLoop(as, vacc0123, vacc4567, w, kc, max_mr, loop_unroll_iters, iters, full_unroll);

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
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 1, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2(xnn_code_buffer* b, size_t max_mr,
                                                                            size_t nc_mod_nr, size_t kc,
                                                                            const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 2, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4(xnn_code_buffer* b, size_t max_mr,
                                                                            size_t nc_mod_nr, size_t kc,
                                                                            const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 4, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8(xnn_code_buffer* b, size_t max_mr,
                                                                            size_t nc_mod_nr, size_t kc,
                                                                            const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 8, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                              size_t nc_mod_nr, size_t kc,
                                                                              const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, kc / sizeof(float), /*full_unroll=*/true, nc_mod_nr,
                           /*use_fma=*/false, params);
}

#if XNN_ARCH_WASMRELAXEDSIMD
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1(xnn_code_buffer* b,
                                                                                       size_t max_mr, size_t nc_mod_nr,
                                                                                       size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_fma_loadsplat_x1";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 1, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2(xnn_code_buffer* b,
                                                                                       size_t max_mr, size_t nc_mod_nr,
                                                                                       size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_fma_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 2, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4(xnn_code_buffer* b,
                                                                                       size_t max_mr, size_t nc_mod_nr,
                                                                                       size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_fma_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 4, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8(xnn_code_buffer* b,
                                                                                       size_t max_mr, size_t nc_mod_nr,
                                                                                       size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_fma_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 8, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf(xnn_code_buffer* b,
                                                                                         size_t max_mr,
                                                                                         size_t nc_mod_nr, size_t kc,
                                                                                         const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_fma_loadsplat_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, kc / sizeof(float), /*full_unroll=*/true, nc_mod_nr,
                           /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1(xnn_code_buffer* b, size_t max_mr,
                                                                                   size_t nc_mod_nr, size_t kc,
                                                                                   const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 1, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2(xnn_code_buffer* b, size_t max_mr,
                                                                                   size_t nc_mod_nr, size_t kc,
                                                                                   const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 2, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4(xnn_code_buffer* b, size_t max_mr,
                                                                                   size_t nc_mod_nr, size_t kc,
                                                                                   const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 4, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8(xnn_code_buffer* b, size_t max_mr,
                                                                                   size_t nc_mod_nr, size_t kc,
                                                                                   const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_x8";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, 8, /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false,
                           params);
}
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                                     size_t nc_mod_nr, size_t kc,
                                                                                     const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, kc / sizeof(float), /*full_unroll=*/true, nc_mod_nr,
                           /*use_fma=*/false, params);
}
#endif
}
