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

  void generate(const char* name, size_t max_mr, size_t k_const, size_t k_per_iteration, bool full_unroll,
                size_t nc_mod_nr, bool use_fma, const jit_gemm_params* jit_gemm_params) {
    assert(!use_fma || XNN_ARCH_WASMRELAXEDSIMD);
    use_fma_ = use_fma;
    ValTypesToInt locals_declaration = {{i32, max_mr * 2 + 2}, {v128, max_mr * 3 + 8}};
    AddFunc<10>({}, name, locals_declaration,
                [&](auto mr, auto nc, auto kc, auto a, auto a_stride, auto w, auto c, auto cm_stride, auto cn_stride,
                    auto params) {
                  const bool is_nc_mod_nr_known = nc_mod_nr != SIZE_MAX;
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

                      InnerLoop(as, vacc0123, vacc4567, w, kc, max_mr, k_per_iteration, k_const, full_unroll);

                      ApplyPostOps(vacc0123);
                      ApplyPostOps(vacc4567);

                      // TODO(b/294356273)
                      StoreArgs store_args{&cs, &vacc0123, &vacc4567, &as, &cn_stride, &kc, &nc};
                      if (nc_mod_nr == 0) {
                        Store8Floats(store_args);
                      } else {
                        IfElse([&] { I32GeU(nc, I32Const(8)); }, [&] { Store8Floats(store_args); },
                               [&] {
                                 if (is_nc_mod_nr_known) {
                                   if (nc_mod_nr & 4) {
                                     Store4Floats(store_args);
                                   }
                                   if (nc_mod_nr & 2) {
                                     Store2Floats(store_args);
                                   }
                                   if (nc_mod_nr & 1) {
                                     Store1Floats(store_args);
                                   }
                                 } else {
                                   MaskedIf([&] { Store4Floats(store_args); }, nc, 4);
                                   MaskedIf([&] { Store2Floats(store_args); }, nc, 2);
                                   MaskedIf([&] { Store1Floats(store_args); }, nc, 1);
                                 }
                                 Return();
                               });
                      }
                    },
                    [&] { I32Ne(nc, I32Const(0)); });
                });
  }

 private:
  template <typename Body>
  void MaskedIf(Body&& body, const Local& nc, size_t mask) {
    If([&] { I32And(nc, I32Const(mask)); }, std::forward<Body>(body));
  }

  void Store8Floats(StoreArgs& args) {
    for (int i = args.max_mr - 1; i >= 0; i--) {
      V128Store(args.cs[i], args.vacc0123[i]);
      V128Store(args.cs[i], args.vacc4567[i], /*offset=*/sizeof(v128_t));
      args.cs[i] = I32Add(args.cs[i], args.cn_stride);
    }
    for (int i = args.max_mr - 1; i >= 0; i--) {
      args.as[i] = I32Sub(args.as[i], args.kc);
    }

    args.nc = I32Sub(args.nc, I32Const(8));
  }

  void Store4Floats(StoreArgs& args) {
    for (int i = args.max_mr - 1; i >= 0; i--) {
      V128Store(args.cs[i], args.vacc0123[i]);
      args.vacc0123[i] = args.vacc4567[i];
      args.cs[i] = I32Add(args.cs[i], I32Const(sizeof(v128_t)));
    }
  }

  void Store2Floats(StoreArgs& args) {
    for (int i = args.max_mr - 1; i >= 0; i--) {
      V128Store64Lane(args.cs[i], args.vacc0123[i], 0);
      args.vacc0123[i] = I64x2Shuffle(args.vacc0123[i], args.vacc0123[i], {1, 1});
      args.cs[i] = I32Add(args.cs[i], I32Const(2 * sizeof(float)));
    }
  }

  void Store1Floats(StoreArgs& args) {
    for (int i = args.max_mr - 1; i >= 0; i--) {
      V128Store32Lane(args.cs[i], args.vacc0123[i], 0);
    }
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
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/4,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x2(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd_x86_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/8,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x4(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd_x86_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/16,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                      size_t nc_mod_nr, size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd_x86_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc,
                           /*k_per_iteration=*/kc / sizeof(float),
                           /*full_unroll=*/true, nc_mod_nr, /*use_fma=*/false, params);
}

#if XNN_ARCH_WASMRELAXEDSIMD
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/4,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/8,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/16,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                      size_t nc_mod_nr, size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc,
                           /*k_per_iteration=*/kc / sizeof(float),
                           /*full_unroll=*/true, nc_mod_nr, /*use_fma=*/true, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/4,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/8,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4(xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr,
                                                                    size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc, /*k_per_iteration=*/16,
                           /*full_unroll=*/false, nc_mod_nr, /*use_fma=*/false, params);
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf(xnn_code_buffer* b, size_t max_mr,
                                                                      size_t nc_mod_nr, size_t kc, const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, kc,
                           /*k_per_iteration=*/kc / sizeof(float),
                           /*full_unroll=*/true, nc_mod_nr, /*use_fma=*/false, params);
}

#endif
}
