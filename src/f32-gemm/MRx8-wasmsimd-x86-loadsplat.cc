#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>

#include <wasm_simd128.h>

#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/wasm-assembler.h>

#include <wasm_simd128.h>


namespace xnnpack {
namespace {
class F32GemmLoadsplatGenerator : public WasmAssembler {
 public:
  explicit F32GemmLoadsplatGenerator(xnn_code_buffer* bf)
      : WasmAssembler(bf) {}

  void generate(const char* name, size_t max_mr, const jit_gemm_params* jit_gemm_params) {
    const size_t num_post_operations = jit_gemm_params->num_post_operations;
    const xnn_post_operation* post_operations = jit_gemm_params->post_operations;
    const float min = jit_gemm_params->f32_minmax.min;
    const float max = jit_gemm_params->f32_minmax.max;

    ValTypesToInt locals_declaration = {{i32, max_mr * 2 + 1}, {v128, max_mr * 3 + 8}};
    AddFunc<10>({}, name, locals_declaration,
                [&](auto mr, auto nc, auto kc, auto a, auto a_stride, auto w, auto c, auto cm_stride, auto cn_stride,
                    auto params) {
                  InitClampLimit(min, max);
                  InitPostOps(post_operations, num_post_operations, params);

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

                      auto k = MakeLocal(kc);
                      DoWhile(
                        [&] {
                          LocalsArray va = MakeLocalsArray(max_mr, v128);

                          auto vb0123 = MakeLocal(V128Load(w, /*offset=*/0));
                          auto vb4567 = MakeLocal(V128Load(w, /*offset=*/sizeof(v128_t)));

                          for (size_t i = 0; i < max_mr; i++) {
                            va[i] = V128Load32Splat(as[i]);
                            vacc0123[i] = F32x4Add(vacc0123[i], F32x4Mul(va[i], vb0123));
                            vacc4567[i] = F32x4Add(vacc4567[i], F32x4Mul(va[i], vb4567));
                            as[i] = I32Add(as[i], I32Const(sizeof(float)));
                          }

                          w = I32Add(w, I32Const(8 * sizeof(float)));
                          k = I32Sub(k, I32Const(sizeof(float)));
                        },
                        [&] { I32Ne(k, I32Const(0)); });

                      Clamp(vacc0123);
                      ApplyPostOps(post_operations, num_post_operations, vacc0123);
                      Clamp(vacc4567);
                      ApplyPostOps(post_operations, num_post_operations, vacc4567);

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
  struct HswishConsts {
    using Local = F32GemmLoadsplatGenerator::Local;
    Local vsixth;
    Local vsix;
    Local vthree;
    Local vzero;
  };

  struct ClampConsts {
    using Local = F32GemmLoadsplatGenerator::Local;
    bool clamp_min;
    bool clamp_max;
    Local vmin;
    Local vmax;
  };

  auto MakeF32x4Const(float value) { return MakeLocal(F32x4Splat(F32Const(value))); }

  auto Hswish(Local& v) {
    Local vacc = MakeLocal(F32x4Add(v, hswish_consts_.vthree));
    v = F32x4Mul(v, hswish_consts_.vsixth);
    vacc = F32x4Pmax(vacc, hswish_consts_.vzero);
    vacc = F32x4Pmin(vacc, hswish_consts_.vsix);
    v = F32x4Mul(vacc, v);
  }

  auto MakeV128Load64Splat(const Local& address, uint32_t offset) {
    return MakeLocal(V128Load64Splat(address, offset));
  }

  void InitPostOps(const xnn_post_operation* ops, size_t num_ops, Local& params) {
    for (size_t i = 0; i < num_ops; i++) {
      switch (ops[i].op_type) {
        case xnn_post_operation_type_hardswish:
          hswish_consts_.vsixth = MakeV128Load64Splat(params, /*offset=*/0);
          hswish_consts_.vthree = MakeV128Load64Splat(params, /*offset=*/2 * sizeof(float));
          hswish_consts_.vsix = MakeV128Load64Splat(params, /*offset=*/4 * sizeof(float));
          hswish_consts_.vzero = MakeF32x4Const(0);
          break;
        default:
          XNN_UNREACHABLE;
      }
      params = I32Add(params, I32Const(6 * sizeof(float)));
    }
  }

  void ApplyPostOps(const xnn_post_operation* ops, size_t num_ops, Local& v) {
    for (size_t i = 0; i < num_ops; i++) {
      switch (ops[i].op_type) {
        case xnn_post_operation_type_hardswish:
          Hswish(v);
          break;
        default:
          XNN_UNREACHABLE;
      }
    }
  }

  void ApplyPostOps(const xnn_post_operation* ops, size_t num_ops, LocalsArray& vs) {
    for (auto& v : vs) ApplyPostOps(ops, num_ops, v);
  }

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

  void InitClampLimit(float min, float max) {
    clamps_consts_.clamp_min = min != -std::numeric_limits<float>::infinity();
    clamps_consts_.clamp_max = max != +std::numeric_limits<float>::infinity();
    if (clamps_consts_.clamp_min) {
      clamps_consts_.vmin = MakeF32x4Const(min);
    }
    if (clamps_consts_.clamp_max) {
      clamps_consts_.vmax = MakeF32x4Const(max);
    }
  }

  void Clamp(Local& value) {
    if (clamps_consts_.clamp_max) {
      value = F32x4Pmin(clamps_consts_.vmax, value);
    }
    if (clamps_consts_.clamp_min) {
      value = F32x4Pmax(clamps_consts_.vmin, value);
    }
  }

  void Clamp(LocalsArray& values) {
    for (auto& value : values) Clamp(value);
  }

  HswishConsts hswish_consts_;
  ClampConsts clamps_consts_;
};

xnn_status generate(xnn_code_buffer* b, const char* name, size_t max_mr, const void* params) {
  xnnpack::F32GemmLoadsplatGenerator generator(b);

  generator.generate(name, max_mr, static_cast<const jit_gemm_params*>(params));
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
xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat(xnn_code_buffer* b, size_t max_mr,
                                                                       size_t nc_mod_nr, size_t kc,
                                                                       const void* params) {
  static const char* kFunctionName = "xnn_generate_f32_gemm_ukernel_6x8__wasmsimd_x86_loadsplat";
  assert(max_mr <= 6);
  return xnnpack::generate(b, kFunctionName, max_mr, params);
}
}
