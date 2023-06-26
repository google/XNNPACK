#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>

#include <wasm_simd128.h>

#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/wasm-assembler.h>


namespace xnnpack {
namespace {
class F32GemmLoadsplatGenerator : public WasmAssembler {
 public:
  explicit F32GemmLoadsplatGenerator(xnn_code_buffer* bf)
      : WasmAssembler(bf) {}

  void generate(const char* name, size_t max_mr, const jit_gemm_params* jit_gemm_params) {
    const float min = jit_gemm_params->f32_minmax.min;
    const float max = jit_gemm_params->f32_minmax.max;
    const bool clamp_min = min != -std::numeric_limits<float>::infinity();
    const bool clamp_max = max != +std::numeric_limits<float>::infinity();

    ValTypesToInt locals_declaration = {
      {i32, max_mr * 2 + 1}, {v128, max_mr * 3 + 2 + static_cast<int>(clamp_min) + static_cast<int>(clamp_max)}};
    AddFunc<10>({}, name, locals_declaration,
                [&](auto mr, auto nc, auto kc, auto a, auto a_stride, auto w, auto c, auto cm_stride, auto cn_stride,
                    auto params) {
                  auto vmin = InitClampLimit(min, clamp_min);
                  auto vmax = InitClampLimit(max, clamp_max);

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

                      Clamp(vacc0123, vmin, vmax, clamp_min, clamp_max);
                      Clamp(vacc4567, vmin, vmax, clamp_min, clamp_max);

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

  Local InitClampLimit(float limit_float, bool is_clamping_enabled) {
    Local limit;
    if (is_clamping_enabled) {
      limit = MakeLocal(F32x4Splat(F32Const(limit_float)));
    }
    return limit;
  }

  void Clamp(Local& value, const Local& vmin, const Local& vmax, bool clamp_min, bool clamp_max) {
    if (clamp_max) {
      value = F32x4Pmin(vmax, value);
    }
    if (clamp_min) {
      value = F32x4Pmax(vmin, value);
    }
  }

  void Clamp(LocalsArray& values, const Local& vmin, const Local& vmax, bool clamp_min, bool clamp_max) {
    for (auto& value : values) {
      Clamp(value, vmin, vmax, clamp_min, clamp_max);
    }
  }
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
