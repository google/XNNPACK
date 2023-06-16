#include <wasm_simd128.h>
#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/wasm-assembler.h>

#include <cstdint>
#include <limits>

namespace xnnpack {
namespace {
class F32GemmLoadsplatGenerator : public WasmAssembler {
 public:
  explicit F32GemmLoadsplatGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {}

  void generate(const char* name, const jit_gemm_params* jit_gemm_params) {
    const float min = jit_gemm_params->f32_minmax.min;
    const float max = jit_gemm_params->f32_minmax.max;
    const bool clamp_min = min != -std::numeric_limits<float>::infinity();
    const bool clamp_max = max != +std::numeric_limits<float>::infinity();

    ValTypesToInt locals_declaration = {
        {i32, 3},
        {v128, 5 + static_cast<int>(clamp_min) + static_cast<int>(clamp_max)}};
    AddFunc<10>(
        {}, name, {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32},
        locals_declaration,
        [&](auto mr, auto nc, auto kc, auto a, auto a_stride, auto w, auto c,
            auto cm_stride, auto cn_stride, auto params) {
          auto a0 = MakeLocal(a);
          auto c0 = MakeLocal(c);

          auto vacc0x0123 = MakeLocal(v128);
          auto vacc0x4567 = MakeLocal(v128);

          Local vmin, vmax;
          InitClampLimit(vmin, min, clamp_min);
          InitClampLimit(vmax, max, clamp_max);

          DoWhile(
              [&] {
                vacc0x0123 = V128Load(w, /*offset=*/0);
                vacc0x4567 = V128Load(w, /*offset=*/sizeof(v128_t));
                w = I32Add(w, I32Const(8 * sizeof(float)));

                auto k = MakeLocal(kc);
                DoWhile(
                    [&] {
                      auto va0 = MakeLocal(V128Load32Splat(a0));
                      a0 = I32Add(a0, I32Const(sizeof(float)));

                      auto vb0123 = MakeLocal(V128Load(w, /*offset=*/0));
                      auto vb4567 = MakeLocal(V128Load(w, /*offset=*/sizeof(v128_t)));

                      w = I32Add(w, I32Const(8 * sizeof(float)));

                      vacc0x0123 = F32x4Add(vacc0x0123, F32x4Mul(va0, vb0123));
                      vacc0x4567 = F32x4Add(vacc0x4567, F32x4Mul(va0, vb4567));

                      k = I32Sub(k, I32Const(sizeof(float)));
                    },
                    [&] { I32Ne(k, I32Const(0)); });

                Clamp(vacc0x0123, vmin, vmax, clamp_min, clamp_max);
                Clamp(vacc0x4567, vmin, vmax, clamp_min, clamp_max);

                IfElse([&] { I32GeU(nc, I32Const(8)); },
                       [&] {
                         V128Store(c0, vacc0x0123);
                         V128Store(c0, vacc0x4567, /*offset=*/sizeof(v128_t));
                         c0 = I32Add(c0, cn_stride);
                         a0 = I32Sub(a0, kc);
                         nc = I32Sub(nc, I32Const(8));
                       },
                       [&] {
                         If([&] { I32And(nc, I32Const(4)); },
                            [&] {
                              V128Store(c0, vacc0x0123);
                              vacc0x0123 = vacc0x4567;
                              c0 = I32Add(c0, I32Const(sizeof(v128_t)));
                            });
                         If([&] { I32And(nc, I32Const(2)); },
                            [&] {
                              V128Store64Lane(c0, vacc0x0123, 0);
                              vacc0x0123 = I64x2Shuffle(vacc0x0123, vacc0x0123, {1, 1});
                              c0 = I32Add(c0, I32Const(2 * sizeof(float)));
                            });
                         If([&] { I32And(nc, I32Const(1)); },
                            [&] { V128Store32Lane(c0, vacc0x0123, 0); });
                         nc = I32Const(0);
                       });
              },
              [&] { I32Ne(nc, I32Const(0)); });
        });
  }
 private:
  void InitClampLimit(Local& limit, float limit_float, bool is_clamping_enabled) {
    if(is_clamping_enabled) {
      limit = MakeLocal(F32x4Splat(F32Const(limit_float)));
    }
  }

  void Clamp(Local& value, const Local& vmin, const Local& vmax, bool clamp_min, bool clamp_max) {
    if (clamp_max) {
      value = F32x4Pmin(vmax, value);
    }
    if (clamp_min) {
      value = F32x4Pmax(vmin, value);
    }
  }
};

}  // namespace
}  // namespace xnnpack

extern "C" {
xnn_status_t xnn_generate_f32_gemm_ukernel_1x8__wasmsimd_x86_loadsplat(
    xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr, size_t kc,
    const void* params) {
  static const char* kFunctionName =
      "xnn_generated_f32_gemm_1x8__wasmsimd_x86_loadsplat";

  assert(max_mr <= 1);
  xnnpack::F32GemmLoadsplatGenerator generator(b);

  generator.generate(kFunctionName,
                     static_cast<const jit_gemm_params*>(params));
  generator.Emit();
  auto finalized = generator.finalize();
  if (finalized == nullptr || generator.error() != xnnpack::Error::kNoError) {
    return xnn_status_uninitialized;
  }
  return xnn_status_success;
}
}
