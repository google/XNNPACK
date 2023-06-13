#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/wasm-assembler.h>

#include <cstdint>

#include <xnnpack/wasm-assembler-macro-define.h>

namespace xnnpack {
namespace {
class F32GemmLoadsplatGenerator : public WasmAssembler {
 public:
  explicit F32GemmLoadsplatGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {}

  void generate(const char* name) {
    ValTypesToInt locals_declaration = {{i32, 3}, {v128, 5}};
    AddFunc<10>(
        {}, name, {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32},
        locals_declaration,
        [&](auto mr, auto nc, auto kc, auto a, auto a_stride, auto w, auto c,
            auto cm_stride, auto cn_stride, auto params) {
          constexpr size_t kFloatSize = sizeof(float);
          constexpr size_t kV128Size = 16;

          auto a0 = MakeLocal(a);
          auto c0 = MakeLocal(c);

          auto vacc0x0123 = MakeLocal(v128);
          auto vacc0x4567 = MakeLocal(v128);

          DO_WHILE(
              {
                vacc0x0123 = V128Load(w, /*offset=*/0);
                vacc0x4567 = V128Load(w, /*offset=*/kV128Size);
                w = I32Add(w, I32Const(8 * kFloatSize));

                auto k = MakeLocal(kc);
                DO_WHILE(
                    {
                      auto va0 = MakeLocal(V128Load32Splat(a0));
                      a0 = I32Add(a0, I32Const(kFloatSize));

                      auto vb0123 = MakeLocal(V128Load(w, /*offset=*/0));
                      auto vb4567 =
                          MakeLocal(V128Load(w, /*offset=*/kV128Size));

                      w = I32Add(w, I32Const(8 * kFloatSize));

                      vacc0x0123 = F32x4Add(vacc0x0123, F32x4Mul(va0, vb0123));
                      vacc0x4567 = F32x4Add(vacc0x4567, F32x4Mul(va0, vb4567));

                      k = I32Sub(k, I32Const(kFloatSize));
                    },
                    I32Ne(k, I32Const(0)));
                IF_ELSE(
                    I32GeU(nc, I32Const(8)),
                    {
                      V128Store(c0, vacc0x0123);
                      V128Store(c0, vacc0x4567, /*offset=*/kV128Size);
                      c0 = I32Add(c0, cn_stride);
                      a0 = I32Sub(a0, kc);
                      nc = I32Sub(nc, I32Const(8));
                    },
                    {
                      IF(I32And(nc, I32Const(4)), {
                        V128Store(c0, vacc0x0123);
                        vacc0x0123 = vacc0x4567;
                        c0 = I32Add(c0, I32Const(kV128Size));
                      });
                      IF(I32And(nc, I32Const(2)), {
                        V128Store64Lane(c0, vacc0x0123, 0);
                        vacc0x0123 =
                            I64x2Shuffle(vacc0x0123, vacc0x0123, {1, 1});
                        c0 = I32Add(c0, I32Const(2 * kFloatSize));
                      });
                      IF(I32And(nc, I32Const(1)),
                         { V128Store32Lane(c0, vacc0x0123, 0); });
                      nc = I32Const(0);
                    });
              },
              I32Ne(nc, I32Const(0)));
        });
  }
};

}  // namespace
}  // namespace xnnpack

#include <xnnpack/wasm-assembler-macro-undef.h>

extern "C" {
xnn_status_t xnn_generate_f32_gemm_ukernel_1x8__wasmsimd_x86_loadsplat(
    xnn_code_buffer* b, size_t max_mr, size_t nc_mod_nr, size_t kc,
    const void* params) {
  static const char* kFunctionName =
      "xnn_generated_f32_gemm_1x8__wasmsimd_x86_loadsplat";

  assert(max_mr <= 1);
  xnnpack::F32GemmLoadsplatGenerator generator(b);

  generator.generate(kFunctionName);
  generator.Emit();
  auto finalized = generator.finalize();
  if (finalized == nullptr || generator.error() != xnnpack::Error::kNoError) {
    return xnn_status_uninitialized;
  }
  return xnn_finalize_code_memory(b);
}
}
