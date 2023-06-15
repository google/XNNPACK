#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/wasm-assembler.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace xnnpack {
namespace {
class VReluGenerator : public WasmAssembler {
 public:
  explicit VReluGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {}

  void generate(const char* name, int k_unroll, bool use_local) {
    ValTypesToInt locals_declaration = {{i32, use_local ? 2 : 1}};
    AddFunc<4>({}, name, {i32, i32, i32, i32}, locals_declaration,
               [&](Local count, Local src, Local dst, Local params) {
                 auto value = MakeLocal(i32);
                 While([&] { I32LeS(I32Const(k_unroll * 4), count); },
                       [&] {
                         for (int k = 0; k < k_unroll; k++) {
                           value = I32Load(src, /*offset=*/k * 4);
                           GenerateReluViaShR(value, use_local);
                           I32Store(dst, value, /*offset=*/k * 4);
                         }
                         src = I32Add(src, I32Const(4 * k_unroll));
                         dst = I32Add(dst, I32Const(4 * k_unroll));
                         count = I32Sub(count, I32Const(k_unroll * 4));
                       });
                 While([&] { I32LtS(I32Const(0), count); },
                       [&] {
                         value = I32Load(src);
                         GenerateReluViaShR(value, use_local);
                         I32Store(dst, value);
                         src = I32Add(src, I32Const(4));
                         dst = I32Add(dst, I32Const(4));
                         count = I32Sub(count, I32Const(4));
                       });
               });
  }

 private:
  void GenerateReluViaShR(Local& value, bool use_locals) {
    if (use_locals)
      GenerateReluViaShRWithLocals(value);
    else
      GenerateReluViaShRNoLocal(value);
  }

  void GenerateReluViaShRNoLocal(Local& value) {
    value = I32And(I32Sub(I32ShrU(value, I32Const(31)), I32Const(1)), value);
  }

  void GenerateReluViaShRWithLocals(Local& value) {
    auto mask = MakeLocal(i32);
    mask = I32ShrU(value, I32Const(31));
    mask = I32Sub(mask, I32Const(1));
    value = I32And(mask, value);
  }
};

}  // namespace
}  // namespace xnnpack

extern "C" {
xnn_status_t xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr(xnn_code_buffer* b,
                                                            size_t k_unroll,
                                                            int use_locals) {
  static const char* kFunctionName = "f32_vrelu_ukernel__jit_wasm32_shr_cnt";

  xnnpack::VReluGenerator generator(b);

  generator.generate(kFunctionName, k_unroll, static_cast<bool>(use_locals));
  generator.Emit();
  auto finalized = generator.finalize();
  if (finalized == nullptr || generator.error() != xnnpack::Error::kNoError) {
    return xnn_status_uninitialized;
  }
  return xnn_status_success;
}
}
