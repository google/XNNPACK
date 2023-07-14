#include <algorithm>

#include <wasm_simd128.h>

#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/wasm-assembler.h>


namespace xnnpack {
namespace internal {

class PostOps : public WasmAssembler {
 public:
  using WasmAssembler::WasmAssembler;

  void InitPostOps(const jit_gemm_params* jit_gemm_params, Local& params) {
    InitClampLimit(jit_gemm_params->f32_minmax.min, jit_gemm_params->f32_minmax.max);
    InitNonClampPostOps(jit_gemm_params->post_operations, jit_gemm_params->num_post_operations, params);
  }

  void ApplyPostOps(LocalsArray& values) {
    Clamp(values);
    ApplyNonClampPostOps(values);
  }

 private:
  Local MakeV128Load64Splat(const Local& address, uint32_t offset) {
    return MakeLocal(V128Load64Splat(address, offset));
  }

  void InitClampLimit(float min, float max) {
    clamps_consts_.clamp_min = min != -std::numeric_limits<float>::infinity();
    clamps_consts_.clamp_max = max != +std::numeric_limits<float>::infinity();
    if (clamps_consts_.clamp_min) {
      clamps_consts_.vmin = MakeLocal(V128Const(min));
    }
    if (clamps_consts_.clamp_max) {
      clamps_consts_.vmax = MakeLocal(V128Const(max));
    }
  }

  void InitNonClampPostOps(const xnn_post_operation* ops, size_t num_ops, Local& params) {
    ops_ = ops;
    num_ops_ = num_ops;
    for (size_t i = 0; i < num_ops; i++) {
      switch (ops[i].op_type) {
        case xnn_post_operation_type_hardswish:
          hswish_consts_.vsixth = MakeV128Load64Splat(params, /*offset=*/0);
          hswish_consts_.vthree = MakeV128Load64Splat(params, /*offset=*/2 * sizeof(float));
          hswish_consts_.vsix = MakeV128Load64Splat(params, /*offset=*/4 * sizeof(float));
          hswish_consts_.vzero = MakeLocal(F32x4Splat(F32Const(0)));
          break;
        default:
          XNN_UNREACHABLE;
      }
      params = I32Add(params, I32Const(6 * sizeof(float)));
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

  void ApplyNonClampPostOps(Local& v) {
    for (size_t i = 0; i < num_ops_; i++) {
      switch (ops_[i].op_type) {
        case xnn_post_operation_type_hardswish:
          Hswish(v);
          break;
        default:
          XNN_UNREACHABLE;
      }
    }
  }

  void ApplyNonClampPostOps(LocalsArray& vs) {
    for (auto& v : vs) ApplyNonClampPostOps(v);
  }

  void Hswish(Local& v) {
    Local vacc = MakeLocal(F32x4Add(v, hswish_consts_.vthree));
    v = F32x4Mul(v, hswish_consts_.vsixth);
    vacc = F32x4Pmax(vacc, hswish_consts_.vzero);
    vacc = F32x4Pmin(vacc, hswish_consts_.vsix);
    v = F32x4Mul(vacc, v);
  }

  struct HswishConsts {
    Local vsixth;
    Local vsix;
    Local vthree;
    Local vzero;
  };

  struct ClampConsts {
    bool clamp_min{};
    bool clamp_max{};
    Local vmin;
    Local vmax;
  };

  const xnn_post_operation* ops_ = nullptr;
  size_t num_ops_{};
  HswishConsts hswish_consts_;
  ClampConsts clamps_consts_;
};

class GemmIGemmLoadsplatCommons : public PostOps {
 public:
  using PostOps::PostOps;

  void InitAccumulators(LocalsArray& vaccs, const Local& w, size_t offset) {
    vaccs[0] = V128Load(w, offset);
    std::for_each(std::next(std::begin(vaccs)), std::end(vaccs), [&](auto& vacc) { vacc = vaccs[0]; });
  }

  void InnerLoop(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& kc, size_t max_mr,
                 size_t loop_unroll_iters) {
    Local k = MakeLocal(kc);
    InnerLoopMainPart(as, vacc0123, vacc4567, w, k, max_mr, loop_unroll_iters);

    const size_t max_iters_left = loop_unroll_iters - 1;
    size_t mask = max_iters_left > 0 ? (1 << static_cast<size_t>(log2(max_iters_left))) : 0;

    if (max_iters_left > 0) {
      If([&] { I32NeZ(k); },
         [&] {
           while (mask > 0) {
             If([&] { I32GeU(k, I32Const(mask * sizeof(float))); },
                [&] { InnerLoopBody(as, vacc0123, vacc4567, w, k, max_mr, mask); });
             mask >>= 1;
           }
         });
    }
  }

 private:
  void InnerLoopBody(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& k, size_t max_mr,
                     size_t loop_unroll_iters) {
    for (size_t unrolled_iter = 0; unrolled_iter < loop_unroll_iters; unrolled_iter++) {
      const auto vb0123 = MakeLocal(V128Load(w, /*offset=*/(2 * unrolled_iter) * sizeof(v128_t)));
      const auto vb4567 = MakeLocal(V128Load(w, /*offset=*/(2 * unrolled_iter + 1) * sizeof(v128_t)));
      for (size_t i = 0; i < max_mr; i++) {
        const auto va = MakeLocal(V128Load32Splat(as[i]));
        vacc0123[i] = F32x4Add(vacc0123[i], F32x4Mul(va, vb0123));
        vacc4567[i] = F32x4Add(vacc4567[i], F32x4Mul(va, vb4567));
        as[i] = I32Add(as[i], I32Const(sizeof(float)));
      }
    }
    w = I32Add(w, I32Const(8 * loop_unroll_iters * sizeof(float)));
    k = I32Sub(k, I32Const(loop_unroll_iters * sizeof(float)));
  }

  void InnerLoopMainPart(LocalsArray& as, LocalsArray& vacc0123, LocalsArray& vacc4567, Local& w, Local& k,
                         size_t max_mr, size_t loop_unroll_iters) {
    const auto body = [&] { InnerLoopBody(as, vacc0123, vacc4567, w, k, max_mr, loop_unroll_iters); };
    if (loop_unroll_iters == 1) {
      DoWhile(body, [&] { I32NeZ(k); });
    } else {
      While([&] { I32GeU(k, I32Const(loop_unroll_iters * sizeof(float))); }, body);
    }
  }
};
}  // namespace internal
}  // namespace xnnpack
