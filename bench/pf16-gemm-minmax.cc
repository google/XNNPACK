// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/pf16-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstdint>
#include <functional>

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"

namespace {

struct ConstantOrFunction {
  ConstantOrFunction(size_t x) : fn([x]() { return x; }) {}  //NOLINT
  ConstantOrFunction(int x) : fn([x]() { return x; }) {}  //NOLINT
  template <typename Fn>
  ConstantOrFunction(Fn fn) : fn(std::move(fn)) {}  //NOLINT

  std::function<size_t()> fn;

  operator size_t() const { return fn(); }  //NOLINT
};

}  // namespace



#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void pf16_gemm_minmax_ukernel_1x32c2__neonsme2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_kai_f16_weights_and_biases,
      xnn_packed_stride_kai_f16_weights_and_biases,
      /*mr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  , /*nr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_nr();
        } else {
          return 0;
        }
      }
  , /*kr=*/2, /*sr=*/1,
      /*mr_packed=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  ,
      /*arch_flags=*/xnn_arch_arm_sme2);
  }

  BENCHMARK_GEMM(pf16_gemm_minmax_ukernel_1x32c2__neonsme2)

  static void pf16_gemm_minmax_ukernel_32x32c2__neonsme2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_kai_f16_weights_and_biases,
      xnn_packed_stride_kai_f16_weights_and_biases,
      /*mr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  , /*nr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_nr();
        } else {
          return 0;
        }
      }
  , /*kr=*/2, /*sr=*/1,
      /*mr_packed=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  ,
      /*arch_flags=*/xnn_arch_arm_sme2);
  }

  BENCHMARK_GEMM(pf16_gemm_minmax_ukernel_32x32c2__neonsme2)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
