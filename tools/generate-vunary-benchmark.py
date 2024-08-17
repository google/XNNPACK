#!/usr/bin/env python
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xngen
import xnncommon


parser = argparse.ArgumentParser(
    description="Vector unary operation microkernel benchmark generator"
)
parser.add_argument(
    "-s",
    "--spec",
    metavar="FILE",
    required=True,
    help="Specification (YAML) file",
)
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C++ source) file",
)
parser.set_defaults(defines=list())


def split_ukernel_name(name: str) -> tuple[str, str, str]:
  """Splits a microkernel name into its components.

  Args:
    name: The kernel name.

  Returns:
    A `tuple` `(op_type, batch_tile, vector_tile, arch, isa)` where
      `op_type`: `str` name of the op.
      `arch`: `str` target architecture.
      `isa`: `str` target ISA.
  """
  match = re.fullmatch(
      r"(?:xnn_|xnn_generate_)(f16|f32)(_(f16|f32))*_v(abs|clamp|elu|gelu|hswish|log|lrelu|neg|relu|rndd|rndne|rndu|rndz|rsqrt|sigmoid|sqr|sqrt|sqrtshift|tanh)_(fact_)?ukernel__(.+)_u(\d+)(v)?",
      name,
  )
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  op_type = match.group(4)

  arch, isa, _ = xnncommon.parse_target_name(target_name=match.group(6))
  return op_type, arch, isa


BENCHMARK_TEMPLATE = """\
BENCHMARK_CAPTURE(${OP_NAME}, ${BENCHMARK_NAME},
                  ${KERNEL},
                  $if CHECK_ISA:
                    ${INIT_PARAMS},
                    ${CHECK_ISA})
                  $else:
                    ${INIT_PARAMS})
  $if DATATYPE == "f16":
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
  $else:
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();"""

BENCHMARK_FUNCTION_TEMPLATE = """\
$if OP_NAME.startswith("rnd"):
  void ${DATATYPE}_v${OP_NAME}(benchmark::State& state, xnn_${DATATYPE}_vround_ukernel_fn ukernel,
                xnn_init_${DATATYPE}_rnd_params_fn init_params = nullptr,
                benchmark::utils::IsaCheckFunction isa_check = nullptr) {
    ${DATATYPE}_vunary_benchmark<xnn_${DATATYPE}_rnd_params>(
        state, ukernel,
        init_params,
        isa_check,
        /*range_min=*/${RANGE_MIN},
        /*range_max=*/${RANGE_MAX});
  }
$elif OP_NAME == "clamp":
  void ${DATATYPE}_v${OP_NAME}(benchmark::State& state, xnn_${DATATYPE}_v${OP_NAME}_ukernel_fn ukernel,
                xnn_init_${DATATYPE}_minmax_params_fn init_params = nullptr,
                benchmark::utils::IsaCheckFunction isa_check = nullptr) {
    ${DATATYPE}_vunary_benchmark<xnn_${DATATYPE}_minmax_params>(
        state, ukernel,
        [init_params](xnn_${DATATYPE}_minmax_params* params) -> size_t {
          $if DATATYPE == "f16":
            init_params(params,
                UINT16_C(0xAC00),  // -1.0h
                UINT16_C(0x3C00));  // 1.0h
          $else:
            init_params(params, -INFINITY, INFINITY);
          return sizeof(*params);
        },
        isa_check,
        /*range_min=*/${RANGE_MIN},
        /*range_max=*/${RANGE_MAX});
  }
$elif OP_NAME in ("abs", "gelu", "log", "neg", "sqr"):
  void ${DATATYPE}_v${OP_NAME}(benchmark::State& state, xnn_${DATATYPE}_v${OP_NAME}_ukernel_fn ukernel,
                xnn_init_${DATATYPE}_default_params_fn init_params = nullptr,
                benchmark::utils::IsaCheckFunction isa_check = nullptr) {
    ${DATATYPE}_vunary_benchmark<xnn_${DATATYPE}_default_params>(
        state, ukernel,
        init_params,
        isa_check,
        /*range_min=*/${RANGE_MIN},
        /*range_max=*/${RANGE_MAX});
  }
$else:
  void ${DATATYPE}_v${OP_NAME}(benchmark::State& state, xnn_${DATATYPE}_v${OP_NAME}_ukernel_fn ukernel,
                xnn_init_${DATATYPE}_${OP_NAME}_params_fn init_params = nullptr,
                benchmark::utils::IsaCheckFunction isa_check = nullptr) {
    ${DATATYPE}_vunary_benchmark<xnn_${DATATYPE}_${OP_NAME}_params>(
        state, ukernel,
        $if OP_NAME == "lrelu":
          [init_params](xnn_${DATATYPE}_${OP_NAME}_params* params) -> size_t {
            $if DATATYPE == "f16":
              init_params(params, UINT16_C(0x1F00));  // 0.01h
            $else:
              init_params(params, 0.01f);
            return sizeof(*params);
          },
        $elif OP_NAME == "elu":
          $if DATATYPE == "f16":
            [init_params](xnn_${DATATYPE}_${OP_NAME}_params* params) -> size_t {
              init_params(params,
                          /*prescale=*/UINT16_C(0x3C00),  // prescale = 1.0h
                          /*alpha=*/UINT16_C(0x3C00),     // alpha = 1.0h
                          /*beta=*/UINT16_C(0x3C00));     // beta = 1.0h
              return sizeof(*params);
            },
          $else:
            [init_params](xnn_${DATATYPE}_${OP_NAME}_params* params) -> size_t {
              init_params(params, /*prescale=*/1.0f, /*alpha=*/1.0f, /*beta=*/1.0f);
              return sizeof(*params);
            },
        $else:
          init_params,
        isa_check,
        /*range_min=*/${RANGE_MIN},
        /*range_max=*/${RANGE_MAX});
  }
"""

RANGE_FOR_OP_NAME = {
    "f16_tanh": (-5.0, 5.0),
    "f32_clamp": (0.0, 10.0),
    "f32_elu": (-20.0, 10.0),
    "f16_elu": (-9.0, 9.0),
    "f32_log": (0.0, 10.0),
    "f32_lrelu": (-5.0, 5.0),
    "f16_lrelu": (-5.0, 5.0),
    "f32_rsqrt": (1e-5, 10.0),
    "f16_rsqrt": (1e-5, 10.0),
    "f32_sqrt": (0.0, 10.0),
    "f16_sqrt": (0.0, 1.0),
}


def generate_benchmark_cases(
    ukernel: str,
    op_type: str,
    isa: str,
    init_fn: str | None = None,
):
  """Generates all tests cases for a Vector Unary Operation micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    op_type: Operation type.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.
    init_fn: C name of the function to initialize microkernel parameters.

  Returns:
    Code for the test case.
  """
  datatype = ukernel.split("_", 2)[1]
  check_isa = xnncommon.generate_isa_utilcheck_macro(isa)
  if check_isa:
    check_isa = f"benchmark::utils::{check_isa}"
  return xngen.preprocess(
      BENCHMARK_TEMPLATE,
      {
          "OP_NAME": f"{datatype}_v{op_type}",
          "BENCHMARK_NAME": ukernel.split("__", 1)[1],
          "KERNEL": ukernel,
          "INIT_PARAMS": init_fn or "/*init_params=*/nullptr",
          "CHECK_ISA": check_isa,
          "DATATYPE": datatype,
      },
  )


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    # Extract the datatype and op from the file name.
    datatype, op_name = (
        os.path.basename(options.spec).split(".", 1)[0].split("-v", 2)
    )

    benchmarks = """\
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "bench/{datatype}-vunary-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"

""".format(specification=options.spec, generator=sys.argv[0], datatype=datatype)

    # Create the benchmark wrapper function.
    range_min, range_max = RANGE_FOR_OP_NAME.get(
        f"{datatype}_{op_name}", (-10.0, 10.0)
    )
    benchmarks += xngen.preprocess(
        BENCHMARK_FUNCTION_TEMPLATE,
        {
            "DATATYPE": datatype,
            "OP_NAME": op_name,
            "RANGE_MIN": range_min,
            "RANGE_MAX": range_max,
        },
    )

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      op_type, arch, isa = split_ukernel_name(name)

      test_case = generate_benchmark_cases(name, op_type, isa, init_fn)
      benchmarks += "\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    # Strip out consecutive preprocessor `endif`/`if` pairs.
    benchmarks = re.sub(
        r"\#endif  // ([^\n]+)\n\n\#if \1\n", "", benchmarks, flags=re.MULTILINE
    )

    # Footer with `main` function.
    benchmarks += "\n\n" + """\
#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
"""

    # Finally, write the file to disk.
    xnncommon.overwrite_if_changed(options.output, benchmarks)


if __name__ == "__main__":
  main(sys.argv[1:])
