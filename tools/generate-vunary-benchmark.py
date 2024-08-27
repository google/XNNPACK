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
    "-k",
    "--ukernel",
    required=True,
    help="Microkernel",
)
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C++ source) file",
)
parser.set_defaults(defines=list())

BENCHMARK_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, 
                                datatype, params_type, init_params)                        
BENCHMARK_CAPTURE(${DATATYPE}_v${OP_NAME}, ukernel, arch_flags, ukernel, init_params)      
  ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)
  ->UseRealTime();"""

BENCHMARK_FUNCTION_TEMPLATE = """\
$if OP_NAME.startswith("rnd"):
  void ${DATATYPE}_v${OP_NAME}(benchmark::State& state, uint64_t arch_flags, xnn_${DATATYPE}_vround_ukernel_fn ukernel,
                xnn_init_${DATATYPE}_rnd_params_fn init_params = nullptr) {
    ${DATATYPE}_vunary_benchmark<xnn_${DATATYPE}_rnd_params>(
        state, ukernel,
        init_params,
        arch_flags,
        /*range_min=*/${RANGE_MIN},
        /*range_max=*/${RANGE_MAX});
  }
$elif OP_NAME == "clamp":
  void ${DATATYPE}_v${OP_NAME}(benchmark::State& state, uint64_t arch_flags, xnn_${DATATYPE}_v${OP_NAME}_ukernel_fn ukernel,
                xnn_init_${DATATYPE}_minmax_params_fn init_params = nullptr) {
    ${DATATYPE}_vunary_benchmark<xnn_${DATATYPE}_minmax_params>(
        state, ukernel,
        [init_params](xnn_${DATATYPE}_minmax_params* params) -> size_t {
          $if DATATYPE == "f16":
            init_params(params, -1.0f, 1.0f);
          $else:
            init_params(params, -INFINITY, INFINITY);
          return sizeof(*params);
        },
        arch_flags,
        /*range_min=*/${RANGE_MIN},
        /*range_max=*/${RANGE_MAX});
  }
$elif OP_NAME in ("abs", "gelu", "log", "neg", "sqr"):
  void ${DATATYPE}_v${OP_NAME}(benchmark::State& state, uint64_t arch_flags, xnn_${DATATYPE}_v${OP_NAME}_ukernel_fn ukernel,
                xnn_init_${DATATYPE}_default_params_fn init_params = nullptr) {
    ${DATATYPE}_vunary_benchmark<xnn_${DATATYPE}_default_params>(
        state, ukernel,
        init_params,
        arch_flags,
        /*range_min=*/${RANGE_MIN},
        /*range_max=*/${RANGE_MAX});
  }
$else:
  void ${DATATYPE}_v${OP_NAME}(benchmark::State& state, uint64_t arch_flags, xnn_${DATATYPE}_v${OP_NAME}_ukernel_fn ukernel,
                xnn_init_${DATATYPE}_${OP_NAME}_params_fn init_params = nullptr) {
    ${DATATYPE}_vunary_benchmark<xnn_${DATATYPE}_${OP_NAME}_params>(
        state, ukernel,
        $if OP_NAME == "lrelu":
          [init_params](xnn_${DATATYPE}_${OP_NAME}_params* params) -> size_t {
            init_params(params, 0.01f);
            return sizeof(*params);
          },
        $elif OP_NAME == "elu":
          $if DATATYPE == "f16":
            [init_params](xnn_${DATATYPE}_${OP_NAME}_params* params) -> size_t {
              init_params(params,
                          /*prescale=*/1.0f,
                          /*alpha=*/1.0f,
                          /*beta=*/1.0f);
              return sizeof(*params);
            },
          $else:
            [init_params](xnn_${DATATYPE}_${OP_NAME}_params* params) -> size_t {
              init_params(params, /*prescale=*/1.0f, /*alpha=*/1.0f, /*beta=*/1.0f);
              return sizeof(*params);
            },
        $else:
          init_params,
        arch_flags,
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

def main(args):
  options = parser.parse_args(args)

  # Extract the datatype and op from the file name.
  datatype, op_name = options.ukernel.split('-')

  benchmarks = """\
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {microkernel}
//   Generator: {generator}

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "bench/{datatype}-vunary-benchmark.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

""".format(microkernel=options.ukernel, generator=sys.argv[0], datatype=datatype)

  op_name = op_name[1:]

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

  benchmarks += xnncommon.make_multiline_macro(xngen.preprocess(
      BENCHMARK_TEMPLATE,
      {
          "OP_NAME": op_name,
          "DATATYPE": datatype,
      },
  ))

  folder = options.ukernel
  if "rnd" in folder:
    folder = folder[0:8]

  benchmarks += f'#include "{xnncommon._XNNPACK_SRC}/{folder}/{options.ukernel}.h"\n'
  benchmarks += "#undef XNN_UKERNEL_WITH_PARAMS\n"

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
