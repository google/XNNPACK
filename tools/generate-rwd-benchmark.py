#!/usr/bin/env python
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='RWD microkernel test generator')
parser.add_argument("-k", "--ukernel", required=True,
                    help="Microkernel")
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=False,
    help="Benchmark output (C++ source) file(s)")
parser.set_defaults(defines=list())


BENCHMARK_FUNCTION_TEMPLATE = """\
void ${DATATYPE}_${OP_NAME}(
  benchmark::State& state, 
  xnn_${DATATYPE}_rwd_ukernel_fn ${OP_NAME},
  xnn_init_${DATATYPE}_default_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t rows = state.range(0);
  const size_t channels = state.range(1);

  std::vector<${DATATYPE2}, AlignedAllocator<${DATATYPE2}, 64>> input(rows * channels + XNN_EXTRA_BYTES / sizeof(${DATATYPE2}));
  std::vector<${DATATYPE2}> output(channels);
  std::vector<${DATATYPE2}> zero(channels + XNN_EXTRA_BYTES / sizeof(${DATATYPE2}), 0.f);
  std::iota(input.begin(), input.end(), 0.0f);

  // Prepare parameters.
  int64_t padding[2] = {0,0};
  int64_t base_dilation = 1;
  int64_t window_dilation = 1;
  int64_t window_dimensions = rows;
  int64_t window_stride = 1;
  ${DATATYPE2} init_value = 0;
  struct xnn_${DATATYPE}_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }

  for (auto _ : state) {
    ${OP_NAME}(rows, channels, input.data(), init_value, padding, base_dilation, window_dilation,
            window_dimensions, window_stride, output.data(), init_params != nullptr ? &params : nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkRWDSUM(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"rows", "channels"});
  b->Args({8, 1024});
  b->Args({16, 1024});
  b->Args({10240, 1024});
}

"""


BENCHMARK_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,
                                datatype, params_type, init_params)
BENCHMARK_CAPTURE(${DATATYPE}_${OP_NAME}, arch_flags, ukernel)      
  ->Apply(BenchmarkRWDSUM)
  ->UseRealTime();"""



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

#include <numeric>
#include <benchmark/benchmark.h>

#include "bench/utils.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/reduce.h"

""".format(microkernel=options.ukernel, generator=sys.argv[0], datatype=datatype)
  
  DT_TYPES = {
  "f32": "float",
  "s32": "int",
  }


  # Create the benchmark wrapper function.
  benchmarks += xngen.preprocess(
      BENCHMARK_FUNCTION_TEMPLATE,
      {
          "DATATYPE": datatype,
          "DATATYPE2": DT_TYPES[datatype],
          "OP_NAME": op_name,
      },
  )
  benchmarks += xnncommon.make_multiline_macro(xngen.preprocess(
      BENCHMARK_TEMPLATE,
      {
          "DATATYPE": datatype,
          "OP_NAME": op_name,
      },
  ))

  folder = options.ukernel
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
