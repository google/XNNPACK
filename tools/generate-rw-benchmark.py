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


parser = argparse.ArgumentParser(description='RW microkernel test generator')
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
  xnn_${DATATYPE}_rw_ukernel_fn ${OP_NAME},
  xnn_init_${DATATYPE}_default_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t rows = state.range(0);
  const size_t batch = state.range(1);

  std::vector<${DATATYPE2}, AlignedAllocator<${DATATYPE2}, 64>> input(rows * batch + XNN_EXTRA_BYTES / sizeof(${DATATYPE2}));
  std::vector<${DATATYPE2}> output(rows);
  std::iota(input.begin(), input.end(), 1);

  // Prepare parameters.
  int64_t padding[2] = {0,0};
  int64_t base_dilation = 1;
  int64_t window_dilation = 1;
  int64_t window_dimensions = batch;
  int64_t window_stride = 1;
  ${DATATYPE2} init_value = 0;
  xnn_${DATATYPE}_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }

  for (auto _ : state) {
    for (int64_t i = 0; i < rows; ++i) {
      ${OP_NAME}(batch * sizeof(${DATATYPE2}), &input[i * batch], init_value, padding, base_dilation, window_dilation,
            window_dimensions, window_stride, &output[i], init_params != nullptr ? &params : nullptr);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkRWSUM(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"channels","rows"});
  b->Args({1, 512});
  b->Args({1, 1024});
  b->Args({1, 8000});
  b->Args({512, 512});
  b->Args({512, 1024});
  b->Args({512, 8000});
  b->Args({1024, 64});
  b->Args({32768, 1});
  b->Args({10240, 1024});
}

"""


BENCHMARK_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,
                                datatype, params_type, init_params)
BENCHMARK_CAPTURE(${DATATYPE}_${OP_NAME}, arch_flags, ukernel)      
  ->Apply(BenchmarkRWSUM)
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
