#!/usr/bin/env python
# Copyright 2023 Google LLC
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


parser = argparse.ArgumentParser(description='PackW microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument(
    "-b",
    "--output-bench",
    metavar="FILE",
    required=False,
    help="Benchmark output (C++ source) file(s)")
parser.set_defaults(defines=list())

def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(x8|x16|x32)_packw_gemm_goi_ukernel_x(\d+)(c(\d+))?(s(\d+))?(v)?__(.+)_u(\d+)(_(.+))?", name)
  assert match is not None
  nr = int(match.group(2))
  if match.group(3):
    kr = int(match.group(4))
  else:
    kr = 1
  if match.group(5):
    sr = int(match.group(6))
  else:
    sr = 1
  vector_tile = match.group(7)
  kblock = int(match.group(9))
  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(8))
  return nr, kr, sr, kblock, vector_tile, arch, isa

PACKW_BENCHMARK_TEMPLATE = """\
static void ${BENCHMARK_NAME}(benchmark::State& state, const char* net) {
  ${DATATYPE}_packw(state,
    ${UKERNEL_NAME},
    $if ISA_CHECK:
      /*nr=*/${NR}${NR_SCALE}, /*kr=*/${KR}, /*sr=*/${SR},
      benchmark::utils::${ISA_CHECK});
    $else:
      /*nr=*/${NR}${NR_SCALE}, /*kr=*/${KR}, /*sr=*/${SR});
}
BENCHMARK_BGEMM(${BENCHMARK_NAME})
"""

def generate_test_cases(ukernel, nr, kr, sr, kblock, vector_tile, isa):
  """Generates all tests cases for a PACKW micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    nr: NR parameter of the PACKW micro-kernel.
    kr: KR parameter of the PACKW micro-kernel.
    sr: SR parameter of the PACKW micro-kernel.
    kblock: unroll factor along the K dimension.
    vector_tile: Indicates if vector tile is specified in vectors rather than
                 elements.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  nr_scale = ""
  if vector_tile:
    ctype = {"x8": "uint8_t", "x16": "uint16_t", "x32": "uint32_t"}[datatype]
    nr_scale = {"rvv": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype}[isa]

  benchmark = xngen.preprocess(PACKW_BENCHMARK_TEMPLATE, {
      "DATATYPE": datatype,
      "BENCHMARK_NAME": test_name,
      "UKERNEL_NAME": ukernel,
      "NR": nr,
      "KR": kr,
      "SR": sr,
      "KBLOCK": kblock,
      "NR_SCALE": nr_scale,
      "NR_SUFFIX": "v" if vector_tile else "",
      "ISA_CHECK": xnncommon.generate_isa_utilcheck_macro(isa),
      "next_prime": next_prime,
    })

  return benchmark


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    bench_output = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <benchmark/benchmark.h>
#include "bench/bgemm.h"
#include "bench/packw-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/packw.h"
""".format(specification=options.spec, generator=sys.argv[0])

    isa_hierarchy = xnncommon._ISA_HIERARCHY_MAP
    benches = [""] * len(isa_hierarchy)

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      nr, kr, sr, kblock, vector_tile, arch, isa = split_ukernel_name(name)

      benchmark = generate_test_cases(name, nr, kr, sr, kblock, vector_tile, isa)

      benches[isa_hierarchy.get(isa, 0)] += \
        "\n\n" + xnncommon.postprocess_test_case(benchmark, arch, isa)

    for arch_idx in reversed(range(len(isa_hierarchy))):
      bench_output += benches[arch_idx]

    bench_output += """\n
#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
"""

    if options.output_bench:
      output_name = options.output_bench
      xnncommon.overwrite_if_changed(output_name, bench_output)
    else:
      raise Exception("options.output_bench must be True")

if __name__ == "__main__":
  main(sys.argv[1:])
