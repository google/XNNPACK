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


parser = argparse.ArgumentParser(
    description="Reduce discontiguous microkernel test generator"
)
parser.add_argument(
    "-t",
    "--tester",
    metavar="TESTER",
    required=True,
    choices=["ReduceMicrokernelTester", "RDSumMicrokernelTester"],
    help="Tester class to be used in the generated test",
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


def split_ukernel_name(name):
  match = re.fullmatch(
      r"xnn_(qs8|qu8|f16_f32acc|f16|f32|s8|u8)_rd(minmax|max|min|sum)?(_minmax)?(_(fp32|rndnu))?_ukernel_((\d+)p)?(\d+)x__(.+)_(c)?(u)?(\d+)(_acc(\d+))?(v)?",
      name,
  )

  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  op_type = {
      "minmax": "MinMax",
      "max": "Max",
      "min": "Min",
      "sum": "Sum",
  }[match.group(2)]

  requantization_type = match.group(5)
  primary_tile = int(match.group(7))
  incremental_tile = int(match.group(8))
  channel_tile = int(match.group(12))
  target_name = match.group(9)
  vector_tile = bool(match.group(15))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(9))
  return (
      requantization_type,
      op_type,
      primary_tile,
      incremental_tile,
      channel_tile,
      vector_tile,
      arch,
      isa,
  )


RDSUM_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  ${TESTER}()
    .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
    .channels(channel_tile)
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  ${TESTER}()
    .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
    .channels(channel_tile)
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      .input_stride(${next_prime(CHANNEL_TILE+1)})
    $else:
      .input_stride(channel_tile+1)
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  ${TESTER}()
    .rows({1, ${PRIMARY_TILE+INCREMENTAL_TILE}})
    .channels(channel_tile)
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  ${TESTER}()
    .rows({1, ${PRIMARY_TILE+INCREMENTAL_TILE}})
    .channels(channel_tile)
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      .input_stride(${next_prime(CHANNEL_TILE+1)})
    $else:
      .input_stride(channel_tile+1)
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  ${TESTER}()
    .rows({1, ${INCREMENTAL_TILE*5 + 1}, ${INCREMENTAL_TILE}})
    .channels(channel_tile)
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  ${TESTER}()
    .rows({1, ${INCREMENTAL_TILE*5 + 1}, ${INCREMENTAL_TILE}})
    .channels(channel_tile)
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      .input_stride(${next_prime(CHANNEL_TILE+1)})
    $else:
      .input_stride(channel_tile+1)
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    ${TESTER}()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels({${CHANNEL_TILE*2}, ${CHANNEL_TILE*8}, ${CHANNEL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    ${TESTER}()
      .channels({${CHANNEL_TILE*2}, ${CHANNEL_TILE*8}, ${CHANNEL_TILE}})
      .rows({1, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    ${TESTER}()
      .channels({${CHANNEL_TILE*2}, ${CHANNEL_TILE*8}, ${CHANNEL_TILE}})
      .rows({1, ${INCREMENTAL_TILE*5 + 1}, ${INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, ${INCREMENTAL_TILE*5 + 1}, ${INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    ${TESTER}()
      .channels({${CHANNEL_TILE*2}, ${CHANNEL_TILE*8}, ${CHANNEL_TILE}})
      .rows({1, ${INCREMENTAL_TILE*5 + 1}, ${INCREMENTAL_TILE}})
      .input_stride(${next_prime(CHANNEL_TILE*16+1)})
      .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, ${INCREMENTAL_TILE*5 + 1}, ${INCREMENTAL_TILE}})
      .input_stride(channel_tile*16+1)
      .Test(${", ".join(TEST_ARGS)});
}

$if CHANNEL_TILE > 1 or CHANNEL_SCALED_TILE != CHANNEL_TILE:
  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels({1, channel_tile})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .channels({1, channel_tile})
      .rows({1, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .channels({1, channel_tile})
      .rows({1, ${INCREMENTAL_TILE*5 + 1}, ${INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile_with_input_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .channels({1, channel_tile})
      .rows({1, ${INCREMENTAL_TILE*5 + 1}, ${INCREMENTAL_TILE}})
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        .input_stride(${next_prime(CHANNEL_TILE+1)})
      $else:
        .input_stride(channel_tile+1)
      .Test(${", ".join(TEST_ARGS)});
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    ${TESTER}()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels({${CHANNEL_TILE+1}, ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}})
      .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels({channel_tile+1, channel_tile*2})
      .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    ${TESTER}()
      .channels({${CHANNEL_TILE+1}, ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}})
      .rows({1, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    ${TESTER}()
      .channels({${CHANNEL_TILE+1}, ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}})
      .rows({1, ${INCREMENTAL_TILE*5}, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, ${INCREMENTAL_TILE*5}, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    ${TESTER}()
      .channels({${CHANNEL_TILE+1}, ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}})
      .rows({1, ${INCREMENTAL_TILE*5}, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .input_stride(${next_prime(CHANNEL_TILE*2+11)})
      .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};

    ${TESTER}()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, ${INCREMENTAL_TILE*5}, ${PRIMARY_TILE+INCREMENTAL_TILE}})
      .input_stride(channel_tile*2+11)
      .Test(${", ".join(TEST_ARGS)});
}

$if TESTER == "RDSumMicrokernelTester":
  TEST(${TEST_NAME}, overflow_accumulator) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    ${TESTER}()
      .rows(${257 + INCREMENTAL_TILE})
      .channels({1, channel_tile*2})
      .Test(${", ".join(TEST_ARGS)});
  }
"""


def generate_test_cases(
    ukernel,
    op_type,
    init_fn,
    requantization_type,
    tester,
    primary_tile,
    incremental_tile,
    channel_tile,
    vector_tile,
    isa,
):
  """Generates all tests cases for a discontiguous reduce micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    op_type: Operation type (MAX/MIN/SUM/etc).
    init_fn: C name of the function to initialize microkernel parameters.
    requantization_type: Requantization type (FP32/RNDNU).
    tester: C++ name of the tester class.
    primary_tile: Number of rows (pixels) processed per one iteration of the
      primary outer loop of the micro-kernel.
    incremental_tile: Number of rows (pixels) processed per one iteration of the
      incremental outer loop of the micro-kernel.
    channel_tile: Number of channels processed per one iteration of the inner
      loops of the micro-kernel.
    vector_tile: Indicates if channels are specified in vectors rather than
      elements.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel]
  if tester == "ReduceMicrokernelTester":
    test_args.append("ReduceMicrokernelTester::OpType::%s" % op_type)
  if init_fn is not None:
    test_args.append(init_fn)
  if requantization_type:
    test_args.append(
        "xnn_%s_requantize_%s" % (datatype.lower(), requantization_type.lower())
    )
  channel_scaled_tile = channel_tile
  if vector_tile:
    ctype = {
        "qs8": "int8_t",
        "qu8": "uint8_t",
        "f16": "uint16_t",
        "f32": "float",
    }[datatype]
    channel_scaled_tile = {
        "rvv": (
            "(%s*xnn_init_hardware_config()->vlenb/sizeof(%s))"
            % (str(channel_tile), ctype)
        )
    }[isa]
  return xngen.preprocess(
      RDSUM_TEST_TEMPLATE,
      {
          "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
          "TEST_ARGS": test_args,
          "TESTER": tester,
          "DATATYPE": datatype,
          "PRIMARY_TILE": primary_tile,
          "INCREMENTAL_TILE": incremental_tile,
          "CHANNEL_TILE": channel_tile,
          "CHANNEL_SCALED_TILE": channel_scaled_tile,
          "CHANNEL_SUFFIX": "v" if vector_tile else "",
          "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
          "next_prime": next_prime,
      },
  )


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tester_header = {
        "ReduceMicrokernelTester": "reduce-microkernel-tester.h",
        "RDSumMicrokernelTester": "rdsum-microkernel-tester.h",
    }[options.tester]

    tests = """\
// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/reduce.h"
#include "test/{tester_header}"
""".format(
        specification=options.spec,
        generator=sys.argv[0],
        tester_header=tester_header,
    )

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      (
          requantization_type,
          op_type,
          primary_tile,
          incremental_tile,
          channel_tile,
          vector_tile,
          arch,
          isa,
      ) = split_ukernel_name(name)

      test_case = generate_test_cases(
          name,
          op_type,
          init_fn,
          requantization_type,
          options.tester,
          primary_tile,
          incremental_tile,
          channel_tile,
          vector_tile,
          isa,
      )
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
