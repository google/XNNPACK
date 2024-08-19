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


parser = argparse.ArgumentParser(description='RDSum microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.match(r"xnn_(qs8|f16_f32acc|f32)_rdsum?(_minmax)?(_(fp32|rndnu))?_ukernel_((\d+)p)?(\d+)x__(.+)_c(\d+)(_acc(\d+))?", name)

  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  requantization_type = match.group(4)
  primary_tile = int(match.group(6))
  incremental_tile = int(match.group(7))
  channel_tile = int(match.group(9))
  target_name = match.group(8)

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(8))
  return requantization_type, primary_tile, incremental_tile, channel_tile, arch, isa


RDSUM_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_2pass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  RDSumMicrokernelTester()
    .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
    .channels(${CHANNEL_TILE})
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_2pass_fulltile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  RDSumMicrokernelTester()
    .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
    .channels(${CHANNEL_TILE})
    .input_stride(${next_prime(CHANNEL_TILE+1)})
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_2pass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(${CHANNEL_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_2pass_subtile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(${CHANNEL_TILE})
      .input_stride(${next_prime(CHANNEL_TILE+1)})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_multipass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(${CHANNEL_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_multipass_fulltile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(${CHANNEL_TILE})
      .input_stride(${next_prime(CHANNEL_TILE+1)})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_2pass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
    RDSumMicrokernelTester()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels(channels)
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_2pass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
    for (size_t rows = 1; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_multipass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
    for (size_t rows = 1; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_multipass_fulltile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
    for (size_t rows = 1; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(${next_prime(CHANNEL_TILE*16+1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

$if CHANNEL_TILE > 1:
  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_2pass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      RDSumMicrokernelTester()
        .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_2pass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      for (size_t rows = 1; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_multipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      for (size_t rows = 1; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_multipass_fulltile_with_input_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      for (size_t rows = 1; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(${next_prime(CHANNEL_TILE+1)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_2pass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    RDSumMicrokernelTester()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels(channels)
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_2pass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    for (size_t rows = 1; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_multipass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    for (size_t rows = 1; rows < ${INCREMENTAL_TILE*5}; rows += ${PRIMARY_TILE+INCREMENTAL_TILE}) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_multipass_fulltile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    for (size_t rows = 1; rows < ${INCREMENTAL_TILE*5}; rows += ${PRIMARY_TILE+INCREMENTAL_TILE}) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(${next_prime(CHANNEL_TILE*2+11)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, overflow_accumulator) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = 1; channels < ${CHANNEL_TILE * 2}; ++channels) {
     RDSumMicrokernelTester()
       .rows(${257 + INCREMENTAL_TILE})
       .channels(channels)
       .Test(${", ".join(TEST_ARGS)});
  }
}
"""


def generate_test_cases(ukernel, init_fn, requantization_type, primary_tile,
                        incremental_tile, channel_tile, isa):
  """Generates all tests cases for a RDSUM micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    requantization_type: Requantization type (FP32/RNDNU).
    primary_tile: Number of rows (pixels) processed per one iteration of the
                  primary outer loop of the micro-kernel.
    incremental_tile: Number of rows (pixels) processed per one iteration of
                      the incremental outer loop of the micro-kernel.
    channel_tile: Number of channels processed per one iteration of the inner
                  loops of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel]
  if init_fn is not None:
    test_args.append(init_fn)
  if requantization_type:
    test_args.append("xnn_%s_requantize_%s" % \
      (datatype.lower(), requantization_type.lower()))
  return xngen.preprocess(RDSUM_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "DATATYPE": datatype,
      "PRIMARY_TILE": primary_tile,
      "INCREMENTAL_TILE": incremental_tile,
      "CHANNEL_TILE": channel_tile,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "rdsum-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      requantization_type, primary_tile, incremental_tile, channel_tile, arch, \
        isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, init_fn, requantization_type,
                                      primary_tile, incremental_tile,
                                      channel_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])

