#!/usr/bin/env python
# Copyright 2019 Google LLC
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
  description='VMulCAddC microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f32)_vmulcaddc(_(minmax))?_ukernel_c(\d+)__(.+)_(\d+)x", name)
  assert match is not None
  channel_tile = int(match.group(4))
  row_tile = int(match.group(6))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(5))
  return channel_tile, row_tile, arch, isa


VMULCADDC_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  VMulCAddCMicrokernelTester()
    .channel_tile(${CHANNEL_TILE})
    .channels(${CHANNEL_TILE})
    .rows(${ROW_TILE})
    .Test(${", ".join(TEST_ARGS)});
}

$if CHANNEL_TILE > 1:
  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*10}; channels += ${CHANNEL_TILE}) {
      VMulCAddCMicrokernelTester()
        .channel_tile(${CHANNEL_TILE})
        .channels(channels)
        .rows(${ROW_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(${CHANNEL_TILE})
        .channels(channels)
        .rows(${ROW_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    VMulCAddCMicrokernelTester()
      .channel_tile(${CHANNEL_TILE})
      .channels(channels)
      .rows(${ROW_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if ROW_TILE > 1:
  TEST(${TEST_NAME}, rows_lt_${ROW_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = 1; rows < ${ROW_TILE}; rows++) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
        VMulCAddCMicrokernelTester()
          .channel_tile(${CHANNEL_TILE})
          .channels(channels)
          .rows(rows)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, rows_div_${ROW_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = ${ROW_TILE*2}; rows <= ${ROW_TILE*4}; rows += ${ROW_TILE}) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
        VMulCAddCMicrokernelTester()
          .channel_tile(${CHANNEL_TILE})
          .channels(channels)
          .rows(rows)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, rows_gt_${ROW_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = ${ROW_TILE+1}; rows < ${ROW_TILE*2}; rows++) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      VMulCAddCMicrokernelTester()
        .channel_tile(${CHANNEL_TILE})
        .channels(channels)
        .rows(rows)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      VMulCAddCMicrokernelTester()
        .channel_tile(${CHANNEL_TILE})
        .channels(channels)
        .rows(rows)
        .input_stride(${next_prime(CHANNEL_TILE*5+1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      VMulCAddCMicrokernelTester()
        .channel_tile(${CHANNEL_TILE})
        .channels(channels)
        .rows(rows)
        .output_stride(${next_prime(CHANNEL_TILE*5+1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      VMulCAddCMicrokernelTester()
        .channel_tile(${CHANNEL_TILE})
        .channels(channels)
        .rows(rows)
        .inplace(true)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      VMulCAddCMicrokernelTester()
        .channel_tile(${CHANNEL_TILE})
        .channels(channels)
        .rows(rows)
        .qmin(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      VMulCAddCMicrokernelTester()
        .channel_tile(${CHANNEL_TILE})
        .channels(channels)
        .rows(rows)
        .qmax(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}
"""


def generate_test_cases(ukernel, channel_tile, row_tile, init_fn, isa):
  """Generates all tests cases for a VMULCADDC micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    channel_tile: Number of channels processed per one iteration of the inner
                  loop of the micro-kernel.
    row_tile: Number of rows processed per one iteration of the outer loop of
              the micro-kernel.
    init_fn: C name of the function to initialize microkernel parameters.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel]
  if init_fn:
    test_args.append(init_fn)
  return xngen.preprocess(VMULCADDC_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "DATATYPE": datatype,
      "CHANNEL_TILE": channel_tile,
      "ROW_TILE": row_tile,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spechannels_file:
    spechannels_yaml = yaml.safe_load(spechannels_file)
    if not isinstance(spechannels_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2019 Google LLC
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
#include "xnnpack/vmulcaddc.h"
#include "vmulcaddc-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spechannels_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      channel_tile, row_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(
        name, channel_tile, row_tile, init_fn, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
