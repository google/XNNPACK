#!/usr/bin/env python
# Copyright 2019 Google LLC
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
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(
<<<<<<< HEAD
    description="IBILINEAR microkernel test generator"
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
      r"xnn_(f16|f32|s8|u8)_ibilinear_ukernel__(.+)_c(\d+)", name
  )
  assert match is not None
  channel_tile = int(match.group(3))
  pixel_tile = 1

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(2))
  return channel_tile, pixel_tile, arch, isa


=======
  description='IBILINEAR microkernel test generator')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["IBilinearMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", metavar="FILE", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


>>>>>>> c2bcc7bd5 (Replace ibilinear yaml with table header)
IBILINEAR_TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, pixel_tile, datatype, weight_type, params_type, init_params) \
XNN_TEST_IBILINEAR_CHANNELS_EQ(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHANNELS_DIV(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHANNELS_LT(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHANNELS_GT(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_PIXELS_DIV(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_PIXELS_LT(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_PIXELS_GT(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_INPUT_OFFSET(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_OUTPUT_STRIDE(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
"""

<<<<<<< HEAD

def generate_test_cases(ukernel, channel_tile, pixel_tile, isa):
  """Generates all tests cases for a BILINEAR micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    channel_tile: Number of channels processed per one iteration of the inner
      loop of the micro-kernel.
    pixel_tile: Number of pixels processed per one iteration of the outer loop
      of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel]
  return xngen.preprocess(
      IBILINEAR_TEST_TEMPLATE,
      {
          "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
          "TEST_FUNC": ukernel,
          "UKERNEL_TYPE": ukernel_type.upper(),
          "DATATYPE": datatype,
          "CHANNEL_TILE": channel_tile,
          "PIXEL_TILE": pixel_tile,
          "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
          "next_prime": next_prime,
      },
  )


=======
>>>>>>> c2bcc7bd5 (Replace ibilinear yaml with table header)
def main(args):
  options = parser.parse_args(args)
  tester = options.tester
  tester_header = {
  "IBilinearMicrokernelTester": "ibilinear-microkernel-tester.h",
  }[tester]
  ukernel = options.ukernel

<<<<<<< HEAD
  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// clang-format off
=======
  tests = """\
>>>>>>> c2bcc7bd5 (Replace ibilinear yaml with table header)
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {ukernel}
//   Generator: {generator}


#include <gtest/gtest.h>
<<<<<<< HEAD
#include "src/xnnpack/common.h"
#include "src/xnnpack/ibilinear.h"
#include "src/xnnpack/isa-checks.h"
#include "test/ibilinear-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])
=======
#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/isa-checks.h"
#include "ibilinear-microkernel-tester.h"
""".format(ukernel=options.ukernel, generator=sys.argv[0])
>>>>>>> c2bcc7bd5 (Replace ibilinear yaml with table header)

  ukernel_parts = options.ukernel.split("-")
  datatype = ukernel_parts[0]
  op = ukernel_parts[1]
  test_args = ["channel_tile"]
  test_args.append("pixel_tile")
  test_args.append("datatype")
  test_args.append("weight_type")
  test_args.append("params_type")
  test_args.append("init_params")
  tests += xnncommon.make_multiline_macro(xngen.preprocess(
    IBILINEAR_TEST_TEMPLATE,
    {
        "TEST_ARGS": test_args,
        "TESTER": tester,
        "DATATYPE": datatype,
    },
  ))
  folder = datatype + "-" + ("ibilinear" if datatype.startswith("f") else op)
  tests += f'#include "{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"
  xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
