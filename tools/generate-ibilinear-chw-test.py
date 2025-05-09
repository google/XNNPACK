#!/usr/bin/env python
# Copyright 2020 Google LLC
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
    description='IBILINEAR microkernel test generator')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["IBilinearMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", metavar="FILE", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


IBILINEAR_TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, pixel_tile, channel_tile, datatype, params_type, init_params) \
XNN_TEST_IBILINEAR_CHW_PIXELS_DIV(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHW_PIXELS_LT(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHW_PIXELS_GT(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHW_CHANNELS_DIV(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHW_CHANNELS_EQ(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHW_CHANNELS_GT(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHW_INPUT_OFFSET(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_IBILINEAR_CHW_INPUT_STRIDE(ukernel, arch_flags, ${", ".join(TEST_ARGS)});
"""

def main(args):
  options = parser.parse_args(args)
  tester = options.tester
  tester_header = {
  "IBilinearMicrokernelTester": "ibilinear-microkernel-tester.h",
  }[tester]
  ukernel = options.ukernel

  tests = """\

// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {ukernel}
//   Generator: {generator}


#include <gtest/gtest.h>

#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/isa-checks.h"
#include "ibilinear-microkernel-tester.h"
""".format(ukernel=options.ukernel, generator=sys.argv[0])

  ukernel_parts = options.ukernel.split("-")
  datatype = ukernel_parts[0]
  op = ukernel_parts[1]
  test_args = ["pixel_tile"]
  test_args.append("channel_tile")
  test_args.append("datatype")
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
  folder = datatype + "-" + ("ibilinear-chw" if datatype.startswith("f") else op)
  tests += f'#include "{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"
  xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
