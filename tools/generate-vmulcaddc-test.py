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
  description='VMULCADDC microkernel test generator')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["VMulCAddCMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", metavar="FILE", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f32)_vmulcaddc_ukernel__(.+)_u(\d+)(_acc(\d+))?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  elements_tile = int(match.group(3))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(2))
  return elements_tile, arch, isa


VMULCADDC_TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile,channel_tile, datatype, params_type, init_params) \
XNN_TEST_VMULCADDC_ROW_DIV(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_ROW_LT(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_ROW_GT(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_CHANNEL_GT(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_CHANNEL_EQ(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_VMULCADDC_CHANNEL_DIV(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_CHANNEL_LT(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_INPUT_STRIDE(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_OUTPUT_STRIDE(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_INPLACE(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_QMAX(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VMULCADDC_QMIN(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
"""

def main(args):
    options = parser.parse_args(args)
    tester = options.tester
    tester_header = {
    "VMulCAddCMicrokernelTester": "vmulcaddc-microkernel-tester.h",
    }[tester]
    ukernel = options.ukernel

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
#include "xnnpack/vmulcaddc.h"
#include "xnnpack/microparams-init.h"
#include "vmulcaddc-microkernel-tester.h"
#include "next_prime.h"
""".format(specification=options.ukernel, generator=sys.argv[0])
    ukernel_parts = options.ukernel.split("-")
    datatype = ukernel_parts[0]
    op = ukernel_parts[1]
    test_args = ["row_tile"]
    test_args.append("channel_tile")
    test_args.append("datatype")
    test_args.append("params_type")
    test_args.append("init_params")
    print("test args",test_args)
    tests += xnncommon.make_multiline_macro(xngen.preprocess(
      VMULCADDC_TEST_TEMPLATE,
      {
          "TEST_ARGS": test_args,
          "TESTER": tester,
          "DATATYPE": datatype,
      },
  ))
    folder = datatype + "-" + ("vmulcaddc" if datatype.startswith("f") else op)
    print("options",options.ukernel)
    tests += f'#include "{xnncommon._XNNPACK_SRC}{folder}/{options.ukernel}.h"\n'
    tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"

    xnncommon.overwrite_if_changed(options.output, tests)

if __name__ == "__main__":
  main(sys.argv[1:])
