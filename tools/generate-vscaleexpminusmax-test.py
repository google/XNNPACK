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
import xngen
import xnncommon


parser = argparse.ArgumentParser(
  description='Vector ScaleExpMinusMax microkernel test generator')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["VScaleExpMinusMaxMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", metavar="FILE", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f32)_vscaleexpminusmax_ukernel__(.+)_u(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  elements_tile = int(match.group(3))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(2))
  return elements_tile, arch, isa


VSCALEEXPMINUSMAX_TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params) \
XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_EQ(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_DIV(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_LT(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_GT(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
XNN_TEST_VSCALEEXPMINUSMAX_SCALE(ukernel,arch_flags,  ${", ".join(TEST_ARGS)});
"""
def main(args):
  options = parser.parse_args(args)
  tester = options.tester
  tester_header = {
  "VScaleExpMinusMaxMicrokernelTester": "vscaleexpminusmax-microkernel-tester.h",
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
#include "xnnpack/vscaleexpminusmax.h"
#include "vscaleexpminusmax-microkernel-tester.h"
""".format(specification=options.ukernel, generator=sys.argv[0])

  ukernel_parts = options.ukernel.split("-")
  datatype = ukernel_parts[0]
  op = ukernel_parts[1]
  test_args = ["element_tile"]
  test_args.append("init_params")
  tests += xnncommon.make_multiline_macro(xngen.preprocess(
    VSCALEEXPMINUSMAX_TEST_TEMPLATE,
    {
        "TEST_ARGS": test_args,
        "TESTER": tester,
        "DATATYPE": datatype,
    },
  ))
  folder = datatype + "-" + ("vscaleexpminusmax" if datatype.startswith("f") else op)
  tests += f'#include "{xnncommon.xnnpack_src()}{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"
  xnncommon.overwrite_if_changed(options.output, tests)

if __name__ == "__main__":
  main(sys.argv[1:])
