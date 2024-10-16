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
import xngen
import xnncommon


parser = argparse.ArgumentParser(
  description='RWD microkernel test generator')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["RWDMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


OP_TYPES = {
  "rwdsum": "Sum",
}

BINOP_TEST_TEMPLATE = """
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, void, init_params) 
XNN_TEST_RWD_CHANNEL_EQ_ROW_EQ(ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_RWD_CHANNEL_EQ_ROW_GT(ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_RWD_CHANNEL_GT_ROW_EQ(ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_RWD_CHANNEL_GT_ROW_GT(ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_RWD_CHANNEL_EQ_ROW_258(ukernel, ${", ".join(TEST_ARGS)});
"""

def main(args):
  options = parser.parse_args(args)

  tester = options.tester
  tester_header = {
    "RWDMicrokernelTester": "rwd-microkernel-tester.h",
  }[tester]
  
  tests = """\
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {microkernel}
//   Generator: {generator}


#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "{tester_header}"
""".format(
  microkernel=options.ukernel,
  generator=sys.argv[0],
  tester_header=tester_header,
  )

  ukernel_parts = options.ukernel.split("-")
  datatype = ukernel_parts[0]
  op = ukernel_parts[1]
  activation = ukernel_parts[2] if len(ukernel_parts) >= 3 else ""
  broadcast_b = False
  if op[-1] == 'c':
    broadcast_b = True
  op_type = OP_TYPES[op]
  test_args = ["ukernel"]
  if tester in ["RWDMicrokernelTester"]:
    test_args.append("%s::OpType::%s" % (tester, op_type))
  test_args.append("init_params")
  tests += xnncommon.make_multiline_macro(xngen.preprocess(
      BINOP_TEST_TEMPLATE,
      {
          "TEST_ARGS": test_args,
          "TESTER": tester,
          "BROADCAST_B": str(broadcast_b).lower(),
          "DATATYPE": datatype,
          "OP_TYPE": op_type,
          "ACTIVATION_TYPE": activation,
      },
  ))
  folder = datatype + "-" + ("vbinary" if datatype.startswith("f") else op)
  tests += f'#include "{xnncommon._XNNPACK_SRC}/{options.ukernel}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"
  xnncommon.overwrite_if_changed(options.output, tests)
  

if __name__ == "__main__":
  main(sys.argv[1:])
