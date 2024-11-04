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
  description='Vector binary operation microkernel test generator')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=[
                    "VCMulMicrokernelTester",
                    "VBinaryMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-b", "--broadcast_b", action="store_true",
                    help='Broadcast the RHS of the operation')
parser.add_argument("-k", "--ukernel", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())

OP_TYPES = {
    "vadd": "Add",
    "vaddc": "Add",
    "vcopysign": "CopySign",
    "vcopysignc": "CopySign",
    "vrcopysign": "RCopySign",
    "vrcopysignc": "RCopySign",
    "vdiv": "Div",
    "vdivc": "Div",
    "vrdiv": "RDiv",
    "vrdivc": "RDiv",
    "vmax": "Max",
    "vmaxc": "Max",
    "vmin": "Min",
    "vminc": "Min",
    "vmul": "Mul",
    "vmulc": "Mul",
    "vcmul": "CMul",
    "vsub": "Sub",
    "vsubc": "Sub",
    "vrsub": "RSub",
    "vrsubc": "RSub",
    "vsqrdiff": "SqrDiff",
    "vsqrdiffc": "SqrDiff",
    "vprelu": "Prelu",
    "vpreluc": "Prelu",
    "vrpreluc": "RPrelu",
}

BINOP_TEST_TEMPLATE = """
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)
XNN_TEST_BINARY_BATCH_EQ(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
XNN_TEST_BINARY_BATCH_DIV(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
XNN_TEST_BINARY_BATCH_LT(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
XNN_TEST_BINARY_BATCH_GT(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});

$if TESTER in ["VMulCMicrokernelTester"]:
  XNN_TEST_BINARY_INPLACE_A(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
$elif ${BROADCAST_B} == "true":
  XNN_TEST_BINARY_INPLACE_A(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
$else:
  XNN_TEST_BINARY_INPLACE_A(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_BINARY_INPLACE_B(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_BINARY_INPLACE_A_AND_B(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});

$if DATATYPE.startswith("q"):
  XNN_TEST_BINARY_A_ZERO_POINT(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_BINARY_B_ZERO_POINT(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_BINARY_Y_ZERO_POINT(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_BINARY_A_SCALE(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_BINARY_B_SCALE(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_BINARY_Y_SCALE(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});

$if "minmax" in ACTIVATION_TYPE:
  XNN_TEST_BINARY_QMIN(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_BINARY_QMAX(ukernel, arch_flags, batch_tile, ${BROADCAST_B}, datatype, ${", ".join(TEST_ARGS)});
"""

def main(args):
  options = parser.parse_args(args)

  tester = options.tester
  tester_header = {
    "VCMulMicrokernelTester": "vcmul-microkernel-tester.h",
    "VBinaryMicrokernelTester": "vbinary-microkernel-tester.h",
  }[tester]
  tests = """\
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {microkernel}
//   Generator: {generator}


#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
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
  if tester in ["VBinaryMicrokernelTester"] and not datatype in ['qs8', 'qu8']:
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
  tests += f'#include "{xnncommon.xnnpack_src()}{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"
  tests = tests.replace("s32-vmulc/s32-vmulc.h", "s32-vmul/s32-vmulc.h")

  xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
