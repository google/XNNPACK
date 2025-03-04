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
  description='Test generator for DWCONV2D CHW micro-kernels')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["DWConv2DMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", metavar="FILE", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params) \
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_EQ(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_DIV(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_LT(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_GT(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_EQ(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_DIV(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_LT(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_GT(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
XNN_TEST_DWCONV2D_OUTPUT_PADDING_TOP_EQ(arch_flags, ukernel, ${", ".join(TEST_ARGS)});
"""


def split_ukernel_name(name):
  match = re.fullmatch(
      r"xnn_(f16|f32)_dwconv2d_chw_ukernel_(\d+)x(\d+)(s2)?p(\d+)__(.+)_(\d+)x(\d+)(v)?(_acc\d+)?",
      name,
  )
  assert match is not None
  kernel_height, kernel_width = int(match.group(2)), int(match.group(3))
  if match.group(4):
    assert match.group(4).startswith("s")
    stride = int(match.group(4)[1:])
  else:
    stride = 1
  padding = int(match.group(5))

  height_tile = int(match.group(7))
  width_tile = int(match.group(8))

  if match.group(9):
    vector_tile = True
  else:
    vector_tile = False

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(6))
  return (
      kernel_height,
      kernel_width,
      stride,
      padding,
      arch,
      isa,
      height_tile,
      width_tile,
      vector_tile,
  )


def main(args):
  options = parser.parse_args(args)

  tester = options.tester
  tester_header = {
  "DWConv2DMicrokernelTester": "dwconv2d-microkernel-tester.h",
  }[tester]
  ukernel = options.ukernel

  tests = """\
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "dwconv2d-microkernel-tester.h"
""".format(specification=options.ukernel, generator=sys.argv[0])
  ukernel_parts = options.ukernel.split("-")
  datatype = ukernel_parts[0]
  op = ukernel_parts[1]
  test_args = ["_kernel_height_"]
  test_args.append("_kernel_width_")
  test_args.append("_subsampling_")
  test_args.append("_padding_")
  test_args.append("_height_tile_")
  test_args.append("_width_tile_")
  test_args.append("datatype")
  test_args.append("params_type")
  test_args.append("init_params")
  tests += xnncommon.make_multiline_macro(xngen.preprocess(
    TEST_TEMPLATE,
    {
        "TEST_ARGS": test_args,
        "TESTER": tester,
        "DATATYPE": datatype,
    },
  ))
  folder = datatype + "-" + ("dwconv2d-chw" if datatype.startswith("f") else op)
  tests += f'#include "{xnncommon.xnnpack_src()}{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"

  xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
