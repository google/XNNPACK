#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import collections
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='XNNPACK generator')
parser.add_argument("-k", "--ukernel", required=True,
                    help="ukernel name")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())

def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  param_spec = common_parts[-1]

  # New transitional naming convention for unipass microkernels.
  m = re.search(r'(\d+)p(\d+)c', param_spec);
  assert(m)
  primary_tile = 0;
  cr = int(m[2])
  kr = int(m[1])
  arch, isa, assembly = xnncommon.parse_target_name(target_name)

  requantization = common_parts[-3]
  if requantization not in ["fp32", "rndnu"]:
    requantization = None

  return primary_tile, cr, kr, requantization, arch, isa, assembly


DWCONV_CREATE_TESTS_CODE = """\
std::vector<DWConvTestParams> CreateTests(
    size_t c_block, bool is_pipelined, size_t cr, size_t kr,
    std::function<void(DWConvMicrokernelTester& tester)> test_func) {
  size_t adj_c_block = is_pipelined ? c_block * 2 : c_block;
  const std::string cbs = std::to_string(c_block);
  const std::string acbs = std::to_string(adj_c_block);

  std::vector<DWConvTestParams> tests;
  tests.reserve(18);

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .channels(c_block)
      , test_func));

  if (is_pipelined) {
    tests.push_back(DWConvTestParams(
        "c_eq_" + std::to_string(c_block * 2),
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .channels(c_block * 2)
        , test_func));
  }

  if (c_block > 1) {
    tests.push_back(DWConvTestParams(
        "c_div_" + cbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
        , test_func)
        .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

    $if ACTIVATION == "MINMAX":
      tests.push_back(DWConvTestParams(
          "c_div_" + cbs + "_with_qmin",
          DWConvMicrokernelTester()
              .channel_tile(cr)
              .kernel_tile(kr)
              .qmin(128)
          , test_func)
          .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

      tests.push_back(DWConvTestParams(
          "c_div_" + cbs + "_with_qmax",
          DWConvMicrokernelTester()
              .channel_tile(cr)
              .kernel_tile(kr)
              .qmax(128)
          , test_func)
          .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_lt_" + acbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
        , test_func)
      .loop_channels(1, adj_c_block - 1));
  }

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
      , test_func)
      .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

  $if ACTIVATION == "MINMAX":
    tests.push_back(DWConvTestParams(
        "c_gt_" + acbs + "_with_qmin",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .qmin(128)
        , test_func)
        .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

    tests.push_back(DWConvTestParams(
        "c_gt_" + acbs + "_with_qmax",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .qmax(128)
        , test_func)
        .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

  tests.push_back(DWConvTestParams(
      "multipixel",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(3)
      , test_func)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "multipixel_with_step",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .width(3)
        , test_func)
        .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1))
        .loop_step(2, kr));

  tests.push_back(DWConvTestParams(
      "multipixel_with_output_stride",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(5)
          .output_stride(xnnpack::NextPrime(cr * 5 + 1))
      , test_func)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  $if ACTIVATION == "MINMAX":
    tests.push_back(DWConvTestParams(
        "multipixel_with_qmin",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .width(3)
            .qmin(128)
        , test_func)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

    tests.push_back(DWConvTestParams(
        "multipixel_with_qmax",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .width(3)
            .qmax(128)
        , test_func)
        .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  $if DATATYPE == "qu8":
    tests.push_back(DWConvTestParams(
        "input_zero_point_only",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .width(3)
            .input_zero_point(255)
            .kernel_zero_point(0)
        , test_func)
        .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

    tests.push_back(DWConvTestParams(
        "kernel_zero_point_only",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .width(3)
            .input_zero_point(0)
            .kernel_zero_point(255)
        , test_func)
        .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "input_offset",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  tests.push_back(DWConvTestParams(
      "zero",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func)
      .loop_zi(0, kr - 1)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  return tests;
}
"""

TEST_TEMPLATE = """\
#define XNN_DWCONV_UNIPASS(arch_flags, ukernel, c_block, is_pipelined, cr, kr, datatype, weights_type, params_type, init_params)
INSTANTIATE_TEST_SUITE_P(
    ukernel, DWConvTest,
    testing::ValuesIn(CreateTests(
        c_block, is_pipelined, cr, kr,
        [](DWConvMicrokernelTester& tester) {
          TEST_REQUIRES_ARCH_FLAGS(arch_flags);
          tester.Test(${", ".join(TEST_ARGS)});
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });
"""

def main(args):
  options = parser.parse_args(args)

  ukernel = options.ukernel

  test_header = """\
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {ukernel}
//   Generator: {generator}


#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/requantization.h"
#include "dwconv-microkernel-tester.h"
#include "next_prime.h"
""".format(ukernel=ukernel, generator=sys.argv[0])

  test_cases = ""

  parts = ukernel.split("-")
  datatype = parts[0]
  folder = datatype + "-dwconv"
  if parts[1] == "qc8w":
    folder = datatype + "-qc8w-dwconv"
    parts.pop(1)
  activation = "minmax" if "minmax" in parts else "linear"
  ukernel_type = "unipass" if "unipass" in parts else "multipass"
  requantization = "fp32" if "fp32" in parts else "rndnu" if "rndnu" in parts else None

  create_tests_args = {
      "UKERNEL_TYPE": ukernel_type.upper(),
      "DATATYPE": datatype,
      "ACTIVATION": activation.upper(),
  }
  create_tests = xngen.preprocess(DWCONV_CREATE_TESTS_CODE, create_tests_args)

  create_tests = (
      "namespace {\n\n"
      + "\n".join([create_tests])
      + "\n}  // namespace\n"
  )
  tests = test_header + "\n" + create_tests + "\n" + test_cases

  test_args = ["ukernel", "init_params"]
  if requantization:
    requantization_datatype = {"qc8": "qs8"}.get(datatype, datatype)
    test_args.append(
        "xnn_%s_requantize_%s" % (requantization_datatype, requantization)
    )

  tests += xnncommon.make_multiline_macro(xngen.preprocess(
      TEST_TEMPLATE,
      {
          "TEST_ARGS": test_args,
      },
  ))

  tests += f'#include "{xnncommon.xnnpack_src()}{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"

  xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
