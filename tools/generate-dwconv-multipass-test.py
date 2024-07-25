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
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Spec (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())

def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  param_spec = common_parts[-1]

  m = re.search(r'(\d+)f(\d+)m(\d+)l(\d+)c(\d+)s(\d+)r', param_spec)
  assert m
  first_pass_tile = int(m[1])
  middle_pass_tile = int(m[2])
  last_pass_tile = int(m[3])
  channel_tile = int(m[4])
  channel_subtile = int(m[5])
  channel_round = int(m[6])
  arch, isa, assembly = xnncommon.parse_target_name(target_name)

  requantization = common_parts[-3]
  if requantization not in ["fp32", "rndnu"]:
    requantization = None

  return (first_pass_tile, middle_pass_tile, last_pass_tile, channel_tile, channel_subtile, channel_round, requantization, arch, isa, assembly)


DWCONV_CREATE_TESTS_CODE = """\
std::vector<DWConvTestParams> CreateTests(
    size_t c_block, size_t adj_c_block, size_t cr, size_t kr,
    size_t first_pass_tile, size_t middle_pass_tile, size_t last_pass_tile,
    size_t channel_subtile, size_t channel_round,
    std::function<void(DWConvMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  const std::string cbs = std::to_string(c_block);
  const std::string acbs = std::to_string(adj_c_block);

  std::vector<DWConvTestParams> tests;
  tests.reserve(17);

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_first_pass_plus_one",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + 1)
          .channels(c_block)
      , test_func, isa_check));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_first_pass_and_last_pass",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + last_pass_tile)
          .channels(c_block)
      , test_func, isa_check));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_multipass",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .channels(c_block)
      , test_func, isa_check)
      .loop_kernel_size(
          first_pass_tile + middle_pass_tile + last_pass_tile,
          first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  if (c_block > 1) {
    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_first_pass_plus_one",
        DWConvMicrokernelTester()
            .first_pass_tile(first_pass_tile)
            .middle_pass_tile(middle_pass_tile)
            .last_pass_tile(last_pass_tile)
            .channel_tile(cr)
            .channel_subtile(channel_subtile)
            .channel_round(channel_round)
            .kernel_size(first_pass_tile + 1)
        , test_func, isa_check)
        .loop_channels(adj_c_block = c_block, cr * 16, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_first_pass_and_last_pass",
        DWConvMicrokernelTester()
            .first_pass_tile(first_pass_tile)
            .middle_pass_tile(middle_pass_tile)
            .last_pass_tile(last_pass_tile)
            .channel_tile(cr)
            .channel_subtile(channel_subtile)
            .channel_round(channel_round)
            .kernel_size(first_pass_tile + last_pass_tile)
        , test_func, isa_check)
      .loop_channels(adj_c_block + c_block, cr * 16, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_multipass",
        DWConvMicrokernelTester()
            .first_pass_tile(first_pass_tile)
            .middle_pass_tile(middle_pass_tile)
            .last_pass_tile(last_pass_tile)
            .channel_tile(cr)
            .channel_subtile(channel_subtile)
            .channel_round(channel_round)
        , test_func, isa_check)
        .loop_channels(adj_c_block + c_block, cr * 16, cr * 3)
        .loop_kernel_size(
            first_pass_tile + middle_pass_tile + last_pass_tile,
            first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

    $if ACTIVATION == "MINMAX":
      tests.push_back(DWConvTestParams(
          "c_div_" + cbs + "_with_qmin",
          DWConvMicrokernelTester()
              .first_pass_tile(first_pass_tile)
              .middle_pass_tile(middle_pass_tile)
              .last_pass_tile(last_pass_tile)
              .channel_tile(cr)
              .channel_subtile(channel_subtile)
              .channel_round(channel_round)
              .kernel_size(first_pass_tile + last_pass_tile)
              .qmin(128)
          , test_func, isa_check)
          .loop_channels(adj_c_block + c_block, cr * 16, cr * 3));

      tests.push_back(DWConvTestParams(
          "c_div_" + cbs + "_with_qmax",
          DWConvMicrokernelTester()
              .first_pass_tile(first_pass_tile)
              .middle_pass_tile(middle_pass_tile)
              .last_pass_tile(last_pass_tile)
              .channel_tile(cr)
              .channel_subtile(channel_subtile)
              .channel_round(channel_round)
              .kernel_size(first_pass_tile + last_pass_tile)
              .qmax(128)
          , test_func, isa_check)
          .loop_channels(adj_c_block + c_block, cr * 16, cr * 3));
  }

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_first_pass_plus_one",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + 1)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, c_block == 1 ? 10 : adj_c_block + c_block));

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_first_pass_and_last_pass",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + last_pass_tile)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, c_block == 1 ? 10 : adj_c_block + c_block));

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_multipass",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, c_block == 1 ? 10 : adj_c_block + c_block)
      .loop_kernel_size(
          first_pass_tile + middle_pass_tile + last_pass_tile,
          first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_first_pass_plus_one_multipixel",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + 1)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_first_pass_and_last_pass_multipixel",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + last_pass_tile)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_multipass_multipixel",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1))
      .loop_kernel_size(
          first_pass_tile + middle_pass_tile + last_pass_tile,
          first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  tests.push_back(DWConvTestParams(
      "multipixel_with_step",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1))
      .loop_kernel_size(
          first_pass_tile + middle_pass_tile + last_pass_tile,
          first_pass_tile + middle_pass_tile * 2 + last_pass_tile)
      .loop_step(2, kr + 1));

  tests.push_back(DWConvTestParams(
      "multipixel_with_output_stride",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .width(5)
          .output_stride(xnnpack::NextPrime(cr * 5 + 1))
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1))
      .loop_kernel_size(first_pass_tile + middle_pass_tile + last_pass_tile,
                        first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  tests.push_back(DWConvTestParams(
      "input_offset",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_channels(adj_c_block + c_block, cr * 16, cr * 3)
      .loop_kernel_size(first_pass_tile + middle_pass_tile + last_pass_tile,
                        first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  return tests;
}
"""

DWCONV_TEST_CODE = """\
INSTANTIATE_TEST_SUITE_P(
    ${TEST_NAME}, DWConvTest,
    testing::ValuesIn(CreateTests(
        /*c_block=*/${CBLOCK}, /*adj_c_block=*/${ADJCBLOCK}, /*cr=*/${CR}, /*kr=*/${KR},
        /*first_pass_tile=*/${FIRST_PASS_TILE}, /*middle_pass_tile=*/${MIDDLE_PASS_TILE}, /*last_pass_tile=*/${LAST_PASS_TILE},
        /*channel_subtile=*/${CHANNEL_SUBTILE}, /*channel_round=*/${CHANNEL_ROUND},
        [](DWConvMicrokernelTester& tester) {
          tester.Test(${",\\n                      ".join(TEST_ARGS)});
        $if ISA_CHECK:
          },
          []() {
            ${ISA_CHECK};
          })),
        $else:
          })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });
"""


def generate_test_cases(ukernel, first_pass_tile, middle_pass_tile, last_pass_tile, cr, c_block,
                        channel_subtile, channel_round, init_fn, requantization, is_pipelined, isa):
  """Generates all tests cases for a DWCONV micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    cr: CR parameter of the DWCONV micro-kernel.
    channel_subtile: channel_subtile parameter of the DWCONV micro-kernel.
    channel_round: channel_round parameter of the DWCONV micro-kernel.
    kr: KR parameter of the DWCONV micro-kernel.
    k_block: Number of C values processed per one iteration of the main loop of
             the micro-kernel.
    init_fn: C name of the function to initialize microkernel parameters.
    requantization: name of the requantization scheme used by the microkernel.
    is_pipelined: Indicates if the micro-kernel is implemented with software
                  pipelining. Additional test cases are generated for software
                  pipelined micro-kernels to separately test prologue + epiloque
                  of the pipelined loop and iteration of the pipelined loop.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  kr = first_pass_tile
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, activation, _ = ukernel.split("_", 4)
  if activation == "ukernel":
    activation = "linear"
  test_args = [ukernel]
  if init_fn:
    test_args.append(init_fn)
    if requantization:
      requantization_datatype = {"qc8": "qs8"}.get(datatype, datatype)
      test_args.append(
          "xnn_%s_requantize_%s" % (requantization_datatype, requantization)
      )

  args = {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "UKERNEL_TYPE": ukernel_type.upper(),
      "DATATYPE": datatype,
      "ACTIVATION": activation.upper(),
      "FIRST_PASS_TILE": first_pass_tile,
      "MIDDLE_PASS_TILE": middle_pass_tile,
      "LAST_PASS_TILE": last_pass_tile,
      "CR": cr,
      "CHANNEL_SUBTILE": channel_subtile,
      "CHANNEL_ROUND": channel_round,
      "KR": kr,
      "CBLOCK": c_block,
      "ADJCBLOCK": 2 * c_block if is_pipelined else c_block,
      "IS_PIPELINED": is_pipelined,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
      "sqrt": math.sqrt,
  }

  return (
      xngen.preprocess(DWCONV_CREATE_TESTS_CODE, args),
      xngen.preprocess(DWCONV_TEST_CODE, args),
  )


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    test_header = """\
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
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
""".format(specification=options.spec, generator=sys.argv[0])

    # Cached `CreateTests` functions.
    idx_from_create_tests_hash = collections.defaultdict(
        lambda: len(idx_from_create_tests_hash) + 1
    )
    create_tests_from_idx = {}

    test_cases = ""

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      pipelined = bool(ukernel_spec.get("pipelined", False))
      first_pass_tile, middle_pass_tile, last_pass_tile, cr, channel_subtile, channel_round, requantization, arch, isa, assembly = split_ukernel_name(name)

      create_tests, test_case = generate_test_cases(
          name,
          first_pass_tile,
          middle_pass_tile,
          last_pass_tile,
          cr,
          cr,
          channel_subtile,
          channel_round,
          init_fn,
          requantization,
          pipelined,
          isa,
      )

      # Store or reuse the `CreateTests` function?
      create_tests_hash = hash(create_tests)
      create_tests_idx = idx_from_create_tests_hash[create_tests_hash]
      if create_tests_idx not in create_tests_from_idx:
        create_tests_from_idx[create_tests_idx] = create_tests.replace(
            "CreateTests(", f"CreateTests{create_tests_idx}("
        )
      test_case = test_case.replace(
          "CreateTests(", f"CreateTests{create_tests_idx}("
      )

      test_cases += "\n\n" + xnncommon.postprocess_test_case(
          test_case, arch, isa, assembly
      )

    create_tests = (
        "namespace {\n\n"
        + "\n".join(create_tests_from_idx.values())
        + "\n}  // namespace\n"
    )
    tests = test_header + "\n" + create_tests + test_cases
    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
