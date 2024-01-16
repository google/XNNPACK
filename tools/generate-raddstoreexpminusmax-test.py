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
  description='RAddStoreExpMinusMax microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f32)_raddstoreexpminusmax_ukernel__(.+)_u(\d+)(v)?(_acc(\d+))?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  elements_tile = int(match.group(3))
  vector_tile = bool(match.group(4))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(2))
  return elements_tile, vector_tile, arch, isa


RADDSTOREEXPMINUSMAX_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, elements_eq_${ELEMENTS_TILE}${ELEMENTS_SUFFIX}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(${ELEMENTS_TILE}${ELEMENTS_SCALE})
    .Test(${TEST_FUNCTION}, ${INIT_FUNCTION});
}

$if ELEMENTS_TILE > 1:
  TEST(${TEST_NAME}, elements_div_${ELEMENTS_TILE}${ELEMENTS_SUFFIX}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if ELEMENTS_SCALE == "":
      for (size_t elements = ${ELEMENTS_TILE*2}; elements < ${ELEMENTS_TILE*10}; elements += ${ELEMENTS_TILE}) {
        RAddStoreExpMinusMaxMicrokernelTester()
          .elements(elements)
          .Test(${TEST_FUNCTION}, ${INIT_FUNCTION});
      }
    $else:
      for (size_t elements = ${ELEMENTS_TILE*2}${ELEMENTS_SCALE};
                  elements < ${ELEMENTS_TILE*10}${ELEMENTS_SCALE};
                  elements += ${ELEMENTS_TILE}${ELEMENTS_SCALE}) {
        RAddStoreExpMinusMaxMicrokernelTester()
          .elements(elements)
          .Test(${TEST_FUNCTION}, ${INIT_FUNCTION});
      }
  }

  TEST(${TEST_NAME}, elements_lt_${ELEMENTS_TILE}${ELEMENTS_SUFFIX}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if ELEMENTS_SCALE == "":
      for (size_t elements = 1; elements < ${ELEMENTS_TILE}; elements++) {
        RAddStoreExpMinusMaxMicrokernelTester()
          .elements(elements)
          .Test(${TEST_FUNCTION}, ${INIT_FUNCTION});
      }
    $else:
      for (size_t elements = 1;
                  elements < ${ELEMENTS_TILE}${ELEMENTS_SCALE};
                  elements++) {
        RAddStoreExpMinusMaxMicrokernelTester()
          .elements(elements)
          .Test(${TEST_FUNCTION}, ${INIT_FUNCTION});
      }
  }

TEST(${TEST_NAME}, elements_gt_${ELEMENTS_TILE}${ELEMENTS_SUFFIX}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if ELEMENTS_SCALE == "":
    for (size_t elements = ${ELEMENTS_TILE+1}; elements < ${10 if ELEMENTS_TILE == 1 else ELEMENTS_TILE*2}; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(${TEST_FUNCTION}, ${INIT_FUNCTION});
    }
  $else:
    for (size_t elements = ${ELEMENTS_TILE}${ELEMENTS_SCALE} + 1;
                elements < ${10 if ELEMENTS_TILE == 1 else ELEMENTS_TILE*2}${ELEMENTS_SCALE};
                elements += ${ELEMENTS_TILE*2}) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(${TEST_FUNCTION}, ${INIT_FUNCTION});
    }
}
"""


def generate_test_cases(ukernel, init_fn, elements_tile, vector_tile, isa):
  """Generates all tests cases for a RAddStoreExpMinusMax micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    elements_tile: Number of batch elements processed per one iteration of the
                   inner loop of the micro-kernel.
    vector_tile: Indicates if elements_tile is specified in vectors rather than
                 elements.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, _ = ukernel.split("_", 2)
  elements_scale = ""
  if vector_tile:
    ctype = {"f16": "uint16_t", "f32": "float"}[datatype]
    elements_scale = {
      "rvv": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype,
      "rvvfp16arith": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype,
    }[isa]


  return xngen.preprocess(RADDSTOREEXPMINUSMAX_TEST_TEMPLATE, {
      "TEST_FUNCTION": ukernel,
      "INIT_FUNCTION": init_fn,
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "DATATYPE": datatype,
      "ELEMENTS_TILE": elements_tile,
      "ELEMENTS_SCALE": elements_scale,
      "ELEMENTS_SUFFIX": "v" if vector_tile else "",
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
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

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/raddstoreexpminusmax.h>
#include "raddstoreexpminusmax-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      elements_tile, vector_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, init_fn, elements_tile, vector_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
