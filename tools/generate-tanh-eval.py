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


parser = argparse.ArgumentParser(description='Tanh evaluation generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def parse_eval_stub_name(name):
  match = re.fullmatch(r"xnn_math_(f16|f32)_tanh__(.+)?", name)
  if match is None:
    raise ValueError("Unexpected evaluation stub name: " + name)

  arch, isa, _ = xnncommon.parse_target_name(target_name=match.group(2))
  return match.group(1), arch, isa


TEST_TEMPLATE = """\
TEST(${TEST_NAME}, positive_saturation) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  MathEvaluationTester()
    .input_range(${SATURATION_LIMIT}f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(${TEST_FUNCTION}, 1.0f);
}

TEST(${TEST_NAME}, negative_saturation) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -${SATURATION_LIMIT}f)
    .TestOutputMatchReference(${TEST_FUNCTION}, -1.0f);
}

TEST(${TEST_NAME}, nan) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  MathEvaluationTester()
    .TestNaN(${TEST_FUNCTION});
}
"""


def generate_test_cases(eval_stub, datatype, isa):
  """Generates all tests cases for a Tanh evaluation stub.

  Args:
    eval_stub: C name of the evaluation stub function.
    datatype: input/output data type abbreviation (f16/f32).
    isa: instruction set required to run the evaluation stub. Generated tests
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  return xngen.preprocess(TEST_TEMPLATE, {
      "TEST_NAME": eval_stub.replace("xnn_math_%s_" % datatype, "").upper(),
      "TEST_FUNCTION": eval_stub,
      "DATATYPE": datatype,
      "SATURATION_LIMIT": {"f16": "0x1.208p+2", "f32": "0x1.205968p+3"}[datatype],
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of evaluation stubs in the spec")

    tests = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>
""".format(specification=options.spec, generator=sys.argv[0])

    for eval_spec in spec_yaml:
      name = eval_spec["name"]
      datatype, arch, isa = parse_eval_stub_name(name)

      test_case = generate_test_cases(name, datatype, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
