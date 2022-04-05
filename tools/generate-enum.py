#!/usr/bin/env python
# Copyright 2022 Google LLC
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
    description="Generates xnn_operator_type enum.")
parser.add_argument(
    "-s",
    "--spec",
    metavar="FILE",
    required=True,
    help="Specification (YAML) file")
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C source) file")
parser.add_argument(
    "-e",
    "--enum",
    metavar="FILE",
    required=True,
    help="Enum to generate")
parser.set_defaults(defines=list())


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of operators in the spec")

    output = """\
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}

#pragma once

enum xnn_{enum}_type {{
""".format(
    specification=options.spec, generator=sys.argv[0], enum=options.enum)

    name = spec_yaml[0]["name"]
    output += "  " + name + " = 0,\n"
    for ukernel_spec in spec_yaml[1:]:
      name = ukernel_spec["name"]
      output += "  " + name + ",\n"

    output += "};"
    txt_changed = True
    if os.path.exists(options.output):
      with codecs.open(options.output, "r", encoding="utf-8") as output_file:
        txt_changed = output_file.read() != output

    if txt_changed:
      with codecs.open(options.output, "w", encoding="utf-8") as output_file:
        output_file.write(output)


if __name__ == "__main__":
  main(sys.argv[1:])
