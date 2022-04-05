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
    description="Generates operator-strings code.")
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


#include <assert.h>
#include <stdint.h>

#include <xnnpack/{enum}-type.h>

""".format(
    specification=options.spec, generator=sys.argv[0], enum=options.enum)

    all_strings = ''
    pos = 0
    offset = "static const uint16_t offset[] = {";
    last_member = ""
    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      string = ukernel_spec["string"]

      all_strings +=  '    "' + string + '\\0"\n'

      offset += str(pos) + ","
      pos += len(string) + 1
      last_member = name

    offset = offset[:-1] + "};"
    output += offset + '\n\n';
    output += """static const char *data =
{all_strings};
""".format(all_strings=all_strings)

    output += """
const char* xnn_{enum}_type_to_string(enum xnn_{enum}_type type) {{
  assert(type <= {last_member});
  return &data[offset[type]];
}}""".format(last_member=last_member, enum=options.enum)

    txt_changed = True
    if os.path.exists(options.output):
      with codecs.open(options.output, "r", encoding="utf-8") as output_file:
        txt_changed = output_file.read() != output

    if txt_changed:
      with codecs.open(options.output, "w", encoding="utf-8") as output_file:
        output_file.write(output)


if __name__ == "__main__":
  main(sys.argv[1:])
