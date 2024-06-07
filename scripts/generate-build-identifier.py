# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generates an identifier from the files in the XNNPack project

This generates a fingerprint of the XNNPack library sources.
"""

import argparse
import hashlib
import os
import sys
import textwrap

parser = argparse.ArgumentParser(
    prog="XNNPackFingerprint",
    description=(
        "Generates a C source file that defines a function that returns a"
        " fingerprint of the given XNNPack source files and writes it to the"
        " output."
    ),
)
parser.add_argument(
    "--output", required=True, action="store", help="Set the output"
)
parser.add_argument("inputs", nargs="+", help="The source files to use to generate the fingerprint.")

FILE_TEMPLATE = """// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Auto-generated file. Do not edit!
//   Generator: scripts/generate-build-identifier.py
//
// The following inputs were used to generate this file.
{genlist}

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static const uint8_t xnn_build_identifier[] = {{
{id_data}
}};

size_t xnn_experimental_get_build_identifier_size() {{
  return sizeof(xnn_build_identifier);
}}

const void* xnn_experimental_get_build_identifier_data() {{
  return xnn_build_identifier;
}}

bool xnn_experimental_check_build_identifier(const void* data, const size_t size) {{
  if(size != xnn_experimental_get_build_identifier_size()) {{
    return false;
  }}
  return !memcmp(data, xnn_build_identifier, size);
}}
"""


def main(args) -> None:
  m = hashlib.sha256()
  for path in args.inputs:
    with open(path, "rb") as f:
      m.update(f.read())
  byte_list = ", ".join(str(b).rjust(3, "x") for b in m.digest())
  byte_list = textwrap.indent(textwrap.fill(byte_list, width=40), "  ").replace("x", " ")
  formated_input_list = "\n".join("// - " + p for p in args.inputs)
  with open(args.output, "w") as out:
    out.write(FILE_TEMPLATE.format(id_data=byte_list, genlist=formated_input_list))


if __name__ == "__main__":
  main(parser.parse_args())
