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


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(
  description='Amalgamation utility for microkernels')
parser.add_argument("-s", "--set", metavar="SET", required=True,
                    help="List of microkernel filenames in the BUILD.bazel file")
parser.add_argument("-i", "--include", metavar="INCLUDE",
                    help="Header file to include (e.g. immintrin.h, arm_neon.h)")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C source) file')


AMALGAMATION_HEADER = """\
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""


def amalgamate_microkernel_sources(source_paths, include_header):
  amalgam_lines = list()
  amalgam_includes = set()
  for filepath in sorted(source_paths):
    with open(filepath, "r", encoding="utf-8") as file:
      source_lines = file.read().splitlines()

    consumed_license = False
    consumed_includes = False
    for line in source_lines:
      if line.startswith("//"):
        if not consumed_license:
          # Skip and generate a standard license header for amalgamated file
          continue
      elif line.lstrip().startswith("#"):
        if not consumed_includes:
          amalgam_includes.add(line)
          continue
        consumed_license = True
      elif not line:
        if not consumed_includes:
          # Skip empty lines until end of headers
          continue
      else:
        consumed_license = True
        consumed_includes = True

      amalgam_lines.append(line)

    amalgam_lines.append("")

  # Multi-line sequence for XOP intrinsics, which don't have a standardized header
  amalgam_includes.discard("#ifdef _MSC_VER")
  amalgam_includes.discard("  #include <intrin.h>")
  amalgam_includes.discard("#else")
  amalgam_includes.discard("  #include <x86intrin.h>")
  amalgam_includes.discard("#endif")

  # Single-line sequences for intrinsics with a standardized header
  amalgam_includes.discard("#include <arm_acle.h>")
  amalgam_includes.discard("#include <arm_fp16.h>")
  amalgam_includes.discard("#include <arm_neon.h>")
  amalgam_includes.discard("#include <emmintrin.h>")
  amalgam_includes.discard("#include <immintrin.h>")
  amalgam_includes.discard("#include <nmmintrin.h>")
  amalgam_includes.discard("#include <smmintrin.h>")
  amalgam_includes.discard("#include <tmmintrin.h>")
  amalgam_includes.discard("#include <xmmintrin.h>")
  amalgam_includes.discard("#include <wasm_simd128.h>")

  amalgam_text = AMALGAMATION_HEADER

  amalgam_text += "\n".join(sorted(inc for inc in amalgam_includes if
                                   not inc.startswith("#include <xnnpack/")))
  if include_header:
    if include_header == "xopintrin.h":
      amalgam_text += "\n\n"
      amalgam_text += "#ifdef _MSC_VER\n"
      amalgam_text += "  #include <intrin.h>\n"
      amalgam_text += "#else\n"
      amalgam_text += "  #include <x86intrin.h>\n"
      amalgam_text += "#endif\n\n"
    else:
      amalgam_text += "\n\n#include <%s>\n\n" % include_header
  else:
    amalgam_text += "\n\n"
  amalgam_text += "\n".join(sorted(inc for inc in amalgam_includes if
                                   inc.startswith("#include <xnnpack/")))
  amalgam_text += "\n\n\n"
  amalgam_text += "\n".join(amalgam_lines)

  return amalgam_text

def main(args):
  options = parser.parse_args(args)

  build_path = os.path.join(ROOT_DIR, "..", "BUILD.bazel")

  with codecs.open(build_path, "r", encoding="utf-8") as build_file:
    build_text = build_file.read()

  pattern = r"\b" + options.set + r"\b\s*=\s*\["
  match = re.search(pattern, build_text)
  if not match:
    raise ValueError(
      "Failed to find file set %s (regex \"%s\") inside the BUILD.bazel file" %
        (options.set, pattern))

  start_pos = match.end()
  end_pos = build_text.find("]", start_pos)

  filenames = [filename.strip()[1:-1] for filename in
               build_text[start_pos:end_pos].split(",")]

  filepaths = [os.path.join(ROOT_DIR, "..", filename)
               for filename in filenames if filename]

  amalgam_text = amalgamate_microkernel_sources(filepaths, options.include)

  txt_changed = True
  if os.path.exists(options.output):
    with codecs.open(options.output, "r", encoding="utf-8") as amalgam_file:
      txt_changed = amalgam_file.read() != amalgam_text

  if txt_changed:
    with codecs.open(options.output, "w", encoding="utf-8") as amalgam_file:
      amalgam_file.write(amalgam_text)


if __name__ == "__main__":
  main(sys.argv[1:])
