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
                    help="List of microkernel filenames in the BUILD file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C source) file')


def main(args):
  options = parser.parse_args(args)

  build_path = os.path.join(ROOT_DIR, "..", "BUILD")

  with codecs.open(build_path, "r", encoding="utf-8") as build_file:
    build_text = build_file.read()

  pattern = r"\b" + options.set + r"\b\s*=\s*\["
  match = re.search(pattern, build_text)
  if not match:
    raise ValueError(
      "Failed to find file set %s (regex \"%s\") inside the BUILD file" %
        (options.set, pattern))

  start_pos = match.end()
  end_pos = build_text.find("]", start_pos)

  fileset = [filename.strip()[1:-1] for filename in
             build_text[start_pos:end_pos].split(",")]

  amalgam_lines = list()
  amalgam_includes = set()
  for filename in sorted(fileset):
    if not filename:
      continue

    filepath = os.path.join(ROOT_DIR, "..", filename)
    with codecs.open(filepath, "r", encoding="utf-8") as file:
      filelines = file.read().splitlines()

    consumed_license = False
    consumed_includes = False
    for line in filelines:
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

  amalgam_includes.discard("#include <emmintrin.h>")
  amalgam_includes.discard("#include <immintrin.h>")
  amalgam_includes.discard("#include <nmmintrin.h>")
  amalgam_includes.discard("#include <smmintrin.h>")
  amalgam_includes.discard("#include <tmmintrin.h>")
  amalgam_includes.discard("#include <xmmintrin.h>")

  amalgam_text = """\
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""

  amalgam_text += "\n".join(sorted(inc for inc in amalgam_includes if
                                   not inc.startswith("#include <xnnpack/")))
  amalgam_text += "\n\n#include <immintrin.h>\n\n"
  amalgam_text += "\n".join(sorted(inc for inc in amalgam_includes if
                                   inc.startswith("#include <xnnpack/")))
  amalgam_text += "\n\n\n"
  amalgam_text += "\n".join(amalgam_lines)



  txt_changed = True
  if os.path.exists(options.output):
    with open(options.output, "r", encoding="utf-8") as amalgam_file:
      txt_changed = amalgam_file.read() != amalgam_text

  if txt_changed:
    with open(options.output, "w", encoding="utf-8") as amalgam_file:
      amalgam_file.write(amalgam_text)


if __name__ == "__main__":
  main(sys.argv[1:])
