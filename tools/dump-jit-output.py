#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dumps JIT codegen given specific JIT parameters.
# Usage is similar to xngen:
#   dump-jit-output.py <path to JIT cc file> --max_mr=6 [--clamp_min]
# E.g.
#   ./tools/dump-jit-output.py \
#     src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.cc
#     --max_mr=6
#
# The parameters prefetch, clamp_min, clamp_max defaults to True if not
# specified on the command line.

import argparse
import codecs
import re
import sys
import xngen
from itertools import chain


parser = argparse.ArgumentParser(description='Dump output of JIT')
parser.add_argument("input", metavar="FILE", nargs=1,
          help="Input file")
parser.add_argument("--prefetch", action="store_true")
parser.add_argument("--clamp_min", action="store_true")
parser.add_argument("--clamp_max", action="store_true")
parser.add_argument("--max_mr", type=int, required=True)
parser.add_argument("-o", "--output",
          help='Output file')
parser.set_defaults(defines=list())


def preprocess(input_text):
  input_lines = input_text.splitlines()
  in_function = False
  output = []
  for i, line in enumerate(input_lines):
    if line.startswith('void Generator::generate'):
      in_function = True
    if not in_function:
      continue
    if line == '}':
      in_function = False
      output.append(line)
      continue
    if line.strip() == '}':
      continue

    o = re.sub(r'(if|else)( +\(.*\)) +{', r'$\1\2:', line)
    o = re.sub(r'&&', 'and', o)
    o = re.sub(r'\|\|', 'or', o)
    output.append(o)
  return output


def call_xngen(text, options):
  input_globals = {
      'prefetch': options.prefetch,
      'clamp_min': options.clamp_min,
      'clamp_max': options.clamp_max,
      'max_mr': options.max_mr,
  }
  return xngen.preprocess(text, input_globals, "codegen")


def main(args):
  options = parser.parse_args(args)
  input_text = codecs.open(options.input[0], "r", encoding="utf-8").read()
  output = preprocess(input_text)
  result = call_xngen("\n".join(output), options)
  if (options.output):
    with codecs.open(options.output, "w", encoding="utf-8") as output_file:
      output_file.write(result)
  else:
    print(result)


if __name__ == "__main__":
  main(sys.argv[1:])
