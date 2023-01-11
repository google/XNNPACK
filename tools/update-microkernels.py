#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import sys


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

ISA_LIST = frozenset({
  'armsimd32',
  'avx',
  'avx2',
  'avx512f',
  'avx512skx',
  'avx512vbmi',
  'f16c',
  'fma',
  'fma3',
  'fp16arith',
  'hexagon',
  'neon',
  'neonbf16',
  'neondot',
  'neonfma',
  'neonfp16',
  'neonfp16arith',
  'neonv8',
  'rvv',
  'scalar',
  'sse',
  'sse2',
  'sse41',
  'ssse3',
  'wasm',
  'wasmrelaxedsimd',
  'wasmpshufb',
  'wasmsimd',
  'xop',
})

ISA_MAP = {
  'wasmpshufb': 'wasmrelaxedsimd',
}

ARCH_LIST = frozenset({
  'aarch32',
  'aarch64',
  'wasm32',
})

parser = argparse.ArgumentParser(
  description='Utility for re-generating microkernel lists')


import re

NUMBERS_REGEX = re.compile('([0-9]+)')

def human_sort_key(text):
  return [int(token) if token.isdigit() else token.lower() for token in NUMBERS_REGEX.split(text)]


def main(args):
  options = parser.parse_args(args)
  root_dir = os.path.normpath(os.path.join(TOOLS_DIR, '..'))
  src_dir = os.path.join(root_dir, 'src')
  ignore_roots = {
    src_dir,
    os.path.join(src_dir, 'amalgam'),
    os.path.join(src_dir, 'enums'),
    os.path.join(src_dir, 'jit'),
    os.path.join(src_dir, 'operators'),
    os.path.join(src_dir, 'subgraph'),
    os.path.join(src_dir, 'tables'),
    os.path.join(src_dir, 'xnnpack'),
  }
  c_microkernels_per_isa = {isa: [] for isa in ISA_LIST if isa not in ISA_MAP}
  c_microkernels_per_isa['neon_aarch64'] = list()
  c_microkernels_per_isa['neonfma_aarch64'] = list()
  c_microkernels_per_isa['neonfp16arith_aarch64'] = list()
  c_microkernels_per_isa['neonbf16_aarch64'] = list()
  asm_microkernels_per_arch = {arch: [] for arch in ARCH_LIST}
  for root, dirs, files in os.walk(src_dir, topdown=False):
    if root in ignore_roots:
      continue
    for name in files:
      if name.endswith('.in'):
        continue
      basename, ext = os.path.splitext(name)
      if ext == '.sollya':
        continue
      filepath = os.path.join(os.path.relpath(root, root_dir), name)
      if ext == '.c':
        arch = None
        for component in basename.split('-'):
          if component in ARCH_LIST:
            arch = component
          elif component in ISA_LIST:
            isa = ISA_MAP.get(component, component)
            key = isa if arch is None else f'{isa}_{arch}'
            c_microkernels_per_isa[key].append(filepath)
            break
        else:
          print('Unknown ISA for C microkernel %s' % filepath)
      elif ext == '.S':
        for component in basename.split('-'):
          if component in ARCH_LIST:
            asm_microkernels_per_arch[component].append(filepath)
            break
        else:
          print('Unknown architecture for assembly microkernel %s' % filepath)

  with open(os.path.join(root_dir, 'microkernels.bzl'), 'w', encoding='utf-8') as microkernels_bzl:
    microkernels_bzl.write('''\
"""
Microkernel filenames lists.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""
''')
    for isa in sorted(c_microkernels_per_isa):
      microkernels_bzl.write(f'\nALL_{isa.upper()}_MICROKERNEL_SRCS = [\n')
      for microkernel in sorted(c_microkernels_per_isa[isa], key=human_sort_key):
        microkernels_bzl.write(f'    "{microkernel}",\n')
      microkernels_bzl.write(']\n')

    for arch in sorted(asm_microkernels_per_arch):
      microkernels_bzl.write(f'\n{arch.upper()}_ASM_MICROKERNEL_SRCS = [\n')
      for microkernel in sorted(asm_microkernels_per_arch[arch], key=human_sort_key):
        microkernels_bzl.write(f'    "{microkernel}",\n')
      microkernels_bzl.write(']\n')

  with open(os.path.join(root_dir, "cmake", 'microkernels.cmake'), 'w', encoding='utf-8') as microkernels_cmake:
    microkernels_cmake.write("""\
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py

""")
    for isa in sorted(c_microkernels_per_isa):
      microkernels_cmake.write(f'\nSET(ALL_{isa.upper()}_MICROKERNEL_SRCS')
      for microkernel in sorted(c_microkernels_per_isa[isa], key=human_sort_key):
        microkernels_cmake.write(f'\n  {microkernel}')
      microkernels_cmake.write(')\n')

    for arch in sorted(asm_microkernels_per_arch):
      microkernels_cmake.write(f'\nSET({arch.upper()}_ASM_MICROKERNEL_SRCS')
      for microkernel in sorted(asm_microkernels_per_arch[arch], key=human_sort_key):
        microkernels_cmake.write(f'\n  {microkernel}')
      microkernels_cmake.write(')\n')


if __name__ == '__main__':
  main(sys.argv[1:])
