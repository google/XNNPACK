#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import sys
import io

import xnncommon


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

ISA_LIST = frozenset({
  'armsimd32',
  'avx',
  'avx2',
  'avx512f',
  'avx512skx',
  'avx512vbmi',
  'avx512vnni',
  'avx512vnnigfni',
  'avx512amx',
  'avxvnni',
  'f16c',
  'fma',
  'fma3',
  'fp16arith',
  'hexagon',
  'neon',
  'neonbf16',
  'neondot',
  'neondotfp16arith',
  'neoni8mm',
  'neonfma',
  'neonfp16',
  'neonfp16arith',
  'neonv8',
  'rvv',
  'rvvfp16arith',
  'scalar',
  'sse',
  'sse2',
  'sse41',
  'ssse3',
  'wasm',
  'wasmblendvps',
  'wasmrelaxedsimd',
  'wasmpshufb',
  'wasmsdot',
  'wasmsimd',
  'xop',
})

ISA_MAP = {
  'wasmblendvps': 'wasmrelaxedsimd',
  'wasmpshufb': 'wasmrelaxedsimd',
  'wasmsdot': 'wasmrelaxedsimd',
}

ARCH_LIST = frozenset({
  'aarch32',
  'aarch64',
  'wasm32',
  'wasmsimd32',
  'wasmrelaxedsimd32',
})

ISA_TO_HEADER_MAP = {
  'armsimd32': 'arm_acle.h',
  'avx': 'immintrin.h',
  'avx2': 'immintrin.h',
  'avx512f': 'immintrin.h',
  'avx512skx': 'immintrin.h',
  'avx512vbmi': 'immintrin.h',
  'avx512vnni': 'immintrin.h',
  'avx512vnnigfni': 'immintrin.h',
  'avx512amx': 'immintrin.h',
  'avxvnni': 'immintrin.h',
  'f16c': 'immintrin.h',
  'fma3': 'immintrin.h',
  'fp16arith': 'arm_fp16.h',
  'neon': 'arm_neon.h',
  'neonbf16': 'arm_neon.h',
  'neondot': 'arm_neon.h',
  'neondotfp16arith': 'arm_neon.h',
  'neoni8mm': 'arm_neon.h',
  'neonfma': 'arm_neon.h',
  'neonfp16': 'arm_neon.h',
  'neonfp16arith': 'arm_neon.h',
  'neonv8': 'arm_neon.h',
  'rvv': 'riscv_vector.h',
  'rvvfp16arith': 'riscv_vector.h',
  'sse': 'immintrin.h',
  'sse2': 'immintrin.h',
  'sse41': 'immintrin.h',
  'ssse3': 'immintrin.h',
  'wasmrelaxedsimd': 'wasm_simd128.h',
  'wasmsimd': 'wasm_simd128.h',
  'xop': 'xopintrin.h',
}

MICROKERNEL_NAME_REGEX = re.compile(
  r"\bxnn_(?:[a-z0-9]+(?:_[a-z0-9]+)*)_ukernel(?:_[a-z0-9]+)*__(?:[a-z0-9]+(?:_[a-z0-9]+)*)\b")

VERIFICATION_IGNORE_SUBDIRS = {
  os.path.join('src', 'math'),
  os.path.join('src', 'math', 'gen'),
  os.path.join('src', 'qs8-requantization'),
  os.path.join('src', 'qu8-requantization'),
}

AMALGAMATION_HEADER = """\
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""


parser = argparse.ArgumentParser(
  description='Utility for re-generating microkernel lists')
parser.add_argument('-a', '--amalgamate', action='store_true',
                    help='Amalgamate production microkernels')


import re

NUMBERS_REGEX = re.compile('([0-9]+)')

def human_sort_key(text):
  return [int(token) if token.isdigit() else token.lower() for token in NUMBERS_REGEX.split(text)]

def amalgamate_microkernel_sources(source_paths, include_header):
  amalgam_lines = list()
  amalgam_includes = set()
  for filepath in sorted(source_paths):
    with open(filepath, 'r', encoding='utf-8') as file:
      source_lines = file.read().splitlines()

    consumed_license = False
    consumed_includes = False
    for line in source_lines:
      if line.startswith('//'):
        if not consumed_license:
          # Skip and generate a standard license header for amalgamated file
          continue
      elif line.lstrip().startswith('#'):
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

    amalgam_lines.append('')

  # Multi-line sequence for XOP intrinsics, which don't have a standardized header
  amalgam_includes.discard('#ifdef _MSC_VER')
  amalgam_includes.discard('  #include <intrin.h>')
  amalgam_includes.discard('#else')
  amalgam_includes.discard('  #include <x86intrin.h>')
  amalgam_includes.discard('#endif')

  # Single-line sequences for intrinsics with a standardized header
  amalgam_includes.discard('#include <arm_acle.h>')
  amalgam_includes.discard('#include <arm_fp16.h>')
  amalgam_includes.discard('#include <arm_neon.h>')
  amalgam_includes.discard('#include <emmintrin.h>')
  amalgam_includes.discard('#include <immintrin.h>')
  amalgam_includes.discard('#include <nmmintrin.h>')
  amalgam_includes.discard('#include <smmintrin.h>')
  amalgam_includes.discard('#include <tmmintrin.h>')
  amalgam_includes.discard('#include <xmmintrin.h>')
  amalgam_includes.discard('#include <riscv_vector.h>')
  amalgam_includes.discard('#include <wasm_simd128.h>')

  amalgam_text = AMALGAMATION_HEADER

  amalgam_text += "\n".join(sorted(inc for inc in amalgam_includes if
                                   not inc.startswith('#include <xnnpack/')))
  if include_header:
    if include_header == 'xopintrin.h':
      amalgam_text += '\n\n'
      amalgam_text += '#ifdef _MSC_VER\n'
      amalgam_text += '  #include <intrin.h>\n'
      amalgam_text += '#else\n'
      amalgam_text += '  #include <x86intrin.h>\n'
      amalgam_text += '#endif\n\n'
    else:
      amalgam_text += '\n\n#include <%s>\n\n' % include_header
  else:
    amalgam_text += '\n\n'
  amalgam_text += '\n'.join(sorted(inc for inc in amalgam_includes if
                                   inc.startswith('#include <xnnpack/')))
  amalgam_text += '\n\n\n'
  amalgam_text += '\n'.join(amalgam_lines)

  return amalgam_text

def make_variable_name(prefix, key, suffix):
  return '_'.join(token for token in [prefix, key.upper(), suffix] if token)

def write_grouped_microkernels_bzl(file, microkernels, prefix, suffix):
  for key in sorted(microkernels):
    variable_name = make_variable_name(prefix, key, suffix)
    file.write(f'\n{variable_name} = [\n')
    for microkernel in sorted(microkernels[key], key=human_sort_key):
      file.write(f'    "{microkernel}",\n')
    file.write(']\n')

def write_grouped_microkernels_cmake(file, microkernels, prefix, suffix):
  for key in sorted(microkernels):
    variable_name = make_variable_name(prefix, key, suffix)
    file.write(f'\nSET({variable_name}')
    for microkernel in sorted(microkernels[key], key=human_sort_key):
      file.write(f'\n  {microkernel}')
    file.write(')\n')

def main(args):
  options = parser.parse_args(args)
  root_dir = os.path.normpath(os.path.join(TOOLS_DIR, '..'))
  src_dir = os.path.join(root_dir, 'src')
  configs_dir = os.path.join(src_dir, 'configs')
  ignore_roots = {
    src_dir,
    os.path.join(src_dir, 'amalgam', 'gen'),
    os.path.join(src_dir, 'configs'),
    os.path.join(src_dir, 'enums'),
    os.path.join(src_dir, 'jit'),
    os.path.join(src_dir, 'operators'),
    os.path.join(src_dir, 'subgraph'),
    os.path.join(src_dir, 'tables'),
    os.path.join(src_dir, 'xnnpack'),
  }
  c_microkernels_per_isa = {isa: [] for isa in ISA_LIST if isa not in ISA_MAP}
  c_microkernels_per_isa['neon_aarch64'] = list()
  c_microkernels_per_isa['neondot_aarch64'] = list()
  c_microkernels_per_isa['neonfma_aarch64'] = list()
  c_microkernels_per_isa['neonfp16arith_aarch64'] = list()
  c_microkernels_per_isa['neonbf16_aarch64'] = list()
  asm_microkernels_per_arch = {arch: [] for arch in ARCH_LIST}
  jit_microkernels_per_arch = {arch: [] for arch in ARCH_LIST}
  microkernel_name_to_filename = dict()
  for root, dirs, files in os.walk(src_dir, topdown=False):
    if root in ignore_roots:
      continue

    for name in files:
      if name.endswith('.in'):
        continue
      basename, ext = os.path.splitext(name)
      if ext == '.sollya':
        continue

      subdir = os.path.relpath(root, root_dir)
      filepath = os.path.join(subdir, name)

      # Build microkernel name -> microkernel filepath mapping
      if options.amalgamate:
        with open(os.path.join(root_dir, filepath), 'r', encoding='utf-8') as f:
          content = f.read()
          microkernels = re.findall(MICROKERNEL_NAME_REGEX, content)
          if not microkernels:
            if subdir not in VERIFICATION_IGNORE_SUBDIRS:
              print('No microkernel found in %s' % filepath)
          else:
            microkernels = sorted(set(microkernels))
            if len(microkernels) > 1:
              if ext == '.cc':
                # In generators it is normal to have multiple implementations in the same file.
                # We need to filter out non-generators though.
                microkernels = sorted(filter(lambda fn: fn.startswith('xnn_generate_'), microkernels))
                if not microkernels and subdir not in VERIFICATION_IGNORE_SUBDIRS:
                  print('No microkernel generator found in %s' % filepath)

                for microkernel in microkernels:
                  if microkernel in microkernel_name_to_filename:
                    print('Duplicate microkernel generator definition: %s and %s' % (microkernel_name_to_filename[microkernel], filepath))
                  else:
                    microkernel_name_to_filename[microkernel] = filepath
              else:
                microkernels_str = ', '.join(microkernels)
                print('Multiple microkernels (%s) found in %s' % (microkernels_str, filepath))
            else:
              microkernel = microkernels[0]
              if microkernel in microkernel_name_to_filename:
                print('Duplicate microkernel definition: %s and %s' % (microkernel_name_to_filename[microkernel], filepath))
              else:
                microkernel_name_to_filename[microkernel] = filepath

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
      elif ext == '.cc':
        for component in basename.split('-'):
          if component in ARCH_LIST:
            jit_microkernels_per_arch[component].append(filepath)
            break
        else:
          print('Unknown architecture for JIT microkernel %s' % filepath)

  with io.StringIO() as microkernels_bzl:
    microkernels_bzl.write('''\
"""
Microkernel filenames lists.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""
''')
    write_grouped_microkernels_bzl(microkernels_bzl, c_microkernels_per_isa, 'ALL', 'MICROKERNEL_SRCS')
    write_grouped_microkernels_bzl(microkernels_bzl, asm_microkernels_per_arch, '', 'ASM_MICROKERNEL_SRCS')
    write_grouped_microkernels_bzl(microkernels_bzl, jit_microkernels_per_arch, '', 'JIT_MICROKERNEL_SRCS')
    microkernels_bzl.seek(0)
    xnncommon.overwrite_if_changed(os.path.join(root_dir, 'microkernels.bzl'), microkernels_bzl.read())

  with io.StringIO() as microkernels_cmake:
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
    write_grouped_microkernels_cmake(microkernels_cmake, c_microkernels_per_isa, 'ALL', 'MICROKERNEL_SRCS')
    write_grouped_microkernels_cmake(microkernels_cmake, asm_microkernels_per_arch, '', 'ASM_MICROKERNEL_SRCS')
    write_grouped_microkernels_cmake(microkernels_cmake, jit_microkernels_per_arch, '', 'JIT_MICROKERNEL_SRCS')
    microkernels_cmake.seek(0)
    xnncommon.overwrite_if_changed(os.path.join(root_dir, 'cmake', 'microkernels.cmake'), microkernels_cmake.read())

  if options.amalgamate:
    # Collect filenames of production microkernels as a set
    prod_microkernels = set()
    for configs_filepath in os.listdir(configs_dir):
      with open(os.path.join(configs_dir, configs_filepath), 'r', encoding='utf-8') as config_file:
        content = config_file.read()
        microkernels = re.findall(MICROKERNEL_NAME_REGEX, content)
        prod_microkernels.update(microkernels)
    prod_microkernels = set(map(microkernel_name_to_filename.get, prod_microkernels))

    for isa_spec, microkernels in c_microkernels_per_isa.items():
      microkernels = sorted(prod_microkernels.intersection(microkernels), key=human_sort_key)
      if microkernels:
        filepaths = [os.path.join(root_dir, filepath) for filepath in microkernels]
        if '_' in isa_spec:
          isa, arch = isa_spec.split('_')
          amalgam_filename = f'{isa}-{arch}.c'
        else:
          isa = isa_spec
          amalgam_filename = f'{isa}.c'
        header = ISA_TO_HEADER_MAP.get(isa)
        amalgam_text = amalgamate_microkernel_sources(filepaths, include_header=header)
        xnncommon.overwrite_if_changed(os.path.join(src_dir, 'amalgam', 'gen', amalgam_filename), amalgam_text)


if __name__ == '__main__':
  main(sys.argv[1:])
