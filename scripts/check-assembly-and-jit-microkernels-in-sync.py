#!/usr/bin/env python3
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import subprocess

def line_in_file(n, f):
  with open(f) as o:
    return any(n in l for l in o)


def find_jit_files():
  paths = list(Path('src').glob('**/*.cc'))
  return [p for p in paths if line_in_file('Converted from', p)]


def check_assembly_file_exists(f):
  p = Path('.')
  assert((p / f).exists())


def get_assembly_file_or_template(converted_from):
  file = converted_from
  with open(converted_from) as f:
    for line in f.readlines():
      if 'Template: ' in line:
        file = line.split()[2]
        break
  if file.startswith('/'):
    file = file[len('/'):]
  return file


def find_assembly_file(jit_file):
  with jit_file.open() as f:
    for line in f.readlines():
      if 'Converted from' in line:
        converted_from = line.split()[3]
        check_assembly_file_exists(converted_from)
        assembly_file = get_assembly_file_or_template(converted_from)
        return assembly_file
    return f'{jit_file} does not have converted from'


def write_lint_markers(assembly_file, jit_file):
  with open(assembly_file) as f:
    lines = f.readlines()
    if any('LINT.IfChange' in l for l in lines):
      # has lint marker, check expected
      if not any(jit_file.name in l for l in lines):
        print(f'{jit_file.name} not found in {assembly_file}')
        assert(False)
      return

  relative_jit_file = jit_file.name
  sed_args = [
      'sed',
      '-i',
      '-e',
      '/# void xnn/i# LINT.IfChange',
      '-e',
      '/\/\/ void xnn/i# LINT.IfChange',
      '-e',
      f'/END_FUNCTION/a# LINT.ThenChange({relative_jit_file})',
      assembly_file,
  ]
  subprocess.run(sed_args)


def main():
  for f in find_jit_files():
    write_lint_markers(find_assembly_file(f), f)


if __name__ == '__main__':
  main()
