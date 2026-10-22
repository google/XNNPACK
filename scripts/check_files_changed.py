#!/usr/bin/python3

import argparse
import os
import re


def find_changed_files(
    script_to_check, file_paths_contain='/', file_edit_dates=None
):
  if file_edit_dates is None:
    file_edit_dates = {}
  outputs = {}

  def get_mtime(path):
    if path not in file_edit_dates:
      try:
        file_edit_dates[path] = os.path.getmtime(path)
      except OSError:
        file_edit_dates[path] = 0
    return file_edit_dates[path]

  with open(script_to_check, 'r', encoding='utf-8') as f:
    content = re.sub(r'\\[ \t]*\r?\n', ' ', f.read())
    for line in content.splitlines():
      if line.lstrip().startswith('#'):
        continue
      file_names = [x.strip() for x in line.split(' ') if file_paths_contain in x]
      if len(file_names) >= 3:
        in_file = file_names[1]
        in_mtime = get_mtime(in_file)
        if in_mtime == 0:
          continue
        for out_file in file_names[2:]:
          get_mtime(out_file)
          outputs[out_file] = in_file
      elif len(file_names) == 2:
        in_file = file_names[0]
        in_mtime = get_mtime(in_file)
        if in_mtime == 0:
          continue
        out_file = file_names[1]
        get_mtime(out_file)
        outputs[out_file] = in_file

  return [
      out_file
      for out_file, in_file in outputs.items()
      if file_edit_dates[in_file] > file_edit_dates[out_file]
  ]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('filename', help='Scripts to read', type=str)
  parser.add_argument(
      '--files-contain', dest='files_contain', default='/', type=str
  )
  args = parser.parse_args()

  for out_file in find_changed_files(args.filename, args.files_contain):
    print(out_file)


if __name__ == '__main__':
  main()
