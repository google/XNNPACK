#!/usr/bin/python3

import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Scripts to read', type=str)
    parser.add_argument('--files-contain', dest='files_contain', default='/', type=str)
    args = parser.parse_args()

    script_to_check = args.filename
    file_paths_contain = args.files_contain

    file_edit_dates = {}
    outputs = {}
    files_exist = {}
    with open(script_to_check, 'r') as f:
        for line in f.readlines():
            if line.lstrip().startswith('#'):continue
            file_names = [x.strip() for x in line.split(' ')]
            # Filter file names based on presence of 'files_contain'
            file_names = [x for x in file_names if file_paths_contain in x]
            if len(file_names) >= 3:
                # outputs[file_names[1]] = file_names[0]
                # i==0 is script. i==1 is input file. i>=1 is for output files
                for i, file_name in enumerate(file_names):
                    if file_name not in files_exist:
                        files_exist[file_name] = os.path.exists(file_name)

                    # if file doesn't exist
                    if not files_exist[file_name]:
                        # if output file does not exist
                        if i > 1:
                            file_edit_dates[file_name] = 0
                        else:
                            continue
                    if file_name not in file_edit_dates:
                        file_edit_dates[file_name] = os.path.getmtime(file_name)
                    if i > 1:
                        outputs[file_name] = file_names[1]

    for out_file, in_file in outputs.items():
        if file_edit_dates[in_file] > file_edit_dates[out_file]:
            print(out_file)


if __name__ == '__main__':
    main()
