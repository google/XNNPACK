#!/usr/bin/python3
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+', help='Input filename')
parser.add_argument('--min_lines', type=int, default=1, help='Specify the min number of lines to rearrange')
args = parser.parse_args()

min_num_lines = args.min_lines

filename_regex = r'^[^\(]*(src|source|unit_test|include)/.*$'

# caveat: Does fully not handle numbers with different number of digits.
def compare_strs(str1, str2, num_sort=True):
    winner = 0 # which one's bigger
    num_sort_mode = False
    for i in range(min(len(str1),len(str2))):
        if winner == 0:
            if num_sort:
                if str1[i].isdigit() and str2[i].isdigit():
                    if num_sort_mode:
                        str1_num += str1[i]
                        str2_num += str2[i]
                    else:
                        num_sort_mode = True
                        str1_num = str1[i]
                        str2_num = str2[i]
                elif num_sort_mode:
                    if str1[i].isdigit(): str1_num += str1[i]
                    if str2[i].isdigit(): str2_num += str2[i]
                    winner = int(str1_num) - int(str2_num)
                    num_sort_mode = False
                    if winner != 0: break
            if str1[i] != str2[i] and not num_sort_mode:
                if str2[i].isdigit():
                    winner = -1
                elif str1[i].isdigit() or str1[i] > str2[i]:
                    winner = 1
                else:
                    winner = -1
    if num_sort_mode: # if it's still in num_sort_mode then find the results
        if len(str1)-1 > i:
            i += 1
            if str1[i].isdigit(): str1_num += str1[i]
        elif len(str2)-1 > i:
            i += 1
            if str2[i].isdigit(): str2_num += str2[i]
        if int(str1_num) > int(str2_num):
            winner = 1
        elif int(str1_num) < int(str2_num):
            winner = -1
        num_sort_mode = False

    if winner == 0:
        if str2[i].isdigit():
            winner = -1
        elif str1[i].isdigit() or str1[i] > str2[i]:
            winner = 1
        else:
            winner = -1
    return winner


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def custom_sort(listy):
    for _ in range(len(listy)-1+1):
        for i in range(len(listy)-1):
            bigger = compare_strs(listy[i], listy[i+1])
            if bigger > 0:
                tmp_str = listy[i]
                listy[i] = listy[i+1]
                listy[i+1] = tmp_str
    return listy

# file_lines = [l.strip(stripchars) for l in file_lines]
# currently sorting by first number in filename. smallest numbers first
# this thing may need the ability to recognize filenames
# file_lines = sorted(file_lines, key=lambda line: int(re.search('\d+', line.split('/')[-1] + ' 0').group(0)))
input_filenames=args.inputs
for input_filename in input_filenames:
    with open(input_filename,'r') as file:
        file_lines = file.read().split('\n')
    output_filename=input_filename
    files_range = [] # is a tuple
    filenames_ranges = []
    for i, line in enumerate(file_lines):
        if re.search(filename_regex, line):
            if files_range == []:
                files_range.append(i)
        else:
            if files_range != []:
                files_range.append(i-1)
                # manages min num lines
                if files_range[1] - files_range[0] + 1 >= min_num_lines:filenames_ranges.append(files_range)
                files_range = []
    if files_range != []:
        files_range.append(i-1)
        # manages min num lines
        if files_range[1] - files_range[0] + 1 >= min_num_lines:filenames_ranges.append(files_range)
        files_range = []


    for x in filenames_ranges:
        # accommodate for a close bracket at the end of the last filename
        if file_lines[x[1]][-1] == ')':
            append_chars = ')'
            file_lines[x[1]] = file_lines[x[1]][:-1]
        else:
            append_chars = ''

        new_lines = sorted(file_lines[x[0]:x[1]+1], key=cmp_to_key(compare_strs))
        for i, l in enumerate(range(x[0], x[1]+1)):
            file_lines[l] = new_lines[i]
            if l == x[1]:
                file_lines[l] += append_chars

    with open(output_filename, 'w') as file:
        file.write('\n'.join(file_lines))
