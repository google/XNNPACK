#!/usr/bin/python3

import argparse
import re
import typing

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="File to validate")
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()
filename = args.filename
verbose = args.verbose

c_and_s_regex = [("c", r"c(\d+)"), ("s", r"s(\d+)")]
short_lines_to_validate = [
    ("mr", r"_config\.mr = (\d+);"),
    ("nr", r"_config\.nr = (\d+);"),
    ("log2_kr", r"_config\.log2_kr = (\d+);"),
    ("log2_sr", r"_config\.log2_sr = (\d+);"),
]

def find_indentation(line):
    return len(line) - len(line.lstrip(' '))

def validate_code_block(f: typing.TextIO, line):
    lines_to_validate = []
    defined_variables = {}
    global line_number
    code_block_indentation = find_indentation(line)
    endgame = False
    while line:
        regex_match = re.search(r'XNN_MR_TO_INDEX\((\d+)', line)
        if regex_match and not endgame:
            first_number_should_be = regex_match.group(1)
            main_part_to_validate = line.split('__')[0].split('_')[-1] # 1x16c4
            first_number = main_part_to_validate.split('x')[0]
            second_number = re.search(r'^\d+', main_part_to_validate.split('x')[1]).group(0)
            parts_to_validate = {
                "first_number_should_be": int(first_number_should_be),
                "main_part_to_validate": main_part_to_validate,
                "first_number": int(first_number),
                "second_number": int(second_number),
            }
            # kr and sr are sometimes present
            for number_name, number_regexp in c_and_s_regex:
                regex_match = re.search(number_regexp, main_part_to_validate)
                if regex_match:
                    parts_to_validate[number_name] = int(regex_match.group(1))
            lines_to_validate.append(parts_to_validate)
        else:
            for var_name, var_regexp in short_lines_to_validate:
                regex_match = re.search(var_regexp, line)
                if regex_match:
                    endgame = True
                    defined_variables[var_name] = int(regex_match.group(1))
                    break
            else:
                if endgame:break

        if find_indentation(line) < code_block_indentation:
            # #else #endif are ok
            if line.strip() in ["#else", "#endif", ""]:
                code_block_indentation = find_indentation(line)
            else:
                break
        line = f.readline()
        line_number += 1

    # go through lines_to_validate and defined_variables and check that everything's as it should be
    first_numbers = [x["first_number"] for x in lines_to_validate]
    second_numbers = [x["second_number"] for x in lines_to_validate]
    if verbose:
        if defined_variables.get("mr") and defined_variables.get("mr") != max(first_numbers):
            yield f'mr == max(first_numbers). mr=={defined_variables.get("mr")} max(first_numbers) == {max(first_numbers)}'
    if defined_variables.get("nr") and {defined_variables.get("nr")} != set(second_numbers):
        yield f'nr should be same as all second_numbers. nr=={defined_variables.get("nr")} set(second_numbers) == {set(second_numbers)}'
    c_value = lines_to_validate[0].get("c")
    s_value = lines_to_validate[0].get("s")
    for line_to_validate in lines_to_validate:
        if line_to_validate.get("first_number_should_be") != line_to_validate.get("first_number"):
            yield (f"XNN_MR_TO_INDEX(x) is not the same as the first number."
                  f" they are {line_to_validate.get('first_number_should_be')} and {line_to_validate.get('first_number')}")
        # if c or r are present for one they should be present for all
        if line_to_validate.get('c') != c_value:
            yield f"c is not the same as previous lines. previously was {c_value}. now is {line_to_validate.get('c')}"
        if line_to_validate.get("s") != s_value:
            yield f"s is not the same as previous lines. previously was {s_value}. now is {line_to_validate.get('s')}"

    # check that sr and kr are there when they should be there (and not there when they shouldn't be)
    if (not (defined_variables.get("log2_kr") == c_value == None) and
            (defined_variables.get("log2_kr") is None or 2 ** defined_variables.get("log2_kr") != c_value)):
        if (defined_variables.get("log2_kr") != None and c_value != None) or verbose:
            yield f"log2_kr does not match c. log2_kr == {defined_variables.get('log2_kr')}. c == {c_value}"
    if (not (defined_variables.get("log2_sr") == s_value == None) and
            (defined_variables.get("log2_sr") is None or 2 ** defined_variables.get("log2_sr") != s_value)):
        if (defined_variables.get("log2_sr") != None and s_value != None) or verbose:
            yield f"log2_sr does not match r. log2_sr == {defined_variables.get('log2_sr')}. s == {s_value}"


with open(filename, 'r') as f:
    line = f.readline()
    line_number = 1
    while line:
        if "XNN_MR_TO_INDEX" in line and line.startswith("      "):
            for warning in validate_code_block(f, line):
                print(f'in file {filename} near line {line_number}: {warning}')
        line = f.readline()
        line_number += 1
