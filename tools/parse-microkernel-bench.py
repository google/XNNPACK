#!/usr/bin/env python
"""Parse microkernel benchmark results into a more readable table.

This script translates google benchmark framework output for microkernel
benchmarks from the following ordering (from google benchmark framework):

for b in benchmark:
  for k in microkernels:
    k/b  (time)  (cpu)  ...

To a table, where the rows are microkernels, and the columns are the benchmarks.
It sorts the microkernels by the geomean of the ratio of the microkernel
time in each benchmark to the best time for that benchmark, and colors the data
to make it easier to find the outliers.

Usage example:

bench/vunary_bench | tee /dev/stderr | tools/parse-microkernel-bench.py
"""

import collections
import math
import pathlib
import re
import sys

BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BOLD = "\033[1m"
RESET = "\033[0m"  # Reset to default color

MAX_COLS = 10


def color(r):
  thresholds = [
      (1.01, WHITE),
      (1.1, GREEN),
      (1.25, YELLOW),
      (2, RED),
  ]

  for threshold, color in thresholds:
    if r < threshold:
      return color
  return MAGENTA


def to_seconds(time, units):
  if units == "s":
    return time
  elif units == "ms":
    return time / 1e3
  elif units == "us":
    return time / 1e6
  elif units == "ns":
    return time / 1e9
  else:
    raise ValueError("Unknown units: %s" % units)


def seconds_to_string(t):
  if t < 1e-7:
    s = "%5.1fns" % (t * 1e9)
  elif t < 1e-4:
    s = "%5.1fus" % (t * 1e6)
  elif t < 1e-1:
    s = "%5.1fms" % (t * 1e3)
  else:
    s = "%5.1fs" % t

  return s.rjust(5)


def ratio_to_string(t, best):
  r = t / best
  if r <= 1.01:
    s = seconds_to_string(t)
  elif r < 100:
    s = "%6.2fx" % r
  else:
    s = "  >100x"

  return color(r) + s.rjust(5) + RESET


def prod_kernels(ukernels):
  """Returns the set of kernels that are used in production."""
  result = set()
  try:
    configs = (pathlib.Path(__file__).parent.parent / "src/configs").glob("*.c")
    for path in configs:
      contents = path.read_text()
      for i in ukernels:
        if i not in result and i in contents:
          result.add(i)
  except Exception:
    pass
  return result


# Parse a row of google benchmark output.
r = re.compile(r"(.*ukernel[^/]*)/([\S]*)\s+(\d\S+) (\S+)\s+(\d\S+)\s+(\S+).*")

# Gather the data into a dictionary {ukernel: {benchmark: time_seconds}}
all_ukernels = collections.defaultdict(dict)
for line in sys.stdin:
  m = r.match(line)
  if m:
    ukernel = m.group(1)
    benchmark = m.group(2)
    time = float(m.group(3))
    units = m.group(4)

    # Remove BM_ and parameter prefixes from the ukernel, if any.
    if ukernel.startswith("BM_"):
      ukernel = ukernel[3:]
    ukernel = ukernel.split("/")[-1]

    all_ukernels[ukernel][benchmark] = to_seconds(time, units)

while all_ukernels:
  # Find the kernels that belong in the next table.
  ukernel_prefix = next(iter(all_ukernels.keys())).split("ukernel")[0]
  ukernels = {
      k: v for k, v in all_ukernels.items() if k.startswith(ukernel_prefix)
  }
  for i in ukernels.keys():
    all_ukernels.pop(i)

  # Get the set of all benchmarks represented here.
  benchmarks = list(
      sorted(set.union(*[set(i.keys()) for i in ukernels.values()]))
  )

  # Make the table layout and header.
  ukernel_width = max([len(i) for i in ukernels.keys()]) + 5
  table_format = "| {} | " + " | ".join(["{:>7}" for i in benchmarks]) + " |"
  print(
      table_format.format(
          "ukernel".ljust(ukernel_width), *range(len(benchmarks))
      )
  )
  print(table_format.format("-" * ukernel_width, *["-" * 7] * len(benchmarks)))

  # Find the best time for each benchmark.
  bests = {}
  ratios = {}
  for i in benchmarks:
    # {ukernel: time} for this benchmark
    data = {j: ukernels[j].get(i, 0) for j in ukernels.keys()}
    best = min(data.values())
    bests[i] = best
    ratios[i] = {j: data[j] / best for j in ukernels.keys()}

  # Get the subset of ukernels that are used in production.
  prod = prod_kernels(ukernels.keys())

  cols = {i[1]: [i[0]] for i in enumerate(benchmarks)}

  # Sort the kernels by the sum of the log of the ratios of the kernel in each
  # benchmark (geomean-ish ordering)
  for ukernel, data in sorted(
      ukernels.items(),
      key=lambda x: sum([math.log(ratios[i].get(x[0], 1)) for i in benchmarks]),
  ):
    # Highlight ukernels found in production configs.
    if ukernel in prod:
      ukernel = ukernel + "*"
      ukernel = BOLD + ukernel.ljust(ukernel_width) + RESET
    else:
      ukernel = ukernel.ljust(ukernel_width)

    # Print the row
    cols = [ratio_to_string(data.get(i, 0), bests[i]) for i in benchmarks]
    print(table_format.format(ukernel, *cols))

  # Print the key for the columns.
  print()
  for i, benchmark in enumerate(benchmarks):
    print("{i}. {benchmark}".format(i=i, benchmark=benchmark))
  print()
  print()
