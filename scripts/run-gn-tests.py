# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs XNNPACK's unit tests on the current machine for the standalone GN build.

Collects everything in the build output directory matching the pattern
out/foo/xnnpack_*_test, and runs them with GoogleTest sharding enabled.
"""

import argparse
import asyncio
import dataclasses
import datetime
import glob
import os
import sys

# Add tests that require sharding here.
LONG_TESTS = frozenset(['xnnpack_operators_test'])


@dataclasses.dataclass
class TestResult:
  stdout: str
  stderr: str
  suite: str
  success: bool
  duration_seconds: float
  shard: int
  num_shards: int


async def run_one_test(
    path_to_executable: os.PathLike[str],
    lock: asyncio.Semaphore,
    *,
    current_shard: int,
    total_shards: int,
) -> TestResult:
  """Runs a single test suite with the given shard index.

  Args:
    path_to_executable: Path to the test binary to execute.
    lock: Semaphore to limit concurrent executions.
    current_shard: The shard index to run.
    total_shards: The total number of shards the test is split into.

  Returns:
    A TestResult structure.
  """
  # Prepare arguments etc
  args = ['gtest_brief=1', 'gtest_color=0']
  env = {
      'GTEST_TOTAL_SHARDS': str(total_shards),
      'GTEST_SHARD_INDEX': str(current_shard),
  }
  # Ensure a maximum level of concurrency.
  async with lock:
    start_time = datetime.datetime.now()
    process = await asyncio.create_subprocess_exec(
        path_to_executable,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    stdout, stderr = await process.communicate()
    await process.wait()
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Await the exit code, if zero, all's well.
    return TestResult(
        stdout=stdout.decode('ascii'),
        stderr=stderr.decode('ascii'),
        suite=os.path.basename(path_to_executable),
        success=process.returncode == 0,
        duration_seconds=duration,
        shard=current_shard,
        num_shards=total_shards,
    )


async def main() -> None:
  """Parses arguments, finds test suites, and runs them."""
  parser = argparse.ArgumentParser(
      'run-gn-tests',
      description="Runs XNNPACK's unit tests on the"
      + ' current machine, and print a summary.',
  )
  parser.add_argument(
      'out_dir', help='Path to a build directory e.g. out/Default'
  )
  parser.add_argument(
      '--shards',
      type=int,
      default=16,
      help='How much to subdivide long test suites.',
  )
  parser.add_argument(
      '--cpus',
      type=int,
      help='The maximum number of test shards that can run at once',
  )
  parser.add_argument(
      '--verbose',
      action='store_true',
      help="Prints test output as they're executing",
  )

  args = parser.parse_args()

  # Figure out how many tests we can run at once, the command line
  # takes precendence.
  detected_concurrency = os.cpu_count()
  concurrency = (
      args.cpus
      if args.cpus
      else (1 if not detected_concurrency else detected_concurrency)
  )
  # The semaphore controls the number of tests that can run.
  semaphore = asyncio.Semaphore(concurrency)

  # Pick up the executables - must be named in this way to work
  test_suites = list(sorted(glob.glob(args.out_dir + '/xnnpack_*_test')))

  print(f'Discovered {len(test_suites)} test suites...')
  # Create the list of tests to run, sharding the long ones.
  task_list = []
  for suite in test_suites:
    num_shards = 1 if os.path.basename(suite) not in LONG_TESTS else args.shards
    for shard in range(num_shards):
      task_list.append(
          run_one_test(
              suite,
              semaphore,
              current_shard=shard,
              total_shards=num_shards,
          )
      )

  # Run and collect the results.
  failures = []
  for result in asyncio.as_completed(task_list):
    result = await result
    description = f'{result.suite} ({result.shard + 1}/{result.num_shards})'
    print(description.ljust(60, '.'), end='', flush=True)
    outcome = (
        f'PASS ({result.duration_seconds:.2f} s)' if result.success else 'FAIL'
    )
    print(outcome.rjust(20, '.'))
    if args.verbose:
      print(result.stdout)
      if result.stderr:
        print('**stderr*')
        print(result.stderr)
    if not result.success:
      failures.append(result)

  # Re-iterate any failures.
  for x in sorted(failures, key=lambda x: x.suite):
    print(x.suite, f'- Shard #{x.shard + 1}', 'stderr:')
    print(x.stderr)
    print('stdout:')
    print(x.stdout)

  # Print a final summary and exit.
  if not failures:
    print('** SUCCESS - ALL TESTS PASS **')
  else:
    print('** TEST FAILURES **')

  sys.exit(int(len(failures) >= 1))


if __name__ == '__main__':
  asyncio.run(main())
