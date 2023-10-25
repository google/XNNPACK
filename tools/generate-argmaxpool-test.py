#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(
  description='ArgMaxPool microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f32)_argmaxpool_ukernel_((\d+)p)?(\d+)x__(.+)_c(\d+)", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  if match.group(2):
    primary_tile = int(match.group(3))
    incremental_tile = int(match.group(4))
  else:
    primary_tile = int(match.group(4))
    incremental_tile = 0

  channel_tile = int(match.group(6))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(5))
  return primary_tile, incremental_tile, channel_tile, arch, isa


ARGMAXPOOL_TEST_TEMPLATE = """\
$if INCREMENTAL_TILE == 0:
  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_unipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE})
      .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
      .channels(${CHANNEL_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_unipass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE})
      .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
      .channels(${CHANNEL_TILE})
      .input_offset(${next_prime(CHANNEL_TILE+1)})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_unipass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(${CHANNEL_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_unipass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(${CHANNEL_TILE})
        .input_offset(${next_prime(CHANNEL_TILE+1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if CHANNEL_TILE > 1:
    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_unipass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_unipass_fulltile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*8)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_unipass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_unipass_subtile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*8)})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_unipass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_unipass_fulltile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_unipass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
        for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_unipass_subtile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
        for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE)})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_unipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_unipass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(channels)
        .input_offset(${next_prime(CHANNEL_TILE*2)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_unipass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_unipass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*2)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if INCREMENTAL_TILE != 0:
  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_twopass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
      .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
      .channels(${CHANNEL_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_twopass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
      .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
      .channels(${CHANNEL_TILE})
      .input_offset(${next_prime(CHANNEL_TILE+1)})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_twopass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(${CHANNEL_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_twopass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(${CHANNEL_TILE})
        .input_offset(${next_prime(CHANNEL_TILE+1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if CHANNEL_TILE > 1:
    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_twopass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_twopass_fulltile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*5)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_twopass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_twopass_subtile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*8)})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_twopass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_twopass_fulltile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_twopass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
        for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_twopass_subtile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
        for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE)})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_twopass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_twopass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(channels)
        .input_offset(${next_prime(CHANNEL_TILE*2)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_twopass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_twopass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*2)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_multipass) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(${CHANNEL_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}_multipass_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
        .channels(${CHANNEL_TILE})
        .input_offset(${next_prime(CHANNEL_TILE+1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if CHANNEL_TILE > 1:
    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_multipass) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements += 3) {
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}_multipass_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements += 3) {
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*8)})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_multipass) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements += 3) {
        for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}_multipass_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements += 3) {
        for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
          ArgMaxPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .channels(channels)
            .input_offset(${CHANNEL_TILE})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_multipass) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements += 3) {
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}_multipass_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements += 3) {
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*2)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if INCREMENTAL_TILE == 0:
  $MIN_POOLING, MAX_POOLING = 2, PRIMARY_TILE
$else:
  $MIN_POOLING, MAX_POOLING = PRIMARY_TILE + 1, PRIMARY_TILE + INCREMENTAL_TILE

TEST(${TEST_NAME}, few_output_pixels) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = ${MIN_POOLING}; pooling_elements <= ${MAX_POOLING}; pooling_elements++) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, few_output_pixels_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = ${MIN_POOLING}; pooling_elements <= ${MAX_POOLING}; pooling_elements++) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*5+1)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, few_output_pixels_with_output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = ${MIN_POOLING}; pooling_elements <= ${MAX_POOLING}; pooling_elements++) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
          .channels(channels)
          .output_stride(${next_prime(CHANNEL_TILE*5+1)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, few_output_pixels_with_step) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = ${MIN_POOLING}; pooling_elements <= ${MAX_POOLING}; pooling_elements++) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
        for (size_t step = 2; step <= pooling_elements; step++) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${", ".join(map(str, filter(bool, [PRIMARY_TILE, INCREMENTAL_TILE])))})
            .step(step)
            .channels(channels)
            .output_stride(${next_prime(CHANNEL_TILE*5+1)})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  }
}
"""


def generate_test_cases(ukernel, primary_tile, incremental_tile, channel_tile,
                        isa):
  """Generates all tests cases for a ARGMAXPOOL micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    primary_tile: Number of rows (pixels) processed per one iteration of the
                  primary outer loop of the micro-kernel.
    incremental_tile: Number of rows (pixels) processed per one iteration of
                      the incremental outer loop of the micro-kernel.
    channel_tile: Number of channels processed per one iteration of the inner
                  loops of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel]
  if not isa:
    test_args.append("ArgMaxPoolMicrokernelTester::Variant::Scalar")
  return xngen.preprocess(ARGMAXPOOL_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "DATATYPE": datatype,
      "PRIMARY_TILE": primary_tile,
      "INCREMENTAL_TILE": incremental_tile,
      "CHANNEL_TILE": channel_tile,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/argmaxpool.h>
#include "argmaxpool-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      primary_tile, incremental_tile, channel_tile, arch, isa = \
        split_ukernel_name(name)

      test_case = generate_test_cases(name, primary_tile, incremental_tile,
                                      channel_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
