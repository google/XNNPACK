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


parser = argparse.ArgumentParser(description='MaxPool microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(s8|u8|s16|f16|f32)_maxpool(_(minmax))?_ukernel_(\d+)p(\d+)x__(.+)_c(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  primary_tile = int(match.group(4))
  incremental_tile = int(match.group(5))
  channel_tile = int(match.group(7))
  vector_tile = bool(match.group(8))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(6))
  return primary_tile, incremental_tile, channel_tile, vector_tile, arch, isa


MAXPOOL_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  MaxPoolMicrokernelTester()
    .pooling_elements(${PRIMARY_TILE})
    .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
    .channels(channel_tile)
    $if DATATYPE in ["s8", "u8"]:
      .qmin(std::numeric_limits<${CTYPE}>::min())
      .qmax(std::numeric_limits<${CTYPE}>::max())
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  MaxPoolMicrokernelTester()
    .pooling_elements(${PRIMARY_TILE})
    .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
    .channels(channel_tile)
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      .input_offset(${next_prime(CHANNEL_TILE+1)})
    $else:
      .input_offset(channel_tile+1)
    $if DATATYPE in ["s8", "u8"]:
      .qmin(std::numeric_limits<${CTYPE}>::min())
      .qmax(std::numeric_limits<${CTYPE}>::max())
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  MaxPoolMicrokernelTester()
    .pooling_elements(${PRIMARY_TILE})
    .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
    .channels(channel_tile)
    .qmin(${QMIN})
    $if DATATYPE in ["s8", "u8"]:
      .qmax(std::numeric_limits<${CTYPE}>::max())
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  MaxPoolMicrokernelTester()
    .pooling_elements(${PRIMARY_TILE})
    .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
    .channels(channel_tile)
    $if DATATYPE in ["s8", "u8"]:
      .qmin(std::numeric_limits<${CTYPE}>::min())
    .qmax(${QMAX})
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      $if DATATYPE in ["s8", "u8"]:
        .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(std::numeric_limits<${CTYPE}>::max())
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        .input_offset(${next_prime(CHANNEL_TILE+1)})
      $else:
        .input_offset(channel_tile+1)
      $if DATATYPE in ["s8", "u8"]:
        .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(std::numeric_limits<${CTYPE}>::max())
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if CHANNEL_TILE > 1 or CHANNEL_SCALED_TILE != CHANNEL_TILE:
  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*8)})
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(channel_tile*8)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(${QMIN})
          $if DATATYPE in ["s8", "u8"]:
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(${QMIN})
          $if DATATYPE in ["s8", "u8"]:
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(${QMAX})
          .Test(${", ".join(TEST_ARGS)});
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(${QMAX})
          .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*8)})
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile*8)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_offset(${next_prime(CHANNEL_TILE)})
        $else:
          .input_offset(channel_tile)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .qmin(${QMIN})
        $if DATATYPE in ["s8", "u8"]:
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(${QMAX})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            .input_offset(${next_prime(CHANNEL_TILE)})
          $else:
            .input_offset(channel_tile)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .input_offset(${next_prime(CHANNEL_TILE*2)})
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .input_offset(channel_tile*2)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .qmin(${QMIN})
        $if DATATYPE in ["s8", "u8"]:
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .qmin(${QMIN})
        $if DATATYPE in ["s8", "u8"]:
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(${QMAX})
        .Test(${", ".join(TEST_ARGS)});
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(${QMAX})
        .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*2)})
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(channel_tile*2)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  MaxPoolMicrokernelTester()
    .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
    .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
    .channels(channel_tile)
    $if DATATYPE in ["s8", "u8"]:
      .qmin(std::numeric_limits<${CTYPE}>::min())
      .qmax(std::numeric_limits<${CTYPE}>::max())
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  MaxPoolMicrokernelTester()
    .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
    .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
    .channels(channel_tile)
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      .input_offset(${next_prime(CHANNEL_TILE+1)})
    $else:
      .input_offset(channel_tile+1)
    $if DATATYPE in ["s8", "u8"]:
      .qmin(std::numeric_limits<${CTYPE}>::min())
      .qmax(std::numeric_limits<${CTYPE}>::max())
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  MaxPoolMicrokernelTester()
    .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
    .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
    .channels(channel_tile)
    .qmin(${QMIN})
    $if DATATYPE in ["s8", "u8"]:
      .qmax(std::numeric_limits<${CTYPE}>::max())
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  MaxPoolMicrokernelTester()
    .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
    .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
    .channels(channel_tile)
    $if DATATYPE in ["s8", "u8"]:
      .qmin(std::numeric_limits<${CTYPE}>::min())
    .qmax(${QMAX})
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      $if DATATYPE in ["s8", "u8"]:
        .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(std::numeric_limits<${CTYPE}>::max())
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        .input_offset(${next_prime(CHANNEL_TILE+1)})
      $else:
        .input_offset(channel_tile+1)
      $if DATATYPE in ["s8", "u8"]:
        .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(std::numeric_limits<${CTYPE}>::max())
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if CHANNEL_TILE > 1 or CHANNEL_SCALED_TILE != CHANNEL_TILE:
  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*5)})
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(channel_tile*5)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(${QMIN})
          $if DATATYPE in ["s8", "u8"]:
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(${QMIN})
          $if DATATYPE in ["s8", "u8"]:
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(${QMAX})
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(${QMAX})
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*8)})
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile*8)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_offset(${next_prime(CHANNEL_TILE)})
        $else:
          .input_offset(channel_tile)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .qmin(${QMIN})
        $if DATATYPE in ["s8", "u8"]:
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(${QMAX})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            .input_offset(${next_prime(CHANNEL_TILE)})
          $else:
            .input_offset(channel_tile)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .input_offset(${next_prime(CHANNEL_TILE*2)})
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .input_offset(channel_tile*2)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .qmin(${QMIN})
        $if DATATYPE in ["s8", "u8"]:
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        .qmin(${QMIN})
        $if DATATYPE in ["s8", "u8"]:
          .qmax(std::numeric_limits<${CTYPE}>::max())
        .Test(${", ".join(TEST_ARGS)});
    }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
    for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(${QMAX})
        .Test(${", ".join(TEST_ARGS)});
    }
  $else:
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channels)
        $if DATATYPE in ["s8", "u8"]:
          .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(${QMAX})
        .Test(${", ".join(TEST_ARGS)});
    }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*2)})
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(channel_tile*2)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      $if DATATYPE in ["s8", "u8"]:
        .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(std::numeric_limits<${CTYPE}>::max())
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        .input_offset(${next_prime(CHANNEL_TILE+1)})
      $else:
        .input_offset(channel_tile+1)
      $if DATATYPE in ["s8", "u8"]:
        .qmin(std::numeric_limits<${CTYPE}>::min())
        .qmax(std::numeric_limits<${CTYPE}>::max())
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      .qmin(${QMIN})
      $if DATATYPE in ["s8", "u8"]:
        .qmax(std::numeric_limits<${CTYPE}>::max())
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  const size_t channel_tile = ${CHANNEL_SCALED_TILE};
  for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      $if DATATYPE in ["s8", "u8"]:
        .qmin(std::numeric_limits<${CTYPE}>::min())
      .qmax(${QMAX})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if CHANNEL_TILE > 1 or CHANNEL_SCALED_TILE != CHANNEL_TILE:
  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*8)})
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile*8)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(${QMIN})
            $if DATATYPE in ["s8", "u8"]:
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(${QMIN})
            $if DATATYPE in ["s8", "u8"]:
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(${QMAX})
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(${QMAX})
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(channel_tile)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(${QMIN})
          $if DATATYPE in ["s8", "u8"]:
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(${QMAX})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*2)})
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(channel_tile*2)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(${QMIN})
          $if DATATYPE in ["s8", "u8"]:
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(${QMIN})
          $if DATATYPE in ["s8", "u8"]:
            .qmax(std::numeric_limits<${CTYPE}>::max())
          .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(${QMAX})
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if DATATYPE in ["s8", "u8"]:
            .qmin(std::numeric_limits<${CTYPE}>::min())
          .qmax(${QMAX})
          .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, few_output_pixels) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}}}) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }
}

TEST(${TEST_NAME}, few_output_pixels_with_input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}}}) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*5+1)})
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile*5+1)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }
}

TEST(${TEST_NAME}, few_output_pixels_with_qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}}}) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(${QMIN})
            $if DATATYPE in ["s8", "u8"]:
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(${QMIN})
            $if DATATYPE in ["s8", "u8"]:
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }
}

TEST(${TEST_NAME}, few_output_pixels_with_qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}}}) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(${QMAX})
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
            .qmax(${QMAX})
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }
}

TEST(${TEST_NAME}, few_output_pixels_with_output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}}}) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .output_stride(${next_prime(CHANNEL_TILE*5+1)})
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .output_stride(channel_tile*5+1)
            $if DATATYPE in ["s8", "u8"]:
              .qmin(std::numeric_limits<${CTYPE}>::min())
              .qmax(std::numeric_limits<${CTYPE}>::max())
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }
}

TEST(${TEST_NAME}, few_output_pixels_with_step) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}}}) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .step(step)
              .channels(channels)
              .output_stride(${next_prime(CHANNEL_TILE*5+1)})
              $if DATATYPE in ["s8", "u8"]:
                .qmin(std::numeric_limits<${CTYPE}>::min())
                .qmax(std::numeric_limits<${CTYPE}>::max())
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .step(step)
              .channels(channels)
              .output_stride(channel_tile*5+1)
              $if DATATYPE in ["s8", "u8"]:
                .qmin(std::numeric_limits<${CTYPE}>::min())
                .qmax(std::numeric_limits<${CTYPE}>::max())
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }
  }
}
"""


def generate_test_cases(ukernel, init_fn, primary_tile, incremental_tile,
                        channel_tile, vector_tile, isa):
  """Generates all tests cases for a MAXPOOL micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    primary_tile: Number of rows (pixels) processed per one iteration of the
                  primary outer loop of the micro-kernel.
    incremental_tile: Number of rows (pixels) processed per one iteration of
                      the incremental outer loop of the micro-kernel.
    channel_tile: Number of channels processed per one iteration of the inner
                  loops of the micro-kernel.
    vector_tile: Indicates if channels are specified in vectors rather than
                 elements.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel, init_fn]
  channel_scaled_tile = channel_tile
  if vector_tile:
    ctype = {"qs8": "int8_t", "qu8": "uint8_t", "f16": "uint16_t", "f32": "float"}[datatype]
    channel_scaled_tile = {"rvv": "(%s*xnn_init_hardware_config()->vlenb/sizeof(%s))" % (str(channel_tile), ctype)}[isa]
  return xngen.preprocess(MAXPOOL_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "DATATYPE": datatype,
      "CTYPE": {"s8": "int8_t", "u8": "uint8_t", "f16": "uint16_t", "f32": "int16_t"}[datatype],
      "QMIN": {"s8": -64, "u8": 64}.get(datatype, -16384),
      "QMAX": {"s8": 64, "u8": 192}.get(datatype, 16384),
      "PRIMARY_TILE": primary_tile,
      "INCREMENTAL_TILE": incremental_tile,
      "CHANNEL_TILE": channel_tile,
      "CHANNEL_SCALED_TILE": channel_scaled_tile,
      "CHANNEL_SUFFIX": "v" if vector_tile else "",
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
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams-init.h"
#include "maxpool-microkernel-tester.h"
#include "next_prime.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec["init"]
      primary_tile, incremental_tile, channel_tile, vector_tile, arch, isa = \
        split_ukernel_name(name)

      test_case = generate_test_cases(name, init_fn, primary_tile,
                                      incremental_tile, channel_tile, vector_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
