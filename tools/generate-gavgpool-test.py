#!/usr/bin/env python
# Copyright 2020 Google LLC
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


parser = argparse.ArgumentParser(description='GAvgPool microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.match(r"xnn_(qs8|qu8|f16|f32)_(gavgpool|rdsum)(_(minmax))?(_(fp32|rndnu))?_ukernel_((\d+)p)?(\d+)x__(.+)_c(\d+)(_acc(\d+))?(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  requantization_type = match.group(6)
  if match.group(7):
    primary_tile = int(match.group(8))
    incremental_tile = int(match.group(9))
  else:
    primary_tile = int(match.group(9))
    incremental_tile = 0
  channel_tile = int(match.group(11))
  vector_tile = bool(match.group(12))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(10))
  return requantization_type, primary_tile, incremental_tile, channel_tile, vector_tile, arch, isa


AVGPOOL_TEST_TEMPLATE = """\
$if INCREMENTAL_TILE == 0:
  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GAvgPoolMicrokernelTester()
      .rows(${PRIMARY_TILE})
      .channels(${CHANNEL_SCALED_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = 1; rows < ${PRIMARY_TILE}; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(${CHANNEL_SCALED_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile_with_input_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GAvgPoolMicrokernelTester()
      .rows(${PRIMARY_TILE})
      .channels(${CHANNEL_SCALED_TILE})
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        .input_stride(${next_prime(CHANNEL_TILE+1)})
      $else:
        .input_stride(${CHANNEL_SCALED_TILE}+1)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GAvgPoolMicrokernelTester()
      .rows(${PRIMARY_TILE})
      .channels(${CHANNEL_SCALED_TILE})
      .qmax(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GAvgPoolMicrokernelTester()
      .rows(${PRIMARY_TILE})
      .channels(${CHANNEL_SCALED_TILE})
      .qmin(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  $if CHANNEL_TILE > 1 or CHANNEL_SCALED_TILE != CHANNEL_TILE:
    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(${PRIMARY_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        for (size_t channels = ${CHANNEL_SCALED_TILE}*2; channels < ${CHANNEL_SCALED_TILE}*8; channels += ${CHANNEL_SCALED_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(${PRIMARY_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          for (size_t rows = 1; rows < ${PRIMARY_TILE}; rows++) {
            GAvgPoolMicrokernelTester()
              .rows(rows)
              .channels(channels)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        for (size_t channels = ${CHANNEL_SCALED_TILE}*2; channels < ${CHANNEL_SCALED_TILE}*8; channels += ${CHANNEL_SCALED_TILE}) {
          for (size_t rows = 1; rows < ${PRIMARY_TILE}; rows++) {
            GAvgPoolMicrokernelTester()
              .rows(rows)
              .channels(channels)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        for (size_t rows = 1; rows < ${PRIMARY_TILE}; rows++) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile_with_qmax) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        for (size_t rows = 1; rows < ${PRIMARY_TILE}; rows++) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        for (size_t rows = 1; rows < ${PRIMARY_TILE}; rows++) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }
$else:
  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GAvgPoolMicrokernelTester()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels(${CHANNEL_SCALED_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile_with_input_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GAvgPoolMicrokernelTester()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels(${CHANNEL_TILE})
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        .input_stride(${next_prime(CHANNEL_TILE+1)})
      $else:
        .input_stride(${CHANNEL_SCALED_TILE}+1)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GAvgPoolMicrokernelTester()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels(${CHANNEL_SCALED_TILE})
      .qmax(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GAvgPoolMicrokernelTester()
      .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
      .channels(${CHANNEL_SCALED_TILE})
      .qmin(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = ${PRIMARY_TILE+1}; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(${CHANNEL_SCALED_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile_with_input_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = ${PRIMARY_TILE+1}; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(${CHANNEL_SCALED_TILE})
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_stride(${next_prime(CHANNEL_TILE+1)})
        $else:
          .input_stride(${CHANNEL_SCALED_TILE}+1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(${CHANNEL_SCALED_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile_with_input_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(${CHANNEL_SCALED_TILE})
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_stride(${next_prime(CHANNEL_TILE+1)})
        $else:
          .input_stride(${CHANNEL_SCALED_TILE}+1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}*2; channels < ${CHANNEL_SCALED_TILE}*8; channels += ${CHANNEL_SCALED_TILE}) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        for (size_t rows = ${PRIMARY_TILE+1}; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}*2; channels < ${CHANNEL_SCALED_TILE}*8; channels += ${CHANNEL_SCALED_TILE}) {
        for (size_t rows = ${PRIMARY_TILE+1}; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}*2; channels < ${CHANNEL_SCALED_TILE}*8; channels += ${CHANNEL_SCALED_TILE}) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile_with_input_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .input_stride(${next_prime(CHANNEL_TILE*16+1)})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}*2; channels < ${CHANNEL_SCALED_TILE}*8; channels += ${CHANNEL_SCALED_TILE}) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .input_stride(${CHANNEL_SCALED_TILE}*16+1)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

  $if CHANNEL_TILE > 1 or CHANNEL_SCALED_TILE != CHANNEL_TILE:
    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile_with_qmax) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        for (size_t rows = ${PRIMARY_TILE+1}; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile_with_input_stride) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t channels = 1; channels < ${CHANNEL_SCALED_TILE}; channels++) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows <= ${INCREMENTAL_TILE*5}; rows += ${INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
              .input_stride(${next_prime(CHANNEL_TILE+1)})
            $else:
              .input_stride(${CHANNEL_SCALED_TILE}+1)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(${PRIMARY_TILE+INCREMENTAL_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_2pass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        for (size_t rows = ${PRIMARY_TILE+1}; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        for (size_t rows = ${PRIMARY_TILE+1}; rows < ${PRIMARY_TILE+INCREMENTAL_TILE}; rows++) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows < ${INCREMENTAL_TILE*5}; rows += ${PRIMARY_TILE+INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows < ${INCREMENTAL_TILE*5}; rows += ${PRIMARY_TILE+INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_fulltile_with_input_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows < ${INCREMENTAL_TILE*5}; rows += ${PRIMARY_TILE+INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .input_stride(${next_prime(CHANNEL_TILE*2+11)})
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t channels = ${CHANNEL_SCALED_TILE}+1; channels < ${CHANNEL_SCALED_TILE}*2; channels++) {
        for (size_t rows = ${PRIMARY_TILE+INCREMENTAL_TILE}; rows < ${INCREMENTAL_TILE*5}; rows += ${PRIMARY_TILE+INCREMENTAL_TILE}) {
          GAvgPoolMicrokernelTester()
            .rows(rows)
            .channels(channels)
            .input_stride(${CHANNEL_SCALED_TILE}*2+11)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

"""


def generate_test_cases(ukernel, init_fn, requantization_type, primary_tile,
                        incremental_tile, channel_tile, vector_tile, isa):
  """Generates all tests cases for a GAVGPOOL micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    requantization_type: Requantization type (FP32/RNDNU).
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
  if requantization_type:
    test_args.append("xnn_%s_requantize_%s" % \
      (datatype.lower(), requantization_type.lower()))
  channel_scaled_tile = channel_tile
  if vector_tile:
    ctype = {"qs8": "int8_t", "qu8": "uint8_t", "f16": "uint16_t", "f32": "float"}[datatype]
    channel_scaled_tile = {"rvv": "(%s*xnn_init_hardware_config()->vlenb/sizeof(%s))" % (str(channel_tile), ctype)}[isa]
  return xngen.preprocess(AVGPOOL_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "DATATYPE": datatype,
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
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "gavgpool-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      requantization_type, primary_tile, incremental_tile, channel_tile, vector_tile, arch, \
        isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, init_fn, requantization_type,
                                      primary_tile, incremental_tile,
                                      channel_tile, vector_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
