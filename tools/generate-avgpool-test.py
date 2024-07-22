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


parser = argparse.ArgumentParser(description='AvgPool microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(qs8|qu8|f16|f32)_[p]?avgpool(_(minmax))?(_(fp32|rndnu))?_ukernel_((\d+)p)?(\d+)x__(.+)_c(\d+)(v)?(_acc(\d+))?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  requantization_type = match.group(5)
  if match.group(6):
    primary_tile = int(match.group(7))
    incremental_tile = int(match.group(8))
  else:
    primary_tile = int(match.group(8))
    incremental_tile = 0
  channel_tile = int(match.group(10))
  vector_tile = bool(match.group(11))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(9))
  return requantization_type, primary_tile, incremental_tile, channel_tile, vector_tile, arch, isa


AVGPOOL_TEST_TEMPLATE = """\
$if INCREMENTAL_TILE == 0:
  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    AvgPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE})
      .pooling_tile(${PRIMARY_TILE})
      .channels(channel_tile)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    AvgPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE})
      .pooling_tile(${PRIMARY_TILE})
      .channels(channel_tile)
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        .input_offset(${next_prime(CHANNEL_TILE+1)})
      $else:
        .input_offset(channel_tile+1)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE})
        .pooling_tile(${PRIMARY_TILE})
        .channels(channel_tile)
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_offset(${next_prime(CHANNEL_TILE+1)})
        $else:
          .input_offset(channel_tile+1)
        .zero_index_mod2(zero_index_mod2)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if DATATYPE == "qu8":
    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channel_tile)
          .input_scale(scale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channel_tile)
          .input_zero_point(zero_point)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_output_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channel_tile)
          .output_scale(scale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_output_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channel_tile)
          .output_zero_point(zero_point)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    AvgPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE})
      .pooling_tile(${PRIMARY_TILE})
      .channels(channel_tile)
      .qmin(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    AvgPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE})
      .pooling_tile(${PRIMARY_TILE})
      .channels(channel_tile)
      .qmax(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${PRIMARY_TILE})
        .channels(channel_tile)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${PRIMARY_TILE})
        .channels(channel_tile)
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_offset(${next_prime(CHANNEL_TILE+1)})
        $else:
          .input_offset(channel_tile+1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE})
          .channels(channel_tile)
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            .input_offset(${next_prime(CHANNEL_TILE+1)})
          $else:
            .input_offset(channel_tile+1)
          .zero_index_mod2(zero_index_mod2)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  $if CHANNEL_TILE > 1 or CHANNEL_SCALED_TILE != CHANNEL_TILE:
    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*8)})
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .input_offset(channel_tile*8)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*8)})
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_offset(channel_tile*8)
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    $if DATATYPE == "qu8":
      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE})
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .input_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE})
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .input_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE})
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .input_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE})
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .input_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_output_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE})
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .output_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE})
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .output_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_output_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE})
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .output_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE})
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .output_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .qmin(128)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .qmin(128)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmax) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .qmax(128)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .qmax(128)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
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
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*8)})
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_offset(channel_tile*8)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .input_offset(${next_prime(CHANNEL_TILE*8)})
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE})
                .channels(channels)
                .input_offset(channel_tile*8)
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            .input_offset(${next_prime(CHANNEL_TILE)})
          $else:
            .input_offset(channel_tile)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
              .input_offset(${next_prime(CHANNEL_TILE)})
            $else:
              .input_offset(channel_tile)
            .zero_index_mod2(zero_index_mod2)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    $if DATATYPE == "qu8":
      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_output_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .output_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_output_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmax) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
        for (size_t channels = 1; channels < channel_tile; channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
              .input_offset(${next_prime(CHANNEL_TILE)})
            $else:
              .input_offset(channel_tile)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
                .input_offset(${next_prime(CHANNEL_TILE)})
              $else:
                .input_offset(channel_tile)
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*2)})
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*2)})
            .zero_index_mod2(zero_index_mod2)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE})
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .input_offset(channel_tile*2)
            .zero_index_mod2(zero_index_mod2)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

  $if DATATYPE == "qu8":
    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_input_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_output_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .output_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .output_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_output_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE})
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE})
          .pooling_tile(${PRIMARY_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*2)})
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE})
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_unipass_subtile_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = 2; pooling_elements < ${PRIMARY_TILE}; pooling_elements++) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*2)})
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE})
              .channels(channels)
              .input_offset(channel_tile*2)
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
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
      for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*5+1)})
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(channel_tile*5+1)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_offset(${next_prime(CHANNEL_TILE*5+1)})
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_offset(channel_tile*5+1)
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }
  }

  $if DATATYPE == "qu8":
    TEST(${TEST_NAME}, few_output_pixels_with_input_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
        for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE})
                  .channels(channels)
                  .input_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE})
                  .channels(channels)
                  .input_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }
    }

    TEST(${TEST_NAME}, few_output_pixels_with_input_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
        for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE})
                  .channels(channels)
                  .input_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE})
                  .channels(channels)
                  .input_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }
    }

    TEST(${TEST_NAME}, few_output_pixels_with_output_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
        for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE})
                  .channels(channels)
                  .output_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE})
                  .channels(channels)
                  .output_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }
    }

    TEST(${TEST_NAME}, few_output_pixels_with_output_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
        for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE})
                  .channels(channels)
                  .output_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE})
                  .channels(channels)
                  .output_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }
    }

  TEST(${TEST_NAME}, few_output_pixels_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmin(128)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmin(128)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmax(128)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmax(128)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels_with_output_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_stride(${next_prime(CHANNEL_TILE*5+1)})
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_stride(channel_tile*5+1)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels_with_step) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, ${PRIMARY_TILE-1}, ${PRIMARY_TILE}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            for (size_t step = 2; step <= pooling_elements; step++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .step(step)
                .channels(channels)
                .output_stride(${next_prime(CHANNEL_TILE*5+1)})
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= 3 * channel_tile; channels += channel_tile-1) {
            for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .step(step)
                .channels(channels)
                .output_stride(channel_tile*5+1)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }
  }
$else:
  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    AvgPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    AvgPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        .input_offset(${next_prime(CHANNEL_TILE+1)})
      $else:
        .input_offset(channel_tile+1)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channel_tile)
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_offset(${next_prime(CHANNEL_TILE+1)})
        $else:
          .input_offset(channel_tile+1)
        .zero_index_mod2(zero_index_mod2)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if DATATYPE == "qu8":
    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channel_tile)
          .input_scale(scale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channel_tile)
          .input_zero_point(zero_point)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_output_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channel_tile)
          .output_scale(scale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_output_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channel_tile)
          .output_zero_point(zero_point)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    AvgPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      .qmin(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    AvgPoolMicrokernelTester()
      .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
      .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
      .channels(channel_tile)
      .qmax(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channel_tile)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channel_tile)
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_offset(${next_prime(CHANNEL_TILE+1)})
        $else:
          .input_offset(channel_tile+1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channel_tile)
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            .input_offset(${next_prime(CHANNEL_TILE+1)})
          $else:
            .input_offset(channel_tile+1)
          .zero_index_mod2(zero_index_mod2)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  $if CHANNEL_TILE > 1 or CHANNEL_SCALED_TILE != CHANNEL_TILE:
    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*5)})
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile*5)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*5)})
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(channel_tile*5)
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    $if DATATYPE == "qu8":
      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_output_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_output_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(128)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(128)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmax) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmax(128)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmax(128)
            .Test(${", ".join(TEST_ARGS)});
        }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
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
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*8)})
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(channel_tile*8)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_offset(${next_prime(CHANNEL_TILE*8)})
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_offset(channel_tile*8)
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_offset) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            .input_offset(${next_prime(CHANNEL_TILE)})
          $else:
            .input_offset(channel_tile)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
              .input_offset(${next_prime(CHANNEL_TILE)})
            $else:
              .input_offset(channel_tile)
            .zero_index_mod2(zero_index_mod2)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    $if DATATYPE == "qu8":
      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_output_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_output_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmax) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
        for (size_t channels = 1; channels < channel_tile; channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
              .input_offset(${next_prime(CHANNEL_TILE)})
            $else:
              .input_offset(channel_tile)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
                .input_offset(${next_prime(CHANNEL_TILE)})
              $else:
                .input_offset(channel_tile)
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(${next_prime(CHANNEL_TILE*2)})
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*2)})
            .zero_index_mod2(zero_index_mod2)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile*2)
            .zero_index_mod2(zero_index_mod2)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
  }

  $if DATATYPE == "qu8":
    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_input_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_output_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_scale(scale)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_output_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_fulltile_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
      for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(${PRIMARY_TILE+INCREMENTAL_TILE})
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*2)})
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_twopass_subtile_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+1}; pooling_elements < ${PRIMARY_TILE+INCREMENTAL_TILE}; pooling_elements++) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*2)})
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(channel_tile*2)
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channel_tile)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channel_tile)
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          .input_offset(${next_prime(CHANNEL_TILE+1)})
        $else:
          .input_offset(channel_tile+1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
          .channels(channel_tile)
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            .input_offset(${next_prime(CHANNEL_TILE+1)})
          $else:
            .input_offset(channel_tile+1)
          .zero_index_mod2(zero_index_mod2)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  $if DATATYPE == "qu8":
    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channel_tile)
            .input_scale(scale)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channel_tile)
            .input_zero_point(zero_point)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_output_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channel_tile)
            .output_scale(scale)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_output_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channel_tile)
            .output_zero_point(zero_point)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channel_tile)
        .qmin(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const size_t channel_tile = ${CHANNEL_SCALED_TILE};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
        .channels(channel_tile)
        .qmax(128)
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
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
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
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*8)})
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(channel_tile*8)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_offset(${next_prime(CHANNEL_TILE*8)})
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_offset(channel_tile*8)
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }

    $if DATATYPE == "qu8":
      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .input_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .input_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .input_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .input_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_output_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .output_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .output_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }

      TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_output_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .output_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .output_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }

    TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*8}; channels += ${CHANNEL_TILE}) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmin(128)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmin(128)
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
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmax(128)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmax(128)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_zero) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        for (size_t channels = 1; channels < channel_tile; channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(channel_tile)
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      }
    }

    $if DATATYPE == "qu8":
      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
          for (size_t channels = 1; channels < channel_tile; channels++) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
          for (size_t channels = 1; channels < channel_tile; channels++) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_output_scale) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
          for (size_t channels = 1; channels < channel_tile; channels++) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        }
      }

      TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_output_zero_point) {
        $if ISA_CHECK:
          ${ISA_CHECK};
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
          for (size_t channels = 1; channels < channel_tile; channels++) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        }
      }

    TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const size_t channel_tile = ${CHANNEL_SCALED_TILE};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        for (size_t channels = 1; channels < channel_tile; channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(128)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmax(128)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(${next_prime(CHANNEL_TILE*2)})
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*2)})
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(channel_tile*2)
              .zero_index_mod2(zero_index_mod2)
              .Test(${", ".join(TEST_ARGS)});
          }
        }
    }
  }

  $if DATATYPE == "qu8":
    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_input_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_output_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
            for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_scale(scale)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }

    TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_output_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
            for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
              AvgPoolMicrokernelTester()
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .output_zero_point(zero_point)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}${CHANNEL_SUFFIX}_multipass_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pooling_elements = ${PRIMARY_TILE+INCREMENTAL_TILE+1}; pooling_elements <= ${PRIMARY_TILE+INCREMENTAL_TILE*3}; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
        for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(128)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmin(128)
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
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmax(128)
            .Test(${", ".join(TEST_ARGS)});
        }
      $else:
        const size_t channel_tile = ${CHANNEL_SCALED_TILE};
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
            .channels(channels)
            .qmax(128)
            .Test(${", ".join(TEST_ARGS)});
        }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
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
      for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(${next_prime(CHANNEL_TILE*5+1)})
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .input_offset(channel_tile*5+1)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels_with_zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_offset(${next_prime(CHANNEL_TILE*5+1)})
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .channels(channels)
                .input_offset(channel_tile*5+1)
                .zero_index_mod2(zero_index_mod2)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }
  }

  $if DATATYPE == "qu8":
    TEST(${TEST_NAME}, few_output_pixels_with_input_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
        for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .input_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .input_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }
    }

    TEST(${TEST_NAME}, few_output_pixels_with_input_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
        for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .input_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .input_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }
    }

    TEST(${TEST_NAME}, few_output_pixels_with_output_scale) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
        for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .output_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
              for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .output_scale(scale)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }
    }

    TEST(${TEST_NAME}, few_output_pixels_with_output_zero_point) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
        for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
          $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
            for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .output_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
          $else:
            const size_t channel_tile = ${CHANNEL_SCALED_TILE};
            for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
              for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
                AvgPoolMicrokernelTester()
                  .output_pixels(output_pixels)
                  .pooling_elements(pooling_elements)
                  .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                  .channels(channels)
                  .output_zero_point(zero_point)
                  .Test(${", ".join(TEST_ARGS)});
              }
            }
        }
      }
    }

  TEST(${TEST_NAME}, few_output_pixels_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmin(128)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmin(128)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmax(128)
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .qmax(128)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels_with_output_stride) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_stride(${next_prime(CHANNEL_TILE*5+1)})
              .Test(${", ".join(TEST_ARGS)});
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
              .channels(channels)
              .output_stride(channel_tile*5+1)
              .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
  }

  TEST(${TEST_NAME}, few_output_pixels_with_step) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{${PRIMARY_TILE+1}, ${PRIMARY_TILE+INCREMENTAL_TILE-1}, ${PRIMARY_TILE+INCREMENTAL_TILE+1}}}) {
        $if CHANNEL_SCALED_TILE == CHANNEL_TILE:
          for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
            for (size_t step = 2; step <= pooling_elements; step++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .step(step)
                .channels(channels)
                .output_stride(${next_prime(CHANNEL_TILE*5+1)})
                .Test(${", ".join(TEST_ARGS)});
            }
          }
        $else:
          const size_t channel_tile = ${CHANNEL_SCALED_TILE};
          for (size_t channels = 1; channels <= 3 * channel_tile; channels += channel_tile-1) {
            for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(${PRIMARY_TILE}, ${INCREMENTAL_TILE})
                .step(step)
                .channels(channels)
                .output_stride(channel_tile*5+1)
                .Test(${", ".join(TEST_ARGS)});
            }
          }
      }
    }
  }
"""


def generate_test_cases(ukernel, init_fn, requantization_type, primary_tile,
                        incremental_tile, channel_tile, vector_tile, isa):
  """Generates all tests cases for a AVGPOOL micro-kernel.

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


#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pavgpool.h"
#include "avgpool-microkernel-tester.h"
#include "next_prime.h"
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
