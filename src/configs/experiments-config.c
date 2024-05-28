// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <experiments-config.h>

static struct xnn_experiment_config experiment_config = {0};

struct xnn_experiment_config* xnn_get_experiment_config() {
  return &experiment_config;
}

