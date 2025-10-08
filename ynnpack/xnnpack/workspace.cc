// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "include/xnnpack.h"

extern "C" {

xnn_status xnn_create_workspace(xnn_workspace_t* workspace_out) {
  *workspace_out = (xnn_workspace_t)3;
  return xnn_status_success;
}
xnn_status xnn_release_workspace(xnn_workspace_t workspace) {
  return xnn_status_success;
}

}  // extern "C"
