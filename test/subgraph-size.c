// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdlib.h>

#include <xnnpack.h>


// A dummy program that calls every Subgraph API function in XNNPACK, for size estimation.
int main(int argc, char** argv) {
  int function_idx = 0;
  if (argc >= 2) {
    function_idx = atoi(argv[1]);
  }

  xnn_initialize(NULL /* allocator */);

  switch (function_idx) {
    case 0:
      xnn_create_subgraph(0, 0, NULL);
      break;
    case 1:
      xnn_delete_subgraph(NULL);
      break;
    case 2:
      xnn_define_tensor_value(NULL, xnn_datatype_invalid, 0, NULL, NULL, 0, 0, NULL);
      break;
    case 3:
      xnn_define_convolution_2d(
        NULL,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0, 0,
        0.0f, 0.0f,
        0, 0, 0, 0, 0);
      break;
    case 4:
      xnn_define_depthwise_convolution_2d(
        NULL,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0.0f, 0.0f,
        0, 0, 0, 0, 0);
      break;
    case 5:
      xnn_define_average_pooling_2d(
        NULL,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0.0f, 0.0f,
        0, 0, 0);
      break;
    case 6:
      xnn_define_max_pooling_2d(
        NULL,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0, 0,
        0.0f, 0.0f,
        0, 0, 0);
      break;
    case 7:
      xnn_define_add2(NULL, 0.0f, 0.0f, 0, 0, 0, 0);
      break;
    case 8:
      xnn_define_multiply2(NULL, 0.0f, 0.0f, 0, 0, 0, 0);
      break;
    case 9:
      xnn_define_prelu(NULL, 0, 0, 0, 0);
      break;
    case 10:
      xnn_define_clamp(NULL, 0.0f, 0.0f, 0, 0, 0);
      break;
    case 11:
      xnn_define_hardswish(NULL, 0, 0, 0);
      break;
    case 12:
      xnn_define_sigmoid(NULL, 0, 0, 0);
      break;
    case 13:
      xnn_define_softmax(NULL, 0, 0, 0);
      break;
    case 14:
      xnn_create_runtime_v2(NULL, NULL, 0, NULL);
      break;
    case 15:
      xnn_setup_runtime(NULL, 0, NULL);
      break;
    case 16:
      xnn_invoke_runtime(NULL);
      break;
    case 17:
      xnn_delete_runtime(NULL);
      break;
  }

  xnn_deinitialize();
}
