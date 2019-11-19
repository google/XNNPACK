// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdlib.h>

#include <xnnpack.h>


// A dummy program that calls every function in XNNPACK, for size estimation.
int main(int argc, char** argv) {
  int function_idx = 0;
  if (argc >= 2) {
    function_idx = atoi(argv[1]);
  }

  xnn_initialize(NULL /* allocator */);

  xnn_operator_t op = NULL;
  switch (function_idx) {
    case -1:
      xnn_delete_operator(op);
      break;
    case 0:
      xnn_run_operator(op, NULL);
      break;
    case 1:
      xnn_create_convolution2d_nhwc_f32(
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0, 0,
        0, 0,
        NULL, NULL,
        -INFINITY, INFINITY,
        0,
        &op);
      break;
    case 2:
      xnn_setup_convolution2d_nhwc_f32(
        op,
        0, 0, 0,
        NULL, NULL,
        NULL);
      break;
    case 3:
      xnn_create_deconvolution2d_nhwc_f32(
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0, 0,
        0, 0,
        NULL, NULL,
        -INFINITY, INFINITY,
        0,
        &op);
      break;
    case 4:
      xnn_setup_deconvolution2d_nhwc_f32(
        op,
        0, 0, 0,
        0, 0,
        NULL, NULL,
        NULL);
      break;
    case 5:
      xnn_create_fully_connected_nc_f32(
        0, 0, 0, 0, NULL, NULL,
        -INFINITY, INFINITY, 0, &op);
      break;
    case 6:
      xnn_setup_fully_connected_nc_f32(
        op, 0, NULL, NULL, NULL);
      break;
    case 7:
      xnn_create_global_average_pooling_nwc_f32(
        0, 0, 0,
        -INFINITY, INFINITY, 0, &op);
      break;
    case 8:
      xnn_setup_global_average_pooling_nwc_f32(
        op, 0, 0, NULL, NULL, NULL);
      break;
    case 9:
      xnn_create_average_pooling2d_nhwc_f32(
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0, 0, 0,
        -INFINITY, INFINITY,
        0, &op);
      break;
    case 10:
      xnn_setup_average_pooling2d_nhwc_f32(
        op, 0, 0, 0, NULL, NULL, NULL);
      break;
    case 11:
      xnn_create_max_pooling2d_nhwc_f32(
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0, 0,
        -INFINITY, INFINITY,
        0, &op);
      break;
    case 12:
      xnn_setup_max_pooling2d_nhwc_f32(
        op, 0, 0, 0, NULL, NULL, NULL);
      break;
    case 13:
      xnn_create_argmax_pooling2d_nhwc_f32(
        0, 0, 0, 0,
        0, 0,
        0, 0, 0,
        -INFINITY, INFINITY,
        0, &op);
      break;
    case 14:
      xnn_setup_argmax_pooling2d_nhwc_f32(
        op, 0, 0, 0, NULL, NULL, NULL, NULL);
      break;
    case 15:
      xnn_create_unpooling2d_nhwc_x32(
        0, 0, 0, 0,
        0, 0,
        0, 0, 0,
        0, &op);
      break;
    case 17:
      xnn_setup_unpooling2d_nhwc_x32(
        op, 0, 0, 0, NULL, NULL, NULL, NULL);
      break;
    case 18:
      xnn_create_channel_shuffle_nc_x32(
        0, 0, 0, 0,
        0, &op);
      break;
    case 19:
      xnn_setup_channel_shuffle_nc_x32(
        op, 0, NULL, NULL, NULL);
      break;
    case 20:
      xnn_create_add_nc_f32(
        0, 0, 0, 0,
        -INFINITY, INFINITY,
        0, &op);
      break;
    case 21:
      xnn_setup_add_nc_f32(
        op, 0, NULL, NULL, NULL, NULL);
      break;
    case 22:
      xnn_create_channel_pad_nc_x32(
        0, 0, 0, 0, 0, NULL, 0, &op);
      break;
    case 23:
      xnn_setup_channel_pad_nc_x32(
        op, 0, NULL, NULL, NULL);
      break;
    case 24:
      xnn_create_clamp_nc_f32(
        0, 0, 0,
        -INFINITY, INFINITY,
        0, &op);
      break;
    case 25:
      xnn_setup_clamp_nc_f32(
        op, 0, NULL, NULL, NULL);
      break;
    case 26:
      xnn_create_hardswish_nc_f32(
        0, 0, 0, 0, &op);
      break;
    case 27:
      xnn_setup_hardswish_nc_f32(
        op, 0, NULL, NULL, NULL);
      break;
    case 28:
      xnn_create_prelu_nc_f32(
        0, 0, 0, NULL, 0, 0, 0, &op);
      break;
    case 29:
      xnn_setup_prelu_nc_f32(
        op, 0, NULL, NULL, NULL);
      break;
  }

  xnn_deinitialize();
}
