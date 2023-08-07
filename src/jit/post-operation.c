// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/allocator.h>
#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/microparams.h>
#include <xnnpack/params.h>
#include <xnnpack/post-operation.h>

char* allocate_and_initialize_post_operation_params(
    size_t num_post_operations,
    const struct xnn_post_operation* post_operations) {

  union {
    union xnn_f32_hswish_params hswish_params;
  } post_op_params;  // Anonymous union to hold params of all valid post operations.

  // Calculate how much space all post operation params will take.
  size_t total_size = 0;
  for (size_t i = 0; i < num_post_operations; i++) {
    const struct xnn_post_operation post_op = post_operations[i];
    switch (post_op.op_type) {
      case xnn_post_operation_type_hardswish:
      {
        const struct xnn_unary_elementwise_config* f32_hswish_config = xnn_init_f32_hswish_config();
        if (f32_hswish_config->init.f32_hswish != NULL) {
          total_size += f32_hswish_config->init.f32_hswish(&post_op_params.hswish_params);
        }
        break;
      }
      default:
        XNN_LOG_UNREACHABLE("unsupported post operation: %u", post_op.op_type);
    }
  }
  // Copy all params compactly into post_operation_params.
  char* post_operation_params = xnn_allocate_zero_memory(total_size);
  char* cur_params = post_operation_params;
  for (size_t i = 0; i < num_post_operations; i++) {
    const struct xnn_post_operation post_op = post_operations[i];
    switch (post_op.op_type) {
      case xnn_post_operation_type_hardswish:
      {
        const struct xnn_unary_elementwise_config* f32_hswish_config = xnn_init_f32_hswish_config();
        if (f32_hswish_config->init.f32_hswish!= NULL) {
          const size_t initialized_size = f32_hswish_config->init.f32_hswish(&post_op_params.hswish_params);
          memcpy(cur_params, &post_op_params.hswish_params, initialized_size);
          cur_params += initialized_size;
        }
        break;
      }
      default:
        XNN_LOG_UNREACHABLE("unsupported post operation: %u", post_op.op_type);
    }
  }
  return post_operation_params;
}

