// clang-format off

#include "ynnpack/base/build_config.h"

YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_fp32, float, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_fp16_fp32, half, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_bf16_fp32, bfloat16, float)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_int8_int32, int8_t, int32_t)
YNN_UNARY_REDUCE_KERNEL(arch_flag::none, sum_squared_uint8_int32, uint8_t, int32_t)
