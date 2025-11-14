#ifndef XNNPACK_SRC_CONFIGS_HARDWARE_UTILS_H_
#define XNNPACK_SRC_CONFIGS_HARDWARE_UTILS_H_

#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"

#ifdef __cplusplus
extern "C" {
#endif

XNN_INTERNAL bool xnn_set_cache_data(
    struct xnn_hardware_config* hardware_config);

#ifdef __cplusplus
}
#endif

#endif  // XNNPACK_SRC_CONFIGS_HARDWARE_UTILS_H_
