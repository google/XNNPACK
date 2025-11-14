#include "src/configs/hardware_utils.h"

#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/log.h"
#include <benchmark/benchmark.h>

bool xnn_set_cache_data(struct xnn_hardware_config* hardware_config) {
  // Get the CPUInfo.
  const benchmark::CPUInfo& cpu_info = benchmark::CPUInfo::Get();

  // Populate the `hardware_config` fields with it.
  for (const auto& cache : cpu_info.caches) {
    if (cache.level == 1 && (cache.type == "Data" || cache.type == "Unified")) {
      hardware_config->l1_data_cache_bytes = cache.size;
      xnn_log_info(
          "l1_data_cache_bytes=%zu, l1_data_cache_line_size=%zu, "
          "l1_data_cache_associativity=%zu, l1_data_cache_num_sets=%zu.",
          hardware_config->l1_data_cache_bytes,
          hardware_config->l1_data_cache_line_size,
          hardware_config->l1_data_cache_associativity,
          hardware_config->l1_data_cache_num_sets);
    } else if (cache.level == 2 &&
               (cache.type == "Data" || cache.type == "Unified")) {
      hardware_config->l2_data_cache_bytes = cache.size;
      xnn_log_info(
          "l2_data_cache_bytes=%zu, l2_data_cache_line_size=%zu, "
          "l2_data_cache_associativity=%zu, l2_data_cache_num_sets=%zu.",
          hardware_config->l2_data_cache_bytes,
          hardware_config->l2_data_cache_line_size,
          hardware_config->l2_data_cache_associativity,
          hardware_config->l2_data_cache_num_sets);
    }
  }
  return true;
}
