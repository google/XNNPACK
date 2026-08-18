// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root of this source tree.

#include "src/xnnpack/microkernel-name-registry.h"

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/cache.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/mutex.h"

// Open-addressing table sized for the ~hundreds of distinct ukernels any
// single build of XNNPACK registers (bounded by the union of XNN_INIT_*
// _UKERNEL call sites across enabled ISAs). 4096 slots keeps load < 50%
// with margin and avoids ever growing.
#define XNN_NAME_TABLE_SIZE 4096
#define XNN_NAME_TABLE_MASK (XNN_NAME_TABLE_SIZE - 1)

struct entry {
  const void* fn_ptr;
  const char* name;
};

static struct entry name_table[XNN_NAME_TABLE_SIZE];

static struct xnn_mutex mutex;
XNN_INIT_ONCE_GUARD(mutex);
static void init_mutex_config() { xnn_mutex_init(&mutex); }

static inline uint32_t hash_fn_ptr(const void* fn_ptr) {
  return murmur_hash3(&fn_ptr, sizeof(fn_ptr), /*seed=*/7);
}

void xnn_register_microkernel_name(const void* fn_ptr, const char* name) {
  if (fn_ptr == NULL || name == NULL) {
    return;
  }
  XNN_INIT_ONCE(mutex);
  xnn_mutex_lock(&mutex);
  uint32_t slot = hash_fn_ptr(fn_ptr) & XNN_NAME_TABLE_MASK;
  for (size_t probes = 0; probes < XNN_NAME_TABLE_SIZE; probes++) {
    if (name_table[slot].fn_ptr == NULL) {
      name_table[slot].name = name;
      name_table[slot].fn_ptr = fn_ptr;
      break;
    }
    if (name_table[slot].fn_ptr == fn_ptr) {
      break;
    }
    slot = (slot + 1) & XNN_NAME_TABLE_MASK;
  }
  xnn_mutex_unlock(&mutex);
}

const char* xnn_lookup_microkernel_name(const void* fn_ptr) {
  if (fn_ptr == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(mutex);
  xnn_mutex_lock(&mutex);
  const char* name = NULL;
  uint32_t slot = hash_fn_ptr(fn_ptr) & XNN_NAME_TABLE_MASK;
  for (size_t probes = 0; probes < XNN_NAME_TABLE_SIZE; probes++) {
    if (name_table[slot].fn_ptr == NULL) {
      break;
    }
    if (name_table[slot].fn_ptr == fn_ptr) {
      name = name_table[slot].name;
      break;
    }
    slot = (slot + 1) & XNN_NAME_TABLE_MASK;
  }
  xnn_mutex_unlock(&mutex);
  return name;
}
