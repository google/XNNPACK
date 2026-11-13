// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-hmx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <HAP_compute_res.h>
#include <HAP_power.h>
#include <assert.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>
#include <pthread.h>
#include <qurt.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/unaligned.h"

#if __HEXAGON_ARCH__ >= 73

// HMX activation operand sizes are limited to 64 KB (32 tiles of 2048 bytes,
// K = 1024) per issue on Hexagon V81; earlier generations stall above ~32 KB
// (K = 512), so `max_k_chunk` is chosen at init.
#define XNN_HMX_MAX_K_CHUNK 1024
#define XNN_HMX_MAX_ACT_K 9216
#define XNN_HMX_MAX_SLOTS 4
#define XNN_HMX_SLOT_BYTES (720 * 1024)

static XNN_INLINE void l2fetch_2d(const void* p, uint32_t stride,
                                  uint32_t width, uint32_t height) {
  const uint64_t control = ((uint64_t) stride << 32) |
                           ((uint64_t) width << 16) | (uint64_t) height;
  __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(control));
}

static XNN_INLINE void xnn_hmx_tile_clear(void) {
  __asm__ __volatile__("mxclracc" ::: "memory");
}

static XNN_INLINE void xnn_hmx_mpy_ub_cm_b(const void* act_addr,
                                           uint32_t act_ctrl,
                                           const void* wgt_addr,
                                           uint32_t wgt_ctrl) {
  __asm__ __volatile__(
      "{\n\t"
      "  activation.ub = mxmem(%0, %1):deep:cm\n\t"
      "  weight.b = mxmem(%2, %3)\n\t"
      "}"
      :
      : "r"(act_addr), "r"(act_ctrl), "r"(wgt_addr), "r"(wgt_ctrl)
      : "memory");
}

// Extract exact 24-bit signed accumulators from odd `:cm` pegs (with even pegs
// zeroed, scale S = 16 at `bias_lo16_addr`, and scale S = 8 at
// `bias_hi8_addr`): `cvt.uh:2x1` with retain (0x0003) writes bits [7:0] to even
// pegs and bits [15:8] to odd pegs of `out_lo16`, while `cvt.ub` with consume
// (0x0002) writes bits [23:16] to odd pegs of `out_hi8`. Loading each bias
// descriptor immediately before its `cvt` works across all HMX revisions
// (including Hexagon V73, which has a single bias register file).
static XNN_INLINE void xnn_hmx_store_cvt_i24_cm(
    void* out_lo16_addr, void* out_hi8_addr, const void* bias_lo16_addr,
    const void* bias_hi8_addr) {
  __asm__ __volatile__(
      "bias = mxmem(%2)\n\t"
      "cvt.uh = acc(%4):2x1\n\t"
      "mxmem(%0, %6):cm = cvt\n\t"
      "bias = mxmem(%3)\n\t"
      "cvt.ub = acc(%5)\n\t"
      "mxmem(%1, %6):cm = cvt"
      :
      : "r"(out_lo16_addr), "r"(out_hi8_addr), "r"(bias_lo16_addr),
        "r"(bias_hi8_addr), "r"((uint32_t) 0x0003), "r"((uint32_t) 0x0002),
        "r"((uint32_t) 0x7FF)
      : "memory");
}

// Drain preceding HVX/HMX stores to VTCM before subsequent reads/writes.
static XNN_INLINE void xnn_hvx_vtcm_fence(const void* vtcm_ptr) {
  HVX_Vector vdummy;
  __asm__ __volatile__("%0 = vmem(%1+#0)"
                       : "=v"(vdummy)
                       : "r"(vtcm_ptr)
                       : "memory");
}

// 64B-aligned Hexagon 2D User DMA (`dm0`) descriptor (`DESC_DESCTYPE_2D`).
struct XNN_ALIGN(64) xnn_hmx_dma_desc_2d {
  uint32_t next;
  uint32_t dstate_order_bypass_desctype_length;
  uint32_t src;
  uint32_t dst;
  uint32_t allocation;
  uint32_t roiheight_roiwidth;
  uint32_t dststride_srcstride;
  uint32_t dstwidthoffset_srcwidthoffset;
  uint32_t pad[8];
};

struct xnn_hmx_worker_slot {
  XNN_ALIGN(64) struct xnn_hmx_dma_desc_2d dma_descs[2][2];
  pthread_mutex_t mutex;
  uint8_t* vtcm_act0;
  HVX_Vector* vtcm_wgt;
  HVX_Vector* vtcm_out;
  HVX_Vector* vtcm_acc;
};

#define XNN_HMX_AUTONOMOUS_PREEMPT_MASK (2U | 4U | 8U)

#ifdef HAP_COMPUTE_RES_AUTONOMOUS_PREEMPTION
typedef compute_res_hmx_mutex_t xnn_hmx_crm_mutex_t;
#else
typedef enum {
  HAP_COMPUTE_RES_PREEMPTION_CAPABILITY = 1,
} xnn_compute_res_capability_id_t;
typedef struct {
  void* mutex;
  void (*lock)(void* mutex);
  void (*unlock)(void* mutex);
} xnn_hmx_crm_mutex_t;
int __attribute__((weak)) crm_query_capability(
    xnn_compute_res_capability_id_t capability_id, unsigned int* data);
int __attribute__((weak)) crm_hmx_lock3(unsigned int context_id,
                                        compute_res_hmx_type_t type,
                                        xnn_hmx_crm_mutex_t* mutex,
                                        unsigned int timeout_us);
int __attribute__((weak)) crm_hmx_unlock3(unsigned int context_id,
                                          compute_res_hmx_type_t type,
                                          xnn_hmx_crm_mutex_t* mutex);
#endif

enum xnn_hmx_lock_mode {
  XNN_HMX_LOCK_NONE = 0,
  XNN_HMX_LOCK_SHARED_V3 = 1,
  XNN_HMX_LOCK_SHARED_V2 = 2,
  XNN_HMX_LOCK_SINGLE_THREAD = 3,
};

struct xnn_hmx_vtcm_context {
  XNN_ALIGN(128) struct xnn_hmx_worker_slot slots[XNN_HMX_MAX_SLOTS];
  pthread_mutex_t init_mutex;
  pthread_mutex_t hmx_exec_mutex;
  pthread_mutex_t dma_mutex;
  bool initialized;
  bool init_failed;
  enum xnn_hmx_lock_mode lock_mode;
  uint32_t context_id;
  int power_ctx;
  uint8_t* vtcm_base;
  HVX_Vector* vtcm_bias;
  size_t num_slots;
  size_t max_k_chunk;
  void* last_dma_desc;
  qurt_thread_t locked_tids[64];
  xnn_hmx_crm_mutex_t locked_crm_mutex[64];
  size_t num_locked_tids;
};

__attribute__((weak)) struct xnn_hmx_vtcm_context xnn_hmx_ctx = {
    .init_mutex = PTHREAD_MUTEX_INITIALIZER,
    .hmx_exec_mutex = PTHREAD_MUTEX_INITIALIZER,
    .dma_mutex = PTHREAD_MUTEX_INITIALIZER,
};

// Must be called with `xnn_hmx_ctx.init_mutex` held.
static bool xnn_hmx_lock_thread(xnn_hmx_crm_mutex_t* out_crm_mutex) {
  const qurt_thread_t tid = qurt_thread_get_id();
  for (size_t i = 0; i < xnn_hmx_ctx.num_locked_tids; ++i) {
    if (xnn_hmx_ctx.locked_tids[i] == tid) {
      *out_crm_mutex = xnn_hmx_ctx.locked_crm_mutex[i];
      return true;
    }
  }
  xnn_hmx_crm_mutex_t crm_mutex = {0};
  switch (xnn_hmx_ctx.lock_mode) {
    case XNN_HMX_LOCK_SHARED_V3:
      if (crm_hmx_lock3 == NULL ||
          crm_hmx_lock3(xnn_hmx_ctx.context_id, HAP_COMPUTE_RES_HMX_SHARED,
                        &crm_mutex, 100000) != 0) {
        return false;
      }
      break;
    case XNN_HMX_LOCK_SHARED_V2:
      if (HAP_compute_res_hmx_lock2(xnn_hmx_ctx.context_id,
                                    HAP_COMPUTE_RES_HMX_SHARED) != 0) {
        return false;
      }
      break;
    default:
      return false;
  }
  if (xnn_hmx_ctx.num_locked_tids < 64) {
    const size_t idx = xnn_hmx_ctx.num_locked_tids++;
    xnn_hmx_ctx.locked_tids[idx] = tid;
    xnn_hmx_ctx.locked_crm_mutex[idx] = crm_mutex;
  }
  *out_crm_mutex = crm_mutex;
  return true;
}

// Must be called with `xnn_hmx_ctx.init_mutex` held.
static bool xnn_init_hmx_vtcm(void) {
  HAP_power_request_t request = {
      .type = HAP_power_set_apptype,
      .apptype = HAP_POWER_COMPUTE_CLIENT_CLASS,
  };
  HAP_power_set(&xnn_hmx_ctx.power_ctx, &request);

  request = (HAP_power_request_t) {
      .type = HAP_power_set_DCVS_v3,
      .dcvs_v3 = {
          .set_dcvs_enable = 1,
          .dcvs_enable = 1,
          .dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE,
          .set_latency = 1,
          .latency = 100,
          .set_core_params = 1,
          .core_params = {
              .min_corner = HAP_DCVS_VCORNER_MAX,
              .max_corner = HAP_DCVS_VCORNER_MAX,
              .target_corner = HAP_DCVS_VCORNER_MAX,
          },
          .set_bus_params = 1,
          .bus_params = {
              .min_corner = HAP_DCVS_VCORNER_MAX,
              .max_corner = HAP_DCVS_VCORNER_MAX,
              .target_corner = HAP_DCVS_VCORNER_MAX,
          },
          .set_sleep_disable = 1,
          .sleep_disable = HAP_DCVS_LPM_LEVEL1,
      },
  };
  if (HAP_power_set(&xnn_hmx_ctx.power_ctx, &request) != 0) {
    request = (HAP_power_request_t) {
        .type = HAP_power_set_DCVS_v2,
        .dcvs_v2 = {
            .dcvs_enable = 1,
            .dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE,
            .set_latency = 1,
            .latency = 100,
            .set_dcvs_params = 1,
            .dcvs_params = {
                .min_corner = HAP_DCVS_VCORNER_MAX,
                .max_corner = HAP_DCVS_VCORNER_MAX,
                .target_corner = HAP_DCVS_VCORNER_MAX,
            },
        },
    };
    HAP_power_set(&xnn_hmx_ctx.power_ctx, &request);
  }

  request = (HAP_power_request_t) {
      .type = HAP_power_set_HVX,
      .hvx.power_up = 1,
  };
  HAP_power_set(&xnn_hmx_ctx.power_ctx, &request);

  request = (HAP_power_request_t) {
      .type = HAP_power_set_HMX,
      .hmx.power_up = 1,
  };
  if (HAP_power_set(&xnn_hmx_ctx.power_ctx, &request) != 0) {
    HAP_power_destroy(&xnn_hmx_ctx.power_ctx);
    return false;
  }

  // VTCM layout (all sections 2048B-aligned):
  //   [0 .. 2KB): shared bias/scale descriptor (2 KB)
  //   Per worker slot (720 KB each, up to XNN_HMX_MAX_SLOTS = 4 slots):
  //     [+0KB   .. +576KB): act0: 32 rows at odd `:cm` pegs, up to K=9216
  //     [+576KB .. +704KB): wgt: 2 x 64KB DMA ping-pong buffers (128 KB)
  //     [+704KB .. +712KB): out: 4 x 2048B `:cm` readout tiles (8 KB)
  //     [+712KB .. +720KB): acc: 32 x 64 x int32 multi-chunk accumulator (8 KB)
  // Pad `compute_res_attr_t` (64B in SDK v5, 160B in SDK v6) so binaries built
  // against older headers do not overflow stack when run on newer DSP firmware.
  union {
    compute_res_attr_t attr;
    unsigned long long pad[24];
  } compute_res;
  size_t acquired_slots = 0;
  unsigned int acquired_bytes = 0;

  unsigned int preempt_caps = 0;
  const bool has_auto_preempt =
      (crm_query_capability != NULL && crm_hmx_lock3 != NULL &&
       crm_query_capability(HAP_COMPUTE_RES_PREEMPTION_CAPABILITY,
                            &preempt_caps) == 0 &&
       (preempt_caps & XNN_HMX_AUTONOMOUS_PREEMPT_MASK) != 0);

  if (has_auto_preempt) {
    for (size_t s = XNN_HMX_MAX_SLOTS; s >= 1; s >>= 1) {
      const unsigned int try_bytes =
          (unsigned int) (4096 + s * XNN_HMX_SLOT_BYTES);
      memset(&compute_res, 0, sizeof(compute_res));
      HAP_compute_res_attr_init(&compute_res.attr);
      HAP_compute_res_attr_set_vtcm_param(&compute_res.attr, try_bytes, 1);
      xnn_hmx_ctx.context_id =
          HAP_compute_res_acquire(&compute_res.attr, 100000);
      if (xnn_hmx_ctx.context_id != 0) {
        acquired_slots = s;
        acquired_bytes = try_bytes;
        break;
      }
    }
    if (xnn_hmx_ctx.context_id != 0) {
      xnn_hmx_crm_mutex_t crm_mutex = {0};
      if (crm_hmx_lock3(xnn_hmx_ctx.context_id, HAP_COMPUTE_RES_HMX_SHARED,
                        &crm_mutex, 100000) == 0) {
        xnn_hmx_ctx.lock_mode = XNN_HMX_LOCK_SHARED_V3;
        xnn_hmx_ctx.locked_crm_mutex[0] = crm_mutex;
      } else {
        HAP_compute_res_release(xnn_hmx_ctx.context_id);
        xnn_hmx_ctx.context_id = 0;
        acquired_slots = 0;
        acquired_bytes = 0;
      }
    }
  }

  if (xnn_hmx_ctx.context_id == 0) {
    for (size_t s = XNN_HMX_MAX_SLOTS; s >= 1; s >>= 1) {
      const unsigned int try_bytes =
          (unsigned int) (4096 + s * XNN_HMX_SLOT_BYTES);
      memset(&compute_res, 0, sizeof(compute_res));
      HAP_compute_res_attr_init(&compute_res.attr);
      HAP_compute_res_attr_set_vtcm_param(&compute_res.attr, try_bytes, 1);
      HAP_compute_res_attr_set_hmx_param(&compute_res.attr, 1);
      xnn_hmx_ctx.context_id =
          HAP_compute_res_acquire(&compute_res.attr, 100000);
      if (xnn_hmx_ctx.context_id != 0) {
        acquired_slots = s;
        acquired_bytes = try_bytes;
        break;
      }
    }
    if (xnn_hmx_ctx.context_id == 0) {
      HAP_power_destroy(&xnn_hmx_ctx.power_ctx);
      return false;
    }
    if (HAP_compute_res_hmx_lock2(xnn_hmx_ctx.context_id,
                                  HAP_COMPUTE_RES_HMX_SHARED) == 0) {
      xnn_hmx_ctx.lock_mode = XNN_HMX_LOCK_SHARED_V2;
    } else if (HAP_compute_res_hmx_lock(xnn_hmx_ctx.context_id) == 0) {
      xnn_hmx_ctx.lock_mode = XNN_HMX_LOCK_SINGLE_THREAD;
    } else {
      HAP_compute_res_release(xnn_hmx_ctx.context_id);
      xnn_hmx_ctx.context_id = 0;
      HAP_power_destroy(&xnn_hmx_ctx.power_ctx);
      return false;
    }
  }

  xnn_hmx_ctx.vtcm_base =
      (uint8_t*) HAP_compute_res_attr_get_vtcm_ptr(&compute_res.attr);
  if (xnn_hmx_ctx.vtcm_base == NULL) {
    HAP_compute_res_release(xnn_hmx_ctx.context_id);
    xnn_hmx_ctx.context_id = 0;
    HAP_power_destroy(&xnn_hmx_ctx.power_ctx);
    return false;
  }
  xnn_hmx_ctx.locked_tids[0] = qurt_thread_get_id();
  xnn_hmx_ctx.num_locked_tids = 1;
  xnn_hmx_ctx.num_slots =
      (xnn_hmx_ctx.lock_mode == XNN_HMX_LOCK_SINGLE_THREAD) ? 1 : acquired_slots;
  memset(xnn_hmx_ctx.vtcm_base, 0, acquired_bytes);

  qurt_arch_version_t arch = {0};
  const unsigned int arch_version =
      (qurt_sysenv_get_arch_version(&arch) == QURT_EOK)
          ? (arch.arch_version & 0xFF)
          : 0;
  xnn_hmx_ctx.max_k_chunk = arch_version >= 0x81 ? XNN_HMX_MAX_K_CHUNK
                                                 : XNN_HMX_MAX_K_CHUNK / 2;

  const uintptr_t vtcm_aligned =
      ((uintptr_t) xnn_hmx_ctx.vtcm_base + 2047) & ~(uintptr_t) 2047;
  xnn_hmx_ctx.vtcm_bias = (HVX_Vector*) vtcm_aligned;
  for (size_t i = 0; i < xnn_hmx_ctx.num_slots; ++i) {
    pthread_mutex_init(&xnn_hmx_ctx.slots[i].mutex, NULL);
    const uintptr_t slot_base = vtcm_aligned + 2048 + i * XNN_HMX_SLOT_BYTES;
    xnn_hmx_ctx.slots[i].vtcm_act0 = (uint8_t*) slot_base;
    xnn_hmx_ctx.slots[i].vtcm_wgt = (HVX_Vector*) (slot_base + 589824);
    xnn_hmx_ctx.slots[i].vtcm_out =
        (HVX_Vector*) (slot_base + 589824 + 131072);
    xnn_hmx_ctx.slots[i].vtcm_acc =
        (HVX_Vector*) (slot_base + 589824 + 131072 + 8192);
  }

  // Area 0 uses S = 16 (16 << 10 = 0x4000) to extract bits [15:0] via
  // `cvt.uh:2x1`, and Area 1 uses S = 8 (8 << 10 = 0x2000) to extract bits
  // [23:16] via `cvt.ub`.
  xnn_hmx_ctx.vtcm_bias[0] = Q6_V_vsplat_R(0x4000);
  xnn_hmx_ctx.vtcm_bias[1] = Q6_V_vsplat_R(0x2000);
  xnn_hvx_vtcm_fence(&xnn_hmx_ctx.vtcm_bias[1]);

  xnn_hmx_ctx.initialized = true;
  return true;
}

__attribute__((destructor)) static void xnn_deinit_hmx_vtcm(void) {
  if (!xnn_hmx_ctx.initialized && !xnn_hmx_ctx.init_failed) {
    return;
  }
  const bool was_initialized = xnn_hmx_ctx.initialized;
  xnn_hmx_ctx.initialized = false;
  xnn_hmx_ctx.init_failed = false;
  if (was_initialized) {
    if (xnn_hmx_ctx.last_dma_desc != NULL) {
      uint32_t dummy = 0;
      __asm__ __volatile__("%0 = dmwait" : "=r"(dummy) : : "memory");
      xnn_hmx_ctx.last_dma_desc = NULL;
    }
    if (xnn_hmx_ctx.num_locked_tids > 0 &&
        xnn_hmx_ctx.locked_tids[0] == qurt_thread_get_id()) {
      switch (xnn_hmx_ctx.lock_mode) {
        case XNN_HMX_LOCK_SHARED_V3:
          if (crm_hmx_unlock3 != NULL) {
            crm_hmx_unlock3(xnn_hmx_ctx.context_id, HAP_COMPUTE_RES_HMX_SHARED,
                            &xnn_hmx_ctx.locked_crm_mutex[0]);
          }
          break;
        case XNN_HMX_LOCK_SHARED_V2:
          HAP_compute_res_hmx_unlock2(xnn_hmx_ctx.context_id,
                                      HAP_COMPUTE_RES_HMX_SHARED);
          break;
        case XNN_HMX_LOCK_SINGLE_THREAD:
          HAP_compute_res_hmx_unlock(xnn_hmx_ctx.context_id);
          break;
        default:
          break;
      }
    }
    if (xnn_hmx_ctx.context_id != 0) {
      HAP_compute_res_release(xnn_hmx_ctx.context_id);
      xnn_hmx_ctx.context_id = 0;
    }
    HAP_power_destroy(&xnn_hmx_ctx.power_ctx);
    for (size_t i = 0; i < xnn_hmx_ctx.num_slots; ++i) {
      pthread_mutex_destroy(&xnn_hmx_ctx.slots[i].mutex);
    }
    xnn_hmx_ctx.num_slots = 0;
    xnn_hmx_ctx.num_locked_tids = 0;
  }
  pthread_mutex_destroy(&xnn_hmx_ctx.dma_mutex);
  pthread_mutex_destroy(&xnn_hmx_ctx.hmx_exec_mutex);
  pthread_mutex_destroy(&xnn_hmx_ctx.init_mutex);
}

static XNN_INLINE uint32_t xnn_dma_poll(void) {
  uint32_t status = 0;
  __asm__ __volatile__("%0 = dmpoll" : "=r"(status) : : "memory");
  return status;
}

static XNN_INLINE void xnn_dma_release_desc(void* desc) {
  __asm__ __volatile__("release(%0):at" : : "r"(desc) : "memory");
}

static XNN_INLINE void xnn_dma_start(void* desc) {
  __asm__ __volatile__(
      "release(%0):at\n\t"
      "dmstart(%0)"
      :
      : "r"(desc)
      : "memory");
}

static XNN_INLINE void xnn_dma_link(void* tail_desc, void* next_desc) {
  __asm__ __volatile__(
      "release(%1):at\n\t"
      "dmlink(%0, %1)"
      :
      : "r"(tail_desc), "r"(next_desc)
      : "memory");
}

// De-interleave `num_rows` 256B weight rows from DDR/L2 (`w_weights`) into
// contiguous 128B VTCM buffers `dst_wgt0` (cols 0..31) and `dst_wgt1`
// (cols 32..63) asynchronously on Hexagon User DMA engine `dm0`.
// `bypass_src = 0` and `allocation = 0` snoop dirty L2 lines coherently
// without polluting L2 on cold DDR streams.
static bool xnn_dma_submit_wgt_2d(struct xnn_hmx_dma_desc_2d* desc0,
                                  struct xnn_hmx_dma_desc_2d* desc1,
                                  void* dst_wgt0, void* dst_wgt1,
                                  const void* w_weights, size_t num_rows) {
  pthread_mutex_lock(&xnn_hmx_ctx.dma_mutex);
  if (xnn_hmx_ctx.last_dma_desc == desc0 ||
      xnn_hmx_ctx.last_dma_desc == desc1) {
    while ((xnn_dma_poll() & 0x3u) == 1u) {
    }
    xnn_hmx_ctx.last_dma_desc = NULL;
  }

  const uint32_t roi = ((uint32_t) num_rows << 16) | 128u;
  const uint32_t strides = (128u << 16) | 256u;
  desc0->next = (uint32_t) (uintptr_t) desc1;
  desc0->dstate_order_bypass_desctype_length = 0x01000000u;
  desc0->src = (uint32_t) (uintptr_t) w_weights;
  desc0->dst = (uint32_t) (uintptr_t) dst_wgt0;
  desc0->allocation = 0u;
  desc0->roiheight_roiwidth = roi;
  desc0->dststride_srcstride = strides;
  desc0->dstwidthoffset_srcwidthoffset = 0u;

  desc1->next = 0u;
  desc1->dstate_order_bypass_desctype_length = 0x41000000u;
  desc1->src = (uint32_t) ((uintptr_t) w_weights + 128u);
  desc1->dst = (uint32_t) (uintptr_t) dst_wgt1;
  desc1->allocation = 0u;
  desc1->roiheight_roiwidth = roi;
  desc1->dststride_srcstride = strides;
  desc1->dstwidthoffset_srcwidthoffset = 0u;

  xnn_dma_release_desc(desc1);
  const uint32_t dm0_status = xnn_dma_poll() & 0x3u;
  bool ok = true;
  if (dm0_status == 0u) {
    xnn_dma_start(desc0);
    xnn_dma_poll();
    xnn_hmx_ctx.last_dma_desc = desc1;
  } else if (dm0_status == 1u && xnn_hmx_ctx.last_dma_desc != NULL) {
    xnn_dma_link(xnn_hmx_ctx.last_dma_desc, desc0);
    xnn_dma_poll();
    xnn_hmx_ctx.last_dma_desc = desc1;
  } else {
    ok = false;
  }
  pthread_mutex_unlock(&xnn_hmx_ctx.dma_mutex);
  return ok;
}

static XNN_INLINE void xnn_dma_wait_desc(
    const struct xnn_hmx_dma_desc_2d* desc) {
  const volatile uint32_t* dstate_word =
      &desc->dstate_order_bypass_desctype_length;
  while ((*dstate_word & 0x80000000u) == 0u) {
    xnn_dma_poll();
  }
}

static void xnn_copy_wgt_to_hmx_1x(HVX_Vector* restrict dst_wgt0,
                                   HVX_Vector* restrict dst_wgt1,
                                   const void* w_weights, size_t num_rows,
                                   size_t total_vecs) {
  if XNN_LIKELY(((uintptr_t) w_weights & 127u) == 0u) {
    const HVX_Vector* restrict w_vec = (const HVX_Vector*) w_weights;
    for (size_t r = 0; r < num_rows; ++r) {
      dst_wgt0[r] = w_vec[0];
      dst_wgt1[r] = w_vec[1];
      w_vec += 2;
    }
  } else {
    const HVX_UVector* restrict w_vec = (const HVX_UVector*) w_weights;
    for (size_t r = 0; r < num_rows; ++r) {
      dst_wgt0[r] = w_vec[0];
      dst_wgt1[r] = w_vec[1];
      w_vec += 2;
    }
  }
  const HVX_Vector vzero = Q6_V_vzero();
  for (size_t r = num_rows; r < total_vecs; ++r) {
    dst_wgt0[r] = vzero;
    dst_wgt1[r] = vzero;
  }
  xnn_hvx_vtcm_fence(&dst_wgt1[total_vecs - 1]);
}

static XNN_INLINE void xnn_finish_wgt_stage(
    const struct xnn_hmx_worker_slot* slot, size_t buf_idx, bool used_dma,
    HVX_Vector* restrict dst_wgt0, HVX_Vector* restrict dst_wgt1,
    const void* w_weights, size_t num_rows, size_t total_vecs) {
  if XNN_LIKELY(used_dma) {
    xnn_dma_wait_desc(&slot->dma_descs[buf_idx][1]);
    if (num_rows < total_vecs) {
      const HVX_Vector vzero = Q6_V_vzero();
      for (size_t r = num_rows; r < total_vecs; ++r) {
        dst_wgt0[r] = vzero;
        dst_wgt1[r] = vzero;
      }
    }
    xnn_hvx_vtcm_fence(&dst_wgt1[total_vecs - 1]);
  } else {
    xnn_copy_wgt_to_hmx_1x(dst_wgt0, dst_wgt1, w_weights, num_rows,
                           total_vecs);
  }
}

// Expand up to 32 pre-packed unsigned uint8 activation rows (`lhs_packed`,
// woven as 32 rows x 32 K-bytes per 1024B tile) into the 32 odd `:cm` spatial
// pegs (2*r + 1) of a 64-peg 2048B tile in `vtcm_act`, leaving the 32 even
// pegs (2*r) zeroed so `cvt.uh = acc:2x1` reads `acc[2*r + 1]` losslessly.
static void xnn_unpack_packed_lhs_cm_32rows(
    uint8_t* restrict vtcm_act, const void* restrict lhs_packed,
    size_t m_count, size_t k_offset, size_t k_per_chunk) {
  assert((k_offset & 31) == 0);
  const size_t t_start = k_offset >> 5;
  const size_t num_tiles = (k_per_chunk + 31) >> 5;
  const size_t num_g = (m_count + 3) >> 2;
  HVX_Vector* restrict dst_vec = (HVX_Vector*) vtcm_act;
  const HVX_Vector vzero = Q6_V_vzero();

  if XNN_LIKELY(((uintptr_t) lhs_packed & 127u) == 0u) {
    const HVX_Vector* restrict src_vec =
        (const HVX_Vector*) lhs_packed + t_start * 8;
    for (size_t t = 0; t < num_tiles; ++t) {
      for (size_t g = 0; g < num_g; ++g) {
        const HVX_VectorPair p = Q6_W_vshuff_VVR(src_vec[g], vzero, -32);
        dst_vec[2 * g + 0] = Q6_V_lo_W(p);
        dst_vec[2 * g + 1] = Q6_V_hi_W(p);
      }
      src_vec += 8;
      dst_vec += 16;
    }
  } else {
    const HVX_UVector* restrict src_uvec =
        (const HVX_UVector*) lhs_packed + t_start * 8;
    for (size_t t = 0; t < num_tiles; ++t) {
      for (size_t g = 0; g < num_g; ++g) {
        const HVX_VectorPair p = Q6_W_vshuff_VVR(src_uvec[g], vzero, -32);
        dst_vec[2 * g + 0] = Q6_V_lo_W(p);
        dst_vec[2 * g + 1] = Q6_V_hi_W(p);
      }
      src_uvec += 8;
      dst_vec += 16;
    }
  }
}


// Reconstruct two 32-column signed int32 rows `(2*v, 2*v + 1)` from `out_lo16`
// (even peg = bits [7:0], odd peg = bits [15:8]) and `out_hi8` (odd peg =
// signed bits [23:16]).
static XNN_INLINE void xnn_unpack_hmx_i24_2rows(const HVX_Vector* out_lo16,
                                                const HVX_Vector* out_hi8,
                                                size_t v, HVX_Vector* out_r0,
                                                HVX_Vector* out_r1) {
  const HVX_VectorPair w_lo = Q6_Wuh_vunpack_Vub(out_lo16[v]);
  const HVX_VectorPair w_hi = Q6_Wh_vunpack_Vb(out_hi8[v]);

  const HVX_VectorPair w_r0_01 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(w_lo));
  const HVX_Vector v_r0_2 = Q6_V_hi_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(w_hi)));
  const HVX_Vector vacc0 =
      Q6_Vw_vaslacc_VwVwR(Q6_V_lo_W(w_r0_01), Q6_V_hi_W(w_r0_01), 8);
  *out_r0 = Q6_Vw_vaslacc_VwVwR(vacc0, v_r0_2, 16);

  const HVX_VectorPair w_r1_01 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(w_lo));
  const HVX_Vector v_r1_2 = Q6_V_hi_W(Q6_Ww_vunpack_Vh(Q6_V_hi_W(w_hi)));
  const HVX_Vector vacc1 =
      Q6_Vw_vaslacc_VwVwR(Q6_V_lo_W(w_r1_01), Q6_V_hi_W(w_r1_01), 8);
  *out_r1 = Q6_Vw_vaslacc_VwVwR(vacc1, v_r1_2, 16);
}

#endif  // __HEXAGON_ARCH__ >= 73
#if __HVX_ARCH__ < 73
static HVX_Vector xnn_rescale_fp32(HVX_Vector vacc, HVX_Vector vscale) {
  XNN_ALIGN(128) int32_t vacc_buffer[32];
  XNN_ALIGN(128) float vscale_buffer[32];
  *((HVX_Vector*) vacc_buffer) = vacc;
  *((HVX_Vector*) vscale_buffer) = vscale;
  for (int i = 0; i < 32; ++i) {
    vacc_buffer[i] =
        (int32_t) lrintf((float) vacc_buffer[i] * vscale_buffer[i]);
  }
  return *((const HVX_Vector*) vacc_buffer);
}
#endif  // __HVX_ARCH__ < 73

static XNN_INLINE void xnn_requantize_and_store_row_qs8(
    int8_t* c_row, size_t nc, HVX_Vector vacc0, HVX_Vector vacc1,
    HVX_Vector vscale0, HVX_Vector vscale1, HVX_Vector vmagic_bias,
    HVX_Vector vmagic_bias_less_zero_point, HVX_Vector voutput_min,
    HVX_Vector voutput_max) {
#if __HVX_ARCH__ >= 81
  HVX_Vector vaf0 = Q6_Vsf_equals_Vw(vacc0);
  HVX_Vector vaf1 = Q6_Vsf_equals_Vw(vacc1);

  vaf0 = xnn_fmadd_f32(vaf0, vscale0, vmagic_bias);
  vaf1 = xnn_fmadd_f32(vaf1, vscale1, vmagic_bias);

  vacc0 = Q6_Vw_vsub_VwVw_sat(vaf0, vmagic_bias_less_zero_point);
  vacc1 = Q6_Vw_vsub_VwVw_sat(vaf1, vmagic_bias_less_zero_point);

  const HVX_Vector vpack_h = Q6_Vh_vpack_VwVw_sat(vacc1, vacc0);
#elif __HVX_ARCH__ >= 73
  // On HVX v73-v79, `Vqf32` has a 24-bit signed 2's complement mantissa (23
  // magnitude bits), so using `vmagic_bias = 6291456.5f` (`1.5 * 2^22`,
  // `0x4AC00000`) places the ULP at 1.0 in `Vqf32` for exact round-to-nearest-
  // even rounding; converting back to 24-bit IEEE-754 `Vsf` shifts the integer
  // left by 1 bit (`0x4AC00000 + (N << 1)`).
  HVX_Vector vaf0 = Q6_Vsf_equals_Vw(vacc0);
  HVX_Vector vaf1 = Q6_Vsf_equals_Vw(vacc1);

  vaf0 = xnn_fmadd_f32(vaf0, vscale0, vmagic_bias);
  vaf1 = xnn_fmadd_f32(vaf1, vscale1, vmagic_bias);

  vacc0 = Q6_Vw_vsub_VwVw_sat(Q6_Vw_vasr_VwR(vaf0, 1),
                              vmagic_bias_less_zero_point);
  vacc1 = Q6_Vw_vsub_VwVw_sat(Q6_Vw_vasr_VwR(vaf1, 1),
                              vmagic_bias_less_zero_point);

  const HVX_Vector vpack_h = Q6_Vh_vpack_VwVw_sat(vacc1, vacc0);
#else
  (void) vmagic_bias;
  vacc0 = xnn_rescale_fp32(vacc0, vscale0);
  vacc1 = xnn_rescale_fp32(vacc1, vscale1);
  HVX_Vector vpack_h = Q6_Vh_vpack_VwVw_sat(vacc1, vacc0);
  vpack_h = Q6_Vh_vadd_VhVh_sat(vpack_h, vmagic_bias_less_zero_point);
#endif
  HVX_Vector vout = Q6_Vb_vpack_VhVh_sat(vpack_h, vpack_h);
  vout = Q6_Vb_vmax_VbVb(vout, voutput_min);
  vout = Q6_Vb_vmin_VbVb(vout, voutput_max);

  const size_t n_store = nc < 64 ? nc : 64;
  Q6_V_vstu_variable(c_row, n_store, vout);
}

// HVX fallback used if HMX/VTCM cannot be acquired.
static void xnn_pqs8_qc8w_gemm_minmax_hvx_fallback(
    size_t mr, size_t nc, size_t kc,
    const void* restrict lhs_packed,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) {
  kc = round_up_po2(kc, 4);
  #if __HVX_ARCH__ >= 81
  const HVX_Vector vmagic_bias = xnn_set1_f32(12582912.0f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(
      INT32_C(0x4B400000) - (int32_t) params->fp32_scalar.output_zero_point);
  #elif __HVX_ARCH__ >= 73
  const HVX_Vector vmagic_bias = xnn_set1_f32(6291456.5f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(
      INT32_C(0x25600000) - (int32_t) params->fp32_scalar.output_zero_point);
  #else
  const HVX_Vector vmagic_bias = Q6_V_vzero();
  const HVX_Vector vmagic_bias_less_zero_point =
      Q6_Vh_vsplat_R(params->fp32_scalar.output_zero_point);
  #endif
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(params->fp32_scalar.output_max);

  do {
    const HVX_Vector vinit0 = *((const HVX_UVector*) w);
    const HVX_Vector vinit1 =
        *((const HVX_UVector*) ((const int32_t*) w + 32));
    w = (const int32_t*) w + 64;

    const void* w_panel = w;
    for (size_t m = 0; m < mr; ++m) {
      const int8_t* w_ptr = (const int8_t*) w_panel;
      HVX_Vector vacc0 = vinit0;
      HVX_Vector vacc1 = vinit1;

      const size_t k_end = round_up_po2(kc, 4);
      for (size_t k_idx = 0; k_idx < k_end; k_idx += 4) {
        const uint8_t* a_ptr =
            (const uint8_t*) lhs_packed + (k_idx >> 5) * 1024 + m * 32 +
            (k_idx & 31);
        const HVX_Vector va0123 = Q6_V_vsplat_R(unaligned_load_u32(a_ptr));

        const HVX_Vector vb0x0123 = *((const HVX_UVector*) w_ptr);
        const HVX_Vector vb1x0123 = *((const HVX_UVector*) (w_ptr + 128));
        w_ptr += 256;

        vacc0 = Q6_Vw_vrmpyacc_VwVubVb(vacc0, va0123, vb0x0123);
        vacc1 = Q6_Vw_vrmpyacc_VwVubVb(vacc1, va0123, vb1x0123);
      }

      const HVX_Vector vscale0 = *((const HVX_UVector*) w_ptr);
      const HVX_Vector vscale1 =
          *((const HVX_UVector*) ((const float*) w_ptr + 32));


      int8_t* c_row = (int8_t*) ((uintptr_t) c + m * cm_stride);
      xnn_requantize_and_store_row_qs8(
          c_row, nc, vacc0, vacc1, vscale0, vscale1, vmagic_bias,
          vmagic_bias_less_zero_point, voutput_min, voutput_max);
    }

    w = (const void*) ((const uintptr_t) w_panel +
                       (round_up_po2(kc, 4) >> 2) * 256 +
                       64 * sizeof(float));
    c = (int8_t*) ((uintptr_t) c + cn_stride);
    nc = (nc > 64) ? (nc - 64) : 0;
  } while (nc != 0);
}

#if __HEXAGON_ARCH__ >= 73

static XNN_INLINE void xnn_hmx_exec_panel_b(
    const xnn_hmx_crm_mutex_t* crm_mutex, bool hmx_already_held,
    const uint8_t* vtcm_act0, uint32_t act_ctrl, const HVX_Vector* cur_wgt0,
    const HVX_Vector* cur_wgt1, uint32_t wgt_ctrl, HVX_Vector* vtcm_out,
    const HVX_Vector* vtcm_bias, size_t step_nc) {
  if XNN_UNLIKELY(!hmx_already_held) {
    pthread_mutex_lock(&xnn_hmx_ctx.hmx_exec_mutex);
    if (crm_mutex->lock != NULL) {
      crm_mutex->lock(crm_mutex->mutex);
    }
  }
  xnn_hmx_tile_clear();
  xnn_hmx_mpy_ub_cm_b(vtcm_act0, act_ctrl, cur_wgt0, wgt_ctrl);
  xnn_hmx_store_cvt_i24_cm(&vtcm_out[0], &vtcm_out[16], &vtcm_bias[0],
                           &vtcm_bias[1]);
  if (step_nc > 32) {
    xnn_hmx_tile_clear();
    xnn_hmx_mpy_ub_cm_b(vtcm_act0, act_ctrl, cur_wgt1, wgt_ctrl);
    xnn_hmx_store_cvt_i24_cm(&vtcm_out[32], &vtcm_out[48], &vtcm_bias[0],
                             &vtcm_bias[1]);
    xnn_hvx_vtcm_fence(&vtcm_out[48]);
  } else {
    xnn_hvx_vtcm_fence(&vtcm_out[16]);
  }
  if XNN_UNLIKELY(!hmx_already_held) {
    if (crm_mutex->unlock != NULL) {
      crm_mutex->unlock(crm_mutex->mutex);
    }
    pthread_mutex_unlock(&xnn_hmx_ctx.hmx_exec_mutex);
  }
}

// Must be called with `slot->mutex` held and the HMX lock registered for this
// thread.
static void xnn_pqs8_qc8w_gemm_minmax_32x64c4__hmx_impl(
    struct xnn_hmx_worker_slot* slot, const xnn_hmx_crm_mutex_t* crm_mutex,
    bool hmx_already_held, bool wgt_already_staged, size_t mr, size_t nc,
    size_t kc, const void* restrict lhs_packed,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) {
  const size_t k_chunk_max = xnn_hmx_ctx.max_k_chunk;
  const HVX_Vector* vtcm_bias = xnn_hmx_ctx.vtcm_bias;
  uint8_t* vtcm_act0 = slot->vtcm_act0;
  HVX_Vector* vtcm_wgt = slot->vtcm_wgt;
  HVX_Vector* vtcm_out = slot->vtcm_out;
  HVX_Vector* vtcm_acc = slot->vtcm_acc;

  #if __HVX_ARCH__ >= 81
  const HVX_Vector vmagic_bias = xnn_set1_f32(12582912.0f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(
      INT32_C(0x4B400000) - (int32_t) params->fp32_scalar.output_zero_point);
  #else
  const HVX_Vector vmagic_bias = xnn_set1_f32(6291456.5f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(
      INT32_C(0x25600000) - (int32_t) params->fp32_scalar.output_zero_point);
  #endif
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(params->fp32_scalar.output_max);

  const size_t num_v = (mr + 1) >> 1;
  const HVX_Vector vzero = Q6_V_vzero();

  if (kc <= k_chunk_max) {
    const size_t num_k_tiles = (kc + 31) >> 5;
    const size_t num_rows = round_up_po2(kc, 4) >> 2;
    const size_t total_vecs = num_k_tiles * 8;
    const size_t half_panel_vecs = (total_vecs + 15) & ~(size_t) 15;
    const uint32_t act_ctrl = (uint32_t) (num_k_tiles * 2048) - 1;
    const uint32_t wgt_ctrl =
        (uint32_t) (num_k_tiles * 1024) - 1;
    const size_t panel_bytes = (num_rows + 2) * 256;

    size_t cur_dma_slot = 0;
    HVX_Vector* cur_wgt0 = vtcm_wgt;
    HVX_Vector* cur_wgt1 = cur_wgt0 + half_panel_vecs;
    const bool use_wgt_dma = (num_rows >= 64) && (nc > 64);
    bool cur_used_dma = false;
    if (!wgt_already_staged) {
      if (use_wgt_dma) {
        cur_used_dma = xnn_dma_submit_wgt_2d(
            &slot->dma_descs[0][0], &slot->dma_descs[0][1], cur_wgt0,
            cur_wgt1, (const uint8_t*) w + 256, num_rows);
      } else {
        l2fetch_2d(w, 256, 256, (uint32_t) (num_rows + 2));
      }
    }
    xnn_unpack_packed_lhs_cm_32rows(vtcm_act0, lhs_packed, mr,
                                    /*k_offset=*/0, kc);
    xnn_hvx_vtcm_fence((HVX_Vector*) &vtcm_act0[num_k_tiles * 2048 - 128]);

    do {
      const size_t step_nc = min(nc, 64);
      const size_t next_dma_slot = cur_dma_slot ^ 1;
      HVX_Vector* next_wgt0 = vtcm_wgt + next_dma_slot * 512;
      HVX_Vector* next_wgt1 = next_wgt0 + half_panel_vecs;
      bool next_used_dma = false;
      if (nc > 64) {
        if (use_wgt_dma) {
          next_used_dma = xnn_dma_submit_wgt_2d(
              &slot->dma_descs[next_dma_slot][0],
              &slot->dma_descs[next_dma_slot][1], next_wgt0, next_wgt1,
              (const uint8_t*) w + panel_bytes + 256, num_rows);
        } else {
          l2fetch_2d((const uint8_t*) w + panel_bytes, 256, 256,
                     (uint32_t) (num_rows + 2));
        }
      }

      const HVX_Vector vpacked_b0 = *((const HVX_UVector*) w);
      const HVX_Vector vpacked_b1 =
          *((const HVX_UVector*) ((const int32_t*) w + 32));
      const HVX_Vector vneg_ksum0 = vpacked_b0;
      const HVX_Vector vneg_ksum1 = vpacked_b1;

      const void* w_weights = (const uint8_t*) w + 256;
      const float* w_tail = (const float*) ((const uint8_t*) w_weights +
                                            num_rows * 256);
      const HVX_Vector vw_scale0 = *((const HVX_UVector*) w_tail);
      const HVX_Vector vw_scale1 = *((const HVX_UVector*) (w_tail + 32));

      if (!wgt_already_staged) {
        xnn_finish_wgt_stage(slot, cur_dma_slot, cur_used_dma, cur_wgt0,
                             cur_wgt1, w_weights, num_rows, total_vecs);
      }

      xnn_hmx_exec_panel_b(crm_mutex, hmx_already_held, vtcm_act0,
                                       act_ctrl, cur_wgt0, cur_wgt1, wgt_ctrl,
                                       vtcm_out, vtcm_bias, step_nc);

      for (size_t v = 0; v < num_v; ++v) {
        HVX_Vector vhmx0_0, vhmx0_1;
        HVX_Vector vhmx1_0 = vzero, vhmx1_1 = vzero;
        xnn_unpack_hmx_i24_2rows(&vtcm_out[0], &vtcm_out[16], v, &vhmx0_0,
                                 &vhmx0_1);
        if (step_nc > 32) {
          xnn_unpack_hmx_i24_2rows(&vtcm_out[32], &vtcm_out[48], v, &vhmx1_0,
                                   &vhmx1_1);
        }
        const size_t m_base = 2 * v;
        int8_t* c_row = (int8_t*) ((uintptr_t) c + m_base * cm_stride);
        xnn_requantize_and_store_row_qs8(
            c_row, step_nc, Q6_Vw_vadd_VwVw(vhmx0_0, vneg_ksum0),
            Q6_Vw_vadd_VwVw(vhmx1_0, vneg_ksum1), vw_scale0, vw_scale1,
            vmagic_bias, vmagic_bias_less_zero_point, voutput_min,
            voutput_max);
        if (m_base + 1 < mr) {
          c_row = (int8_t*) ((uintptr_t) c_row + cm_stride);
          xnn_requantize_and_store_row_qs8(
              c_row, step_nc, Q6_Vw_vadd_VwVw(vhmx0_1, vneg_ksum0),
              Q6_Vw_vadd_VwVw(vhmx1_1, vneg_ksum1), vw_scale0, vw_scale1,
              vmagic_bias, vmagic_bias_less_zero_point, voutput_min,
              voutput_max);
        }
      }

      w = (const uint8_t*) w + panel_bytes;
      c = (int8_t*) ((uintptr_t) c + cn_stride);
      nc = (nc > 64) ? (nc - 64) : 0;
      cur_dma_slot = next_dma_slot;
      cur_wgt0 = next_wgt0;
      cur_wgt1 = next_wgt1;
      cur_used_dma = next_used_dma;
    } while (nc != 0);
  } else {
    const bool prepack_all_act = (kc <= XNN_HMX_MAX_ACT_K);
    const size_t first_chunk_k = k_chunk_max;
    const size_t first_chunk_rows = first_chunk_k >> 2;
    const size_t first_half_vecs =
        (((first_chunk_k >> 5) * 8) + 15) & ~(size_t) 15;
    size_t cur_buf = 0;
    bool cur_used_dma = xnn_dma_submit_wgt_2d(
        &slot->dma_descs[0][0], &slot->dma_descs[0][1], vtcm_wgt,
        vtcm_wgt + first_half_vecs, (const uint8_t*) w + 256,
        first_chunk_rows);

    if (prepack_all_act) {
      size_t last_fence_bytes = 0;
      for (size_t k_offset = 0; k_offset < kc; k_offset += k_chunk_max) {
        const size_t chunk_k = min(kc - k_offset, k_chunk_max);
        const size_t chunk_tiles = (chunk_k + 31) >> 5;
        xnn_unpack_packed_lhs_cm_32rows(vtcm_act0 + k_offset * 64,
                                        lhs_packed, mr, k_offset, chunk_k);
        last_fence_bytes = k_offset * 64 + chunk_tiles * 2048;
      }
      xnn_hvx_vtcm_fence((HVX_Vector*) &vtcm_act0[last_fence_bytes - 128]);
    }

    do {
      const size_t step_nc = min(nc, 64);
      const HVX_Vector vpacked_b0 = *((const HVX_UVector*) w);
      const HVX_Vector vpacked_b1 =
          *((const HVX_UVector*) ((const int32_t*) w + 32));
      w = (const int32_t*) w + 64;

      const HVX_Vector vneg_ksum0 = vpacked_b0;
      const HVX_Vector vneg_ksum1 = vpacked_b1;

      for (size_t k_offset = 0; k_offset < kc; k_offset += k_chunk_max) {
        const size_t chunk_k = min(kc - k_offset, k_chunk_max);
        const size_t chunk_tiles = (chunk_k + 31) >> 5;
        const size_t chunk_rows =
            round_up_po2(chunk_k, 4) >> 2;
        const size_t chunk_total_vecs = chunk_tiles * 8;
        const size_t chunk_half_vecs = (chunk_total_vecs + 15) & ~(size_t) 15;
        const uint32_t act_ctrl = (uint32_t) (chunk_tiles * 2048) - 1;
        const uint32_t wgt_ctrl =
            (uint32_t) (chunk_tiles * 1024) - 1;

        const void* cur_w_weights = w;
        w = (const uint8_t*) w + chunk_rows * 256;

        const size_t next_buf = cur_buf ^ 1;
        HVX_Vector* next_base = vtcm_wgt + next_buf * 512;
        bool next_used_dma = false;
        if (k_offset + k_chunk_max < kc) {
          const size_t next_k = min(kc - (k_offset + k_chunk_max),
                                    k_chunk_max);
          const size_t next_tiles = (next_k + 31) >> 5;
          const size_t next_rows =
              round_up_po2(next_k, 4) >> 2;
          const size_t next_half_vecs =
              (next_tiles * 8 + 15) & ~(size_t) 15;
          next_used_dma = xnn_dma_submit_wgt_2d(
              &slot->dma_descs[next_buf][0], &slot->dma_descs[next_buf][1],
              next_base, next_base + next_half_vecs, w, next_rows);
        } else if (nc > 64) {
          next_used_dma = xnn_dma_submit_wgt_2d(
              &slot->dma_descs[next_buf][0], &slot->dma_descs[next_buf][1],
              next_base, next_base + first_half_vecs,
              (const uint8_t*) w + 512,
              first_chunk_rows);
        }

        const uint8_t* chunk_act0 = vtcm_act0;
        if (prepack_all_act) {
          chunk_act0 = vtcm_act0 + k_offset * 64;
        } else {
          xnn_unpack_packed_lhs_cm_32rows(vtcm_act0, lhs_packed, mr,
                                          k_offset, chunk_k);
          xnn_hvx_vtcm_fence(
              (HVX_Vector*) &vtcm_act0[chunk_tiles * 2048 - 128]);
        }

        HVX_Vector* cur_wgt0 = vtcm_wgt + cur_buf * 512;
        HVX_Vector* cur_wgt1 = cur_wgt0 + chunk_half_vecs;
        xnn_finish_wgt_stage(slot, cur_buf, cur_used_dma, cur_wgt0, cur_wgt1,
                             cur_w_weights, chunk_rows, chunk_total_vecs);

        xnn_hmx_exec_panel_b(crm_mutex, hmx_already_held,
                                         chunk_act0, act_ctrl, cur_wgt0,
                                         cur_wgt1, wgt_ctrl, vtcm_out,
                                         vtcm_bias, step_nc);

        for (size_t v = 0; v < num_v; ++v) {
          HVX_Vector vacc0_0, vacc0_1;
          HVX_Vector vacc1_0 = vzero, vacc1_1 = vzero;
          xnn_unpack_hmx_i24_2rows(&vtcm_out[0], &vtcm_out[16], v, &vacc0_0,
                                   &vacc0_1);
          if (step_nc > 32) {
            xnn_unpack_hmx_i24_2rows(&vtcm_out[32], &vtcm_out[48], v,
                                     &vacc1_0, &vacc1_1);
          }
          HVX_Vector* acc_ptr = &vtcm_acc[v * 4];
          if (k_offset == 0) {
            acc_ptr[0] = vacc0_0;
            acc_ptr[1] = vacc1_0;
            acc_ptr[2] = vacc0_1;
            acc_ptr[3] = vacc1_1;
          } else {
            acc_ptr[0] = Q6_Vw_vadd_VwVw(acc_ptr[0], vacc0_0);
            acc_ptr[1] = Q6_Vw_vadd_VwVw(acc_ptr[1], vacc1_0);
            acc_ptr[2] = Q6_Vw_vadd_VwVw(acc_ptr[2], vacc0_1);
            acc_ptr[3] = Q6_Vw_vadd_VwVw(acc_ptr[3], vacc1_1);
          }
        }

        cur_buf = next_buf;
        cur_used_dma = next_used_dma;
      }

      const float* w_tail = (const float*) w;
      const HVX_Vector vw_scale0 = *((const HVX_UVector*) w_tail);
      const HVX_Vector vw_scale1 = *((const HVX_UVector*) (w_tail + 32));
      w = w_tail + 64;

      for (size_t m = 0; m < mr; ++m) {
        int8_t* c_row = (int8_t*) ((uintptr_t) c + m * cm_stride);
        const HVX_Vector vacc0 =
            Q6_Vw_vadd_VwVw(vtcm_acc[m * 2 + 0], vneg_ksum0);
        const HVX_Vector vacc1 =
            Q6_Vw_vadd_VwVw(vtcm_acc[m * 2 + 1], vneg_ksum1);
        xnn_requantize_and_store_row_qs8(
            c_row, step_nc, vacc0, vacc1, vw_scale0, vw_scale1, vmagic_bias,
            vmagic_bias_less_zero_point, voutput_min, voutput_max);
      }

      c = (int8_t*) ((uintptr_t) c + cn_stride);
      nc = (nc > 64) ? (nc - 64) : 0;
    } while (nc != 0);
  }
}

#endif  // __HEXAGON_ARCH__ >= 73

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x64c4__hmx_get_mr(void) {
  return 32;
}

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x64c4__hmx_get_nr(void) {
  return 64;
}

void xnn_pqs8_qc8w_gemm_minmax_32x64c4_hmx(
    size_t m, size_t nc, size_t kc, const void* lhs_packed, const void* w,
    void* dst, size_t dst_stride_row, size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
assert(m != 0);
assert(nc != 0);
assert(kc != 0);
assert(lhs_packed != NULL);
assert(w != NULL);
assert(dst != NULL);
assert(dst_stride_col == sizeof(int8_t));
(void) dst_stride_col;

const size_t lhs_tile_bytes = divide_round_up(kc, 32) * 1024;
const size_t cn_stride = 64 * sizeof(int8_t);

#if __HEXAGON_ARCH__ >= 73
  xnn_hmx_crm_mutex_t crm_mutex = {0};
  pthread_mutex_lock(&xnn_hmx_ctx.init_mutex);
  if XNN_UNLIKELY(!xnn_hmx_ctx.initialized && !xnn_hmx_ctx.init_failed) {
    xnn_hmx_ctx.init_failed = !xnn_init_hmx_vtcm();
  }
  const bool hmx_ready =
      xnn_hmx_ctx.initialized && xnn_hmx_lock_thread(&crm_mutex);
  const bool hold_hmx_outer = (xnn_hmx_ctx.num_locked_tids <= 1);
  pthread_mutex_unlock(&xnn_hmx_ctx.init_mutex);

  if XNN_LIKELY(hmx_ready) {
    const size_t num_slots = xnn_hmx_ctx.num_slots;
    const size_t start_slot = ((size_t) qurt_thread_get_id()) % num_slots;
    size_t slot_idx = start_slot;
    bool acquired = false;
    for (size_t i = 0; i < num_slots; ++i) {
      const size_t try_idx = (start_slot + i) % num_slots;
      if (pthread_mutex_trylock(&xnn_hmx_ctx.slots[try_idx].mutex) == 0) {
        slot_idx = try_idx;
        acquired = true;
        break;
      }
    }
    if (!acquired) {
      pthread_mutex_lock(&xnn_hmx_ctx.slots[slot_idx].mutex);
    }
    if XNN_LIKELY(hold_hmx_outer) {
      pthread_mutex_lock(&xnn_hmx_ctx.hmx_exec_mutex);
      if (crm_mutex.lock != NULL) {
        crm_mutex.lock(crm_mutex.mutex);
      }
    }
    const bool single_panel_wgt =
        (nc <= 64) && (kc <= xnn_hmx_ctx.max_k_chunk);
    for (size_t m_off = 0; m_off < m; m_off += 32) {
      const size_t tile_m = min(m - m_off, (size_t) 32);
      const void* lhs_tile =
          (const uint8_t*) lhs_packed + (m_off >> 5) * lhs_tile_bytes;
      int8_t* c_tile = (int8_t*) ((uintptr_t) dst + m_off * dst_stride_row);
      xnn_pqs8_qc8w_gemm_minmax_32x64c4__hmx_impl(
          &xnn_hmx_ctx.slots[slot_idx], &crm_mutex, hold_hmx_outer,
          /*wgt_already_staged=*/(m_off > 0 && single_panel_wgt), tile_m,
          nc, kc, lhs_tile, w, c_tile, dst_stride_row, cn_stride, params);
    }
    if XNN_LIKELY(hold_hmx_outer) {
      if (crm_mutex.unlock != NULL) {
        crm_mutex.unlock(crm_mutex.mutex);
      }
      pthread_mutex_unlock(&xnn_hmx_ctx.hmx_exec_mutex);
    }
    pthread_mutex_unlock(&xnn_hmx_ctx.slots[slot_idx].mutex);
    return;
  }
#endif  // __HEXAGON_ARCH__ >= 73

  for (size_t m_off = 0; m_off < m; m_off += 32) {
    const size_t tile_m = min(m - m_off, (size_t) 32);
    const void* lhs_tile =
        (const uint8_t*) lhs_packed + (m_off >> 5) * lhs_tile_bytes;
    int8_t* c_tile = (int8_t*) ((uintptr_t) dst + m_off * dst_stride_row);
    xnn_pqs8_qc8w_gemm_minmax_hvx_fallback(
        tile_m, nc, kc, lhs_tile, w, c_tile, dst_stride_row, cn_stride,
        params);
  }
}

void xnn_pqs8_qc8w_gemm_minmax_ukernel_32x64c4__hmx(
    size_t m, size_t nc, size_t kc, const void* lhs_packed, const void* w,
    void* dst, size_t dst_stride_row, size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
  xnn_pqs8_qc8w_gemm_minmax_32x64c4_hmx(m, nc, kc, lhs_packed, w, dst,
                                        dst_stride_row, dst_stride_col,
                                        params);
}
