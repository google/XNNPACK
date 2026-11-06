// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-hmx.c.in
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
#include <pthread.h>
#include <qurt.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/simd/s32-hvx.h"
#include "src/xnnpack/unaligned.h"

#if __HEXAGON_ARCH__ >= 73

#define XNN_HMX_MAX_K_CHUNK 1024
#define XNN_HMX_MAX_ACT_K 9216
#define XNN_HMX_MAX_SLOTS 4
#define XNN_HMX_SLOT_BYTES (720 * 1024)


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

static XNN_INLINE void xnn_hvx_vtcm_fence(const void* vtcm_ptr) {
  HVX_Vector vdummy;
  __asm__ __volatile__("%0 = vmem(%1+#0)"
                       : "=v"(vdummy)
                       : "r"(vtcm_ptr)
                       : "memory");
}

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
  const HVX_UVector* w_vec = (const HVX_UVector*) w_weights;
  for (size_t r = 0; r < num_rows; ++r) {
    dst_wgt0[r] = w_vec[0];
    dst_wgt1[r] = w_vec[1];
    w_vec += 2;
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

// Gather `[k_offset .. k_offset + chunk_k)` from the flattened `num_p *
// kc_aligned` im2col K-dimension (`kc_aligned = round_up_po2(kc, 4)`), zeroing
// the `[kc .. kc_aligned)` padding per kernel element so activation indices
// match `xnn_pack_qs8_conv_goki_w` weights 1-for-1 across HMX K-tiles.
static void xnn_pack_act_igemm_qs8_to_qu8_cm_32rows(
    uint8_t* restrict vtcm_act, const int8_t** a, size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    size_t mr, size_t kc, size_t kc_aligned, size_t k_offset,
    size_t chunk_k) XNN_OOB_READS {
  const size_t num_v = (mr + 1) >> 1;
  const size_t chunk_tiles = (chunk_k + 31) >> 5;
  const size_t padded_k = chunk_tiles * 32;
  HVX_Vector* restrict vtcm_vec = (HVX_Vector*) vtcm_act;
  const HVX_Vector vzero = Q6_V_vzero();
  const HVX_Vector v0x80 = Q6_Vb_vsplat_R(-128);

  XNN_ALIGN(128) uint8_t row0[XNN_HMX_MAX_K_CHUNK];
  XNN_ALIGN(128) uint8_t row1[XNN_HMX_MAX_K_CHUNK];

  for (size_t v = 0; v < num_v; ++v) {
    const size_t r0 = 2 * v;
    const bool has_r1 = (r0 + 1 < mr);

    size_t dst_pos = 0;
    size_t cur_p = k_offset / kc_aligned;
    size_t cur_k = k_offset % kc_aligned;
    while (dst_pos < chunk_k) {
      const size_t span = min(chunk_k - dst_pos, kc_aligned - cur_k);
      const size_t valid_len = (cur_k < kc) ? min(span, kc - cur_k) : 0;
      const size_t pad_len = span - valid_len;

      const int8_t* a0 = a[cur_p * 32 + r0];
      a0 = (a0 != zero) ? (const int8_t*) ((uintptr_t) a0 + a_offset)
                        : zero_data;
      const uint8_t* s0 = (const uint8_t*) a0 + cur_k;
      uint8_t* d0 = row0 + dst_pos;
      size_t rem = valid_len;
      for (; rem >= 128; rem -= 128) {
        *((HVX_UVector*) d0) = Q6_V_vxor_VV(*((const HVX_UVector*) s0), v0x80);
        s0 += 128;
        d0 += 128;
      }
      if (rem != 0) {
        Q6_V_vstu_variable(
            d0, rem, Q6_V_vxor_VV(*((const HVX_UVector*) s0), v0x80));
        d0 += rem;
      }
      for (size_t z = 0; z < pad_len; ++z) {
        d0[z] = 0;
      }

      if (has_r1) {
        const int8_t* a1 = a[cur_p * 32 + r0 + 1];
        a1 = (a1 != zero) ? (const int8_t*) ((uintptr_t) a1 + a_offset)
                          : zero_data;
        const uint8_t* s1 = (const uint8_t*) a1 + cur_k;
        uint8_t* d1 = row1 + dst_pos;
        size_t rem1 = valid_len;
        for (; rem1 >= 128; rem1 -= 128) {
          *((HVX_UVector*) d1) =
              Q6_V_vxor_VV(*((const HVX_UVector*) s1), v0x80);
          s1 += 128;
          d1 += 128;
        }
        if (rem1 != 0) {
          Q6_V_vstu_variable(
              d1, rem1, Q6_V_vxor_VV(*((const HVX_UVector*) s1), v0x80));
          d1 += rem1;
        }
        for (size_t z = 0; z < pad_len; ++z) {
          d1[z] = 0;
        }
      }

      dst_pos += span;
      ++cur_p;
      cur_k = 0;
    }

    if (chunk_k < padded_k) {
      memset(row0 + chunk_k, 0, padded_k - chunk_k);
      if (has_r1) {
        memset(row1 + chunk_k, 0, padded_k - chunk_k);
      }
    }

    const HVX_Vector* vrow0 = (const HVX_Vector*) row0;
    const HVX_Vector* vrow1 = (const HVX_Vector*) row1;
    size_t t = 0;
    size_t vec_idx = 0;
    for (; t + 4 <= chunk_tiles; t += 4, ++vec_idx) {
      const HVX_Vector v_r0 = vrow0[vec_idx];
      const HVX_Vector v_r1 = has_r1 ? vrow1[vec_idx] : vzero;
      const HVX_VectorPair p0 = Q6_W_vshuff_VVR(v_r0, vzero, -32);
      const HVX_VectorPair p1 = Q6_W_vshuff_VVR(v_r1, vzero, -32);
      const HVX_VectorPair q01 =
          Q6_W_vshuff_VVR(Q6_V_lo_W(p1), Q6_V_lo_W(p0), -64);
      const HVX_VectorPair q23 =
          Q6_W_vshuff_VVR(Q6_V_hi_W(p1), Q6_V_hi_W(p0), -64);
      vtcm_vec[(t + 0) * 16 + v] = Q6_V_lo_W(q01);
      vtcm_vec[(t + 1) * 16 + v] = Q6_V_hi_W(q01);
      vtcm_vec[(t + 2) * 16 + v] = Q6_V_lo_W(q23);
      vtcm_vec[(t + 3) * 16 + v] = Q6_V_hi_W(q23);
    }
    if (t < chunk_tiles) {
      const size_t rem_tiles = chunk_tiles - t;
      const HVX_Vector v_r0 = vrow0[vec_idx];
      const HVX_Vector v_r1 = has_r1 ? vrow1[vec_idx] : vzero;
      const HVX_VectorPair p0 = Q6_W_vshuff_VVR(v_r0, vzero, -32);
      const HVX_VectorPair p1 = Q6_W_vshuff_VVR(v_r1, vzero, -32);
      const HVX_VectorPair q01 =
          Q6_W_vshuff_VVR(Q6_V_lo_W(p1), Q6_V_lo_W(p0), -64);
      const HVX_VectorPair q23 =
          Q6_W_vshuff_VVR(Q6_V_hi_W(p1), Q6_V_hi_W(p0), -64);
      vtcm_vec[(t + 0) * 16 + v] = Q6_V_lo_W(q01);
      if (rem_tiles > 1) {
        vtcm_vec[(t + 1) * 16 + v] = Q6_V_hi_W(q01);
      }
      if (rem_tiles > 2) {
        vtcm_vec[(t + 2) * 16 + v] = Q6_V_lo_W(q23);
      }
    }
  }
}

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

static XNN_INLINE void xnn_dequantize_and_store_row_qd8(
    float* c_row, size_t nc, HVX_Vector vhmx0, HVX_Vector vhmx1,
    HVX_Vector vzp_bias0, HVX_Vector vzp_bias1, HVX_Vector vinput_scale,
    HVX_Vector vw_scale0, HVX_Vector vw_scale1, HVX_Vector vbias0,
    HVX_Vector vbias1, HVX_Vector vmin, HVX_Vector vmax) {
  const HVX_Vector vacc0 = Q6_Vw_vadd_VwVw(vhmx0, vzp_bias0);
  HVX_Vector vout0 = xnn_mul_f32(xnn_cvt_f32_s32(vacc0), vinput_scale);
  vout0 = xnn_fmadd_f32(vout0, vw_scale0, vbias0);
  vout0 = xnn_min_f32(xnn_max_f32(vout0, vmin), vmax);

  if XNN_LIKELY(nc >= 64) {
    const HVX_Vector vacc1 = Q6_Vw_vadd_VwVw(vhmx1, vzp_bias1);
    HVX_Vector vout1 = xnn_mul_f32(xnn_cvt_f32_s32(vacc1), vinput_scale);
    vout1 = xnn_fmadd_f32(vout1, vw_scale1, vbias1);
    vout1 = xnn_min_f32(xnn_max_f32(vout1, vmin), vmax);
    *((HVX_UVector*) c_row) = vout0;
    *((HVX_UVector*) (c_row + 32)) = vout1;
  } else if (nc > 32) {
    const HVX_Vector vacc1 = Q6_Vw_vadd_VwVw(vhmx1, vzp_bias1);
    HVX_Vector vout1 = xnn_mul_f32(xnn_cvt_f32_s32(vacc1), vinput_scale);
    vout1 = xnn_fmadd_f32(vout1, vw_scale1, vbias1);
    vout1 = xnn_min_f32(xnn_max_f32(vout1, vmin), vmax);
    *((HVX_UVector*) c_row) = vout0;
    Q6_V_vstu_variable(c_row + 32, (nc - 32) * sizeof(float), vout1);
  } else {
    Q6_V_vstu_variable(c_row, nc * sizeof(float), vout0);
  }
}

#endif  // __HEXAGON_ARCH__ >= 73

static void xnn_qd8_f32_qc8w_igemm_minmax_hvx_fallback(
    size_t mr, size_t nc, size_t kc, size_t ks, const int8_t** restrict a,
    const void* restrict w, float* restrict c, size_t cm_stride,
    size_t cn_stride, size_t a_offset, const int8_t* zero,
    const int8_t* zero_data, const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
      XNN_OOB_READS {
  const size_t kc_aligned = round_up_po2(kc, 4);
  const size_t num_p = ks / (32 * sizeof(void*));
  const HVX_Vector vmin = Q6_V_vsplat_R(float_as_uint32(params->scalar.min));
  const HVX_Vector vmax = Q6_V_vsplat_R(float_as_uint32(params->scalar.max));
  const HVX_Vector vzp = Q6_V_vsplat_R(quantization_params->zero_point);
  const HVX_Vector vinput_scale =
      Q6_V_vsplat_R(float_as_uint32(quantization_params->inv_scale));

  do {
    const HVX_Vector vinit0 = *((const HVX_UVector*) w);
    const HVX_Vector vinit1 =
        *((const HVX_UVector*) ((const int32_t*) w + 32));
    w = (const int32_t*) w + 64;

    const void* w_panel = w;
    const int8_t* w_end =
        (const int8_t*) w_panel + num_p * (kc_aligned >> 2) * 256;
    const HVX_Vector vscale0 = *((const HVX_UVector*) w_end);
    const HVX_Vector vscale1 =
        *((const HVX_UVector*) ((const float*) w_end + 32));
    const HVX_Vector vbias0 =
        *((const HVX_UVector*) ((const float*) w_end + 64));
    const HVX_Vector vbias1 =
        *((const HVX_UVector*) ((const float*) w_end + 96));

    for (size_t m = 0; m < mr; ++m) {
      const int8_t* w_ptr = (const int8_t*) w_panel;
      HVX_Vector vacc0 = xnn_mul_s32(vinit0, vzp);
      HVX_Vector vacc1 = xnn_mul_s32(vinit1, vzp);

      for (size_t p_idx = 0; p_idx < num_p; ++p_idx) {
        const int8_t* a_row = a[p_idx * 32 + m];
        a_row = (a_row != zero)
                    ? (const int8_t*) ((uintptr_t) a_row + a_offset)
                    : zero_data;

        size_t k = kc_aligned;
        for (; k >= 4; k -= 4) {
          const HVX_Vector va0123 = Q6_V_vsplat_R(unaligned_load_s32(a_row));
          a_row += 4;
          const HVX_Vector vb0 = *((const HVX_UVector*) w_ptr);
          const HVX_Vector vb1 = *((const HVX_UVector*) (w_ptr + 128));
          w_ptr += 256;
          vacc0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0, va0123, vb0);
          vacc1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1, va0123, vb1);
        }
      }

      float* c_row = (float*) ((uintptr_t) c + m * cm_stride);
      HVX_Vector vout0 = xnn_mul_f32(xnn_cvt_f32_s32(vacc0), vinput_scale);
      vout0 = xnn_fmadd_f32(vout0, vscale0, vbias0);
      vout0 = xnn_min_f32(xnn_max_f32(vout0, vmin), vmax);
      if (nc >= 64) {
        HVX_Vector vout1 = xnn_mul_f32(xnn_cvt_f32_s32(vacc1), vinput_scale);
        vout1 = xnn_fmadd_f32(vout1, vscale1, vbias1);
        vout1 = xnn_min_f32(xnn_max_f32(vout1, vmin), vmax);
        *((HVX_UVector*) c_row) = vout0;
        *((HVX_UVector*) (c_row + 32)) = vout1;
      } else if (nc > 32) {
        HVX_Vector vout1 = xnn_mul_f32(xnn_cvt_f32_s32(vacc1), vinput_scale);
        vout1 = xnn_fmadd_f32(vout1, vscale1, vbias1);
        vout1 = xnn_min_f32(xnn_max_f32(vout1, vmin), vmax);
        *((HVX_UVector*) c_row) = vout0;
        Q6_V_vstu_variable(c_row + 32, (nc - 32) * sizeof(float), vout1);
      } else {
        Q6_V_vstu_variable(c_row, nc * sizeof(float), vout0);
      }
    }

    w = (const float*) w_end + 128;
    c = (float*) ((uintptr_t) c + cn_stride);
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

static void xnn_qd8_f32_qc8w_igemm_minmax_32x64c4__hmx_impl(
    struct xnn_hmx_worker_slot* slot, const xnn_hmx_crm_mutex_t* crm_mutex,
    bool hmx_already_held, size_t mr, size_t nc, size_t kc, size_t ks,
    const int8_t** restrict a, const void* restrict w, float* restrict c,
    size_t cm_stride, size_t cn_stride, size_t a_offset, const int8_t* zero,
    const int8_t* zero_data, const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) {
  const size_t k_chunk_max = xnn_hmx_ctx.max_k_chunk;
  const HVX_Vector* vtcm_bias = xnn_hmx_ctx.vtcm_bias;
  uint8_t* vtcm_act0 = slot->vtcm_act0;
  HVX_Vector* vtcm_wgt = slot->vtcm_wgt;
  HVX_Vector* vtcm_out = slot->vtcm_out;
  HVX_Vector* vtcm_acc = slot->vtcm_acc;

  const HVX_Vector vmin = Q6_V_vsplat_R(float_as_uint32(params->scalar.min));
  const HVX_Vector vmax = Q6_V_vsplat_R(float_as_uint32(params->scalar.max));
  const HVX_Vector vzp_adj =
      Q6_V_vsplat_R((int32_t) quantization_params->zero_point + 128);
  const HVX_Vector vinput_scale =
      Q6_V_vsplat_R(float_as_uint32(quantization_params->inv_scale));

  const size_t num_v = (mr + 1) >> 1;
  const HVX_Vector vzero = Q6_V_vzero();
  const size_t kc_aligned = round_up_po2(kc, 4);
  const size_t num_p = ks / (32 * sizeof(void*));
  const size_t total_k = num_p * kc_aligned;

  if (total_k <= k_chunk_max) {
    const size_t num_k_tiles = (total_k + 31) >> 5;
    const size_t num_rows = total_k >> 2;
    const size_t total_vecs = num_k_tiles * 8;
    const size_t half_panel_vecs = (total_vecs + 15) & ~(size_t) 15;
    const uint32_t act_ctrl = (uint32_t) (num_k_tiles * 2048) - 1;
    const uint32_t wgt_ctrl = (uint32_t) (num_k_tiles * 1024) - 1;
    const size_t panel_bytes = (num_rows + 3) * 256;

    size_t cur_buf = 0;
    HVX_Vector* cur_wgt0 = vtcm_wgt;
    HVX_Vector* cur_wgt1 = cur_wgt0 + half_panel_vecs;
    bool cur_used_dma = xnn_dma_submit_wgt_2d(
        &slot->dma_descs[0][0], &slot->dma_descs[0][1], cur_wgt0, cur_wgt1,
        (const uint8_t*) w + 256, num_rows);

    xnn_pack_act_igemm_qs8_to_qu8_cm_32rows(vtcm_act0, a, a_offset, zero,
                                            zero_data, mr, kc, kc_aligned,
                                            /*k_offset=*/0, total_k);
    xnn_hvx_vtcm_fence((HVX_Vector*) &vtcm_act0[num_k_tiles * 2048 - 128]);

    do {
      const size_t step_nc = min(nc, 64);
      const size_t next_buf = cur_buf ^ 1;
      HVX_Vector* next_wgt0 = vtcm_wgt + next_buf * 512;
      HVX_Vector* next_wgt1 = next_wgt0 + half_panel_vecs;
      bool next_used_dma = false;
      if (nc > 64) {
        next_used_dma = xnn_dma_submit_wgt_2d(
            &slot->dma_descs[next_buf][0], &slot->dma_descs[next_buf][1],
            next_wgt0, next_wgt1, (const uint8_t*) w + panel_bytes + 256,
            num_rows);
      }

      const HVX_Vector vneg_ksum0 = *((const HVX_UVector*) w);
      const HVX_Vector vneg_ksum1 =
          *((const HVX_UVector*) ((const int32_t*) w + 32));
      const HVX_Vector vzp_bias0 = xnn_mul_s32(vneg_ksum0, vzp_adj);
      const HVX_Vector vzp_bias1 = xnn_mul_s32(vneg_ksum1, vzp_adj);

      const void* w_weights = (const uint8_t*) w + 256;
      const float* w_tail = (const float*) ((const uint8_t*) w_weights +
                                            num_rows * 256);
      const HVX_Vector vw_scale0 = *((const HVX_UVector*) w_tail);
      const HVX_Vector vw_scale1 = *((const HVX_UVector*) (w_tail + 32));
      const HVX_Vector vbias0 = *((const HVX_UVector*) (w_tail + 64));
      const HVX_Vector vbias1 = *((const HVX_UVector*) (w_tail + 96));

      xnn_finish_wgt_stage(slot, cur_buf, cur_used_dma, cur_wgt0, cur_wgt1,
                           w_weights, num_rows, total_vecs);

      xnn_hmx_exec_panel_b(crm_mutex, hmx_already_held, vtcm_act0, act_ctrl,
                           cur_wgt0, cur_wgt1, wgt_ctrl, vtcm_out, vtcm_bias,
                           step_nc);

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
        float* c_row = (float*) ((uintptr_t) c + m_base * cm_stride);
        xnn_dequantize_and_store_row_qd8(
            c_row, step_nc, vhmx0_0, vhmx1_0, vzp_bias0, vzp_bias1,
            vinput_scale, vw_scale0, vw_scale1, vbias0, vbias1, vmin, vmax);
        if (m_base + 1 < mr) {
          c_row = (float*) ((uintptr_t) c_row + cm_stride);
          xnn_dequantize_and_store_row_qd8(
              c_row, step_nc, vhmx0_1, vhmx1_1, vzp_bias0, vzp_bias1,
              vinput_scale, vw_scale0, vw_scale1, vbias0, vbias1, vmin, vmax);
        }
      }

      w = (const uint8_t*) w + panel_bytes;
      c = (float*) ((uintptr_t) c + cn_stride);
      nc = (nc > 64) ? (nc - 64) : 0;
      cur_buf = next_buf;
      cur_wgt0 = next_wgt0;
      cur_wgt1 = next_wgt1;
      cur_used_dma = next_used_dma;
    } while (nc != 0);
  } else {
    const bool prepack_all_act = (total_k <= XNN_HMX_MAX_ACT_K);
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
      for (size_t k_offset = 0; k_offset < total_k; k_offset += k_chunk_max) {
        const size_t chunk_k = min(total_k - k_offset, k_chunk_max);
        const size_t chunk_tiles = (chunk_k + 31) >> 5;
        xnn_pack_act_igemm_qs8_to_qu8_cm_32rows(
            vtcm_act0 + k_offset * 64, a, a_offset, zero, zero_data, mr, kc,
            kc_aligned, k_offset, chunk_k);
        last_fence_bytes = k_offset * 64 + chunk_tiles * 2048;
      }
      xnn_hvx_vtcm_fence((HVX_Vector*) &vtcm_act0[last_fence_bytes - 128]);
    }

    do {
      const size_t step_nc = min(nc, 64);
      const HVX_Vector vneg_ksum0 = *((const HVX_UVector*) w);
      const HVX_Vector vneg_ksum1 =
          *((const HVX_UVector*) ((const int32_t*) w + 32));
      w = (const int32_t*) w + 64;

      const HVX_Vector vzp_bias0 = xnn_mul_s32(vneg_ksum0, vzp_adj);
      const HVX_Vector vzp_bias1 = xnn_mul_s32(vneg_ksum1, vzp_adj);

      for (size_t k_offset = 0; k_offset < total_k; k_offset += k_chunk_max) {
        const size_t chunk_k = min(total_k - k_offset, k_chunk_max);
        const size_t chunk_tiles = (chunk_k + 31) >> 5;
        const size_t chunk_rows = chunk_k >> 2;
        const size_t chunk_total_vecs = chunk_tiles * 8;
        const size_t chunk_half_vecs = (chunk_total_vecs + 15) & ~(size_t) 15;
        const uint32_t act_ctrl = (uint32_t) (chunk_tiles * 2048) - 1;
        const uint32_t wgt_ctrl = (uint32_t) (chunk_tiles * 1024) - 1;

        const void* cur_w_weights = w;
        w = (const uint8_t*) w + chunk_rows * 256;

        const size_t next_buf = cur_buf ^ 1;
        HVX_Vector* next_base = vtcm_wgt + next_buf * 512;
        bool next_used_dma = false;
        if (k_offset + k_chunk_max < total_k) {
          const size_t next_k =
              min(total_k - (k_offset + k_chunk_max), k_chunk_max);
          const size_t next_tiles = (next_k + 31) >> 5;
          const size_t next_rows = next_k >> 2;
          const size_t next_half_vecs =
              (next_tiles * 8 + 15) & ~(size_t) 15;
          next_used_dma = xnn_dma_submit_wgt_2d(
              &slot->dma_descs[next_buf][0], &slot->dma_descs[next_buf][1],
              next_base, next_base + next_half_vecs, w, next_rows);
        } else if (nc > 64) {
          next_used_dma = xnn_dma_submit_wgt_2d(
              &slot->dma_descs[next_buf][0], &slot->dma_descs[next_buf][1],
              next_base, next_base + first_half_vecs,
              (const uint8_t*) w + 768, first_chunk_rows);
        }

        const uint8_t* chunk_act0 = vtcm_act0;
        if (prepack_all_act) {
          chunk_act0 = vtcm_act0 + k_offset * 64;
        } else {
          xnn_pack_act_igemm_qs8_to_qu8_cm_32rows(
              vtcm_act0, a, a_offset, zero, zero_data, mr, kc, kc_aligned,
              k_offset, chunk_k);
          xnn_hvx_vtcm_fence(
              (HVX_Vector*) &vtcm_act0[chunk_tiles * 2048 - 128]);
        }

        HVX_Vector* cur_wgt0 = vtcm_wgt + cur_buf * 512;
        HVX_Vector* cur_wgt1 = cur_wgt0 + chunk_half_vecs;
        xnn_finish_wgt_stage(slot, cur_buf, cur_used_dma, cur_wgt0, cur_wgt1,
                             cur_w_weights, chunk_rows, chunk_total_vecs);

        xnn_hmx_exec_panel_b(crm_mutex, hmx_already_held, chunk_act0,
                             act_ctrl, cur_wgt0, cur_wgt1, wgt_ctrl, vtcm_out,
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
      const HVX_Vector vbias0 = *((const HVX_UVector*) (w_tail + 64));
      const HVX_Vector vbias1 = *((const HVX_UVector*) (w_tail + 96));
      w = w_tail + 128;

      for (size_t m = 0; m < mr; ++m) {
        float* c_row = (float*) ((uintptr_t) c + m * cm_stride);
        xnn_dequantize_and_store_row_qd8(
            c_row, step_nc, vtcm_acc[m * 2 + 0], vtcm_acc[m * 2 + 1],
            vzp_bias0, vzp_bias1, vinput_scale, vw_scale0, vw_scale1, vbias0,
            vbias1, vmin, vmax);
      }

      c = (float*) ((uintptr_t) c + cn_stride);
      nc = (nc > 64) ? (nc - 64) : 0;
    } while (nc != 0);
  }
}

#endif  // __HEXAGON_ARCH__ >= 73

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_32x64c4__hmx(
    size_t mr, size_t nc, size_t kc, size_t ks, const int8_t** restrict a,
    const void* restrict w, float* restrict c, size_t cm_stride,
    size_t cn_stride, size_t a_offset, const int8_t* zero,
    const int8_t* zero_data, const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
      XNN_OOB_READS {
  assert(mr != 0);
  assert(mr <= 32);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (32 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(quantization_params != NULL);

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
    xnn_qd8_f32_qc8w_igemm_minmax_32x64c4__hmx_impl(
        &xnn_hmx_ctx.slots[slot_idx], &crm_mutex, hold_hmx_outer, mr, nc, kc,
        ks, a, w, c, cm_stride, cn_stride, a_offset, zero,
        zero_data, params, quantization_params);
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

  xnn_qd8_f32_qc8w_igemm_minmax_hvx_fallback(
      mr, nc, kc, ks, a, w, c, cm_stride, cn_stride, a_offset, zero,
      zero_data, params, quantization_params);
}
