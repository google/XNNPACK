// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif
#if defined(__ANDROID__)
  #include <android/log.h>
#endif
#if defined(__hexagon__)
  #include <qurt_printf.h>
#endif

#ifndef XNN_LOG_TO_STDIO
  #if defined(__ANDROID__)
    #define XNN_LOG_TO_STDIO 0
  #else
    #define XNN_LOG_TO_STDIO 1
  #endif
#endif

#include <xnnpack/log.h>


/* Messages up to this size are formatted entirely on-stack, and don't allocate heap memory */
#define XNN_LOG_STACK_BUFFER_SIZE 1024

#ifdef _WIN32
  #define XNN_LOG_NEWLINE_LENGTH 2

  #define XNN_LOG_STDERR STD_ERROR_HANDLE
  #define XNN_LOG_STDOUT STD_OUTPUT_HANDLE
#elif defined(__hexagon__)
  #define XNN_LOG_NEWLINE_LENGTH 1

  #define XNN_LOG_STDERR 0
  #define XNN_LOG_STDOUT 0
#else
  #define XNN_LOG_NEWLINE_LENGTH 1

  #define XNN_LOG_STDERR STDERR_FILENO
  #define XNN_LOG_STDOUT STDOUT_FILENO
#endif

#if XNN_LOG_TO_STDIO
static void xnn_vlog(int output_handle, const char* prefix, size_t prefix_length, const char* format, va_list args) {
  char stack_buffer[XNN_LOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a second vsnprintf call is needed */
  va_list args_copy;
  va_copy(args_copy, args);

  memcpy(stack_buffer, prefix, prefix_length * sizeof(char));
  assert((prefix_length + XNN_LOG_NEWLINE_LENGTH) * sizeof(char) <= XNN_LOG_STACK_BUFFER_SIZE);

  const int format_chars = vsnprintf(
    &stack_buffer[prefix_length],
    XNN_LOG_STACK_BUFFER_SIZE - (prefix_length + XNN_LOG_NEWLINE_LENGTH) * sizeof(char),
    format,
    args);
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    goto cleanup;
  }
  const size_t format_length = (size_t) format_chars;
  if ((prefix_length + format_length + XNN_LOG_NEWLINE_LENGTH) * sizeof(char) > XNN_LOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    const size_t heap_buffer_size = (prefix_length + format_length + XNN_LOG_NEWLINE_LENGTH) * sizeof(char);
    #if _WIN32
      heap_buffer = HeapAlloc(GetProcessHeap(), 0, heap_buffer_size);
    #else
      heap_buffer = malloc(heap_buffer_size);
    #endif
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    /* Copy pre-formatted prefix into the on-heap buffer */
    memcpy(heap_buffer, prefix, prefix_length * sizeof(char));
    vsnprintf(&heap_buffer[prefix_length], (format_length + XNN_LOG_NEWLINE_LENGTH) * sizeof(char), format, args_copy);
    out_buffer = heap_buffer;
  }
  #ifdef _WIN32
    out_buffer[prefix_length + format_length] = '\r';
    out_buffer[prefix_length + format_length + 1] = '\n';

    DWORD bytes_written;
    WriteFile(
      GetStdHandle((DWORD) output_handle),
      out_buffer, (prefix_length + format_length + XNN_LOG_NEWLINE_LENGTH) * sizeof(char),
      &bytes_written, NULL);
  #elif defined(__hexagon__)
    qurt_printf("%s", out_buffer);
  #else
    out_buffer[prefix_length + format_length] = '\n';

    ssize_t bytes_written = write(output_handle, out_buffer, (prefix_length + format_length + XNN_LOG_NEWLINE_LENGTH) * sizeof(char));
    (void) bytes_written;
  #endif

cleanup:
  #ifdef _WIN32
    HeapFree(GetProcessHeap(), 0, heap_buffer);
  #else
    free(heap_buffer);
  #endif
  va_end(args_copy);
}
#elif defined(__ANDROID__) && XNN_LOG_LEVEL > XNN_LOG_NONE
  static const char xnnpack_module[] = "XNNPACK";
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_DEBUG
  void xnn_vlog_debug(const char* format, va_list args) {
    #if XNN_LOG_TO_STDIO
      static const char debug_prefix[17] = {
        'D', 'e', 'b', 'u', 'g', ' ', '(', 'X', 'N', 'N', 'P', 'A', 'C', 'K', ')', ':', ' '
      };
      xnn_vlog(XNN_LOG_STDOUT, debug_prefix, 17, format, args);
    #elif defined(__ANDROID__)
      __android_log_vprint(ANDROID_LOG_DEBUG, xnnpack_module, format, args);
    #else
      #error "Platform-specific implementation required"
    #endif
  }
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_INFO
  void xnn_vlog_info(const char* format, va_list args) {
    #if XNN_LOG_TO_STDIO
      static const char info_prefix[16] = {
        'N', 'o', 't', 'e', ' ', '(', 'X', 'N', 'N', 'P', 'A', 'C', 'K', ')', ':', ' '
      };
      xnn_vlog(XNN_LOG_STDOUT, info_prefix, 16, format, args);
    #elif defined(__ANDROID__)
      __android_log_vprint(ANDROID_LOG_INFO, xnnpack_module, format, args);
    #else
      #error "Platform-specific implementation required"
    #endif
  }
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_WARNING
  void xnn_vlog_warning(const char* format, va_list args) {
    #if XNN_LOG_TO_STDIO
      static const char warning_prefix[20] = {
        'W', 'a', 'r', 'n', 'i', 'n', 'g', ' ', 'i', 'n', ' ', 'X', 'N', 'N', 'P', 'A', 'C', 'K', ':', ' '
      };
      xnn_vlog(XNN_LOG_STDERR, warning_prefix, 20, format, args);
    #elif defined(__ANDROID__)
      __android_log_vprint(ANDROID_LOG_WARN, xnnpack_module, format, args);
    #else
      #error "Platform-specific implementation required"
    #endif
  }
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_ERROR
  void xnn_vlog_error(const char* format, va_list args) {
    #if XNN_LOG_TO_STDIO
      static const char error_prefix[18] = {
        'E', 'r', 'r', 'o', 'r', ' ', 'i', 'n', ' ', 'X', 'N', 'N', 'P', 'A', 'C', 'K', ':', ' '
      };
      xnn_vlog(XNN_LOG_STDERR, error_prefix, 18, format, args);
    #elif defined(__ANDROID__)
      __android_log_vprint(ANDROID_LOG_ERROR, xnnpack_module, format, args);
    #else
      #error "Platform-specific implementation required"
    #endif
  }
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_FATAL
  void xnn_vlog_fatal(const char* format, va_list args) {
    #if XNN_LOG_TO_STDIO
      static const char fatal_prefix[24] = {
        'F', 'a', 't', 'a', 'l', ' ', 'e', 'r', 'r', 'o', 'r', ' ', 'i', 'n', ' ', 'X', 'N', 'N', 'P', 'A', 'C', 'K', ':', ' '
      };
      xnn_vlog(XNN_LOG_STDERR, fatal_prefix, 24, format, args);
    #elif defined(__ANDROID__)
      __android_log_vprint(ANDROID_LOG_FATAL, xnnpack_module, format, args);
    #else
      #error "Platform-specific implementation required"
    #endif
  }
#endif
