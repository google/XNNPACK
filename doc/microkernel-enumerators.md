# Microkernel enumerators

XNNPACK frequently needs to define code for each microkernel, such as tests,
benchmarks, and declaring functions. To facilitate this, XNNPACK uses a pattern
where a header invokes a macro for each microkernel. To use these headers,
`#define` the appropriate macro, and then `#include` the microkernel enumerator
header.

For example, to print the name of every `f32-vtanh` microkernel and its batch
size:

```
#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype) \
    printf("%s %d\n", #fn_name, batch_tile);
#include "src/f32-vtanh/f32-vtanh.h"
#undef XNN_UKERNEL
```

It's good practice to `#undef` the macro after including the header, to avoid
accidental contamination of other macro enumerator header usages.

## Adding new kernels

The microkernel enumerator header associated with a new microkernel should be
the only place where manual edits are necessary. After manually adding the
microkernel to the enumerator header, tests and benchmarks will be generated
automatically.
