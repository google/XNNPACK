# Depthwise convolution microkernels

This document describes how depthwise convolution (DWCONV) microkernels work.

All depthwise convolution microkernels live in `src/*-dwconv`, e.g.
[`src/f32-dwconv`](https://github.com/google/XNNPACK/tree/master/src/f32-dwconv).

The simplest microkernel to look at is probably
[`f32-dwconv-up2x3-scalar.c`](../src/f32-dwconv/gen/f32-dwconv-up2x3-scalar.c).

Key parameters:

- channel tile, how many channels the microkernel can process in each iteration
- kernel tile, how many weights (kernel elements, each element is # channels values) the microkernel reads in each
  iteration. This can be greater than the actual number of kernel elements.

## High level description

Each call to the DWCONV microkernel will produce 1 row of output.

For each element of this row of output, DWCONV will produce `channel_tile`
number of outputs in the main loop, with a separate loop to handle remainders
(remainder loop).

In each iteration of the main loop, the microkernel will read `channel_tile` biases, `channel_tile * kernel_tile`
inputs, `channel_tile * kernel_tile` weights, and, optionally, `channel_tile` of per-channel scales,
perform the convolution, then write `channel_tile` outputs.

In the remainder loop, the microkernel will read `remainder_channels` biases,
`remainder_channels * kernel_tile` inputs, `remainder_channels * kernel_tile`
weights, perform the convolution, and write `remainder_channels` outputs.

## Microkernel arguments

```
void xnn_f32_dwconv_ukernel_up2x3__scalar(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
```

- `channels`, number of output channels to compute
- `output_width`, number of produced pixels
- `input`, pointer to input indirection buffer
- `weights`, pointer to weights
- `output`, pointer to output
- `input_stride`, number of bytes to add to the indirection buffer to advance to the input pointers corresponding to the
  next output element
- `output_increment`, number of bytes to get to the next output element
- `input_offset`, offset to add to pointers from indirection buffer, unless these pointers match the zero pointer
- `zero`, pointer to zero buffer
- `params`, min/max values for clamping the output

## Packing

Based on the high level description of the microkernel, we will have to pack the
weights such that we have:

- `channel_tile` biases
- `channel_tile * kernel_tile` weights

Repeated `round_up(channels, channel_tile)` times.

## Indirection buffer

The indirection buffer is packed such that the `channel_tile * kernel_tile`
pointers to input required for computing a single output is adjacent to each
other. A simple way to pack it will then be:

```
input  kernel  output

ABC    ab      WX
DEF    cd      YZ
GHI

uncompressed indirection buffer for first row of output
ABDEBCEF
```

This requires `kernel_tile * output_width` pointers.

We can compress this if we pack the input pointers column first:

```
column first uncompressed:
ADBEBECF
```

Notice that `BE` is repeated. So we can elide it, provided that we tell the
microkernel how much to skip over to get to the input pointers for the next
output element (it is not just `kernel_tile`), that's what `input_stride` is
for.

```
column first compressed:
ADBECF
```

The weights similarly have to be packed column first.

## Compressed indirection buffer

One drawback of the indirection buffer is that its size is proportional to
output size, which can be very big.

```
input  kernel

ABC    ab
DEF    cd
GHI
JKL
MNO
PQR
STU
VWX

indirection buffers:
ADBECF
DGEHFI
GJHKIL
JMKNLO
MPNQOR
PSQTRU
SVTWUX
```

Notice how along each column of the indirection buffer, each element is a
constant offset, + 3, away. This allows us to compress along the Y dimension. We
only need to have 1 row of indirection buffer, and specify a constant offset for
each input as `y index * 3`.

```
indirection buffer (1 row):
ABCDEF

constant offset: 3 (added to each input)

row 0: ADBECF + 0 * 3 = ADBECF
row 1: ADBECF + 1 * 3 = DGEHFI
row 2: ADBECF + 2 * 3 = GJHKIL
row 3: ADBECF + 3 * 3 = JMKNLO
row 4: ADBECF + 4 * 3 = MPNQOR
row 5: ADBECF + 5 * 3 = PSQTRU
row 6: ADBECF + 6 * 3 = SVTWUX
```

This constant offset is calculate in the compute function (in operator-run.c)
based on `output_y`, and passed as `input_offset`.

Note that left and right padding does not affect this: if we are reading from
padding, the input pointer in the indirection buffer is set to the zero buffer,
and in the microkernels, we check if the input pointer is the zero buffer
*before* we add the input offset.

### Top and bottom padding

The above compression scheme is incomplete as it does not handle top and bottom
padding.

```
input  kernel

000
ABC    ab
DEF    cd
GHI
000

(uncompressed) indirection buffers:
0A0B0C
ADBECF
DGEHFI
G0H0I0
```

The above compression scheme will result in inputs like so:

```
row 0: 0A0B0C + 0 * 3 = 0A0B0C
row 1: 0A0B0C + 1 * 3 = 0D0E0F (wrong, due to zero buffer)
```

Due to the padding and zero buffer, the constant offset calculation does not
work. So, we separate out the top section of the indirection buffer to account
for top padding (and by the same logic, the bottom padding).

The compressed indirection buffer thus has 3 sections: top, middle (compressed),
and bottom. The top and bottom account for top and bottom padding respectively
and are not compressed, so there is a row of indirection buffers for each row of
output in these sections. The compressed section is always 1 row.

```
input  kernel

000
ABC    ab
DEF    cd
GHI
000

compressed indirection buffers:
0A0B0C (top)
ADBECF (middle) (1 less row than uncompressed)
G0H0I0 (compressed)
```

In the compute function, we will then need to consider which row of output we
are computing, as that will determine the row of indirection buffer to read, and
also the input offset to use. With the above example:

output_y | section | row in indirection buffer | input offset
-------- | ------- | ------------------------- | ------------
0        | top     | 0                         | 0
1        | mid     | 1                         | 0
2        | mid     | 1                         | 3
3        | bot     | 0                         | 0