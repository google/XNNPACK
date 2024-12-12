
#################################### Scalar ###################################
# C8 Packing
tools/xngen src/qb4-packw/kr-scalar.c.in -D NR=16 -D KR=8 -D -o src/qb4-packw/gen/qb4-packw-x16c8-gemm-goi-scalar.c

# C4 Packing
tools/xngen src/qb4-packw/kr-scalar.c.in -D NR=16 -D KR=4 -D -o src/qb4-packw/gen/qb4-packw-x16c4-gemm-goi-scalar.c

#################################### NeonDot ###################################
# C4 Packing
tools/xngen src/qb4-packw/c4-neondot.c.in -D NR=16 -D KR=4 -D -o src/qb4-packw/gen/qb4-packw-x16c4-gemm-goi-neondot.c
