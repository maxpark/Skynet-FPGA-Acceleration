# Skynet-FPGA-Acceleration
* **Platform**: ZCU104 Evaluation Board
* **Software**: SDx 2019.1

## Run test
Load the bit file in board_files folder and then run
```
$ sudo ./board_files/skynet.elf
```

## Implementation Detail
* The Algorithm is accelerated by moving the PWCONV1X1 function to hardware using a Systolic Array of size 8x8
* The 8x8 PE array and the register array are completly partitioned in all dimension. The 8x143 input A_buffer and B_buffer are completly partitioned in dimension 1. the globel array FM1, FM2, FM3, FM4 are completly partitioned in dimension 1 and WBUF1x1 are completly partitioned in dimension 2.
* The part of systolic array or the PE function are pipelined in the outer loop, read and write operations are pipelined in the inner loop.

## HLS Report
The performance and resources utilization are shown in reports folder, the matrix multiplication in PWCONV1X1 are pipelined with interval 2. But it is worth to note that the read and write operation takes a really long time in the whole optimized function, so this function can be futher optimized by optimizing the I/O.
