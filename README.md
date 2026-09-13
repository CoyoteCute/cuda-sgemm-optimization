## Environment

| | |
|---|---|
| GPU | RTX 3060 Ti (sm_86, 38 SMs, ~448 GB/s) |
| OS | Windows + WSL2, Ubuntu 22.04.5 |
| Toolkit | CUDA 12.9 (nvcc V12.9.86), driver 615.65.06 |
| Profiler | Nsight Compute 2025.2.1 |

Under WSL, `ncu` fails with `ERR_NVGPUCTRPERM` until GPU performance
counters are opened up in the Windows NVIDIA Control Panel
(Desktop → Enable Developer Settings, then Developer → Manage GPU
Performance Counters → allow all users), followed by `wsl --shutdown`.

Verify the toolchain and profiler counter access:

    $nvcc -arch=sm_86 tools/smoke_test.cu -o build/smoke
    NVIDIA GeForce RTX 3060 Ti  sm_86  38 SMs  448 GB/s

    $ncu --set basic build/smoke
    
    [1433] smoke@127.0.0.1
    k(float *) (1, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.71
    SM Frequency                    Ghz         1.38
    Elapsed Cycles                cycle         2616
    Memory Throughput                 %         1.42
    DRAM Throughput                   %         1.42
    Duration                         us         1.89
    L1/TEX Cache Throughput           %        50.89
    L2 Cache Throughput               %         1.22
    SM Active Cycles              cycle        23.58
    Compute (SM) Throughput           %         0.00
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full
          waves across all SMs. Look at Launch Statistics for more details.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                    32
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      1
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              38
    Stack Size                                                  1024
    Threads                                   thread              32
    # TPCs                                                        19
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.00
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 97.37%
          The grid for this launch is configured to execute only 1 block, which is less than the GPU's 38
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel
          concurrently with other workloads, consider reducing the block size to have at least one block per
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)
          description for more details on launch configurations.

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block          128
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block           48
    Theoretical Active Warps per SM        warp           16
    Theoretical Occupancy                     %        33.33
    Achieved Occupancy                        %         2.08
    Achieved Active Warps Per SM           warp            1
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 93.75%
          The difference between calculated theoretical (33.3%) and measured achieved occupancy (2.1%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 66.67%
          The 4.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
          hardware maximum of 12. This kernel's theoretical occupancy (33.3%) is limited by the number of blocks that
          can fit on the SM, and the required amount of shared memory.

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle          180
    Total DRAM Elapsed Cycles        cycle       101376
    Average L1 Active Cycles         cycle        23.58
    Total L1 Elapsed Cycles          cycle       100706
    Average L2 Active Cycles         cycle       150.29
    Total L2 Elapsed Cycles          cycle        59424
    Average SM Active Cycles         cycle        23.58
    Total SM Elapsed Cycles          cycle       100706
    Average SMSP Active Cycles       cycle         5.83
    Total SMSP Elapsed Cycles        cycle       402824
    -------------------------- ----------- ------------
