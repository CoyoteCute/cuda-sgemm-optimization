# Plan

Kernel ladder. Each stage targets the bottleneck the previous stage created.

| # | Kernel              | Attacks                        | Expected new bottleneck |
|---|---------------------|--------------------------------|-------------------------|
| 0 | naive               | -                              | uncoalesced global      |
| 1 | coalesced           | global access pattern          | global traffic volume   |
| 2 | smem block tile     | global reuse                   | SMEM traffic (2 ld/FMA) |
| 3 | register tile 1D    | SMEM reuse, one axis           | still 1/TM+1 ld/FMA     |
| 4 | register tile 2D    | SMEM reuse, both axes          | SMEM bank conflicts     |
| 5 | ...                 | (to be decided from profile)   |                         |

Rule: no stage is accepted without a before/after ncu profile and the
metric that explains the delta.

## Curriculum order (guide = CUDA Programming Guide, Release 13.4)

1. Occupancy + register tiling — 2.3.3.3, 2.3.3.4, 2.3.7, 3.2.2.2, 3.2.6
2. Shared memory access patterns — 2.3.3.2, 2.3.4.2   [deferred]
3. Error checking — 2.1.7, esp. 2.1.7.2 async errors  [deferred]
4. Warp-level + reductions — 5.4.6, 2.3.5, 3.2.3, 4.4
5. Async in-kernel — 3.2.4.2, 4.10, 4.11, 4.12
6. TMA — 4.12.2                         (needs sm_90+; not this GPU)
7. Clusters / DSMEM / work stealing — 2.1.10, 3.1.2, 2.3.3.8, 4.13
8. Tile kernels (cuTile) — 1.2.2.3, 2.4                [last]

correct FP32 kernels report max rel ~2e-5 at M=512, N=K=1024, seeds 42/1337.