#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

#define ELEM_SIZE 2
#define VECTOR_LENGTH 1024

#define PE_PER_CHANNEL 2

#define NUM_HBM_CHANNELS 4

#define A0_OFFSET 0
#define A1_OFFSET (A0_OFFSET + ARCH_HBM_NODE_ADDR_SPACE)
#define A2_OFFSET (A1_OFFSET + ARCH_HBM_NODE_ADDR_SPACE)
#define A3_OFFSET (A2_OFFSET + ARCH_HBM_NODE_ADDR_SPACE)

#define B0_OFFSET A0_OFFSET + (VECTOR_LENGTH * ELEM_SIZE)
#define B1_OFFSET (B0_OFFSET + ARCH_HBM_NODE_ADDR_SPACE)
#define B2_OFFSET (B1_OFFSET + ARCH_HBM_NODE_ADDR_SPACE)
#define B3_OFFSET (B2_OFFSET + ARCH_HBM_NODE_ADDR_SPACE)


void vector_copy(uint32_t A, uint32_t B, uint32_t K)
{
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();

    if (cluster_id < 8) {
        // Each 2 clusters handle one HBM channel
        uint32_t local_buffer = 0;
        uint32_t channel_id = cluster_id / PE_PER_CHANNEL; // 2 clusters per channel
        uint32_t channel_offset = cluster_id % PE_PER_CHANNEL; // 2 clusters per channel
        uint32_t size_per_cluster = VECTOR_LENGTH * ELEM_SIZE / PE_PER_CHANNEL;
        uint32_t A_base = A + (channel_id * ARCH_HBM_NODE_ADDR_SPACE) + (channel_offset * size_per_cluster);
        uint32_t B_base = B + (channel_id * ARCH_HBM_NODE_ADDR_SPACE) + (channel_offset * size_per_cluster);

        // Copy A to local buffer
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(local(local_buffer), hbm_addr(A_base), size_per_cluster);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        // Copy local buffer to B
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(hbm_addr(B_base), local(local_buffer), size_per_cluster);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }
}


void main(uint32_t A, uint32_t B, uint32_t K)
{
    K = VECTOR_LENGTH;
    A = A0_OFFSET;
    B = B0_OFFSET;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    uint32_t eoc_val = 0;
    flex_timer_start();
    vector_copy(A, B, K);
    flex_global_barrier_xy();
    flex_timer_end();
    flex_eoc(eoc_val);
    return 0;
}