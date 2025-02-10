
// #include <cuda_runtime.h>
// #include <dace/dace.h>
#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

typedef struct GEMM_state_t {
    int filler;
}GEMM_state_t;



int __dace_init_cuda(struct GEMM_state_t *__state, int K, int M, int N);
int __dace_exit_cuda(struct GEMM_state_t *__state);

uint32_t A_base;
uint32_t A_height;
uint32_t A_width;
uint32_t A_tile_height;
uint32_t A_tile_width;
uint32_t B_base;
uint32_t B_height;
uint32_t B_width;
uint32_t B_tile_height;
uint32_t B_tile_width;
uint32_t C_base;
uint32_t C_height;
uint32_t C_width;
uint32_t C_tile_height;
uint32_t C_tile_width;
void nested_main_0_0_11(uint32_t A, uint32_t B, uint32_t accumulator, uint32_t K, uint32_t N) {
    uint32_t local_A;
    local_A = 8192;
    uint32_t local_B;
    local_B = 40960;
    long long bK;

    {
        // Start of state block_tiled_init
        //Framecode generating state block_tiled_init...
        // A = A;
        // copy_memory: A -> local_A, [128, 64], [K, 1], [64, 1], A, local_A
        // is_sync = False
        // SoftHier_HBM -> SoftHier_TCDM 2D
        if(flex_is_dm_core())
        {
            const int tile_width = A_tile_width;
            const int tile_height = A_tile_height;
            const int row_start_offset = ((A - A_base) / 2) / A_width;
            const int col_start_offset = ((A - A_base) / 2) % A_width;
            const int col_start_temp = 0 + col_start_offset;
            const int col_start = col_start_temp % A_width;
            const int row_start = 0 + row_start_offset + col_start_temp / A_width;
            const int tile_row_index = row_start/tile_height;
            const int tile_col_index = col_start/tile_width;
            const int tile_row_offset = row_start%tile_height;
            const int tile_col_offset = col_start%tile_width;
            const int tile_index = tile_row_index*2 + tile_col_index;
            const int channel_id = 0 + (tile_index % 4) * 1;
            const int num_blocks_per_tile = (tile_height/128) * (tile_width/64);
            const int num_blocks_in_previous_tiles_in_channel = (tile_index / 4) * num_blocks_per_tile;
            const int block_row_index = tile_row_offset/128;
            const int block_col_index = tile_col_offset/64;
            const int block_index = block_row_index * (tile_width/64) + block_col_index;
            const int total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
            const int block_addr = A_base + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 128 * 64 * 2;
            flex_dma_async_1d(local(local_A), hbm_addr(block_addr), 128*64*2);
        }
        // local_A = local_A;
        // B = B;
        // copy_memory: B -> local_B, [64, 32], [N, 1], [32, 1], B, local_B
        // is_sync = False
        // SoftHier_HBM -> SoftHier_TCDM 2D
        if(flex_is_dm_core())
        {
            const int tile_width = B_tile_width;
            const int tile_height = B_tile_height;
            const int row_start_offset = ((B - B_base) / 2) / B_width;
            const int col_start_offset = ((B - B_base) / 2) % B_width;
            const int col_start_temp = 0 + col_start_offset;
            const int col_start = col_start_temp % B_width;
            const int row_start = 0 + row_start_offset + col_start_temp / B_width;
            const int tile_row_index = row_start/tile_height;
            const int tile_col_index = col_start/tile_width;
            const int tile_row_offset = row_start%tile_height;
            const int tile_col_offset = col_start%tile_width;
            const int tile_index = tile_row_index*4 + tile_col_index;
            const int channel_id = 4 + (tile_index % 4) * 1;
            const int num_blocks_per_tile = (tile_height/64) * (tile_width/32);
            const int num_blocks_in_previous_tiles_in_channel = (tile_index / 4) * num_blocks_per_tile;
            const int block_row_index = tile_row_offset/64;
            const int block_col_index = tile_col_offset/32;
            const int block_index = block_row_index * (tile_width/32) + block_col_index;
            const int total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
            const int block_addr = B_base + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 64 * 32 * 2;
            flex_dma_async_1d(local(local_B), hbm_addr(block_addr), 64*32*2);
        }
        // local_B = local_B;
        if (flex_is_dm_core())
        {
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
        // End of state block_tiled_init

    }
    for (bK = 0; (bK < (K - 64)); bK = (bK + 64)) {
        {
            // Start of state block_tiled_double_buffered
            //Framecode generating state block_tiled_double_buffered...
            // local_A = local_A;
            // local_B = local_B;
            // accumulator = accumulator;
            if (flex_is_first_core())
            {
                uint32_t _in_local_a = local_A + (8192 * ((bK / 64) % 2)) * 2;
                uint32_t _in_local_b = local_B + (2048 * ((bK / 64) % 2)) * 2;
                uint32_t _in_accumulator = accumulator;

                ///////////////////
                flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_FP_16);
                flex_redmule_wait();
                ///////////////////

            }
            // accumulator = accumulator;
            // A = A;
            // copy_memory: A -> local_A, [128, 64], [K, 1], [64, 1], A + (bK + 64), local_A + (8192 * (((bK / 64) + 1) % 2))
            // is_sync = False
            // SoftHier_HBM -> SoftHier_TCDM 2D
            if(flex_is_dm_core())
            {
                const int tile_width = A_tile_width;
                const int tile_height = A_tile_height;
                const int row_start_offset = ((A - A_base) / 2) / A_width;
                const int col_start_offset = ((A - A_base) / 2) % A_width;
                const int col_start_temp = bK + 64 + col_start_offset;
                const int col_start = col_start_temp % A_width;
                const int row_start = 0 + row_start_offset + col_start_temp / A_width;
                const int tile_row_index = row_start/tile_height;
                const int tile_col_index = col_start/tile_width;
                const int tile_row_offset = row_start%tile_height;
                const int tile_col_offset = col_start%tile_width;
                const int tile_index = tile_row_index*2 + tile_col_index;
                const int channel_id = 0 + (tile_index % 4) * 1;
                const int num_blocks_per_tile = (tile_height/128) * (tile_width/64);
                const int num_blocks_in_previous_tiles_in_channel = (tile_index / 4) * num_blocks_per_tile;
                const int block_row_index = tile_row_offset/128;
                const int block_col_index = tile_col_offset/64;
                const int block_index = block_row_index * (tile_width/64) + block_col_index;
                const int total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                const int block_addr = A_base + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 128 * 64 * 2;
                flex_dma_async_1d(local(local_A + (8192 * (((bK / 64) + 1) % 2)) * 2), hbm_addr(block_addr), 128*64*2);
            }
            // local_A = local_A;
            // B = B;
            // copy_memory: B -> local_B, [64, 32], [N, 1], [32, 1], B + (N * (bK + 64)), local_B + (2048 * (((bK / 64) + 1) % 2))
            // is_sync = False
            // SoftHier_HBM -> SoftHier_TCDM 2D
            if(flex_is_dm_core())
            {
                const int tile_width = B_tile_width;
                const int tile_height = B_tile_height;
                const int row_start_offset = ((B - B_base) / 2) / B_width;
                const int col_start_offset = ((B - B_base) / 2) % B_width;
                const int col_start_temp = 0 + col_start_offset;
                const int col_start = col_start_temp % B_width;
                const int row_start = bK + 64 + row_start_offset + col_start_temp / B_width;
                const int tile_row_index = row_start/tile_height;
                const int tile_col_index = col_start/tile_width;
                const int tile_row_offset = row_start%tile_height;
                const int tile_col_offset = col_start%tile_width;
                const int tile_index = tile_row_index*4 + tile_col_index;
                const int channel_id = 4 + (tile_index % 4) * 1;
                const int num_blocks_per_tile = (tile_height/64) * (tile_width/32);
                const int num_blocks_in_previous_tiles_in_channel = (tile_index / 4) * num_blocks_per_tile;
                const int block_row_index = tile_row_offset/64;
                const int block_col_index = tile_col_offset/32;
                const int block_index = block_row_index * (tile_width/32) + block_col_index;
                const int total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                const int block_addr = B_base + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 64 * 32 * 2;
                flex_dma_async_1d(local(local_B + (2048 * (((bK / 64) + 1) % 2)) * 2), hbm_addr(block_addr), 64*32*2);
            }
            // local_B = local_B;
            if (flex_is_dm_core())
            {
                flex_dma_async_wait_all();
            }
            flex_intra_cluster_sync();
            // End of state block_tiled_double_buffered

        }

    }
    {
        // Start of state block_tiled_final_computation
        //Framecode generating state block_tiled_final_computation...
        // local_A = local_A;
        // local_B = local_B;
        // accumulator = accumulator;
        if (flex_is_first_core())
        {
            uint32_t _in_local_a = local_A + (8192 * ((bK / 64) % 2)) * 2;
            uint32_t _in_local_b = local_B + (2048 * ((bK / 64) % 2)) * 2;
            uint32_t _in_accumulator = accumulator;

            ///////////////////
            flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_FP_16);
            flex_redmule_wait();
            ///////////////////

        }
        // accumulator = accumulator;
        flex_intra_cluster_sync();
        // End of state block_tiled_final_computation

    }
}



int __dace_init_cuda(struct GEMM_state_t *__state, int K, int M, int N) {
    
    

    // __state->gpu_context = new dace::cuda::Context(2, 1);

    

    return 0;
}

int __dace_exit_cuda(struct GEMM_state_t *__state) {
    
    int __err = 0;
    // delete __state->gpu_context;
    return __err;
}


void gemm_entry_0_0_0(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N) {
    {
        // TEST KERNEL SCOPE
        flex_global_barrier_xy();
        uint32_t cluster_id = flex_get_cluster_id();
        uint32_t core_id = flex_get_core_id();
        {
            for (auto i = 0; i < M; i += 512) {
                for (auto j = 0; j < N; j += 128) {
                    {
                        // TEST DEVICE SCOPE
                        int gi = get_pos(cluster_id).x;
                        int gj = get_pos(cluster_id).y;
                        if (gi <= 3) {
                            if (gj <= 3) {
                                // Minels: [0, 0], Maxels: [3, 3]
                                // Configure RedMule Here
                                if(flex_is_first_core())
                                {
                                    flex_redmule_config(128, 64, 32);
                                }
                                flex_intra_cluster_sync();
                                {
                                    for (auto ci = 0; ci < 128; ci += 128) {
                                        for (auto cj = 0; cj < 32; cj += 32) {
                                            uint32_t accumulator;
                                            accumulator = 0;
                                            // DACE_ACL_CHECK(aclrtMemset(accumulator, 0, 4096 * sizeof(dace::float16)));

                                            if(flex_is_dm_core())
                                            {
                                                flex_dma_async_1d(local(accumulator), zomem(0), 8192);
                                                flex_dma_async_wait_all();
                                            }

                                            // accumulator = accumulator;
                                            // Nested SDFG nested_main begin
                                            nested_main_0_0_11(A + (K * (((128 * ci) + (128 * gi)) + i)) * 2, B + (((32 * cj) + (32 * gj)) + j) * 2, accumulator, K, N);
                                            // accumulator = accumulator;
                                            // copy_memory: accumulator -> C, [128, 32], [32, 1], [N, 1], accumulator, C + ((((N * (((128 * ci) + (128 * gi)) + i)) + (32 * cj)) + (32 * gj)) + j)
                                            // is_sync = True
                                            // SoftHier_TCDM -> SoftHier_HBM
                                            if(flex_is_dm_core())
                                            {
                                                const int tile_width = C_tile_width;
                                                const int tile_height = C_tile_height;
                                                const int row_start_offset = ((C - C_base) / 2) / C_width;
                                                const int col_start_offset = ((C - C_base) / 2) % C_width;
                                                const int col_start_temp = 32*cj + 32*gj + j + col_start_offset;
                                                const int col_start = col_start_temp % C_width;
                                                const int row_start = 128*ci + 128*gi + i + row_start_offset + col_start_temp / C_width;
                                                const int tile_row_index = row_start/tile_height;
                                                const int tile_col_index = col_start/tile_width;
                                                const int tile_row_offset = row_start%tile_height;
                                                const int tile_col_offset = col_start%tile_width;
                                                const int tile_index = tile_row_index*4 + tile_col_index;
                                                const int channel_id = 0 + (tile_index % 4) * 1;
                                                const int num_blocks_per_tile = (tile_height/128) * (tile_width/32);
                                                const int num_blocks_in_previous_tiles_in_channel = (tile_index / 4) * num_blocks_per_tile;
                                                const int block_row_index = tile_row_offset/128;
                                                const int block_col_index = tile_col_offset/32;
                                                const int block_index = block_row_index * (tile_width/32) + block_col_index;
                                                const int total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                                                const int block_addr = C_base + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 128 * 32 * 2;
                                                flex_dma_async_1d(hbm_addr(block_addr), local(accumulator), 128*32*2);
                                                flex_dma_async_wait_all();
                                                if (flex_get_cluster_id()==0)
                                                {
                                                    printf("%x %x\n", row_start, col_start);
                                                    printf("%x\n", ((uint16_t *)(hbm_addr(block_addr)))[0]);
                                                }
                                            }
                                            flex_intra_cluster_sync();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    flex_intra_cluster_sync();
                    // Finished deivelevel scope
                }
            }
        }
    }
}


void main(GEMM_state_t *__state, uint32_t A, uint32_t B, uint32_t C, uint32_t K, uint32_t M, uint32_t N);
void main(GEMM_state_t *__state, uint32_t A, uint32_t B, uint32_t C, uint32_t K, uint32_t M, uint32_t N)
{
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    A = ((uint32_t *)(hbm_addr(0)))[0];
    B = ((uint32_t *)(hbm_addr(4)))[0];
    C = ((uint32_t *)(hbm_addr(8)))[0];
    K = ((uint32_t *)(hbm_addr(12)))[0];
    M = ((uint32_t *)(hbm_addr(16)))[0];
    N = ((uint32_t *)(hbm_addr(20)))[0];
    A_base = A;
    B_base = B;
    C_base = C;
    A_height = M;
    A_width = K;
    A_tile_height = M/4;
    A_tile_width = K/2;
    B_height = K;
    B_width = N;
    B_tile_height = K/2;
    B_tile_width = N/4;
    C_height = M;
    C_width = N;
    C_tile_height = M/4;
    C_tile_width = N/4;

    // if (flex_is_first_core() && (flex_get_cluster_id()==0))
    // {
        // printf("A: %x\n", A);
        // printf("B: %x\n", B);
        // printf("C: %x\n", C);
        // printf("K: %x\n", K);
        // printf("M: %x\n", M);
        // printf("N: %x\n", N);
    // }
    // if (flex_is_first_core() && (flex_get_cluster_id()==0))
    // {
        // printf("%x\n", ((uint32_t *)(hbm_addr(A)))[0]);
        // printf("%x\n", ((uint32_t *)(hbm_addr(B)))[0]);
        // printf("%x\n", ((uint32_t *)(hbm_addr(C)))[0]);
    // }
    uint32_t eoc_val = 0;
    flex_global_barrier_xy();
    flex_timer_start();
    gemm_entry_0_0_0(A, B, C, K, M, N);
    flex_global_barrier_xy();
    flex_timer_end();
    flex_eoc(eoc_val);
    return 0;
}

