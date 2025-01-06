
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
            for (auto i = 0; i < M; i += 256) {
                for (auto j = 0; j < N; j += 64) {
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
                                    flex_redmule_config(64, 32, 16);
                                }
                                flex_intra_cluster_sync();
                                {
                                    for (auto ci = 0; ci < 64; ci += 64) {
                                        for (auto cj = 0; cj < 16; cj += 16) {
                                            uint32_t accumulator;
                                            accumulator = 0;
                                            // DACE_ACL_CHECK(aclrtMemset(accumulator, 0, 1024 * sizeof(dace::float16)));

                                            if(flex_is_dm_core())
                                            {
                                                flex_dma_async_1d(local(accumulator), zomem(0), 2048);
                                                flex_dma_async_wait_all();
                                            }

                                            // accumulator = accumulator;
                                            {
                                                for (auto bK = 0; bK < K; bK += 32) {
                                                    uint32_t local_A;
                                                    local_A = 2048;
                                                    uint32_t local_B;
                                                    local_B = 6144;
                                                    // local_A = local_A;
                                                    // copy_memory: A -> local_A, [64, 32], [K, 1], [32, 1], A + ((K * (((64 * ci) + (64 * gi)) + i)) + bK), local_A
                                                    // is_sync = True
                                                    // SoftHier_HBM -> SoftHier_TCDM 2D
                                                    if(flex_is_dm_core())
                                                    {
                                                        const int tile_width = K/8;
                                                        const int tile_height = M/4;
                                                        const int row_start = 64*ci + 64*gi + i;
                                                        const int col_start = bK;
                                                        const int tile_row_index = row_start/tile_height;
                                                        const int tile_col_index = col_start/tile_width;
                                                        const int tile_row_offset = row_start%tile_height;
                                                        const int tile_col_offset = col_start%tile_width;
                                                        const int tile_index = tile_row_index*8 + tile_col_index;
                                                        const int channel_id = 0 + (tile_index % 4) * 1;
                                                        const int num_blocks_per_tile = (tile_height/64) * (tile_width/32);
                                                        const int num_blocks_in_previous_tiles_in_channel = (tile_index / 4) * num_blocks_per_tile;
                                                        const int block_row_index = tile_row_offset/64;
                                                        const int block_col_index = tile_col_offset/32;
                                                        const int block_index = block_row_index * (tile_width/32) + block_col_index;
                                                        const int total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                                                        const int block_addr = A + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 64 * 32 * 2;
                                                        // if (flex_get_cluster_id()==0)
                                                        // {
                                                        //     flex_print(" "); flex_print_int(tile_row_index); flex_print(" ");
                                                        //     flex_print(" "); flex_print_int(tile_col_index); flex_print(" ");
                                                        //     flex_print(" "); flex_print_int(tile_row_offset); flex_print(" ");
                                                        //     flex_print(" "); flex_print_int(tile_col_offset); flex_print(" ");
                                                        //     flex_print(" "); flex_print_int(block_row_index); flex_print(" ");
                                                        //     flex_print(" "); flex_print_int(block_col_index); flex_print(" ");
                                                        //     flex_print(" "); flex_print_int(block_addr); flex_print("\n");
                                                        // }
                                                        flex_dma_async_1d(local(local_A), hbm_addr(block_addr), 64*32*2);
                                                        flex_dma_async_wait_all();
                                                    }
                                                    flex_intra_cluster_sync();
                                                    // local_B = local_B;
                                                    // copy_memory: B -> local_B, [32, 16], [N, 1], [16, 1], B + ((((N * bK) + (16 * cj)) + (16 * gj)) + j), local_B
                                                    // is_sync = True
                                                    // SoftHier_HBM -> SoftHier_TCDM 2D
                                                    if(flex_is_dm_core())
                                                    {
                                                        const int tile_width = N/2;
                                                        const int tile_height = K/2;
                                                        const int row_start = bK;
                                                        const int col_start = 16*cj + 16*gj + j;
                                                        const int tile_row_index = row_start/tile_height;
                                                        const int tile_col_index = col_start/tile_width;
                                                        const int tile_row_offset = row_start%tile_height;
                                                        const int tile_col_offset = col_start%tile_width;
                                                        const int tile_index = tile_row_index*2 + tile_col_index;
                                                        const int channel_id = 0 + (tile_index % 4) * 1;
                                                        const int num_blocks_per_tile = (tile_height/32) * (tile_width/16);
                                                        const int num_blocks_in_previous_tiles_in_channel = (tile_index / 4) * num_blocks_per_tile;
                                                        const int block_row_index = tile_row_offset/32;
                                                        const int block_col_index = tile_col_offset/16;
                                                        const int block_index = block_row_index * (tile_width/16) + block_col_index;
                                                        const int total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                                                        const int block_addr = B + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 32 * 16 * 2;
                                                        flex_dma_async_1d(local(local_B), hbm_addr(block_addr), 32*16*2);
                                                        flex_dma_async_wait_all();
                                                    }
                                                    flex_intra_cluster_sync();
                                                    if (flex_is_first_core())
                                                    {
                                                        uint32_t _in_local_a = local_A;
                                                        uint32_t _in_local_b = local_B;
                                                        uint32_t _in_accumulator = accumulator;

                                                        ///////////////////
                                                        flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_FP_16);
                                                        flex_redmule_wait();
                                                        ///////////////////

                                                    }
                                                    flex_intra_cluster_sync();
                                                }
                                            }
                                            // accumulator = accumulator;
                                            // copy_memory: accumulator -> C, [64, 16], [16, 1], [N, 1], accumulator, C + ((((N * (((64 * ci) + (64 * gi)) + i)) + (16 * cj)) + (16 * gj)) + j)
                                            // is_sync = True
                                            // SoftHier_TCDM -> SoftHier_HBM
                                            if(flex_is_dm_core())
                                            {
                                                const int tile_width = N/2;
                                                const int tile_height = M/2;
                                                const int row_start = 64*ci + 64*gi + i;
                                                const int col_start = 16*cj + 16*gj + j;
                                                const int tile_row_index = row_start/tile_height;
                                                const int tile_col_index = col_start/tile_width;
                                                const int tile_row_offset = row_start%tile_height;
                                                const int tile_col_offset = col_start%tile_width;
                                                const int tile_index = tile_row_index*2 + tile_col_index;
                                                const int channel_id = 0 + (tile_index % 4) * 1;
                                                const int num_blocks_per_tile = (tile_height/64) * (tile_width/16);
                                                const int num_blocks_in_previous_tiles_in_channel = (tile_index / 4) * num_blocks_per_tile;
                                                const int block_row_index = tile_row_offset/64;
                                                const int block_col_index = tile_col_offset/16;
                                                const int block_index = block_row_index * (tile_width/16) + block_col_index;
                                                const int total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                                                const int block_addr = C + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 64 * 16 * 2;
                                                
                                                flex_dma_async_1d(hbm_addr(block_addr), local(accumulator), 64*16*2);
                                                flex_dma_async_wait_all();
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

    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("A: %x\n", A);
        printf("B: %x\n", B + ARCH_HBM_NODE_ADDR_SPACE);
        printf("C: %x\n", C + ARCH_HBM_NODE_ADDR_SPACE);
        printf("K: %x\n", K);
        printf("M: %x\n", M);
        printf("N: %x\n", N);
    }
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("%x\n", ((uint32_t *)(hbm_addr(A)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(B)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(C)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(A + ARCH_HBM_NODE_ADDR_SPACE)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(B + ARCH_HBM_NODE_ADDR_SPACE)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE)))[0]);
    }
    uint32_t eoc_val = 0;
    flex_global_barrier_xy();
    flex_timer_start();
    gemm_entry_0_0_0(A, B, C, K, M, N);
    flex_global_barrier_xy();
    flex_timer_end();
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        flex_print_int(((uint32_t *)(hbm_addr(C)))[0]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C)))[M*N/16 - 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C)))[M*N/16    ]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C)))[M*N/16 + 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C)))[M*N/8 - 1]); flex_print("\n");





        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE)))[0]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE)))[M*N/16 - 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE)))[M*N/16    ]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE)))[M*N/16 + 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE)))[M*N/8 - 1]); flex_print("\n");

        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 2)))[0]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 2)))[M*N/16 - 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 2)))[M*N/16    ]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 2)))[M*N/16 + 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 2)))[M*N/8 - 1]); flex_print("\n");

        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 3)))[0]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 3)))[M*N/16 - 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 3)))[M*N/16    ]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 3)))[M*N/16 + 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C + ARCH_HBM_NODE_ADDR_SPACE * 3)))[M*N/8 - 1]); flex_print("\n");
        
    }
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}

