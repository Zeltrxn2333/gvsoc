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

void nested_main_0_0_11(uint32_t A, uint32_t B, uint32_t accumulator, uint32_t K, uint32_t N) {
    uint32_t local_B;
    local_B = 2048;
    uint32_t local_A;
    local_A = 4096;
    long long bK;

    {
        // Start of state block_tiled_init
        //Framecode generating state block_tiled_init...
        // copy_memory: A -> local_A, [64, 32], [K, 1], [32, 1], A, local_A
        // is_sync = False
        // SoftHier_HBM -> SoftHier_TCDM 2D
        if(flex_is_dm_core())
        {
            flex_dma_async_2d(local(local_A), hbm_addr(A), 32*2, 32*2, K*2, 64);
            flex_dma_async_wait_all();
        }
        // copy_memory: B -> local_B, [32, 16], [N, 1], [16, 1], B, local_B
        // is_sync = False
        // SoftHier_HBM -> SoftHier_TCDM 2D
        if(flex_is_dm_core())
        {
            flex_dma_async_2d(local(local_B), hbm_addr(B), 16*2, 16*2, N*2, 32);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
        // End of state block_tiled_init

    }
    //Framecode generating state loop_block_tiled_guard...

    for (bK = 0; (bK < (K - 32)); bK = (bK + 32)) {
        {
            // Start of state block_tiled_double_buffered
            //Framecode generating state block_tiled_double_buffered...
            if (flex_is_first_core())
            {
                uint32_t _in_local_a = local_A + (2048 * ((bK / 32) % 2)) * 2;
                uint32_t _in_local_b = local_B + (512 * ((bK / 32) % 2)) * 2;
                uint32_t _in_accumulator = accumulator;

                ///////////////////
                flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_FP_16);
                flex_redmule_wait();
                ///////////////////

            }
            // copy_memory: A -> local_A, [64, 32], [K, 1], [32, 1], A + (bK + 32), local_A + (2048 * (((bK / 32) + 1) % 2))
            // is_sync = False
            // SoftHier_HBM -> SoftHier_TCDM 2D
            if(flex_is_dm_core())
            {
                flex_dma_async_2d(local(local_A + (2048 * (((bK / 32) + 1) % 2)) * 2), hbm_addr(A + (bK + 32) * 2), 32*2, 32*2, K*2, 64);
                flex_dma_async_wait_all();
            }
            // copy_memory: B -> local_B, [32, 16], [N, 1], [16, 1], B + (N * (bK + 32)), local_B + (512 * (((bK / 32) + 1) % 2))
            // is_sync = False
            // SoftHier_HBM -> SoftHier_TCDM 2D
            if(flex_is_dm_core())
            {
                flex_dma_async_2d(local(local_B + (512 * (((bK / 32) + 1) % 2)) * 2), hbm_addr(B + (N * (bK + 32)) * 2), 16*2, 16*2, N*2, 32);
                flex_dma_async_wait_all();
            }
            flex_intra_cluster_sync();
            // End of state block_tiled_double_buffered

        }
        //Framecode generating state loop_block_tiled_latch...

    }
    {
        // Start of state block_tiled_final_computation
        //Framecode generating state block_tiled_final_computation...
        if (flex_is_first_core())
        {
            uint32_t _in_local_a = local_A + (2048 * ((bK / 32) % 2)) * 2;
            uint32_t _in_local_b = local_B + (512 * ((bK / 32) % 2)) * 2;
            uint32_t _in_accumulator = accumulator;

            ///////////////////
            flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_FP_16);
            flex_redmule_wait();
            ///////////////////

        }
        flex_intra_cluster_sync();
        // End of state block_tiled_final_computation

    }
}



int __dace_init_cuda(struct GEMM_state_t *__state, int K, int M, int N) {
    
    

    // __state->gpu_context = new dace::cuda::Context(3, 1);

    

    return 0;
}

int __dace_exit_cuda(struct GEMM_state_t *__state) {
    
    int __err = 0;
    // delete __state->gpu_context;
    return __err;
}


void gemm_entry_0_0_0(uint32_t A, uint32_t B, uint32_t C, uint32_t K, uint32_t M, uint32_t N) {
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

                                            // Nested SDFG nested_main begin
                                            nested_main_0_0_11(A + (K * (((64 * ci) + (64 * gi)) + i)) * 2, B + (((16 * cj) + (16 * gj)) + j) * 2, accumulator, K, N);
                                            // copy_memory: accumulator -> C, [64, 16], [16, 1], [N, 1], accumulator, C + ((((N * (((64 * ci) + (64 * gi)) + i)) + (16 * cj)) + (16 * gj)) + j)
                                            // is_sync = True
                                            // SoftHier_TCDM -> SoftHier_HBM
                                            if(flex_is_dm_core())
                                            {
                                                flex_dma_async_2d_dummy(hbm_addr(C + ((((N * (((64 * ci) + (64 * gi)) + i)) + (16 * cj)) + (16 * gj)) + j) * 2), local(accumulator), 16*2, N*2, 16*2, 64);
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
        printf("B: %x\n", B);
        printf("C: %x\n", C);
        printf("K: %x\n", K);
        printf("M: %x\n", M);
        printf("N: %x\n", N);
    }
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("%x\n", ((uint32_t *)(hbm_addr(A)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(B)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(C)))[0]);
    }
    uint32_t eoc_val = 0;
    
    flex_timer_start();
    gemm_entry_0_0_0(A, B, C, K, M, N);
    flex_global_barrier_xy();
    flex_timer_end();
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        flex_print_int(((uint32_t *)(hbm_addr(C)))[0]); flex_print("\n");

        flex_print_int(((uint32_t *)(hbm_addr(C)))[M*N/4 - 1]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C)))[M*N/4    ]); flex_print("\n");
        flex_print_int(((uint32_t *)(hbm_addr(C)))[M*N/4 + 1]); flex_print("\n");

        flex_print_int(((uint32_t *)(hbm_addr(C)))[M*N/2 - 1]); flex_print("\n");
    }
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}