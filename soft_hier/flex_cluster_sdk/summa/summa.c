
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

void nested_main_0_0_13(uint32_t A, uint32_t B, uint32_t accumulator, uint32_t K, uint32_t N, uint32_t gi, uint32_t gj) {
    uint32_t local_B;
    local_B = 8192;
    uint32_t local_A;
    local_A = 24576;
    long long _c;

    //Framecode generating state init...
    //Framecode generating state init_sync...
    for (_c = 0; (_c <= (K / 64)); _c = (_c + 1)) {
        //Framecode generating state start...
        if ((_c > 0)) {
            {
                // Start of state compute
                //Framecode generating state compute...
                // local_A = local_A;
                // local_B = local_B;
                // accumulator = accumulator;
                if (flex_is_first_core())
                {
                    uint32_t _in_local_a = local_A + (4096 * (_c % 2)) * 2;
                    uint32_t _in_local_b = local_B + (4096 * (_c % 2)) * 2;
                    uint32_t _in_accumulator = accumulator;

                    ///////////////////
                    flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_FP_16);
                    flex_redmule_wait();
                    ///////////////////

                }
                // accumulator = accumulator;
                // End of state compute

            }
        }
        if ((_c < (K / 64))) {
            //Framecode generating state empty_comm...
            if ((gi == 0)) {
                {
                    // Start of state local_B_hbm
                    //Framecode generating state local_B_hbm...
                    // B = B;
                    // copy_memory: B -> local_B, [64, 64], [N, 1], [64, 1], B + ((64 * N) * _c), local_B + (4096 * ((_c + 1) % 2))
                    // is_sync = True
                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core())
                    {
                        flex_dma_async_2d(local(local_B + (4096 * ((_c + 1) % 2)) * 2), hbm_addr(B + ((64 * N) * _c) * 2), 64*2, 64*2, N*2, 64);
                        flex_dma_async_wait_all();
                    }
                    // local_B = local_B;
                    if (flex_is_dm_core())
                    {
                        flex_dma_async_1d_broadcast(remote_xy(gi + 3,gj,local_B+((_c + 1) % 2) * 8192), local(local_B + (4096 * ((_c + 1) % 2))*2), 8192);
                        flex_dma_async_wait_all();
                    }
                    // s_local_B = s_local_B;
                    // End of state local_B_hbm

                }
            } else if ((gi > 0)) {
                {
                    // Start of state local_B_tcdm
                    //Framecode generating state local_B_tcdm...
                    // s_local_B = s_local_B;
                    // copy_memory: s_local_B -> local_B
                    // is_sync = False
                    // local_B = local_B;
                    // End of state local_B_tcdm

                }
            }
            if ((gj == 0)) {
                {
                    // Start of state local_A_hbm
                    //Framecode generating state local_A_hbm...
                    // A = A;
                    // copy_memory: A -> local_A, [64, 64], [K, 1], [64, 1], A + (64 * _c), local_A + (4096 * ((_c + 1) % 2))
                    // is_sync = True
                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core())
                    {
                        flex_dma_async_2d(local(local_A + (4096 * ((_c + 1) % 2)) * 2), hbm_addr(A + (64 * _c) * 2), 64*2, 64*2, K*2, 64);
                        flex_dma_async_wait_all();
                    }
                    // local_A = local_A;
                    if (flex_is_dm_core())
                    {
                        flex_dma_async_1d_broadcast(remote_xy(gi,gj + 3,local_A+((_c + 1) % 2) * 8192), local(local_A + (4096 * ((_c + 1) % 2))*2), 8192);
                        flex_dma_async_wait_all();
                    }
                    // s_local_A = s_local_A;
                    // End of state local_A_hbm

                }
            } else if ((gj > 0)) {
                {
                    // Start of state local_A_tcdm
                    //Framecode generating state local_A_tcdm...
                    // s_local_A = s_local_A;
                    // copy_memory: s_local_A -> local_A
                    // is_sync = False
                    // local_A = local_A;
                    // End of state local_A_tcdm

                }
            }
        }
        {
            // Start of state sync
            //Framecode generating state sync...
            {

                ///////////////////

                if (flex_is_dm_core()) {
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                flex_global_barrier_xy();

                ///////////////////

            }
            // End of state sync

        }

    }
}



int __dace_init_cuda(struct GEMM_state_t *__state, int K, int M, int N) {
    
    

    // __state->gpu_context = new dace::cuda::Context(1, 1);

    

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
                for (auto j = 0; j < N; j += 256) {
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
                                    flex_redmule_config(64, 64, 64);
                                }
                                flex_intra_cluster_sync();
                                {
                                    for (auto ci = 0; ci < 64; ci += 64) {
                                        for (auto cj = 0; cj < 64; cj += 64) {
                                            uint32_t accumulator;
                                            accumulator = 0;
                                            if (flex_is_dm_core())
                                            {
                                                flex_dma_async_1d(local(accumulator), zomem(0), 64*64*2);
                                                flex_dma_async_wait_all();
                                            }

                                            // accumulator = accumulator;
                                            {
                                                for (auto bK = 0; bK < K; bK += K) {
                                                    // Nested SDFG nested_main begin
                                                    nested_main_0_0_13(A + ((K * (((64 * ci) + (64 * gi)) + i)) + bK) * 2, B + ((((N * bK) + (64 * cj)) + (64 * gj)) + j) * 2, accumulator, K, N, gi, gj);
                                                }
                                            }
                                            // accumulator = accumulator;
                                            // copy_memory: accumulator -> C, [64, 64], [64, 1], [N, 1], accumulator, C + ((((N * (((64 * ci) + (64 * gi)) + i)) + (64 * cj)) + (64 * gj)) + j)
                                            // is_sync = True
                                            // SoftHier_TCDM -> SoftHier_HBM
                                            if(flex_is_dm_core())
                                            {
                                                flex_dma_async_2d_dummy(hbm_addr(C + ((((N * (((64 * ci) + (64 * gi)) + i)) + (64 * cj)) + (64 * gj)) + j) * 2), local(accumulator), 64*2, N*2, 64*2, 64);
                                            }
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


void test_broadcast(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N) {
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    int gi = get_pos(cluster_id).x;
    int gj = get_pos(cluster_id).y;
    uint32_t local_test = 10;
    if (gi == 0 && gj ==0)
    {
        if(flex_is_dm_core())
        {
            flex_dma_async_1d(local(local_test), hbm_addr(B), 128);
            flex_dma_async_wait_all();
            flex_dma_async_1d_broadcast(remote_xy(0,3,local_test), local(local_test), 128);
            flex_dma_async_wait_all();
        }
    }

    flex_global_barrier_xy();
    if (gi == 0 && gj == 3)
    {
        if(flex_is_dm_core())
        {
            for (auto ii = 0; ii < 64; ii++)
            {
                uint16_t local_test_val = ((uint16_t *)(local(local_test)))[ii];
                if(local_test_val != 0x4000)
                {
                    printf("local_test: %x\n", local_test_val);
                }
            }
        }
    }
    flex_global_barrier_xy();
    if (gi == 0 && gj == 2)
    {
        if(flex_is_dm_core())
        {
            for (auto ii = 0; ii < 64; ii++)
            {
                uint16_t local_test_val = ((uint16_t *)(local(local_test)))[ii];
                if(local_test_val != 0x4000)
                {
                    printf("local_test: %x\n", local_test_val);
                }
            }
        }
    }
    flex_global_barrier_xy();
    if (gi == 0 && gj == 1)
    {
        if(flex_is_dm_core())
        {
            for (auto ii = 0; ii < 64; ii++)
            {
                uint16_t local_test_val = ((uint16_t *)(local(local_test)))[ii];
                if(local_test_val != 0x4000)
                {
                    printf("local_test: %x\n", local_test_val);
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
    uint32_t G = ((uint32_t *)(hbm_addr(24)))[0];

    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("A: %x\n", A);
        printf("B: %x\n", B);
        printf("C: %x\n", C);
        printf("K: %x\n", K);
        printf("M: %x\n", M);
        printf("N: %x\n", N);
        printf("G: %x\n", G);
    }
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("%x\n", ((uint16_t *)(hbm_addr(A)))[0]);
        printf("%x\n", ((uint16_t *)(hbm_addr(B)))[0]);
        printf("%x\n", ((uint16_t *)(hbm_addr(C)))[0]);
    }
    uint32_t eoc_val = 0;
    flex_global_barrier_xy();
    flex_timer_start();
    gemm_entry_0_0_0(A, B, C, K, M, N);
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
    flex_timer_end();
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        for (auto gi = 0; gi < M; gi++){
            for (auto gj = 0; gj < N; gj++){
                if (((uint16_t *)(hbm_addr(C)))[gj + gi * N] != ((uint16_t *)(hbm_addr(G)))[gj + gi * N]){
                    printf("%d, %d, %x, %x\n", gi, gj, ((uint16_t *)(hbm_addr(C)))[gj + gi * N], ((uint16_t *)(hbm_addr(G)))[gj + gi * N]);
                    // break;
                }
            }
        }
    }
    
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}

