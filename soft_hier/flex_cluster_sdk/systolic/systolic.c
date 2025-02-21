
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


void nested_main_0_0_13(uint32_t A, uint32_t B, uint32_t accumulator, uint32_t K, uint32_t M, uint32_t N, uint32_t gi, uint32_t gj, uint32_t i, uint32_t j) {
    uint32_t local_A;
    local_A = 8192;
    uint32_t local_B;
    local_B = 24576;
    long long _c;

    //Framecode generating state init...
    for (_c = (((i + j) || 0) * (gi + gj)); (_c < ((((i < (M - (4 * 64))) || (j < (N - (4 * 64)))) ? ((gi + gj) + 1) : 7) + (K / 64))); _c = (_c + 1)) {
        //Framecode generating state systolic_start...
        if (((_c > (gi + gj)) && (_c <= ((gi + gj) + (K / 64))))) {
            {
                // Start of state systolic_compute
                //Framecode generating state systolic_compute...
                // accumulator = accumulator;
                // s_local_A = s_local_A;
                // local_A = local_A;
                // s_local_B = s_local_B;
                // local_B = local_B;
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
                // End of state systolic_compute

            }
        }
        if (((_c >= (gi + gj)) && (_c <= ((gi + gj) + (K / 64))))) {
            //Framecode generating state empty_comm...
            if ((gj == 0)) {
                {
                    // Start of state local_A_hbm
                    //Framecode generating state local_A_hbm...
                    // A = A;
                    // copy_memory: A -> s_local_A
                    // is_sync = False
                    if (flex_is_dm_core())
                    {
                        flex_dma_async_2d(remote_xy(gi,gj,local_A+((_c + 1) % 2) * 8192), hbm_addr(A + (((64 * _c) - (64 * gi)) - (64 * gj))*2), 64*2, 64*2, K*2, 64);
                        flex_dma_async_wait_all();
                    }
                    // s_local_A = s_local_A;
                    // local_A = local_A;
                    // copy_memory: local_A -> s_local_A
                    // is_sync = False
                    if (flex_is_dm_core())
                    {
                        bare_dma_start_1d(remote_xy(gi,((gj + 1) % 4),local_A+((_c + 1) % 2) * 8192), local(local_A + (4096 * (_c % 2))*2), 8192);
                        flex_dma_async_wait_all();
                    }
                    // s_local_A = s_local_A;
                    // End of state local_A_hbm

                }
            } else if (((gj > 0) && (gj < (4 - 1)))) {
                {
                    // Start of state local_A_tcdm
                    //Framecode generating state local_A_tcdm...
                    // local_A = local_A;
                    // copy_memory: local_A -> s_local_A
                    // is_sync = False
                    if (flex_is_dm_core())
                    {
                        bare_dma_start_1d(remote_xy(gi,((gj + 1) % 4),local_A+((_c + 1) % 2) * 8192), local(local_A + (4096 * (_c % 2))*2), 8192);
                        flex_dma_async_wait_all();
                    }
                    // s_local_A = s_local_A;
                    // End of state local_A_tcdm

                }
            }
            if ((gi == 0)) {
                {
                    // Start of state local_B_hbm
                    //Framecode generating state local_B_hbm...
                    // B = B;
                    // copy_memory: B -> s_local_B
                    // is_sync = False
                    if (flex_is_dm_core())
                    {
                        flex_dma_async_2d(remote_xy(gi,gj,local_B+((_c + 1) % 2) * 8192), hbm_addr(B + (N * (((64 * _c) - (64 * gi)) - (64 * gj)))*2), 64*2, 64*2, N*2, 64);
                        flex_dma_async_wait_all();
                    }
                    // s_local_B = s_local_B;
                    // local_B = local_B;
                    // copy_memory: local_B -> s_local_B
                    // is_sync = False
                    if (flex_is_dm_core())
                    {
                        bare_dma_start_1d(remote_xy(((gi + 1) % 4),gj,local_B+((_c + 1) % 2) * 8192), local(local_B + (4096 * (_c % 2))*2), 8192);
                        flex_dma_async_wait_all();
                    }
                    // s_local_B = s_local_B;
                    // End of state local_B_hbm

                }
            } else if (((gi > 0) && (gi < (4 - 1)))) {
                {
                    // Start of state local_B_tcdm
                    //Framecode generating state local_B_tcdm...
                    // local_B = local_B;
                    // copy_memory: local_B -> s_local_B
                    // is_sync = False
                    if (flex_is_dm_core())
                    {
                        bare_dma_start_1d(remote_xy(((gi + 1) % 4),gj,local_B+((_c + 1) % 2) * 8192), local(local_B + (4096 * (_c % 2))*2), 8192);
                        flex_dma_async_wait_all();
                    }
                    // s_local_B = s_local_B;
                    // End of state local_B_tcdm

                }
            }
        }
        {
            // Start of state canon_sync
            //Framecode generating state canon_sync...
            {

                ///////////////////

                if (flex_is_dm_core()) {
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                flex_global_barrier_xy();

                ///////////////////

            }
            // End of state canon_sync

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
                                            // DACE_ACL_CHECK(aclrtMemset(accumulator, 0, 4096 * sizeof(dace::float16)));

                                            if(flex_is_dm_core())
                                            {
                                                flex_dma_async_1d(local(accumulator), zomem(0), 8192);
                                                flex_dma_async_wait_all();
                                            }

                                            // accumulator = accumulator;
                                            {
                                                for (auto bK = 0; bK < K; bK += K) {
                                                    // Nested SDFG nested_main begin
                                                    nested_main_0_0_13(A + ((K * (((64 * ci) + (64 * gi)) + i)) + bK) * 2, B + ((((N * bK) + (64 * cj)) + (64 * gj)) + j) * 2, accumulator, K, M, N, gi, gj, i, j);
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

void test_remote_dma(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N)
{
    uint32_t localA;
    localA = 8192;
    uint32_t localB;
    localB = 24576;
    long long _c;

    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    int gi = get_pos(cluster_id).x;
    int gj = get_pos(cluster_id).y;
    if (gi == 0 && gj == 0)
    {
        if (flex_is_dm_core())
        {
            flex_dma_async_2d(local(localA), hbm_addr(A), 64*2, 64*2, K*2, 64);
            flex_dma_async_wait_all();
            for (auto bi = 0; bi < 64; bi++)
            {
                for (auto bj = 0; bj < 64; bj++)
                {
                    uint16_t local_a_val = ((uint16_t *)(local(localA)))[bi*64 + bj];
                    if (local_a_val != ((uint16_t *)(hbm_addr(A)))[bi*N+bj])
                    {
                        printf("local_a_HBM: %x\n", local_a_val);
                        printf("bi: %d\n", bi);
                        break;
                    }
                }
                    
            }
        }


        if (flex_is_dm_core())
        {
            bare_dma_start_1d(remote_xy(gi,((gj + 1) % 4),localA+8192), local(localA), 8192);
            flex_dma_async_wait_all();
        }

        flex_intra_cluster_sync();
    }
    flex_global_barrier_xy(); 

    if (gi == 0 && gj == 1){
        if (core_id == 0)
        {
            for (auto ai = 0; ai < 64; ai++)
            {
                for (auto aj = 0; aj < 64; aj++)
                {
                    uint16_t local_a_val = ((uint16_t *)(local(localA+8192)))[ai*64+aj];
                    if (local_a_val != ((uint16_t *)(hbm_addr(A)))[ai*N+aj])
                    {
                        printf("local_a: %x\n", local_a_val);
                        printf("ai: %d\n", ai);
                        break;
                    }
                }
                
            }
        }
    }   
}

void test_redmule(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N)
{
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    uint32_t local_A;
    local_A = 8192;
    uint32_t local_B;
    local_B = 24576;
    
    
    if (cluster_id == 0)
    {
        if (core_id == 0)
        {
            flex_redmule_config(1, 1, 1);
        }

        uint32_t accumulator;
        accumulator = 4;
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(local(accumulator), zomem(0), 8192);
            flex_dma_async_wait_all();
        }
        for(auto ii=0; ii < 4096; ii++)
        {
            if (flex_is_first_core())
            {
                uint32_t _in_local_a = local_A;
                uint32_t _in_local_b = local_B;
                uint32_t _in_accumulator = accumulator;

                ((uint16_t *)local_A)[0] = 0x3c00;
                ((uint16_t *)local_B)[0] = 0x3c00;
                ///////////////////
                flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_FP_16);
                flex_redmule_wait();
                ///////////////////
                printf("accumulator: %x\n", ((uint16_t *)local(accumulator))[0]);

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
        // printf("%x\n", ((uint32_t *)(hbm_addr(A)))[0]);
        // printf("%x\n", ((uint32_t *)(hbm_addr(B)))[0]);
        // printf("%x\n", ((uint32_t *)(hbm_addr(C)))[0]);
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
                    break;
                }
            }
        }
    }
    
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}

