
// #include <cuda_runtime.h>
// #include <dace/dace.h>
#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

#define floor(x) ((x))

typedef struct GEMM_state_t {
    int filler;
}GEMM_state_t;



int __dace_init_cuda(struct GEMM_state_t *__state);
int __dace_exit_cuda(struct GEMM_state_t *__state);


void nested_main_1_0_4(uint32_t A, uint32_t B, uint32_t accumulator, uint32_t gi, uint32_t gj) {
    uint32_t local_A;
    local_A = 512;
    uint32_t local_B;
    local_B = 1536;
    long long _c;

    //Framecode generating state init...
    for (_c = (floor((gi / 2)) + floor((gj / 2))); (_c <= ((floor((gi / 2)) + floor((gj / 2))) + (256 / 16))); _c = (_c + 1)) {
        //Framecode generating state start...
        if (((_c > (floor((gi / 2)) + floor((gj / 2)))) && (_c <= ((floor((gi / 2)) + floor((gj / 2))) + (256 / 16))))) {
            {
                // Start of state compute
                //Framecode generating state compute...
                // local_A = local_A;
                // local_B = local_B;
                // accumulator = accumulator;
                if (flex_is_first_core())
                {
                    uint32_t _in_local_a = local_A + (256 * (_c % 2)) * 2;
                    uint32_t _in_local_b = local_B + (256 * (_c % 2)) * 2;
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
        if (((_c >= (floor((gi / 2)) + floor((gj / 2)))) && (_c < ((floor((gi / 2)) + floor((gj / 2))) + (256 / 16))))) {
            //Framecode generating state local_248...
            if (((floor((gi / 2)) == 0) && ((gi % 2) == 0))) {
                {
                    // Start of state local_434
                    //Framecode generating state local_434...
                    // B = B;
                    // copy_memory: B -> local_B, [16, 16], [256, 1], [16, 1], B + (((4096 * _c) - (4096 * (gi / 2))) - (4096 * (gj / 2))), local_B + (256 * ((_c + 1) % 2))
                    // is_sync = True
                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core())
                    {
                        flex_dma_sync_2d(local(local_B + (256 * ((_c + 1) % 2)) * 2), hbm_addr(B + (((4096 * _c) - (4096 * (gi / 2))) - (4096 * (gj / 2))) * 2), 16*2, 16*2, 256*2, 16);
                        flex_dma_async_wait_all();
                    }
                    // local_B = local_B;
                    if (flex_is_dm_core())
                    {
                        flex_dma_async_1d_broadcast(dace_remote_xy(gi + 1,gj,local_B+((_c + 1) % 2) * 512,8), local(local_B + (256 * ((_c + 1) % 2))*2), 512);
                        flex_dma_async_wait_all();
                    }
                    // s_local_B = s_local_B;
                    // End of state local_434

                }
            } else if (((floor((gi / 2)) > 0) && ((gi % 2) == 0))) {
                {
                    // Start of state local_886
                    //Framecode generating state local_886...
                    // s_local_B = s_local_B;
                    // copy_memory: s_local_B -> local_B
                    // is_sync = True
                    if (flex_is_dm_core())
                    {
                        bare_dma_start_1d(local(local_B + (256 * ((_c + 1) % 2))*2), dace_remote_xy(((gi + 7) % 8),gj,local_B+(_c % 2) * 512,8), 512);
                        flex_dma_async_wait_all();
                    }
                    // local_B = local_B;
                    if (flex_is_dm_core())
                    {
                        flex_dma_async_1d_broadcast(dace_remote_xy(gi + 1,gj,local_B+((_c + 1) % 2) * 512,8), local(local_B + (256 * ((_c + 1) % 2))*2), 512);
                        flex_dma_async_wait_all();
                    }
                    // s_local_B = s_local_B;
                    // End of state local_886

                }
            } else if (((gi % 2) > 0)) {
                {
                    // Start of state local_494
                    //Framecode generating state local_494...
                    // s_local_B = s_local_B;
                    // copy_memory: s_local_B -> local_B
                    // is_sync = False
                    // local_B = local_B;
                    // End of state local_494

                }
            }
            if (((floor((gj / 2)) == 0) && ((gj % 2) == 0))) {
                {
                    // Start of state local_963
                    //Framecode generating state local_963...
                    // A = A;
                    // copy_memory: A -> local_A, [16, 16], [256, 1], [16, 1], A + (((16 * _c) - (16 * (gi / 2))) - (16 * (gj / 2))), local_A + (256 * ((_c + 1) % 2))
                    // is_sync = True
                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core())
                    {
                        flex_dma_sync_2d(local(local_A + (256 * ((_c + 1) % 2)) * 2), hbm_addr(A + (((16 * _c) - (16 * (gi / 2))) - (16 * (gj / 2))) * 2), 16*2, 16*2, 256*2, 16);
                        flex_dma_async_wait_all();
                    }
                    // local_A = local_A;
                    if (flex_is_dm_core())
                    {
                        flex_dma_async_1d_broadcast(dace_remote_xy(gi,gj + 1,local_A+((_c + 1) % 2) * 512,8), local(local_A + (256 * ((_c + 1) % 2))*2), 512);
                        flex_dma_async_wait_all();
                    }
                    // s_local_A = s_local_A;
                    // End of state local_963

                }
            } else if (((floor((gj / 2)) > 0) && ((gj % 2) == 0))) {
                {
                    // Start of state local_780
                    //Framecode generating state local_780...
                    // s_local_A = s_local_A;
                    // copy_memory: s_local_A -> local_A
                    // is_sync = True
                    if (flex_is_dm_core())
                    {
                        bare_dma_start_1d(local(local_A + (256 * ((_c + 1) % 2))*2), dace_remote_xy(gi,((gj + 7) % 8),local_A+(_c % 2) * 512,8), 512);
                        flex_dma_async_wait_all();
                    }
                    // local_A = local_A;
                    if (flex_is_dm_core())
                    {
                        flex_dma_async_1d_broadcast(dace_remote_xy(gi,gj + 1,local_A+((_c + 1) % 2) * 512,8), local(local_A + (256 * ((_c + 1) % 2))*2), 512);
                        flex_dma_async_wait_all();
                    }
                    // s_local_A = s_local_A;
                    // End of state local_780

                }
            } else if (((gj % 2) > 0)) {
                {
                    // Start of state local_286
                    //Framecode generating state local_286...
                    // s_local_A = s_local_A;
                    // copy_memory: s_local_A -> local_A
                    // is_sync = False
                    // local_A = local_A;
                    // End of state local_286

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

void nested_main_0_0_9(uint32_t A, uint32_t B, uint32_t C, uint32_t gi, uint32_t gj, uint32_t i, uint32_t j) {

    {
        // Start of state pre_shift
        //Framecode generating state pre_shift...
        {

            ///////////////////

            if ((i == 0) && (j == 0))
            {
                for (int sync_iter = 0; sync_iter < floor(gi/2)+floor(gj/2); sync_iter++){
                    flex_global_barrier_xy();
                }
                if (flex_is_dm_core()) {
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                flex_global_barrier_xy();
            }

            ///////////////////

        }
        // End of state pre_shift

    }
    {
        // Start of state block
        //Framecode generating state block...
        uint32_t accumulator;
        accumulator = 0;
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(local(accumulator), zomem(0), 512);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        // accumulator = accumulator;
        // A = A;
        // B = B;
        {
            for (int bK = 0; bK < 256; bK += 256) {
                // Nested SDFG nested_main begin
                nested_main_1_0_4(A + bK * 2, B + (256 * bK) * 2, accumulator, gi, gj);
            }
        }
        // accumulator = accumulator;
        // copy_memory: accumulator -> C, [16, 16], [16, 1], [256, 1], accumulator, C
        // is_sync = True
        // SoftHier_TCDM -> SoftHier_HBM
        if(flex_is_dm_core())
        {
            flex_dma_sync_2d(hbm_addr(C), local(accumulator), 16*2, 256*2, 16*2, 16);
        }
        // C = C;
        // End of state block

    }
    {
        // Start of state post_shift
        //Framecode generating state post_shift...
        {

            ///////////////////

            if ((i >= 256 - 8*16) && (j >= 256 - 8*16))
            {
                for (int sync_iter = 0; sync_iter < 8 - 1 - (floor(gi/2)+floor(gj/2)) - 1; sync_iter++){
                    flex_global_barrier_xy();
                }
                if (flex_is_dm_core()) {
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                flex_global_barrier_xy();
            }

            ///////////////////

        }
        // End of state post_shift

    }
}



int __dace_init_cuda(struct GEMM_state_t *__state) {
    
    

    // __state->gpu_context = new dace::cuda::Context(1, 2);

    

    return 0;
}

int __dace_exit_cuda(struct GEMM_state_t *__state) {
    
    int __err = 0;
    // delete __state->gpu_context;
    return __err;
}


void gemm_entry_0_0_0(const uint32_t A, const uint32_t B, const uint32_t C) {
    {
        // TEST KERNEL SCOPE
        flex_global_barrier_xy();
        uint32_t cluster_id = flex_get_cluster_id();
        uint32_t core_id = flex_get_core_id();
        {
            for (int i = 0; i < 256; i += 128) {
                for (int j = 0; j < 256; j += 128) {
                    {
                        // TEST DEVICE SCOPE
                        int gi = cluster_id % 8;
                        int gj = cluster_id / 8;
                        if (gi <= 7) {
                            if (gj <= 7) {
                                // Minels: [0, 0], Maxels: [7, 7]
                                // Configure RedMule Here
                                if(flex_is_first_core())
                                {
                                    flex_redmule_config(16, 16, 16);
                                }
                                flex_intra_cluster_sync();
                                {
                                    for (int ci = 0; ci < 16; ci += 16) {
                                        for (int cj = 0; cj < 16; cj += 16) {
                                            // Nested SDFG nested_main begin
                                            nested_main_0_0_9(A + (((4096 * ci) + (4096 * gi)) + (256 * i)) * 2, B + (((16 * cj) + (16 * gj)) + j) * 2, C + ((((((4096 * ci) + (16 * cj)) + (4096 * gi)) + (16 * gj)) + (256 * i)) + j) * 2, gi, gj, i, j);
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
        printf("%x\n", ((uint32_t *)(hbm_addr(A)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(B)))[0]);
        printf("%x\n", ((uint32_t *)(hbm_addr(C)))[0]);
    }
    uint32_t eoc_val = 0;
    flex_global_barrier_xy();
    flex_timer_start();
    gemm_entry_0_0_0(A, B, C);
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
    flex_timer_end();
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        for (int gi = 0; gi < M; gi++){
            for (int gj = 0; gj < N; gj++){
                if (((uint16_t *)(hbm_addr(C)))[gj + gi * N] != ((uint16_t *)(hbm_addr(G)))[gj + gi * N]){
                    printf("%d, %d, %x, %x\n", gi, gj, ((uint16_t *)(hbm_addr(C)))[gj + gi * N], ((uint16_t *)(hbm_addr(G)))[gj + gi * N]);
                    // break;
                }
            }
        }
    }
    
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return;
}

// if (flex_is_dm_core())
//         {
//             flex_dma_async_1d(local(accumulator), zomem(0), 512);
//             flex_dma_async_wait_all();
//         }
//         flex_intra_cluster_sync();
