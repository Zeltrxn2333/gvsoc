
// #include <cuda_runtime.h>
// #include <dace/dace.h>
#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

typedef struct soft_hier_vec_copy_state_t {
    int filler;
}soft_hier_vec_copy_state_t;



int __dace_init_cuda(struct soft_hier_vec_copy_state_t *__state);
int __dace_exit_cuda(struct soft_hier_vec_copy_state_t *__state);



int __dace_init_cuda(struct soft_hier_vec_copy_state_t *__state) {
    
    

    // __state->gpu_context = new dace::cuda::Context(1, 1);

    

    return 0;
}

int __dace_exit_cuda(struct soft_hier_vec_copy_state_t *__state) {
    
    int __err = 0;
    // delete __state->gpu_context;
    return __err;
}


void copy_map_outer_0_0_2(uint32_t soft_hier_A, uint32_t soft_hier_B) {
    {
        // TEST KERNEL SCOPE
        flex_global_barrier_xy();
        uint32_t cluster_id = flex_get_cluster_id();
        uint32_t core_id = flex_get_core_id();
        for (auto i = 0; i < 8192*4; i += 8192) {
            {
                // TEST DEVICE SCOPE
                uint32_t frag_A;
                frag_A = 0;
                uint32_t frag_B;
                frag_B = 1024;
                int ii = (512 * cluster_id);
                if (ii <= 8191) {
                    // Minels: [0], Maxels: [8191]
                    flex_intra_cluster_sync();
                    // copy_memory: soft_hier_A -> frag_A, [512], [1], [1], soft_hier_A + (i + ii), frag_A
                    // is_sync = True
                    // SoftHier_HBM -> SoftHier_TCDM
                    if(flex_is_dm_core())
                    {
                        flex_dma_async_1d(local(frag_A), hbm_addr(soft_hier_A + (i + ii) * 2), 512*2);
                        flex_dma_async_wait_all();
                    }
                    if (core_id == 0 && cluster_id == 10) {
                        // printf("cluster_id: %d, soft_hier_A+i+ii: %d\n", cluster_id, soft_hier_A + (i + ii));
                    }
                    flex_intra_cluster_sync();
                    // copy_memory: frag_A -> frag_B, [512], [1], [1], frag_A, frag_B
                    // is_sync = True
                    // SoftHier_TCDM -> SoftHier_TCDM
                    if(flex_is_dm_core())
                    {
                        flex_dma_async_1d(local(frag_B), local(frag_A), 512*2);
                        flex_dma_async_wait_all();
                    }
                    flex_intra_cluster_sync();
                    // copy_memory: frag_B -> soft_hier_B, [512], [1], [1], frag_B, soft_hier_B + (i + ii)
                    // is_sync = True
                    // SoftHier_TCDM -> SoftHier_HBM
                    if(flex_is_dm_core())
                    {
                        flex_dma_async_1d(hbm_addr(soft_hier_B + (i + ii) * 2), local(frag_B), 512*2);
                        flex_dma_async_wait_all();
                    }
                    flex_intra_cluster_sync();
                }
            }
            flex_intra_cluster_sync();
            // Finished deivelevel scope
        }
    }
}


void main(soft_hier_vec_copy_state_t *__state, uint32_t soft_hier_A, uint32_t soft_hier_B);
void main(soft_hier_vec_copy_state_t *__state, uint32_t soft_hier_A, uint32_t soft_hier_B)
{
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    soft_hier_A = ((uint32_t *)(hbm_addr(0)))[0];
    soft_hier_B = ((uint32_t *)(hbm_addr(4)))[0];

    uint32_t eoc_val = 0;
    flex_global_barrier_xy();
    flex_timer_start();
    copy_map_outer_0_0_2(soft_hier_A, soft_hier_B);
    flex_global_barrier_xy();
    flex_timer_end();
    flex_eoc(eoc_val);
    return 0;
}

