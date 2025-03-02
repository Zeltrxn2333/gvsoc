
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
            bare_dma_start_1d(local(localA), hbm_addr(A), 8192);
            bare_dma_start_1d(local(localB), hbm_addr(B+12*ARCH_HBM_NODE_ADDR_SPACE), 8192);
            flex_dma_async_wait_all();
            bare_dma_start_1d(remote_xy(gi, ((gj + 1) % 4), localA), local(localA+8192), 8192);
            bare_dma_start_1d(remote_xy(((gi + 1) % 4), gj, localB), local(localB+8192), 8192);
            flex_dma_async_wait_all();
        
        }
    }
    flex_global_barrier_xy(); 
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
    test_remote_dma(A, B, C, K, M, N);
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
    flex_timer_end();    
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}

