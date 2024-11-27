#include "flex_runtime.h"
#include "example_one_cluster_gemm.h"
#include <math.h>

#define SOFT_HIER_A_HBM 0
#define SOFT_HIER_B_HBM 512*2
#define SOFT_HIER_C_HBM (SOFT_HIER_B_HBM + 512*2)
int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    flex_timer_start();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    // example_one_cluster_gemm();
    // copy_map_outer_0_0_4(SOFT_HIER_A_HBM, SOFT_HIER_B_HBM);
    gemm_entry_0_0_0(SOFT_HIER_A_HBM, SOFT_HIER_B_HBM, SOFT_HIER_C_HBM, 512, 512, 512);
    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_timer_end();
    flex_eoc(eoc_val);
    return 0;
}