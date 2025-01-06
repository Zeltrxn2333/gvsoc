#include "flex_runtime.h"

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_start();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    if (flex_is_dm_core() && flex_get_cluster_id() == 0)
    {
        flex_print("hello: ");flex_print_int(25);flex_print(" is my id\n");
    }

    // example_one_cluster_gemm();
    // copy_map_outer_0_0_4(SOFT_HIER_A_HBM, SOFT_HIER_B_HBM);
    // gemm_entry_0_0_0(SOFT_HIER_A_HBM, SOFT_HIER_B_HBM, SOFT_HIER_C_HBM, 512, 512, 512);
    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_end();
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}