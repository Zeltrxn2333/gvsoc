#ifndef _EXAMPLE_ONE_CLUSTER_GEMM_H_
#define _EXAMPLE_ONE_CLUSTER_GEMM_H_

#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

// GEMM M-N-K           : 1024-1024-1024
// Elem Size            : FP16
// Assumption           : Data are already tiled in HBM

#define ELEM_SIZE      2
#define GEMM_DIMENSION 1024
#define GEMM_SIZE_BYTE (GEMM_DIMENSION * GEMM_DIMENSION * ELEM_SIZE)
#define TILE_DIMENSION 256
#define TILE_SIZE_BYTE (TILE_DIMENSION * TILE_DIMENSION * ELEM_SIZE)
#define TILES_PER_DIM  (GEMM_DIMENSION/TILE_DIMENSION)

#define X_HBM_OFFSET   0
#define W_HBM_OFFSET   (X_HBM_OFFSET + GEMM_SIZE_BYTE)
#define Y_HBM_OFFSET   (W_HBM_OFFSET + GEMM_SIZE_BYTE)
#define Z_HBM_OFFSET   (Y_HBM_OFFSET + GEMM_SIZE_BYTE)

#define X_L1_OFFSET1   0
#define W_L1_OFFSET1   (X_L1_OFFSET1 + TILE_SIZE_BYTE)
#define X_L1_OFFSET2   (W_L1_OFFSET1 + TILE_SIZE_BYTE)
#define W_L1_OFFSET2   (X_L1_OFFSET2 + TILE_SIZE_BYTE)
#define YZ_L1_OFFSET   (W_L1_OFFSET2 + TILE_SIZE_BYTE)

void example_one_cluster_gemm(){
    flex_global_barrier_xy();//Global barrier

    uint32_t CID = flex_get_cluster_id();//Get cluster ID

    //Initialize RedMule Paramters
    if (flex_is_first_core() && (CID == 0))//Use the first core in cluster 0 to configure RedMule
    {
        //Configure M-N-K of tile that RedMule will accelerate
        flex_redmule_config(TILE_DIMENSION, TILE_DIMENSION, TILE_DIMENSION);
    }

    flex_global_barrier_xy();//Global barrier

    if (CID == 0)//Only let Cluster 0 to work
    {
        //Iterate over every Z tiles
        for (int row = 0; row < TILES_PER_DIM; ++row)
        {
            for (int col = 0; col < TILES_PER_DIM; ++col)
            {
                //Start address of tiles
                uint32_t X_hbm_addr = X_HBM_OFFSET + row * TILES_PER_DIM * TILE_SIZE_BYTE;
                uint32_t W_hbm_addr = W_HBM_OFFSET + col * TILE_SIZE_BYTE;
                uint32_t Y_hbm_addr = Y_HBM_OFFSET + row * TILES_PER_DIM * TILE_SIZE_BYTE + col * TILE_SIZE_BYTE;
                uint32_t Z_hbm_addr = Z_HBM_OFFSET + row * TILES_PER_DIM * TILE_SIZE_BYTE + col * TILE_SIZE_BYTE;


                //Preload Y,X,W tiles
                if (flex_is_dm_core()){//Use DM core to trigger DMA transcations
                    //Trigger DMA transaction: move Y tile from HBM to L1
                    flex_dma_async_1d(local(YZ_L1_OFFSET),hbm_addr(Y_hbm_addr), TILE_SIZE_BYTE);

                    //Trigger DMA transaction: move X tile from HBM to L1
                    flex_dma_async_1d(local(X_L1_OFFSET1),hbm_addr(X_hbm_addr), TILE_SIZE_BYTE);

                    //Trigger DMA transaction: move W tile from HBM to L1
                    flex_dma_async_1d(local(W_L1_OFFSET1),hbm_addr(W_hbm_addr), TILE_SIZE_BYTE);

                    //Wait all DMA transaction done
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();//Cluster barrier

                //Compute X and W to YZ in L1 + preload next X and W
                for (int i = 0; i < (TILES_PER_DIM-1); ++i)
                {
                    //Calculate next X and W tile address in HBM
                    X_hbm_addr += TILE_SIZE_BYTE;
                    W_hbm_addr += TILES_PER_DIM * TILE_SIZE_BYTE;

                    //Ping-Pong operations on Double-Buffering X and W
                    if (i%2 == 0)
                    {
                        if (flex_is_dm_core()){//Use DM core to trigger DMA transcations
                            //Trigger DMA transaction: move X tile from HBM to L1
                            flex_dma_async_1d(local(X_L1_OFFSET2),hbm_addr(X_hbm_addr), TILE_SIZE_BYTE);

                            //Trigger DMA transaction: move W tile from HBM to L1
                            flex_dma_async_1d(local(W_L1_OFFSET2),hbm_addr(W_hbm_addr), TILE_SIZE_BYTE);

                            //Wait all DMA transaction done
                            flex_dma_async_wait_all();
                        }

                        if (flex_is_first_core())//Use the first core in cluster 0 to configure and trigger RedMule
                        {
                            //Configure tile address in L1 and run RedMule acceleration
                            flex_redmule_trigger(X_L1_OFFSET1, W_L1_OFFSET1, YZ_L1_OFFSET, REDMULE_FP_16);

                            //Wait RedMule Done
                            flex_redmule_wait();
                        }
                    } else {
                        if (flex_is_dm_core()){//Use DM core to trigger DMA transcations
                            //Trigger DMA transaction: move X tile from HBM to L1
                            flex_dma_async_1d(local(X_L1_OFFSET1),hbm_addr(X_hbm_addr), TILE_SIZE_BYTE);

                            //Trigger DMA transaction: move W tile from HBM to L1
                            flex_dma_async_1d(local(W_L1_OFFSET1),hbm_addr(W_hbm_addr), TILE_SIZE_BYTE);

                            //Wait all DMA transaction done
                            flex_dma_async_wait_all();
                        }

                        if (flex_is_first_core())//Use the first core in cluster 0 to configure and trigger RedMule
                        {
                            //Configure tile address in L1 and run RedMule acceleration
                            flex_redmule_trigger(X_L1_OFFSET2, W_L1_OFFSET2, YZ_L1_OFFSET, REDMULE_FP_16);

                            //Wait RedMule Done
                            flex_redmule_wait();
                        }
                    }
                    flex_intra_cluster_sync();//Cluster barrier
                }

                //Last Computation
                if (flex_is_first_core())
                {
                    flex_redmule_trigger(X_L1_OFFSET2, W_L1_OFFSET2, YZ_L1_OFFSET, REDMULE_FP_16);
                    flex_redmule_wait();
                }
                flex_intra_cluster_sync();//Cluster barrier

                //Store Z tile
                if (flex_is_dm_core()){
                    //Trigger DMA transaction: move Z tile from L1 to HBM
                    flex_dma_async_1d(hbm_addr(Z_hbm_addr),local(YZ_L1_OFFSET), TILE_SIZE_BYTE);

                    //Wait all DMA transaction done
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();//Cluster barrier
            }
        }
    }

    flex_global_barrier_xy();//Global barrier

}


#define FRAG_A_SIZE (1024)
#define FRAG_A_ADDR (0)
#define FRAG_B_SIZE (1024)
#define FRAG_B_ADDR ((0 + FRAG_A_SIZE))
void copy_map_outer_0_0_4(uint32_t soft_hier_A, uint32_t soft_hier_B) {
    {
        // TEST KERNEL SCOPE
        flex_global_barrier_xy();
        uint32_t cluster_id = flex_get_cluster_id();
        uint32_t core_id = flex_get_core_id();
        int i = 0;
        flex_global_barrier_xy();
        {
            // TEST DEVICE SCOPE
            uint32_t frag_A = FRAG_A_ADDR;
            uint32_t frag_B = FRAG_B_ADDR;
            int ii = (512 * cluster_id);
            if (ii <= 8191) {
                // Minels: [0], Maxels: [8191]
                // SoftHier_HBM -> SoftHier_TCDM
                if(flex_is_dm_core())
                {
                    flex_dma_async_1d(local(frag_A), hbm_addr(soft_hier_A + (i + ii)), 512*2);
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                // SoftHier_TCDM -> SoftHier_TCDM
                if(flex_is_dm_core())
                {
                    flex_dma_async_1d(local(frag_B), local(frag_A), 512*2);
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                // SoftHier_TCDM -> SoftHier_HBM
                if(flex_is_dm_core())
                {
                    flex_dma_async_1d(hbm_addr(soft_hier_B + (i + ii)), local(frag_B), 512*2);
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
            }
        }
        flex_global_barrier_xy();
        // Finished deivelevel scope
    }
}



#define ACCUMULATOR_SIZE (2048)
#define ACCUMULATOR_ADDR (0)
#define LOCAL_A_SIZE (2048)
#define LOCAL_A_ADDR ((0 + ACCUMULATOR_SIZE))
#define LOCAL_B_SIZE (2048)
#define LOCAL_B_ADDR (((0 + ACCUMULATOR_SIZE) + LOCAL_A_SIZE))


void gemm_entry_0_0_0(uint32_t A, uint32_t B, uint32_t C, uint32_t K, uint32_t M, uint32_t N) {
    {
        // TEST KERNEL SCOPE
        flex_global_barrier_xy();
        uint32_t cluster_id = flex_get_cluster_id();
        uint32_t core_id = flex_get_core_id();
        {
            for (auto j = 0; j < M; j += 256) {
                for (auto i = 0; i < N; i += 256) {
            // int i = 0;
            // int j = 0;
                    flex_global_barrier_xy();
                    {
                        // TEST DEVICE SCOPE
                        uint32_t accumulator = ACCUMULATOR_ADDR;
                        uint32_t local_A = LOCAL_A_ADDR;
                        uint32_t local_B = LOCAL_B_ADDR;
                        int gj = get_pos(cluster_id).x;
                        int gi = get_pos(cluster_id).y;
                        if (gj <= 3) {
                            if (gi <= 3) {
                                // Minels: [0, 0], Maxels: [3, 3]
                                // Configure RedMule Here
                                if(flex_is_first_core())
                                {
                                    flex_redmule_config(32, 32, 32);
                                }
                                // CPU Scope <dace.sdfg.scope.ScopeSubgraphView object at 0x7f7af6e9e3d0>
                                {
                                    for (auto ci = 0; ci < 64; ci += 32) {
                                        for (auto cj = 0; cj < 64; cj += 32) {
                                            // CPU Scope <dace.sdfg.scope.ScopeSubgraphView object at 0x7f7af6e9e8d0>
                                            {
                                                for (auto bK = 0; bK < K; bK += 32) {
                                                    // SoftHier: Emitting copy from A to local_A
                                                    // copy_memory: A -> local_A, [32, 32], [K, 1], [32, 1], A + ((K * (((32 * ci) + (64 * gi)) + i)) + bK), local_A
                                                    // subset.string_list() = ['32*ci + 64*gi + i:32*ci + 64*gi + i + 32', 'bK:bK + 32']
                                                    // SoftHier_HBM -> SoftHier_TCDM 2D
                                                    if(flex_is_dm_core())
                                                    {
                                                        flex_dma_async_2d(local(local_A), hbm_addr(A + ((K * (((32 * ci) + (64 * gi)) + i)) + bK)), 32*2, 32*2, K*2, 32);
                                                        flex_dma_async_wait_all();
                                                    }
                                                    flex_intra_cluster_sync();
                                                    // SoftHier: Emitting copy from B to local_B
                                                    // copy_memory: B -> local_B, [32, 32], [N, 1], [32, 1], B + ((((N * bK) + (32 * cj)) + (64 * gj)) + j), local_B
                                                    // subset.string_list() = ['bK:bK + 32', '32*cj + 64*gj + j:32*cj + 64*gj + j + 32']
                                                    // SoftHier_HBM -> SoftHier_TCDM 2D
                                                    if(flex_is_dm_core())
                                                    {
                                                        flex_dma_async_2d(local(local_B), hbm_addr(B + ((((N * bK) + (32 * cj)) + (64 * gj)) + j)), 32*2, 32*2, N*2, 32);
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
                                                    // SoftHier: Emitting copy from accumulator to accumulator
                                                    // copy_memory: accumulator -> accumulator, [1024], [1], [1], accumulator, accumulator
                                                }
                                            }
                                            // SoftHier: Emitting copy from accumulator to C
                                            // copy_memory: accumulator -> C, [32, 32], [32, 1], [N, 1], accumulator, C + ((((N * (((32 * ci) + (64 * gi)) + i)) + (32 * cj)) + (64 * gj)) + j)
                                            // subset.string_list() = ['32*ci + 64*gi + i:32*ci + 64*gi + i + 32', '32*cj + 64*gj + j:32*cj + 64*gj + j + 32']
                                            // SoftHier_TCDM -> SoftHier_HBM
                                            if(flex_is_dm_core())
                                            {
                                                flex_dma_async_2d(hbm_addr(C + ((((N * (((32 * ci) + (64 * gi)) + i)) + (32 * cj)) + (64 * gj)) + j)), local(accumulator), 32*2, N*2, 32*2, 32);
                                                flex_dma_async_wait_all();
                                            }
                                            flex_intra_cluster_sync();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    flex_global_barrier_xy();
                    // Finished deivelevel scope
                }
            }
        }
    }
}

#endif