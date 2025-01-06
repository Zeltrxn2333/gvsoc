bash -c 'cd /usr/scratch/badile111/dace4softhier/gvsoc && \
        source sourceme.sh && \   
        ./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary ./sw_build/softhier.elf run --preload ./output.elf'