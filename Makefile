CMAKE_FLAGS ?= -j 6
CMAKE ?= cmake

TARGETS ?= rv32;rv64

export PATH:=$(CURDIR)/gapy/bin:$(PATH)
export PATH:=$(CURDIR)/third_party:$(PATH)

all: checkout build

checkout:
	git submodule update --recursive --init

.PHONY: build

build:
	# Change directory to curdir to avoid issue with symbolic links
	cd $(CURDIR) && $(CMAKE) -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
		-DCMAKE_INSTALL_PREFIX=install \
		-DGVSOC_MODULES="$(CURDIR)/core/models;$(CURDIR)/pulp;$(MODULES)" \
		-DGVSOC_TARGETS="${TARGETS}" \
		-DCMAKE_SKIP_INSTALL_RPATH=false

	cd $(CURDIR) && $(CMAKE) --build build $(CMAKE_FLAGS)
	cd $(CURDIR) && $(CMAKE) --install build


clean:
	rm -rf build install

######################################################################
## 				Make Targets for DRAMSys Integration 				##
######################################################################

SYSTEMC_VERSION := 2.3.3
SYSTEMC_GIT_URL := https://github.com/accellera-official/systemc.git
SYSTEMC_INSTALL_DIR := $(PWD)/third_party/systemc_install

update:
	cd pulp && git diff > ../soft_hier/gvsoc_pulp.patch
	cd core && git add models/cpu && git diff --cached > ../soft_hier/gvsoc_core.patch

drmasys_apply_patch:
	git submodule update --init --recursive
	if cd core && git apply --check ../soft_hier/gvsoc_core.patch; then \
		git apply ../soft_hier/gvsoc_core.patch;\
	fi
	if cd pulp && git apply --check ../soft_hier/gvsoc_pulp.patch; then \
		git apply ../soft_hier/gvsoc_pulp.patch;\
	fi
	cp -rf soft_hier/flex_cluster pulp/pulp/chips/flex_cluster


build-systemc: third_party/systemc_install/lib64/libsystemc.so

third_party/systemc_install/lib64/libsystemc.so:
	mkdir -p $(SYSTEMC_INSTALL_DIR)
	cd third_party && \
	git clone $(SYSTEMC_GIT_URL) && \
	cd systemc && git fetch --tags && git checkout $(SYSTEMC_VERSION) && \
	mkdir build && cd build && \
	$(CMAKE) -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=$(SYSTEMC_INSTALL_DIR) -DCMAKE_INSTALL_LIBDIR=lib64 .. && \
	make && make install

build-dramsys: build-systemc third_party/DRAMSys/libDRAMSys_Simulator.so

third_party/DRAMSys/libDRAMSys_Simulator.so:
	mkdir -p third_party/DRAMSys
	cp soft_hier/libDRAMSys_Simulator.so third_party/DRAMSys/
	echo "Check Library Functionality"
	cd soft_hier/build_dynlib_from_github_dramsys5/dynamic_load/ && \
	gcc main.c -ldl
	@if soft_hier/build_dynlib_from_github_dramsys5/dynamic_load/a.out ; then \
        echo "Test libaray succeeded"; \
		rm soft_hier/build_dynlib_from_github_dramsys5/dynamic_load/a.out; \
    else \
		rm soft_hier/build_dynlib_from_github_dramsys5/dynamic_load/a.out; \
		rm third_party/DRAMSys/libDRAMSys_Simulator.so; \
		echo "Test libaray failed, We need to rebuild the library, tasks around 40 min"; \
		echo -n "Do you want to proceed? (y/n) "; \
		read -t 30 -r user_input; \
		if [ "$$user_input" = "n" ]; then echo "oops, I see, your time is precious, see you next time"; exit 1; fi; \
		echo "Go Go Go!" ; \
		cd soft_hier/build_dynlib_from_github_dramsys5 && make all; \
		cp DRAMSys/build/lib/libDRAMSys_Simulator.so ../../third_party/DRAMSys/ ; \
		make clean; \
    fi

build-configs: core/models/memory/dramsys_configs

core/models/memory/dramsys_configs:
	cp -rf soft_hier/dramsys_configs core/models/memory/

build-toolchain: third_party/toolchain

third_party/toolchain:
	mkdir -p third_party/toolchain
	cd third_party/toolchain && \
	wget https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-centos-7.tar.bz2 &&\
	tar -xvjf v1.0.16-pulp-riscv-gcc-centos-7.tar.bz2

softhier_preparation: drmasys_apply_patch build-systemc build-dramsys build-configs build-toolchain

clean_preparation:
	rm -rf third_party

######################################################################
## 				Make Targets for SoftHier Simulator 				##
######################################################################

config_file ?= "soft_hier/flex_cluster/flex_cluster_arch.py"
ifdef cfg
	config_file = "$(cfg)"
endif

config:
	rm -rf pulp/pulp/chips/flex_cluster
	cp -rf soft_hier/flex_cluster pulp/pulp/chips/flex_cluster
	cp $(config_file) pulp/pulp/chips/flex_cluster/flex_cluster_arch.py
	python3 soft_hier/flex_cluster_utilities/config.py $(config_file)

hw:
	make config
	make TARGETS=pulp.chips.flex_cluster.flex_cluster all


######################################################################
## 				Make Targets for SoftHier Software	 				##
######################################################################

sw_cmake_arg ?= ""
ifdef app
	app_path = $(abspath $(app))
	sw_cmake_arg = "-DSRC_DIR=$(app_path)"
endif

sw:
	rm -rf sw_build && mkdir sw_build
	cd sw_build && $(CMAKE) $(sw_cmake_arg) ../soft_hier/flex_cluster_sdk/ && make

clean_sw:
	rm -rf sw_build

######################################################################
## 				Make Targets for SoftHier HW + SW	 				##
######################################################################

hs:
	make config
	make TARGETS=pulp.chips.flex_cluster.flex_cluster all
	make sw

######################################################################
## 				Make Targets for Run Simulator		 				##
######################################################################

preload_arg ?= ""
ifdef pld
	pld_path = $(abspath $(pld))
	preload_arg = --preload $(pld_path)
endif
run:
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary sw_build/softhier.elf run $(preload_arg) --trace=/chip/cluster_0/redmule

runv:
	./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary sw_build/softhier.elf run $(preload_arg) --trace=redmule --trace=idma --trace=cluster_registers | tee sw_build/analyze_trace.txt

######################################################################
## 				Make Targets for Trace Analyzer		 				##
######################################################################

trace_file ?= sw_build/analyze_trace.txt
ifdef trace
	trace_file = $(trace)
endif
pfto:
	python soft_hier/flex_cluster_utilities/trace_perfetto/parse.py $(trace_file) sw_build/roi.json
	python soft_hier/flex_cluster_utilities/trace_perfetto/visualize.py sw_build/roi.json -o sw_build/perfetto.json
