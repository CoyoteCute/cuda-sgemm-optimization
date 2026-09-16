NVCC ?= nvcc
CXX  ?= g++
ARCH ?= sm_86
OPT  ?= -O2
INC  := -Isrc
# -MMD -MP: emit a .d per object listing the headers it pulled in, so editing
# gemm.h or harness.cuh rebuilds what actually included it.
DEP  := -MMD -MP

# Adding a kernel means dropping a .cu in src/kernels and naming it in
# src/kernels/kernels.h + src/registry.cu. The build picks it up on its own.
CU_SRC  := $(wildcard src/kernels/*.cu) src/registry.cu tests/test_gemm.cu
CPP_SRC := src/gemm_cpu.cpp
OBJ     := $(CU_SRC:%.cu=build/%.o) $(CPP_SRC:%.cpp=build/%.o)

.PHONY: all test smoke clean
all: build/test_gemm

test: build/test_gemm
	./build/test_gemm $(K)

build/test_gemm: $(OBJ)
	@mkdir -p $(dir $@)
	$(NVCC) -arch=$(ARCH) $^ -o $@

build/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) -arch=$(ARCH) $(OPT) $(INC) $(DEP) -c $< -o $@

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(OPT) $(INC) $(DEP) -c $< -o $@

smoke: tools/smoke_test.cu
	@mkdir -p build
	$(NVCC) -arch=$(ARCH) $< -o build/smoke

clean:
	rm -rf build/src build/tests build/test_gemm

-include $(OBJ:.o=.d)
