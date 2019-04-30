CXX=g++
CXXFLAGS=-std=c++14 -I../src/cutf -I../src
OMPFLAGS=-fopenmp
NVCC=nvcc
NVCCFLAGS=$(CXXFLAGS) --compiler-bindir=$(CXX) -Xcompiler=$(OMPFLAGS) -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70
SRCDIR=.
SRCS=$(shell find $(SRCDIR) -maxdepth 1 -name '*.cu' -o -name '*.cpp')
OBJDIR=objs
HEADERS=$(shell find $(SRCDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')

test: matrix_copy_16x16.test matrix_copy_32x16.test

%.test: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.test: $(SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf *.test

