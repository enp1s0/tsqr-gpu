CXX=g++
CXXFLAGS=-std=c++14 -I../src/cutf -I../src
OMPFLAGS=-fopenmp
NVCC=nvcc
NVCCFLAGS=$(CXXFLAGS) --compiler-bindir=$(CXX) -Xcompiler=$(OMPFLAGS) -gencode arch=compute_75,code=sm_75
SRCDIR=.
SRCS=$(shell find $(SRCDIR) -maxdepth 1 -name '*.cu' -o -name '*.cpp')
OBJDIR=objs
HEADERS=$(shell find $(SRCDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')

test: matrix_copy_16x16.test matrix_copy_32x16.test calc_batchsize.test matrix_copy_32x16_t.test matrix_copy_1w.test

%.test: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.test: $(SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

library_link.test: library_link.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -I../src ../lib/libtsqr.a -lcublas

clean:
	rm -rf *.test

