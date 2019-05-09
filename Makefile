CXX=g++
CXXFLAGS=-std=c++14 -I./src/cutf -O3
OMPFLAGS=-fopenmp
NVCC=nvcc
NVCCFLAGS=$(CXXFLAGS)  --compiler-bindir=$(CXX) -Xcompiler=$(OMPFLAGS) -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -lcublas --ptxas-options=-v -rdc=true
SRCDIR=src
SRCS=$(shell find src -maxdepth 1 -name '*.cu' -o -name '*.cpp')
OBJDIR=objs
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
HEADERS=$(shell find $(SRCDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')
TARGET=tsqr-gpu.out
SL_OBJS=$(OBJDIR)/tsqr.o $(OBJDIR)/tcqr32x16.o

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

library: $(SL_OBJS)
	[ -d lib ] || mkdir lib
	$(NVCC) $(NVCCFLAGS) $+ -dlink -lcudart -shared -o lib/libtsqr.so

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -c -o $@


clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)

