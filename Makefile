CXX=g++
COMMONFLAGS=-std=c++14 -I./src/cutf -I./src/wmma-extension -I./src/runtime-status -DRS_GIT_BRANCH="\"$(shell git branch | grep '\*' | sed -e 's/.* //')\"" -DRS_GIT_COMMIT="\"$(shell git rev-parse HEAD)\""
CXXFLAGS=-O3 -Wall -fopenmp
NVCC=nvcc
NVCCFLAGS=$(COMMONFLAGS) --compiler-bindir=$(CXX) -Xcompiler="$(CXXFLAGS)" -rdc=true --expt-relaxed-constexpr
NVCCFLAGS+=-lcublas -lcusolver -lcurand
NVCCFLAGS+=-gencode arch=compute_70,code=sm_70
NVCCFLAGS+=-gencode arch=compute_75,code=sm_75
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
SRCDIR=src
SRCS=$(shell find src -maxdepth 1 -name '*.cu' -o -name '*.cpp')
OBJDIR=objs
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
HEADERS=$(shell find $(SRCDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')
TARGET=tsqr-gpu.out

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
