CXX=g++
CXXFLAGS=-std=c++11 -I../
# CUDA
NVCC=nvcc
# HIP
HIPPATH=/opt/rocm/hip
HIPCC=$(HIPPATH)/bin/hipcc
HIP_INC=$(HIPPATH)/include
HIP_LIB=$(HIPPATH)/lib
HIPFLAGS=-I$(HIP_INC) -L$(HIP_LIB) $(CXXFLAGS)

git_status.test : git_status.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) -DRS_GIT_BRANCH="\"$(shell git branch | grep '\*' | sed -e 's/.* //')\"" -DRS_GIT_COMMIT="\"$(shell git rev-parse HEAD)\""

process_status.test : process_status.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)

cuda_status.test : cuda_status.cpp
	$(NVCC) $< -o $@ $(CXXFLAGS)

hip_status.test : hip_status.cpp
	$(HIPCC) $< -o $@ $(HIPFLAGS)

env_status.test : env_status.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)

date_status.test : date_status.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) -DRS_DATE_BUILD="\"$(shell date +%s)\""

clean:
	rm -f *.test
