# TSQR on TensorCores

![TSQR](https://gitlab.momo86.net/mutsuki/tsqr-gpu/raw/master/docs/tsqr.svg)

## How to build test
```
git clone [this remote repository] --recursive
cd tsqr-gpu
make
```

## How to build shared library
```
git clone [this remote repository] --recursive
cd tsqr-gpu
make -f Makefile.library
```

You can find `libtcqr.a` in `lib` directory.

## Sample
### TSQR
```cpp
#include <tsqr.hpp>

using comute_t = float;
constexpr bool UseTC = true;
constexpr bool Refine = true;
constexpr bool Reorthogonalization = true;

// size of input matrix
constexpr std::size_t M = 9211;
constexpr std::size_t N = 16;

// allocate input matrix
compute_t *d_a;
cudaMalloc((void**)&d_a, sizeof(compute_t) * N * N);

// allocate output matrices
float *d_r, *d_q;
cudaMalloc((void**)&d_r, sizeof(compute_t) * N * N);
cudaMalloc((void**)&d_q, sizeof(compute_t) * M * N);

// allocate working memory
mtk::tsqr::buffer buffer;
buffer.allocate(M, N);

// TSQR
mtk::tsqr::tsqr16<UseTC, Refine, Reorthogonalization>(
	d_q, M,
	d_r, N,
	d_a, M,
	M, N,
	buffer
	);
```

### BlockQR
```cpp
#include <blockqr.hpp>

using comute_t = float;
constexpr bool UseTC = true;
constexpr bool Refine = true;
constexpr bool Reorthogonalization = true;

// size of input matrix
constexpr std::size_t M = 9211;
constexpr std::size_t N = 51;

// allocate input matrix
compute_t *d_a;
cudaMalloc((void**)&d_a, sizeof(compute_t) * N * N);

// allocate output matrices
float *d_r, *d_q;
cudaMalloc((void**)&d_r, sizeof(compute_t) * N * N);
cudaMalloc((void**)&d_q, sizeof(compute_t) * M * N);

// allocate working memory
mtk::tsqr::buffer buffer;
buffer.allocate(M, N);

// cuBLAS
cublasHandle_t cublas_handle;
cublasCreateHandle(cublas_handle);

// BlockQR
mtk::qr::qr<UseTC, Refine, Reorthogonalization>(
	d_q, M,
	d_r, N,
	d_a, M,
	M, N,
	buffer,
	cublas_handle
	);
```

### Build
```
nvcc -std=c++11 -arch=sm_70 tsqr-sample.cu -I/path/to/`include` -L/path/to/`lib` -ltcqr
```


## Environment
### Software
- C++ (C++11 or later)
- CUDA (9.2 or later)

### Hardware
- NVIDIA GPU

## Dependencies
- cutf : [https://github.com/enp1s0/cutf](https://github.com/enp1s0/cutf)
- gemm_core : [https://gitlab.momo86.net/mutsuki/gemm_core](https://gitlab.momo86.net/mutsuki/gemm_core)
- wmma-extension : [https://gitlab.momo86.net/mutsuki/wmma-extension](https://gitlab.momo86.net/mutsuki/wmma-extension)
