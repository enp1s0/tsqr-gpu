# TSQR on TensorCores

![TSQR](https://gitlab.momo86.net/mutsuki/tsqr-gpu/raw/master/docs/tsqr.svg)

## How to build shared library
```
git clone [this remote repository] --recursive
cd tsqr-gpu
make -f Makefile.library
```

You can find `libtsqr.a` in `lib` directory.

## Sample
```cpp
#include <tsqr.hpp>

constexpr bool UseTC = true;

// size of input matrix
constexpr std::size_t M = 9211;
constexpr std::size_t N = 16;

// allocate input matrix
float *d_a;

// allocate output matrices
float *d_r, *d_q;
cudaMalloc((void**)&d_r, sizeof(float) * N * N);
cudaMalloc((void**)&d_q, sizeof(float) * M * N);

// allocate working memory
typename mtk::tsqr::get_working_q_type<T, UseTC>::type *d_wq;
typename mtk::tsqr::get_working_r_type<T, UseTC>::type *d_wr;
cudaMalloc((void**)&d_wr, sizeof(typename mtk::tsqr::get_working_q_type<T, UseTC>::type) * mtk::tsqr::get_working_q_size(M, N));
cudaMalloc((void**)&d_wq, sizeof(typename mtk::tsqr::get_working_r_type<T, UseTC>::type) * mtk::tsqr::get_working_q_size(M, N));

// TSQR
mtk::tsqr::tsqr16<UseTC, float>(
	d_q, d_r,
	d_a, M, N,
	d_wq,
	d_wr
	);
```

### Build
```
nvcc -std=c++11 -arch=sm_70 tsqr-sample.cu /path/to/libtsqr.a -I/path/to/[tsqr-gpu/src/tsqr.hpp]
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

