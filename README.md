# TSQR on TensorCores

![TSQR](https://gitlab.momo86.net/mutsuki/tsqr-gpu/raw/master/docs/tsqr.svg)

## How to build shared library
```
git clone [this remote repository] --recursive
cd tsqr-gpu
make -f Makefile.library
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

