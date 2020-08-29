#include <cutf/experimental/exponent.hpp>
#include "experimental.hpp"

namespace {
template <class T>
__global__ void force_experimental_kernel(T* const ptr, const int min_exponent, const std::size_t size) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= size) return;

	ptr[tid] = cutf::experimental::exponent::min_exponent<T>(ptr[tid], min_exponent);
}
} // noname namespace

template <class T>
void mtk::experimental::force_exponent(T* const ptr, const int min_exponent, const std::size_t size, cudaStream_t const cuda_stream) {
	constexpr std::size_t block_size = 1lu << 8;
	
	force_experimental_kernel<T><<<(size + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(ptr, min_exponent, size);
}

template void mtk::experimental::force_exponent<half >(half * const, const int, const std::size_t, cudaStream_t const);
template void mtk::experimental::force_exponent<float>(float* const, const int, const std::size_t, cudaStream_t const);
