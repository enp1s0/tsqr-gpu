#include <cmath>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>
#include "validation.hpp"
#include "matrix_copy.cuh"
#include "utils.hpp"

template <>
float mtk::validation::check_orthogonality16<float>(
		const float* const matrix,
		const std::size_t m,
		const unsigned n
		) {
	auto d_qqt = cutf::memory::get_device_unique_ptr<float>(n * n);
	auto h_qqt = cutf::memory::get_host_unique_ptr<float>(n * n);
	for(std::size_t i = 0; i < n; i++) {
		for(std::size_t j = 0; j < n; j++) {
			h_qqt.get()[i + n * j] = (i == j) ? 1.0f : 0.0f;
		}
	}
	cutf::memory::copy(d_qqt.get(), h_qqt.get(), n * n);

	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	const auto alpha = 1.0f, beta = -1.0f;
	cutf::cublas::gemm(
			*cublas.get(),
			CUBLAS_OP_T, CUBLAS_OP_N,
			n, n, m,
			&alpha,
			matrix, m,
			matrix, m,
			&beta,
			d_qqt.get(), n
			);
	cutf::memory::copy(h_qqt.get(), d_qqt.get(), n * n);
	float sum = 0;
	for(std::size_t i = 0; i < n * n; i++) {
		const auto tmp = h_qqt.get()[i];
		sum += tmp * tmp;
	}
	return std::sqrt(sum / n);
}

__global__ void convert_f2h(float* const dst, const half* const src, const std::size_t size){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= size) return;

	dst[tid] = cutf::type::cast<float>(src[tid]);
}

template <>
float mtk::validation::check_orthogonality16<half>(
		const half* const matrix,
		const std::size_t m,
		const unsigned n
		) {
	constexpr std::size_t block_size = 256;
	auto d_q_f32 = cutf::memory::get_device_unique_ptr<float>(n * m);
	auto d_qqt = cutf::memory::get_device_unique_ptr<float>(n * n);
	auto h_qqt = cutf::memory::get_host_unique_ptr<float>(n * n);
	for(std::size_t i = 0; i < n; i++) {
		for(std::size_t j = 0; j < n; j++) {
			h_qqt.get()[i + n * j] = (i == j) ? 1.0f : 0.0f;
		}
	}
	cutf::memory::copy(d_qqt.get(), h_qqt.get(), n * n);
	convert_f2h<<<(m * n + block_size - 1) / block_size, block_size>>>(d_q_f32.get(), matrix, m * n);

	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	const auto alpha = 1.0f, beta = -1.0f;
	cutf::cublas::gemm(
			*cublas.get(),
			CUBLAS_OP_T, CUBLAS_OP_N,
			n, n, m,
			&alpha,
			d_q_f32.get(), m,
			d_q_f32.get(), m,
			&beta,
			d_qqt.get(), n
			);
	cutf::memory::copy(h_qqt.get(), d_qqt.get(), n * n);
	float sum = 0;
	for(std::size_t i = 0; i < n * n; i++) {
		const auto tmp = h_qqt.get()[i];
		sum += tmp * tmp;
	}
	return std::sqrt(sum / n);
}
