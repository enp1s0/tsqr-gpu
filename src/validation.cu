#include <cmath>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include "validation.hpp"
#include "matrix_copy.cuh"
#include "utils.hpp"

float mtk::validation::check_orthogonality16(
		const float* const matrix,
		const std::size_t m,
		const unsigned n
		){
	auto d_qqt = cutf::memory::get_device_unique_ptr<float>(n * n);
	auto h_qqt = cutf::memory::get_host_unique_ptr<float>(n * n);
	for(std::size_t i = 0; i < n; i++){
		for(std::size_t j = 0; j < n; j++){
			h_qqt.get()[i + n * j] = (i == j) ? 1.0f : 0.0f;
		}
	}
	cutf::memory::copy(d_qqt.get(), h_qqt.get(), n * n);

	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	float alpha = 1.0f, beta = -1.0f;
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
	for(std::size_t i = 0; i < n * n; i++){
		const auto tmp = h_qqt.get()[i];
		sum += tmp * tmp;
	}
	return std::sqrt(sum / n);
}
