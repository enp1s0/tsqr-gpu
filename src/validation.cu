#include <cmath>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>
#include "validation.hpp"
#include "matrix_copy.cuh"
#include "utils.hpp"

template <class T>
float mtk::validation::check_orthogonality16(
		const T* const matrix,
		const std::size_t m,
		const unsigned n
		) {
	auto d_qqt = cutf::memory::get_device_unique_ptr<T>(n * n);
	auto h_qqt = cutf::memory::get_host_unique_ptr<T>(n * n);
	for(std::size_t i = 0; i < n; i++) {
		for(std::size_t j = 0; j < n; j++) {
			h_qqt.get()[i + n * j] = cutf::type::cast<T>((i == j) ? 1.0f : 0.0f);
		}
	}
	cutf::memory::copy(d_qqt.get(), h_qqt.get(), n * n);

	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	const auto alpha = cutf::type::cast<T>(1.0f), beta = cutf::type::cast<T>(-1.0f);
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
		const auto tmp = cutf::type::cast<float>(h_qqt.get()[i]);
		sum += tmp * tmp;
	}
	return std::sqrt(sum / n);
}


template float mtk::validation::check_orthogonality16<float>(const float* const matrix, const std::size_t m, const unsigned n);
template float mtk::validation::check_orthogonality16<half>(const half* const matrix, const std::size_t m, const unsigned n);
