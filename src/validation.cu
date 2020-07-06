#include <cmath>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>
#include "validation.hpp"
#include "matrix_copy.cuh"
#include "utils.hpp"

template<class T>
__global__ void convert_2d(double* const dst, const T* const src, const std::size_t size){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= size) return;

	dst[tid] = cutf::type::cast<T>(src[tid]);
}

template <class T>
double mtk::validation::check_orthogonality16(
		const T* const matrix,
		const std::size_t m,
		const unsigned n
		) {
	constexpr std::size_t block_size = 256;
	auto d_q_f64 = cutf::memory::get_device_unique_ptr<double>(n * m);
	auto d_qqt = cutf::memory::get_device_unique_ptr<double>(n * n);
	auto h_qqt = cutf::memory::get_host_unique_ptr<double>(n * n);
	for(std::size_t i = 0; i < n; i++) {
		for(std::size_t j = 0; j < n; j++) {
			h_qqt.get()[i + n * j] = (i == j) ? 1.0f : 0.0f;
		}
	}
	cutf::memory::copy(d_qqt.get(), h_qqt.get(), n * n);
	convert_2d<<<(m * n + block_size - 1) / block_size, block_size>>>(d_q_f64.get(), matrix, m * n);

	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	const double alpha = 1.0f, beta = -1.0f;
	cutf::cublas::gemm(
			*cublas.get(),
			CUBLAS_OP_T, CUBLAS_OP_N,
			n, n, m,
			&alpha,
			d_q_f64.get(), m,
			d_q_f64.get(), m,
			&beta,
			d_qqt.get(), n
			);
	cutf::memory::copy(h_qqt.get(), d_qqt.get(), n * n);
	double sum = 0;
	for(std::size_t i = 0; i < n * n; i++) {
		const auto tmp = h_qqt.get()[i];
		sum += tmp * tmp;
	}
	return std::sqrt(sum / n);
}

template double mtk::validation::check_orthogonality16<double>(const double* const, const std::size_t, const unsigned);
template double mtk::validation::check_orthogonality16<float>(const float* const, const std::size_t, const unsigned);
template double mtk::validation::check_orthogonality16<half>(const half* const, const std::size_t, const unsigned);


template <class T>
void mtk::validation::check_submatrix_orthogonality(
		const T* const matrix,
		const std::size_t m,
		const unsigned n
		) {
	constexpr std::size_t block_size = 256;
	auto d_q_f64 = cutf::memory::get_device_unique_ptr<double>(n * m);
	auto d_qqt = cutf::memory::get_device_unique_ptr<double>(n * n);
	auto h_qqt = cutf::memory::get_host_unique_ptr<double>(n * n);
	for(std::size_t i = 0; i < n; i++) {
		for(std::size_t j = 0; j < n; j++) {
			h_qqt.get()[i + n * j] = (i == j) ? 1.0f : 0.0f;
		}
	}
	cutf::memory::copy(d_qqt.get(), h_qqt.get(), n * n);
	convert_2d<<<(m * n + block_size - 1) / block_size, block_size>>>(d_q_f64.get(), matrix, m * n);

	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	const double alpha = 1.0f, beta = -1.0f;
	cutf::cublas::gemm(
			*cublas.get(),
			CUBLAS_OP_T, CUBLAS_OP_N,
			n, n, m,
			&alpha,
			d_q_f64.get(), m,
			d_q_f64.get(), m,
			&beta,
			d_qqt.get(), n
			);
	cutf::memory::copy(h_qqt.get(), d_qqt.get(), n * n);

	const auto submatrix_n = 16;
	const auto submatrix_size = n / submatrix_n;
	auto submatrix_orthogonality_matrix = cutf::memory::get_host_unique_ptr<double>(submatrix_size * submatrix_size);
	for (unsigned si = 0; si < submatrix_size; si++) {
		for (unsigned sj = 0; sj < submatrix_size; sj++) {
			double orthogonality = 0;
			for (unsigned i = 0; i < submatrix_n; i++) {
				for (unsigned j = 0; j < submatrix_n; j++) {
					const auto v = h_qqt.get()[si * submatrix_n + i + (sj * submatrix_n + j) * n];
					orthogonality += v * v;
				}
			}
			submatrix_orthogonality_matrix.get()[si + sj * submatrix_size] = std::sqrt(orthogonality / submatrix_n);
		}
	}
	mtk::utils::print_matrix(submatrix_orthogonality_matrix.get(), submatrix_size, submatrix_size, submatrix_size, "sub ort matrix");
}

template void mtk::validation::check_submatrix_orthogonality<double>(const double* const, const std::size_t, const unsigned);
template void mtk::validation::check_submatrix_orthogonality<float>(const float* const, const std::size_t, const unsigned);
template void mtk::validation::check_submatrix_orthogonality<half>(const half* const, const std::size_t, const unsigned);

template <class T>
void mtk::validation::multi_orthogonality(const T* const ptr, const std::size_t ldm, const std::size_t m, const std::size_t n, const std::size_t size, cudaStream_t stream) {
	cudaStreamSynchronize(stream);
	auto h_mem = cutf::memory::get_host_unique_ptr<T>(m * n * size);
	cutf::memory::copy_async(h_mem.get(), ptr, m * n, stream);
	cudaStreamSynchronize(stream);
	double avg_orth = 0.0;
	for (std::size_t b = 0; b < size; b++) {
		double tmp = 0.0;
		for (unsigned i = 0; i < n; i++) {
			for (unsigned j = 0; j < n; j++) {
				double c = 0.0;
				for (unsigned k = 0; k < m; k++) {
					c += cutf::type::cast<double>(h_mem.get()[i * ldm + b * m + k]) * cutf::type::cast<double>(h_mem.get()[j * ldm + b * m + k]);
				}
				double t = (c - (i == j ? 1.0 : 0.0));
				tmp += t * t;
			}
		}
		tmp = std::sqrt(tmp / n);
		std::printf("%5lu : %e\n", b, tmp);
		avg_orth += tmp;
	}
	std::printf("avg : %e\n", avg_orth / size);
}

template void mtk::validation::multi_orthogonality<half >(const half * const ptr, const std::size_t ldm, const std::size_t m, const std::size_t n, const std::size_t size, cudaStream_t stream);
template void mtk::validation::multi_orthogonality<float>(const float* const ptr, const std::size_t ldm, const std::size_t m, const std::size_t n, const std::size_t size, cudaStream_t stream);
