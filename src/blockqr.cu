#include <cutf/error.hpp>
#include <cutf/type.hpp>
#include "blockqr.hpp"

constexpr std::size_t tsqr_colmun_size = 16;

std::size_t get_working_memory_size(const std::size_t n) {
	return tsqr_colmun_size * n;
}

template <class T, bool UseTC, bool Refinement>
void mtk::qr::qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		const T* a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::tsqr::get_working_q_type<T, UseTC, Refinement>::type* const wq_ptr,
		typename mtk::tsqr::get_working_r_type<T, UseTC, Refinement>::type* const wr_ptr,
		typename mtk::qr::get_working_memory_type<T, UseTC, Refinement>::type* const wm_ptr,
		cublasHandle_t const main_cublas_handle, cublasHandle_t const sub_cublas_handle) {

	const auto column_block_size = (n + tsqr_colmun_size - 1) / tsqr_colmun_size;
	
	cudaStream_t main_cuda_stream;
	CUTF_HANDLE_ERROR(cublasGetStream(main_cublas_handle, &main_cuda_stream));

	// QR factorization of each block
	for (std::size_t b = 0; b < column_block_size; b++) {
		const auto current_block_n = std::min(tsqr_colmun_size, n - b * tsqr_colmun_size);
		const auto previous_block_n = b * tsqr_colmun_size;
		// compute R12
		const auto one = cutf::type::cast<T>(1.0f);
		const auto zero = cutf::type::cast<T>(0.0f);
		const auto minus_two = cutf::type::cast<T>(-2.0f);
		CUTF_HANDLE_ERROR(cutf::cublas::gemm(
					sub_cublas_handle,
					previous_block_n, current_block_n, m,
					&one,
					q_ptr, ldq,
					a_ptr + lda * previous_block_n,
					&zero,
					r_ptr, ldr
					));
		
		//QR factorization of A'
		mtk::tsqr::tsqr16<UseTC, Refinement>(
				q_ptr + previous_block_n * ldq, ldq,
				r_ptr + previous_block_n * ldr + previous_block_n, ldr,
				a_ptr + previous_block_n * ldq, lda,
				m, tsqr_colmun_size,
				wq_ptr,
				wr_ptr
				);
		CUTF_HANDLE_ERROR(cutf::cublas::gemm(
					main_cublas_handle,
					CUBLAS_OP_T, CUBLAS_OP_N,
					previous_block_n, tsqr_colmun_size, m,
					&one,
					q_ptr, ldq,
					q_ptr + ldq * previous_block_n, ldq,
					&zero,
					wm_ptr, previous_block_n
					));
		CUTF_HANDLE_ERROR(cutf::cublas::gemm(
					main_cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					m, tsqr_colmun_size, previous_block_n,
					&minus_two,
					q_ptr, ldq,
					wm_ptr, previous_block_n,
					&one,
					q_ptr + ldq * previous_block_n, ldq
					));
	}
}
