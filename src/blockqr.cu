#include <cutf/error.hpp>
#include <cutf/type.hpp>
#include "blockqr.hpp"

constexpr std::size_t tsqr_colmun_size = 16;

std::size_t mtk::qr::get_working_q_size(const std::size_t m) {
	return mtk::tsqr::get_working_r_size(m, tsqr_colmun_size);
}
std::size_t mtk::qr::get_working_r_size(const std::size_t m) {
	return mtk::tsqr::get_working_r_size(m, tsqr_colmun_size);
}

template <bool UseTC, bool Refinement, class T>
void mtk::qr::qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<T, UseTC, Refinement>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<T, UseTC, Refinement>::type* const wr_ptr,
		cublasHandle_t const main_cublas_handle, cublasHandle_t const sub_cublas_handle) {

	const auto column_block_size = (n + tsqr_colmun_size - 1) / tsqr_colmun_size;
	
	cudaStream_t main_cuda_stream;
	CUTF_HANDLE_ERROR(cublasGetStream(main_cublas_handle, &main_cuda_stream));
	cudaStream_t sub_cuda_stream;
	CUTF_HANDLE_ERROR(cublasGetStream(sub_cublas_handle, &sub_cuda_stream));

	// QR factorization of each block
	for (std::size_t b = 0; b < column_block_size; b++) {
		CUTF_HANDLE_ERROR(cudaStreamSynchronize(main_cuda_stream));
		CUTF_HANDLE_ERROR(cudaStreamSynchronize(sub_cuda_stream));

		const auto current_block_n = std::min(tsqr_colmun_size, n - b * tsqr_colmun_size);
		const auto previous_block_n = b * tsqr_colmun_size;
		const auto one = cutf::type::cast<T>(1.0f);
		const auto zero = cutf::type::cast<T>(0.0f);
		const auto minus_one = cutf::type::cast<T>(-1.0f);
		if (b != 0) {
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						main_cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						previous_block_n, tsqr_colmun_size, m,
						&one,
						q_ptr, ldq,
						a_ptr + lda * previous_block_n, lda,
						&zero,
						r_ptr + ldr * previous_block_n, ldr
						));
			// compute A'
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						main_cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						m, current_block_n, previous_block_n,
						&minus_one,
						q_ptr, ldq,
						r_ptr + ldr * previous_block_n, ldr,
						&one,
						a_ptr + lda * previous_block_n, lda
						));
		}
		CUTF_HANDLE_ERROR(cudaStreamSynchronize(main_cuda_stream));
		//QR factorization of A'
		mtk::tsqr::tsqr16<UseTC, Refinement>(
				q_ptr + previous_block_n * ldq, ldq,
				r_ptr + previous_block_n * ldr + previous_block_n, ldr,
				a_ptr + previous_block_n * lda, lda,
				m, tsqr_colmun_size,
				wq_ptr,
				wr_ptr
				);
	}
}

template void mtk::qr::qr<false, false, float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, false, false>::type* const, typename mtk::qr::get_working_r_type<float, false, false>::type* const, cublasHandle_t const, cublasHandle_t const);
template void mtk::qr::qr<true , false, float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, true , false>::type* const, typename mtk::qr::get_working_r_type<float, true , false>::type* const, cublasHandle_t const, cublasHandle_t const);
template void mtk::qr::qr<true , true , float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, true , true >::type* const, typename mtk::qr::get_working_r_type<float, true , true >::type* const, cublasHandle_t const, cublasHandle_t const);
template void mtk::qr::qr<false, false, half >(half * const, const std::size_t, half * const, const std::size_t, half * const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<half , false, false>::type* const, typename mtk::qr::get_working_r_type<half , false, false>::type* const, cublasHandle_t const, cublasHandle_t const);
template void mtk::qr::qr<true , false, half >(half * const, const std::size_t, half * const, const std::size_t, half * const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<half , true , false>::type* const, typename mtk::qr::get_working_r_type<half , true , false>::type* const, cublasHandle_t const, cublasHandle_t const);
