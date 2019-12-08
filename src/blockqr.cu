#include <cutf/error.hpp>
#include <cutf/type.hpp>
#include "blockqr.hpp"

// #define PROFILE_BREAKDOWN

#ifdef PROFILE_BREAKDOWN
#include <chrono>

namespace  {
template <class T> std::string get_type_name();
template <> std::string get_type_name<float>() {return "float";}
template <> std::string get_type_name<half>() {return "half";}
} // namespace
#endif

constexpr std::size_t tsqr_colmun_size = 16;

std::size_t mtk::qr::get_working_q_size(const std::size_t m) {
	return mtk::tsqr::get_working_q_size(m, tsqr_colmun_size);
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
		cublasHandle_t const cublas_handle) {

	const auto column_block_size = (n + tsqr_colmun_size - 1) / tsqr_colmun_size;
	
	cudaStream_t cuda_stream;
	CUTF_HANDLE_ERROR(cublasGetStream(cublas_handle, &cuda_stream));

	cublasMath_t original_math_mode;
	CUTF_HANDLE_ERROR(cublasGetMathMode(cublas_handle, &original_math_mode));

	if (UseTC && !Refinement) {
		CUTF_HANDLE_ERROR(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
	} else {
		CUTF_HANDLE_ERROR(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
	}

#ifdef PROFILE_BREAKDOWN
	std::size_t gemm_0_count = 0lu;
	std::size_t gemm_1_count = 0lu;
	std::size_t tsqr_count = 0lu;
#endif

	// QR factorization of each block
	for (std::size_t b = 0; b < column_block_size; b++) {
		CUTF_HANDLE_ERROR(cudaStreamSynchronize(cuda_stream));

		const auto current_block_n = std::min(tsqr_colmun_size, n - b * tsqr_colmun_size);
		const auto previous_block_n = b * tsqr_colmun_size;
		const auto one = cutf::type::cast<T>(1.0f);
		const auto zero = cutf::type::cast<T>(0.0f);
		const auto minus_one = cutf::type::cast<T>(-1.0f);
#ifdef PROFILE_BREAKDOWN
		std::chrono::time_point<std::chrono::system_clock> t0, t1, t2, t3, t4;
#endif
		if (b != 0) {
#ifdef PROFILE_BREAKDOWN
			t0 = std::chrono::system_clock::now();
#endif
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						previous_block_n, tsqr_colmun_size, m,
						&one,
						q_ptr, ldq,
						a_ptr + lda * previous_block_n, lda,
						&zero,
						r_ptr + ldr * previous_block_n, ldr
						));
#ifdef PROFILE_BREAKDOWN
			CUTF_HANDLE_ERROR(cudaStreamSynchronize(cuda_stream));
			t1 = std::chrono::system_clock::now();
#endif
			// compute A'
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						m, current_block_n, previous_block_n,
						&minus_one,
						q_ptr, ldq,
						r_ptr + ldr * previous_block_n, ldr,
						&one,
						a_ptr + lda * previous_block_n, lda
						));
#ifdef PROFILE_BREAKDOWN
			CUTF_HANDLE_ERROR(cudaStreamSynchronize(cuda_stream));
			t2 = std::chrono::system_clock::now();
#endif
		}
		CUTF_HANDLE_ERROR(cudaStreamSynchronize(cuda_stream));

		//QR factorization of A'
#ifdef PROFILE_BREAKDOWN
		CUTF_HANDLE_ERROR(cudaStreamSynchronize(cuda_stream));
		t3 = std::chrono::system_clock::now();
#endif
		mtk::tsqr::tsqr16<UseTC, Refinement>(
				q_ptr + previous_block_n * ldq, ldq,
				r_ptr + previous_block_n * ldr + previous_block_n, ldr,
				a_ptr + previous_block_n * lda, lda,
				m, tsqr_colmun_size,
				wq_ptr,
				wr_ptr,
				cuda_stream
				);
		CUTF_HANDLE_ERROR(cudaStreamSynchronize(cuda_stream));
#ifdef PROFILE_BREAKDOWN
		t4 = std::chrono::system_clock::now();

		if (b != 0) {
			gemm_0_count += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
			gemm_1_count += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		}
		tsqr_count += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
#endif
	}
#ifdef PROFILE_BREAKDOWN
	const auto time_sum = gemm_0_count + gemm_1_count + tsqr_count;
	std::printf("# BlockQR breakdown\n");
	std::printf("Size   : %lu x %lu\n", m, n);
	std::printf("Type   : %s\n", get_type_name<T>().c_str());
	std::printf("UseTC  : %s\n", (UseTC ? "YES" : "NO"));
	std::printf("Refine : %s\n", (Refinement ? "YES" : "NO"));
	std::printf("GEMM-0 : %e[s] (%e%%)\n", gemm_0_count / 1.0e6, static_cast<double>(gemm_0_count) / time_sum * 100);
	std::printf("GEMM-1 : %e[s] (%e%%)\n", gemm_1_count / 1.0e6, static_cast<double>(gemm_1_count) / time_sum * 100);
	std::printf("TSQR   : %e[s] (%e%%)\n", tsqr_count / 1.0e6, static_cast<double>(tsqr_count) / time_sum * 100);
#endif
	CUTF_HANDLE_ERROR(cublasSetMathMode(cublas_handle, original_math_mode));
}

template void mtk::qr::qr<false, false, float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, false, false>::type* const, typename mtk::qr::get_working_r_type<float, false, false>::type* const, cublasHandle_t const);
template void mtk::qr::qr<true , false, float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, true , false>::type* const, typename mtk::qr::get_working_r_type<float, true , false>::type* const, cublasHandle_t const);
template void mtk::qr::qr<true , true , float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, true , true >::type* const, typename mtk::qr::get_working_r_type<float, true , true >::type* const, cublasHandle_t const);
template void mtk::qr::qr<false, false, half >(half * const, const std::size_t, half * const, const std::size_t, half * const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<half , false, false>::type* const, typename mtk::qr::get_working_r_type<half , false, false>::type* const, cublasHandle_t const);
template void mtk::qr::qr<true , false, half >(half * const, const std::size_t, half * const, const std::size_t, half * const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<half , true , false>::type* const, typename mtk::qr::get_working_r_type<half , true , false>::type* const, cublasHandle_t const);
