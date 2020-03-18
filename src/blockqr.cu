#include <cutf/cublas.hpp>
#include <cutf/error.hpp>
#include <cutf/type.hpp>
#include "blockqr.hpp"

// #define PROFILE_BREAKDOWN
// #define PROFILE_BREAKDOWN_CSV

#ifdef PROFILE_BREAKDOWN
#include <chrono>

namespace  {
template <class T> std::string get_type_name();
template <> std::string get_type_name<float>() {return "float";}
template <> std::string get_type_name<half>() {return "half";}
} // namespace
#endif


std::size_t mtk::qr::get_working_q_size(const std::size_t m) {
	return mtk::tsqr::get_working_q_size(m, tsqr_colmun_size);
}
std::size_t mtk::qr::get_working_r_size(const std::size_t m) {
	return mtk::tsqr::get_working_r_size(m, tsqr_colmun_size);
}
std::size_t mtk::qr::get_working_l_size(const std::size_t m) {
	return mtk::tsqr::get_working_l_size(m);
}

namespace {
template <bool UseTC, bool Refinement, class T, class CORE_T>
mtk::qr::state_t block_qr_core(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<T, UseTC, Refinement>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<T, UseTC, Refinement>::type* const wr_ptr,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const cublas_handle) {
	const auto column_block_size = (n + mtk::qr::tsqr_colmun_size - 1) / mtk::qr::tsqr_colmun_size;
	
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

		const auto current_block_n = std::min(mtk::qr::tsqr_colmun_size, n - b * mtk::qr::tsqr_colmun_size);
		const auto previous_block_n = b * mtk::qr::tsqr_colmun_size;
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
						previous_block_n, current_block_n, m,
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
		mtk::tsqr::tsqr16<UseTC, Refinement, T, CORE_T>(
				q_ptr + previous_block_n * ldq, ldq,
				r_ptr + previous_block_n * ldr + previous_block_n, ldr,
				a_ptr + previous_block_n * lda, lda,
				m, current_block_n,
				wq_ptr,
				wr_ptr,
				d_wl_ptr,
				h_wl_ptr,
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
#ifdef PROFILE_BREAKDOWN_CSV
	std::printf("%lu,%lu,%s,%s,%d,%d,%e,%e,%e,%e\n",
			m, n,
			get_type_name<T>().c_str(),
			get_type_name<CORE_T>().c_str(),
			(UseTC ? 1 : 0),
			(Refinement ? 1 : 0),
			(gemm_0_count + gemm_1_count) / 1.0e6, static_cast<double>(gemm_0_count + gemm_1_count) / time_sum * 100,
			tsqr_count / 1.0e6, static_cast<double>(tsqr_count) / time_sum * 100
			);
#else
	std::printf("# BlockQR breakdown\n");
	std::printf("Size   : %lu x %lu\n", m, n);
	std::printf("Type   : %s\n", get_type_name<T>().c_str());
	std::printf("C Type : %s\n", get_type_name<CORE_T>().c_str());
	std::printf("UseTC  : %s\n", (UseTC ? "YES" : "NO"));
	std::printf("Refine : %s\n", (Refinement ? "YES" : "NO"));
	std::printf("GEMM-0 : %e[s] (%e%%)\n", gemm_0_count / 1.0e6, static_cast<double>(gemm_0_count) / time_sum * 100);
	std::printf("GEMM-1 : %e[s] (%e%%)\n", gemm_1_count / 1.0e6, static_cast<double>(gemm_1_count) / time_sum * 100);
	std::printf("TSQR   : %e[s] (%e%%)\n", tsqr_count / 1.0e6, static_cast<double>(tsqr_count) / time_sum * 100);
#endif
#endif
	CUTF_HANDLE_ERROR(cublasSetMathMode(cublas_handle, original_math_mode));

	return mtk::qr::success_factorization;
}

template <bool UseTC, bool Refinement, class T, class CORE_T>
mtk::qr::state_t block_qr_reorthogonalization_core(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<T, UseTC, Refinement>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<T, UseTC, Refinement>::type* const wr_ptr,
		T* const w_reorth,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const cublas_handle) {
	const auto column_block_size = (n + mtk::qr::tsqr_colmun_size - 1) / mtk::qr::tsqr_colmun_size;

	T* const a2_ptr = w_reorth;
	T* const r2_ptr = a2_ptr + m * mtk::qr::tsqr_colmun_size;
	T* const r12_2_ptr = r2_ptr + mtk::qr::tsqr_colmun_size * mtk::qr::tsqr_colmun_size;
	T* const r3_ptr = r12_2_ptr + mtk::qr::tsqr_colmun_size * mtk::qr::tsqr_colmun_size;

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

		const auto current_block_n = std::min(mtk::qr::tsqr_colmun_size, n - b * mtk::qr::tsqr_colmun_size);
		const auto previous_block_n = b * mtk::qr::tsqr_colmun_size;
		const auto one = cutf::type::cast<T>(1.0f);
		const auto zero = cutf::type::cast<T>(0.0f);
		const auto minus_one = cutf::type::cast<T>(-1.0f);

		if (b != 0) {
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						previous_block_n, current_block_n, m,
						&one,
						q_ptr, ldq,
						a_ptr + lda * previous_block_n, lda,
						&zero,
						r_ptr + ldr * previous_block_n, ldr
						));
			// compute A'
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						m, current_block_n, previous_block_n,
						&minus_one,
						q_ptr, ldq,
						r_ptr + ldr * previous_block_n, ldr,
						&one,
						a2_ptr, lda
						));
			mtk::tsqr::tsqr16<UseTC, Refinement, T, CORE_T>(
					q_ptr + previous_block_n * ldq, ldq,
					r2_ptr, current_block_n,
					a2_ptr, lda,
					m, current_block_n,
					wq_ptr,
					wr_ptr,
					d_wl_ptr,
					h_wl_ptr,
					cuda_stream
					);
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						current_block_n, current_block_n, m,
						&one,
						q_ptr + previous_block_n * ldq, ldq,
						a2_ptr, lda,
						&zero,
						r12_2_ptr, current_block_n
						));
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						m, current_block_n, previous_block_n,
						&minus_one,
						q_ptr + previous_block_n * ldq, ldq,
						r12_2_ptr, current_block_n,
						&one,
						a2_ptr, lda
						));
			mtk::tsqr::tsqr16<UseTC, Refinement, T, CORE_T>(
					q_ptr + previous_block_n * ldq, ldq,
					r3_ptr, current_block_n,
					a2_ptr, lda,
					m, current_block_n,
					wq_ptr,
					wr_ptr,
					d_wl_ptr,
					h_wl_ptr,
					cuda_stream
					);
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						previous_block_n, current_block_n, current_block_n,
						&one,
						r12_2_ptr, current_block_n,
						r2_ptr, current_block_n,
						&one,
						r_ptr + ldr * previous_block_n, ldr
						));
			CUTF_HANDLE_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						current_block_n, current_block_n, current_block_n,
						&one,
						r3_ptr, current_block_n,
						r2_ptr, current_block_n,
						&zero,
						r_ptr + previous_block_n * ldr + previous_block_n, ldr
						));
		} else {
			mtk::tsqr::tsqr16<UseTC, Refinement, T, CORE_T>(
					q_ptr + previous_block_n * ldq, ldq,
					r_ptr + previous_block_n * ldr + previous_block_n, ldr,
					a_ptr + previous_block_n * lda, lda,
					m, current_block_n,
					wq_ptr,
					wr_ptr,
					d_wl_ptr,
					h_wl_ptr,
					cuda_stream
					);
		}
		CUTF_HANDLE_ERROR(cudaStreamSynchronize(cuda_stream));
	}
	CUTF_HANDLE_ERROR(cublasSetMathMode(cublas_handle, original_math_mode));

	return mtk::qr::success_factorization;
}

} // namespace

template <bool UseTC, bool Refinement, bool Reorthoganalize, class T, class CORE_T>
mtk::qr::state_t mtk::qr::qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<T, UseTC, Refinement>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<T, UseTC, Refinement>::type* const wr_ptr,
		T* const reorth_r,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const cublas_handle) {

	if (n > m || m == 0 || n == 0) {
		return mtk::qr::error_invalid_matrix_size;
	}

	if (Reorthoganalize) {
		return block_qr_reorthogonalization_core<UseTC, Refinement, T, CORE_T>(
				q_ptr, ldq,
				r_ptr, ldr,
				a_ptr, lda,
				m, n,
				wq_ptr, wr_ptr,
				reorth_r,
				d_wl_ptr, h_wl_ptr,
				cublas_handle
				);
	} else {
		return block_qr_core<UseTC, Refinement, T, CORE_T>(
				q_ptr, ldq,
				r_ptr, ldr,
				a_ptr, lda,
				m, n,
				wq_ptr, wr_ptr,
				d_wl_ptr, h_wl_ptr,
				cublas_handle
				);
	}
}

template mtk::qr::state_t mtk::qr::qr<false, false, false, float, float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, false, false>::type* const, typename mtk::qr::get_working_r_type<float, false, false>::type* const, float* const, unsigned* const, unsigned* const, cublasHandle_t const);
template mtk::qr::state_t mtk::qr::qr<true , false, false, float, float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, true , false>::type* const, typename mtk::qr::get_working_r_type<float, true , false>::type* const, float* const, unsigned* const, unsigned* const, cublasHandle_t const);
template mtk::qr::state_t mtk::qr::qr<true , true , false, float, float>(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, true , true >::type* const, typename mtk::qr::get_working_r_type<float, true , true >::type* const, float* const, unsigned* const, unsigned* const, cublasHandle_t const);
template mtk::qr::state_t mtk::qr::qr<false, false, false, half , half >(half * const, const std::size_t, half * const, const std::size_t, half * const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<half , false, false>::type* const, typename mtk::qr::get_working_r_type<half , false, false>::type* const, half*  const, unsigned* const, unsigned* const, cublasHandle_t const);
template mtk::qr::state_t mtk::qr::qr<true , false, false, half , half >(half * const, const std::size_t, half * const, const std::size_t, half * const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<half , true , false>::type* const, typename mtk::qr::get_working_r_type<half , true , false>::type* const, half*  const, unsigned* const, unsigned* const, cublasHandle_t const);
template mtk::qr::state_t mtk::qr::qr<true , false, false, float, half >(float* const, const std::size_t, float* const, const std::size_t, float* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<float, true , false>::type* const, typename mtk::qr::get_working_r_type<float, true , false>::type* const, float* const, unsigned* const, unsigned* const, cublasHandle_t const);
