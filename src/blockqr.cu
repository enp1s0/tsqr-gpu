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


std::size_t mtk::qr::get_working_q_size(const std::size_t m, const std::size_t n) {
	return mtk::tsqr::get_working_q_size(m, std::min(tsqr_colmun_size, n));
}
std::size_t mtk::qr::get_working_r_size(const std::size_t m, const std::size_t n) {
	return mtk::tsqr::get_working_r_size(m, std::min(tsqr_colmun_size, n));
}
std::size_t mtk::qr::get_working_l_size(const std::size_t m) {
	return mtk::tsqr::get_working_l_size(m);
}

namespace {
template <mtk::qr::compute_mode mode, class T>
mtk::qr::state_t block_qr_core(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<mode>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<mode>::type* const wr_ptr,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const cublas_handle) {
	const auto column_block_size = (n + mtk::qr::tsqr_colmun_size - 1) / mtk::qr::tsqr_colmun_size;
	
	cudaStream_t cuda_stream;
	CUTF_CHECK_ERROR(cublasGetStream(cublas_handle, &cuda_stream));

	cublasMath_t original_math_mode;
	CUTF_CHECK_ERROR(cublasGetMathMode(cublas_handle, &original_math_mode));

	if (mode == mtk::qr::fp16_tc_nocor || mode == mtk::qr::fp32_tc_nocor) {
		CUTF_CHECK_ERROR(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
	} else {
		CUTF_CHECK_ERROR(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
	}

#ifdef PROFILE_BREAKDOWN
	std::size_t gemm_0_count = 0lu;
	std::size_t gemm_1_count = 0lu;
	std::size_t tsqr_count = 0lu;
#endif

	// QR factorization of each block
	for (std::size_t b = 0; b < column_block_size; b++) {
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

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
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
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
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			t1 = std::chrono::system_clock::now();
#endif
			// compute A'
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
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
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			t2 = std::chrono::system_clock::now();
#endif
		}
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

		//QR factorization of A'
#ifdef PROFILE_BREAKDOWN
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
		t3 = std::chrono::system_clock::now();
#endif
		mtk::tsqr::tsqr16<mtk::qr::get_tsqr_compute_mode<mode>()>(
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
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
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
	std::printf("%lu,%lu,%s,%s,%d,%d,%d,%e,%e,%e,%e\n",
			m, n,
			get_type_name<T>().c_str(),
			get_type_name<CORE_T>().c_str(),
			(UseTC ? 1 : 0),
			(Correction ? 1 : 0),
			0,
			(gemm_0_count + gemm_1_count) / 1.0e6, static_cast<double>(gemm_0_count + gemm_1_count) / time_sum * 100,
			tsqr_count / 1.0e6, static_cast<double>(tsqr_count) / time_sum * 100
			);
#else
	std::printf("# BlockQR breakdown\n");
	std::printf("Size   : %lu x %lu\n", m, n);
	std::printf("Type   : %s\n", get_type_name<T>().c_str());
	std::printf("UseTC  : %s\n", (UseTC ? "YES" : "NO"));
	std::printf("Correction : %s\n", (Correction ? "YES" : "NO"));
	std::printf("Reorth : %s\n", "NO");
	std::printf("GEMM-0 : %e[s] (%e%%)\n", gemm_0_count / 1.0e6, static_cast<double>(gemm_0_count) / time_sum * 100);
	std::printf("GEMM-1 : %e[s] (%e%%)\n", gemm_1_count / 1.0e6, static_cast<double>(gemm_1_count) / time_sum * 100);
	std::printf("TSQR   : %e[s] (%e%%)\n", tsqr_count / 1.0e6, static_cast<double>(tsqr_count) / time_sum * 100);
#endif
#endif
	CUTF_CHECK_ERROR(cublasSetMathMode(cublas_handle, original_math_mode));

	return mtk::qr::success_factorization;
}

template <mtk::qr::compute_mode mode, class T>
mtk::qr::state_t block_qr_reorthogonalization_core(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<mode>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<mode>::type* const wr_ptr,
		T* const w_reorth,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const cublas_handle) {
	const auto column_block_size = (n + mtk::qr::tsqr_colmun_size - 1) / mtk::qr::tsqr_colmun_size;

	T* const r2_ptr = w_reorth;
	T* const s2_ptr = r2_ptr + mtk::qr::tsqr_colmun_size * mtk::qr::tsqr_colmun_size;
	T* const w_ptr = s2_ptr + m * mtk::qr::tsqr_colmun_size;

#ifdef PROFILE_BREAKDOWN
	std::size_t gemm_count = 0lu;
	std::size_t tsqr_count = 0lu;
#endif

	cudaStream_t cuda_stream;
	CUTF_CHECK_ERROR(cublasGetStream(cublas_handle, &cuda_stream));

	cublasMath_t original_math_mode;
	CUTF_CHECK_ERROR(cublasGetMathMode(cublas_handle, &original_math_mode));

	if (mode == mtk::qr::fp16_tc_nocor || mode == mtk::qr::fp32_tc_nocor) {
		CUTF_CHECK_ERROR(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
	} else {
		CUTF_CHECK_ERROR(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
	}

	// QR factorization of each block
	for (std::size_t b = 0; b < column_block_size; b++) {
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

		const auto current_block_n = std::min(mtk::qr::tsqr_colmun_size, n - b * mtk::qr::tsqr_colmun_size);
		const auto previous_block_n = b * mtk::qr::tsqr_colmun_size;
		const auto one = cutf::type::cast<T>(1.0f);
		const auto zero = cutf::type::cast<T>(0.0f);
		const auto minus_one = cutf::type::cast<T>(-1.0f);

		if (b != 0) {
#ifdef PROFILE_BREAKDOWN
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			const auto t_0 = std::chrono::system_clock::now();
#endif
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						previous_block_n, current_block_n, m,
						&one,
						q_ptr, ldq,
						a_ptr + lda * previous_block_n, lda,
						&zero,
						r_ptr + ldr * previous_block_n, ldr
						));
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
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
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			const auto t_1 = std::chrono::system_clock::now();
			gemm_count += std::chrono::duration_cast<std::chrono::microseconds>(t_1 - t_0).count();
#endif
			mtk::tsqr::tsqr16<mtk::qr::get_tsqr_compute_mode<mode>()>(
					q_ptr + previous_block_n * ldq, ldq,
					r2_ptr, mtk::qr::tsqr_colmun_size,
					a_ptr + previous_block_n * lda, lda,
					m, current_block_n,
					wq_ptr,
					wr_ptr,
					d_wl_ptr,
					h_wl_ptr,
					cuda_stream
					);
#ifdef PROFILE_BREAKDOWN
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			const auto t_2 = std::chrono::system_clock::now();
			tsqr_count += std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
#endif
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						previous_block_n, current_block_n, m,
						&one,
						q_ptr, ldq,
						q_ptr + previous_block_n * ldq, ldq,
						&zero,
						s2_ptr, m
						));
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						m, current_block_n, previous_block_n,
						&minus_one,
						q_ptr, ldq,
						s2_ptr, m,
						&one,
						q_ptr + previous_block_n * ldq, ldq
						));
#ifdef PROFILE_BREAKDOWN
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			const auto t_3 = std::chrono::system_clock::now();
			gemm_count += std::chrono::duration_cast<std::chrono::microseconds>(t_3 - t_2).count();
#endif
			mtk::tsqr::tsqr16<mtk::qr::get_tsqr_compute_mode<mode>()>(
					q_ptr + previous_block_n * ldq, ldq,
					w_ptr, mtk::qr::tsqr_colmun_size,
					q_ptr + previous_block_n * ldq, ldq,
					m, current_block_n,
					wq_ptr,
					wr_ptr,
					d_wl_ptr,
					h_wl_ptr,
					cuda_stream
					);
#ifdef PROFILE_BREAKDOWN
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			const auto t_4 = std::chrono::system_clock::now();
			tsqr_count += std::chrono::duration_cast<std::chrono::microseconds>(t_4 - t_3).count();
#endif
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						previous_block_n, current_block_n, current_block_n,
						&one,
						s2_ptr, m,
						r2_ptr, mtk::qr::tsqr_colmun_size,
						&one,
						r_ptr + ldr * previous_block_n, ldr
						));
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						current_block_n, current_block_n, current_block_n,
						&one,
						w_ptr, mtk::qr::tsqr_colmun_size,
						r2_ptr, mtk::qr::tsqr_colmun_size,
						&zero,
						r_ptr + ldr * previous_block_n + previous_block_n, ldr
						));
#ifdef PROFILE_BREAKDOWN
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			const auto t_5 = std::chrono::system_clock::now();
			gemm_count += std::chrono::duration_cast<std::chrono::microseconds>(t_5 - t_4).count();
#endif
		} else {
#ifdef PROFILE_BREAKDOWN
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			const auto t_0 = std::chrono::system_clock::now();
#endif
			mtk::tsqr::tsqr16<mtk::qr::get_tsqr_compute_mode<mode>()>(
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
#ifdef PROFILE_BREAKDOWN
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
			const auto t_1 = std::chrono::system_clock::now();
			tsqr_count += std::chrono::duration_cast<std::chrono::microseconds>(t_1 - t_0).count();
#endif
		}
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	}

#ifdef PROFILE_BREAKDOWN
	const auto time_sum = gemm_count + tsqr_count;
#ifdef PROFILE_BREAKDOWN_CSV
	std::printf("%lu,%lu,%s,%s,%d,%d,%d,%e,%e,%e,%e\n",
			m, n,
			get_type_name<T>().c_str(),
			get_type_name<CORE_T>().c_str(),
			(UseTC ? 1 : 0),
			(Correction ? 1 : 0),
			0,
			(gemm_0_count + gemm_1_count) / 1.0e6, static_cast<double>(gemm_0_count + gemm_1_count) / time_sum * 100,
			tsqr_count / 1.0e6, static_cast<double>(tsqr_count) / time_sum * 100
			);
#else
	std::printf("# BlockQR breakdown\n");
	std::printf("Size   : %lu x %lu\n", m, n);
	std::printf("Type   : %s\n", get_type_name<T>().c_str());
	std::printf("C Type : %s\n", get_type_name<CORE_T>().c_str());
	std::printf("UseTC  : %s\n", (UseTC ? "YES" : "NO"));
	std::printf("Correction : %s\n", (Correction ? "YES" : "NO"));
	std::printf("Reorth : %s\n", "YES");
	std::printf("GEMM   : %e[s] (%e%%)\n", gemm_count / 1.0e6, static_cast<double>(gemm_count) / time_sum * 100);
	std::printf("TSQR   : %e[s] (%e%%)\n", tsqr_count / 1.0e6, static_cast<double>(tsqr_count) / time_sum * 100);
#endif
#endif

	CUTF_CHECK_ERROR(cublasSetMathMode(cublas_handle, original_math_mode));

	return mtk::qr::success_factorization;
}

} // namespace

template <mtk::qr::compute_mode mode, bool Reorthoganalize>
mtk::qr::state_t mtk::qr::qr(
		typename mtk::qr::get_io_type<mode>::type* const q_ptr, const std::size_t ldq,
		typename mtk::qr::get_io_type<mode>::type* const r_ptr, const std::size_t ldr,
		typename mtk::qr::get_io_type<mode>::type* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<mode>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<mode>::type* const wr_ptr,
		typename mtk::qr::get_io_type<mode>::type* const reorth_r,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const cublas_handle) {

	if (n > m || m == 0 || n == 0) {
		return mtk::qr::error_invalid_matrix_size;
	}

	if (Reorthoganalize) {
		return block_qr_reorthogonalization_core<mode>(
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
		return block_qr_core<mode>(
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


#define BQR_TEMPLATE_INSTANCE(mode, reorth) template mtk::qr::state_t mtk::qr::qr<mode, reorth>(mtk::qr::get_io_type<mode>::type* const, const std::size_t, mtk::qr::get_io_type<mode>::type* const, const std::size_t, mtk::qr::get_io_type<mode>::type* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::qr::get_working_q_type<mode>::type* const, typename mtk::qr::get_working_r_type<mode>::type* const, typename mtk::qr::get_io_type<mode>::type* const, unsigned* const, unsigned* const, cublasHandle_t const);
BQR_TEMPLATE_INSTANCE(mtk::qr::compute_mode::fp16_notc    , true);
BQR_TEMPLATE_INSTANCE(mtk::qr::compute_mode::fp32_notc    , true);
BQR_TEMPLATE_INSTANCE(mtk::qr::compute_mode::fp16_tc_nocor, true);
BQR_TEMPLATE_INSTANCE(mtk::qr::compute_mode::fp32_tc_nocor, true);
BQR_TEMPLATE_INSTANCE(mtk::qr::compute_mode::fp32_tc_cor  , true);
