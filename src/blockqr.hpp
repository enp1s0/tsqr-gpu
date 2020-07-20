#ifndef __BLOCKQR_HPP__
#define __BLOCKQR_HPP__
#include <cublas_v2.h>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include "tsqr.hpp"

namespace mtk {
namespace qr {

enum compute_mode {
	fp16_notc,
	fp32_notc,
	fp16_tc_nocor,
	fp32_tc_nocor,
	tf32_tc_nocor,
	fp32_tc_cor,
	tf32_tc_cor,
	tf32_tc_cor_emu,
	mixed_tc_cor,
};

constexpr std::size_t tsqr_colmun_size = 16;

using state_t = int;
const state_t success_factorization = 0;
const state_t error_invalid_matrix_size = 1;

template <mtk::qr::compute_mode>
constexpr mtk::tsqr::compute_mode get_tsqr_compute_mode();
#define BQR_GET_TCQR_COMPUTE_MODE(mode) template<> constexpr mtk::tsqr::compute_mode get_tsqr_compute_mode<mtk::qr::compute_mode::mode>() {return mtk::tsqr::compute_mode::mode;}
BQR_GET_TCQR_COMPUTE_MODE(fp16_notc      );
BQR_GET_TCQR_COMPUTE_MODE(fp32_notc      );
BQR_GET_TCQR_COMPUTE_MODE(fp16_tc_nocor  );
BQR_GET_TCQR_COMPUTE_MODE(fp32_tc_nocor  );
BQR_GET_TCQR_COMPUTE_MODE(tf32_tc_nocor  );
BQR_GET_TCQR_COMPUTE_MODE(fp32_tc_cor    );
BQR_GET_TCQR_COMPUTE_MODE(tf32_tc_cor    );
BQR_GET_TCQR_COMPUTE_MODE(tf32_tc_cor_emu);
BQR_GET_TCQR_COMPUTE_MODE(mixed_tc_cor   );

template <mtk::qr::compute_mode mode>
struct get_working_q_type{using type = typename mtk::tsqr::get_working_q_type<get_tsqr_compute_mode<mode>>::type;};

template <mtk::qr::compute_mode mode>
struct get_working_r_type{using type = typename mtk::tsqr::get_working_r_type<get_tsqr_compute_mode<mode>>::type;};

template <mtk::qr::compute_mode mode>
struct get_io_type{using type = typename mtk::tsqr::get_io_type<get_tsqr_compute_mode<mode>>::type;};

// get working memory size
std::size_t get_working_q_size(const std::size_t m, const std::size_t n);
std::size_t get_working_r_size(const std::size_t m, const std::size_t n);
std::size_t get_working_l_size(const std::size_t m);

template <mtk::qr::compute_mode mode, bool Reorthogonalize>
struct buffer {
	typename get_working_q_type<mode>::type* dwq;
	typename get_working_r_type<mode>::type* dwr;
	typename get_io_type<mode>::type* dw_reorth_r;
	unsigned* dl;
	unsigned* hl;

	std::size_t total_memory_size;

	// constructor
	buffer() : dwq(nullptr), dwr(nullptr), dl(nullptr), hl(nullptr), total_memory_size(0lu) {}
	// destructor
	~buffer() {
		destroy();
	}

	void allocate(const std::size_t m, const std::size_t n) {
		if (dwq != nullptr || dwr != nullptr || dl != nullptr || hl != nullptr) {
			throw std::runtime_error("The buffer has been already allocated");
		}
		const auto wq_size = sizeof(typename get_working_q_type<mode>::type) * get_working_q_size(m, n);
		const auto wr_size = sizeof(typename get_working_r_type<mode>::type) * get_working_r_size(m, n);
		const auto l_size = sizeof(unsigned) * get_working_l_size(m);
		cudaMalloc(reinterpret_cast<void**>(&dwq), wq_size);
		cudaMalloc(reinterpret_cast<void**>(&dwr), wr_size);
		cudaMalloc(reinterpret_cast<void**>(&dl), l_size);
		cudaMallocHost(reinterpret_cast<void**>(&hl), l_size);
		total_memory_size = wq_size + wr_size + l_size;

		// Allocate additional working memory for reorthogonalization
		if (Reorthogonalize) {
			const auto reorth_r_size = sizeof(typename get_io_type<mode>::type) * (tsqr_colmun_size * tsqr_colmun_size * 2 + m * tsqr_colmun_size);
			cudaMalloc(reinterpret_cast<void**>(&dw_reorth_r), reorth_r_size);
			total_memory_size += reorth_r_size;
		}
	}

	void destroy() {
		cudaFree(dwq); dwq = nullptr;
		cudaFree(dwr); dwr = nullptr;
		if (Reorthogonalize) {
			cudaFree(dw_reorth_r); dw_reorth_r = nullptr;
		}
		cudaFree(dl); dl = nullptr;
		cudaFreeHost(hl); hl = nullptr;
	}

	void allocate_host(const std::size_t m, const std::size_t n) {
		if (dwq != nullptr || dwr != nullptr || dl != nullptr || hl != nullptr) {
			throw std::runtime_error("The buffer has been already allocated");
		}
		const auto wq_size = sizeof(typename get_working_q_type<mode>::type) * get_working_q_size(m, n);
		const auto wr_size = sizeof(typename get_working_r_type<mode>::type) * get_working_r_size(m, n);
		const auto l_size = sizeof(unsigned) * get_working_l_size(m);
		cudaMallocHost(reinterpret_cast<void**>(&dwq), wq_size);
		cudaMallocHost(reinterpret_cast<void**>(&dwr), wr_size);
		cudaMallocHost(reinterpret_cast<void**>(&dl), l_size);
		cudaMallocHost(reinterpret_cast<void**>(&hl), l_size);
		total_memory_size = wq_size + wr_size + l_size;

		// Allocate additional working memory for reorthogonalization
		if (Reorthogonalize) {
			const auto reorth_r_size = sizeof(typename get_io_type<mode>::type) * (tsqr_colmun_size * tsqr_colmun_size * 2 + m * tsqr_colmun_size);
			cudaMallocHost(reinterpret_cast<void**>(&dw_reorth_r), reorth_r_size);
			total_memory_size += reorth_r_size;
		}
	}

	void destroy_host() {
		cudaFreeHost(dwq); dwq = nullptr;
		cudaFreeHost(dwr); dwr = nullptr;
		if (Reorthogonalize) {
			cudaFreeHost(dw_reorth_r); dw_reorth_r = nullptr;
		}
		cudaFreeHost(dl); dl = nullptr;
		cudaFreeHost(hl); hl = nullptr;
	}
	std::size_t get_device_memory_size() const {
		return total_memory_size;
	}
};

template <mtk::qr::compute_mode mode, bool Reorthogonalize>
state_t qr(
		typename mtk::qr::get_io_type<mode>::type* const q_ptr, const std::size_t ldq,
		typename mtk::qr::get_io_type<mode>::type* const r_ptr, const std::size_t ldr,
		typename mtk::qr::get_io_type<mode>::type* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<mode>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<mode>::type* const wr_ptr,
		typename mtk::qr::get_io_type<mode>::type* const reorth_r_ptr,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const main_cublas_handle);

template <mtk::qr::compute_mode mode, bool Reorthogonalize>
inline state_t qr(
		typename mtk::qr::get_io_type<mode>::type* const q_ptr, const std::size_t ldq,
		typename mtk::qr::get_io_type<mode>::type* const r_ptr, const std::size_t ldr,
		typename mtk::qr::get_io_type<mode>::type* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		buffer<mode, Reorthogonalize>& bf,
		cublasHandle_t const main_cublas_handle) {
	return qr<mode, Reorthogonalize>(
			q_ptr, ldq,
			r_ptr, ldr,
			a_ptr, lda,
			m, n,
			bf.dwq,
			bf.dwr,
			bf.dw_reorth_r,
			bf.dl,
			bf.hl,
			main_cublas_handle
			);
}
} // namespace qr
} // namespace mtk

#endif /* end of include guard */
