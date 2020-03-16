#ifndef __BLOCKQR_HPP__
#define __BLOCKQR_HPP__
#include <cublas_v2.h>
#include <cstddef>
#include <cstdlib>
#include "tsqr.hpp"

namespace mtk {
namespace qr {

constexpr std::size_t tsqr_colmun_size = 16;

using state_t = int;
const state_t success_factorization = 0;
const state_t error_invalid_matrix_size = 1;

template <class T, bool UseTC, bool Refine>
struct get_working_q_type{using type = typename mtk::tsqr::get_working_q_type<T, UseTC, Refine>::type;};

template <class T, bool UseTC, bool Refine>
struct get_working_r_type{using type = typename mtk::tsqr::get_working_r_type<T, UseTC, Refine>::type;};

// get working memory size
std::size_t get_working_q_size(const std::size_t m);
std::size_t get_working_r_size(const std::size_t m);
std::size_t get_working_l_size(const std::size_t m);

template <class T, bool UseTC, bool Refine, bool Reorthogonalize>
struct buffer {
	typename get_working_q_type<T, UseTC, Refine>::type* dwq;
	typename get_working_r_type<T, UseTC, Refine>::type* dwr;
	T* dw_reorth_r;
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
		const auto wq_size = sizeof(typename get_working_q_type<T, UseTC, Refine>::type) * get_working_q_size(m);
		const auto wr_size = sizeof(typename get_working_r_type<T, UseTC, Refine>::type) * get_working_r_size(m);
		const auto l_size = sizeof(unsigned) * get_working_l_size(m);
		const auto reorth_r_size = sizeof(T) * tsqr_colmun_size * tsqr_colmun_size * 3;
		cudaMalloc(reinterpret_cast<void**>(&dwq), wq_size);
		cudaMalloc(reinterpret_cast<void**>(&dwr), wr_size);
		cudaMalloc(reinterpret_cast<void**>(&dl), l_size);
		cudaMallocHost(reinterpret_cast<void**>(&hl), l_size);
		total_memory_size = wq_size + wr_size + l_size;
		if (Reorthogonalize) {
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
	std::size_t get_device_memory_size() const {
		return total_memory_size;
	}
};

template <bool UseTC, bool Refinement, bool Reorthogonalize, class T, class CORE_T = T>
state_t qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<T, UseTC, Refinement>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<T, UseTC, Refinement>::type* const wr_ptr,
		T* const reorth_r_ptr,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const main_cublas_handle);

template <bool UseTC, bool Reorthogonalize, bool Refinement, class T, class CORE_T = T>
inline state_t qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		buffer<T, UseTC, Refinement, Reorthogonalize>& bf,
		cublasHandle_t const main_cublas_handle) {
	return qr<UseTC, Refinement, T, CORE_T>(
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
