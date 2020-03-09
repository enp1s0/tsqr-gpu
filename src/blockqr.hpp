#ifndef __BLOCKQR_HPP__
#define __BLOCKQR_HPP__
#include <cublas_v2.h>
#include <cstddef>
#include <cstdlib>
#include "tsqr.hpp"

namespace mtk {
namespace qr {

template <class T, bool UseTC, bool Refine>
struct get_working_q_type{using type = typename mtk::tsqr::get_working_q_type<T, UseTC, Refine>::type;};

template <class T, bool UseTC, bool Refine>
struct get_working_r_type{using type = typename mtk::tsqr::get_working_r_type<T, UseTC, Refine>::type;};

// get working memory size
std::size_t get_working_q_size(const std::size_t m);
std::size_t get_working_r_size(const std::size_t m);
std::size_t get_working_l_size(const std::size_t m);

template <class T, bool UseTC, bool Refine>
struct buffer : public mtk::tsqr::buffer<T, UseTC, Refine> {
	typename get_working_q_type<T, UseTC, Refine>::type* dwq;
	typename get_working_r_type<T, UseTC, Refine>::type* dwr;
	unsigned* dl;
	unsigned* hl;

	buffer() : mtk::tsqr::buffer<T, UseTC, Refine>() {}
	~buffer() {
		destroy();
	}

	void allocate(const std::size_t m, const std::size_t n) {
		const auto wq_size = sizeof(typename get_working_q_type<T, UseTC, Refine>::type) * get_working_q_size(m);
		const auto wr_size = sizeof(typename get_working_r_type<T, UseTC, Refine>::type) * get_working_r_size(m);
		const auto l_size = sizeof(unsigned) * get_working_l_size(m);
		cudaMalloc(reinterpret_cast<void**>(&dwq), wq_size);
		cudaMalloc(reinterpret_cast<void**>(&dwr), wr_size);
		cudaMalloc(reinterpret_cast<void**>(&dl), l_size);
		cudaMallocHost(reinterpret_cast<void**>(&hl), l_size);
		mtk::tsqr::buffer<T, UseTC, Refine>::total_memory_size = wq_size + wr_size + l_size;
	}

	void destroy() {
		mtk::tsqr::buffer<T, UseTC, Refine>::destroy();
	}
};

template <bool UseTC, bool Refinement, class T, class CORE_T = T>
void qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::qr::get_working_q_type<T, UseTC, Refinement>::type* const wq_ptr,
		typename mtk::qr::get_working_r_type<T, UseTC, Refinement>::type* const wr_ptr,
		unsigned* const d_wl_ptr,
		unsigned* const h_wl_ptr,
		cublasHandle_t const main_cublas_handle);

template <bool UseTC, bool Refinement, class T, class CORE_T = T>
inline void qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		buffer<T, UseTC, Refinement>& bf,
		cublasHandle_t const main_cublas_handle) {
	qr<UseTC, Refinement, T, CORE_T>(
			q_ptr, ldq,
			r_ptr, ldr,
			a_ptr, lda,
			m, n,
			bf.dwq,
			bf.dwr,
			bf.dl,
			bf.hl,
			main_cublas_handle
			);
}
} // namespace qr
} // namespace mtk

#endif /* end of include guard */
