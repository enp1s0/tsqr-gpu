#ifndef __BLOCKQR_HPP__
#define __BLOCKQR_HPP__
#include <cublas_v2.h>
#include <cstddef>
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
} // namespace qr
} // namespace mtk

#endif /* end of include guard */
