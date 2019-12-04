#ifndef __BLOCKQR_HPP__
#define __BLOCKQR_HPP__
#include <cutf/cublas.hpp>
#include <cstddef>
#include "tsqr.hpp"

namespace mtk {
namespace qr {

template <class T, bool UseTC, bool Refine>
struct get_working_memory_type{using type = T;};

std::size_t get_working_memory_size(const std::size_t n);

template <class T, bool UseTC, bool Refinement>
void qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		const T* a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::tsqr::get_working_q_type<T, UseTC, Refinement>::type* const wq_ptr,
		typename mtk::tsqr::get_working_r_type<T, UseTC, Refinement>::type* const wr_ptr,
		typename mtk::qr::get_working_memory_type<T, UseTC, Refinement>::type* const wm_ptr,
		cublasHandle_t const main_cublas_handle, cublasHandle_t const sub_cublas_handle);
} // namespace qr
} // namespace mtk

#endif /* end of include guard */
