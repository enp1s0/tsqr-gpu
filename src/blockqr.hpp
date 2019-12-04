#ifndef __BLOCKQR_HPP__
#define __BLOCKQR_HPP__
#include <cutf/cublas.hpp>
#include <cstddef>

namespace mtk {
template <class T, bool UseTC, bool Refinement>
void qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		const std::size_t m, const std::size_t n,
		const T* a_ptr, const std::size_t lda,
		cublasHandle_t const main_cublas_handle, cublasHandle_t const sub_cublas_handle);
} // namespace mtk

#endif /* end of include guard */
