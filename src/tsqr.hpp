#ifndef __TSQR_HPP__
#define __TSQR_HPP__
#include <cstddef>
#include <cuda_fp16.h>

namespace mtk {
namespace tsqr {
// get batch size
std::size_t get_batch_size_log2(const std::size_t m);
std::size_t get_batch_size(const std::size_t m);
// get working memory type
template <class T, bool UseTC, bool Refine>
struct get_working_q_type{using type = T;};
template <> struct get_working_q_type<float, true, false>{using type = half;};

template <class T, bool UseTC, bool Refine>
struct get_working_r_type{using type = T;};

// get working memory size
std::size_t get_working_q_size(const std::size_t m, const std::size_t n);
std::size_t get_working_r_size(const std::size_t m, const std::size_t n);

template <bool UseTC, bool Refine, class T>
void tsqr16(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		const T* const a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n,
		typename get_working_q_type<T, UseTC, Refine>::type* const working_q_ptr,
		typename get_working_r_type<T, UseTC, Refine>::type* const working_r_ptr,
		cudaStream_t const cuda_stream = nullptr);
} // namespace tsqr
} // namespace mtk

#endif /* end of include guard */
