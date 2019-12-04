#ifndef __TCQR_HPP__
#define __TCQR_HPP__

#include <cstddef>

namespace mtk {
namespace tcqr{
template <bool UseTC, bool Refine, class Q_T, class R_T, class A_T>
void qr32x16(
		Q_T* const q, const std::size_t ldq,
		R_T* const r, const std::size_t ldr,
		const A_T* const a, const std::size_t lda,
		const unsigned int m, const unsigned int n,
		cudaStream_t const cuda_stream = nullptr);

template <bool UseTC, bool Refine, class Q_T, class R_T, class A_T>
void qr32x16_batched(
		Q_T *const q, const std::size_t ldq,
		R_T *const r, const std::size_t ldr,
		const A_T *const a, const std::size_t lda,
		const unsigned int m, const unsigned int n,
		const std::size_t batch_size, const unsigned* a_start_position,
		cudaStream_t const cuda_stream = nullptr);
}
} // namespace mtk

#endif /* end of include guard */
