#ifndef __TCQR_HPP__
#define __TCQR_HPP__

#include <cstddef>

namespace mtk {
namespace tcqr{

enum compute_mode {
	fp16_notc,
	fp16_tc_nocor,
	fp32_notc,
	fp32_tc_cor,
	fp32_tc_nocor,
	mixed_tc_cor,
	tf32_tc_cor,
	tf32_tc_cor_emu,
	tf32_tc_nocor,
	tf32_tc_nocor_emu,
};

template <compute_mode mode, class Q_T, class R_T, class A_T>
void qr32x16(
		Q_T* const q, const std::size_t ldq,
		R_T* const r, const std::size_t ldr,
		const A_T* const a, const std::size_t lda,
		const unsigned int m, const unsigned int n,
		cudaStream_t const cuda_stream = nullptr);

template <compute_mode mode, class Q_T, class R_T, class A_T>
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
